#!/usr/bin/env python3
"""Probe P4: is CAM++ (spk_embed.CamPlus, trained on Chinese speech) a usable speaker vector for English speech?

Speaker verification on two held-out sets, never trained on:
  dev-clean   LibriTTS-R dev-clean: each speaker's utterances concatenated in id order and cut into non-overlapping
              10 s segments, at most 20 per speaker (evenly spaced over the speaker's audio, so they span chapters)
  vctk-dev    the VCTK Dev speakers of the journal Dev pair list: every source_path (source_speaker) and E path
              (target_speaker) cut into non-overlapping 3 s segments (shorter files give none), at most 30 per
              speaker, evenly spaced over the speaker's files in path order
Trials: every segment against every other segment, genuine when both are of the same speaker; scores are cosines of
the L2-normalised CAM++ embeddings; EER by probe_speaker_info.eer. As a Mimi-native reference the same trials are
scored with the 512-d mean Mimi latent of each segment (MimiEncoder, the last TRIM frames dropped as in the shards),
standardised per dimension over the set and cosine-scored (probe_speaker_info.standardise).

Go/no-go (PLAN.md section 4, P4): keep CAM++ if its EER is <= 5% on dev-clean and <= 8% on vctk-dev; otherwise the
Mimi-AAM vector (K3) becomes the default in P6.

Outputs in --out: p4_campplus.json (per set: EER of CAM++ and of the Mimi latent, n_segments, n_speakers, n_trials,
n_genuine; the gate and the decision) and per-set embedding caches p4_<set>.npz, reused when the set's segment plan
(files, segment length, cap) is unchanged. Pair lists copied from the old account hold /datastore/nvquoc/crt paths:
--path_map /datastore/nvquoc/crt=/datastore/ceai/crt rewrites that prefix (repeatable; first match wins).

    python probe_campplus.py --raw $RAW --pairs $J/pairs/dev.jsonl --out $WORK_DIR/mvc/results
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import encode_mimi as em
import probe_speaker_info as ps
import spk_embed as se

WORK = '/datastore/ceai/crt/mimi_vc'
RAW = f'{WORK}/data_storage/dataset_raw/LibriTTS_R'
PAIRS = f'{WORK}/benchmark/journal/pairs/dev.jsonl'
OUT = f'{WORK}/mvc/results'
GATES = {'dev-clean': 0.05, 'vctk-dev': 0.08}


def parse_map(items):
    out = []
    for s in items:
        old, sep, new = s.partition('=')
        if not sep or not old:
            raise SystemExit(f'--path_map wants OLD=NEW, got {s!r}')
        out.append((old, new))
    return out


def map_path(p, maps):
    for old, new in maps:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def libri_files(raw, subset='dev-clean'):
    """speaker -> wav paths in utterance-id order."""
    utts = em.find_utterances(raw, subset)
    if not utts:
        raise SystemExit(f'{subset}: no wav files under {raw}/{subset}')
    out = {}
    for u in sorted(utts, key=lambda u: (u['speaker'], u['id'])):
        out.setdefault(u['speaker'], []).append(u['path'])
    return out


def vctk_files(rows, maps=()):
    """speaker -> sorted unique paths: source_path of source_speaker and E paths of target_speaker."""
    out = {}
    for r in rows:
        for spk, paths in ((r['source_speaker'], [r['source_path']]), (r['target_speaker'], r['E']['paths'])):
            out.setdefault(spk, set()).update(map_path(p, maps) for p in paths)
    return {s: sorted(v) for s, v in sorted(out.items())}


def cut(wav, n):
    return [wav[i * n:(i + 1) * n] for i in range(len(wav) // n)]


def spread(n, k):
    """Indices of at most k of n items, evenly spaced, in order, first and last included."""
    if n <= k:
        return list(range(n))
    return [int(i) for i in np.round(np.linspace(0, n - 1, k))]


def concat_segments(files, reader, seg_s, max_seg):
    """(segment id, speaker, wav): each speaker's files concatenated in the given order, then cut."""
    n = int(round(seg_s * em.SR))
    for spk, paths in files.items():
        segs = cut(np.concatenate([np.asarray(reader(p), dtype=np.float32) for p in paths]), n)
        for i in spread(len(segs), max_seg):
            yield f'{spk}_c{i:03d}', spk, segs[i]


def file_segments(files, reader, seg_s, max_seg):
    """(segment id, speaker, wav): every file cut on its own; a speaker's pieces pooled in file order, then capped."""
    n = int(round(seg_s * em.SR))
    for spk, paths in files.items():
        segs = [(f'{Path(p).stem}_{j}', w) for p in paths for j, w in enumerate(cut(reader(p), n))]
        for i in spread(len(segs), max_seg):
            yield segs[i][0], spk, segs[i][1]


def embed(segments, campplus, mimi, batch=16):
    """-> segment ids, speakers, CAM++ [N, 192], mean Mimi latent of the stored frames [N, 512]."""
    ids, spk, C, M, buf = [], [], [], [], []

    def flush():
        for w, (lat, _) in zip(buf, mimi(buf)):
            M.append(np.asarray(lat[:em.stored_frames(len(w))], dtype=np.float64).mean(0))
        buf.clear()

    for sid, s, w in segments:
        ids.append(sid)
        spk.append(s)
        C.append(np.asarray(campplus(w, em.SR), dtype=np.float32))
        buf.append(w)
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    return ids, spk, np.stack(C), np.stack(M).astype(np.float32)


def l2(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def trials(F, speakers):
    """F [N, D] L2-normalised -> cosine scores of all N (N - 1) / 2 segment pairs: (genuine, impostor)."""
    spk = np.asarray(speakers)
    i, j = np.triu_indices(len(F), 1)
    S = (F @ F.T)[i, j]
    same = spk[i] == spk[j]
    return S[same], S[~same]


def score(F, speakers):
    g, imp = trials(F, speakers)
    return dict(eer=ps.eer(g, imp), n_trials=int(len(g) + len(imp)), n_genuine=int(len(g)))


def plan_key(files, seg_s, max_seg, concat):
    spec = dict(seg_s=seg_s, max_seg=max_seg, concat=concat,
                files=[[s, Path(p).name] for s, paths in files.items() for p in paths])
    return hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:16]


def save_npz(path, **arrays):
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def run_set(name, files, reader, campplus, mimi, out, seg_s, max_seg, concat, batch=16, log=print):
    key = plan_key(files, seg_s, max_seg, concat)
    cache = Path(out) / f'p4_{name}.npz'
    hit = None
    if cache.exists():
        with np.load(cache) as z:
            if str(z['key']) == key:
                hit = [str(x) for x in z['ids']], [str(x) for x in z['speakers']], z['campplus'], z['mimi']
    if hit:
        ids, spk, C, M = hit
        log(f'{name}: {len(ids)} cached segment embeddings ({cache})')
    else:
        t0 = time.time()
        segs = (concat_segments if concat else file_segments)(files, reader, seg_s, max_seg)
        ids, spk, C, M = embed(segs, campplus, mimi, batch)
        save_npz(cache, key=np.array(key), ids=np.array(ids), speakers=np.array(spk), campplus=C, mimi=M)
        log(f'{name}: embedded {len(ids)} segments in {time.time() - t0:.0f}s')
    cam, lat = score(l2(C), spk), score(ps.standardise(np.asarray(M, dtype=np.float64)), spk)
    per = np.unique(np.asarray(spk), return_counts=True)[1]
    gate = GATES[name]
    return dict(seg_s=seg_s, max_segments_per_speaker=max_seg, n_files=sum(len(v) for v in files.values()),
                n_segments=len(ids), n_speakers=len(per), segments_per_speaker=[int(per.min()), int(per.max())],
                n_trials=cam['n_trials'], n_genuine=cam['n_genuine'],
                eer={'campplus': cam['eer'], 'mimi_lat': lat['eer']}, gate_eer=gate, passed=cam['eer'] <= gate)


def decide(sets):
    if not all(n in sets for n in GATES):
        return None, 'incomplete: both dev-clean and vctk-dev are needed for the decision'
    if all(sets[n]['passed'] for n in GATES):
        return True, 'keep CAM++ as the global speaker vector'
    return False, 'CAM++ fails: the Mimi-AAM vector (K3) becomes the P6 default'


def probe(a, campplus, mimi, reader=em.read_wav, log=print):
    maps = parse_map(a.path_map)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    sets = {}
    for name in a.sets.split(','):
        if name == 'dev-clean':
            files = libri_files(a.raw, 'dev-clean')
            sets[name] = run_set(name, files, reader, campplus, mimi, a.out, a.seg_libri, a.max_libri, True,
                                 a.batch, log)
        elif name == 'vctk-dev':
            rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
            files = vctk_files(rows, maps)
            sets[name] = run_set(name, files, reader, campplus, mimi, a.out, a.seg_vctk, a.max_vctk, False,
                                 a.batch, log)
        else:
            raise SystemExit(f'unknown set {name!r}: use dev-clean and/or vctk-dev')
    keep, decision = decide(sets)
    res = dict(probe='P4 CAM++ sanity', sets=sets, keep_campplus=keep, decision=decision,
               config=dict(raw=a.raw, pairs=a.pairs, path_map=a.path_map, score='cosine', eer='probe_speaker_info.eer',
                           mimi_lat='mean of stored frames, standardised over the set'))
    path = Path(a.out) / 'p4_campplus.json'
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)
    for name, r in sets.items():
        log(f"{name}: {r['n_segments']} segments of {r['seg_s']:g} s, {r['n_speakers']} speakers, {r['n_trials']} "
            f"trials ({r['n_genuine']} genuine): EER CAM++ {100 * r['eer']['campplus']:.2f}% "
            f"(gate {100 * r['gate_eer']:g}%: {'pass' if r['passed'] else 'FAIL'}), "
            f"Mimi mean latent {100 * r['eer']['mimi_lat']:.2f}%")
    log(f'P4: {decision} -> {path}')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default=RAW, help='LibriTTS-R root holding dev-clean/')
    ap.add_argument('--pairs', default=PAIRS, help='the journal Dev pair list')
    ap.add_argument('--path_map', action='append', default=[], metavar='OLD=NEW')
    ap.add_argument('--sets', default='dev-clean,vctk-dev')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--seg_libri', type=float, default=10.0)
    ap.add_argument('--max_libri', type=int, default=20)
    ap.add_argument('--seg_vctk', type=float, default=3.0)
    ap.add_argument('--max_vctk', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    probe(a, se.CamPlus(a.device), em.MimiEncoder(a.device), log=lambda m: print(m, flush=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
