#!/usr/bin/env python3
"""Probe P0b: speaker identity in Mimi's streams with the text held fixed (VCTK Dev shared sentences).

In P0 (probe_speaker_info.py) every speaker reads different text, so only accuracy above a lexical control counts.
Here every speaker reads the same sentences, so anything above chance is voice or accent. From the Dev pair list,
each row with a shared_sentence_id gives the source speaker's recording of that sentence (source_path) and, when set,
the target speaker's recording of the same sentence (P). This builds a speaker x sentence table of wav paths and keeps
the sentences recorded by at least --min_speakers speakers. Each utterance is encoded clean and after K perturbed
draws (perturb.perturb, seeded by encode_perturbed.utt_seed(seed, '<speaker>_<sentence>', k), as the training data).
The last TRIM frames are dropped, as in the shards. Streams per utterance:
  cb0 / pcb0_k          bag of CB0 tokens of the clean / perturbed audio (tf-idf, L2-normalised)
  cb0_emb / pcb0_emb_k  mean CB0 codeword embedding (E0 = MimiIO.cb0_table()), standardised over the probe set
  cb1                   bag of CB1 tokens (an acoustic level, for scale)
  lat                   mean of the continuous latent, standardised over the probe set (as P0)
Leave-one-sentence-out: for each held-out sentence, a speaker's centroid is the mean of its vectors on the other
sentences, and the held-out utterances go to the nearest centroid (cosine): speaker-ID accuracy, chance
1/n_speakers. Sex: the nearest of the sex centroids built from the other sentences, balanced accuracy (mean recall
per sex; sex from source_gender / target_gender). sex_acc_unseen also leaves the test speaker out of the sex
centroids, so a stream that only identifies speakers cannot score on sex through the speaker's own other sentences.

Gate (feeds the P0 decision): E1 is needed if perturbed CB0 still gives speaker-ID >= 5x chance or sex accuracy
>= 0.80; the perturbed value is the higher of the bag and the embedding stream, each averaged over the K draws.
Outputs: <out>/p0b_parallel_text.json (results and gate) and <out>/p0b_features.npz (token counts and latent means
per utterance; reused when the utterance list, K and seed are unchanged, so a rerun only re-scores).

    python probe_parallel_text.py --pairs $J/pairs/dev.jsonl --out $WORK_DIR/mvc/results/p0b --K 2 --workers 8
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import encode_mimi as em
import encode_perturbed as ep
import perturb
from probe_speaker_info import bag, standardise, tfidf

RATIO, SEX_MAX = 5.0, 0.80


def map_path(p, maps):
    for old, new in maps:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_table(rows, min_speakers=10, maps=()):
    """Pair rows -> ({sentence id: {speaker: wav path}} of the sentences with >= min_speakers speakers,
    {speaker: sex} of the speakers in it)."""
    table, sex = defaultdict(dict), {}
    for r in rows:
        sex.setdefault(r['source_speaker'], r['source_gender'])
        sex.setdefault(r['target_speaker'], r['target_gender'])
        s = r.get('shared_sentence_id')
        if not s:
            continue
        table[s].setdefault(r['source_speaker'], map_path(r['source_path'], maps))
        if r.get('P'):
            table[s].setdefault(r['target_speaker'], map_path(r['P'], maps))
    table = {s: dict(sorted(v.items())) for s, v in sorted(table.items()) if len(v) >= min_speakers}
    return table, {p: sex[p] for p in sorted({p for v in table.values() for p in v})}


def items_of(table):
    return [(spk, sent, path) for sent, v in table.items() for spk, path in v.items()]


def _load(args):
    path, seeds, reader, fn = args
    w = reader(path)
    return [w] + [fn(w, em.SR, np.random.default_rng(s)) for s in seeds]


def encode_items(enc, items, K, seed, reader=em.read_wav, perturb_fn=ep.default_perturb, pool=None, log=print):
    """(speaker, sentence, path) items -> per-utterance token counts and latent means of the clean audio and of K
    perturbed draws (stored frames only), plus the CB0 agreement of each draw with the clean audio."""
    jobs = [(path, [ep.utt_seed(seed, f'{spk}_{sent}', k) for k in range(K)], reader, perturb_fn)
            for spk, sent, path in items]
    loaded = pool.map(_load, jobs, chunksize=4) if pool else map(_load, jobs)
    N = len(items)
    out = dict(cb0=np.zeros((N, 2048), np.int32), cb1=np.zeros((N, 2048), np.int32), lat=np.zeros((N, 512)),
               pcb0=np.zeros((K, N, 2048), np.int32), agree=np.zeros((K, N)), frames=np.zeros(N, np.int64))
    for i, wavs in enumerate(loaded):
        n = em.stored_frames(len(wavs[0]))
        lat, cod = enc([wavs[0]])[0]
        out['cb0'][i], out['cb1'][i] = bag(cod[:n, 0]), bag(cod[:n, 1])
        out['lat'][i] = np.asarray(lat[:n], dtype=np.float64).mean(0)
        out['frames'][i] = n
        for k, w in enumerate(wavs[1:]):
            pc = enc([w])[0][1][:n, 0]
            out['pcb0'][k, i] = bag(pc)
            out['agree'][k, i] = float(np.mean(pc == cod[:n, 0]))
        if (i + 1) % 50 == 0 or i + 1 == N:
            log(f'encoded {i + 1}/{N} utterances')
    return out


def streams(feat, E0, K):
    """Token counts and latent means -> {stream: [N, D] L2-normalised features}."""
    def emb(c):
        return standardise(c @ E0 / np.maximum(c.sum(1, keepdims=True), 1))
    S = dict(cb0=tfidf(feat['cb0'].astype(np.float64)), cb0_emb=emb(feat['cb0']),
             cb1=tfidf(feat['cb1'].astype(np.float64)), lat=standardise(feat['lat']))
    for k in range(K):
        S[f'pcb0_{k}'] = tfidf(feat['pcb0'][k].astype(np.float64))
        S[f'pcb0_emb_{k}'] = emb(feat['pcb0'][k])
    return S


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _balanced(hits):
    return float(np.mean([np.mean(h) for h in hits.values()]))


def loso(F, speakers, sentences, sexes):
    """Leave-one-sentence-out nearest-centroid (cosine) speaker-ID and balanced sex accuracy. F [N, D], one speaker,
    sentence id and sex per row. Centroids use only the other sentences' rows."""
    spk, sent, sx = np.asarray(speakers), np.asarray(sentences), np.asarray(sexes)
    names, labels = sorted(set(speakers)), sorted(set(sexes))
    correct, hits, hits_u = 0, defaultdict(list), defaultdict(list)
    for s in sorted(set(sentences)):
        te, tr = np.nonzero(sent == s)[0], np.nonzero(sent != s)[0]
        cand = [p for p in names if np.any(spk[tr] == p)]
        C = _unit(np.stack([F[tr[spk[tr] == p]].mean(0) for p in cand]))
        correct += int(np.sum(np.asarray(cand)[(F[te] @ C.T).argmax(1)] == spk[te]))
        G = _unit(np.stack([F[tr[sx[tr] == g]].mean(0) for g in labels]))
        for i, g in zip(te, np.asarray(labels)[(F[te] @ G.T).argmax(1)]):
            hits[sx[i]].append(g == sx[i])
            other = tr[spk[tr] != spk[i]]
            Gu = _unit(np.stack([F[other[sx[other] == g]].mean(0) for g in labels]))
            hits_u[sx[i]].append(labels[int((Gu @ F[i]).argmax())] == sx[i])
    n = len(spk)
    return dict(acc=correct / n, chance=1 / len(names), sex_acc=_balanced(hits), sex_acc_unseen=_balanced(hits_u),
                n_test=n, n_speakers=len(names), n_sentences=len(set(sentences)))


def gate(res, K, chance, ratio=RATIO, sex_max=SEX_MAX):
    acc = max(float(np.mean([res[f'{t}_{k}']['acc'] for k in range(K)])) for t in ('pcb0', 'pcb0_emb'))
    sex = max(float(np.mean([res[f'{t}_{k}']['sex_acc'] for k in range(K)])) for t in ('pcb0', 'pcb0_emb'))
    return dict(e1_needed=bool(acc >= ratio * chance or sex >= sex_max), pcb0_acc=acc, pcb0_sex_acc=sex,
                chance=chance, rule=f'perturbed-CB0 speaker-ID >= {ratio:g}x chance or sex accuracy >= {sex_max:.2f}'
                                    f' (higher of bag and embedding stream, each averaged over the {K} draws)')


def _save_npz(path, **arrays):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def features(a, items, enc_fn, log=print, **kw):
    """Cached features of the items: the npz in a.out is reused when its utterance list, K and seed match."""
    path = os.path.join(a.out, 'p0b_features.npz')
    meta = json.dumps(dict(items=[list(t) for t in items], K=a.K, seed=a.seed))
    if os.path.exists(path):
        z = np.load(path)
        if str(z['meta']) == meta:
            log(f'features reused from {path}')
            return {k: z[k] for k in z.files if k != 'meta'}
    enc, E0 = enc_fn()
    feat = encode_items(enc, items, a.K, a.seed, log=log, **kw)
    feat['E0'] = np.asarray(E0, dtype=np.float64)
    _save_npz(path, meta=np.array(meta), **feat)
    return feat


def run(a, enc_fn, log=print, **kw):
    """enc_fn() -> (MimiEncoder-like callable, E0 [2048, 256]), called only when the features are not cached; kw
    (reader, perturb_fn, pool) go to encode_items."""
    os.makedirs(a.out, exist_ok=True)
    table, sex = build_table(load_rows(a.pairs), a.min_speakers, a.path_map)
    items = items_of(table)
    log(f'{len(items)} utterances: {len(table)} sentences with >= {a.min_speakers} of {len(sex)} speakers')
    feat = features(a, items, enc_fn, log, **kw)
    spk, sent = [t[0] for t in items], [t[1] for t in items]
    S = streams(feat, feat['E0'], a.K)
    res = {name: loso(F, spk, sent, [sex[p] for p in spk]) for name, F in S.items()}
    g = gate(res, a.K, 1 / len(sex))
    out = dict(probe='P0b', pairs=a.pairs, n_utts=len(items), n_speakers=len(sex), n_sentences=len(table),
               min_speakers=a.min_speakers, sentences={s: len(v) for s, v in table.items()}, speakers=sex, K=a.K,
               seed=a.seed, perturb=perturb.PRESET, cb0_agreement=[float(x) for x in feat['agree'].mean(1)],
               streams=res, gate=g)
    _save_json(os.path.join(a.out, 'p0b_parallel_text.json'), out)
    for name, r in res.items():
        log(f"{name:11s} speaker-ID {r['acc']:.3f} ({r['acc'] / r['chance']:.1f}x chance {r['chance']:.3f}), "
            f"sex {r['sex_acc']:.3f} (unseen speaker {r['sex_acc_unseen']:.3f})")
    log(f"CB0 agreement clean vs perturbed: {', '.join(f'{x:.3f}' for x in out['cb0_agreement'])}")
    log(f"gate: perturbed CB0 speaker-ID {g['pcb0_acc']:.3f} ({g['pcb0_acc'] / g['chance']:.1f}x chance), "
        f"sex {g['pcb0_sex_acc']:.3f} -> E1 {'NEEDED' if g['e1_needed'] else 'not needed'}")
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', required=True, help='the Dev pair list (pairs/dev.jsonl)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--min_speakers', type=int, default=10)
    ap.add_argument('--path_map', action='append', default=[], metavar='OLD=NEW',
                    help='rewrite a path prefix of the pair list (repeatable)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    a.path_map = [tuple(m.split('=', 1)) for m in a.path_map]
    return a


def main(argv=None):
    a = parse(argv)

    def enc_fn():
        import mimi_io
        io = mimi_io.MimiIO(a.device)
        return io.enc, io.cb0_table()
    with ProcessPoolExecutor(a.workers) as pool:
        run(a, enc_fn, log=lambda m: print(m, flush=True), pool=pool)
    return 0


if __name__ == '__main__':
    sys.exit(main())
