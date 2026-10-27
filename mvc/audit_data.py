#!/usr/bin/env python3
"""Per-utterance audit of the Mimi-VC training data: which utterances are fit to train on.

Pass 1 (per shard, heavy: reads every wav and the latents; optional UTMOS on the GPU) writes <base>.audit.jsonl, one
JSON line per shard row (not .json, which the shard_*.json globs of every other script would pick up):
  sr, dur_s, finite, peak, clip_frac (|x| >= 0.999), rms_db (dBFS over speech frames), speech_frac (20 ms frames above
  -70 dBFS and within 40 dB of the utterance's 95th-percentile frame energy), speech_frac_core (the same with up to
  1 s of edge silence on each side left out of the denominator), lead_sil_s, trail_sil_s, frames_ok (the shard's
  stored frames match the audio length), lat_rms (RMS of the stored latent), utmos (UTMOS22-strong at 16 kHz, --utmos).
Pass 2 (all shards together; runs alone with --summarize) adds what needs the whole corpus or the other streams:
spk_cos (CAM++ cosine to the mean of the speaker's OTHER utterances, from .spk.npy), spk_other (the highest cosine to
any other speaker's centroid), spk_z (robust z of spk_cos within the speaker), agree<k> (CB0 agreement with perturbed
draw k, from .pcodes<k>.npy), lat_z (robust z of lat_rms over the corpus); then applies RULES and writes
<out>/audit_summary.json (counts and hours per flag, per-speaker and per-duration breakdowns, the audited and
unaudited shards), <out>/exclude.txt (one utterance id per line, the --exclude_utts input of train.py, which refuses
to train on a shard the summary does not list as audited) and <out>/flags.jsonl (every flagged utterance). A second
run that would change an existing exclude.txt stops unless --force.

RULES were fixed on 25 Oct 2026 before any filtered training run, and are not tuned on training results:
  unreadable, not finite, frames mismatch, clipped (clip_frac > 0.001), low speech (speech_frac_core < 0.5),
  speaker outlier (spk_cos < spk_other - 0.05, i.e. clearly closer to another speaker than to its own, or spk_z < -4
  with at least 5 utterances of the speaker), perturbation extreme (every agree<k> < 0.15, or every one > 0.9),
  latent outlier
  (|lat_z| > 6), low MOS (utmos < 2.5). Long silence (lead or trail > 1.0 s) is only reported. When a speaker would
  lose more than 25 % of its hours to the speaker rule, its flags become speaker_review (reported, not excluded).
A rule that flags more than 5 % of the trainable (>= 2 s) hours it could see (review_triggered in the summary) is
reviewed once, on the audit distribution alone, before the first filtered run; any change is logged in PLAN.md A0.
Review 1 (25 Oct, after audit job 5650): margin 0.05 on the speaker rule, every-draw perturbation rule, shares over
the hours each rule covers.

    python audit_data.py --shards .../mvc/mimi/train-clean-100,... --raw .../LibriTTS_R --utmos --out results/audit_v1
    python audit_data.py --shards ... --summarize --out results/audit_v1
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import encode_mimi as em
import encode_perturbed as ep

FRAME_S = 0.02
RULES = dict(clip_frac=1e-3, speech_frac=0.5, spk_margin=0.05, spk_z=-4.0, spk_min_utts=5, agree_lo=0.15, agree_hi=0.9,
             lat_z=6.0, utmos=2.5, silence_s=1.0, floor_db=-70.0, speaker_review_share=0.25, review_share=0.05,
             trainable_s=2.0)
EXCLUDING = ('unreadable', 'not_finite', 'frames_mismatch', 'clipped', 'low_speech', 'speaker_outlier',
             'perturbation_extreme', 'latent_outlier', 'low_mos')


def audio_stats(w, sr):
    """Level, clipping and speech-activity statistics of one signal (float, full scale 1)."""
    w = np.asarray(w, dtype=np.float64)
    out = dict(sr=int(sr), dur_s=round(len(w) / sr, 3), finite=bool(np.all(np.isfinite(w))))
    if not out['finite'] or len(w) == 0:
        return out
    hop = max(1, int(FRAME_S * sr))
    n = len(w) // hop
    e = (w[:n * hop].reshape(n, hop) ** 2).mean(1) if n else np.array([np.mean(w ** 2)])
    db = 10 * np.log10(e + 1e-12)
    ref = np.percentile(db, 95)
    active = (db > ref - 40) & (db > RULES['floor_db'])
    idx = np.flatnonzero(active)
    e_max = int(round(RULES['silence_s'] / FRAME_S))
    core = (active.sum() / max(1, n - min(idx[0], e_max) - min(n - 1 - idx[-1], e_max))) if idx.size else 0.0
    out.update(peak=round(float(np.abs(w).max()), 4), clip_frac=round(float(np.mean(np.abs(w) >= 0.999)), 6),
               speech_frac=round(float(active.mean()), 4), speech_frac_core=round(float(core), 4),
               rms_db=round(float(10 * np.log10(e[active].mean() + 1e-12)), 2) if idx.size else -120.0,
               lead_sil_s=round(float(idx[0] * FRAME_S), 2) if idx.size else out['dur_s'],
               trail_sil_s=round(float((n - 1 - idx[-1]) * FRAME_S), 2) if idx.size else out['dur_s'])
    return out


def _read(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32', always_2d=False)
    return (w.mean(1) if w.ndim > 1 else w), sr


def _row_stats(args):
    path, frames, reader = args
    try:
        w, sr = reader(path)
    except Exception as e:                              # noqa: BLE001 -- recorded as a flag
        return dict(error=f'{type(e).__name__}: {e}'[:200])
    s = audio_stats(w, sr)
    n24 = len(w) if sr == em.SR else int(round(len(w) * em.SR / sr))
    s['frames_ok'] = bool(em.stored_frames(n24) == frames)
    return s


def audit_shard(base, raw, pool=None, reader=_read, mos=None, log=print):
    """Pass 1 for one shard: <base>.audit.jsonl (skipped when it exists)."""
    out = base + '.audit.jsonl'
    if os.path.exists(out):
        return False
    rows, lat, _ = em.load_shard(base)
    jobs = [(ep.wav_path(raw, r), r['frames'], reader) for r in rows]
    t0 = time.time()
    stats = list(pool.map(_row_stats, jobs, chunksize=16)) if pool else [_row_stats(j) for j in jobs]
    for r, s in zip(rows, stats):
        x = np.asarray(lat[r['offset']:r['offset'] + r['frames']], dtype=np.float32)
        s['lat_rms'] = round(float(np.sqrt(np.mean(x ** 2))), 5) if x.size and np.all(np.isfinite(x)) else None
    if mos is not None:
        import librosa
        for (path, _, _), s in zip(jobs, stats):
            if 'error' not in s and s.get('finite'):
                w, sr = reader(path)
                s['utmos'] = round(mos.score(librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=16000)), 3)
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        f.writelines(json.dumps(x) + '\n' for x in stats)
    os.replace(tmp, out)
    log(f'{base}: {len(rows)} utts audited in {time.time() - t0:.0f}s')
    return True


def robust_z(x, ref=None):
    ref = x if ref is None else ref
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    return (x - med) / max(mad, 1e-6)


def shard_key(base):
    """subset/shard name, the same wherever the shards are staged."""
    return f'{Path(base).parent.name}/{Path(base).name}'


def collect(dirs, K=2):
    """(utterances of every audited shard with their pass-1 stats plus the pass-2 streams that exist, audited shard
    keys, unaudited shard keys)."""
    utts, audited, missing = [], [], []
    for d in dirs:
        for js in sorted(Path(d).glob('shard_*.json')):
            base = str(js)[:-5]
            if not os.path.exists(base + '.audit.jsonl'):
                missing.append(shard_key(base))
                continue
            audited.append(shard_key(base))
            rows, _, codes = em.load_shard(base)
            stats = [json.loads(x) for x in open(base + '.audit.jsonl')]
            spk = np.load(base + '.spk.npy') if os.path.exists(base + '.spk.npy') else None
            pc = [np.load(f'{base}.pcodes{k}.npy', mmap_mode='r') for k in range(K) if os.path.exists(f'{base}.pcodes{k}.npy')]
            for i, (r, s) in enumerate(zip(rows, stats)):
                a, n = r['offset'], r['frames']
                u = dict(id=r['id'], speaker=r['speaker'], subset=r.get('subset'), **s)
                if spk is not None:
                    u['_emb'] = np.asarray(spk[i], np.float32)
                for k, p in enumerate(pc):
                    u[f'agree{k}'] = round(float(np.mean(np.asarray(p[a:a + n, 0]) == np.asarray(codes[a:a + n, 0]))), 4)
                utts.append(u)
    return utts, audited, missing


def add_corpus_stats(utts, rules=RULES):
    by = {}
    for u in utts:
        if '_emb' in u and 'error' not in u and u.get('finite', True) and np.all(np.isfinite(u['_emb'])):
            by.setdefault(u['speaker'], []).append(u)
    names = sorted(by)
    norm = lambda E: E / np.maximum(np.linalg.norm(E, axis=-1, keepdims=True), 1e-9)  # noqa: E731
    cent = norm(np.stack([norm(np.stack([u['_emb'] for u in by[s]])).mean(0) for s in names])) if names else None
    for si, s in enumerate(names):
        us = by[s]
        E = norm(np.stack([u['_emb'] for u in us]))
        tot = E.sum(0)
        cos = np.array([float(E[i] @ (tot - E[i]) / max(np.linalg.norm(tot - E[i]), 1e-9)) if len(us) > 1 else 1.0
                        for i in range(len(us))])               # leave-one-out centroid of the speaker
        other = E @ cent.T
        other[:, si] = -np.inf
        other = other.max(1) if len(names) > 1 else np.full(len(us), -1.0)
        z = robust_z(cos) if len(us) >= rules['spk_min_utts'] else np.zeros(len(us))
        for u, c, o, zz in zip(us, cos, other, z):
            u['spk_cos'], u['spk_other'], u['spk_z'] = round(float(c), 4), round(float(o), 4), round(float(zz), 2)
    us = [u for u in utts if u.get('lat_rms') is not None]
    if us:
        for u, zz in zip(us, robust_z(np.array([u['lat_rms'] for u in us]))):
            u['lat_z'] = round(float(zz), 2)
    for u in utts:
        u.pop('_emb', None)
    return utts


def flags(u, rules=RULES):
    """Reasons to exclude (EXCLUDING) plus report-only ones, for one utterance."""
    if 'error' in u:
        return ['unreadable']
    f = []
    if not u.get('finite', True):
        f.append('not_finite')
    if u.get('frames_ok') is False:
        f.append('frames_mismatch')
    if u.get('clip_frac', 0) > rules['clip_frac']:
        f.append('clipped')
    if u.get('speech_frac_core', u.get('speech_frac', 1)) < rules['speech_frac']:
        f.append('low_speech')
    if 'spk_cos' in u and (u['spk_cos'] < u['spk_other'] - rules['spk_margin'] or u['spk_z'] < rules['spk_z']):
        f.append('speaker_outlier')
    ag = [u[k] for k in u if k.startswith('agree')]
    if ag and (max(ag) < rules['agree_lo'] or min(ag) > rules['agree_hi']):
        f.append('perturbation_extreme')
    if abs(u.get('lat_z', 0)) > rules['lat_z']:
        f.append('latent_outlier')
    if u.get('utmos') is not None and u['utmos'] < rules['utmos']:
        f.append('low_mos')
    if max(u.get('lead_sil_s', 0), u.get('trail_sil_s', 0)) > rules['silence_s']:
        f.append('long_silence')
    return f


def summarize(dirs, out, K=2, force=False, log=print):
    utts, audited, missing = collect(dirs, K)
    utts = add_corpus_stats(utts)
    if not utts:
        raise SystemExit('no audited shard under ' + ','.join(dirs))
    if missing:
        log(f'WARNING: {len(missing)} shards are not audited and are not covered by exclude.txt: {missing[:5]} ...')
    secs = lambda us: sum(u.get('dur_s', 0) for u in us)  # noqa: E731
    hours = lambda us: round(secs(us) / 3600, 3)  # noqa: E731
    fl = {u['id']: flags(u) for u in utts}
    by_spk = {}
    for u in utts:
        by_spk.setdefault(u['speaker'], []).append(u)
    review = []
    for s, us in by_spk.items():                       # a speaker the speaker rule would gut is reviewed, not cut
        lost = secs([u for u in us if 'speaker_outlier' in fl[u['id']]])
        if lost > RULES['speaker_review_share'] * secs(us):
            review.append(s)
            for u in us:
                fl[u['id']] = ['speaker_review' if x == 'speaker_outlier' else x for x in fl[u['id']]]
    ex = {i for i, f in fl.items() if any(x in EXCLUDING for x in f)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text = ''.join(i + '\n' for i in sorted(ex))
    old = out / 'exclude.txt'
    if old.exists() and old.read_text() != text and not force:
        raise SystemExit(f'{old} exists with a different list; use --force or a new --out')
    old.write_text(text)
    with open(out / 'flags.jsonl', 'w') as f:
        for u in utts:
            if fl[u['id']]:
                f.write(json.dumps(dict(u, flags=fl[u['id']])) + '\n')
    total = hours(utts)
    names = sorted({x for f in fl.values() for x in f})
    fh = {x: hours([u for u in utts if x in fl[u['id']]]) for x in names}
    need = dict(speaker_outlier='spk_cos', speaker_review='spk_cos', perturbation_extreme='agree0', low_mos='utmos')
    train_ok = lambda u: u.get('dur_s', 0) >= RULES['trainable_s']  # noqa: E731 -- train.py drops shorter ones
    share = {}                                         # over the trainable hours the rule could see
    for x in names:
        cov = [u for u in utts if train_ok(u) and (x not in need or u.get(need[x]) is not None)]
        share[x] = round(secs([u for u in cov if x in fl[u['id']]]) / max(secs(cov), 1e-9), 4)
    spk_share = {s: secs([u for u in us if u['id'] in ex]) / max(secs(us), 1e-9) for s, us in by_spk.items()}
    bins = [(2, 3), (3, 5), (5, 10), (10, 1e9)]                     # trainable durations only
    dur = {f'{a}-{b if b < 1e9 else "inf"} s': round(sum(u['id'] in ex for u in utts if a <= u.get('dur_s', 0) < b)
                                                    / max(1, sum(a <= u.get('dur_s', 0) < b for u in utts)), 4) for a, b in bins}
    keys = ('dur_s', 'rms_db', 'speech_frac', 'speech_frac_core', 'clip_frac', 'spk_cos', 'spk_other', 'utmos', 'agree0', 'lat_rms')
    pct = {k: [round(float(np.percentile(v, q)), 4) for q in (1, 5, 50, 95, 99)]
           for k in keys if (v := [u[k] for u in utts if u.get(k) is not None and np.isfinite(u[k])])}
    v = np.array(list(spk_share.values()))
    res = dict(utterances=len(utts), hours=total, excluded=len(ex), excluded_hours=hours([u for u in utts if u['id'] in ex]),
               flag_counts={x: sum(x in f for f in fl.values()) for x in names}, flag_hours=fh, flag_share_of_hours=share,
               review_triggered=[x for x in names if x in EXCLUDING and share[x] > RULES['review_share']],
               speakers=dict(n=len(by_spk), excluded_share_p50_90_99=[round(float(np.percentile(v, q)), 4) for q in (50, 90, 99)],
                             over_25pct=sorted([(s, round(hours(by_spk[s]), 3), round(x, 3)) for s, x in spk_share.items() if x > 0.25],
                                               key=lambda t: -t[2]),
                             left_below_2_utts=sorted(s for s, us in by_spk.items() if sum(u['id'] not in ex for u in us) < 2),
                             speaker_review=sorted(review)),
               excluded_share_by_duration=dur, percentiles_1_5_50_95_99=pct, rules=RULES, excluding=list(EXCLUDING),
               by_subset={s: dict(utterances=sum(u['subset'] == s for u in utts), excluded=sum(u['subset'] == s and u['id'] in ex for u in utts))
                          for s in sorted({u['subset'] for u in utts})},
               coverage=dict(spk_cos=sum('spk_cos' in u for u in utts), agree=sum('agree0' in u for u in utts),
                             utmos=sum(u.get('utmos') is not None for u in utts)),
               audited_shards=audited, unaudited_shards=missing,
               exclude_sha256=hashlib.sha256(text.encode()).hexdigest())
    json.dump(res, open(out / 'audit_summary.json', 'w'), indent=1)
    log(f"{res['utterances']} utterances ({total} h): exclude {res['excluded']} ({res['excluded_hours']} h); "
        f"share of hours {share}; review triggered {res['review_triggered']}; coverage {res['coverage']}; "
        f"{len(missing)} shards not audited")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True, help='comma-separated shard directories (<out>/<subset>)')
    ap.add_argument('--raw', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--utmos', action='store_true')
    ap.add_argument('--journal', default='', help='benchmark/journal dir (metrics.UTMOS) for --utmos')
    ap.add_argument('--summarize', action='store_true', help='pass 2 only')
    ap.add_argument('--force', action='store_true', help='replace an existing exclude.txt that differs')
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    dirs = [d for d in a.shards.split(',') if d]
    if not a.summarize:
        if not a.raw:
            raise SystemExit('--raw is needed for pass 1')
        mos = None
        if a.utmos:
            sys.path.insert(0, a.journal)
            import metrics
            mos = metrics.UTMOS(a.device)
        with ProcessPoolExecutor(a.workers) as pool:
            for d in dirs:
                for js in sorted(Path(d).glob('shard_*.json')):
                    audit_shard(str(js)[:-5], a.raw, pool, mos=mos, log=lambda m: print(m, flush=True))
    summarize(dirs, a.out, a.K, a.force, log=lambda m: print(m, flush=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
