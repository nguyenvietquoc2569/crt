#!/usr/bin/env python3
"""Acoustic guards next to UTMOS (26 Oct, bottleneck plan guard set G): measurements that catch outputs UTMOS may
reward although they are worse speech (buzz, monotone, lost voicing). For every metric row (fields wav and source_wav,
as written by lab_rows.py or probe_recon.py), both at 16 kHz with Praat at a 10 ms step:
  hnr, hnr_src       mean harmonics-to-noise ratio (dB) over voiced frames;
  voiced_kept        share of the source's voiced frames that are voiced in the output (frame-aligned; the converters
                     keep the source timing);
  f0_pcc             correlation of log F0 between output and source on frames voiced in both;
  st_sd_ratio        semitone standard deviation of the output's F0 over the source's (1 = same pitch movement);
  rms_db_diff        output minus source RMS level in dB.
Writes one JSON line per row to --out and the means to --out with .summary.json.

    python acoustic_guard.py --rows .../metrics/dev_30.jsonl --out .../guard/dev_30.jsonl --workers 8
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SR = 16000
FIELDS = ('hnr', 'hnr_src', 'voiced_kept', 'f0_pcc', 'st_sd_ratio', 'rms_db_diff')


def load16(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type='soxr_hq')
    return x.astype(np.float64)


def f0_hnr(x):
    """Praat F0 (Hz, 0 = unvoiced) and harmonicity (dB, <= -100 where Praat finds no periodicity), both at 10 ms."""
    import parselmouth
    s = parselmouth.Sound(x, SR)
    f0 = s.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array['frequency']
    h = s.to_harmonicity_cc(time_step=0.01, minimum_pitch=60).values[0]
    return np.asarray(f0), np.asarray(h)


def mean_hnr(h):
    v = h[h > -100]
    return float(v.mean()) if len(v) else float('nan')


def rms_db(x):
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12))


def compare(out, src):
    """The guard fields for an output signal against its source (both 16 kHz float arrays)."""
    fo, ho = f0_hnr(out)
    fs, hs = f0_hnr(src)
    n = min(len(fo), len(fs))
    a, b = fo[:n], fs[:n]
    both = (a > 0) & (b > 0)
    pcc = float(np.corrcoef(np.log(a[both]), np.log(b[both]))[0, 1]) if both.sum() > 10 else float('nan')
    st = lambda v: 12 * np.log2(v[v > 0])  # noqa: E731
    sd_o = float(np.std(st(a))) if (a > 0).sum() > 10 else float('nan')
    sd_s = float(np.std(st(b))) if (b > 0).sum() > 10 else float('nan')
    kept = float(((a > 0) & (b > 0)).sum() / max((b > 0).sum(), 1))
    return dict(hnr=mean_hnr(ho), hnr_src=mean_hnr(hs), voiced_kept=kept, f0_pcc=pcc,
                st_sd_ratio=sd_o / sd_s if sd_s and np.isfinite(sd_s) and sd_s > 0 else float('nan'),
                rms_db_diff=rms_db(out) - rms_db(src))


def one(r):
    try:
        g = compare(load16(r['wav']), load16(r['source_wav']))
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the summary; it is recorded
        g = dict(error=f'{type(e).__name__}: {e}')
    return dict(pair_id=r['pair_id'], system=r.get('system'), **g)


def summarize(rows):
    out = dict(n=len(rows), n_error=sum('error' in r for r in rows))
    for k in FIELDS:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])], float)
        out[k] = float(v.mean()) if len(v) else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.rows)]
    rows = [r for r in rows if r.get('wav') and r.get('source_wav')]
    with Pool(a.workers) as p:
        res = p.map(one, rows, chunksize=4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    s = summarize(res)
    Path(a.out[:-len('.jsonl')] + '.summary.json' if a.out.endswith('.jsonl') else a.out + '.summary.json').write_text(
        json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
