#!/usr/bin/env python3
"""Ground-truth-aligned latents for R5 (decoder_ft.py): MimiFlow-VC converts training utterances in reconstruction
mode, exactly as at inference (vc_adapter.MimiVC.convert_latent), and the generated latent of every utterance is
stored frame-aligned with its real audio.

For every k-th utterance of the shards (seeded order, speakers with at least 4 utterances, audit exclusions left out)
the source is the utterance itself and the reference other utterances of the same speaker, up to --ref_s seconds, as
in probe_recon.plan. Output: <out>/gta_<NNNN>.npz with ids, offsets, frames and lat [sum frames, 512] float16, one
file per --chunk utterances; a rerun skips the ids already written.

    python gta_latents.py --shards .../mvc/mimi/train-clean-100 --raw .../LibriTTS_R --every 3 \
        --cfg '{"run_dir": ".../runs/p6_V"}' --exclude results/audit_tc100/exclude.txt --out .../mvc/gta/p6_V
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import encode_mimi as em
import encode_perturbed as ep
import probe_recon as pr


def plan(rows, every, ref_s, exclude=(), seed=0):
    """[(utterance row, reference rows)] for every `every`-th usable utterance (references never hold it)."""
    keep = [r for r in rows if r['id'] not in exclude]
    items = pr.plan(keep, len(keep), ref_s, seed=seed, n_enrol=1)
    return [(u, ref) for i, (u, ref, _) in enumerate(items) if i % every == 0]


def done_ids(out):
    ids = set()
    for f in sorted(Path(out).glob('gta_*.npz')):
        ids |= {str(i) for i in np.load(f)['ids']}
    return ids


def write_chunk(out, k, ids, lats):
    off = np.cumsum([0] + [len(x) for x in lats])[:-1]
    path = Path(out) / f'gta_{k:04d}.npz'
    tmp = str(path) + '.tmp.npz'
    np.savez(tmp, ids=np.array(ids), offsets=off, frames=np.array([len(x) for x in lats]),
             lat=np.concatenate(lats).astype(np.float16))
    os.replace(tmp, path)
    return path


def run(adapter, items, raw, out, chunk=2000, reader=em.read_wav, ref_s=30.0, log=print):
    Path(out).mkdir(parents=True, exist_ok=True)
    have = done_ids(out)
    k = len(list(Path(out).glob('gta_*.npz')))
    ids, lats, t0, n = [], [], time.time(), 0
    for u, refs in items:
        if u['id'] in have:
            continue
        src = reader(ep.wav_path(raw, u))
        ref = np.concatenate([reader(ep.wav_path(raw, r)) for r in refs])[:int(ref_s * em.SR)]
        np.random.seed(int.from_bytes(u['id'].encode()[:4], 'little'))
        x = adapter.convert_latent(src, ref)
        if not 0 <= len(x) - u['frames'] <= em.TRIM:      # shards drop the last TRIM frames (encode_mimi.stored_frames)
            raise SystemExit(f"{u['id']}: generated {len(x)} frames, the shard stores {u['frames']}")
        x = x[:u['frames']]
        ids.append(u['id'])
        lats.append(np.asarray(x, dtype=np.float32))
        n += 1
        if len(ids) == chunk:
            log(f'  {write_chunk(out, k, ids, lats)}: {n} utterances, {time.time() - t0:.0f}s')
            k, ids, lats = k + 1, [], []
    if ids:
        log(f'  {write_chunk(out, k, ids, lats)}: {n} utterances, {time.time() - t0:.0f}s')
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--cfg', required=True, help='vc_adapter settings JSON')
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--ref_s', type=float, default=30.0)
    ap.add_argument('--exclude', default='')
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    rows = []
    for d in a.shards.split(','):
        for js in sorted(Path(d).glob('shard_*.json')):
            rows += json.load(open(js))
    ex = set(Path(a.exclude).read_text().split()) if a.exclude else set()
    items = plan(rows, a.every, a.ref_s, ex)
    import vc_adapter
    ad = vc_adapter.MimiVC()
    ad.load(a.device, **json.loads(a.cfg))
    n = run(ad, items, a.raw, a.out, a.chunk, ref_s=a.ref_s, log=lambda m: print(m, flush=True))
    print(f'{n} of {len(items)} planned utterances converted -> {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
