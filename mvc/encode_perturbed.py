#!/usr/bin/env python3
"""Mimi codes of information-perturbed audio (perturb.py), frame-aligned with the shards of encode_mimi.py.

For each shard <base> of a subset and each draw k < K this writes <base>.pcodes<k>.npy [N, 32] int16: the 32 codes
of every utterance after an independent random perturbation (formant shift, pitch randomisation, random EQ), cut to
the same stored frames as the clean shard; and the soft-CB0 profile of the same frames, <base>.psoft<k>.idx.npy
[N, 8] int16 and .d2.npy [N, 8] float16: the 8 CB0 codewords nearest to the perturbed semantic latent
sem.input_proj(lat) and their squared distances (a soft content stream between hard CB0 and the continuous latent). Perturbation keeps the length, so frame t of the perturbed codes is frame
t of the clean latent: a model can take content from the perturbed CB0 and the voice from elsewhere. The draws are
seeded by (seed, utterance id, k), so a rerun reproduces them; a shard whose files exist is skipped.

    python encode_perturbed.py --shards .../mvc/mimi/train-clean-100 --raw .../LibriTTS_R --K 2 --workers 8
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import encode_mimi as em


def utt_seed(seed, uid, k):
    return int.from_bytes(hashlib.sha256(f'{seed}:{uid}:{k}'.encode()).digest()[:8], 'little')


def wav_path(raw, r):
    return str(Path(raw) / r['subset'] / r['speaker'] / r['chapter'] / f"{r['id']}.wav")


def _perturb_job(args):
    path, seeds, reader, perturb_fn = args
    w = reader(path)
    t0 = time.time()
    out = [perturb_fn(w, em.SR, np.random.default_rng(s)) for s in seeds]
    if time.time() - t0 > 30:                      # names the utterance behind a perturb.py timeout message
        print(f'slow perturbation: {path} ({len(w) / em.SR:.1f} s of audio) took {time.time() - t0:.0f} s',
              file=sys.stderr, flush=True)
    return out


def default_perturb(w, sr, rng):
    import perturb
    return perturb.perturb(w, sr, rng)


def soft_cb0(enc, lat, top=8):
    """lat [T, 512] -> (idx [T, top] int64, d2 [T, top] float32): nearest CB0 codewords of sem.input_proj(lat)."""
    torch = enc.torch
    sem = enc.m.quantizer.semantic_residual_vector_quantizer
    cb = sem.layers[0].codebook
    with torch.no_grad():
        E0 = cb.embed_sum.float() / cb.cluster_usage.float().clamp(min=cb.epsilon)[:, None]
        z = sem.input_proj(torch.as_tensor(lat, dtype=torch.float32, device=enc.device).T[None])[0].T.float()
        d2 = torch.cdist(z, E0.to(z.device)) ** 2
        v, i = d2.topk(top, dim=1, largest=False)
    return i.cpu().numpy(), v.cpu().numpy()


def encode_shard(enc, base, raw, K, seed, pool=None, reader=em.read_wav, perturb_fn=default_perturb,
                 max_batch_samples=24000 * 240, log=print, top=8):
    outs = [f'{base}.pcodes{k}.npy' for k in range(K)]
    if all(os.path.exists(o) for o in outs):
        return False
    rows, _, codes = em.load_shard(base)
    jobs = [(wav_path(raw, r), [utt_seed(seed, r['id'], k) for k in range(K)], reader, perturb_fn) for r in rows]
    t0 = time.time()
    pert = list(pool.map(_perturb_job, jobs, chunksize=4)) if pool else [_perturb_job(j) for j in jobs]
    t1 = time.time()
    res = [np.zeros((len(codes), 32), np.int16) for _ in range(K)]
    sidx = [np.zeros((len(codes), top), np.int16) for _ in range(K)]
    sd2 = [np.zeros((len(codes), top), np.float16) for _ in range(K)]
    for k in range(K):
        wavs = [p[k] for p in pert]
        for b in em.batches([len(w) for w in wavs], max_batch_samples, max_batch=1):
            i = b[0]
            r = rows[i]
            lat, cod = enc([wavs[i]])[0]
            if len(cod) < r['frames']:
                raise SystemExit(f"{r['id']}: perturbed audio gave {len(cod)} frames, the shard stores {r['frames']}")
            a, n = r['offset'], r['frames']
            res[k][a:a + n] = cod[:n]
            ii, dd = soft_cb0(enc, lat[:n], top)
            sidx[k][a:a + n], sd2[k][a:a + n] = ii, dd
    for k, o in enumerate(outs):
        for path, arr in ((f'{base}.psoft{k}.idx.npy', sidx[k]), (f'{base}.psoft{k}.d2.npy', sd2[k]), (o, res[k])):
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                np.save(f, arr)
            os.replace(tmp, path)                  # the pcodes file last: its existence marks the draw complete
    agree = float(np.mean(res[0][:, 0] == np.asarray(codes[:, 0])))
    log(f'{base}: {len(rows)} utts, perturb {t1 - t0:.0f}s, encode {time.time() - t1:.0f}s, '
        f'CB0 agreement clean vs perturbed {agree:.3f}')
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True, help='comma-separated shard directories (<out>/<subset>)')
    ap.add_argument('--raw', required=True)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    enc = em.MimiEncoder(a.device)
    with ProcessPoolExecutor(a.workers) as pool:
        for d in a.shards.split(','):
            for js in sorted(Path(d).glob('shard_*.json')):
                encode_shard(enc, str(js)[:-5], a.raw, a.K, a.seed, pool, log=lambda m: print(m, flush=True))
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
