#!/usr/bin/env python3
"""Encode a speech corpus with Mimi once, for the Mimi-VC models: the continuous encoder latent and the 32 RVQ codes.

For every utterance this stores
  lat   [T, 512] float16   the encoder output after the transformer and the 12.5 Hz downsampling, before the RVQ
                           (decoder input when unquantised: sem.out(sem.in(lat)) + ac.out(ac.in(lat)))
  codes [T, 32]  int16     MimiModel's codes of the same latent (CB0 semantic, CB1-31 acoustic)
in shards of about --shard_frames frames:  <out>/<subset>/shard_<n>.lat.npy, .codes.npy and .json (one row per
utterance: id, speaker, chapter, subset, offset, frames, samples, text). A shard is written atomically (tmp + rename),
so an interrupted job resumes at the first missing shard.

Batches hold utterances of similar length, right-padded. Mimi's encoder is causal (causal convolutions, a causal
transformer), so an utterance's frames do not depend on the padding after it, except its last frames: their receptive
field reaches past the end, where a batch holds the deeper layers' response to zero input but a lone utterance holds
each layer's own zero padding (seen on job 5625: one of 7 codes differed). So the last TRIM frames of every utterance
are not stored, and the first batch is checked against one-by-one encoding on the stored frames; if those differ,
the job falls back to one-by-one encoding (exact, slower).

    python encode_mimi.py --root .../LibriTTS_R --subsets train-clean-100,dev-clean --out .../mvc/mimi
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FRAME_RATE = 12.5
SR = 24000
HOP = int(SR / FRAME_RATE)          # 1920 samples per Mimi frame
TRIM = 2                            # trailing frames not stored (padding-dependent)


def find_utterances(root, subset):
    """LibriTTS(-R) layout <subset>/<speaker>/<chapter>/<speaker>_<chapter>_<a>_<b>.wav with .normalized.txt."""
    base = Path(root) / subset
    out = []
    for wav in sorted(base.glob('*/*/*.wav')):
        spk, chap = wav.parent.parent.name, wav.parent.name
        txt = wav.with_suffix('.normalized.txt')
        out.append(dict(id=wav.stem, speaker=spk, chapter=chap, subset=subset, path=str(wav),
                        text=txt.read_text().strip() if txt.exists() else ''))
    return out


def n_frames(samples):
    """Mimi frames of an input of `samples` samples (the encoder pads the last frame)."""
    return -(-samples // HOP)


def stored_frames(samples):
    """Frames kept per utterance: all but the last TRIM."""
    return max(0, n_frames(samples) - TRIM)


def first_diff(a, b, tol=1e-3):
    """Index of the first frame where latents a, b [T, D] differ by more than tol, or None."""
    bad = np.nonzero(np.abs(a - b).max(1) > tol)[0]
    return int(bad[0]) if len(bad) else None


def batches(lengths, max_batch_samples, max_batch=32):
    """Positions grouped by similar length; each group's padded size stays under max_batch_samples."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > max_batch_samples or len(cur) >= max_batch:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def shard_plan(utts, frames, shard_frames):
    """Consecutive (speaker-sorted) utterances grouped into shards of about shard_frames frames."""
    shards, cur, tot = [], [], 0
    for i, u in enumerate(utts):
        cur.append(i)
        tot += frames[i]
        if tot >= shard_frames:
            shards.append(cur)
            cur, tot = [], 0
    if cur:
        shards.append(cur)
    return shards


class MimiEncoder:
    """Mimi's encoder path, exactly as transformers' MimiModel._encode_frame, also returning the latent."""

    def __init__(self, device='cuda', model=None):
        import torch
        self.torch = torch
        self.device = device
        if model is None:
            from transformers import MimiModel
            model = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=torch.float32).to(device).eval()
        self.m = model

    def __call__(self, wavs):
        """wavs: list of 1-d float arrays (24 kHz) -> list of (lat [T, 512] float32, codes [T, 32] int64)."""
        torch, m = self.torch, self.m
        lens = [len(w) for w in wavs]
        L = max(lens)
        x = torch.zeros(len(wavs), 1, L, device=self.device)
        for i, w in enumerate(wavs):
            x[i, 0, :len(w)] = torch.as_tensor(w, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            emb = m.encoder(x)
            h = m.encoder_transformer(emb.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
            lat = m.downsample(h)                                   # [B, 512, T]
            codes = m.quantizer.encode(lat).transpose(0, 1)         # [B, 32, T]
        out = []
        for i, n in enumerate(lens):
            t = n_frames(n)
            out.append((lat[i, :, :t].T.float().cpu().numpy(), codes[i, :, :t].T.cpu().numpy()))
        return out


def write_shard(path_base, rows, lats, codes):
    """Atomic write of one shard: <base>.lat.npy, .codes.npy, then .json (the json marks the shard complete)."""
    base = str(path_base)
    lat = np.concatenate(lats).astype(np.float16)
    cod = np.concatenate(codes).astype(np.int16)
    for suf, arr in (('.lat.npy', lat), ('.codes.npy', cod)):
        tmp = base + suf + '.tmp'
        with open(tmp, 'wb') as f:
            np.save(f, arr)
        os.replace(tmp, base + suf)
    tmp = base + '.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(tmp, base + '.json')


def load_shard(path_base):
    base = str(path_base)
    return (json.load(open(base + '.json')), np.load(base + '.lat.npy', mmap_mode='r'),
            np.load(base + '.codes.npy', mmap_mode='r'))


def read_wav(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32', always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w


def encode_subset(enc, root, subset, out, shard_frames=400000, max_batch_samples=24000 * 240, max_sec=30.0,
                  check_first=True, log=print, reader=read_wav, fallback=True):
    utts = find_utterances(root, subset)
    if not utts:
        raise SystemExit(f'{subset}: no wav files under {root}/{subset}')
    import soundfile as sf
    samples = [sf.info(u['path']).frames if reader is read_wav else len(reader(u['path'])) for u in utts]
    keep = [i for i, n in enumerate(samples) if 0.5 * SR <= n <= max_sec * SR]
    utts, samples = [utts[i] for i in keep], [samples[i] for i in keep]
    frames = [stored_frames(n) for n in samples]
    plan = shard_plan(utts, frames, shard_frames)
    odir = Path(out) / subset
    odir.mkdir(parents=True, exist_ok=True)
    log(f'{subset}: {len(utts)} utterances ({sum(samples) / SR / 3600:.1f} h, {len(set(u["speaker"] for u in utts))} '
        f'speakers) in {len(plan)} shards')
    checked, max_batch = not check_first, 32
    for si, idx in enumerate(plan):
        base = odir / f'shard_{si:04d}'
        if Path(str(base) + '.json').exists():
            continue
        t0 = time.time()
        wavs = {i: reader(utts[i]['path']) for i in idx}
        res = {}
        for b in batches([samples[i] for i in idx], max_batch_samples, max_batch):
            items = [idx[j] for j in b]
            outs = enc([wavs[i] for i in items]) if len(items) > 1 else [enc([wavs[items[0]]])[0]]
            for i, o in zip(items, outs):
                res[i] = o
            if not checked and len(items) > 1:
                for i in items[:3] + items[-2:]:
                    lat1, cod1 = enc([wavs[i]])[0]
                    t = frames[i]
                    d = float(np.abs(lat1[:t] - res[i][0][:t]).max())
                    same = float((cod1[:t] == res[i][1][:t]).mean())
                    log(f'batch check {utts[i]["id"]} ({len(lat1)} frames, {t} stored): max |lat diff| {d:.2e}, '
                        f'code agreement {same:.4f}, first differing frame {first_diff(lat1, res[i][0])}')
                    if d > 1e-2 or same < 0.99:
                        if not fallback:
                            raise SystemExit('batched encoding differs from one-by-one encoding on stored frames; stop')
                        log('batched encoding differs on stored frames: falling back to one-by-one encoding')
                        max_batch = 1
                        for k in items:
                            res[k] = enc([wavs[k]])[0]
                        break
                checked = True
        rows, lats, cods, off = [], [], [], 0
        for i in idx:
            lat, cod = res[i][0][:frames[i]], res[i][1][:frames[i]]
            assert lat.shape[0] == frames[i] == cod.shape[0], (utts[i]['id'], lat.shape, frames[i])
            r = {k: utts[i][k] for k in ('id', 'speaker', 'chapter', 'subset', 'text')}
            r.update(offset=off, frames=frames[i], samples=samples[i])
            rows.append(r)
            lats.append(lat)
            cods.append(cod)
            off += frames[i]
        write_shard(base, rows, lats, cods)
        log(f'{subset} shard {si + 1}/{len(plan)}: {len(idx)} utts, {off} frames, {time.time() - t0:.0f}s')
    return len(plan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--subsets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard_frames', type=int, default=400000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    enc = MimiEncoder(a.device)
    for s in a.subsets.split(','):
        encode_subset(enc, a.root, s, a.out, a.shard_frames, log=lambda m: print(m, flush=True))
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
