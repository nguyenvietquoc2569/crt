#!/usr/bin/env python3
"""Optional continuous content stream for Mimi-VC: Whisper-small encoder features at Mimi's frame rate.

Whisper's encoder emits one 768-d vector per 20 ms (50 Hz) of 16 kHz audio, so 4 consecutive vectors cover exactly
one 80 ms Mimi frame: features(wav) averages them into [T_mimi, 768]. Utterances are at most 30 s (Whisper's window);
the output is cut to the utterance's own frames.

As a corpus pass, for each shard and each perturbed draw k it re-creates the perturbed audio with the same seeds as
encode_perturbed.py (utt_seed(seed, id, k)) and stores <base>.pwhisper<k>.npy [N, 768] float16, frame-aligned with the
clean shard (stored frames only). Whisper features are rich and also carry speaker information, so the stream is only
ever computed on perturbed audio for training, and its leakage is checked on Dev (SIM_src) before it is kept.

    python whisper_feats.py --shards .../mimi/train-clean-100 --raw .../LibriTTS_R --K 1 --workers 8
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import encode_mimi as em
import encode_perturbed as ep

MODEL = 'openai/whisper-small'
SUB = 4                                   # 50 Hz Whisper frames per 12.5 Hz Mimi frame


class WhisperFeat:
    def __init__(self, device='cuda', model=None, fe=None):
        import torch
        self.torch, self.device = torch, device
        if model is None:
            from transformers import WhisperFeatureExtractor, WhisperModel
            model = WhisperModel.from_pretrained(MODEL, torch_dtype=torch.float32).encoder
            fe = WhisperFeatureExtractor.from_pretrained(MODEL)
        self.enc, self.fe = model.to(device).eval(), fe

    def __call__(self, wav24k):
        import librosa
        torch = self.torch
        w = librosa.resample(np.asarray(wav24k, dtype=np.float32), orig_sr=em.SR, target_sr=16000)
        if len(w) > 30 * 16000:
            raise ValueError('longer than Whisper\'s 30 s window')
        mel = self.fe(w, sampling_rate=16000, return_tensors='pt').input_features.to(self.device)
        with torch.no_grad():
            h = self.enc(mel).last_hidden_state[0].float().cpu().numpy()          # [1500, 768] for 30 s
        T = em.n_frames(len(wav24k))
        h = h[:T * SUB]
        if len(h) < T * SUB:
            h = np.pad(h, ((0, T * SUB - len(h)), (0, 0)), mode='edge')
        return h.reshape(T, SUB, -1).mean(1)


def _perturbed(args):
    path, seed, reader, fn = args
    return fn(reader(path), em.SR, np.random.default_rng(seed))


def feats_shard(wf, base, raw, K, seed, pool=None, reader=em.read_wav, perturb_fn=ep.default_perturb, log=print):
    outs = [f'{base}.pwhisper{k}.npy' for k in range(K)]
    if all(os.path.exists(o) for o in outs):
        return False
    rows, _, codes = em.load_shard(base)
    for k, o in enumerate(outs):
        if os.path.exists(o):
            continue
        jobs = [(ep.wav_path(raw, r), ep.utt_seed(seed, r['id'], k), reader, perturb_fn) for r in rows]
        wavs = list(pool.map(_perturbed, jobs, chunksize=4)) if pool else [_perturbed(j) for j in jobs]
        F = np.zeros((len(codes), 768), np.float16)
        for r, w in zip(rows, wavs):
            F[r['offset']:r['offset'] + r['frames']] = wf(w)[:r['frames']]
        tmp = o + '.tmp'
        with open(tmp, 'wb') as f:
            np.save(f, F)
        os.replace(tmp, o)
        log(f'{base}: whisper features of perturbed draw {k}, {len(rows)} utts')
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    wf = WhisperFeat(a.device)
    with ProcessPoolExecutor(a.workers) as pool:
        for d in a.shards.split(','):
            for js in sorted(Path(d).glob('shard_*.json')):
                feats_shard(wf, str(js)[:-5], a.raw, a.K, a.seed, pool, log=lambda m: print(m, flush=True))
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
