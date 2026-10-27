#!/usr/bin/env python3
"""Global speaker embeddings for Mimi-VC conditioning: CAM++ (3D-Speaker / FunASR), as Seed-VC uses it.

CAM++ is deliberately NOT one of the evaluation models (WavLM-base-plus-SV, ECAPA-TDNN), so conditioning on it cannot
inflate the reported similarity. Feature recipe as Seed-VC's inference: 16 kHz, kaldi fbank 80 bins, no dither,
per-utterance mean removed, CAMPPlus(feat_dim=80, embedding_size=192). The code is imported from the Seed-VC checkout
on ceai (modules/campplus), the weights are its campplus_cn_common.bin.

    python spk_embed.py --shards .../mvc/mimi/train-clean-100 --raw .../LibriTTS_R    # writes <base>.spk.npy
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

import encode_mimi as em

SEEDVC = '/datastore/ceai/crt/baselines/seed-vc'


class CamPlus:
    def __init__(self, device='cuda', seedvc=SEEDVC, ckpt=None, model=None):
        import torch
        import torchaudio
        self.torch, self.ta, self.device = torch, torchaudio, device
        if model is None:
            sys.path.insert(0, seedvc)
            from modules.campplus.DTDNN import CAMPPlus
            model = CAMPPlus(feat_dim=80, embedding_size=192)
            model.load_state_dict(torch.load(ckpt or os.path.join(seedvc, 'campplus_cn_common.bin'), map_location='cpu'))
        self.model = model.to(device).eval()

    def __call__(self, wav, sr=em.SR):
        torch = self.torch
        with torch.no_grad():
            w = torch.as_tensor(np.asarray(wav, dtype=np.float32))[None]
            if sr != 16000:
                w = self.ta.functional.resample(w, sr, 16000)
            feat = self.ta.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            return self.model(feat[None].to(self.device))[0].float().cpu().numpy()


def embed_shard(emb, base, raw, reader=em.read_wav, log=print):
    out = f'{base}.spk.npy'
    if os.path.exists(out):
        return False
    rows, _, _ = em.load_shard(base)
    from encode_perturbed import wav_path
    E = np.stack([emb(reader(wav_path(raw, r))) for r in rows]).astype(np.float32)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, E)
    os.replace(tmp, out)
    log(f'{base}: {len(rows)} speaker embeddings')
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    emb = CamPlus(a.device)
    for d in a.shards.split(','):
        for js in sorted(Path(d).glob('shard_*.json')):
            embed_shard(emb, str(js)[:-5], a.raw, log=lambda m: print(m, flush=True))
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
