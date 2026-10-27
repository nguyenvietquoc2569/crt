#!/usr/bin/env python3
"""DNSMOS P.835 (SIG, BAK, OVRL) and P.808 MOS: a second, non-SSL naturalness predictor beside UTMOS (26 Oct, user
approved the download). A port of Microsoft's DNS-Challenge DNSMOS/dnsmos_local.py (non-personalised models
DNSMOS/sig_bak_ovr.onnx and DNSMOS/model_v8.onnx, in --models): 16 kHz audio, clips shorter than 9.01 s repeated to
9.01 s, 9.01 s windows at 1 s hops, polynomial calibration of the P.835 outputs, means over windows.

    python dnsmos.py --models .../dnsmos --rows .../metrics/dev_30.jsonl --out .../dnsmos/dev_30.jsonl

--norm_lufs L first brings every clip to integrated loudness L LUFS (ITU-R BS.1770, gated; pyloudnorm): DNSMOS rates
quieter clips higher, and the systems compared differ in level by several dB.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SR = 16000
INPUT_LENGTH = 9.01
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
FIELDS = ('SIG', 'BAK', 'OVRL', 'P808')


class DNSMOS:
    def __init__(self, models, threads=1):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = threads
        self.primary = ort.InferenceSession(str(Path(models) / 'sig_bak_ovr.onnx'), o, providers=['CPUExecutionProvider'])
        self.p808 = ort.InferenceSession(str(Path(models) / 'model_v8.onnx'), o, providers=['CPUExecutionProvider'])

    @staticmethod
    def melspec(audio):
        import librosa
        m = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(m, ref=np.max) + 40) / 40).T

    def score(self, audio):
        """Scores of a 16 kHz float signal."""
        audio = np.asarray(audio, np.float32)
        n = int(INPUT_LENGTH * SR)
        while len(audio) < n:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
        acc = {k: [] for k in FIELDS}
        for i in range(hops):
            seg = audio[i * SR: i * SR + n]
            if len(seg) < n:
                continue
            sig, bak, ovr = self.primary.run(None, {'input_1': seg[None].astype(np.float32)})[0][0]
            p808 = self.p808.run(None, {'input_1': self.melspec(seg[:-160]).astype(np.float32)[None]})[0][0][0]
            acc['SIG'].append(P_SIG(sig))
            acc['BAK'].append(P_BAK(bak))
            acc['OVRL'].append(P_OVR(ovr))
            acc['P808'].append(p808)
        return {k: float(np.mean(v)) for k, v in acc.items()}


def load16(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR, 'HQ')
    return x


def normalize_lufs(x, target):
    """x scaled to `target` LUFS integrated loudness (BS.1770 gating), or unchanged if it is silent."""
    import pyloudnorm
    lo = pyloudnorm.Meter(SR).integrated_loudness(np.asarray(x, np.float64))
    if not np.isfinite(lo):
        return x
    return (np.asarray(x, np.float64) * 10 ** ((target - lo) / 20)).astype(np.float32)


def summarize(rows):
    out = dict(n=len(rows), n_error=sum('error' in r for r in rows))
    for k in FIELDS:
        v = [r[k] for r in rows if k in r]
        out[k] = float(np.mean(v)) if v else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', required=True)
    ap.add_argument('--rows', required=True, help='metric rows (jsonl with wav and pair_id)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--key', default='wav', help='row field holding the audio path')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--norm_lufs', type=float, default=None, help='loudness-normalise to this LUFS first')
    a = ap.parse_args(argv)
    m = DNSMOS(a.models, a.threads)
    res = []
    for l in open(a.rows):
        r = json.loads(l)
        try:
            x = load16(r[a.key])
            if a.norm_lufs is not None:
                x = normalize_lufs(x, a.norm_lufs)
            res.append(dict(pair_id=r['pair_id'], system=r.get('system'), **m.score(x)))
        except Exception as e:  # noqa: BLE001 - recorded, the summary counts it
            res.append(dict(pair_id=r.get('pair_id'), system=r.get('system'), error=f'{type(e).__name__}: {e}'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    s = summarize(res)
    Path(str(a.out)[:-len('.jsonl')] + '.summary.json' if a.out.endswith('.jsonl') else a.out + '.summary.json').write_text(
        json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
