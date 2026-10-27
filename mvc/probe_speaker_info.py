#!/usr/bin/env python3
"""Probe P0: how much speaker identity do Mimi's streams carry, and how much does perturbation remove?

On held-out speakers (e.g. LibriTTS-R dev-clean shards) every utterance becomes one feature vector per stream:
  cb0      bag of CB0 tokens (tf-idf, L2-normalised)        the content stream a VC model would condition on
  pcb0_k   the same for the CB0 of perturbed audio (encode_perturbed.py, draw k)
  cb1      bag of CB1 tokens (an acoustic level, for scale)
  lat      mean of the continuous latent (512-d, standardised)
Each speaker's CHAPTERS are split in half by a fixed seed (enrolment chapters vs test chapters), so book and chapter
vocabulary cannot carry the identification; speakers with a single chapter are left out. A transcript bag-of-words
stream ('text') is the lexical control: only accuracy above it is speaker information. Reported per stream:
speaker-ID accuracy of the nearest centroid (cosine) on the test chapters, chance 1/n_speakers, and the verification
EER of utterance-vs-centroid cosine scores. Lower values for pcb0 than for cb0 mean perturbation removed speaker
information; the content side is measured by the CB0 agreement that encode_perturbed.py logs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import encode_mimi as em


def bag(tokens, vocab=2048):
    return np.bincount(np.asarray(tokens, dtype=np.int64), minlength=vocab).astype(np.float64)


def tfidf(X):
    df = (X > 0).sum(0)
    idf = np.log((1 + len(X)) / (1 + df)) + 1
    Y = X * idf
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def standardise(X):
    Y = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def eer(genuine, impostor):
    s = np.concatenate([genuine, impostor])
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y) / max(1, y.sum())
    fp = np.cumsum(1 - y) / max(1, (1 - y).sum())
    fnr = 1 - tp
    i = int(np.argmin(np.abs(fnr - fp)))
    return float((fnr[i] + fp[i]) / 2)


def evaluate(F, speakers, seed=0, chapters=None):
    """F [N, D] L2-normalised features, speakers [N] -> nearest-centroid accuracy and EER on held-out data: the
    test chapters when chapters [N] are given (speakers with one chapter are left out), else a random utterance half."""
    rng = np.random.default_rng(seed)
    spk = np.asarray(speakers)
    names = sorted(set(speakers))
    enrol, test, kept = [], [], []
    for s in names:
        idx = np.nonzero(spk == s)[0]
        if chapters is not None:
            ch = sorted({chapters[i] for i in idx})
            if len(ch) < 2:
                continue
            rng.shuffle(ch)
            e_ch = set(ch[:len(ch) // 2])
            e = np.array([i for i in idx if chapters[i] in e_ch])
            t = [i for i in idx if chapters[i] not in e_ch]
        else:
            rng.shuffle(idx)
            h = max(1, len(idx) // 2)
            e, t = idx[:h], list(idx[h:])
        enrol.append(e)
        kept.append(s)
        test.extend((i, len(enrol) - 1) for i in t)
    names = kept
    C = np.stack([F[e].mean(0) for e in enrol])
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    ti = np.array([i for i, _ in test])
    ty = np.array([c for _, c in test])
    S = F[ti] @ C.T
    acc = float((S.argmax(1) == ty).mean())
    gen = S[np.arange(len(ty)), ty]
    mask = np.ones_like(S, dtype=bool)
    mask[np.arange(len(ty)), ty] = False
    return dict(acc=acc, chance=1 / len(names), eer=eer(gen, S[mask]), n_test=int(len(ty)), n_speakers=len(names))


def text_bag(texts):
    vocab = {}
    for t in texts:
        for w in t.lower().split():
            vocab.setdefault(w.strip('.,;:!?"\''), len(vocab))
    X = np.zeros((len(texts), max(1, len(vocab))))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            X[i, vocab[w.strip('.,;:!?"\'')]] += 1
    return X


def features(shard_dirs, K):
    rows_all, streams = [], {'cb0': [], 'cb1': [], 'lat': []}
    for k in range(K):
        streams[f'pcb0_{k}'] = []
    for d in shard_dirs:
        for js in sorted(Path(d).glob('shard_*.json')):
            base = str(js)[:-5]
            rows, lat, cod = em.load_shard(base)
            pc = [np.load(f'{base}.pcodes{k}.npy', mmap_mode='r') for k in range(K)]
            for r in rows:
                a, b = r['offset'], r['offset'] + r['frames']
                streams['cb0'].append(bag(cod[a:b, 0]))
                streams['cb1'].append(bag(cod[a:b, 1]))
                streams['lat'].append(np.asarray(lat[a:b], dtype=np.float64).mean(0))
                for k in range(K):
                    streams[f'pcb0_{k}'].append(bag(pc[k][a:b, 0]))
                rows_all.append(r)
    F = {name: (standardise(np.stack(v)) if name == 'lat' else tfidf(np.stack(v))) for name, v in streams.items()}
    F['text'] = tfidf(text_bag([r.get('text', '') for r in rows_all]))
    return F, [r['speaker'] for r in rows_all], [f"{r['speaker']}_{r.get('chapter', '')}" for r in rows_all]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    F, spk, chap = features(a.shards.split(','), a.K)
    res = {name: evaluate(X, spk, chapters=chap) for name, X in F.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(f"{name:8s} speaker-ID acc {r['acc']:.3f} (chance {r['chance']:.3f}), EER {r['eer']:.3f}, "
              f"{r['n_test']} test utts / {r['n_speakers']} speakers", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
