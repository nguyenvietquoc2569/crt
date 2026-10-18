#!/usr/bin/env python3
"""
Stability of all 32 Mimi codebooks under small edits, for the CRT journal (CRT refines CB1-18; the mimi2vec pilot
reported CB0-7 only). Re-uses the interpreter project's same-slot copies read-only:
  <codes_dir>/{clean,pink20,gain3,rt32}/<uid>.npz, codes int16 [32, T]; an empty [32, 0] file marks a clipped copy.
Alignment as m2v_slot.py `count`: the frame shift s in (0, -1, 1, -2, 2) maximising CB0 agreement (first maximum
wins), clips below --min_agree CB0 agreement dropped.
Per copy and level k: flip rate (share of aligned frames whose token differs from the clean encode), share of flips
whose CB0..k-1 prefix is unchanged, and, from the Mimi codebooks (embedding_sum / cluster_usage through the level's
output_proj, the decoder-summed space), the share of flips that land among the clean token's K nearest codewords
(null (K-1)/2047) and the median flip distance relative to the median random-pair distance of that level.
Writes <out>.json and <out>.md.
"""
import argparse
import glob
import json
import os

import numpy as np

L, V, KS = 32, 2048, (2, 8, 32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--codes_dir', default='/Users/harry/Projects/NCS/interpreter/m2v/codes_slot')
    p.add_argument('--weights', default='/Users/harry/Projects/NCS/interpreter/m2v/weights/'
                                        'tokenizer-e351c8d8-checkpoint125.safetensors')
    p.add_argument('--versions', default='pink20,gain3,rt32')
    p.add_argument('--min_agree', type=float, default=0.3)
    p.add_argument('--out', required=True)
    return p.parse_args()


def codewords(path):
    """[32, 2048, 512] decoder-space codewords: CB0 from rvq_first, CB1-31 from rvq_rest."""
    from safetensors import safe_open
    f = safe_open(path, 'np')
    out = []
    for q, n in (('rvq_first', 1), ('rvq_rest', L - 1)):
        w = f.get_tensor(f'quantizer.{q}.output_proj.weight')[:, :, 0]                       # [512, 256]
        for i in range(n):
            pre = f'quantizer.{q}.vq.layers.{i}._codebook.'
            e = f.get_tensor(pre + 'embedding_sum') / np.maximum(f.get_tensor(pre + 'cluster_usage'), 1e-5)[:, None]
            out.append((e @ w.T).astype(np.float32))
    return np.stack(out)


def align(c, d):
    best = (-1.0, 0)
    for s in (0, -1, 1, -2, 2):
        lo, hi = max(0, -s), min(c.shape[1], d.shape[1] - s)
        if hi > lo:
            ag = float((c[0, lo:hi] == d[0, lo + s:hi + s]).mean())
            best = max(best, (ag, s), key=lambda z: z[0])
    ag, s = best
    lo, hi = max(0, -s), min(c.shape[1], d.shape[1] - s)
    return ag, c[:, lo:hi], d[:, lo + s:hi + s]


def main():
    a = parse_args()
    cw = codewords(a.weights)
    nn, dist, med = [], [], []
    for k in range(L):
        sq = (cw[k] ** 2).sum(1)
        dk = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * cw[k] @ cw[k].T, 0))
        o = dk.copy()
        np.fill_diagonal(o, -1)
        nn.append(np.argsort(o, 1)[:, :max(KS)])
        dist.append(dk)
        med.append(float(np.median(dk[np.triu_indices(V, 1)])))
    uids = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{a.codes_dir}/clean/*.npz'))
    res = {'clips_clean': len(uids), 'min_agree': a.min_agree, 'Ks': KS, 'copies': {}}
    for v in a.versions.split(','):
        frames, flips, prefix = 0, np.zeros(L, np.int64), np.zeros(L, np.int64)
        in_nbr = {K: np.zeros(L, np.int64) for K in KS}
        fd = [[] for _ in range(L)]
        n = {'clips': 0, 'missing': 0, 'clipped': 0, 'dropped': 0}
        for u in uids:
            f = f'{a.codes_dir}/{v}/{u}.npz'
            if not os.path.exists(f):
                n['missing'] += 1
                continue
            d = np.load(f)['codes'].astype(np.int64)
            if d.shape[1] == 0:
                n['clipped'] += 1
                continue
            c = np.load(f'{a.codes_dir}/clean/{u}.npz')['codes'].astype(np.int64)
            ag, cc, dd = align(c, d)
            if ag < a.min_agree:
                n['dropped'] += 1
                continue
            n['clips'] += 1
            diff = dd != cc
            same = np.cumprod(~diff, axis=0).astype(bool)
            frames += cc.shape[1]
            for k in range(L):
                m = diff[k]
                flips[k] += m.sum()
                prefix[k] += (m if k == 0 else m & same[k - 1]).sum()
                if m.any():
                    ct, dt = cc[k][m], dd[k][m]
                    for K in KS:
                        in_nbr[K][k] += (nn[k][ct, :K] == dt[:, None]).any(1).sum()
                    fd[k].append(dist[k][ct, dt] / med[k])
        res['copies'][v] = {**n, 'frames': int(frames),
                            'flip_rate': (flips / max(frames, 1)).tolist(),
                            'prefix_unchanged_share': (prefix / np.maximum(flips, 1)).tolist(),
                            'flip_in_nbr': {str(K): (in_nbr[K] / np.maximum(flips, 1)).tolist() for K in KS},
                            'flip_in_nbr_null': {str(K): (K - 1) / (V - 1) for K in KS},
                            'flip_dist_rel_median': [float(np.median(np.concatenate(x))) if x else None for x in fd]}
    with open(a.out + '.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    vs = list(res['copies'])
    lines = [f'# Mimi token stability, all 32 codebooks ({res["clips_clean"]} clips; same-slot copies of the mimi2vec '
             'pilot)', '', 'flip % per copy | flips landing in the clean token\'s 8 nearest codewords (null 0.34 %) | '
             'median flip distance / median random-pair distance', '',
             '| CB | ' + ' | '.join(f'flip {v}' for v in vs) + ' | ' + ' | '.join(f'in-nbr8 {v}' for v in vs)
             + ' | ' + ' | '.join(f'dist {v}' for v in vs) + ' |', '|---' * (1 + 3 * len(vs)) + '|']
    for k in range(L):
        r = [res['copies'][v] for v in vs]
        lines.append(f'| {k} | ' + ' | '.join(f"{x['flip_rate'][k] * 100:.1f}" for x in r) + ' | '
                     + ' | '.join(f"{x['flip_in_nbr']['8'][k] * 100:.1f}" for x in r) + ' | '
                     + ' | '.join(f"{x['flip_dist_rel_median'][k]:.2f}" if x['flip_dist_rel_median'][k] is not None
                                  else '–' for x in r) + ' |')
    with open(a.out + '.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print({v: {k: res['copies'][v][k] for k in ('clips', 'missing', 'clipped', 'dropped', 'frames')} for v in vs})


if __name__ == '__main__':
    main()
