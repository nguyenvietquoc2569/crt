#!/usr/bin/env python3
"""
Codeword clustering of CRT's probabilities: the feasibility check for a neighbourhood gate / soft decoding in the CRT
journal extension (idea from the mimi2vec discussion, 10 Oct 2026).

Question: when CRT (v5.6.5) is unsure, is its probability spread over codewords that are close in Mimi's decoder
space (so a gate on neighbourhood mass could act where the top-1 gate cannot), or over unrelated codewords?

Input: run_conversions.py dumps made with --dump_logits (crt_logits: n_cb x T x 2048 float16 for CB1-18, plus
retrieved_codes 32 x T). Codeword geometry, from the acoustic RVQ codebook vectors (embed_sum / cluster_usage, read
like run_conversions.cb0_table reads CB0; equal to layer.decode(arange(2048))), nearest neighbours per level (each
codeword is its own 1st neighbour), two variants:
  dec_l2   Euclidean after the acoustic quantizer's output projection, the space whose per-level vectors the
           decoder sums (primary)
  rvq_cos  cosine in the 256-d RVQ space before output_proj (the interpreter project's mimi2vec geometry, for
           cross-checking with its nbr_overlap_codewords numbers)

Per refined level and frame, with p = softmax(logits) in float32 and t = argmax p:
  p1        p[t], the published gate statistic
  nbr_K     mass on t and its K-1 nearest codewords
  top_K     mass on the K most probable tokens (the most any K-set can hold)
  rand_K    p[t] + (K-1)/2047 * (1 - p[t]): expected mass of t and K-1 uniformly random other codewords (null)
  CI_K      (mean nbr_K - mean rand_K) / (mean top_K - mean rand_K): 0 = no more clustered than chance,
            1 = the K most probable tokens are exactly t and its K-1 nearest codewords
Gate opening: share of frames with p1 > tau vs nbr_K > tau.
Move size: on frames where t differs from the retrieved (FAISS) token, the share where the retrieved token lies in
t's K-neighbourhood (null (K-1)/2047), and the t-to-retrieved distance relative to the median distance between
random codeword pairs of that level.
Orphaned overrides (geometry-free): at the published gate tau = 0.4, a level-k token is changed when p1 > tau and
t != retrieved; it is orphaned when the parent level k-1 keeps its retrieved token (gate closed) although CRT's
argmax there differs, so the output mixes CRT's child with a parent CRT did not predict. Mimi's residual levels form
a cascade (a deep token's meaning depends on the levels above), so orphans may be where gated overrides hurt.
Writes <out>.json and <out>.md.
"""
import argparse
import glob
import json
import os

import numpy as np

KS = (2, 4, 8, 16, 32)
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dumps', required=True, help='run_conversions dumps directory (<out_dir>/dumps/<set>/<budget>)')
    p.add_argument('--out', required=True, help='output path prefix (.json and .md are added)')
    p.add_argument('--mimi', default='kyutai/mimi')
    return p.parse_args()


def codeword_geometries(mimi_id):
    """{'dec_l2': [n_levels, 2048, 512] codewords after output_proj, 'rvq_cos': [n_levels, 2048, 256] L2-normalised
    RVQ-space codewords (Euclidean distance on them orders neighbours as cosine does)}, for CB1..CB31."""
    import torch
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained(mimi_id).float().eval()
    rvq = mimi.quantizer.acoustic_residual_vector_quantizer
    dec, cos = [], []
    with torch.no_grad():
        for layer in rvq.layers:
            cb = layer.codebook
            e = cb.embed_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]          # [2048, 256]
            cos.append(torch.nn.functional.normalize(e, dim=1).numpy().astype(np.float32))
            proj = getattr(rvq, 'output_proj', None)
            if proj is not None:
                e = e @ proj.weight[:, :, 0].T                                         # [2048, 512]; bias cancels
            dec.append(e.numpy().astype(np.float32))
    return {'dec_l2': np.stack(dec), 'rvq_cos': np.stack(cos)}


def neighbours(cw, kmax):
    """Indices [2048, kmax] of each codeword's kmax nearest codewords (itself first), and the median random-pair
    distance of the level."""
    sq = (cw ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * cw @ cw.T, 0.0))
    order = dist.copy()
    np.fill_diagonal(order, -1.0)
    nn = np.argsort(order, axis=1)[:, :kmax]
    iu = np.triu_indices(len(cw), 1)
    return nn, dist, float(np.median(dist[iu]))


def main():
    a = parse_args()
    files = sorted(glob.glob(os.path.join(a.dumps, '*.npz')))
    if not files:
        raise SystemExit(f'no dumps in {a.dumps}')
    geoms = codeword_geometries(a.mimi)
    acc = {}       # level -> lists of per-frame arrays
    argmax_agree = [0, 0]
    for f in files:
        z = np.load(f)
        if 'crt_logits' not in z:
            raise SystemExit(f'{f}: no crt_logits (run run_conversions.py with --dump_logits)')
        first = int(z['crt_first_cb'])
        lg = z['crt_logits'].astype(np.float32)                     # [n_cb, T, 2048]
        lg -= lg.max(-1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(-1, keepdims=True)
        t = p.argmax(-1)                                            # [n_cb, T]
        argmax_agree[0] += int((t == z['crt_argmax']).sum())
        argmax_agree[1] += t.size
        retr = z['retrieved_codes']
        for i in range(p.shape[0]):
            lv = first + i
            d = acc.setdefault(lv, {k: [] for k in ('p', 't', 'retr')})
            d['p'].append(p[i])
            d['t'].append(t[i])
            d['retr'].append(retr[lv])
    res = {'n_dumps': len(files), 'argmax_agreement_with_harness': argmax_agree[0] / argmax_agree[1],
           'Ks': KS, 'taus': TAUS, 'levels': {}}
    kmax = max(KS)
    for lv in sorted(acc):
        P = np.concatenate(acc[lv]['p'])                            # [N, 2048]
        T = np.concatenate(acc[lv]['t'])
        R = np.concatenate(acc[lv]['retr']).astype(np.int64)
        rows = np.arange(len(T))
        p1 = P[rows, T]
        cum_top = np.cumsum(-np.sort(-P, axis=1)[:, :kmax], axis=1)
        moved = T != R
        r = {'n_frames': int(len(T)), 'p1_median': float(np.median(p1)), 'p1_mean': float(p1.mean()),
             'gate_top1': {str(tau): float((p1 > tau).mean()) for tau in TAUS},
             'moved_share': float(moved.mean())}
        if lv - 1 in acc:   # orphaned overrides at tau 0.4 (frames align: both levels come from the same dumps)
            Pp = np.concatenate(acc[lv - 1]['p'])
            Tp = np.concatenate(acc[lv - 1]['t'])
            Rp = np.concatenate(acc[lv - 1]['retr']).astype(np.int64)
            changed = (p1 > 0.4) & moved
            orphan = changed & (Pp[rows, Tp] <= 0.4) & (Tp != Rp)
            r['changed_tau0.4'] = int(changed.sum())
            r['orphan_share_tau0.4'] = float(orphan.sum() / changed.sum()) if changed.any() else None
        for g, cws in geoms.items():
            r[g] = geometry_stats(P, T, R, p1, cum_top, moved, *neighbours(cws[lv - 1], kmax))
        res['levels'][str(lv)] = r
    with open(a.out + '.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    lines = [f'# Codeword clustering of CRT probabilities ({len(files)} pairs)', '',
             f'argmax from dumped float16 logits agrees with the harness argmax on '
             f'{res["argmax_agreement_with_harness"]:.4f} of tokens. Geometry dec_l2 = Euclidean after output_proj '
             '(primary); rvq_cos = cosine in the 256-d RVQ space.', '',
             '| CB | p1 median | nbr8 median | CI_4 | CI_8 | CI_16 | CI_8 rvq_cos | gate τ0.4 top1 → nbr8 | moved | '
             'FAISS token in nbr8 when moved (null) | orphaned overrides τ0.4 |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for lv, r in res['levels'].items():
        g = r['dec_l2']
        k8 = g['K']['8']
        ci = lambda gg, k: f"{gg['K'][k]['CI']:.2f}" if gg['K'][k]['CI'] is not None else '–'
        rin = f"{k8['retr_in_nbr_when_moved']:.3f}" if k8['retr_in_nbr_when_moved'] is not None else '–'
        orph = (f"{r['orphan_share_tau0.4']:.2f} of {r['changed_tau0.4']}" if r.get('orphan_share_tau0.4') is not None
                else '–')
        lines.append(f"| {lv} | {r['p1_median']:.3f} | {k8['nbr_median']:.3f} | {ci(g, '4')} | {ci(g, '8')} | "
                     f"{ci(g, '16')} | {ci(r['rvq_cos'], '8')} | {r['gate_top1']['0.4']:.3f} → "
                     f"{k8['gate_nbr']['0.4']:.3f} | {r['moved_share']:.3f} | {rin} ({k8['retr_in_nbr_null']:.4f}) | "
                     f"{orph} |")
    with open(a.out + '.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def geometry_stats(P, T, R, p1, cum_top, moved, nn, dist, med):
    """Neighbourhood statistics of one level under one codeword geometry."""
    rows = np.arange(len(T))
    cum_nbr = np.cumsum(P[rows[:, None], nn[T]], axis=1)           # [N, kmax], column k-1 = nbr_k
    r = {'move_dist_rel_median': float(np.median(dist[T[moved], R[moved]]) / med) if moved.any() else None, 'K': {}}
    for K in KS:
        nbr, top = cum_nbr[:, K - 1], cum_top[:, K - 1]
        rnd = p1 + (K - 1) / 2047.0 * (1.0 - p1)
        den = top.mean() - rnd.mean()
        in_nbr = (nn[T[moved], :K] == R[moved][:, None]).any(1) if moved.any() else np.array([])
        r['K'][str(K)] = {
            'nbr_mean': float(nbr.mean()), 'nbr_median': float(np.median(nbr)),
            'top_mean': float(top.mean()), 'rand_mean': float(rnd.mean()),
            'CI': float((nbr.mean() - rnd.mean()) / den) if den > 1e-9 else None,
            'gate_nbr': {str(tau): float((nbr > tau).mean()) for tau in TAUS},
            'retr_in_nbr_when_moved': float(in_nbr.mean()) if in_nbr.size else None,
            'retr_in_nbr_null': (K - 1) / 2047.0,
            'kth_nbr_dist_rel_median': float(np.median(dist[np.arange(2048), nn[:, K - 1]]) / med)}
    return r


if __name__ == '__main__':
    main()
