#!/usr/bin/env python3
"""
Truth-set evaluation of CRT decoding rules (idea B1 in paper/journal_extension/crt_math_ideas.md; round 2 = T2.2 a-f of
paper/journal_extension/crt_math_ideas_verified.md).

Self-conversion pairs (pairs/self.jsonl): a target speaker's own recording of a shared-text sentence, which make_pairs
keeps out of that speaker's reference pool, converted to the same speaker with the pair's own reference. The ideal
output is the source itself, so the truth is its Mimi codes (src_codes in the run_conversions dump).

For every rule of crt_decode_lab.py (the same code, with decoding replaced by returning the latent), per pair:
  err   mean over frames of ||W (a_hat - a_true)||^2, W = the acoustic output_proj (decoder-space squared error)
  nerr  err / err(faiss) pooled over pairs (below 1 = closer to the truth than FAISS)
  acc   token accuracy vs the truth on CB1-18 (code-valued rules only)
  energy  mean ||W a_hat||^2 / mean ||W a_true||^2 (well below 1 = over-smoothed: a posterior mean can lower the
          squared error by shrinking toward the codebook centre while sounding muffled, so judge with the audio metrics)
  err_full / nerr_full / energy_full  the same on the full decoder input Wsem e0[c0] + W a, so rules that change CB0
          (rerank, lmbr, path rules) are scored on the semantic part too (T2.2 a)
Grids: temperature T for the posterior mean (continuous and re-quantised), shrinkage weight l toward FAISS. Round-2
code-valued lab systems are included; src0_* / mnn* / thr* output or test with the source CB0, which IS the truth CB0
here, so they are sanity rows only.
Round 2 (T2.2), with speaker = the target of the pair id (self__p225_014 -> p225):
  a. per-frame and per-pair errors and energies of every system -> <out>_frames.npz
  b. NLL of the true CB1-18 tokens under softmax(l / T): global T, per-level T_k, T_k + copy bias beta_k on the
     retrieved logit before / T (concave in beta; mu shrinks toward FAISS by s = 1 / (1 + (e^(beta/T) - 1) p(r))), T_k +
     a log pi_T - b log pi_s (kernel-smoothed target-reference unigram over the speaker-balanced reference unigram);
     speaker-grouped 2 x 5-fold CV, held-out delta NLL with a pair-bootstrap CI
  c. levels as players, delta_k = W(x_k - e_k(r_k)) with x_k = posterior mean at T* (= the global T of b, fitted by
     NLL; the squared error of pm_cont_T keeps falling with T by over-smoothing, so its grid best is not used), the
     argmax codeword or the codebook mean (CRT-free control): exact Shapley values phi_k = -<delta_k, 2 r + sum_l
     delta_l> (r = W(a_F - a_true)), w*_k = -g_k / G_kk, cosine -g_k / sqrt(G_kk L(0)), box-QP min_{w in [0,1]^n}
     2 g'w + w'Gw (L-BFGS-B) and the best of all 2^n subsets; one-sided speaker-level t tests of phi_k > 0 with Holm.
     Systems boxqp_cont / boxqp_rq / boxqpH_cont (weights per entropy decile) / boxqp_cent_cont / shapS_argmax / chow
     are cross-fitted: T*, weights, subsets and tau come from the other speaker half
  d. per-level FAISS token accuracy on all pairs -> tau_chow.json next to <out> (crt_decode_lab --tau_chow)
  e. published gate: mean exact single-swap gain -2 <W delta, W(a_F - a_true)> - ||W delta||^2 per conf decile
  f. oracle headroom: per-frame best of HEAD, best of the 8 candidate frames (vertex) and of their convex hull
     (simplex) on the full decoder input, and the same with the candidate sets shuffled across frames (null)
Writes <out>.json, <out>.md, <out>_frames.npz and tau_chow.json.
"""
import argparse
import functools
import glob
import json
import os
import re
import sys

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crt_decode_lab as lab_mod  # noqa: E402

TS = (0.25, 0.5, 0.75, 1.0, 1.5)
LS = (0.25, 0.5, 0.75, 1.0)
TGRID = 2.0 ** (np.arange(-16, 17) / 8)             # calibration temperatures 0.25..4, exactly 1 at index 16
PRIOR_A, PRIOR_B, PRIOR_K = (0.0, 0.25, 0.5, 1.0), (0.0, 0.25, 0.5), 8   # prior family grid; kernel = codeword + 7 NN
SANITY = ('src0_', 'mnn', 'thr')
HEAD = {'faiss': 'faiss', 'tau0.4': 'tau0.4', 'pm_rq_T*': r'pm_rq_T[\d.]+', 'shrink*': r'shrink[\d.]+_(cont|rq)',
        'knn4_rq': 'knn4_rq', 'lmbr*': r'lmbr_.*', 'rerank_g1': 'rerank_g1'}
SECOND = ('boxqp_cont', 'boxqp_rq', 'boxqpH_cont', 'boxqp_cent_cont', 'shapS_argmax', 'chow')
NB = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dumps', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conv_dir', default='', help='manifest dir for reference file boundaries (default <x>/<set>/'
                   '<budget> for --dumps <x>/dumps/<set>/<budget>; without one, only encoding chunks are boundaries)')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def systems():
    s = ['faiss', 'faiss_rq', 'tau0.2', 'tau0.4', 'tau0.6', 'argmax', 'nbr8_tau0.4', 'pm18_rq', 'spk0.5_pm_rq',
         'rerank_g0', 'rerank_g1', 'knn4_cont', 'knn4_rq', 'knncrt_rq']
    s += [f'pm_cont_T{t:g}' for t in TS] + [f'pm_rq_T{t:g}' for t in TS]
    s += [f'shrink{l:g}_cont' for l in LS] + [f'shrink{l:g}_rq' for l in LS]
    # round-2 code-valued rules; chow runs once tau_chow is fitted (second pass), gv / pois are latent-valued
    s += [x for x in lab_mod.DEFAULT_SYSTEMS_R2.split(',') if x not in s and not x.startswith(('chow', 'gv', 'pois'))]
    return s


def speaker_of(pid):
    """Target speaker of a pair id: <set>__<utt>__<target>, or <set>__<spk>_<utt> for self-conversion pairs."""
    p = pid.split('__')
    return p[2] if len(p) > 2 else p[1].split('_')[0]


def manifest_rows(a):
    d = a.conv_dir or os.path.join(*os.path.abspath(a.dumps).rsplit(os.sep + 'dumps' + os.sep, 1))
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, 'manifest*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get('system') == 'faiss' and not r.get('error'):
                    rows[r['pair_id']] = r
    return rows


def unigram(lab, counts):
    """Kernel-smoothed (a codeword and its PRIOR_K - 1 nearest in decoder space), add-1 token distribution per level."""
    sm = np.stack([c[lab.nbrs(k, PRIOR_K).cpu().numpy()].sum(1) for k, c in enumerate(counts)])
    return (sm + 1.0) / (sm.sum(1, keepdims=True) + sm.shape[1])


def ref_counts(ref, n_cb):
    return np.stack([np.bincount(ref[:, 1 + k], minlength=2048) for k in range(n_cb)])


def truth(lab, z, W):
    torch, dev = lab.torch, lab.device
    src = torch.as_tensor(z['src_codes'].astype(np.int64), device=dev)                  # [32, T]
    retr = torch.as_tensor(z['retrieved_codes'].astype(np.int64), device=dev)
    a_true, a_F = lab.latent(src[1:]), lab.latent(retr[1:])
    return {'src': src, 'retr': retr, 'y': a_true @ W.T, 's0': lab.E0dec[src[0]], 'aF': a_F, 'aFd': a_F @ W.T,
            'n_cb': len(z['crt_logits'])}


def record(PF, acc, s, lab, tr, cb0, yh, codes=None):
    """Per-frame decoder-space errors and energies of one output: CB0 tokens cb0 [T], acoustic W a_hat yh [T, 512]."""
    y0 = lab.E0dec[cb0]
    q = {'err_ac': ((yh - tr['y']) ** 2).sum(1), 'err_full': ((y0 - tr['s0'] + yh - tr['y']) ** 2).sum(1),
         'en_ac': (yh ** 2).sum(1), 'en_full': ((y0 + yh) ** 2).sum(1), 'cb0_changed': (cb0 != tr['retr'][0]).float()}
    for k, v in q.items():
        PF.setdefault(s, {}).setdefault(k, []).append(v.cpu().numpy().astype(np.float64))
    if codes is not None:
        n = tr['n_cb']
        acc.setdefault(s, []).append((codes[1:1 + n] == tr['src'][1:1 + n]).float().mean(1).cpu().numpy())


def deltas(lab, logits, retr, ts=()):
    """Single-level moves delta_k = W(x_k - e_k(r_k)) [n_cb, T, 512] for x_k = the posterior mean at each T of ts
    (key T), the argmax codeword and the usage-weighted codebook mean ('cent'), each with the frame entropy (mean
    over levels, at T for the posterior means, at T = 1 otherwise); plus the argmax tokens and max-softmax at T = 1."""
    torch, dev = lab.torch, lab.device
    n_cb = len(logits)
    ar = torch.arange(n_cb, device=dev)[:, None]
    Ed = lab.Edec[:n_cb]
    er = Ed[ar, retr[1:1 + n_cb]]
    ent = lambda P: -(P * torch.log(P.clamp(min=1e-30))).sum(-1).mean(0)                # noqa: E731
    out = {}
    for T in ts:
        P = torch.as_tensor(lab_mod.softmax(logits, T), device=dev)
        out[T] = (torch.einsum('ktc,kcd->ktd', P, Ed) - er, ent(P))
    P = torch.as_tensor(lab_mod.softmax(logits, 1.0), device=dev)
    (conf, t1), H1 = P.max(-1), ent(P)
    out['argmax'] = (Ed[ar, t1] - er, H1)
    out['cent'] = (lab.ebar_dec[:n_cb, None] - er, H1)
    return out, t1, conf


def proj_simplex(v):
    u = v.sort(1, descending=True).values
    css = u.cumsum(1) - 1
    rho = (u - css / v.new_tensor(np.arange(1, v.shape[1] + 1)) > 0).sum(1, keepdim=True)
    return (v - css.gather(1, rho - 1) / rho).clamp(min=0)


def cand_oracles(torch, Yr, yt, cand, iters=500):
    """Per frame: min over the candidate frames (vertex) and over their convex hull (simplex: FISTA projected gradient
    started at the best vertex, result capped by it) of ||y - y_true||^2 on the full decoder input."""
    Y = Yr[cand].double()                                                               # [T, C, 512]
    dv = ((Y - yt[:, None]) ** 2).sum(-1)
    vert = dv.min(1).values
    Q, c = Y @ Y.transpose(1, 2), (Y @ yt[..., None])[..., 0]
    st = 1 / torch.linalg.eigvalsh(Q)[:, -1:].clamp(min=1e-12)
    x = v = torch.nn.functional.one_hot(dv.argmin(1), Y.shape[1]).double()
    t = 1.0
    for _ in range(iters):
        xn = proj_simplex(v - st * ((Q @ v[..., None])[..., 0] - c))
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v, x, t = xn + (t - 1) / tn * (xn - x), xn, tn
    simp = ((x[:, None] @ Y)[:, 0] - yt).pow(2).sum(-1)
    return vert.cpu().numpy(), simp.clamp(max=vert).cpu().numpy()


@functools.lru_cache(None)
def subsets(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64)


def game_fit(g, G, L0, binary=True):
    """min over w in [0,1]^n (L-BFGS-B) and over binary w (all 2^n) of L(w) / L0 = 1 + (2 g'w + w'Gw) / L0."""
    g, G = g / max(L0, 1e-30), G / max(L0, 1e-30)
    f = lambda w: (2 * g @ w + w @ G @ w, 2 * g + 2 * G @ w)                             # noqa: E731
    w = scipy.optimize.minimize(f, np.zeros(len(g)), jac=True, method='L-BFGS-B', bounds=[(0, 1)] * len(g),
                                options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 2000}).x
    out = {'w': w, 'risk_rel': 1 + f(w)[0]}
    if binary:
        X = subsets(len(g))
        J = 2 * X @ g + ((X @ G) * X).sum(1)
        out.update(S=X[J.argmin()].astype(bool), S_risk_rel=1 + J.min())
    return out


def shapley(g, G, L):
    """phi_k / L0 (exact Shapley value of the error reduction), w*_k = -g_k / G_kk and the cosine; batched over the
    leading axes."""
    dG = np.diagonal(G, axis1=-2, axis2=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -(2 * g + G.sum(-1)) / L[..., None], -g / dG, -g / np.sqrt(dG * L[..., None])


def holm(p):
    p = np.nan_to_num(p, nan=1.0)
    o = np.argsort(p)
    adj = np.empty_like(p)
    adj[o] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    return adj


def lst(x, nd=4):
    return [lst(v, nd) for v in x] if np.ndim(x) else (round(float(x), nd) if np.isfinite(x) else None)


def level_game(g, G, L0, spk_f, n_spk, rng):
    """Shapley values, w*, cosines, box-QP and best subset of one game from per-frame g [N, n], G [N, n, n], L0 [N];
    speaker-cluster bootstrap CIs, one-sided t tests over speakers of phi_k > 0 with Holm."""
    gs = np.stack([g[spk_f == s].sum(0) for s in range(n_spk)])
    Gs = np.stack([G[spk_f == s].sum(0) for s in range(n_spk)])
    Ls = np.array([L0[spk_f == s].sum() for s in range(n_spk)])
    phi, ws, cs = shapley(gs.sum(0), Gs.sum(0), Ls.sum())
    cnt = np.stack([np.bincount(r, minlength=n_spk) for r in rng.integers(0, n_spk, (NB, n_spk))]).astype(np.float64)
    ci = [np.nanpercentile(x, [2.5, 97.5], 0).T for x in shapley(cnt @ gs, np.einsum('bs,skl->bkl', cnt, Gs), cnt @ Ls)]
    with np.errstate(divide='ignore', invalid='ignore'):
        p = scipy.stats.ttest_1samp(shapley(gs, Gs, Ls)[0], 0, alternative='greater').pvalue
    fit = game_fit(gs.sum(0), Gs.sum(0), Ls.sum())
    ph = holm(p)
    return {'phi_rel': lst(phi), 'phi_rel_ci': lst(ci[0]), 'p_one_sided': lst(p), 'p_holm': lst(ph),
            'significant_holm': [bool(x < 0.05) for x in ph], 'w_star': lst(ws), 'w_star_ci': lst(ci[1]),
            'cos': lst(cs), 'cos_ci': lst(ci[2]), 'all_levels_risk_rel': lst(1 - phi.sum()),
            'boxqp_w': lst(fit['w']), 'boxqp_risk_rel': lst(fit['risk_rel']),
            'S_star': [int(k) + 1 for k in np.flatnonzero(fit['S'])], 'S_star_risk_rel': lst(fit['S_risk_rel'])}


def cal_arrays(CAL, off):
    """Frame arrays A [N, n_cb, G], ly, lr, eq of the calibration statistics and the per-pair NLL SP [P, n_cb, G] of
    the true tokens under softmax(l / T) for every T of TGRID."""
    A, ly, lr, eq = (np.concatenate(CAL[k]).astype(np.float64) for k in ('A', 'ly', 'lr', 'eq'))
    th = 1 / TGRID
    return A, ly, lr, eq, np.add.reduceat(np.logaddexp(A, th * lr[..., None]) - th * ly[..., None], off, 0)


def calibration(CAL, PN, fp, off, spk, rng):
    """T2.2 b: in-sample fits and speaker-grouped 2 x 5-fold CV of NLL models of the true CB1-18 tokens."""
    A, ly, lr, eq, SP = cal_arrays(CAL, off)
    PN = np.stack(PN).astype(np.float64)                                                 # [P, n_cb, G, nA, nB]
    th, n_cb, n_pairs = 1 / TGRID, A.shape[1], len(off)
    g1 = int(np.flatnonzero(TGRID == 1)[0])
    ntok = np.diff(np.r_[off, len(fp)]) * n_cb
    ks = np.arange(n_cb)

    def nll_tb(k, g, b):                                                                 # [N] at T index g, copy bias b
        return np.logaddexp(A[:, k, g], th[g] * lr[:, k] + b) - th[g] * ly[:, k] - b * eq[:, k]

    def fit_tb(trp):
        fr, par = trp[fp], []
        for k in ks:
            Ak, lrk, lyk, eqk = A[fr, k], lr[fr, k, None], ly[fr, k, None], eq[fr, k, None]
            b = np.zeros(len(th))
            for _ in range(100):                                                     # Newton on the concave log-lik
                s = scipy.special.expit(th * lrk + b - Ak)
                step = np.clip((eqk.sum() - s.sum(0)) / ((s * (1 - s)).sum(0) + 1e-12), -2, 2)
                b = np.clip(b + step, -20, 20)
                if np.abs(step).max() < 1e-10:
                    break
            g = int((np.logaddexp(Ak, th * lrk + b) - th * lyk - b * eqk).sum(0).argmin())
            par.append((g, b[g]))
        return par

    def ev_tb(par, tep):
        return np.add.reduceat(sum(nll_tb(k, g, b) for k, (g, b) in enumerate(par)), off)[tep]

    def fit_pr(trp):
        S = PN[trp].sum(0)
        ia, ib = np.unravel_index(S.min(1).sum(0).argmin(), S.shape[2:])
        return S[:, :, ia, ib].argmin(1), ia, ib

    models = {'T1': (lambda trp: None, lambda p, tep: SP[tep, :, g1].sum(1)),
              'T_global': (lambda trp: SP[trp].sum((0, 1)).argmin(), lambda g, tep: SP[tep, :, g].sum(1)),
              'T_level': (lambda trp: SP[trp].sum(0).argmin(1), lambda gk, tep: SP[tep][:, ks, gk].sum(1)),
              'T_level_beta': (fit_tb, ev_tb),
              'T_level_prior': (fit_pr, lambda p, tep: PN[tep][:, ks, p[0], p[1], p[2]].sum(1))}
    n_spk = spk.max() + 1
    nf = min(5, n_spk)
    held = {m: np.zeros(n_pairs) for m in models}
    for rep in range(2):
        fold = np.empty(n_spk, int)
        fold[rng.permutation(n_spk)] = np.arange(n_spk) % nf
        for f in range(nf):
            te = fold[spk] == f
            for m, (fit, ev) in models.items():
                held[m][te] += ev(fit(~te), te) / 2
    idx = rng.integers(0, n_pairs, (NB, n_pairs))

    def dstat(x, y):
        d, b = x - y, (x - y)[idx].sum(1) / ntok[idx].sum(1)
        ci = np.percentile(b, [2.5, 97.5])
        return {'delta_nll': lst(d.sum() / ntok.sum(), 5), 'ci': lst(ci, 5), 'keep': bool(ci[1] < 0)}
    allp = np.ones(n_pairs, bool)
    gG, gk, tb, pr = models['T_global'][0](allp), models['T_level'][0](allp), fit_tb(allp), fit_pr(allp)
    s_med = [float(np.median(1 / (1 + np.expm1(b) * scipy.special.expit(th[g] * lr[:, k] - A[:, k, g]))))
             for k, (g, b) in enumerate(tb)]
    tok_level = ntok.sum() / n_cb
    return {'n_tokens': int(ntok.sum()), 'cv': f'speaker-grouped 2 x {nf}-fold',
            'T_grid': [float(TGRID[0]), float(TGRID[-1]), len(TGRID)],
            'prior_grid': {'a': PRIOR_A, 'b': PRIOR_B, 'kernel_nn': PRIOR_K},
            'heldout_nll': {m: lst(h.sum() / ntok.sum(), 5) for m, h in held.items()},
            'delta_vs_T1': {m: dstat(held[m], held['T1']) for m in models if m != 'T1'},
            'terms': {'T_global over T1': dstat(held['T_global'], held['T1']),
                      'T_level over T_global': dstat(held['T_level'], held['T_global']),
                      'beta over T_level': dstat(held['T_level_beta'], held['T_level']),
                      'prior over T_level': dstat(held['T_level_prior'], held['T_level'])},
            'fit_all': {'T_global': float(TGRID[gG]), 'T_level': lst(TGRID[gk], 3),
                        'beta': {'T': lst(TGRID[[g for g, _ in tb]], 3), 'beta': lst([TGRID[g] * b for g, b in tb]),
                                 's_median': lst(s_med)},
                        'prior': {'T': lst(TGRID[pr[0]], 3), 'a': PRIOR_A[pr[1]], 'b': PRIOR_B[pr[2]]}},
            'nll_per_cb_T1': lst(SP[:, :, g1].sum(0) / tok_level),
            'nll_per_cb_T_level': lst(SP.sum(0)[ks, gk] / tok_level)}


def monotonicity(conf, gain, sw, fp, n_pairs, rng):
    """T2.2 e per level: mean single-swap gain per conf decile (pair-bootstrap CI), whether the deciles with a positive
    mean form one upper interval, the gain-maximising threshold, and the gains of opening there and at tau = 0.4."""
    idx = rng.integers(0, n_pairs, (NB, n_pairs))
    out = []
    for k in range(conf.shape[1]):
        c, d = conf[:, k], gain[:, k]
        b = np.digitize(c, np.quantile(c, np.arange(1, 10) / 10))
        S, N, Q = (np.zeros((n_pairs, 10)) for _ in range(3))
        np.add.at(S, (fp, b), d)
        np.add.at(N, (fp, b), 1)
        np.add.at(Q, (fp, b), sw[:, k])
        with np.errstate(invalid='ignore', divide='ignore'):
            m, bm = S.sum(0) / N.sum(0), S[idx].sum(1) / N[idx].sum(1)
        pos = np.flatnonzero(m > 0)
        up = not len(pos) or pos[0] + len(pos) == 10                    # the empty set is an upper interval too
        o = np.argsort(-c)
        cs = np.cumsum(d[o])
        j = int(cs.argmax())
        out.append({'decile_mean': lst(m, 5), 'decile_ci': lst(np.nanpercentile(bm, [2.5, 97.5], 0).T, 5),
                    'decile_conf_mid': lst([np.median(c[b == i]) if (b == i).any() else np.nan for i in range(10)], 3),
                    'swap_share': lst(Q.sum(0) / np.maximum(N.sum(0), 1), 3),
                    'positive_deciles': [int(i) + 1 for i in pos],
                    'single_upper_interval': bool(up),
                    'gate': 'never' if not len(pos) else 'upper' if up else 'non-monotone',
                    'best_threshold': lst(c[o][j], 4) if cs[j] > 0 else None,
                    'gain_best': lst(cs[j] / len(d) if cs[j] > 0 else 0.0, 5),
                    'gain_tau0.4': lst(d[c > 0.4].sum() / len(d), 5),
                    'spearman_conf_gain': lst(scipy.stats.spearmanr(c, d)[0])})
    return out


def headroom(PF, names, off, ORC, rng):
    """T2.2 f on the full decoder-input error: per-frame best of HEAD, candidate-frame vertex / simplex oracles and
    their shuffled-candidate nulls, each against the best fixed system of HEAD."""
    E = {s: np.concatenate(PF[s]['err_full']) for s in names}
    K = {k: min((s for s in names if re.fullmatch(p, s)), key=lambda s: E[s].sum()) for k, p in HEAD.items()
         if any(re.fullmatch(p, s) for s in names)}
    M = np.stack([E[s] for s in K.values()])
    best = int(M.sum(1).argmin())
    n_pairs = len(off)
    idx = rng.integers(0, n_pairs, (NB, n_pairs))

    def gain(x, y):                                                     # 1 - sum x / sum y, pair-bootstrap CI
        px, py = np.add.reduceat(x, off), np.add.reduceat(y, off)
        return {'gain': lst(1 - px.sum() / py.sum()),
                'ci': lst(np.percentile(1 - px[idx].sum(1) / py[idx].sum(1), [2.5, 97.5]))}
    F, B = E['faiss'], M[best]
    orc = {'best_of_K': M.min(0), **{k: np.concatenate(v) for k, v in ORC.items()}}
    out = {'K': K, 'nerr_full': {k: lst(E[s].sum() / F.sum()) for k, s in K.items()}, 'best_fixed': list(K)[best],
           'win_share': {k: lst(np.mean(M.argmin(0) == i), 3) for i, k in enumerate(K)}}
    for k, v in orc.items():
        out[k] = {'nerr_full': lst(v.sum() / F.sum()), 'gain_vs_best_fixed': gain(v, B)}
    for k in ('vertex', 'simplex'):
        out[k]['gain_over_null'] = gain(orc[k], orc['null_' + k])
    # no null for best_of_K: permuting each system's frames breaks the frame difficulty all K share, so its min falls
    # far below the aligned best_of_K and would always kill; T2.2 g carries its own held-out kill rule instead
    out['null_scope'] = 'candidate oracles (vertex, simplex) only; kill_M51_M26 uses the 5 % arm alone'
    out['kill_M51_M26'] = out['best_of_K']['gain_vs_best_fixed']['gain'] < 0.05
    out['kill_M49_M40'] = (out['vertex']['gain_vs_best_fixed']['gain'] < 0.05
                           or out['vertex']['gain_over_null']['ci'][0] <= 0)
    return out


def game_stats(torch, GV, D, rr):
    """Append per-frame g [T, n], G [T, n, n] and entropy of every move set of deltas() to GV."""
    for v, (Dv, H) in D.items():
        d = GV.setdefault(v, {'g': [], 'G': [], 'H': []})
        d['g'].append(torch.einsum('ktd,td->tk', Dv, rr).cpu().numpy())
        d['G'].append(torch.einsum('ktd,ltd->tkl', Dv, Dv).cpu().numpy())
        d['H'].append(H.cpu().numpy())


def main():
    import torch
    torch.set_grad_enabled(False)   # real Mimi weights (e.g. output_proj) require grad; nothing here needs autograd
    a = parse_args()
    lab = lab_mod.Lab(a.device)
    lab.decode_codes = lambda codes32: ('codes', codes32)                 # return outputs instead of audio
    lab.decode_latent = lambda cb0, ac: ('lat', (cb0, ac))
    torch, dev = lab.torch, lab.device
    W = lab.ac.output_proj.weight[:, :, 0].detach().float()                          # [512, 256]
    names = systems()
    files = sorted(glob.glob(os.path.join(a.dumps, '*.npz')))
    pids = [os.path.basename(f)[:-4] for f in files]
    spk_names = sorted({speaker_of(p) for p in pids})
    spk = np.array([spk_names.index(speaker_of(p)) for p in pids])
    rows = manifest_rows(a)
    n_cb = len(np.load(files[0])['crt_logits'])
    ar = torch.arange(lab.E.shape[0], device=dev)[:, None]
    cnt = np.zeros((len(spk_names), n_cb, 2048))                           # speaker-balanced reference unigram
    for f, s in zip(files, spk):
        cnt[s] += ref_counts(np.load(f)['ref_codes'].astype(np.int64), n_cb)
    lps = torch.as_tensor(np.log(np.mean([unigram(lab, c) for c in cnt], 0)), device=dev, dtype=torch.float32)
    PF, acc, ET, nll_T = {}, {}, {'ac': [], 'full': []}, {t: [] for t in TS}
    CAL, PN = {k: [] for k in ('A', 'ly', 'lr', 'eq')}, []
    GV = {v: {'g': [], 'G': [], 'H': []} for v in ('argmax', 'cent')}            # + posterior means at T* after b
    L0, CONF, SW, ORC = [], [], [], {k: [] for k in ('vertex', 'simplex', 'null_vertex', 'null_simplex')}
    for i, f in enumerate(files):
        z = dict(np.load(f))
        tr = truth(lab, z, W)
        ET['ac'].append((tr['y'] ** 2).sum(1).cpu().numpy().astype(np.float64))
        ET['full'].append(((tr['y'] + tr['s0']) ** 2).sum(1).cpu().numpy().astype(np.float64))
        for s, ((kind, obj), _) in lab_mod.run_pair(lab, z, names, row=rows.get(pids[i])).items():
            if kind == 'codes':
                record(PF, acc, s, lab, tr, obj[0], lab.latent(obj[1:]) @ W.T, obj)
            else:
                record(PF, acc, s, lab, tr, obj[0], obj[1] @ W.T)
        # b. calibration statistics (A = log sum over c != r of exp(l_c / T) per grid T: every T / copy-bias model)
        logits = z['crt_logits'].astype(np.float32)
        L = torch.as_tensor(logits, device=dev)
        y, r = tr['src'][1:1 + n_cb], tr['retr'][1:1 + n_cb]
        ly, lr = L.gather(-1, y[..., None])[..., 0], L.gather(-1, r[..., None])[..., 0]
        Lm = L.scatter(-1, r[..., None], float('-inf'))
        CAL['A'].append(torch.stack([torch.logsumexp(Lm / T, -1) for T in TGRID], -1).transpose(0, 1).cpu().numpy())
        CAL['ly'].append(ly.T.cpu().numpy())
        CAL['lr'].append(lr.T.cpu().numpy())
        CAL['eq'].append((y == r).T.float().cpu().numpy())
        for t in TS:
            nll_T[t].append(float((torch.logsumexp(L / t, -1) - ly / t).sum()))
        ref = z['ref_codes'].astype(np.int64)
        lpt = torch.as_tensor(np.log(unigram(lab, ref_counts(ref, n_cb))), device=dev, dtype=torch.float32)
        pri = torch.stack([pa * lpt - pb * lps for pa in PRIOR_A for pb in PRIOR_B])[:, :, None]   # [nA nB, n_cb, 1, C]
        yi = y[None, ..., None].expand(len(pri), -1, -1, 1)
        PN.append(torch.stack([(torch.logsumexp(x, -1) - x.gather(-1, yi)[..., 0]).sum(-1)
                               for x in (L[None] / T + pri for T in TGRID)], -1)
                  .reshape(len(PRIOR_A), len(PRIOR_B), n_cb, len(TGRID)).permute(2, 3, 0, 1).cpu().numpy())
        # c / e. per-frame games of the levels
        D, t1, conf = deltas(lab, logits, tr['retr'])
        rr = tr['aFd'] - tr['y']                                                            # W(a_F - a_true)
        L0.append((rr ** 2).sum(1).cpu().numpy())
        game_stats(torch, GV, D, rr)
        CONF.append(conf.T.cpu().numpy())
        SW.append((t1 != r).T.float().cpu().numpy())
        # f. candidate-frame oracles on the full decoder input, and with candidate sets shuffled across frames
        rc = torch.as_tensor(ref.T.copy(), device=dev)
        Yr = lab.E0dec[rc[0]] + lab.Edec[ar, rc[1:]].sum(0)
        yt = (tr['s0'] + tr['y']).double()
        cand = z['cand_idx'].astype(np.int64)
        for k, c in (('', cand), ('null_', cand[np.random.default_rng(i).permutation(len(cand))])):
            vx, sx = cand_oracles(torch, Yr, yt, torch.as_tensor(c, device=dev))
            ORC[k + 'vertex'].append(vx)
            ORC[k + 'simplex'].append(sx)
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(files)}', flush=True)
    n_pairs = len(files)
    nfr = np.array([len(x) for x in L0])
    off = np.r_[0, np.cumsum(nfr)[:-1]]
    fp = np.repeat(np.arange(n_pairs), nfr)
    L0 = np.concatenate(L0).astype(np.float64)
    # d. per-level FAISS token accuracy (= acc_per_cb of faiss) on all pairs for the lab's chow gate
    tau = np.stack(acc['faiss']).mean(0)
    tau_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), 'tau_chow.json')
    json.dump({'tau': [float(x) for x in tau]}, open(tau_path, 'w'))
    # c. cross-fit on speaker halves: T* (global NLL T of b), tau, box-QP weights (global, per entropy decile), subsets
    rng = np.random.default_rng(1)
    half = np.zeros(len(spk_names), int)
    half[rng.permutation(len(spk_names))[len(spk_names) // 2:]] = 1
    half = half[spk]
    tau_h = [np.stack(acc['faiss'])[half == h].mean(0) for h in (0, 1)]
    SP = cal_arrays(CAL, off)[-1]                                           # = the T_global fit of calibration()
    T_all = float(TGRID[SP.sum((0, 1)).argmin()])
    T_h = [float(TGRID[SP[half == h].sum((0, 1)).argmin()]) for h in (0, 1)]
    MU = {T: {'err': [], 'en': []} for T in {T_all, *T_h}}
    for f in files:                                                         # posterior-mean games at the fitted T*
        z = dict(np.load(f))
        tr = truth(lab, z, W)
        D, _, _ = deltas(lab, z['crt_logits'].astype(np.float32), tr['retr'], tuple(MU))
        game_stats(torch, GV, {T: D[T] for T in MU}, tr['aFd'] - tr['y'])
        for T in MU:
            yh = tr['aFd'] + D[T][0].sum(0)
            MU[T]['err'].append(float(((yh - tr['y']) ** 2).sum(1).mean()))
            MU[T]['en'].append(float((yh ** 2).sum(1).mean()))
    GV = {v: {k: np.concatenate(x).astype(np.float64) for k, x in d.items()} for v, d in GV.items()}
    fits = []
    for h in (0, 1):
        m, T = half[fp] == h, T_h[h]
        fit = {'T': T, 'v': 'T%g' % T, **{u: game_fit(GV[x]['g'][m].sum(0), GV[x]['G'][m].sum(0), L0[m].sum())
                                          for u, x in (('mu', T), ('argmax', 'argmax'), ('cent', 'cent'))}}
        H = GV[T]['H'][m]
        fit['edges'] = np.quantile(H, np.arange(1, 10) / 10)
        b = np.digitize(H, fit['edges'])
        fit['wH'] = np.stack([game_fit(GV[T]['g'][m][b == j].sum(0), GV[T]['G'][m][b == j].sum(0),
                                       L0[m][b == j].sum(), binary=False)['w'] for j in range(10)])
        fits.append(fit)
    qcheck = []
    for i, f in enumerate(files):                                           # second pass: cross-fitted systems, chow
        z = dict(np.load(f))
        tr = truth(lab, z, W)
        logits = z['crt_logits'].astype(np.float32)
        fit = fits[1 - half[i]]
        v = fit['T']
        D, t1, _ = deltas(lab, logits, tr['retr'], (v,))
        Dm, H = D[v]
        w = torch.as_tensor(fit['mu']['w'], device=dev, dtype=torch.float32)
        record(PF, acc, 'boxqp_cont', lab, tr, tr['retr'][0], tr['aFd'] + torch.einsum('k,ktd->td', w, Dm))
        sl, wn = slice(off[i], off[i] + nfr[i]), fit['mu']['w']                  # direct error vs the quadratic form
        q = L0[sl] + 2 * GV[v]['g'][sl] @ wn + np.einsum('tkl,k,l->t', GV[v]['G'][sl], wn, wn)
        qcheck.append(float(np.abs(q - PF['boxqp_cont']['err_ac'][-1]).max() / L0.mean()))
        P = torch.as_tensor(lab_mod.softmax(logits, v), device=dev)
        a_hat = tr['aF'] + sum(w[k] * (P[k] @ lab.E[k] - lab.E[k][tr['retr'][1 + k]]) for k in range(n_cb))
        c = tr['retr'].clone()
        c[1:] = lab.rq(a_hat)
        record(PF, acc, 'boxqp_rq', lab, tr, c[0], lab.latent(c[1:]) @ W.T, c)
        wb = torch.as_tensor(fit['wH'][np.digitize(H.cpu().numpy(), fit['edges'])], device=dev, dtype=torch.float32)
        record(PF, acc, 'boxqpH_cont', lab, tr, tr['retr'][0], tr['aFd'] + torch.einsum('tk,ktd->td', wb, Dm))
        wc = torch.as_tensor(fit['cent']['w'], device=dev, dtype=torch.float32)
        record(PF, acc, 'boxqp_cent_cont', lab, tr, tr['retr'][0],
               tr['aFd'] + torch.einsum('k,ktd->td', wc, D['cent'][0]))
        c = tr['retr'].clone()
        S = torch.as_tensor(fit['argmax']['S'], device=dev)
        c[1:1 + n_cb][S] = t1[S]
        record(PF, acc, 'shapS_argmax', lab, tr, c[0], lab.latent(c[1:]) @ W.T, c)
        lab.tau_chow = tau_h[1 - half[i]].astype(np.float32)
        (_, c), _ = lab_mod.run_pair(lab, z, ['chow'], row=rows.get(pids[i]))['chow']
        record(PF, acc, 'chow', lab, tr, c[0], lab.latent(c[1:]) @ W.T, c)
    names = names + list(SECOND)
    err, errf, energy, enf = ({s: np.array([x.mean() for x in PF[s][q]]) for s in names}
                              for q in ('err_ac', 'err_full', 'en_ac', 'en_full'))
    e_true, e_true_full = (np.array([x.mean() for x in ET[q]]) for q in ('ac', 'full'))
    base, basef = np.mean(err['faiss']), np.mean(errf['faiss'])
    out = {'n_pairs': len(files), 'systems': {}}
    rng = np.random.default_rng(0)
    for s in names:
        e = np.array(err[s])
        r = e / np.array(err['faiss'])
        idx = rng.integers(0, len(e), size=(2000, len(e)))
        nb = e[idx].mean(1) / np.array(err['faiss'])[idx].mean(1)
        nbf = errf[s][idx].mean(1) / errf['faiss'][idx].mean(1)
        d = {'err': float(e.mean()), 'nerr': float(e.mean() / base),
             'nerr_ci': [float(np.percentile(nb, 2.5)), float(np.percentile(nb, 97.5))],
             'share_pairs_better_than_faiss': float((r < 1).mean()),
             'energy_ratio': float(np.mean(energy[s]) / np.mean(e_true)),
             'err_full': float(errf[s].mean()), 'nerr_full': float(errf[s].mean() / basef),
             'nerr_full_ci': [float(np.percentile(nbf, 2.5)), float(np.percentile(nbf, 97.5))],
             'share_pairs_better_than_faiss_full': float((errf[s] < errf['faiss']).mean()),
             'energy_full_ratio': float(np.mean(enf[s]) / np.mean(e_true_full)),
             'cb0_changed': float(np.concatenate(PF[s]['cb0_changed']).mean())}
        if acc.get(s):
            A = np.stack(acc[s])
            d['acc_cb1_18'] = float(A.mean())
            d['acc_per_cb'] = [round(float(x), 4) for x in A.mean(0)]
        m = re.fullmatch(r'pm_(cont|rq)_T([\d.]+)', s)
        if m:
            d['nll_cb1_18'] = float(np.sum(nll_T[float(m.group(2))]) / (nfr.sum() * n_cb))
        if s.startswith(SANITY):
            d['sanity_only'] = True
        if s in SECOND:
            d['cross_fit'] = True
        out['systems'][s] = d
    rng = np.random.default_rng(2)
    out['calibration'] = calibration(CAL, PN, fp, off, spk, rng)
    v_all = 'T%g' % T_all
    mu = lambda T: {'nerr': lst(np.mean(MU[T]['err']) / base),                                      # noqa: E731
                    'energy_ratio': lst(np.mean(MU[T]['en']) / np.mean(e_true))}
    edge = lambda T, g: bool(T in (g[0], g[-1]))                                                    # noqa: E731
    tb = min(TS, key=lambda t: out['systems'][f'pm_cont_T{t:g}']['nerr'])
    db = out['systems'][f'pm_cont_T{tb:g}']
    out['levels'] = {
        'T_star': v_all, 'T_star_value': T_all, 'T_star_rule': 'global T of the calibration (b), NLL over all pairs',
        'T_star_at_grid_edge': edge(T_all, TGRID), 'mu_T_star': mu(T_all),
        'best_pm_cont_T_grid': {'T': tb, 'at_grid_edge': edge(tb, TS), 'nerr': lst(db['nerr']),
                                'energy_ratio': lst(db['energy_ratio'])},
        'unit': 'L(0) = summed FAISS acoustic frame error',
        'games': {k: level_game(GV[x]['g'], GV[x]['G'], L0, spk[fp], len(spk_names), rng)
                  for k, x in (('mu_' + v_all, T_all), ('argmax', 'argmax'), ('cent', 'cent'))},
        'interaction_rel': lst(GV[T_all]['G'].sum(0) / L0.sum()),
        'cross_fit': [{'fit_speakers': [spk_names[j] for j in sorted(set(spk[half == h]))], 'T_star': ft['v'],
                       'T_star_value': ft['T'], 'T_star_at_grid_edge': edge(ft['T'], TGRID), 'mu_T_star': mu(ft['T']),
                       'tau_chow': lst(tau_h[h]), 'boxqp_w': lst(ft['mu']['w']), 'boxqp_w_cent': lst(ft['cent']['w']),
                       'S_star_argmax': [int(k) + 1 for k in np.flatnonzero(ft['argmax']['S'])],
                       'entropy_edges': lst(ft['edges']), 'wH': lst(ft['wH'])} for h, ft in enumerate(fits)],
        'boxqp_quadratic_check_max_rel': max(qcheck)}
    out['tau_chow'] = {'file': tau_path, 'tau': lst(tau)}
    gain = -2 * GV['argmax']['g'] - np.diagonal(GV['argmax']['G'], axis1=1, axis2=2)    # exact single-swap gain
    out['gate_monotonicity'] = {'unit': 'mean FAISS acoustic frame error', 'levels': monotonicity(
        np.concatenate(CONF), gain / L0.mean(), np.concatenate(SW), fp, n_pairs, rng)}
    out['oracle_headroom'] = headroom(PF, [s for s in names if not s.startswith(SANITY)], off, ORC, rng)
    out['frames_npz'] = a.out + '_frames.npz'
    json.dump(out, open(a.out + '.json', 'w'), indent=1)
    np.savez_compressed(a.out + '_frames.npz', systems=np.array(names), pair_ids=np.array(pids),
                        speakers=np.array([spk_names[s] for s in spk]), frame_pair=fp, n_frames=nfr, L0=L0,
                        **{k: np.stack([np.concatenate(PF[s][k]) for s in names]).astype(np.float32)
                           for k in ('err_ac', 'err_full', 'en_ac', 'en_full', 'cb0_changed')},
                        **{'pair_' + k: np.stack([x[s] for s in names]) for k, x in
                           (('err_ac', err), ('err_full', errf), ('en_ac', energy), ('en_full', enf))},
                        **{'en_true_' + k: np.concatenate(x) for k, x in ET.items()},
                        **{'oracle_' + k: np.concatenate(x) for k, x in ORC.items()},
                        conf=np.concatenate(CONF), swap_gain=gain)
    L = [f'# Truth-set (self-conversion) latent error, {len(files)} pairs', '',
         'nerr = decoder-space squared latent error relative to FAISS (below 1 = closer to the truth); 95 % bootstrap '
         'CI over pairs; acc = token accuracy vs the truth on CB1-18.', '',
         '| rule | nerr | 95 % CI | pairs better than FAISS | energy | acc CB1-18 | acc CB1 | acc CB18 |',
         '|---|---|---|---|---|---|---|---|']
    tag = lambda s: s + (' (sanity)' if s.startswith(SANITY) else ' (cross-fit)' if s in SECOND else '')  # noqa
    f3 = lambda x, f='.3f': 'nan' if x is None else format(x, f)                                             # noqa
    for s, d in sorted(out['systems'].items(), key=lambda kv: kv[1]['nerr']):
        ac = d.get('acc_per_cb')
        L.append(f"| {tag(s)} | {d['nerr']:.3f} | [{d['nerr_ci'][0]:.3f}, {d['nerr_ci'][1]:.3f}] | "
                 f"{d['share_pairs_better_than_faiss']:.2f} | {d['energy_ratio']:.2f} | "
                 f"{d.get('acc_cb1_18', float('nan')):.3f} | "
                 f"{ac[0] if ac else float('nan'):.3f} | {ac[-1] if ac else float('nan'):.3f} |")
    L += ['', '## Full decoder-input error (T2.2 a)', '',
          'nerr_full = squared error of the full decoder input Wsem e0[c0] + W a relative to FAISS (95 % bootstrap CI '
          'over pairs); energy_full likewise; CB0 changed = share of frames whose CB0 is not the retrieved one; NLL = '
          'mean NLL of the true CB1-18 tokens under softmax(logits / T). (sanity) rows use the source CB0, which is '
          'the truth here; (cross-fit) rows are fitted on the other speaker half.', '',
          '| rule | nerr_full | 95 % CI | pairs better than FAISS | energy_full | CB0 changed | NLL CB1-18 |',
          '|---|---|---|---|---|---|---|']
    for s, d in sorted(out['systems'].items(), key=lambda kv: kv[1]['nerr_full']):
        L.append(f"| {tag(s)} | {d['nerr_full']:.3f} | [{d['nerr_full_ci'][0]:.3f}, {d['nerr_full_ci'][1]:.3f}] | "
                 f"{d['share_pairs_better_than_faiss_full']:.2f} | {d['energy_full_ratio']:.2f} | "
                 f"{d['cb0_changed']:.2f} | {d.get('nll_cb1_18', float('nan')):.3f} |")
    C = out['calibration']
    fa = C['fit_all']
    L += ['', '## Calibration of the CRT heads (T2.2 b)', '',
          f"Held-out NLL per true CB1-18 token ({C['n_tokens']} tokens, {C['cv']} CV); delta vs the published T = 1 "
          'and per added term with a 95 % pair-bootstrap CI; keep = CI below 0.', '',
          '| model | held-out NLL | delta vs T = 1 | 95 % CI |', '|---|---|---|---|']
    for m, h in C['heldout_nll'].items():
        dd = C['delta_vs_T1'].get(m)
        L.append(f'| {m} | {h:.4f} | ' + (f"{dd['delta_nll']:+.4f} | [{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}] |"
                                          if dd else '0 | |'))
    L += ['', '| term | delta NLL | 95 % CI | keep |', '|---|---|---|---|']
    L += [f"| {m} | {dd['delta_nll']:+.4f} | [{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}] | "
          f"{'yes' if dd['keep'] else 'no'} |" for m, dd in C['terms'].items()]
    L += ['', f"Fitted on all pairs: global T = {fa['T_global']:.3f}; prior family a = {fa['prior']['a']:g}, "
          f"b = {fa['prior']['b']:g}. beta_k = copy bias on the retrieved logit before the temperature T_k; s_k = "
          'median adaptive shrinkage weight of the copy bias (mu_k - e_k(r) is scaled by s).', '',
          '| CB | NLL at T = 1 | NLL at T_k | T_k | T_k with beta | beta_k | s_k |',
          '|---|---|---|---|---|---|---|']
    for k in range(n_cb):
        L.append(f"| {k + 1} | {f3(C['nll_per_cb_T1'][k])} | {f3(C['nll_per_cb_T_level'][k])} | "
                 f"{f3(fa['T_level'][k], '.2f')} | {f3(fa['beta']['T'][k], '.2f')} | "
                 f"{f3(fa['beta']['beta'][k], '+.3f')} | {f3(fa['beta']['s_median'][k])} |")
    lv, G_ = out['levels'], out['levels']['games']
    gm, ga, gc = G_['mu_' + v_all], G_['argmax'], G_['cent']
    L += ['', '## Levels as players: Shapley values and joint shrinkage (T2.2 c)', '',
          f"T* = {T_all:.3f}, the global T of (b) fitted by NLL on all pairs"
          + (' (WARNING: at the edge of the T grid)' if lv['T_star_at_grid_edge'] else '')
          + f"; the posterior mean at T* has nerr {f3(lv['mu_T_star']['nerr'])}, energy "
          f"{f3(lv['mu_T_star']['energy_ratio'], '.2f')}. The best pm_cont_T of the squared-error grid is T = {tb:g}"
          + (' (WARNING: at the grid edge, so not an optimum; a larger T lowers the error by over-smoothing)'
             if lv['best_pm_cont_T_grid']['at_grid_edge'] else '')
          + f" with nerr {db['nerr']:.3f}, energy {db['energy_ratio']:.2f}; it is not used as T*. "
          'Values relative to L(0), the FAISS acoustic error. phi = exact Shapley '
          "value of the error reduction from CRT's move at a level (sum = 1 - L(all) / L(0)), 95 % speaker-bootstrap "
          f'CI; p = one-sided t test of phi > 0 over {len(spk_names)} speakers, Holm over {n_cb}; w* = -g_k / G_kk; '
          'cos = -g_k / sqrt(G_kk L(0)); box w = joint weight in [0, 1]; S* = best of all binary subsets. argmax / '
          'cent = the argmax codeword / the CRT-free codebook mean as the move.', '',
          '| CB | phi | 95 % CI | p Holm | w* | cos | box w | in S* | phi argmax | in S* argmax | phi cent | '
          'box w cent |', '|---' * 12 + '|']
    for k in range(n_cb):
        L.append(f"| {k + 1} | {f3(gm['phi_rel'][k], '.4f')} | [{f3(gm['phi_rel_ci'][k][0], '.4f')}, "
                 f"{f3(gm['phi_rel_ci'][k][1], '.4f')}] | {f3(gm['p_holm'][k])} | {f3(gm['w_star'][k])} | "
                 f"{f3(gm['cos'][k])} | {f3(gm['boxqp_w'][k])} | {'x' if k + 1 in gm['S_star'] else ''} | "
                 f"{f3(ga['phi_rel'][k], '.4f')} | {'x' if k + 1 in ga['S_star'] else ''} | "
                 f"{f3(gc['phi_rel'][k], '.4f')} | {f3(gc['boxqp_w'][k])} |")
    L += ['', '| game | L(all) / L(0) | box-QP L(w) / L(0) | L(S*) / L(0) |', '|---|---|---|---|']
    L += [f"| {m} | {f3(g_['all_levels_risk_rel'], '.4f')} | {f3(g_['boxqp_risk_rel'], '.4f')} | "
          f"{f3(g_['S_star_risk_rel'], '.4f')} |" for m, g_ in G_.items()]
    L += ['', 'Cross-fit (T*, weights, subsets and the chow tau fitted on one speaker half, applied to the other; rows '
          'marked (cross-fit) above): ' + '; '.join(
              f"half {h}: T* = {c_['T_star_value']:.3f}" + (' (grid edge)' if c_['T_star_at_grid_edge'] else '')
              + f" (energy {f3(c_['mu_T_star']['energy_ratio'], '.2f')}), S* argmax = {c_['S_star_argmax']}"
              for h, c_ in enumerate(lv['cross_fit'])) +
          f"; quadratic-form check {lv['boxqp_quadratic_check_max_rel']:.1e}."]
    L += ['', '## Per-level FAISS token accuracy (T2.2 d)', '', f'Written to {tau_path} for crt_decode_lab --tau_chow: '
          + ', '.join(f'CB{k + 1} {x:.3f}' for k, x in enumerate(tau)) + ' (all pairs; the chow row above uses the '
          'other speaker half).']
    L += ['', '## Monotonicity of the published gate (T2.2 e)', '',
          'Mean exact single-swap gain -2 <W delta, W(a_F - a_true)> - ||W delta||^2 (delta = e(argmax) - '
          'e(retrieved); > 0 = the swap lowers the error) per max-softmax decile d1-d10, in % of the mean FAISS '
          'acoustic frame error; upper = the positive deciles form one upper interval (never = no positive decile, '
          'the gate should never open); best = gain-maximising '
          'threshold and the mean gain of opening the gate there, vs at tau = 0.4.', '',
          '| CB | ' + ' | '.join(f'd{j}' for j in range(1, 11)) + ' | upper | best threshold | gain best | '
          'gain tau 0.4 |', '|---' * 15 + '|']
    for k, m_ in enumerate(out['gate_monotonicity']['levels']):
        L.append(f'| {k + 1} | ' + ' | '.join(f3(None if x is None else 100 * x, '.2f') for x in m_['decile_mean'])
                 + f" | {dict(upper='yes', never='never').get(m_['gate'], 'no')} | {f3(m_['best_threshold'])} | "
                 f"{100 * m_['gain_best']:.2f} | {100 * m_['gain_tau0.4']:.2f} |")
    H_ = out['oracle_headroom']
    L += ['', '## Oracle headroom (T2.2 f)', '',
          'Full decoder-input error. best_of_K = per-frame best of '
          + ', '.join(f'{k} = {s}' for k, s in H_['K'].items()) + f"; best fixed = {H_['best_fixed']}. vertex / "
          'simplex = best candidate frame / best convex combination of the 8 candidate frames; null_* = the same with '
          'the candidate sets shuffled across frames (the null exists for the candidate oracles only). gain = 1 - '
          'err / err(best fixed), 95 % pair-bootstrap CI.', '',
          '| oracle | nerr_full | gain vs best fixed | 95 % CI |', '|---|---|---|---|']
    for k in ('best_of_K', 'vertex', 'simplex', 'null_vertex', 'null_simplex'):
        gg = H_[k]['gain_vs_best_fixed']
        L.append(f"| {k} | {H_[k]['nerr_full']:.3f} | {gg['gain']:+.3f} | [{gg['ci'][0]:+.3f}, {gg['ci'][1]:+.3f}] |")
    vn = H_['vertex']['gain_over_null']
    L += ['', f"Vertex gain over its null {vn['gain']:+.3f} [{vn['ci'][0]:+.3f}, {vn['ci'][1]:+.3f}]; kill M51/M26 "
          f"(best-of-K gain < 5 %; no null, T2.2 g has its own held-out kill rule): {H_['kill_M51_M26']}; kill M49/M40 "
          f"(vertex gain < 5 % or not above its null): {H_['kill_M49_M40']}."]
    open(a.out + '.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
