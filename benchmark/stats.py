#!/usr/bin/env python3
"""
Inference for the CRT journal extension: ANALYSIS_PLAN_v1.md §5.2 (crossed bootstrap, p-values, Holm, gate L, failed
conversions, secondary tests) and §5.3 (variance components, size check). CPU only; reads metrics.py rows (as
metrics.load_results) and make_pairs.py pair rows. No conversion or model is touched.

  Crossed bootstrap, B = 10,000 replicates, seed 20260928. A replicate draws speakers with replacement and contains
      every pair whose source and target were both drawn, weighted by the product of their draw counts:
        linked       Dev and LTC-2k (primary): one pool, the sources and targets of the pairs, drawn once; a speaker's
                     count serves both roles.
        independent  VCTK-2k; on Dev and LTC-2k v0's version, the sensitivity analysis: sources and targets drawn
                     separately (all source draws of a batch, then all target draws).
      The pool is the speakers of the pairs that enter the comparison. Every statistic is a function of weighted sums of
      per-pair columns, computed per replicate as c_src' M c_tgt from the cell sums M[s, t] over the pairs of source s
      and target t (the same as summing the pair weights). A replicate in which the endpoint has no pair of positive
      weight, or zero reference words, is redrawn from the same generator until it has both; the redraws are counted.
      Every call starts from the seed, so endpoints on the same pairs share their replicates.
  Continuous metrics (SIM, UTMOS, dB-MCD, SIM_src, M9): mean of the per-pair paired differences. Corpus WER / CER
      (metrics 'wer', 'cer'): 100 * total errors (S + D + I) / total reference words (N; cS + cD + cI over cN for CER)
      per system and replicate, the definition of metrics.corpus_wer (reimplemented, with metrics.load_results, because
      metrics.py imports librosa at import time; --self_test checks both against metrics.py). Capped corpus WER / CER
      (M3c, §0.3 row 20; metrics 'wer_capped', 'cer_capped', corpus_wer(capped=True)): each pair's errors are
      min(E_p, N_p) before any sum, so the cap does not depend on the replicate.
  95 % CI: percentile; SE: standard deviation of the replicates; two-sided p = min(1, 2 min(1 + #{Δ* <= 0},
      1 + #{Δ* >= 0}) / (B + 1)). Holm: reject the i-th smallest p while p_(i) <= alpha / (m - i + 1); a missing p
      counts in m and is not rejected. Gate L (H0 Δ_src,g >= +0.01): rejected when the upper CI limit is below +0.01;
      gate_l_strata runs it in the same- and cross-gender strata of a set (VCTK-2k: pairs whose source has an E set).
  Failed conversions (no row, or a row with an error) are left out of a comparison and counted per system; a row
      without error whose value is missing or not finite (e.g. dB-MCD with no frame left) is left out as undefined.
      When the two systems' failure counts differ, a worst-case sensitivity analysis: WER / CER every reference unit of
      the failed pair deleted (E = N, within the cap; N from the other system's row, else from n_ref), a continuous
      metric the failing system's worst finite value in the rows given (worst='min': SIM, SECS, UTMOS; 'max': dB-MCD,
      SIM_src).
  Secondary (v0): two-sided Wilcoxon signed-rank on the per-pair differences (per-pair WER % for WER, capped for
      wer_capped; zero differences dropped) and the matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-); possibly
      anti-conservative, since pairs share speakers.
  §5.3: REML fit of d_p = mu + a_source + b_target + c_source×target + e_p (statsmodels MixedLM in variance-component
      form, one group; fitted with each optimiser of VC_METHODS, the highest REML log-likelihood kept, an optimiser
      that fails left out and listed). For WER
      d_p = 100 (E_A,p - E_B,p) / N̄, N̄ the mean reference words of the pairs, so mean(d_p) is the corpus-WER
      difference (wer_linearised(capped=True): E = min(S + D + I, N) per pair, the capped difference). Predicted
      SE² = σa²/n_s + σb²/n_t + σc²/n_c + σe²/n with the counts of the test
      list's pairs that enter the endpoint; MDE = 3.53 SE against 2.0 WER points, 0.010 SIM, 0.05 UTMOS, 0.10 dB; gate L
      power 2.80 SE_g <= 0.01 - Δ_dev,g (no power under any design when Δ_dev,g >= 0.01); rebuild rule: VCTK-2k uses
      the 10-target variant iff at least one VCTK-2k condition fails with 5 targets and passes with 10.
CLI: python stats.py --self_test [--screen_dir paper/journal_extension/results] (unit tests, then cross-checks on the
50-pair screen files: metrics.corpus_wer, capped WER, the plan's §2.6 numbers, C1 vs FAISS, gate L, variance
components).
"""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

B = 10000
SEED = 20260928
LINKED, INDEPENDENT = 'linked', 'independent'
WER_KEY = 'wer_large_v3'
MARGIN_L = 0.01
MDE_Z = 3.53     # z_{1-0.05/14} + z_0.80 = 2.69 + 0.84 (§5.3 item 4)
GATE_Z = 2.80    # z_0.975 + z_0.80 (§5.3 item 5)
THRESH = {'wer': 2.0, 'sim': 0.010, 'utmos': 0.05, 'mcd': 0.10}
STRATA = {'same': ('F2F', 'M2M'), 'cross': ('F2M', 'M2F')}
COUNTS = ('wer', 'cer', 'wer_capped', 'cer_capped')
SCREEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'journal_extension', 'results')


# ── metrics.py row semantics ─────────────────────────────────

def load_results(paths):
    """metrics.load_results: rows of one or more out_jsonl files, keeping the last row per (pair_id, system)."""
    out = {}
    for path in [paths] if isinstance(paths, str) else paths:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r['pair_id'], r['system'])] = r
    return list(out.values())


def _missing(v):
    return v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))


def _require(rows, keys):
    bad = [(r.get('pair_id'), r.get('system')) for r in rows
           if 'error' in r or any(_missing(r.get(k)) for k in keys)]
    if bad:
        raise ValueError(f'{len(bad)} of {len(rows)} rows have an error or lack a finite {list(keys)}, e.g. {bad[:3]}')


def corpus_wer(rows, key=WER_KEY, primary_only=False, capped=False):
    """metrics.corpus_wer: corpus WER and CER (%) from the per-row edit counts under `key`; raises if a row has an error
    or lacks `key`; primary_only=True keeps the wer_primary rows (and raises if a row lacks wer_primary). capped=True:
    capped corpus WER and CER (M3c), each row's errors min(E, N) before the sums."""
    if primary_only:
        _require(rows, ['wer_primary'])
        rows = [r for r in rows if r['wer_primary']]
    _require(rows, [key])
    e = [r[key] for r in rows]
    cap = min if capped else lambda err, n: err
    n, cn = sum(x['N'] for x in e), sum(x['cN'] for x in e)
    return {'n_utts': len(e),
            'wer': 100.0 * sum(cap(x['S'] + x['D'] + x['I'], x['N']) for x in e) / n if n else float('nan'),
            'cer': 100.0 * sum(cap(x['cS'] + x['cD'] + x['cI'], x['cN']) for x in e) / cn if cn else float('nan')}


def failed(row):
    """A failed conversion: no row, or a row with an error."""
    return row is None or 'error' in row


def _value(row, metric, wer_key=WER_KEY):
    """Per-pair columns of a row without error: (errors, reference units) for the WER / CER metrics (errors
    min(E, N) for wer_capped / cer_capped), (value,) otherwise; None when the value is missing or not finite."""
    if metric in COUNTS:
        x = row.get(wer_key)
        if _missing(x):
            return None
        e, n = ((x['S'] + x['D'] + x['I'], x['N']) if metric.startswith('wer') else
                (x['cS'] + x['cD'] + x['cI'], x['cN']))
        return (min(e, n), n) if metric.endswith('_capped') else (e, n)
    v = row.get(metric)
    return None if _missing(v) else (float(v),)


def margin_rows(rows_tgt, rows_src):
    """M9 per pair, {pair_id: {'m9': SIM_src - SIM_tgt}} from the sim_wavlm of a system's <system>@src and <system>
    rows; a pair where either is missing, failed or not finite gets an error row."""
    out = {}
    for p in set(rows_tgt) | set(rows_src):
        t, s = rows_tgt.get(p), rows_src.get(p)
        vt = None if failed(t) else _value(t, 'sim_wavlm')
        vs = None if failed(s) else _value(s, 'sim_wavlm')
        out[p] = {'m9': vs[0] - vt[0]} if vt and vs else {'error': 'no SIM_tgt or SIM_src'}
    return out


def stratum(genders):
    return next((g for g, v in STRATA.items() if genders in v), None)


# ── crossed bootstrap ────────────────────────────────────────

def _counts(idx, n):
    c = np.zeros((idx.shape[0], n))
    np.add.at(c, (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), idx.ravel()), 1)
    return c


class Design:
    """The speakers of the pairs of one comparison, in pair order, and the bootstrap mode."""

    def __init__(self, src, tgt, mode=LINKED):
        if mode not in (LINKED, INDEPENDENT):
            raise ValueError(f'bootstrap mode {mode!r}')
        self.mode = mode
        if mode == LINKED:
            self.srcs = self.tgts = sorted(set(src) | set(tgt))
        else:
            self.srcs, self.tgts = sorted(set(src)), sorted(set(tgt))
        si, ti = ({s: i for i, s in enumerate(x)} for x in (self.srcs, self.tgts))
        self.si = np.array([si[s] for s in src], int)
        self.ti = np.array([ti[t] for t in tgt], int)

    def draw(self, rng, m):
        """Draw counts of m replicates, (c_src, c_tgt), each m x pool size (the same array when linked)."""
        ns, nt = len(self.srcs), len(self.tgts)
        cs = _counts(rng.integers(0, ns, (m, ns)), ns)
        return (cs, cs) if self.mode == LINKED else (cs, _counts(rng.integers(0, nt, (m, nt)), nt))

    def pair_weights(self, cs, ct):
        """m x n_pairs weights c_src[source] * c_tgt[target] (the definition; bootstrap() uses cell sums)."""
        return cs[:, self.si] * ct[:, self.ti]


def bootstrap(design, cols, stat, den=(), B=B, seed=SEED):
    """Replicates of stat(S), S = the weighted sums of the per-pair columns `cols` (n_pairs x k) with a column of ones
    appended (S[:, -1] = total weight). A replicate is valid when S[:, -1] > 0 and S[:, d] > 0 for d in den; invalid
    ones are redrawn. Returns (full-sample estimate, B replicates, number of redraws); ValueError if the full sample is
    not valid."""
    n = len(design.si)
    cols = np.column_stack([np.asarray(cols, float).reshape(n, -1), np.ones(n)])
    M = np.zeros((len(design.srcs), len(design.tgts), cols.shape[1]))
    np.add.at(M, (design.si, design.ti), cols)
    ns, nt, k = M.shape

    def sums(cs, ct):
        return ((cs @ M.reshape(ns, nt * k)).reshape(-1, nt, k) * ct[:, :, None]).sum(1)

    def valid(S):
        return (S[:, -1] > 0) & np.all(S[:, list(den)] > 0, axis=1)
    full = cols.sum(0)[None]
    if not valid(full)[0]:
        raise ValueError('no pair of positive weight or zero reference words in the full sample')
    rng = np.random.default_rng(seed)
    S = sums(*design.draw(rng, B))
    redraws = 0
    for b in np.flatnonzero(~valid(S)):
        while True:
            redraws += 1
            s = sums(*design.draw(rng, 1))
            if valid(s)[0]:
                S[b] = s[0]
                break
    return float(stat(full)[0]), stat(S), redraws


def p_two_sided(reps):
    """§5.2: p = min(1, 2 min(1 + #{Δ* <= 0}, 1 + #{Δ* >= 0}) / (B + 1))."""
    reps = np.asarray(reps)
    return min(1.0, 2.0 * min(1 + int(np.sum(reps <= 0)), 1 + int(np.sum(reps >= 0))) / (len(reps) + 1))


def summary(est, reps, redraws, design, diff):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    out = {'est': est, 'lo': float(lo), 'hi': float(hi), 'se': float(np.std(reps, ddof=1)), 'redraws': redraws,
           'B': len(reps), 'mode': design.mode, 'n': len(design.si), 'n_src': len(design.srcs),
           'n_tgt': len(design.tgts)}
    if diff:
        out['p'] = p_two_sided(reps)
    return out


def _ratio(S):
    return 100.0 * S[:, 0] / S[:, 1]


def _ratio_diff(S):
    return 100.0 * (S[:, 0] / S[:, 1] - S[:, 2] / S[:, 3])


def _mean(S):
    return S[:, 0] / S[:, -1]


# ── endpoints ────────────────────────────────────────────────

def collect(rows_list, pids, metric, wer_key=WER_KEY):
    """The pairs of pids where every system ({pair_id: row} each) has a row without error and a defined value.
    Returns (pairs, per-pair columns per system (n x 2 for the WER / CER metrics, n x 1 otherwise), failures per
    system, number of undefined pairs)."""
    fails = [sum(failed(rows.get(p)) for p in pids) for rows in rows_list]
    ok, cols, undefined = [], [[] for _ in rows_list], 0
    for p in pids:
        rs = [rows.get(p) for rows in rows_list]
        if any(failed(r) for r in rs):
            continue
        vs = [_value(r, metric, wer_key) for r in rs]
        if any(v is None for v in vs):
            undefined += 1
            continue
        ok.append(p)
        for c, v in zip(cols, vs):
            c.append(v)
    k = 2 if metric in COUNTS else 1
    return ok, [np.array(c, float).reshape(len(ok), k) for c in cols], fails, undefined


def point(rows, pids, metric, wer_key=WER_KEY):
    """One system's point estimate over the pairs of pids without failure and with a defined value (corpus WER / CER
    in %, else the mean), or None. Returns (estimate, pairs used)."""
    ok, (c,), _, _ = collect([rows], pids, metric, wer_key)
    if not ok:
        return None, 0
    if metric in COUNTS:
        return (float(100.0 * c[:, 0].sum() / c[:, 1].sum()) if c[:, 1].sum() else None), len(ok)
    return float(c[:, 0].mean()), len(ok)


def level(rows, pids, spk, metric, mode=LINKED, wer_key=WER_KEY):
    """One system's level with its bootstrap CI and SE over pids (spk: pair_id -> (source, target)). Returns a dict
    (est None when no pair is usable)."""
    ok, (c,), (n_fail,), undefined = collect([rows], pids, metric, wer_key)
    out = {'metric': metric, 'n_pairs': len(pids), 'fail': n_fail, 'undefined': undefined, 'n': len(ok), 'est': None}
    if not ok:
        return out
    design = Design([spk[p][0] for p in ok], [spk[p][1] for p in ok], mode)
    try:
        est, reps, red = (bootstrap(design, c, _ratio, den=(1,)) if metric in COUNTS else bootstrap(design, c, _mean))
    except ValueError as e:
        return dict(out, note=str(e))
    return dict(out, **summary(est, reps, red, design, diff=False))


def _diff(ok, ca, cb, spk, metric, mode):
    design = Design([spk[p][0] for p in ok], [spk[p][1] for p in ok], mode)
    if metric in COUNTS:
        est, reps, red = bootstrap(design, np.column_stack([ca, cb]), _ratio_diff, den=(1, 3))
        a, b = 100.0 * ca[:, 0].sum() / ca[:, 1].sum(), 100.0 * cb[:, 0].sum() / cb[:, 1].sum()
        n_pos = ca[:, 1] > 0
        sec = wilcoxon_rb(100.0 * (ca[n_pos, 0] / ca[n_pos, 1] - cb[n_pos, 0] / cb[n_pos, 1]))
    else:
        est, reps, red = bootstrap(design, ca[:, 0] - cb[:, 0], _mean)
        a, b = float(ca[:, 0].mean()), float(cb[:, 0].mean())
        sec = wilcoxon_rb(ca[:, 0] - cb[:, 0])
    return dict(summary(est, reps, red, design, diff=True), a=a, b=b, wilcoxon=sec)


def compare(rows_a, rows_b, pids, spk, metric, mode=LINKED, worst=None, n_ref=None, wer_key=WER_KEY):
    """Paired comparison A - B of one endpoint over the pairs pids (§5.2). rows_a / rows_b: {pair_id: row} of the two
    systems on the set (all their rows: the worst case of a continuous metric is taken over them); spk: pair_id ->
    (source, target); metric: 'wer', 'cer', 'wer_capped', 'cer_capped' or a row key; worst: 'min' or 'max' for a
    continuous metric's worst case (None: no sensitivity analysis); n_ref: pair_id -> reference units, for a WER / CER
    worst case on a pair where both systems failed. Returns a dict: est (difference of corpus WERs or mean
    difference), lo, hi, se, p, a and b (each system on the pairs used), n, failures per system, undefined pairs,
    redraws, wilcoxon; worst_case when the failure counts differ."""
    ok, (ca, cb), (fa, fb), undefined = collect([rows_a, rows_b], pids, metric, wer_key)
    out = {'metric': metric, 'n_pairs': len(pids), 'fail_a': fa, 'fail_b': fb, 'undefined': undefined, 'n': len(ok),
           'est': None}
    try:
        if ok:
            out.update(_diff(ok, ca, cb, spk, metric, mode))
        if fa != fb:
            out['worst_case'] = _worst_case(rows_a, rows_b, pids, spk, metric, mode, worst, n_ref or {}, wer_key)
    except ValueError as e:
        out['note'] = str(e)
    return out


def _worst_case(rows_a, rows_b, pids, spk, metric, mode, worst, n_ref, wer_key):
    """§5.2 last bullet: every failed conversion of either system imputed with its worst case."""
    if metric not in COUNTS and worst not in ('min', 'max'):
        return None

    def worst_of(rows):
        v = [x[0] for x in (_value(r, metric, wer_key) for r in rows.values() if not failed(r)) if x is not None]
        return (min(v) if worst == 'min' else max(v)) if v else None
    wv = None if metric in COUNTS else (worst_of(rows_a), worst_of(rows_b))
    ok, ca, cb, imputed, lost = [], [], [], [0, 0], 0
    for p in pids:
        rs = (rows_a.get(p), rows_b.get(p))
        vs = [None if failed(r) else _value(r, metric, wer_key) for r in rs]
        if any(not failed(r) and v is None for r, v in zip(rs, vs)):
            continue        # undefined, not a failed conversion: left out as in the primary analysis
        if metric in COUNTS:
            n = next((v[1] for v in vs if v is not None), n_ref.get(p))
            fills = [None if n is None else (n, n)] * 2
        else:
            fills = [None if x is None else (x,) for x in wv]
        fail = [failed(r) for r in rs]
        if any(f and x is None for f, x in zip(fail, fills)):
            lost += 1
            continue
        for i in range(2):
            if fail[i]:
                vs[i] = fills[i]
                imputed[i] += 1
        ok.append(p)
        ca.append(vs[0])
        cb.append(vs[1])
    out = {'imputed_a': imputed[0], 'imputed_b': imputed[1], 'unimputable': lost, 'n': len(ok)}
    if metric not in COUNTS:
        out.update(worst_a=wv[0], worst_b=wv[1])
    if ok:
        out.update(_diff(ok, np.array(ca, float), np.array(cb, float), spk, metric, mode))
    return out


def wilcoxon_rb(d):
    """Two-sided Wilcoxon signed-rank test on the per-pair differences d (zeros and non-finite values dropped) and the
    matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-)."""
    from scipy.stats import rankdata, wilcoxon
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if not len(d):
        return {'n': 0, 'p': None, 'r_rb': None}
    r = rankdata(np.abs(d))
    rp, rm = float(r[d > 0].sum()), float(r[d < 0].sum())
    return {'n': len(d), 'p': float(wilcoxon(d).pvalue), 'r_rb': (rp - rm) / (rp + rm), 'w_plus': rp, 'w_minus': rm}


def holm(pvals, alpha=0.05):
    """§5.2: sort the m p-values and reject the i-th smallest while p_(i) <= alpha / (m - i + 1). Reject flags in input
    order; a None p counts in m and is never rejected."""
    p = [1.0 if x is None else x for x in pvals]
    rej = [False] * len(p)
    for i, k in enumerate(sorted(range(len(p)), key=lambda j: p[j])):
        if pvals[k] is None or p[k] > alpha / (len(p) - i):
            break
        rej[k] = True
    return rej


def gate_l(diff, margin=MARGIN_L):
    """Gate L in one stratum (§5.1, §5.2): H0 Δ_src,g >= margin is rejected when the upper limit of the two-sided 95 %
    percentile CI of Δ_src,g = SIM_src(A) - SIM_src(B) (a compare() result) is below the margin; None if no CI."""
    return None if diff.get('hi') is None else bool(diff['hi'] < margin)


def gate_l_strata(src_a, src_b, pairs, mode=LINKED, src_e=None):
    """Gate L of one set in both strata: compare() of the sim_wavlm of the <A>@src and <B>@src rows ({pair_id: row})
    over each stratum's pairs (make_pairs.py rows) whose source speaker has an E set (src_e; None = every source),
    with the worst case 'max' (highest SIM_src). Returns {stratum: compare() result + rejects, n_no_source_E}."""
    spk = {p['pair_id']: (p['source_speaker'], p['target_speaker']) for p in pairs}
    out = {}
    for g in STRATA:
        sub, no_e = endpoint_pairs(pairs, 'gate_' + g, src_e)
        d = compare(src_a, src_b, [p['pair_id'] for p in sub], spk, 'sim_wavlm', mode=mode, worst='max')
        out[g] = dict(d, rejects=gate_l(d), n_no_source_E=no_e)
    return out


# ── §5.3 variance components and size check ──────────────────

def design_counts(src, tgt):
    src, tgt = list(src), list(tgt)
    return {'n': len(src), 'n_s': len(set(src)), 'n_t': len(set(tgt)), 'n_c': len(set(zip(src, tgt)))}


def wer_linearised(e_a, e_b, n_words, capped=False):
    """§5.3 item 2: d_p = 100 (E_A,p - E_B,p) / N̄, N̄ = the mean reference words of the pairs; capped=True uses the
    capped error counts min(E_p, N_p) (M3c), so mean(d_p) is the capped corpus-WER difference."""
    e_a, e_b, n = (np.asarray(x, float) for x in (e_a, e_b, n_words))
    if capped:
        e_a, e_b = np.minimum(e_a, n), np.minimum(e_b, n)
    return 100.0 * (e_a - e_b) / np.mean(n)


VC_METHODS = (('bfgs', 'lbfgs', 'cg'), ('powell',), ('nm',))


def variance_components(d, src, tgt):
    """§5.3 item 2: REML estimates of d_p = mu + a_source + b_target + c_source×target + e_p, statsmodels MixedLM in its
    variance-component form with one group. The REML estimate is the maximiser of the restricted likelihood, and near
    the boundary (a component at 0) MixedLM's default gradient chain (bfgs, lbfgs, cg) can stop below it, so the model
    is fitted with each optimiser of VC_METHODS and the fit with the highest REML log-likelihood is kept. An optimiser
    that raises (e.g. LinAlgError: MixedLM inverts the Hessian, singular at some boundary fits) or ends at a non-finite
    log-likelihood is left out and listed in fit_errors; ValueError if none is left. Returns mu, sigma2_a, sigma2_b,
    sigma2_c, sigma2_e, the design counts, the method kept, every kept method's REML log-likelihood, fit_errors,
    converged and the kept fit's warnings (numpy floating-point warnings of trial points excluded)."""
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.DataFrame({'d': np.asarray(d, float), 'src': [str(s) for s in src], 'tgt': [str(t) for t in tgt]})
    df['cell'] = df['src'] + '|' + df['tgt']
    df['g'] = 0
    model = smf.mixedlm('d ~ 1', df, groups='g', re_formula='0',
                        vc_formula={'a': '0 + C(src)', 'b': '0 + C(tgt)', 'c': '0 + C(cell)'})
    fits, errors = [], {}
    for method in VC_METHODS:
        name = '/'.join(method)
        with warnings.catch_warnings(record=True) as w, np.errstate(all='ignore'):    # results are computed lazily
            warnings.simplefilter('always')
            try:
                fit = model.fit(reml=True, method=list(method))
                vc = dict(zip(model.exog_vc.names, (float(x) for x in fit.vcomp)))
                out = {'mu': float(np.asarray(fit.fe_params)[0]), 'sigma2_a': vc['a'], 'sigma2_b': vc['b'],
                       'sigma2_c': vc['c'], 'sigma2_e': float(fit.scale), 'llf': float(fit.llf),
                       'converged': bool(fit.converged)}
            except (np.linalg.LinAlgError, ValueError) as e:
                errors[name] = f'{type(e).__name__}: {e}'
                continue
        if np.isfinite(out['llf']):
            fits.append((name, out, sorted({str(x.message) for x in w if not issubclass(x.category, RuntimeWarning)})))
        else:
            errors[name] = f'REML log-likelihood {out["llf"]} (e.g. every d_p equal)'
    if not fits:
        raise ValueError(f'no REML fit: {errors}')
    method, best, msgs = max(fits, key=lambda x: x[1]['llf'])
    return {**{k: v for k, v in best.items() if k != 'llf'}, **design_counts(src, tgt), 'method': method,
            'reml_llf': {m: x['llf'] for m, x, _ in fits}, 'fit_errors': errors, 'warnings': msgs}


def predict_se(vc, counts):
    """§5.3 item 3: SE = sqrt(σa²/n_s + σb²/n_t + σc²/n_c + σe²/n)."""
    return math.sqrt(vc['sigma2_a'] / counts['n_s'] + vc['sigma2_b'] / counts['n_t'] +
                     vc['sigma2_c'] / counts['n_c'] + vc['sigma2_e'] / counts['n'])


def endpoint_pairs(pairs, kind, src_e=None):
    """The make_pairs.py rows of a test list that enter an endpoint: wer the wer_primary pairs, sim / utmos all pairs,
    mcd the pairs with a parallel recording, gate_same / gate_cross the stratum's pairs whose source speaker has a
    source-E set (src_e: those speakers; None = every source). Returns (pairs, pairs left out for want of an E set)."""
    if kind == 'wer':
        return [p for p in pairs if p['wer_primary']], 0
    if kind == 'mcd':
        return [p for p in pairs if p.get('P')], 0
    if kind.startswith('gate_'):
        g = [p for p in pairs if stratum(p['genders']) == kind[5:]]
        keep = [p for p in g if src_e is None or p['source_speaker'] in src_e]
        return keep, len(g) - len(keep)
    return list(pairs), 0


def size_check(conds, designs, src_e=None):
    """§5.3 items 3-7. conds: [{'vctk2k': endpoint name or None, 'ltc2k': name or None, 'kind': wer | sim | utmos | mcd
    | gate_same | gate_cross, 'vc': variance_components() or None, 'delta_dev': the Dev Δ_src,g (gate kinds)}];
    designs: {'vctk2k_5': pair rows, 'vctk2k_10': pair rows or None, 'ltc2k': pair rows or None}; src_e: {design:
    VCTK-2k source speakers with a source-E set, or None}. Returns {'rows': one per condition and design, 'rebuild':
    the VCTK-2k decision}."""
    src_e = src_e or {}
    rows = []
    for c in conds:
        for dn, pairs in designs.items():
            name = c['ltc2k' if dn == 'ltc2k' else 'vctk2k']
            if name is None or pairs is None:
                continue
            sub, no_e = endpoint_pairs(pairs, c['kind'], src_e.get(dn) if dn != 'ltc2k' else None)
            r = {'endpoint': name, 'design': dn, 'kind': c['kind'],
                 **design_counts([p['source_speaker'] for p in sub], [p['target_speaker'] for p in sub]),
                 'n_left_out_no_source_E': no_e, 'se': None, 'pass': None}
            if c['kind'].startswith('gate_') and dn != 'ltc2k' and src_e.get(dn) is None:
                r['note'] = 'no source-E file: every source speaker counted as having an E set'
            if c.get('vc') and sub:
                r['se'] = predict_se(c['vc'], r)
                if c['kind'].startswith('gate_'):
                    r.update(lhs=GATE_Z * r['se'], rhs=MARGIN_L - c['delta_dev'], delta_dev=c['delta_dev'])
                    r['pass'] = bool(c['delta_dev'] < MARGIN_L and r['lhs'] <= r['rhs'])
                    if c['delta_dev'] >= MARGIN_L:
                        r['note'] = 'no power under any design: Δ_dev,g >= 0.01'
                else:
                    r.update(mde=MDE_Z * r['se'], threshold=THRESH[c['kind']])
                    r['pass'] = bool(r['mde'] <= r['threshold'])
            rows.append(r)
    by = {(r['endpoint'], r['design']): r for r in rows}
    names = [c['vctk2k'] for c in conds if c['vctk2k']]
    reb = {'conditions': names, 'use': None, 'triggers': [],
           'evaluable': designs.get('vctk2k_5') is not None and designs.get('vctk2k_10') is not None}
    if reb['evaluable']:
        reb['triggers'] = [n for n in names if by[(n, 'vctk2k_5')]['pass'] is False
                           and by[(n, 'vctk2k_10')]['pass'] is True]
        reb['use'] = '10' if reb['triggers'] else '5'
    chosen = 'vctk2k_10' if reb['use'] == '10' else 'vctk2k_5'
    reb['underpowered'] = [n for n in names if (n, chosen) in by and by[(n, chosen)]['pass'] is not True]
    reb['underpowered_design'] = chosen
    return {'rows': rows, 'rebuild': reb}


# ── self-test ────────────────────────────────────────────────

def _import_metrics():
    """benchmark/metrics.py for the cross-checks; a stub librosa (unused by corpus_wer / load_results) where the real
    one is not installed."""
    import types
    try:
        import librosa  # noqa: F401
    except ImportError:
        sys.modules['librosa'] = types.ModuleType('librosa')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import metrics
    return metrics


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f'  ok  {msg}', flush=True)


def _unit_tests():
    rng = np.random.default_rng(1)
    print('unit tests', flush=True)
    _check(p_two_sided(np.r_[np.full(9990, -1.0), np.full(10, 1.0)]) == 2 * 11 / 10001,
           'p = 2 min(1 + #{<=0}, 1 + #{>=0}) / (B + 1) on 9,990 negative and 10 positive replicates')
    _check(p_two_sided(np.r_[-1.0, 0.0, 1.0]) == 1.0 and p_two_sided(np.full(10000, 3.0)) == 2 / 10001,
           'p is capped at 1 and is 2 / (B + 1) when every replicate is on one side')
    _check(holm([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True] and holm([0.01, 0.02, 0.03]) == [True] * 3
           and holm([0.001, None, 0.02]) == [True, False, True] and holm([0.03, None]) == [False, False],
           'Holm step-down (0.005 <= 0.0125, 0.01 <= 0.0167, 0.03 > 0.025 stops); None counts in m')

    src = [f's{rng.integers(0, 6)}' for _ in range(60)]
    tgt = [f's{(int(s[1]) + 1 + rng.integers(0, 5)) % 6}' for s in src]
    cols = rng.normal(size=(60, 3))
    for mode in (LINKED, INDEPENDENT):
        d = Design(src, tgt, mode)
        cs, ct = d.draw(np.random.default_rng(5), 7)
        M = np.zeros((len(d.srcs), len(d.tgts), 3))
        np.add.at(M, (d.si, d.ti), cols)
        cell = np.einsum('bs,stk,bt->bk', cs, M, ct)
        _check(np.allclose(cell, d.pair_weights(cs, ct) @ cols) and (cs.sum(1) == len(d.srcs)).all(),
               f'{mode}: cell sums c_src\' M c_tgt equal the pair-weight sums; each replicate draws pool-size speakers')
    d = Design(['a', 'a', 'b'], ['b', 'c', 'a'], LINKED)
    w = d.pair_weights(np.array([[2.0, 0, 1]]), np.array([[2.0, 0, 1]]))
    _check(d.srcs == ['a', 'b', 'c'] and w.tolist() == [[0.0, 2.0, 0.0]],
           'linked: counts a=2, b=0, c=1 give weights 0, 2, 0 to a->b, a->c, b->a (b undrawn in both roles)')
    d = Design(['a', 'a', 'b'], ['a', 'c', 'a'], INDEPENDENT)
    w = d.pair_weights(np.array([[0.0, 2]]), np.array([[1.0, 1]]))
    _check(d.srcs == ['a', 'b'] and d.tgts == ['a', 'c'] and w.tolist() == [[0.0, 0.0, 2.0]],
           'independent: source a undrawn, b twice; targets a, c once: weights 0, 0, 2 (roles are separate identities)')

    x = rng.normal(size=60)
    est, reps, red = bootstrap(Design(src, tgt), x, _mean)
    _, reps2, _ = bootstrap(Design(src, tgt), x, _mean)
    _check(abs(est - x.mean()) < 1e-12 and np.array_equal(reps, reps2) and len(reps) == B,
           'mean: full-sample estimate is the plain mean; replicates are reproducible from the seed')
    sub = Design(['a', 'c'], ['b', 'd'], INDEPENDENT)       # a -> b has no reference word: P(invalid) = 7/16
    _, r1, red1 = bootstrap(sub, np.column_stack([[3.0, 1.0], [0.0, 5.0]]), _ratio, den=(1,))
    _check(len(r1) == B and np.isfinite(r1).all() and 0.7 * B < red1 < 0.85 * B,
           f'zero reference words: invalid replicates redrawn ({red1} redraws, expected 7/9 B), B finite replicates')
    try:
        bootstrap(Design(['a'], ['b']), np.column_stack([[2.0], [0.0]]), _ratio, den=(1,))
        _check(False, 'zero reference words in the full sample raises')
    except ValueError:
        _check(True, 'zero reference words in the full sample raises ValueError')

    def wrow(pid, s, n, extra=None):
        return dict({'pair_id': pid, WER_KEY: {'S': s, 'D': 0, 'I': 0, 'N': n, 'cS': s, 'cD': 0, 'cI': 0, 'cN': 5 * n},
                     'sim_wavlm': 0.5 + 0.01 * s, 'wer_primary': True}, **(extra or {}))
    spk = {f'p{i}': (f's{i % 4}', f't{i % 3}') for i in range(12)}
    ra = {f'p{i}': wrow(f'p{i}', i % 3, 10) for i in range(12)}
    rb = {f'p{i}': wrow(f'p{i}', 1, 10) for i in range(12)}
    ra['p0'] = {'pair_id': 'p0', 'error': 'RuntimeError: x'}
    del ra['p1']
    rb['p2'] = {'pair_id': 'p2', 'error': 'RuntimeError: x'}
    rb['p3']['sim_wavlm'] = float('nan')
    c = compare(ra, rb, sorted(spk), spk, 'wer', mode=INDEPENDENT)
    ok = [p for p in sorted(spk) if p not in ('p0', 'p1', 'p2')]
    want = 100 * sum(ra[p][WER_KEY]['S'] for p in ok) / (10 * len(ok)) - 10.0
    _check(c['fail_a'] == 2 and c['fail_b'] == 1 and c['n'] == 9 and abs(c['est'] - want) < 1e-12,
           'compare WER: failures counted per system (2, 1), pairs with a failure left out, est = corpus-WER '
           'difference')
    wc = c['worst_case']
    e_a = sum(ra[p][WER_KEY]['S'] for p in ok + ['p2']) + 10 + 10
    _check(wc['imputed_a'] == 2 and wc['imputed_b'] == 1 and wc['n'] == 12
           and abs(wc['est'] - (100 * e_a / 120 - 100 * (11 + 10) / 120)) < 1e-12,
           'worst case WER: every reference word of a failed pair deleted (E = N), N from the other system')
    keys = ('est', 'lo', 'hi', 'se', 'p', 'redraws')
    cc = {m: compare(ra, rb, sorted(spk), spk, m, mode=INDEPENDENT) for m in ('wer_capped', 'cer', 'cer_capped')}
    _check(all(cc['wer_capped'][k] == c[k] for k in keys) and cc['wer_capped']['worst_case']['est'] == wc['est']
           and all(cc['cer_capped'][k] == cc['cer'][k] for k in keys)
           and corpus_wer([ra[p] for p in ok], capped=True) == corpus_wer([ra[p] for p in ok]),
           'no pair above 100 %: capped WER / CER equal WER / CER (estimate, CI, SE, p, worst case, corpus_wer)')
    rc = dict(ra, p4=wrow('p4', 60, 10))        # 60 errors on 10 words and on 50 characters
    cap_w = 100 * sum(min(rc[p][WER_KEY]['S'], 10) for p in ok) / (10 * len(ok))
    cap_c = 100 * sum(min(rc[p][WER_KEY]['cS'], 50) for p in ok) / (50 * len(ok))
    c2 = compare(rc, rb, sorted(spk), spk, 'wer_capped', mode=INDEPENDENT)
    e_c = sum(min(rc[p][WER_KEY]['S'], 10) for p in ok + ['p2']) + 10 + 10
    cw2 = corpus_wer([rc[p] for p in ok], capped=True)
    _check(abs(point(rc, ok, 'wer_capped')[0] - cap_w) < 1e-12 and cap_w < point(rc, ok, 'wer')[0]
           and abs(c2['est'] - (cap_w - 10.0)) < 1e-12
           and abs(c2['worst_case']['est'] - (100 * e_c / 120 - 100 * 21 / 120)) < 1e-12
           and abs(cw2['wer'] - cap_w) < 1e-12 and abs(cw2['cer'] - cap_c) < 1e-12,
           'capped: a pair with 60 errors on 10 words (50 characters) counts 10 (50) in point, compare, its worst case '
           '(E = N, within the cap) and corpus_wer(capped=True)')
    c = compare(ra, rb, sorted(spk), spk, 'sim_wavlm', worst='min')
    wc = c['worst_case']
    _check(c['undefined'] == 1 and c['n'] == 8 and wc['worst_a'] == 0.5 and wc['worst_b'] == 0.51 and wc['n'] == 11,
           'continuous: NaN without error is undefined (left out, not imputed); worst case = the failing system\'s '
           'lowest SIM')
    _check(compare(ra, ra, sorted(spk), spk, 'sim_wavlm')['est'] == 0.0 and 'worst_case' not in
           compare(ra, ra, sorted(spk), spk, 'sim_wavlm'), 'equal failure counts: no worst-case analysis')
    m9 = margin_rows({'x': {'sim_wavlm': 0.9}, 'y': {'sim_wavlm': 0.8}}, {'x': {'sim_wavlm': 0.7}})
    _check(abs(m9['x']['m9'] + 0.2) < 1e-12 and 'error' in m9['y'],
           'M9 rows: D = SIM_src - SIM_tgt; no @src row -> error')

    from scipy.stats import wilcoxon
    dd = np.array([1.5, -0.5, 2.0, 0.0, 3.0, -1.0, 2.5])
    wr = wilcoxon_rb(dd)
    _check(abs(wr['p'] - wilcoxon(dd[dd != 0]).pvalue) < 1e-12 and wr['n'] == 6 and wr['w_plus'] == 18
           and wr['w_minus'] == 3 and abs(wr['r_rb'] - 15 / 21) < 1e-12 and wilcoxon_rb([1, 2, 3])['r_rb'] == 1.0,
           'Wilcoxon signed-rank (zeros dropped) and rank-biserial (R+ - R-) / (R+ + R-) = 15/21')
    _check(gate_l({'hi': 0.0099}) is True and gate_l({'hi': 0.01}) is False and gate_l({}) is None,
           'gate L rejects iff the CI upper limit is below +0.01')

    rs = np.random.default_rng(3)
    ns, nt = 30, 30
    sa, sb, sc, se = 4.0, 1.0, 0.25, 0.0625
    a, b = rs.normal(0, 2, ns), rs.normal(0, 1, nt)
    s_ = rs.integers(0, ns, 1200)
    t_ = rs.integers(0, nt, 1200)
    cells = {}
    cc = np.array([cells.setdefault((i, j), rs.normal(0, 0.5)) for i, j in zip(s_, t_)])
    y = 1 + a[s_] + b[t_] + cc + rs.normal(0, 0.25, len(s_))
    vc = variance_components(y, [f's{i}' for i in s_], [f't{j}' for j in t_])
    est = [vc['sigma2_a'], vc['sigma2_b'], vc['sigma2_c'], vc['sigma2_e']]
    _check(vc['converged'] and all(0.5 * v < e < 2 * v for e, v in zip(est, (sa, sb, sc, se)))
           and vc['reml_llf'][vc['method']] == max(vc['reml_llf'].values()),
           f'REML variance components of a simulated crossed design: {np.round(est, 3).tolist()} for '
           f'{[sa, sb, sc, se]} (kept {vc["method"]}, the highest REML log-likelihood of {vc["reml_llf"]})')
    _check(vc['n_s'] == ns and vc['n_c'] == len(cells), 'design counts n_s, n_t, n_c, n from the pairs')
    rz = np.random.default_rng(0)
    s20, t20 = [f's{i % 20}' for i in range(200)], [f's{(7 * i + 3) % 20}' for i in range(200)]
    vz = variance_components(np.r_[np.zeros(190), rz.normal(0, 30, 10)], s20, t20)
    _check(set(vz['reml_llf']) | set(vz['fit_errors']) == {'/'.join(m) for m in VC_METHODS}
           and vz['reml_llf'][vz['method']] == max(vz['reml_llf'].values()),
           f"boundary fit: an optimiser that raises is left out, the others kept (kept {vz['method']}; failed "
           f"{vz['fit_errors'] or 'none'})")
    try:
        variance_components(np.zeros(200), s20, t20)
        _check(False, 'every d_p equal raises')
    except ValueError as e:
        _check('no REML fit' in str(e), 'every d_p equal: no finite REML log-likelihood -> ValueError per optimiser')
    ea, eb, nw = rs.integers(0, 9, 50), rs.integers(0, 9, 50), rs.integers(4, 20, 50)
    _check(abs(wer_linearised(ea, eb, nw).mean() - 100 * (ea.sum() - eb.sum()) / nw.sum()) < 1e-9,
           'linearised WER: mean of 100 (E_A - E_B) / N̄ equals the corpus-WER difference')
    _check(abs(wer_linearised(ea, eb, nw, capped=True).mean()
               - 100 * (np.minimum(ea, nw).sum() - np.minimum(eb, nw).sum()) / nw.sum()) < 1e-9
           and (np.r_[ea, eb] > np.r_[nw, nw]).any(),
           'capped linearised WER: the mean equals the capped corpus-WER difference (some E_p > N_p)')
    v1 = {'sigma2_a': 4.0, 'sigma2_b': 1.0, 'sigma2_c': 2.0, 'sigma2_e': 50.0}
    cnt = {'n_s': 20, 'n_t': 20, 'n_c': 100, 'n': 1000}
    _check(abs(predict_se(v1, cnt) - math.sqrt(0.2 + 0.05 + 0.02 + 0.05)) < 1e-12,
           'SE² = σa²/n_s + σb²/n_t + σc²/n_c + σe²/n')

    def plist(n_tg):        # 8 sources x 4 utterances (2 wer_primary, 2 with P) x n_tg of 10 targets
        return [{'source_speaker': f's{i}', 'target_speaker': f't{(i + u + j) % 10}', 'wer_primary': u % 2 == 0,
                 'P': 'x' if u % 2 else None, 'genders': ('F2F', 'M2F', 'F2M', 'M2M')[(i + (i + u + j) % 10) % 4]}
                for i in range(8) for u in range(4) for j in range(n_tg)]
    vc_w = {'sigma2_a': 0.0, 'sigma2_b': 0.0, 'sigma2_c': 0.0, 'sigma2_e': 40.0}
    vc_g = {'sigma2_a': 0.0, 'sigma2_b': 0.0, 'sigma2_c': 0.0, 'sigma2_e': 0.0005}
    conds = [{'vctk2k': 'P1', 'ltc2k': 'P2', 'kind': 'wer', 'vc': vc_w},
             {'vctk2k': 'L1s', 'ltc2k': 'L2s', 'kind': 'gate_same', 'vc': vc_g, 'delta_dev': -0.01},
             {'vctk2k': 'L1c', 'ltc2k': 'L2c', 'kind': 'gate_cross', 'vc': vc_g, 'delta_dev': 0.02}]
    e2 = {'vctk2k_5': {'s0', 's1'}, 'vctk2k_10': {'s0', 's1'}}
    sc = size_check(conds, {'vctk2k_5': plist(5), 'vctk2k_10': plist(10), 'ltc2k': plist(10)}, e2)
    by = {(r['endpoint'], r['design']): r for r in sc['rows']}
    p5, p10 = by[('P1', 'vctk2k_5')], by[('P1', 'vctk2k_10')]
    _check(p5['n'] == 80 and abs(p5['mde'] - 3.53 * math.sqrt(40 / 80)) < 1e-12 and p5['pass'] is False
           and p10['n'] == 160 and p10['pass'] is True and sc['rebuild']['use'] == '10'
           and sc['rebuild']['triggers'] == ['P1'] and ('P2', 'ltc2k') in by,
           f"size check: MDE = 3.53 SE vs 2.0 points, P1 {p5['mde']:.2f} with 5 targets, {p10['mde']:.2f} with 10 "
           f"-> 10 targets; LTC-2k endpoints named P2, L2s, L2c")
    l5, l10 = by[('L1s', 'vctk2k_5')], by[('L1s', 'vctk2k_10')]
    _check(l5['n_left_out_no_source_E'] > 0 and l5['n_s'] == 2 and abs(l5['lhs'] - 2.8 * l5['se']) < 1e-12
           and abs(l5['rhs'] - 0.02) < 1e-12 and l5['pass'] is True and l10['pass'] is True
           and by[('L1c', 'vctk2k_10')]['pass'] is False and 'no power' in by[('L1c', 'vctk2k_10')]['note']
           and sc['rebuild']['underpowered'] == ['L1c'],
           'gate L power 2.80 SE_g <= 0.01 - Δ_dev,g on the pairs whose source has an E set; Δ_dev,g >= 0.01: no power')
    sc = size_check(conds[1:], {'vctk2k_5': plist(5), 'vctk2k_10': plist(10), 'ltc2k': None}, e2)
    _check(sc['rebuild']['use'] == '5' and not sc['rebuild']['triggers'] and sc['rebuild']['underpowered'] == ['L1c'],
           'size check: no condition fails with 5 and passes with 10 -> 5 targets, underpowered endpoints listed')
    sc = size_check(conds, {'vctk2k_5': plist(5), 'vctk2k_10': None, 'ltc2k': None})
    _check(sc['rebuild']['use'] is None and not sc['rebuild']['evaluable'] and sc['rebuild']['underpowered'] ==
           ['P1', 'L1c'] and 'no source-E file' in [r for r in sc['rows'] if r['endpoint'] == 'L1s'][0]['note'],
           'size check without the 10-target list: no decision; without a source-E file every source counts (noted)')


def _screen_tests(screen_dir):
    f = os.path.join(screen_dir, 'anchor_lab', 'metrics.jsonl')
    fs = os.path.join(screen_dir, 'anchor_lab', 'metrics_src.jsonl')
    if not (os.path.exists(f) and os.path.exists(fs)):
        print(f'screen tests skipped: no {f}', flush=True)
        return
    print(f'screen cross-checks ({screen_dir})', flush=True)
    metrics = _import_metrics()
    rows, rsrc = load_results(f), load_results(fs)
    _check(sorted(map(json.dumps, rows)) == sorted(map(json.dumps, metrics.load_results(f))),
           'load_results equals metrics.load_results on anchor_lab/metrics.jsonl')
    by = {}
    for r in rows + rsrc:
        by.setdefault(r['system'], {})[r['pair_id']] = r
    n_eq = n_lower = 0
    for s in sorted(x for x in by if not x.endswith('@src')):
        rs = list(by[s].values())
        for po in (False, True):
            ours, theirs = corpus_wer(rs, primary_only=po), metrics.corpus_wer(rs, primary_only=po)
            if ours != theirs:
                raise AssertionError(f'corpus_wer differs from metrics.corpus_wer: {s} primary_only={po}')
            sub = [r for r in rs if r['wer_primary'] or not po]
            cap = corpus_wer(rs, primary_only=po, capped=True)
            for m in ('wer', 'cer'):
                over = any(_value(r, m)[0] > _value(r, m)[1] for r in sub)
                if (not (cap[m] < ours[m] if over else cap[m] == ours[m])
                        or abs(point(by[s], [r['pair_id'] for r in sub], m + '_capped')[0] - cap[m]) > 1e-9):
                    raise AssertionError(f'capped {m} wrong: {s} primary_only={po}')
                n_lower, n_eq = n_lower + over, n_eq + (not over)
    _check(True, f'corpus_wer equals metrics.corpus_wer (WER, CER, n) on all {len(by) // 2} systems, all and '
                 f'wer_primary rows')
    _check(True, f'capped WER / CER (corpus_wer(capped=True) = point wer_capped / cer_capped) equal the uncapped ones '
                 f'in the {n_eq} cases (system, rows, WER or CER) without a pair above 100 %, lower in the other '
                 f'{n_lower}')
    fa, c1 = by['anc_faiss'], by['anc_R3_s0_pm_rq_T1']
    ids = sorted(fa)
    prim = [p for p in ids if fa[p]['wer_primary']]
    spk = {p: (fa[p]['source_speaker'], fa[p]['target_speaker']) for p in ids}
    w = {s: tuple(round(point(by[s], p, m)[0], 2) for m in ('wer', 'wer_capped') for p in (ids, prim))
         for s in ('anc_faiss', 'anc_R1_r0_tau0.4', 'anc_src0_faiss', 'anc_R3_s0_pm_rq_T1')}
    _check(w == {'anc_faiss': (126.02, 204.55, 95.45, 99.24), 'anc_R1_r0_tau0.4': (107.97, 115.15, 94.8, 98.48),
                 'anc_src0_faiss': (148.46, 111.36, 96.42, 98.48), 'anc_R3_s0_pm_rq_T1': (18.21, 26.52, 18.21, 26.52)},
           f'plan §2.6 corpus WER (50 / 20 wer_primary), uncapped and capped: FAISS 126.02 / 204.55 and 95.45 / '
           f'99.24, C1 18.21 / 26.52 in both ({w})')
    loop = 'dev__p226_064__p230'
    e_loop, e_prim = _value(fa[loop], 'wer'), sum(_value(fa[p], 'wer')[0] for p in prim)
    lv = {m: level(fa, prim, spk, m) for m in ('wer', 'wer_capped')}
    _check(point(fa, [loop], 'wer_capped')[0] == 100.0 < point(fa, [loop], 'wer')[0]
           and (e_loop[0], e_prim) == (112, 270)
           and lv['wer_capped']['est'] < lv['wer']['est'] and lv['wer_capped']['hi'] < lv['wer']['hi'],
           f"Whisper-loop pair {loop}: FAISS {e_loop[0]} errors on {e_loop[1]} words ({e_loop[0]} of its {e_prim} on "
           f"the 20 wer_primary pairs, plan §0.3 row 20): capped 100.00 % < uncapped "
           f"{point(fa, [loop], 'wer')[0]:.2f} %; FAISS on the 20: capped {lv['wer_capped']['est']:.2f} "
           f"[{lv['wer_capped']['lo']:.2f}, {lv['wer_capped']['hi']:.2f}], uncapped {lv['wer']['est']:.2f} "
           f"[{lv['wer']['lo']:.2f}, {lv['wer']['hi']:.2f}]")
    lc = {m: level(c1, prim, spk, m) for m in ('wer', 'wer_capped')}
    _check(all(lc['wer'][k] == lc['wer_capped'][k] for k in ('est', 'lo', 'hi', 'se', 'redraws')),
           f"C1 has no pair above 100 %: capped = uncapped WER on the 20 wer_primary pairs, bootstrap replicates "
           f"included ({lc['wer_capped']['est']:.2f} [{lc['wer_capped']['lo']:.2f}, {lc['wer_capped']['hi']:.2f}])")
    sims = {s: round(point(by[s], ids, 'sim_wavlm')[0], 3) for s in ('anc_faiss', 'anc_R3_s0_pm_rq_T1')}
    src_ff, src_mf = (round(point(by['anc_R3_s0_pm_rq_T1@src'], [p for p in ids if fa[p]['genders'] == g],
                                  'sim_wavlm')[0], 3) for g in ('F2F', 'M2F'))
    _check(sims == {'anc_faiss': 0.919, 'anc_R3_s0_pm_rq_T1': 0.819} and (src_ff, src_mf) == (0.768, 0.670),
           f'plan §2.6 SIM tgt FAISS 0.919, C1 0.819; C1 SIM src F→F 0.768, M→F 0.670 ({sims}, {src_ff}, {src_mf})')
    m9 = margin_rows(c1, by['anc_R3_s0_pm_rq_T1@src'])
    _check(round(point(m9, ids, 'm9')[0], 3) == -0.098 and sum(m9[p]['m9'] > 0 for p in ids) == 10,
           'plan §2.6 mean D of C1 -0.098, 10 of 50 pairs with D > 0')
    cw = compare(c1, fa, prim, spk, 'wer')
    cs = compare(c1, fa, ids, spk, 'sim_wavlm', worst='min')
    _check(abs(cw['est'] - (cw['a'] - cw['b'])) < 1e-9 and round(cw['a'], 2) == 26.52 and cw['hi'] < 0,
           f"C1 - FAISS WER (20 wer_primary, linked): {cw['est']:+.2f} [{cw['lo']:+.2f}, {cw['hi']:+.2f}] "
           f"p {cw['p']:.4f}, {cw['n_src']} speakers in the pool, {cw['redraws']} redraws; Wilcoxon p "
           f"{cw['wilcoxon']['p']:.2g}, r_rb {cw['wilcoxon']['r_rb']:+.2f}")
    _check(cs['hi'] < 0 and abs(cs['est'] - (0.819 - 0.919)) < 0.001,
           f"C1 - FAISS SIM (50): {cs['est']:+.3f} [{cs['lo']:+.3f}, {cs['hi']:+.3f}] p {cs['p']:.4f}")
    ind = compare(c1, fa, ids, spk, 'sim_wavlm', mode=INDEPENDENT)
    _check(ind['n_src'] == 20 and ind['n_tgt'] == 10 and cs['n_src'] == cs['n_tgt'] == 20,
           f"independent (sensitivity): 20 sources x 10 targets, CI [{ind['lo']:+.3f}, {ind['hi']:+.3f}]; linked "
           f"pool 20")
    prows = [dict(fa[p], pair_id=p) for p in ids]
    gl = gate_l_strata(by['anc_R3_s0_pm_rq_T1@src'], by['anc_faiss@src'], prows)
    gl_e = gate_l_strata(by['anc_R3_s0_pm_rq_T1@src'], by['anc_faiss@src'], prows, src_e={'p226', 'p225'})
    for g, d in gl.items():
        _check(d['n'] == sum(stratum(r['genders']) == g for r in prows) and d['rejects'] == (d['hi'] < 0.01)
               and gl_e[g]['n'] + gl_e[g]['n_no_source_E'] == d['n'],
               f"gate L {g}-gender ({d['n']} pairs, screen: M→F or F→F only): Δ_src {d['est']:+.3f} "
               f"[{d['lo']:+.3f}, {d['hi']:+.3f}] -> {'rejects' if d['rejects'] else 'does not reject'} H0 "
               f"Δ >= +0.01; with a source-E filter {gl_e[g]['n']} kept, {gl_e[g]['n_no_source_E']} left out")
    vc = variance_components([c1[p]['sim_wavlm'] - fa[p]['sim_wavlm'] for p in ids], [spk[p][0] for p in ids],
                             [spk[p][1] for p in ids])
    _check(all(vc[k] >= 0 for k in ('sigma2_a', 'sigma2_b', 'sigma2_c', 'sigma2_e')),
           f"variance components of the C1 - FAISS SIM differences (50 pairs, {vc['n_c']} cells): "
           f"σa² {vc['sigma2_a']:.2e}, σb² {vc['sigma2_b']:.2e}, σc² {vc['sigma2_c']:.2e}, σe² {vc['sigma2_e']:.2e}, "
           f"kept {vc['method']} (REML log-likelihoods {vc['reml_llf']}), converged {vc['converged']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--self_test', action='store_true', help='unit tests, then the screen cross-checks')
    p.add_argument('--screen_dir', default=SCREEN_DIR, help='results/ of the 50-pair screen (anchor_lab/ is read)')
    a = p.parse_args()
    if not a.self_test:
        p.error('nothing to do: stats.py is a library (select_crt_al.py imports it); --self_test runs its tests')
    _unit_tests()
    _screen_tests(a.screen_dir)
    print('stats.py self-test: OK', flush=True)


if __name__ == '__main__':
    main()
