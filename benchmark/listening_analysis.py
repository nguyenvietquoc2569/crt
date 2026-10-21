#!/usr/bin/env python3
"""
Analysis of the listening test of ANALYSIS_PLAN_v1.md §5.4, from the raters' result files of the page in
benchmark/listening/page/ and the key.json of benchmark/listening_stimuli.py. CPU only.

Screening (§5.4), fixed in advance and applied before anything else. A rater is excluded who
  - did not pass the headphone check at the start (the page stops there),
  - rates natural target speech (the gold MOS items, condition nat) <= 2 on more than one gold item,
  - fails an attention item (a rating other than the one the item asks for), or
  - gives |CMOS| >= 2 on 5 or more of the 8 hidden identical FAISS-FAISS pairs.
A result file without a block counts as not having done it; the same rater id twice is an error.
Analysis of the screened raters: MOS and SMOS per condition, and the CMOS preference for CRT-AL (the B-relative-to-A
rating, sign-flipped when CRT-AL was A) in the CRT-AL vs FAISS block (P8) and the CRT-AL vs CRT τ = 0.4 block (K8). Each
comes with a 95 % percentile CI from a crossed bootstrap over raters and items (B = 10,000, seed 20260928: raters and
items drawn with replacement, independently; a rating's weight is the product of the two draw counts). The CMOS
blocks also get the two-sided p of §5.2 and the counts; a block enters its family only with at least 15 screened
raters who completed it (§5.4), which analyze_tests.py checks from n_raters. Secondary: a mixed model with crossed
rater and item random intercepts (statsmodels MixedLM, variance components). Descriptive: the hidden pairs' mean
signed and absolute CMOS (bias), the paired MOS differences between the codec systems, session length, and
how many kept raters gave each answer to the page's background questions (age group, first language, English,
hearing, listening device).
Pilot mode (the 5-rater pilot): session length per block, and from the pilot's CMOS variance components the
smallest rater count in 15-20 with 80 % power at a CMOS difference of 0.2 (2.80 SE <= 0.2); if none, 20 raters and
the MDE they reach. Whether a session passes 60 min decides the two halves of MOS / SMOS (§5.4).
  python listening_analysis.py final --key key.json --results results/ --out LISTENING_RESULTS_v1.md   (+ cmos.json)
  python listening_analysis.py pilot --key key.json --results pilot/ --out LISTENING_PILOT_v1.md
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

import stats

B, SEED = 10000, 20260928
MIN_RATERS, CMOS_TARGET, POWER_Z, SESSION_MIN = 15, 0.2, 2.80, 60.0
BACKGROUND_KEYS = ('age', 'first_language', 'english', 'hearing', 'device')   # the page's BACKGROUND (lt_core.js)
GOLD_MAX, GOLD_FAILS, HIDDEN_ABS, HIDDEN_FAILS = 2, 1, 2, 5
CODEC_PAIRS = (('crtal', 'faiss'), ('crtal', 'crt04'), ('faiss', 'nat_mimi'), ('crtal', 'nat_mimi'))


def load_results(paths):
    out = {}
    for p in paths:
        r = json.load(open(p))
        if r['rater_id'] in out:
            raise SystemExit(f'ERROR: rater {r["rater_id"]} appears twice ({p})')
        out[r['rater_id']] = r
    return out


def result_files(arg):
    return sorted(glob.glob(os.path.join(arg, '*.json'))) if os.path.isdir(arg) else [arg]


# ── screening ────────────────────────────────────────────────

def screen(res, key):
    """{rater: [reasons]}; an empty list is a rater who stays."""
    trials, clips = key['trials'], key['clips']
    out = {}
    for rid, r in res.items():
        why = []
        if not (r.get('headphone') or {}).get('passed'):
            why.append('headphone check not passed')
        resp = r.get('responses') or {}
        gold_low = sum(1 for x in resp.get('mos', []) if trials.get(x['tid'], {}).get('kind') == 'gold'
                       and x.get('rating') is not None and x['rating'] <= GOLD_MAX)
        if gold_low > GOLD_FAILS:
            why.append(f'natural target speech rated <= {GOLD_MAX} on {gold_low} gold items')
        for block in ('mos', 'smos', 'cmos'):
            for x in resp.get(block, []):
                t = trials.get(x['tid'], {})
                if t.get('kind') == 'attention' and x.get('rating') != t['expected']:
                    why.append(f'attention item {x["tid"]} ({block}): {x.get("rating")} instead of {t["expected"]}')
        hid = [x for x in resp.get('cmos', []) if trials.get(x['tid'], {}).get('kind') == 'hidden']
        big = sum(1 for x in hid if x.get('rating') is not None and abs(x['rating']) >= HIDDEN_ABS)
        if big >= HIDDEN_FAILS:
            why.append(f'|CMOS| >= {HIDDEN_ABS} on {big} of {len(hid)} hidden identical pairs')
        out[rid] = why
    return out


# ── ratings ──────────────────────────────────────────────────

def ratings(res, key, keep, block):
    """[(rater, item, condition, value)] of the main trials of a block (MOS, SMOS) for the raters in keep."""
    trials, clips = key['trials'], key['clips']
    out = []
    for rid in keep:
        for x in (res[rid].get('responses') or {}).get(block, []):
            t = trials.get(x['tid'], {})
            if t.get('kind') not in ('main', 'gold') or x.get('rating') is None:
                continue
            out.append((rid, t['pair_id'], clips[x['stim']]['condition'], float(x['rating'])))
    return out


def cmos_values(res, key, keep, kind):
    """[(rater, item, preference for CRT-AL)] of a CMOS block (P8, K8); hidden: the signed B-vs-A rating."""
    trials, clips = key['trials'], key['clips']
    out = []
    for rid in keep:
        for x in (res[rid].get('responses') or {}).get('cmos', []):
            t = trials.get(x['tid'], {})
            if t.get('kind') != kind or x.get('rating') is None:
                continue
            v = float(x['rating'])
            if kind != 'hidden':
                ca, cb = clips[x['a']]['condition'], clips[x['b']]['condition']
                if {ca, cb} != {'crtal', t['y']}:
                    raise SystemExit(f'ERROR: trial {x["tid"]} played {ca} and {cb}, the key says crtal and {t["y"]}')
                v = v if cb == 'crtal' else -v
            out.append((rid, t['pair_id'], v))
    return out


def crossed(values, B_=B, seed=SEED):
    """(mean, replicates) of [(rater, item, value)] under the crossed rater x item bootstrap."""
    raters = sorted({v[0] for v in values})
    items = sorted({v[1] for v in values})
    ri, ii = {r: k for k, r in enumerate(raters)}, {i: k for k, i in enumerate(items)}
    S = np.zeros((len(raters), len(items)))
    N = np.zeros_like(S)
    for r, i, v in values:
        S[ri[r], ii[i]] += v
        N[ri[r], ii[i]] += 1
    rng = np.random.default_rng(seed)
    reps = np.empty(B_)
    for lo in range(0, B_, 1000):
        m = min(1000, B_ - lo)
        cr = stats._counts(rng.integers(0, len(raters), (m, len(raters))), len(raters))
        ci = stats._counts(rng.integers(0, len(items), (m, len(items))), len(items))
        num = np.einsum('br,ri,bi->b', cr, S, ci)
        den = np.einsum('br,ri,bi->b', cr, N, ci)
        bad = den <= 0
        while bad.any():
            k = int(bad.sum())
            cr[bad] = stats._counts(rng.integers(0, len(raters), (k, len(raters))), len(raters))
            ci[bad] = stats._counts(rng.integers(0, len(items), (k, len(items))), len(items))
            num[bad] = np.einsum('br,ri,bi->b', cr[bad], S, ci[bad])
            den[bad] = np.einsum('br,ri,bi->b', cr[bad], N, ci[bad])
            bad = den <= 0
        reps[lo:lo + m] = num / den
    return float(S.sum() / N.sum()), reps


def summarise(values, diff=False):
    if not values:
        return {'n': 0, 'est': None}
    est, reps = crossed(values)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    out = {'est': est, 'lo': float(lo), 'hi': float(hi), 'se': float(np.std(reps, ddof=1)), 'n': len(values),
           'n_raters': len({v[0] for v in values}), 'n_items': len({v[1] for v in values})}
    if diff:
        out['p'] = stats.p_two_sided(reps)
    return out


def mixed(values):
    """Secondary: value ~ 1 + (1 | rater) + (1 | item), REML, as stats.variance_components fits it."""
    try:
        vc = stats.variance_components([v[2] for v in values], [v[0] for v in values], [v[1] for v in values])
    except Exception as e:                                     # a degenerate pilot (e.g. all ratings equal)
        return {'error': f'{type(e).__name__}: {e}'}
    return {'mu': vc['mu'], 'sigma2_rater': vc['sigma2_a'], 'sigma2_item': vc['sigma2_b'],
            'sigma2_cell': vc['sigma2_c'], 'sigma2_e': vc['sigma2_e'], 'method': vc['method']}


def paired_diff(rat, a, b):
    """Per rater and item, MOS(a) - MOS(b) where both were rated: [(rater, item, difference)]."""
    by = {}
    for r, i, c, v in rat:
        by.setdefault((r, i), {})[c] = v
    return [(r, i, d[a] - d[b]) for (r, i), d in sorted(by.items()) if a in d and b in d]


def final(res, key):
    sc = screen(res, key)
    keep = sorted(r for r, why in sc.items() if not why)
    out = {'screening': {r: why for r, why in sorted(sc.items())}, 'kept': keep, 'n_kept': len(keep)}
    for block in ('mos', 'smos'):
        rat = ratings(res, key, keep, block)
        conds = sorted({c for _, _, c, _ in rat})
        out[block] = {c: summarise([(r, i, v) for r, i, cc, v in rat if cc == c]) for c in conds}
        if block == 'mos':
            out['mos_codec_diffs'] = {f'{a}-{b}': summarise(paired_diff(rat, a, b), diff=True)
                                      for a, b in CODEC_PAIRS if a in conds and b in conds}
    for kind in ('P8', 'K8'):
        v = cmos_values(res, key, keep, kind)
        s = summarise(v, diff=True)
        s['n_raters'] = len({x[0] for x in v})
        s['mixed'] = mixed(v) if v else None
        out[kind] = s
    hid = cmos_values(res, key, keep, 'hidden')
    out['hidden'] = dict(summarise(hid), mean_abs=float(np.mean([abs(x[2]) for x in hid])) if hid else None)
    out['session_min'] = session_minutes(res, keep)
    out['background'] = background(res, keep)
    return out


def background(res, keep):
    """Descriptive: how many kept raters gave each answer to the page's background questions; a file without the
    answers (or without one of them) counts as 'not answered'."""
    out = {}
    for k in BACKGROUND_KEYS:
        n = {}
        for r in keep:
            v = (res[r].get('background') or {}).get(k) or 'not answered'
            n[v] = n.get(v, 0) + 1
        out[k] = dict(sorted(n.items()))
    return out


def session_minutes(res, keep):
    """Session length in minutes: wall clock from the page's started to finished (the 60-minute rule of §5.4), and
    the time on trial screens (block_ms, without instructions and breaks) beside it."""
    from datetime import datetime
    wall, trial = [], []
    for r in keep:
        x = res[r]
        if x.get('started') and x.get('finished'):
            t0, t1 = (datetime.fromisoformat(x[k].replace('Z', '+00:00')) for k in ('started', 'finished'))
            wall.append((t1 - t0).total_seconds() / 60)
        if x.get('block_ms'):
            trial.append(sum(v or 0 for v in x['block_ms'].values()) / 60000)
    use = wall or trial
    return {'median': float(np.median(use)) if use else None, 'max': float(max(use)) if use else None,
            'n': len(use), 'basis': 'wall clock' if wall else 'trial screens',
            'trial_screens_median': float(np.median(trial)) if trial else None}


def pilot(res, key):
    """Session length and the rater count for 80 % power at a CMOS difference of 0.2 (§5.4)."""
    sc = screen(res, key)
    keep = sorted(r for r, why in sc.items() if not why)
    out = {'screening': sc, 'kept': keep, 'session_min': session_minutes(res, keep),
           'background': background(res, keep)}
    out['halves'] = bool(out['session_min']['median'] and out['session_min']['median'] > SESSION_MIN)
    for kind in ('P8', 'K8'):
        v = cmos_values(res, key, keep, kind)
        n_items = len({x[1] for x in v})
        m = mixed(v) if v else {'error': 'no ratings'}
        row = {'mixed': m, 'n_items': n_items}
        if 'error' not in m:
            def se(n_r):
                return math.sqrt(m['sigma2_rater'] / n_r + m['sigma2_item'] / n_items +
                                 (m['sigma2_cell'] + m['sigma2_e']) / (n_r * n_items))
            row['se_by_raters'] = {n: se(n) for n in range(MIN_RATERS, 21)}
            ok = [n for n in range(MIN_RATERS, 21) if POWER_Z * se(n) <= CMOS_TARGET]
            row['raters'] = ok[0] if ok else 20
            row['mde_at_raters'] = POWER_Z * se(row['raters'])
        out[kind] = row
    return out


# ── report ───────────────────────────────────────────────────

def f(x, n=3):
    return 'n/a' if x is None else f'{x:.{n}f}'


def ci(x):
    return f'{f(x.get("est"), 2)} [{f(x.get("lo"), 2)}, {f(x.get("hi"), 2)}]' if x and x.get('est') is not None \
        else 'n/a'


def render_final(o):
    L = ['# Listening test results v1 (ANALYSIS_PLAN_v1.md §5.4)', '',
         f'Screened raters kept: {o["n_kept"]} of {len(o["screening"])}. Crossed bootstrap over raters and items, '
         f'B = {B}, seed {SEED}.', '', '## Screening', '']
    L += [f'- {r}: {"kept" if not w else "excluded: " + "; ".join(w)}' for r, w in o['screening'].items()]
    for block, title in (('mos', 'MOS (naturalness)'), ('smos', 'SMOS (similarity to the target)')):
        L += ['', f'## {title}', '', '| Condition | Mean [95 % CI] | ratings | raters | items |', '|---|---|---|---|---|']
        L += [f'| {c} | {ci(x)} | {x["n"]} | {x.get("n_raters", 0)} | {x.get("n_items", 0)} |'
              for c, x in o[block].items()]
    L += ['', '### MOS differences between the codec systems', '', '| Pair | Δ [95 % CI] | p |', '|---|---|---|']
    L += [f'| {k} | {ci(x)} | {f(x.get("p"), 4)} |' for k, x in o['mos_codec_diffs'].items()]
    L += ['', '## CMOS: preference for CRT-AL (−3..+3)', '',
          '| Block | Mean [95 % CI] | p | screened raters | enters its family (≥ 15 raters) | mixed model μ |',
          '|---|---|---|---|---|---|']
    for kind, label in (('P8', 'P8: CRT-AL vs FAISS'), ('K8', 'K8: CRT-AL vs CRT τ = 0.4')):
        x = o[kind]
        L.append(f'| {label} | {ci(x)} | {f(x.get("p"), 4)} | {x["n_raters"]} | {x["n_raters"] >= MIN_RATERS} | '
                 f'{f((x.get("mixed") or {}).get("mu"))} |')
    h = o['hidden']
    L += ['', f'Hidden identical pairs: mean signed CMOS {ci(h)}, mean |CMOS| {f(h.get("mean_abs"))}.',
          f'Session length: median {f(o["session_min"]["median"], 1)} min, max {f(o["session_min"]["max"], 1)} min.']
    L += ['', '## Listeners (kept raters, descriptive)', ''] + render_background(o['background'])
    return '\n'.join(L) + '\n'


def render_background(bg):
    return [f'- {k}: ' + ', '.join(f'{v} {n}' for v, n in bg[k].items()) for k in BACKGROUND_KEYS]


def render_pilot(o):
    s = o['session_min']
    L = ['# Listening test pilot (ANALYSIS_PLAN_v1.md §5.4)', '',
         f'Raters kept after screening: {len(o["kept"])} of {len(o["screening"])}.',
         f'Session length: median {f(s["median"], 1)} min, max {f(s["max"], 1)} min -> '
         + ('two balanced halves of MOS / SMOS per rater' if o['halves'] else 'full sessions') + '.', '']
    for kind in ('P8', 'K8'):
        x = o[kind]
        if 'raters' in x:
            L.append(f'- {kind}: {x["raters"]} raters (2.80 SE = {f(x["mde_at_raters"])}; target 0.2).')
        else:
            L.append(f'- {kind}: no estimate ({x["mixed"].get("error")}).')
    L += ['', 'Listeners (kept raters):'] + render_background(o['background'])
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('mode', choices=('final', 'pilot'))
    ap.add_argument('--key', required=True)
    ap.add_argument('--results', required=True, help='a folder of result files or one file')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    key = json.load(open(a.key))
    res = load_results(result_files(a.results))
    o = final(res, key) if a.mode == 'final' else pilot(res, key)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fo:
        fo.write(render_final(o) if a.mode == 'final' else render_pilot(o))
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as fo:
        json.dump(o, fo, indent=1, default=float)
    if a.mode == 'final':
        cmos = {k: {x: o[k].get(x) for x in ('est', 'lo', 'hi', 'p', 'n_raters')} for k in ('P8', 'K8')}
        with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'cmos.json'), 'w') as fo:
            json.dump(cmos, fo, indent=1)
    print(open(a.out).read())
    return 0


if __name__ == '__main__':
    sys.exit(main())
