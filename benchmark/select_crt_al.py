#!/usr/bin/env python3
"""
Dev selection of ANALYSIS_PLAN_v1.md §4 (S2 τ rule §4.2, S1u setting and E17 slot §4.3, CRT-AL §4.4 with its
pre-specified checks), the size check of §5.3 and the listening-item draw of §5.4. Writes the selection record of
§4.6 (--out, default benchmark/DEV_SELECTION_RECORD_v1.md; also printed), <out stem>.json with every number and
<out stem>.listening_items.json (the items and the seeded replacement order; its sha256 is in the record). CPU only;
statistics from stats.py: linked crossed bootstrap of the Dev speakers, B = 10,000, seed 20260928.

Rows. --metrics: metrics.py files (the D1 anchor-lab metrics and metrics_src, the D2 decode-lab metrics after
lab_rows.py, the Mimi copies, the natural controls), each read as metrics.load_results; a (pair_id, system) in two
files is an error. Rows of another set or budget than dev / 30 and pair_ids outside --pairs are ignored and counted.
--rename old=new maps other row names onto SYSTEMS (a name without @src renames its @src rows too); the screen
harness maps r3a_* / r4p_* / lab_* rows this way. --pairs (pairs/dev.jsonl) is the pair universe: wer_primary,
genders, speakers and P come from it, and its row order k gives the 50-pair screen, k % 20 == 0 (--shard 0/20).
Common set (§4.1): the --pairs pairs with a row without error for S1, every S2 τ, A5 and every candidate and an @src
row without error for S1 and every candidate. A required system without any row is reported missing and left out of
that requirement, so the rule still runs on what exists; the record is then INCOMPLETE, as it is when a rule, a
condition, the 950-pair check or a size-check fit cannot be evaluated, when C1 differs from its anchor-lab twin beyond
the tolerance, or when an input file is missing or a job of --jobs is not in --sacct. Over the common set: WER / CER
(Whisper large-v3, corpus) over its wer_primary pairs, SIM, SECS, UTMOS over all of it, dB-MCD over its pairs with
P, SIM_src and M9 (D = SIM_src - SIM_tgt; share D > 0) per gender stratum (same F2F, M2M; cross F2M, M2F). A value
missing or not finite in a row without error leaves that pair out of that quantity (stats.py counts it).
WER (§0.3 row 20): capped corpus WER (M3c) in E4, steps 1-2 of §4.4 and its tolerance sensitivity, S1u, E17, the
gender-pair table and the size check; v0's uncapped corpus WER in the §4.2 τ rule. Every WER table prints both, and
§4.4 re-run with uncapped WER in E4 and steps 1-2 gives the 'uncapped' sensitivity line (reported, never used).
Choices the plan leaves open (each named in the record where it acts):
  §4.2  'tie' = equal WER at two decimals (plan), broken by the lower unrounded dB-MCD, then by the lower τ;
        SIM >= FAISS and the SIM deficit unrounded.
  §4.3  S1u and E17 on the common pairs where their own rows have no error, FAISS and S2 recomputed on those pairs;
        S1u ties as §4.2; E17 'lower WER' = lower at two decimals.
  §4.4  E1-E3 on unrounded point estimates; a condition that cannot be computed (e.g. no @src rows) counts as not met,
        shows as n/a and makes the record INCOMPLETE; F1 ranks by step 4's order; F2 compares the leakage excess and
        then SIM at three decimals (step 4's SIM tie), breaks remaining ties by the lower candidate number and puts
        candidates without SIM_src last. 950-pair check: §4.4 on the common pairs outside the screen, S2 = the τ_dev
        of all pairs.
  C1    cand_pm_rq_T1; compared per pair with anc_R3_s0_pm_rq_T1 (D1) where both exist: identical Whisper hypotheses
        and |ΔSIM| <= 0.001, |ΔUTMOS| <= 0.01 (the §8.1 item 9 tolerance).
Size check (§5.3): Dev differences P = CRT-AL - S1, K = CRT-AL - S3, gate L = CRT-AL@src - S1@src per stratum (under
F3: V = S2 - S1 only), WER linearised with the capped error counts (the uncapped Dev mean printed beside) ->
stats.variance_components -> stats.size_check on --vctk_pairs (5 targets), --vctk10_pairs and --ltc_pairs, with the
source-E files for gate L; the wer_primary count of each VCTK-2k list from its summary.json (and counted from the
list).
Listening items (§5.4): per test set (VCTK-2k: the variant the size check picks, the 5-target list when it cannot
decide) and gender combination, the pair_ids sorted, then shuffled with random.Random('20260928:listening:<set>:<g>')
(the make_pairs.rng convention); the first 4 are the items, the rest is the replacement order.
Evidence from files: --repro JSON files, each with a top-level boolean "passed" (the §4.4 reproduction check: D1 / D2
vs D0 vs the screen); a false one marks the record STOPPED (still written, exit 1). --sacct files of `sacct -P -X -o
JobID,JobName,Submit,End,State` output, with --jobs role=jobid for 2634, D0, D1, D2. --baselines: a JSON list of the
baselines that will run (§2.4; copied). The sha256 of every input file is recorded.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats  # noqa: E402

SRC = '@src'
TAU_GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))
SYSTEMS = {
    'S1': 'anc_faiss',                                              # FAISS (crt_anchor_lab, D1)
    'S2': {t: f'anc_R1_r0_tau{t:g}' for t in TAU_GRID},            # CRT(τ), published anchor, τ grid (D1)
    'S3': 'anc_R1_r0_tau0.4',                                       # CRT τ = 0.4, the SPIE setting
    'A5': 'anc_src0_faiss',                                         # source CB0 swapped into FAISS, no CRT
    'C1': 'cand_pm_rq_T1', 'C2': 'cand_pm18_rq', 'C3': 'cand_gv1_pm_cont_T1', 'C4': 'cand_gv1_shrink0.85_cont',
    'C5': 'cand_gv0.5_shrink0.75_cont',                             # D2, crt_decode_lab --out_dir .../cand
    'C1_anchor': 'anc_R3_s0_pm_rq_T1',                              # C1 decoded inside crt_anchor_lab (D1)
    'S1u': ('front_us32_w0.2_0.2_0.1', 'front_us32_w0.5_0.3_0.3'),  # D2, --out_dir .../front
    'E17': 'front_lmbr_k2_g1_T1',
    'controls': ('source_copy', 'target_copy', 'mimi_copy_src', 'mimi_copy_P'),
}
CANDS = ('C1', 'C2', 'C3', 'C4', 'C5')
SIM_FLOOR, MARGIN = 0.80, 0.01
TOLS = (0.0, 1.0, 2.0)
SCREEN_MOD = 20
GENDERS = ('F2F', 'F2M', 'M2F', 'M2M')
PER_CELL = 4
QUANT = {   # name: (rows, pair subset, metric); wer / cer capped (M3c, WER(X) of §4.1), wer_u / cer_u uncapped
    'wer': ('tgt', 'wer', 'wer_capped'), 'cer': ('tgt', 'wer', 'cer_capped'), 'wer_u': ('tgt', 'wer', 'wer'),
    'cer_u': ('tgt', 'wer', 'cer'), 'sim': ('tgt', 'all', 'sim_wavlm'),
    'secs': ('tgt', 'all', 'secs_resemblyzer'), 'utmos': ('tgt', 'all', 'utmos'), 'mcd': ('tgt', 'par', 'mcd_db'),
    'src_same': ('src', 'same', 'sim_wavlm'), 'src_cross': ('src', 'cross', 'sim_wavlm'),
    'm9_same': ('m9', 'same', 'm9'), 'm9_cross': ('m9', 'cross', 'm9')}
PREC = {'wer': 2, 'cer': 2, 'wer_u': 2, 'cer_u': 2, 'sim': 3, 'secs': 3, 'utmos': 2, 'mcd': 2, 'src_same': 3,
        'src_cross': 3, 'm9_same': 3, 'm9_cross': 3}
SIZE_KINDS = (('wer', 1, 'wer'), ('sim', 3, 'sim_wavlm'), ('utmos', 5, 'utmos'), ('mcd', 7, 'mcd_db'))
BASELINE_KEYS = ('id', 'system', 'repo_commit', 'checkpoint_sha256', 'sample_rate', 'command')
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DEV_SELECTION_RECORD_v1.md')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help='pairs/dev.jsonl (the pair universe and the screen order)')
    p.add_argument('--metrics', required=True, help='comma list of metrics.py files (target-side and @src rows)')
    p.add_argument('--rename', default='', help='comma list old=new of row system names')
    p.add_argument('--vctk_pairs', default='', help='pairs/vctk2k.jsonl (5 targets per source utterance)')
    p.add_argument('--vctk_summary', default='', help='its summary.json (default: next to it)')
    p.add_argument('--vctk_source_e', default='', help='its vctk2k_source_E.json (default: next to it)')
    p.add_argument('--vctk10_pairs', default='', help='the 10-target VCTK-2k list (make_pairs --vctk_targets 10)')
    p.add_argument('--vctk10_summary', default='', help='its summary.json (default: next to it)')
    p.add_argument('--vctk10_source_e', default='', help='its vctk2k_source_E.json (default: next to it)')
    p.add_argument('--ltc_pairs', default='', help='pairs/ltc2k.jsonl')
    p.add_argument('--repro', default='', help='comma list of reproduction-check JSON files {"passed": bool, ...}')
    p.add_argument('--sacct', default='', help='comma list of `sacct -P -X` output files')
    p.add_argument('--jobs', default='2634=2634', help='comma list role=jobid (roles 2634, D0, D1, D2)')
    p.add_argument('--baselines', default='', help='JSON list of the baselines that will run (§2.4)')
    p.add_argument('--out', default=DEFAULT_OUT, help='record markdown; the JSON and the item list go next to it')
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def next_to(path, explicit, name):
    """An explicit path, else `name` in the directory of `path` if it exists there."""
    if explicit or not path:
        return explicit
    p = os.path.join(os.path.dirname(path), name)
    return p if os.path.exists(p) else ''


def renamed(system, ren):
    if system in ren:
        return ren[system]
    base = system[:-len(SRC)] if system.endswith(SRC) else None
    return ren[base] + SRC if base in ren and not ren[base].endswith(SRC) else system


# ── data ─────────────────────────────────────────────────────

class Dev:
    """Rows by system ({system: {pair_id: row}}) and the pair list; quantities of QUANT over a list of pair_ids."""

    def __init__(self, R, pairs):
        self.R = R
        self.info = {p['pair_id']: p for p in pairs}
        self.spk = {k: (p['source_speaker'], p['target_speaker']) for k, p in self.info.items()}
        self._m9 = {}

    def rows(self, system, kind='tgt'):
        if kind == 'tgt':
            return self.R.get(system, {})
        if kind == 'src':
            return self.R.get(system + SRC, {})
        if system not in self._m9:
            self._m9[system] = stats.margin_rows(self.rows(system), self.rows(system, 'src'))
        return self._m9[system]

    def has(self, system, kind='tgt'):
        """The system has rows of this kind (m9: both its target-side and its @src rows)."""
        if not system:
            return False
        tgt, src = bool(self.R.get(system)), bool(self.R.get(system + SRC))
        return {'tgt': tgt, 'src': src, 'm9': tgt and src}[kind]

    def ok(self, system, pid, kind='tgt'):
        return not stats.failed(self.rows(system, kind).get(pid))

    def subset(self, pids, which):
        keep = {'all': lambda p: True, 'wer': lambda p: p['wer_primary'], 'par': lambda p: bool(p.get('P')),
                'same': lambda p: stats.stratum(p['genders']) == 'same',
                'cross': lambda p: stats.stratum(p['genders']) == 'cross'}.get(which, lambda p: p['genders'] == which)
        return [k for k in pids if keep(self.info[k])]

    def point(self, system, pids, q, sub=None):
        kind, s, metric = QUANT[q]
        if not self.has(system, kind):
            return None
        return stats.point(self.rows(system, kind), self.subset(pids, sub or s), metric)[0]

    def level(self, system, pids, q):
        kind, s, metric = QUANT[q]
        return stats.level(self.rows(system, kind), self.subset(pids, s), self.spk, metric) \
            if self.has(system, kind) else None

    def diff(self, a, b, pids, q):
        kind, s, metric = QUANT[q]
        if not (self.has(a, kind) and self.has(b, kind)):
            return None
        return stats.compare(self.rows(a, kind), self.rows(b, kind), self.subset(pids, s), self.spk, metric)

    def share_pos(self, system, pids, sub):
        if not self.has(system, 'm9'):
            return None, 0
        v = [self.rows(system, 'm9')[k]['m9'] for k in self.subset(pids, sub)
             if self.ok(system, k, 'm9')]
        return (float(np.mean(np.array(v) > 0)) if v else None), len(v)


def load_metrics(paths, ren):
    rows, files = {}, []
    for path in paths:
        got = stats.load_results(path)
        files.append({'path': os.path.abspath(path), 'sha256': sha256(path), 'rows': len(got)})
        for r in got:
            r = dict(r, system=renamed(r['system'], ren))
            k = (r['pair_id'], r['system'])
            if k in rows:
                raise SystemExit(f'ERROR: {k} occurs in two --metrics files (or twice after --rename); '
                                 f'last in {path}')
            rows[k] = r
    return rows, files


# ── §4.2, §4.3 ───────────────────────────────────────────────

def _r(x, n=None, missing=math.inf):
    """Sort key: x rounded to n decimals (None: unrounded); a missing value sorts as `missing`."""
    return missing if x is None else x if n is None else round(x, n)


def tau_rule(dev, common):
    s1 = SYSTEMS['S1']
    sim1 = dev.point(s1, common, 'sim')
    grid, notes = [], []
    for t in TAU_GRID:
        s = SYSTEMS['S2'][t]
        g = {'tau': t, 'system': s, 'present': dev.has(s)}
        if g['present']:        # v0's rule: uncapped WER (wer_u) is its input, capped WER printed beside (§0.3 row 20)
            g.update({q: dev.point(s, common, q) for q in ('wer_u', 'wer', 'sim', 'mcd')})
            g['meets'] = None if g['sim'] is None or sim1 is None else bool(g['sim'] >= sim1)
            g['deficit'] = None if g['sim'] is None or sim1 is None else sim1 - g['sim']
        grid.append(g)
    out = {'grid': grid, 'sim_s1': sim1, 'tau': None, 'system': None, 'step': None, 'notes': notes,
           'missing': [g['system'] for g in grid if not g['present']]}
    use = [g for g in grid if g['present'] and g['wer_u'] is not None and g['sim'] is not None]
    if not use or sim1 is None:
        notes.append('not computable: no τ system with WER and SIM, or no FAISS SIM')
        return out
    if out['missing']:
        notes.append(f'run on {len(use)} of {len(TAU_GRID)} τ values; missing {out["missing"]}')
    ok = [g for g in use if g['meets']]
    if ok:
        best = min(round(g['wer_u'], 2) for g in ok)
        tied = [g for g in ok if round(g['wer_u'], 2) == best]
        pick, step = tied[0], 1
        if len(tied) > 1:
            low = min(_r(g['mcd']) for g in tied)
            tied2 = [g for g in tied if _r(g['mcd']) == low]
            pick, step = tied2[0], 2
            notes.append(f'step 2: τ {[g["tau"] for g in tied]} tie at {best:.2f} % uncapped WER; the lower dB-MCD '
                         f'decides')
            if len(tied2) > 1:
                notes.append(f'τ {[g["tau"] for g in tied2]} also tie on dB-MCD: the lowest τ taken (not in the plan)')
    else:
        pick, step = min(use, key=lambda g: (g['deficit'], g['tau'])), 3
        notes.append('no τ meets SIM >= FAISS: step 3, the smallest SIM deficit')
    out.update(tau=pick['tau'], system=pick['system'], step=step)
    return out


def s1u_rule(dev, common):
    s1, sets = SYSTEMS['S1'], SYSTEMS['S1u']
    have = [s for s in sets if dev.has(s)]
    out = {'settings': list(sets), 'missing': [s for s in sets if s not in have], 'setting': None, 'how': None}
    if not have:
        out['how'] = 'not computable: no S1u rows'
        return out
    pids = [k for k in common if all(dev.ok(s, k) for s in have)]
    out['n_pairs'] = len(pids)
    out['q'] = {s: {q: dev.point(s, pids, q) for q in ('wer', 'wer_u', 'sim', 'mcd', 'utmos')} for s in have + [s1]}
    q = out['q']
    if q[s1]['sim'] is None or any(q[s]['wer'] is None or q[s]['sim'] is None for s in have):
        out['how'] = 'not computable: missing WER or SIM'
        return out
    meets = [s for s in have if q[s]['sim'] >= q[s1]['sim']]
    if len(meets) == 2:
        w = [round(q[s]['wer'], 2) for s in meets]
        if w[0] != w[1]:
            out.update(setting=meets[int(w[1] < w[0])], how='both have SIM >= FAISS: the lower capped WER')
        else:
            out.update(setting=min(meets, key=lambda s: (_r(q[s]['mcd']), meets.index(s))),
                       how='both have SIM >= FAISS and equal capped WER at two decimals: the lower dB-MCD')
    elif len(meets) == 1:
        out.update(setting=meets[0], how='the only setting with SIM >= FAISS')
    else:
        out.update(setting=max(have, key=lambda s: (q[s]['sim'], -have.index(s))),
                   how='neither setting has SIM >= FAISS: the higher SIM')
    if out['missing']:
        out['how'] += f' (only {have} present)'
    return out


def e17_rule(dev, common, s2):
    e, s1 = SYSTEMS['E17'], SYSTEMS['S1']
    out = {'system': e, 'enters': None}
    if not dev.has(e) or not s2:
        out['how'] = 'not computable: ' + ('no E17 rows' if not dev.has(e) else 'no S2')
        return out
    pids = [k for k in common if dev.ok(e, k)]
    q = {s: {x: dev.point(s, pids, x) for x in ('wer', 'wer_u', 'sim')} for s in (e, s2, s1)}
    out.update(n_pairs=len(pids), q=q)
    if any(v is None for d in q.values() for v in d.values()):
        out['how'] = 'not computable: missing WER or SIM'
        return out
    lower, sim_ok = round(q[e]['wer'], 2) < round(q[s2]['wer'], 2), q[e]['sim'] >= q[s1]['sim']
    out.update(enters=bool(lower and sim_ok), how=f"capped WER {q[e]['wer']:.2f} {'<' if lower else '>='} S2 "
               f"{q[s2]['wer']:.2f} (two decimals; uncapped {q[e]['wer_u']:.2f} vs {q[s2]['wer_u']:.2f}); SIM "
               f"{q[e]['sim']:.3f} {'>=' if sim_ok else '<'} FAISS {q[s1]['sim']:.3f}")
    return out


# ── §4.4 ─────────────────────────────────────────────────────

def best_sim(T, q):
    """Step 3-4: the highest SIM; ties at three decimals -> lower dB-MCD (two decimals) -> higher UTMOS (two
    decimals) -> lower candidate number."""
    return min(T, key=lambda c: (-_r(q[c]['sim'], 3, -math.inf), _r(q[c]['mcd'], 2),
                                 -_r(q[c]['utmos'], 2, -math.inf), c))


def e4_met(d):
    """E4 from {'S2': compare(), 'A5': compare()}: both upper CI limits below 0; None when one cannot be computed."""
    v = [None if x is None or x.get('hi') is None else bool(x['hi'] < 0) for x in d.values()]
    return False if False in v else None if None in v else True


def crt_al_rule(dev, pids, s2, levels=False, wer='wer'):
    """§4.4 with `wer` the WER of E4 and steps 1-2: 'wer' capped (the rule), 'wer_u' uncapped (the sensitivity analysis
    of §0.3 row 20, reported, never used). E4 and the step-2 differences are computed with both, for the record."""
    s1, a5 = SYSTEMS['S1'], SYSTEMS['A5']
    lab = 'capped' if wer == 'wer' else 'uncapped'
    cands = [c for c in CANDS if dev.has(SYSTEMS[c])]
    out = {'n_pairs': len(pids), 'n_wer_primary': len(dev.subset(pids, 'wer')), 's2': s2, 'wer': wer,
           'candidates': cands, 'missing': [c for c in CANDS if c not in cands], 'trace': [], 'choice': None,
           'fallback': None}
    if not pids or not cands:
        out['trace'].append('no pairs: not computable' if not pids else 'no candidate rows: not computable')
        return out
    refs = [s for s in (s1, s2, a5) if dev.has(s)]
    sysname = {c: SYSTEMS[c] for c in cands}
    q = {c: {k: dev.point(sysname[c], pids, k) for k in QUANT} for c in cands}
    ref = {s: {k: dev.point(s, pids, k) for k in QUANT} for s in refs}
    out.update(q=q, ref=ref)
    if levels:
        out['levels'] = {x: {k: dev.level(SYSTEMS.get(x, x), pids, k) for k in QUANT} for x in cands + refs}
        out['share_pos'] = {x: {g: dev.share_pos(SYSTEMS.get(x, x), pids, g) for g in stats.STRATA}
                            for x in cands + refs}
    r1 = ref.get(s1, {})
    e4 = {c: {w: {'S2': dev.diff(sysname[c], s2, pids, w), 'A5': dev.diff(sysname[c], a5, pids, w)}
              for w in ('wer', 'wer_u')} for c in cands}
    flags = {}
    for c in cands:
        e3 = [None if q[c][f'src_{g}'] is None or r1.get(f'src_{g}') is None else
              bool(q[c][f'src_{g}'] <= r1[f'src_{g}'] + MARGIN) for g in stats.STRATA]
        flags[c] = {'E1': None if q[c]['sim'] is None else bool(q[c]['sim'] >= SIM_FLOOR),
                    'E2': None if q[c]['utmos'] is None or r1.get('utmos') is None else
                    bool(q[c]['utmos'] >= r1['utmos']),
                    'E3': False if False in e3 else None if None in e3 else True,
                    'E4': e4_met(e4[c][wer])}
    out.update(e4=e4, flags=flags)

    def meets(c, keys):
        return all(flags[c][k] is True for k in keys)
    tr = out['trace']
    elig = [c for c in cands if meets(c, ('E1', 'E2', 'E3', 'E4'))]
    out['eligible'] = elig
    na = [f'{c} {k}' for c in cands for k, v in flags[c].items() if v is None]
    if na:
        tr.append(f'not computable, counted as not met: {", ".join(na)}')
    tr.append(f'eligible (E1-E4): {elig or "none"}')
    if elig:
        m = min(elig, key=lambda c: (round(q[c][wer], 2), c))
        d = {w: {c: dev.diff(sysname[c], sysname[m], pids, w) for c in elig if c != m} for w in ('wer', 'wer_u')}
        dw = d[wer]
        s = {c: 0.0 if c == m else dw[c]['se'] for c in elig}
        delta = {c: 0.0 if c == m else dw[c]['est'] for c in elig}
        T = [c for c in elig if delta[c] <= s[c]]
        out.update(c_m=m, diff_vs_m=d, s_k=s, T=T, choice=best_sim(T, q))
        tr.append(f'step 1: C_m = {m}, the lowest {lab} WER ({q[m][wer]:.2f} %; ties at two decimals -> lower number)')
        tr.append(f'step 2: s_k = bootstrap SE of {lab} WER(Ck) - WER(C_m); T = ' + '{Ck: WER(Ck) - WER(C_m) <= s_k} = '
                  + ', '.join(f'{c} ({delta[c]:+.2f} vs {s[c]:.2f})' for c in elig) + f' -> T = {T}')
        tr.append(f'steps 3-4: highest SIM in T (ties at three decimals -> dB-MCD, UTMOS, number): {out["choice"]} '
                  f'(SIM ' + ', '.join(f'{c} {q[c]["sim"]:.3f}' for c in T) + ')')
        Ts = {f'fixed tolerance {t:g} points': [c for c in elig if delta[c] <= t] for t in TOLS}
        Ts['95 % CI of WER(Ck) − WER(C_m) contains 0'] = [c for c in elig if c == m or dw[c]['lo'] <= 0 <= dw[c]['hi']]
        out['sensitivity'] = {k: {'T': v, 'choice': best_sim(v, q)} for k, v in Ts.items()}
    elif any(meets(c, ('E3', 'E4')) for c in cands):
        pool = [c for c in cands if meets(c, ('E3', 'E4'))]
        out.update(fallback='F1', choice=best_sim(pool, q))
        c = out['choice']
        out['deficits'] = {'E1': None if q[c]['sim'] is None else q[c]['sim'] - SIM_FLOOR,
                           'E2': None if q[c]['utmos'] is None or r1.get('utmos') is None else
                           q[c]['utmos'] - r1['utmos']}
        tr.append(f'F1: candidates meeting E3 and E4: {pool}; the highest SIM (step 4 order): {c}; SIM - 0.80 = '
                  f'{fm(out["deficits"]["E1"], 3, True)}, UTMOS - UTMOS(S1) = {fm(out["deficits"]["E2"], 2, True)}')
    elif any(meets(c, ('E4',)) for c in cands):
        pool = [c for c in cands if meets(c, ('E4',))]

        def excess(c):
            v = [q[c][f'src_{g}'] - r1[f'src_{g}'] for g in stats.STRATA
                 if q[c][f'src_{g}'] is not None and r1.get(f'src_{g}') is not None]
            return max(v) if len(v) == len(stats.STRATA) else None
        ex = {c: excess(c) for c in pool}
        out.update(fallback='F2', excess=ex,
                   choice=min(pool, key=lambda c: (_r(ex[c], 3), -_r(q[c]['sim'], 3, -math.inf), c)))
        tr.append(f'F2: no candidate meets E3 and E4; E4 holds for {pool}; leakage excess max_g SRC_g(Ck) - '
                  f'SRC_g(S1): ' + ', '.join(f'{c} {"n/a" if ex[c] is None else f"{ex[c]:+.3f}"}' for c in pool)
                  + f' -> {out["choice"]} (ties at three decimals: higher SIM (three decimals), then lower number; '
                    f'n/a ranks last)')
    else:
        out['fallback'] = 'F3'
        tr.append('F3: no candidate meets E4: no CRT-AL; the primary system reverts to S2 and the families to v0 §5 '
                  '(V primary with v0 P8; K and L dropped); no candidate runs on the test sets')
    return out


def c1_check(dev, common):
    a, b = SYSTEMS['C1'], SYSTEMS['C1_anchor']
    out = {'systems': [a, b], 'compared': dev.has(a) and dev.has(b)}
    if not out['compared']:
        out['missing'] = [s for s in (a, b) if not dev.has(s)]
        return out
    ra, rb = dev.rows(a), dev.rows(b)
    ids = [k for k in common if dev.ok(a, k) and dev.ok(b, k)]

    def dmax(key, x, y, ks=ids):
        v = [abs(x[k][key] - y[k][key]) for k in ks
             if all(isinstance(r[k].get(key), (int, float)) and math.isfinite(r[k][key]) for r in (x, y))]
        return max(v) if v else None
    hyp = sum(ra[k].get(stats.WER_KEY, {}).get('hyp') == rb[k].get(stats.WER_KEY, {}).get('hyp') for k in ids)
    out.update(n=len(ids), same_hyp=hyp, max_dsim=dmax('sim_wavlm', ra, rb), max_dutmos=dmax('utmos', ra, rb),
               max_dmcd=dmax('mcd_db', ra, rb), wer=[dev.point(s, ids, 'wer') for s in (a, b)],
               wer_u=[dev.point(s, ids, 'wer_u') for s in (a, b)])
    sa, sb = dev.rows(a, 'src'), dev.rows(b, 'src')
    sid = [k for k in ids if not stats.failed(sa.get(k)) and not stats.failed(sb.get(k))]
    out['n_src'], out['max_dsim_src'] = len(sid), dmax('sim_wavlm', sa, sb, sid)
    out['within_tolerance'] = (hyp == len(ids) and (out['max_dsim'] or 0) <= 0.001 and
                               (out['max_dutmos'] or 0) <= 0.01) if ids else None
    out['identical'] = out['within_tolerance'] and not any(out[k] for k in ('max_dsim', 'max_dutmos', 'max_dmcd'))
    return out


def gender_table(dev, common):
    out = {}
    for x in ['S1'] + [c for c in CANDS if dev.has(SYSTEMS[c])]:
        s = SYSTEMS[x]
        out[x] = {}
        for g in GENDERS:
            ids = dev.subset(common, g)
            sp, n9 = dev.share_pos(s, ids, 'all')
            out[x][g] = {'n': len(ids), 'n_wer_primary': len(dev.subset(ids, 'wer')), 'wer': dev.point(s, ids, 'wer'),
                         'wer_u': dev.point(s, ids, 'wer_u'),
                         'sim': dev.point(s, ids, 'sim'), 'utmos': dev.point(s, ids, 'utmos'),
                         'sim_src': dev.point(s, ids, 'src_same', sub='all'),
                         'm9': dev.point(s, ids, 'm9_same', sub='all'), 'share_d_pos': sp, 'n_m9': n9}
    return out


def controls_table(dev, common):
    out = {}
    for s in SYSTEMS['controls']:
        if not dev.has(s):
            continue
        ids = [k for k in common if dev.ok(s, k)]
        out[s] = {'n': len(ids), **{q: dev.point(s, ids, q) for q in ('wer', 'wer_u', 'sim', 'utmos', 'mcd')},
                  'sim_src': dev.point(s, ids, 'src_same', sub='all')}
    return out


# ── §5.3 size check ──────────────────────────────────────────

def vc_inputs(dev, a, b, pids, kind, metric, rows='tgt', capped=True):
    """Per-pair Dev differences of a - b for the variance components: WER linearised over the wer_primary pairs, with
    the capped error counts min(E, N) (M3c) unless capped=False."""
    sub = dev.subset(pids, {'wer': 'wer', 'mcd': 'par', 'gate_same': 'same', 'gate_cross': 'cross'}.get(kind, 'all'))
    ok, (ca, cb), _, _ = stats.collect([dev.rows(a, rows), dev.rows(b, rows)], sub, metric)
    if kind == 'wer':
        if not np.array_equal(ca[:, 1], cb[:, 1]):
            raise ValueError(f'{a} and {b} differ in reference words on some pairs')
        d = stats.wer_linearised(ca[:, 0], cb[:, 0], ca[:, 1], capped=capped) if ok else []
    else:
        d = ca[:, 0] - cb[:, 0]
    return np.asarray(d, float), [dev.spk[k][0] for k in ok], [dev.spk[k][1] for k in ok]


def size_conditions(dev, common, sel, s2):
    s1, s3 = SYSTEMS['S1'], SYSTEMS['S3']
    if sel.get('choice'):
        a, fams, gate = SYSTEMS[sel['choice']], (('P', s1), ('K', s3)), True
    else:
        a, fams, gate = s2, (('V', s1),), False
    conds = []
    specs = [(fam, b, kind, k, metric, 'tgt') for fam, b in fams for kind, k, metric in SIZE_KINDS]
    if gate:
        specs += [('L', s1, f'gate_{g}', g[0], 'sim_wavlm', 'src') for g in stats.STRATA]
    for fam, b, kind, k, metric, rows in specs:
        c = {'kind': kind, 'a': a, 'b': b, 'vc': None, 'delta_dev': None}
        if fam == 'L':
            c.update(vctk2k=f'L1{k}', ltc2k=f'L2{k}', a=a + SRC, b=b + SRC)
        else:
            c.update(vctk2k=f'{fam}{k}', ltc2k=None if kind == 'mcd' else f'{fam}{k + 1}')
        try:
            if not (a and dev.has(a, rows) and dev.has(b, rows)):
                raise ValueError('no rows')
            capped = fam != 'V'     # V's WER hypotheses keep v0's uncapped WER (§0.3 row 20)
            d, src, tgt = vc_inputs(dev, a, b, common, kind, metric, rows, capped=capped)
            c.update(dev_n=len(d), dev_mean=float(d.mean()) if len(d) else None)
            if kind == 'wer':
                do = vc_inputs(dev, a, b, common, kind, metric, rows, capped=not capped)[0]
                c['wer_capped'] = capped
                c['dev_mean_uncapped' if capped else 'dev_mean_capped'] = float(do.mean()) if len(do) else None
            if fam == 'L':
                c['delta_dev'] = c['dev_mean']
            c['vc'] = stats.variance_components(d, src, tgt)
        except Exception as e:      # a fit that fails (or has no data) leaves the condition unevaluated, reported
            c['error'] = f'{type(e).__name__}: {e}'
        conds.append(c)
    return conds


def source_e_speakers(path):
    """make_pairs.py <set>_source_E.json {source speaker: {paths, durs, total_s}} -> the speakers with an E set."""
    return {s for s, e in json.load(open(path)).items() if (e.get('paths') if isinstance(e, dict) else e)}


def wer_primary_count(summary_path, pairs):
    out = {'counted': sum(bool(p['wer_primary']) for p in pairs) if pairs is not None else None, 'summary': None}
    if summary_path:
        out['summary'] = json.load(open(summary_path)).get('sets', {}).get('vctk2k', {}).get('n_pairs_wer_primary')
    return out


# ── §5.4 listening items ─────────────────────────────────────

def listening_order(pairs, name):
    """Per gender combination: the set's pair_ids sorted, then shuffled with the seeded stream of make_pairs.rng."""
    out = {}
    for g in GENDERS:
        ids = sorted(p['pair_id'] for p in pairs if p['genders'] == g)
        random.Random(':'.join(str(t) for t in (stats.SEED, 'listening', name, g))).shuffle(ids)
        out[g] = ids
    return out


def listening_draw(lists, variant, path):
    """lists: {set: (pair rows, pairs file)}; writes the item file and returns its description."""
    sets, items = {}, []
    for name, (pairs, src) in lists.items():
        order = listening_order(pairs, name)
        info = {p['pair_id']: p for p in pairs}
        sets[name] = {'pairs': os.path.abspath(src), 'pairs_sha256': sha256(src), 'order': order}
        for g in GENDERS:
            for i, k in enumerate(order[g][:PER_CELL]):
                items.append({'set': name, 'genders': g, 'rank': i, 'pair_id': k,
                              'source_speaker': info[k]['source_speaker'], 'target_speaker': info[k]['target_speaker']})
    doc = {'seed': stats.SEED, 'per_cell': PER_CELL, 'vctk2k_variant': variant, 'rule': 'ANALYSIS_PLAN_v1 §5.4: '
           'a failed item is replaced by the next pair_id of its set and gender combination in "order"',
           'items': items, 'sets': sets}
    with open(path, 'w') as f:
        json.dump(doc, f, indent=1)
    return {'path': os.path.abspath(path), 'sha256': sha256(path), 'n_items': len(items), 'items': items,
            'vctk2k_variant': variant, 'sets': {k: {'pairs': v['pairs'], 'pairs_sha256': v['pairs_sha256'],
                                                    'n_order': {g: len(v['order'][g]) for g in GENDERS}}
                                                for k, v in sets.items()}}


# ── evidence files ───────────────────────────────────────────

def read_sacct(paths, jobs):
    rows = []
    for path in paths:
        lines = [x.rstrip('\n') for x in open(path) if x.strip()]
        head = lines[0].split('|') if lines else []
        rows += [dict(zip(head, x.split('|'))) for x in lines[1:]]
    out = {}
    for role, jid in jobs.items():
        m = [r for r in rows if r.get('JobID') == jid or str(r.get('JobID', '')).startswith(jid + '_')]
        out[role] = {'job_id': jid, 'rows': m, 'submit': min((r.get('Submit') for r in m), default=None),
                     'end': max((r.get('End') for r in m), default=None),
                     'state': sorted({r.get('State') for r in m}) or None}
    return out


def read_repro(paths):
    out = []
    for path in paths:
        d = json.load(open(path))
        out.append({'path': os.path.abspath(path), 'sha256': sha256(path), 'passed': d.get('passed'),
                    'summary': d.get('summary'), 'content': d})
    return out


# ── record ───────────────────────────────────────────────────

def fm(x, p=2, sign=False):
    return '–' if x is None else f'{x:+.{p}f}' if sign else f'{x:.{p}f}'


def fci(d, p=2, sign=False):
    if not d or d.get('est') is None:
        return '–'
    return f"{fm(d['est'], p, sign)} [{fm(d['lo'], p, sign)}, {fm(d['hi'], p, sign)}]"


def yn(v):
    return {True: 'yes', False: 'no', None: 'n/a'}[v]


def render(res):
    L = ['# Dev selection record (ANALYSIS_PLAN_v1 §4.6)', '',
         f"Written {res['written']} by `benchmark/select_crt_al.py` (sha256 {res['scripts']['select_crt_al.py']}) "
         f"with `benchmark/stats.py` (sha256 {res['scripts']['stats.py']}). Linked crossed bootstrap of the Dev "
         f"speakers, B = {stats.B:,}, seed {stats.SEED}; 95 % percentile CIs.", '',
         f"**Status: {res['status']}**", '']
    L += [f'- {x}' for x in res['problems']] + ([''] if res['problems'] else [])

    sel, chk, inp, rb = res['selection'], res['check_950'], res['inputs'], res['size']['rebuild']
    L += ['## Decisions', '',
          f"- S2: τ_dev = {res['tau_rule']['tau']} (`{res['tau_rule']['system']}`), decided by step "
          f"{res['tau_rule']['step']} of §4.2" + ''.join(f'; {n}' for n in res['tau_rule']['notes']),
          f"- S1u: `{res['s1u']['setting']}` ({res['s1u']['how']})",
          f"- E17 (`{res['e17']['system']}`) enters the test runs: {yn(res['e17']['enters'])} "
          f"({res['e17'].get('how')})",
          f"- CRT-AL: {sel['choice'] or 'none'}" + (f" (`{SYSTEMS[sel['choice']]}`)" if sel['choice'] else '')
          + (f", fallback {sel['fallback']}" if sel['fallback'] else ''),
          '- 950-pair check: ' + (f"{'; '.join(chk['trace'])} ({chk['n_pairs']} common pairs outside the screen)"
                                  if not chk.get('q')
                                  else f"{chk['choice'] or 'none'}"
                                  + (f", fallback {chk['fallback']}" if chk['fallback'] else '')
                                  + (' (same as on all pairs)' if chk['choice'] == sel['choice'] else
                                     ' (differs; CRT-AL stays as chosen on all pairs)')),
          '- VCTK-2k variant for T1: ' + (f"{rb['use']} targets" if rb['use']
                                          else 'not decided (needs both the 5- and the 10-target list)'),
          '- Listening items: ' + (f"{res['listening']['n_items']}, file `{res['listening']['path']}` sha256 "
                                   f"{res['listening']['sha256']}" if res['listening'].get('path')
                                   else 'not drawn (no test pair list given)'), '']

    L += ['## Inputs', '', '| file | rows read | sha256 |', '|---|---|---|']
    L += [f"| `{f['path']}` | {f.get('rows', '–')} | {f['sha256']} |" for f in inp['files'] + inp['evidence']]
    c = inp['common']
    L += ['', f"Pair list `{inp['pairs']['path']}` ({inp['pairs']['n']} pairs, sha256 {inp['pairs']['sha256']}); "
              f"screen = rows k % {SCREEN_MOD} == 0 ({inp['screen_n']} pairs). Ignored rows: "
              f"{inp['ignored'] or 'none'}.",
          f"Common set (§4.1): {c['n']} pairs ({c['n_wer_primary']} wer_primary, {c['n_parallel']} with P, "
          f"{c['n_same']} same-gender, {c['n_cross']} cross-gender); pairs with no row of any required system: "
          f"{c['n_no_rows']}. Excluded per required system (no row or an error row, among the pairs that have rows): "
          + (', '.join(f'`{s}` {n}' for s, n in c['excluded'].items() if n) or 'none') + '.',
          f"Required systems without rows: {', '.join(f'`{s}`' for s in inp['missing']) or 'none'}.", '']

    rp = res['repro']
    L += ['## Reproduction check (§4.4, §8.1 item 9)', '']
    L += [f"- `{r['path']}` (sha256 {r['sha256']}): passed = {r['passed']}"
          + (f"; {r['summary']}" if r['summary'] else '') for r in rp] or ['- not given (--repro)']
    L += ['', '## Jobs (sacct)', '']
    if res['jobs']:
        L += ['| role | job ID | Submit | End | State |', '|---|---|---|---|---|']
        L += [f"| {k} | {v['job_id']} | {v['submit'] or 'not in the sacct files'} | {v['end'] or '–'} | "
              f"{', '.join(v['state'] or ['–'])} |" for k, v in res['jobs'].items()]
    for role in ('2634', 'D0', 'D1', 'D2'):
        if role not in res['jobs']:
            L.append(f'- {role}: no job ID given (--jobs)')
    L.append('')

    tr = res['tau_rule']
    L += ['## S2: the v0 τ rule (§4.2)', '', f"FAISS SIM {fm(tr['sim_s1'], 3)}. WER over the common wer_primary pairs; "
          f"the rule uses v0's uncapped corpus WER, the capped WER (M3c) is printed beside it (§0.3 row 20).", '',
          '| τ | system | uncapped WER % (rule) | capped WER % | SIM | SIM − FAISS | dB-MCD | SIM ≥ FAISS |',
          '|---|---|---|---|---|---|---|---|']
    for g in tr['grid']:
        if not g['present']:
            L.append(f"| {g['tau']} | `{g['system']}` | missing | | | | | |")
        else:
            L.append(f"| {g['tau']} | `{g['system']}` | {fm(g['wer_u'])} | {fm(g['wer'])} | {fm(g['sim'], 3)} | "
                     f"{fm(None if g['deficit'] is None else -g['deficit'], 3, True)} | {fm(g['mcd'])} | "
                     f"{yn(g['meets'])} |")
    L += ['', f"τ_dev = {tr['tau']}, step {tr['step']}." + ''.join(f' {n}.' for n in tr['notes']), '']

    s = res['s1u']
    L += ['## S1u setting and E17 (§4.3)', '']
    if s.get('q'):
        L += [f"On {s['n_pairs']} common pairs; the S1u rule uses the capped WER (M3c), the uncapped WER is printed "
              f"beside it:", '', '| system | capped WER % (rule) | uncapped WER % | SIM | UTMOS | dB-MCD |',
              '|---|---|---|---|---|---|']
        L += [f"| `{k}` | {fm(v['wer'])} | {fm(v['wer_u'])} | {fm(v['sim'], 3)} | {fm(v['utmos'])} | {fm(v['mcd'])} |"
              for k, v in s['q'].items()]
    L += ['', f"S1u: `{s['setting']}` — {s['how']}." + (f" Missing: {s['missing']}." if s['missing'] else ''), '']
    e = res['e17']
    L += [f"E17 `{e['system']}` enters: {yn(e['enters'])} — {e.get('how')}"
          + (f" (on {e['n_pairs']} pairs)" if e.get('n_pairs') is not None else '') + '.', '']

    L += render_rule(sel, 'CRT-AL on all common pairs (§4.4)', levels=True)
    L += render_rule(res['check_950'], 'Pre-specified check: without the screen (950-pair rule)', levels=False)

    L += ['## Sensitivity of the choice (§4.4)', '']
    if sel.get('sensitivity'):
        L += ['| rule in step 2 (capped WER) | T | choice |', '|---|---|---|']
        L += [f"| one standard error (the rule) | {', '.join(sel['T'])} | {sel['choice']} |"]
        L += [f"| {k} | {', '.join(v['T'])} | {v['choice']} |" for k, v in sel['sensitivity'].items()]
    else:
        L.append(f"not applicable: no candidate is eligible ({sel['fallback'] or 'not computable'}), so step 2 is not "
                 f"used")
    su = res['selection_uncapped']
    how = (f"fallback {su['fallback']}" if su['fallback'] else
           f"eligible {su['eligible']}, C_m = {su['c_m']}, T = {su['T']}" if su['choice'] else '; '.join(su['trace']))
    L += ['', f"Uncapped: with uncapped corpus WER in E4 and steps 1–2, the §4.4 rule picks {su['choice'] or 'none'} "
              f"({how}); " + ('the same as with capped WER' if su['choice'] == sel['choice'] else
                              f"capped WER picks {sel['choice'] or 'none'}, and CRT-AL stays as chosen")
          + ' (§0.3 row 20; reported, never used).']
    L.append('')

    ck = res['c1_check']
    L += ['## C1 identity check', '']
    if ck['compared']:
        L.append(f"`{ck['systems'][0]}` vs `{ck['systems'][1]}` on {ck['n']} common pairs: identical Whisper "
                 f"hypotheses {ck['same_hyp']}/{ck['n']}, max |ΔSIM| {fm(ck['max_dsim'], 4)}, max |ΔUTMOS| "
                 f"{fm(ck['max_dutmos'], 4)}, max |ΔdB-MCD| {fm(ck['max_dmcd'], 4)}, capped WER {fm(ck['wer'][0])} vs "
                 f"{fm(ck['wer'][1])} % (uncapped {fm(ck['wer_u'][0])} vs {fm(ck['wer_u'][1])} %); @src pairs "
                 f"{ck['n_src']}, max |ΔSIM_src| {fm(ck['max_dsim_src'], 4)}. "
                 f"Within the §8.1 item 9 tolerance: {yn(ck['within_tolerance'])}; identical: {yn(ck['identical'])}.")
    else:
        L.append(f"not compared: no rows for {ck['missing']}")
    L.append('')

    L += ['## By gender pair (§4.4; point estimates)', '',
          '| system | pair | n | capped WER % (n primary) | uncapped WER % | SIM | UTMOS | SIM_src | mean D | D > 0 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for x, by in res['gender'].items():
        for g, v in by.items():
            L.append(f"| {x} | {g} | {v['n']} | {fm(v['wer'])} ({v['n_wer_primary']}) | {fm(v['wer_u'])} | "
                     f"{fm(v['sim'], 3)} | {fm(v['utmos'])} | {fm(v['sim_src'], 3)} | {fm(v['m9'], 3, True)} | "
                     f"{fm(v['share_d_pos'], 2)} |")
    L.append('')

    if res['controls']:
        L += ['## Controls (descriptive)', '',
              '| system | n | capped WER % | uncapped WER % | SIM | UTMOS | dB-MCD | SIM_src |',
              '|---|---|---|---|---|---|---|---|']
        L += [f"| `{k}` | {v['n']} | {fm(v['wer'])} | {fm(v['wer_u'])} | {fm(v['sim'], 3)} | {fm(v['utmos'])} | "
              f"{fm(v['mcd'])} | {fm(v['sim_src'], 3)} |" for k, v in res['controls'].items()]
        L.append('')

    L += render_size(res['size'])
    L += ['## Baselines that will run (§2.4)', '']
    if res['baselines'] is None:
        L.append('not given (--baselines)')
    else:
        L += ['| ID | system | repository commit | checkpoint sha256 | sample rate | command |',
              '|---|---|---|---|---|---|']
        L += [f"| {b.get('id')} | {b.get('system')} | {b.get('repo_commit')} | {b.get('checkpoint_sha256')} | "
              f"{b.get('sample_rate')} | `{b.get('command')}` |" for b in res['baselines']]
    L.append('')

    ls = res['listening']
    L += ['## Listening-test items (§5.4)', '']
    if ls.get('items'):
        L += [f"{ls['n_items']} items from VCTK-2k ({ls['vctk2k_variant']}-target list) and LTC-2k, {PER_CELL} per "
              f"gender combination and set, seed {stats.SEED}. File `{ls['path']}`, sha256 {ls['sha256']}; it also "
              f"holds each cell's full seeded order (the replacement order).", '', '| set | pair | rank | pair_id |',
              '|---|---|---|---|']
        L += [f"| {i['set']} | {i['genders']} | {i['rank']} | `{i['pair_id']}` |" for i in ls['items']]
    L += [f'- {n}' for n in ls.get('notes', [])]
    L.append('')
    return '\n'.join(L) + '\n'


def render_rule(r, title, levels):
    L = [f'## {title}', '']
    if not r.get('q'):
        return L + [f"{'; '.join(r['trace'])} ({r['n_pairs']} common pairs"
                    + (' outside the screen)' if 'screen' in title else ')'), '']
    L.append(f"{r['n_pairs']} pairs ({r['n_wer_primary']} wer_primary); S2 = `{r['s2']}`. Candidates without rows: "
             f"{r['missing'] or 'none'}. E4 and steps 1–2 use the {'capped' if r['wer'] == 'wer' else 'uncapped'} WER; "
             f"capped (M3c) and uncapped corpus WER are both printed (§0.3 row 20).")
    if levels and r.get('levels'):
        L += ['', '| system | capped WER % | uncapped WER % | capped CER % | uncapped CER % | SIM | SECS | UTMOS | '
                  'SRC same | SRC cross | D same (share > 0) | D cross (share > 0) | dB-MCD |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for x, lv in r['levels'].items():
            name = f"{x} `{SYSTEMS[x]}`" if x in SYSTEMS else f'`{x}`'
            sp = r['share_pos'][x]
            L.append(f"| {name} | " + ' | '.join(
                fci(lv[k], PREC[k], k.startswith('m9')) + (f" ({fm(sp[k[3:]][0], 2)})" if k.startswith('m9') else '')
                for k in ('wer', 'wer_u', 'cer', 'cer_u', 'sim', 'secs', 'utmos', 'src_same', 'src_cross', 'm9_same',
                          'm9_cross', 'mcd'))
                + ' |')
        red = collections.defaultdict(set)
        for lv in r['levels'].values():
            for k, v in lv.items():
                if v and v.get('est') is not None:
                    red[k].add(v['redraws'])
        und = [f"{x} {k} {v['undefined']}" for x, lv in r['levels'].items() for k, v in lv.items()
               if v and v.get('undefined')]
        L += ['', 'Bootstrap redraws per quantity (replicates without a pair of positive weight or reference words; '
                  'the same for every system on the same pairs): '
              + ', '.join(f"{k} {'/'.join(map(str, sorted(v)))}" for k, v in red.items()) + '. '
              + f"Values missing or not finite in rows without error (left out): {', '.join(und) or 'none'}."]
    L += ['', '| candidate | SIM (E1 ≥ 0.80) | UTMOS (E2 ≥ FAISS) | E3 same / cross (≤ FAISS + 0.01) | '
              'WER − S2, capped (E4) | WER − A5, capped (E4) | WER − S2, uncapped | WER − A5, uncapped (E4 with '
              'uncapped WER) | eligible |', '|---|---|---|---|---|---|---|---|---|']
    s1 = r['ref'].get(SYSTEMS['S1'], {})
    for c in r['candidates']:
        q, f, e4 = r['q'][c], r['flags'][c], r['e4'][c]
        e3 = ' / '.join(f"{fm(q[f'src_{g}'], 3)} vs "
                        f"{fm(None if s1.get(f'src_{g}') is None else s1[f'src_{g}'] + MARGIN, 3)}"
                        for g in stats.STRATA)
        L.append(f"| {c} | {fm(q['sim'], 3)} {yn(f['E1'])} | {fm(q['utmos'])} vs {fm(s1.get('utmos'))} {yn(f['E2'])} | "
                 f"{e3} {yn(f['E3'])} | {fci(e4['wer']['S2'], 2, True)} | {fci(e4['wer']['A5'], 2, True)} "
                 f"{yn(e4_met(e4['wer']))} | {fci(e4['wer_u']['S2'], 2, True)} | {fci(e4['wer_u']['A5'], 2, True)} "
                 f"{yn(e4_met(e4['wer_u']))} | {yn(c in r['eligible'])} |")
    if r.get('s_k'):
        dm = r['diff_vs_m']
        L += ['', '| candidate | WER(Ck) − WER(C_m), capped | s_k (its SE) | WER(Ck) − WER(C_m), uncapped | its SE | '
                  'in T |', '|---|---|---|---|---|---|']
        L += [f"| {c} | " + ('0 (C_m) | 0.00 | 0 (C_m) | 0.00' if c == r['c_m'] else
                             f"{fci(dm['wer'][c], 2, True)} | {fm(r['s_k'][c])} | {fci(dm['wer_u'][c], 2, True)} | "
                             f"{fm(dm['wer_u'][c].get('se'))}") + f" | {yn(c in r['T'])} |" for c in r['eligible']]
    L += ['', 'Trace:', ''] + [f'- {t}' for t in r['trace']]
    L += ['', f"**Choice: {r['choice'] or 'none'}**" + (f" (fallback {r['fallback']})" if r['fallback'] else ''), '']
    return L


def render_size(sz):
    L = ['## Size check (§5.3)', '']
    if sz.get('note'):
        L += [sz['note'], '']
    L += ['Variance components (REML, MixedLM, one group; the optimiser with the highest REML log-likelihood is kept) '
          'of the Dev per-pair differences; WER linearised as 100 (E_A − E_B) / N̄ with the capped error counts '
          'E = min(S + D + I, N) (M3c) for P and K, and with v0\'s uncapped counts for V (F3 only), the corpus-WER '
          'difference of the other definition printed beside its Dev mean.', '',
          '| endpoint | A − B | Dev n | Dev mean (WER: as fitted) | Dev mean, other WER definition | σa² | σb² | '
          'σc² | σe² | converged, optimiser |', '|---|---|---|---|---|---|---|---|---|---|']
    for c in sz['conds']:
        v = c.get('vc')
        name = c['vctk2k'] + (f" / {c['ltc2k']}" if c['ltc2k'] else '')
        if v is None:
            L.append(f"| {name} | `{c['a']}` − `{c['b']}` | – | – | – | {c.get('error')} | | | | |")
        else:
            du = c.get('dev_mean_uncapped', c.get('dev_mean_capped'))
            fit = '' if 'wer_capped' not in c else ' (capped)' if c['wer_capped'] else ' (uncapped)'
            oth = '' if 'wer_capped' not in c else ' (uncapped)' if c['wer_capped'] else ' (capped)'
            L.append(f"| {name} | `{c['a']}` − `{c['b']}` | {c['dev_n']} | {c['dev_mean']:+.4g}{fit} | "
                     f"{'–' if du is None else f'{du:+.4g}{oth}'} | "
                     f"{v['sigma2_a']:.3g} | {v['sigma2_b']:.3g} | {v['sigma2_c']:.3g} | {v['sigma2_e']:.3g} | "
                     f"{yn(v['converged'])}, {v['method']}"
                     + (f" ({'; '.join(v['warnings'])})" if v['warnings'] else '')
                     + (f"; failed: {'; '.join(f'{m} {e}' for m, e in v['fit_errors'].items())}"
                        if v.get('fit_errors') else '') + ' |')
    L += ['', 'Predicted for the test lists (SE² = σa²/n_s + σb²/n_t + σc²/n_c + σe²/n; MDE = 3.53 SE; gate L: '
          '2.80 SE_g ≤ 0.01 − Δ_dev,g). Approximate: cell imbalance is ignored and the components are treated as '
          'known.', '',
          '| endpoint | design | n | n_s | n_t | n_c | SE | MDE or 2.80 SE | threshold | pass |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in sz['rows']:
        gate = r['kind'].startswith('gate_')
        lhs = r.get('lhs') if gate else r.get('mde')
        thr = (f"0.01 − ({fm(r.get('delta_dev'), 4, True)}) = {fm(r.get('rhs'), 4)}"
               if gate and r.get('rhs') is not None else fm(r.get('threshold'), 3))
        L.append(f"| {r['endpoint']} | {r['design']} | {r['n']} | {r['n_s']} | {r['n_t']} | {r['n_c']} | "
                 f"{fm(r['se'], 4)} | {fm(lhs, 4)} | {thr} | {yn(r['pass'])} |"
                 + (f" {r['note']}" if r.get('note') else '')
                 + (f" ({r['n_left_out_no_source_E']} pairs without a source E left out)"
                    if r.get('n_left_out_no_source_E') else ''))
    rb = sz['rebuild']
    L += ['', 'Rebuild rule: ' + (f"conditions failing with 5 targets and passing with 10: {rb['triggers'] or 'none'} "
                                     f"-> T1 uses the {rb['use']}-target VCTK-2k list."
                                     if rb['evaluable'] else 'not evaluable (no 10-target list given).')
          + f" Underpowered on {rb['underpowered_design']}: {rb['underpowered'] or 'none'}.",
          'wer_primary pairs (summary.json / counted in the list): ' + ', '.join(
              f"VCTK-2k {k[7:]} targets {v['summary'] if v['summary'] is not None else '–'} / "
              f"{v['counted'] if v['counted'] is not None else '–'}" for k, v in sz['wer_primary'].items()) + '.', '']
    return L


# ── main ─────────────────────────────────────────────────────

def main():
    a = parse_args()
    ren = dict(x.split('=', 1) for x in a.rename.split(',') if x)
    pairs = read_jsonl(a.pairs)
    if len({p['pair_id'] for p in pairs}) != len(pairs) or {p['set'] for p in pairs} != {'dev'}:
        raise SystemExit(f'ERROR: {a.pairs} is not a Dev pair list with unique pair_ids')
    raw, files = load_metrics([x for x in a.metrics.split(',') if x], ren)
    info = {p['pair_id']: p for p in pairs}
    R, ignored = collections.defaultdict(dict), collections.Counter()
    for (pid, s), r in raw.items():
        if r.get('set') != 'dev' or str(r.get('budget')) != '30':
            ignored['other set or budget'] += 1
        elif pid not in info:
            ignored['pair_id not in --pairs'] += 1
        else:
            R[s][pid] = r
    dev = Dev(dict(R), pairs)
    screen = {p['pair_id'] for k, p in enumerate(pairs) if k % SCREEN_MOD == 0}
    req = [SYSTEMS['S1'], *SYSTEMS['S2'].values(), SYSTEMS['A5'], *(SYSTEMS[c] for c in CANDS)]
    req += [SYSTEMS['S1'] + SRC, *(SYSTEMS[c] + SRC for c in CANDS)]
    present = [s for s in req if s in R]
    missing = [s for s in req if s not in R]
    order = [p['pair_id'] for p in pairs]
    with_rows = [k for k in order if any(k in R[s] for s in present)]
    common = [k for k in with_rows if all(not stats.failed(R[s].get(k)) for s in present)]
    excluded = {s: sum(stats.failed(R[s].get(k)) for k in with_rows) for s in present}

    tau = tau_rule(dev, common)
    s2 = tau['system']
    s1u = s1u_rule(dev, common)
    e17 = e17_rule(dev, common, s2)
    sel = crt_al_rule(dev, common, s2, levels=True)
    sel_u = crt_al_rule(dev, common, s2, wer='wer_u')
    chk = crt_al_rule(dev, [k for k in common if k not in screen], s2)

    lists = {}
    for key, path, summ, se in (('vctk2k_5', a.vctk_pairs, a.vctk_summary, a.vctk_source_e),
                                ('vctk2k_10', a.vctk10_pairs, a.vctk10_summary, a.vctk10_source_e)):
        lists[key] = {'pairs': path, 'rows': read_jsonl(path) if path else None,
                      'summary': next_to(path, summ, 'summary.json'),
                      'source_e': next_to(path, se, 'vctk2k_source_E.json')}
    ltc = read_jsonl(a.ltc_pairs) if a.ltc_pairs else None
    conds = size_conditions(dev, common, sel, s2)
    size = stats.size_check(conds, {'vctk2k_5': lists['vctk2k_5']['rows'], 'vctk2k_10': lists['vctk2k_10']['rows'],
                                    'ltc2k': ltc},
                            {k: source_e_speakers(v['source_e']) if v['source_e'] else None for k, v in lists.items()})
    size['conds'] = [{k: v for k, v in c.items()} for c in conds]
    size['wer_primary'] = {k: wer_primary_count(v['summary'], v['rows']) for k, v in lists.items()}
    if not sel.get('choice'):
        size['note'] = ('No CRT-AL (F3 or not computable): V = S2 − S1 only; K and L are dropped.'
                        if sel.get('fallback') == 'F3' else 'No CRT-AL could be computed: V = S2 − S1 only.')

    stem = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    variant = size['rebuild']['use'] or '5'
    vkey = 'vctk2k_10' if variant == '10' else 'vctk2k_5'
    lsets = {n: (rows, path) for n, rows, path in (('vctk2k', lists[vkey]['rows'], lists[vkey]['pairs']),
                                                   ('ltc2k', ltc, a.ltc_pairs)) if rows is not None}
    listening = listening_draw(lsets, variant, stem + '.listening_items.json') if lsets else {}
    listening['notes'] = [f'no {n} list given: no items from it' for n in ('vctk2k', 'ltc2k') if n not in lsets]
    if size['rebuild']['use'] is None and 'vctk2k' in lsets:
        listening['notes'].append('the size check could not choose the VCTK-2k variant: items drawn from the 5-target '
                                  'list given')

    jobs = dict(x.split('=', 1) for x in a.jobs.split(',') if x)
    sacct = [x for x in a.sacct.split(',') if x]
    job_rows = read_sacct(sacct, jobs)
    c1 = c1_check(dev, common)
    repro = read_repro([x for x in a.repro.split(',') if x])
    baselines = json.load(open(a.baselines)) if a.baselines else None
    evidence = [*sacct, *(x['path'] for x in repro), *([a.baselines] if a.baselines else []),
                *(v[k] for v in lists.values() for k in ('pairs', 'summary', 'source_e') if v[k]),
                *([a.ltc_pairs] if a.ltc_pairs else [])]

    problems = []
    if missing:
        problems.append(f'{len(missing)} required systems have no rows (left out of the common set): {missing}')
    if len(pairs) != 1000 or len(with_rows) != len(pairs):
        problems.append(f'rows cover {len(with_rows)} of the {len(pairs)} --pairs pairs (full Dev: 1,000)')
    for k, v in (('S1u', s1u['missing']), ('E17', [] if dev.has(SYSTEMS['E17']) else [SYSTEMS['E17']])):
        if v:
            problems.append(f'{k} systems without rows: {v}')
    if not common:
        problems.append('the common set (§4.1) is empty')
    for what, v, how in (('S2 (§4.2)', tau['tau'], '; '.join(tau['notes'])),
                         ('S1u (§4.3)', s1u['setting'], s1u['how']), ('E17 (§4.3)', e17['enters'], e17.get('how'))):
        if v is None:
            problems.append(f'{what}: {how}')
    for what, r in (('CRT-AL (§4.4)', sel), ('950-pair check', chk)):
        na = [f'{c} {k}' for c, f in r.get('flags', {}).items() for k, v in f.items() if v is None]
        if r['choice'] is None and r['fallback'] is None:
            problems.append(f'{what}: ' + '; '.join(r['trace']))
        elif na:
            problems.append(f'{what}: not computable, counted as not met: {", ".join(na)}')
    if c1['compared'] and c1['within_tolerance'] is False:
        problems.append(f'C1 identity check: `{SYSTEMS["C1"]}` differs from `{SYSTEMS["C1_anchor"]}` beyond the §8.1 '
                        f'item 9 tolerance')
    if any(c['vc'] is None for c in conds):
        problems.append('size check (§5.3): no variance components for '
                        + ', '.join(f"{c['vctk2k']} ({c.get('error')})" for c in conds if c['vc'] is None))
    if not repro:
        problems.append('reproduction check not given (--repro)')
    roles = ('2634', 'D0', 'D1', 'D2')
    if not sacct or not all(r in jobs for r in roles):
        problems.append('job IDs / sacct times of 2634, D0, D1, D2 not all given (--jobs, --sacct)')
    elif any(job_rows[r]['submit'] is None for r in roles):
        problems.append(f"job IDs not in the --sacct files: {[jobs[r] for r in roles if not job_rows[r]['submit']]}")
    if not size['rebuild']['evaluable']:
        problems.append('10-target VCTK-2k list not given: the rebuild rule cannot be applied')
    for k, v in lists.items():
        if v['rows'] is not None and not v['source_e']:
            problems.append(f'{k}: no source-E file: gate L counts assume every source speaker has an E set')
    if lists['vctk2k_5']['rows'] is None or ltc is None:
        problems.append('test pair lists not all given (--vctk_pairs, --ltc_pairs)')
    if baselines is None:
        problems.append('baselines not given (--baselines)')
    elif not all(b.get(k) for b in baselines for k in BASELINE_KEYS):
        problems.append(f'--baselines: an entry lacks one of {list(BASELINE_KEYS)} (§2.4, §4.6)')
    stopped = any(r['passed'] is not True for r in repro)
    status = ('STOPPED: the reproduction check did not pass (§4.4); this output is reported but is not the selection'
              if stopped else 'INCOMPLETE: not a valid selection record' if problems else 'complete')

    here = os.path.dirname(os.path.abspath(__file__))
    res = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': status, 'problems': problems,
           'scripts': {f: sha256(os.path.join(here, f)) for f in ('select_crt_al.py', 'stats.py')},
           'systems': SYSTEMS, 'rename': ren,
           'inputs': {'pairs': {'path': os.path.abspath(a.pairs), 'sha256': sha256(a.pairs), 'n': len(pairs)},
                      'files': files, 'evidence': [{'path': os.path.abspath(x), 'sha256': sha256(x)} for x in evidence],
                      'ignored': dict(ignored), 'missing': missing, 'screen_n': len(screen),
                      'common': {'n': len(common), 'n_wer_primary': len(dev.subset(common, 'wer')),
                                 'n_parallel': len(dev.subset(common, 'par')),
                                 'n_same': len(dev.subset(common, 'same')), 'n_cross': len(dev.subset(common, 'cross')),
                                 'n_no_rows': len(pairs) - len(with_rows), 'excluded': excluded,
                                 'n_screen': len(screen & set(common))}},
           'repro': repro, 'jobs': job_rows, 'baselines': baselines,
           'tau_rule': tau, 's1u': s1u, 'e17': e17, 'selection': sel, 'selection_uncapped': sel_u, 'check_950': chk,
           'c1_check': c1, 'gender': gender_table(dev, common),
           'controls': controls_table(dev, common), 'size': size, 'listening': listening}
    md = render(res)
    open(a.out, 'w').write(md)
    with open(stem + '.json', 'w') as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else sorted(o)
                  if isinstance(o, set) else str(o))
    print(md)
    print(f'-> {a.out}, {stem}.json', flush=True)
    if stopped:
        sys.exit(1)


if __name__ == '__main__':
    main()
