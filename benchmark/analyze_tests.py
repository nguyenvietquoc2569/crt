#!/usr/bin/env python3
"""
The test-set statistics of ANALYSIS_PLAN_v1.md §5 and the outcome of §7, computed from job M's metrics (RUNBOOK_v1.md
§7) and run once (§8.3). CPU only; every interval and p-value comes from stats.py (crossed bootstrap, B = 10,000, seed
20260928; VCTK-2k independent, LTC-2k linked).

Systems, from the committed selection record (--record): CRT-AL (C2-C5: cand_<code>, C1: anc_R3_s0_pm_rq_T1), S1
FAISS (faiss), S3 CRT τ = 0.4 (crt_tau0.4), S2 CRT(τ_dev) (crt_tau<τ_dev>; = S3 when τ_dev = 0.4) and the baselines
the record lists (knnvc, freevc, seedvc, facodec). Under fallback F3 (no CRT-AL) the script refuses: the record of
19 Oct 2026 selected C2.

Families (§5.1); a hypothesis is Δ = A - B over the pairs of its endpoint where neither system failed:
  P  CRT-AL - FAISS: P1 capped WER VCTK-2k (wer_primary pairs), P2 capped WER LTC-2k, P3 / P4 SIM, P5 / P6 UTMOS,
     P7 dB-MCD (VCTK-2k pairs with P); P8 the CMOS CRT-AL vs FAISS (--cmos). Holm over 8, or over 7 when P8 is dropped
     (fewer than 15 screened raters). P8 is pending when --cmos is not given or lacks it (the listening test follows
     this run, §8.3); every decision is then given twice: Holm over 7 ('if P8 is dropped') and Holm over 8 with P8
     counted and not rejected ('whatever P8 shows', the strictest case for P1-P7), and so is the outcome.
  L  gate L: SIM_src(CRT-AL) - SIM_src(FAISS) per stratum (same, cross gender) on both sets, H0 Δ >= +0.01 rejected
     when the upper limit of the 95 % CI is below +0.01; claimed only if all four reject. VCTK-2k: pairs whose source
     has no E set (--source_e) are left out and counted. Beside it: the M9 difference (mean D(CRT-AL) - mean D(FAISS),
     D = SIM_src - SIM_tgt) with its CI, the share of pairs with D > 0 of both systems, and on VCTK-2k the natural
     other-speaker level (mean SIM_src of target_copy on the stratum's pairs with P). A VCTK-2k pass where CRT-AL's
     mean SIM_src is below that level, over the stratum's gate pairs or over the same pairs with P, is 'not
     interpretable as absence of leakage'.
  K  CRT-AL - CRT τ = 0.4 on the endpoints of P1-P7, K8 the CMOS CRT-AL vs CRT τ = 0.4; Holm as P.
  V  CRT(τ_dev) - FAISS on the endpoints of P1-P7 with v0's uncapped corpus WER; Holm over 7.
  S  CRT-AL - each listed baseline on capped WER, SIM and UTMOS on both sets; Holm over all of them (6 per baseline).
Sensitivities, each with the decisions (reject and the system favoured) that change: P ∪ K (Holm over 16, 14 without
CMOS); the other WER definition (uncapped for P1, P2, K1, K2 and S's WER, capped for V1, V2); LTC-2k with v0's
independent bootstrap (families and L2); failed conversions at their worst case (stats.compare, when the two systems'
failure counts differ), for the families and for gate L (highest SIM_src).
Outcome (§7): A, B or C from P's significant effects (lower WER and dB-MCD, higher SIM, UTMOS and CMOS favour that
system) and 'leakage not excluded' when gate L does not reject all four (framing at most B).
Descriptive (CIs only): every system's levels per set (capped and uncapped WER of Whisper-large-v3 and of Whisper-base,
capped CER, SIM, SECS, UTMOS, dB-MCD and the F0 metrics on the pairs with P, SIM_src, SECS_src, M9 with the share
D > 0), failure counts, EER (metrics.sv_trials, WavLM-SV), coverage (M7 override and change coverage of the gated
systems, the decode-lab fields), the gender-pair table of FAISS, S2, S3 and CRT-AL, and the budget sweep at 3, 30 and
180 s (FAISS, S2, S3, CRT-AL, kNN-VC; LTC-2k at 180 s without the shortfall targets). Wilcoxon signed-rank and the
rank-biserial r of every hypothesis are printed as possibly anti-conservative (§5.2).

Inputs (all required): --m_dir with t1..t4/metrics/<set>_<budget>{,_src}.jsonl and t1..t4/m.done as job M writes them
(every part done under the same selection.env), the two pair lists (VCTK-2k: the record's variant), the VCTK-2k
source-E file, the record (status complete). Every system of the families has one row per pair at 30 s. Writes --out
(.md) and <out stem>.json with every number and the sha256 of every input.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

import stats

SRC = '@src'
SETS = ('vctk2k', 'ltc2k')
MODE = {'vctk2k': stats.INDEPENDENT, 'ltc2k': stats.LINKED}
CAND_ROW = {'C1': 'anc_R3_s0_pm_rq_T1', 'C2': 'cand_pm18_rq', 'C3': 'cand_gv1_pm_cont_T1',
            'C4': 'cand_gv1_shrink0.85_cont', 'C5': 'cand_gv0.5_shrink0.75_cont'}
BASELINE_ROW = {'S5': 'knnvc', 'S6': 'freevc', 'S7': 'seedvc', 'S8': 'facodec'}
# quantity: metric of stats.py, pair subset, +1 when higher is better, worst case of a failed conversion
Q = {'wer': ('wer_capped', 'wer', -1, None), 'wer_u': ('wer', 'wer', -1, None),
     'wer_b': ('wer_capped', 'wer', -1, None), 'wer_bu': ('wer', 'wer', -1, None),
     'cer': ('cer_capped', 'wer', -1, None), 'sim': ('sim_wavlm', 'all', 1, 'min'),
     'secs': ('secs_resemblyzer', 'all', 1, 'min'), 'utmos': ('utmos', 'all', 1, 'min'),
     'mcd': ('mcd_db', 'par', -1, 'max'), 'lf0': ('lf0_rmse', 'par', -1, None), 'vuv': ('vuv_err', 'par', -1, None),
     'f0pcc': ('f0_pcc', 'par', 1, None), 'sim_src': ('sim_wavlm', 'all', -1, 'max'),
     'secs_src': ('secs_resemblyzer', 'all', -1, 'max'), 'm9': ('m9', 'all', -1, None)}
WER_KEY_OF = {'wer_b': 'wer_base', 'wer_bu': 'wer_base'}
SRC_QUANTITIES = ('sim_src', 'secs_src')
LABEL = {'wer': 'capped WER %', 'wer_u': 'WER %', 'wer_b': 'capped WER % (Whisper-base)',
         'wer_bu': 'WER % (Whisper-base)', 'cer': 'capped CER %', 'sim': 'SIM', 'secs': 'SECS', 'utmos': 'UTMOS',
         'mcd': 'dB-MCD', 'lf0': 'log-F0 RMSE', 'vuv': 'V/UV error', 'f0pcc': 'F0 PCC', 'sim_src': 'SIM_src',
         'secs_src': 'SECS_src', 'm9': 'M9 D'}
ENDPOINTS = [('vctk2k', 'wer'), ('ltc2k', 'wer'), ('vctk2k', 'sim'), ('ltc2k', 'sim'), ('vctk2k', 'utmos'),
             ('ltc2k', 'utmos'), ('vctk2k', 'mcd')]
S_ENDPOINTS = [('vctk2k', 'wer'), ('ltc2k', 'wer'), ('vctk2k', 'sim'), ('ltc2k', 'sim'), ('vctk2k', 'utmos'),
               ('ltc2k', 'utmos')]
PAR_ONLY = ('mcd', 'lf0', 'vuv', 'f0pcc')
MIN_RATERS = 15
COVERAGE = ('changed_cb1_18', 'changed_cb19_31', 'cb0_changed', 'latent_dist_to_faiss', 'energy_rel', 'orphan_share')
COVERAGE_DICTS = ('override_coverage', 'change_coverage')
NOTES = ['Every number comes from H200 MIG 2g.35gb slices with Mimi in float16 (§3.2).',
         'Seed-VC keeps only the first 25 s of the reference, so at 30 s and 180 s it sees the first 25 s of R (§2.4).',
         'FreeVC was trained on VCTK ("VCTK-seen"); its VCTK-2k rows carry that flag (§2.4, §5.1).',
         'Wilcoxon signed-rank p and the rank-biserial r are possibly anti-conservative, since pairs share speakers '
         '(§5.2).']
VARIANT_DROPPED, VARIANT_STRICT = 'P8 dropped (Holm over 7)', 'P8 counted, not rejected (Holm over 8)'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ── data ─────────────────────────────────────────────────────

class TestSet:
    """One test list: its pairs and the metrics rows {budget: {system: {pair_id: row}}}."""

    def __init__(self, name, pairs, rows, src_e=None):
        self.name, self.mode, self.src_e = name, MODE[name], src_e
        self.pairs = pairs
        self.info = {p['pair_id']: p for p in pairs}
        self.spk = {k: (p['source_speaker'], p['target_speaker']) for k, p in self.info.items()}
        self.R = rows
        self.n_ref = {}
        for systems in rows.values():
            for rs in systems.values():
                for k, r in rs.items():
                    x = r.get(stats.WER_KEY)
                    if k not in self.n_ref and isinstance(x, dict) and 'N' in x:
                        self.n_ref[k] = x['N']

    def rows(self, system, budget=30):
        return self.R.get(budget, {}).get(system, {})

    def has(self, system, budget=30):
        return bool(self.rows(system, budget))

    def subset(self, which, pids=None):
        keep = {'all': lambda p: True, 'wer': lambda p: p['wer_primary'], 'par': lambda p: bool(p.get('P'))}
        f = keep[which] if which in keep else (lambda p: p['genders'] == which)
        return [p['pair_id'] for p in self.pairs if f(p) and (pids is None or p['pair_id'] in pids)]

    def m9_rows(self, system, budget=30):
        return stats.margin_rows(self.rows(system, budget), self.rows(system + SRC, budget))

    def quantity_rows(self, system, q, budget=30):
        if q == 'm9':
            return self.m9_rows(system, budget)
        return self.rows(system + SRC if q in SRC_QUANTITIES else system, budget)

    def level(self, system, q, budget=30, pids=None):
        """stats.level of a system's quantity over the quantity's pairs (within pids when given)."""
        sub = self.subset(Q[q][1], pids)
        return stats.level(self.quantity_rows(system, q, budget), sub, self.spk, Q[q][0], mode=self.mode,
                           wer_key=WER_KEY_OF.get(q, stats.WER_KEY))


def load_set(m_dir, name, files, pairs):
    """{budget: {system: {pair_id: row}}} from (part, budget) files of job M; a (pair_id, system) in two files, a row of
    another set or budget, or a pair_id outside the list is an error."""
    R, inputs, ids = {}, [], {p['pair_id'] for p in pairs}
    for part, budget in files:
        for suffix in ('', '_src'):
            path = os.path.join(m_dir, part, 'metrics', f'{name}_{budget}{suffix}.jsonl')
            if not os.path.exists(path):
                raise SystemExit(f'ERROR: {path} missing: the statistics run once, on every output of job M')
            got = stats.load_results(path)
            inputs.append({'path': os.path.abspath(path), 'sha256': sha256(path), 'rows': len(got)})
            for r in got:
                if r.get('set') not in (None, name) or int(r.get('budget', budget)) != budget:
                    raise SystemExit(f'ERROR: {path}: row {r["pair_id"]} {r["system"]} has set {r.get("set")}, '
                                     f'budget {r.get("budget")}')
                if r['pair_id'] not in ids:
                    raise SystemExit(f'ERROR: {path}: pair {r["pair_id"]} is not in the {name} list')
                bucket = R.setdefault(budget, {}).setdefault(r['system'], {})
                if r['pair_id'] in bucket:
                    raise SystemExit(f'ERROR: ({r["pair_id"]}, {r["system"]}) at {budget} s occurs in two files')
                bucket[r['pair_id']] = r
    return R, inputs


def check_done(m_dir, parts):
    """Every part of job M finished (m.done, written only when every step passed its row checks), all under the same
    selection.env. Returns {part: the m.done line}."""
    out, shas = {}, {}
    for part in parts:
        path = os.path.join(m_dir, part, 'm.done')
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {path} missing: part {part} of job M has not finished')
        out[part] = open(path).read().strip()
        m = re.search(r'selection sha256 ([0-9a-f]{64})', out[part])
        if not m:
            raise SystemExit(f'ERROR: {path} names no selection sha256: {out[part]!r}')
        shas[part] = m.group(1)
    if len(set(shas.values())) > 1:
        raise SystemExit(f'ERROR: the parts of job M ran under different selection.env files: {shas}')
    return out


def read_pairs(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def systems_from_record(rec):
    if rec.get('status') != 'complete':
        raise SystemExit(f'ERROR: the selection record has status {rec.get("status")!r}, not complete')
    choice = rec['selection'].get('choice')
    if choice is None:
        raise SystemExit('ERROR: the record ends in fallback F3; this script implements the CRT-AL families only')
    tau = rec['tau_rule']['tau']
    return {'crt_al': CAND_ROW[choice], 'choice': choice, 'faiss': 'faiss', 's3': 'crt_tau0.4',
            's2': f'crt_tau{tau:g}', 'tau_dev': tau,
            'baselines': [BASELINE_ROW[b['id']] for b in rec.get('baselines') or []],
            'vctk_targets': str(rec['size']['rebuild'].get('use') or '5')}


# ── hypotheses ───────────────────────────────────────────────

def favours(res, q, names):
    """Which of names (A, B) a difference favours; None without an estimate or at 0."""
    if res.get('est') is None or res['est'] == 0:
        return None
    return names[0] if (res['est'] > 0) == (Q[q][2] > 0) else names[1]


def hypothesis(sets, hid, s, q, a, b, wer, mode=None):
    ts = sets[s]
    metric = wer if q == 'wer' else Q[q][0]
    pids = ts.subset(Q[q][1])
    res = stats.compare(ts.quantity_rows(a, q), ts.quantity_rows(b, q), pids, ts.spk, metric,
                        mode=mode or ts.mode, worst=Q[q][3], n_ref=ts.n_ref)
    label = LABEL[{'wer_capped': 'wer', 'wer': 'wer_u'}.get(metric, q)]
    return {'id': hid, 'set': s, 'quantity': q, 'label': label, 'sys_a': a, 'sys_b': b, **res}   # res: a, b values


def apply_holm(hyps, extra=()):
    """Holm over the hypotheses plus `extra` (CMOS rows); returns the reject flags of hyps and extra, in order."""
    flags = stats.holm([h.get('p') for h in hyps] + [x.get('p') for x in extra])
    return flags[:len(hyps)], flags[len(hyps):]


def cmos_entry(cmos, key):
    """A CMOS block (P8 or K8): pending when --cmos is not given or lacks the block (the listening test has not
    reported it); in the family when at least MIN_RATERS screened raters completed it; dropped otherwise (§5.4)."""
    if cmos is None or key not in cmos:
        return {'id': key, 'status': 'pending', 'p': None}
    x = dict(cmos[key] or {})
    n = x.get('n_raters') or 0
    ok = n >= MIN_RATERS and x.get('p') is not None
    return dict(x, id=key, status='in family' if ok else 'dropped', p=x.get('p') if ok else None)


def holm_flags(hyps, c=None):
    """The family's Holm with its CMOS block c (None: no CMOS in the family). Returns (flags, strict flags or None,
    the CMOS flag or None, label); strict flags exist when c is pending: Holm with c counted and never rejected."""
    n = len(hyps)
    if c is None:
        flags, _ = apply_holm(hyps)
        return flags, None, None, f'over {n}'
    if c['status'] == 'in family':
        flags, (cf,) = apply_holm(hyps, [c])
        return flags, None, cf, f'over {n + 1}'
    flags, _ = apply_holm(hyps)
    if c['status'] == 'dropped':
        return flags, None, None, f'over {n} ({c["id"]} dropped)'
    strict, _ = apply_holm(hyps, [c])
    return flags, strict, None, f'over {n} if {c["id"]} is dropped; over {n + 1} whatever {c["id"]} shows'


def set_flags(hyps, flags, strict=None):
    for i, h in enumerate(hyps):
        names = (h['sys_a'], h['sys_b'])
        h['reject'] = flags[i]
        h['favours'] = favours(h, h['quantity'], names) if flags[i] else None
        if strict is not None:
            h['reject_strict'] = strict[i]
            h['favours_strict'] = favours(h, h['quantity'], names) if strict[i] else None


def family(sets, fam, a, b, eps, wer, cmos=None, cmos_key=None, mode_ltc=None):
    hyps = [hypothesis(sets, f'{fam}{i}', s, q, a, b, wer, mode=mode_ltc if s == 'ltc2k' else None)
            for i, (s, q) in enumerate(eps, 1)]
    c = cmos_entry(cmos, cmos_key) if cmos_key else None
    flags, strict, cf, label = holm_flags(hyps, c)
    set_flags(hyps, flags, strict)
    out = {'family': fam, 'a': a, 'b': b, 'hypotheses': hyps, 'holm': label}
    if c is not None:
        c['reject'] = cf
        c['favours'] = (a if c.get('est', 0) > 0 else b) if cf else None
        out['cmos'] = c
    return out


def s_family(sets, systems, wer, mode_ltc=None):
    hyps = []
    for bl in systems['baselines']:
        for i, (s, q) in enumerate(S_ENDPOINTS, 1):
            h = hypothesis(sets, f'S-{bl}-{i}', s, q, systems['crt_al'], bl, wer,
                           mode=mode_ltc if s == 'ltc2k' else None)
            if bl == 'freevc' and s == 'vctk2k':
                h['flag'] = 'VCTK-seen'
            hyps.append(h)
    flags, _, _, label = holm_flags(hyps)
    set_flags(hyps, flags)
    return {'family': 'S', 'a': systems['crt_al'], 'b': 'baselines', 'hypotheses': hyps, 'holm': label}


def gate(sets, a, b, mode_ltc=None):
    """Gate L on both sets, its worst case and its descriptive companions (§5.1, §5.2)."""
    out = {}
    for s in SETS:
        ts = sets[s]
        mode = mode_ltc if (s == 'ltc2k' and mode_ltc) else ts.mode
        res = stats.gate_l_strata(ts.rows(a + SRC), ts.rows(b + SRC), ts.pairs, mode=mode,
                                  src_e=ts.src_e if s == 'vctk2k' else None)
        m9a, m9b = ts.m9_rows(a), ts.m9_rows(b)
        for g, r in res.items():
            pids = [p['pair_id'] for p in ts.pairs if stats.stratum(p['genders']) == g
                    and (s != 'vctk2k' or ts.src_e is None or p['source_speaker'] in ts.src_e)]
            if r.get('worst_case'):
                r['worst_case']['rejects'] = stats.gate_l(r['worst_case'])
            r['m9_diff'] = stats.compare(m9a, m9b, pids, ts.spk, 'm9', mode=mode)
            r['share_d_pos'] = {'crt_al': share_pos(m9a, pids), 'faiss': share_pos(m9b, pids)}
            r['natural_other_speaker'], r['interpretable'] = None, True
            if s == 'vctk2k':
                par = [k for k in pids if ts.info[k].get('P')]
                lvl, n = stats.point(ts.rows('target_copy' + SRC), par, 'sim_wavlm')
                same, n_same = stats.point(ts.rows(a + SRC), par, 'sim_wavlm')
                r['natural_other_speaker'] = {'sim_src_target_copy': lvl, 'n': n,
                                              'sim_src_crt_al_same_pairs': same, 'n_same_pairs': n_same}
                below = lvl is not None and any(x is not None and x < lvl for x in (r.get('a'), same))
                r['interpretable'] = not (r['rejects'] and below)
            out[f'{s}/{g}'] = r
    ids = {'vctk2k/same': 'L1s', 'vctk2k/cross': 'L1c', 'ltc2k/same': 'L2s', 'ltc2k/cross': 'L2c'}
    for k, r in out.items():
        r['id'] = ids[k]
    wc = {r['id']: (r['worst_case']['rejects'] if r.get('worst_case') else r['rejects']) for r in out.values()}
    return {'strata': out, 'all_reject': all(r['rejects'] is True for r in out.values()),
            'worst_case': {'used': [r['id'] for r in out.values() if r.get('worst_case')], 'rejects': wc,
                           'all_reject': all(v is True for v in wc.values())}}


def share_pos(rows, pids):
    v = [rows[k]['m9'] for k in pids if k in rows and not stats.failed(rows[k])]
    return {'share': float(np.mean(np.array(v) > 0)) if v else None, 'n': len(v)}


def decisions(fam, strict=False):
    """{hypothesis: (reject, system favoured)}; strict: the flags with a pending CMOS counted, where they exist."""
    d = {}
    for h in fam['hypotheses']:
        if strict and 'reject_strict' in h:
            d[h['id']] = (h['reject_strict'], h['favours_strict'])
        else:
            d[h['id']] = (h['reject'], h['favours'])
    c = fam.get('cmos') or {}
    if c.get('status') == 'in family':
        d[c['id']] = (c['reject'], c.get('favours'))
    return d


def pending(fam):
    return (fam.get('cmos') or {}).get('status') == 'pending'


def outcome(P, L, crt_al, faiss, strict=False):
    d = decisions(P, strict)
    fav_a = [k for k, (r, fv) in d.items() if r and fv == crt_al]
    fav_b = [k for k, (r, fv) in d.items() if r and fv == faiss]
    pattern = 'A' if fav_a and not fav_b else 'B' if fav_a and fav_b else 'C'
    framing = pattern if L['all_reject'] else ('B' if pattern == 'A' else pattern)
    return {'pattern': pattern, 'favour_crt_al': fav_a, 'favour_faiss': fav_b, 'gate_l_all_reject': L['all_reject'],
            'gate_l_failed': [r['id'] for r in L['strata'].values() if r['rejects'] is not True],
            'leakage_not_excluded': not L['all_reject'], 'framing': framing,
            'not_interpretable': [r['id'] for r in L['strata'].values() if not r['interpretable']]}


def outcomes(P, L, crt_al, faiss):
    """The §7 outcome; while P8 is pending, under both Holm variants (they can differ)."""
    if pending(P):
        return {'p8_pending': True, 'variants': {VARIANT_DROPPED: outcome(P, L, crt_al, faiss),
                                                 VARIANT_STRICT: outcome(P, L, crt_al, faiss, strict=True)}}
    return {'p8_pending': False, 'variants': {'final': outcome(P, L, crt_al, faiss)}}


def changed(primary, other):
    """Hypotheses whose decision (reject, system favoured) differs between two runs of a family; with a pending CMOS
    under both Holm variants."""
    out = {}
    for name, strict in (('holm', False),) + ((('holm_strict', True),) if pending(primary) else ()):
        a, b = decisions(primary, strict), decisions(other, strict)
        out[name] = sorted(k for k in a if a[k] != b.get(k))
    return out


# ── sensitivities ────────────────────────────────────────────

def _holm_decisions(hyps, extra):
    flags, eflags = apply_holm(hyps, extra)
    d = {h['id']: (r, favours(h, h['quantity'], (h['sys_a'], h['sys_b'])) if r else None)
         for h, r in zip(hyps, flags)}
    for x, r in zip(extra, eflags):
        if x['status'] == 'in family':
            d[x['id']] = (r, x.get('favours') if r else None)
    return d


def union_pk(P, K):
    hyps = P['hypotheses'] + K['hypotheses']
    inn = [x for x in (P.get('cmos'), K.get('cmos')) if x and x['status'] == 'in family']
    pend = [x for x in (P.get('cmos'), K.get('cmos')) if x and x['status'] == 'pending']
    out = {'m': len(hyps) + len(inn), 'decisions': _holm_decisions(hyps, inn)}
    base = {**decisions(P), **decisions(K)}
    out['changed'] = {'holm': sorted(k for k, v in out['decisions'].items() if v != base.get(k))}
    if pend:
        out['m_strict'] = len(hyps) + len(inn) + len(pend)
        out['decisions_strict'] = _holm_decisions(hyps, inn + pend)
        base_s = {**decisions(P, True), **decisions(K, True)}
        out['changed']['holm_strict'] = sorted(k for k, v in out['decisions_strict'].items() if v != base_s.get(k))
    return out


def worst_case_family(fam):
    """The family's Holm redone with each hypothesis's worst-case Δ and p where stats.compare computed one."""
    hyps = []
    for h in fam['hypotheses']:
        wc = h.get('worst_case') or {}
        hyps.append(dict(h, p=wc.get('p', h.get('p')), est=wc.get('est', h.get('est'))) if wc.get('p') is not None
                    else dict(h))
    c = fam.get('cmos')
    flags, strict, _, _ = holm_flags(hyps, c)
    set_flags(hyps, flags, strict)
    other = dict(fam, hypotheses=hyps)
    return {'worst_case_used': [h['id'] for h in fam['hypotheses'] if h.get('worst_case')],
            'changed': changed(fam, other)}


# ── descriptive ──────────────────────────────────────────────

ORDER = ['faiss', 's2', 'crt_tau0.4', 'crt_argmax', 'front_us32', 'front_lmbr', 'anc_', 'cand_', 'knnvc', 'freevc',
         'seedvc', 'facodec', 'source_copy', 'target_copy', 'mimi_copy_src', 'mimi_copy_P']
LEVEL_QUANTITIES = ('wer', 'wer_u', 'wer_b', 'wer_bu', 'cer', 'sim', 'secs', 'utmos', 'mcd', 'lf0', 'vuv', 'f0pcc',
                    'sim_src', 'secs_src', 'm9')


def system_order(names, systems):
    def key(n):
        for i, pre in enumerate(ORDER):
            if (pre == 's2' and n == systems['s2']) or (pre != 's2' and n.startswith(pre)):
                return (i, n)
        return (len(ORDER), n)
    return sorted(names, key=key)


def _finite_mean(v):
    v = [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    return float(np.mean(v)) if v else None


def coverage(rs):
    """Means of the coverage fields over a system's rows without error: the decode-lab and anchor-lab numbers, and the
    M7 override / change coverage dicts of run_conversions.py (overall and per codebook)."""
    ok = [r for r in rs.values() if not stats.failed(r)]
    out = {k: _finite_mean(r.get(k) for r in ok) for k in COVERAGE if any(k in r for r in ok)}
    for k in COVERAGE_DICTS:
        ds = [r[k] for r in ok if isinstance(r.get(k), dict)]
        if ds:
            per = sorted({c for d in ds for c in (d.get('per_cb') or {})}, key=lambda c: int(re.sub(r'\D', '', c) or 0))
            out[k] = {'overall': _finite_mean(d.get('overall') for d in ds),
                      'per_cb': {c: _finite_mean((d.get('per_cb') or {}).get(c) for d in ds) for c in per}}
    return {k: v for k, v in out.items() if v is not None}


def levels(ts, systems, budget=30, quantities=LEVEL_QUANTITIES):
    names = [n for n in ts.R.get(budget, {}) if not n.endswith(SRC)]
    out = {}
    for n in system_order(names, systems):
        row = {}
        for q in quantities:
            if q in PAR_ONLY and ts.name == 'ltc2k':
                continue
            if (q in SRC_QUANTITIES or q == 'm9') and not ts.has(n + SRC, budget):
                continue
            row[q] = ts.level(n, q, budget)
        rs = ts.rows(n, budget)
        row['fail'] = sum(stats.failed(r) for r in rs.values())
        row['shortfall'] = sum(bool(r.get('shortfall')) for r in rs.values())
        row['n_rows'] = len(rs)
        if ts.has(n + SRC, budget):
            row['share_d_pos'] = share_pos(ts.m9_rows(n, budget), ts.subset('all'))
        cov = coverage(rs)
        if cov:
            row['coverage'] = cov
        if n == 'freevc' and ts.name == 'vctk2k':
            row['flag'] = 'VCTK-seen'
        out[n] = row
    return out


def gender_table(ts, systems):
    out = {}
    for n in dict.fromkeys((systems['faiss'], systems['s2'], systems['s3'], systems['crt_al'])):
        if not ts.has(n):
            continue
        out[n] = {}
        for g in ('F2F', 'F2M', 'M2F', 'M2M'):
            pids = set(ts.subset(g))
            out[n][g] = {q: ts.level(n, q, pids=pids) for q in ('wer', 'sim', 'utmos', 'sim_src')
                         if q != 'sim_src' or ts.has(n + SRC)}
    return out


def eer_table(ts, budget=30):
    from metrics import eer, sv_trials
    out = {}
    for n, rs in ts.R.get(budget, {}).items():
        if n.endswith(SRC) or n in ('source_copy', 'target_copy', 'mimi_copy_src', 'mimi_copy_P'):
            continue
        ok = [r for r in rs.values() if not stats.failed(r) and r.get('emb_wavlm') and r.get('ref_emb_wavlm')]
        try:
            out[n] = eer(*sv_trials(ok))['eer'] if ok else None
        except ValueError as e:
            out[n] = None
            print(f'  EER {ts.name} {n}: {e}', flush=True)
    return out


def budget_sweep(sets, systems):
    out = {}
    names = [systems['faiss'], systems['s2'], systems['s3'], systems['crt_al'], 'knnvc']
    for s in SETS:
        ts = sets[s]
        for n in dict.fromkeys(names):
            for b in (3, 30, 180):
                if not ts.has(n, b):
                    continue
                for q in ('wer', 'wer_u', 'sim', 'utmos') + (('mcd',) if s == 'vctk2k' else ()):
                    out.setdefault(s, {}).setdefault(n, {}).setdefault(q, {})[b] = ts.level(n, q, b)
    return out


def required(sets, systems):
    """Every system the families and the sweep need, per set and budget, and the family systems with one row per pair
    at 30 s; returns (missing, short)."""
    need = []
    core = [systems['faiss'], systems['s2'], systems['s3'], systems['crt_al']]
    for s in SETS:
        for n in core + systems['baselines']:
            need.append((s, 30, n))
        for n in (systems['crt_al'], systems['faiss']):
            need.append((s, 30, n + SRC))
        for b in (3, 180):
            for n in core + (['knnvc'] if 'knnvc' in systems['baselines'] else []):
                need.append((s, b, n))
    if any(p.get('P') for p in sets['vctk2k'].pairs):
        need.append(('vctk2k', 30, 'target_copy' + SRC))
    need = list(dict.fromkeys(need))
    missing = [x for x in need if not sets[x[0]].has(x[2], x[1])]
    short = [(s, b, n, len(sets[s].rows(n, b)), len(sets[s].pairs)) for s, b, n in need
             if b == 30 and (s, b, n) not in missing and not n.startswith('target_copy')
             and len(sets[s].rows(n, b)) != len(sets[s].pairs)]
    return missing, short


# ── report ───────────────────────────────────────────────────

def f(x, n=3):
    return 'n/a' if x is None else f'{x:.{n}f}'


def digits(q):
    return 2 if q in ('wer', 'wer_u', 'wer_b', 'wer_bu', 'cer', 'mcd') else 3


def ci(x, d):
    return f'{f(x["est"], d)} [{f(x.get("lo"), d)}, {f(x.get("hi"), d)}]' if x and x.get('est') is not None else 'n/a'


def decision_text(reject, fav):
    return ('reject' + (f', favours {fav}' if fav else '')) if reject else 'not rejected'


def fmt_h(h):
    n = digits(h['quantity'])
    holm = decision_text(h.get('reject'), h.get('favours'))
    if 'reject_strict' in h:
        holm += f'; whatever the CMOS shows: {decision_text(h["reject_strict"], h.get("favours_strict"))}'
    flag = f' [{h["flag"]}]' if h.get('flag') else ''
    wc = h.get('worst_case')
    wcs = f'; worst case Δ {f(wc.get("est"), n)}, p {f(wc.get("p"), 4)}' if wc else ''
    w = h.get('wilcoxon') or {}
    return (f'| {h["id"]}{flag} | {h["set"]} | {h["label"]} | {f(h.get("a"), n)} | {f(h.get("b"), n)} | '
            f'{f(h.get("est"), n)} [{f(h.get("lo"), n)}, {f(h.get("hi"), n)}] | {f(h.get("p"), 4)} | {holm} | '
            f'{h.get("n", 0)} ({h.get("fail_a", 0)}/{h.get("fail_b", 0)} failed, {h.get("undefined", 0)} undefined, '
            f'{h.get("redraws", 0)} redraws){wcs} | {f(w.get("p"), 4)} ({f(w.get("r_rb"))}) |')


HEAD = ('| Hypothesis | Set | Endpoint | A | B | Δ = A − B [95 % CI] | p | Holm decision | n pairs | '
        'Wilcoxon p (r_rb) |\n|---|---|---|---|---|---|---|---|---|---|')


def fmt_changed(ch):
    parts = [f'{", ".join(ch["holm"]) or "none"}']
    if 'holm_strict' in ch:
        parts.append(f'whatever the CMOS shows: {", ".join(ch["holm_strict"]) or "none"}')
    return '; '.join(parts)


def render(res):
    sy = res['systems']
    L = ['# Test-set results v1 (ANALYSIS_PLAN_v1.md §5, §7)', '',
         f'Written {res["written"]} by `benchmark/analyze_tests.py` (sha256 {res["script_sha256"]}; stats.py '
         f'{res["stats_sha256"]}). CRT-AL = {sy["choice"]} (`{sy["crt_al"]}`), S2 = `{sy["s2"]}` (τ_dev '
         f'{sy["tau_dev"]:g}), baselines {", ".join(sy["baselines"]) or "none"}. Crossed bootstrap B = {stats.B}, '
         f'seed {stats.SEED}: VCTK-2k independent, LTC-2k linked.', ''] + [f'- {x}' for x in NOTES] + ['']
    o = res['outcome']
    L += ['## Outcome (§7)', '']
    if o['p8_pending']:
        L += ['P8 (CMOS CRT-AL vs FAISS) comes from the listening test, which follows this run (§8.3). Until it '
              'reports, the outcome is given under both Holm variants of family P.', '']
    for name, v in o['variants'].items():
        gl = ('rejects in all four strata' if v['gate_l_all_reject'] else
              'does not reject in ' + ', '.join(v['gate_l_failed']))
        ni = f'; not interpretable as absence of leakage: {", ".join(v["not_interpretable"])}' \
            if v['not_interpretable'] else ''
        L += [f'- **{name}**: pattern {v["pattern"]} (favour CRT-AL: {", ".join(v["favour_crt_al"]) or "none"}; '
              f'favour FAISS: {", ".join(v["favour_faiss"]) or "none"}); gate L {gl}{ni}; framing **{v["framing"]}**'
              + (' (leakage not excluded: at most B)' if v['leakage_not_excluded'] else '') + '.']
    L.append('')
    for key, title in (('P', 'Family P (primary): CRT-AL − FAISS'), ('K', 'Family K: CRT-AL − CRT τ = 0.4'),
                       ('V', 'Family V (v0 primary): CRT(τ_dev) − FAISS, uncapped WER'),
                       ('S', 'Family S: CRT-AL − baselines')):
        fam = res['families'][key]
        L += [f'## {title}', '', f'Holm {fam["holm"]}.', '', HEAD] + [fmt_h(h) for h in fam['hypotheses']]
        c = fam.get('cmos')
        if c:
            L.append(f'| {c["id"]} | both | CMOS | | | {f(c.get("est"))} [{f(c.get("lo"))}, {f(c.get("hi"))}] | '
                     f'{f(c.get("p"), 4)} | {c["status"]}'
                     f'{"; " + decision_text(c["reject"], c.get("favours")) if c["status"] == "in family" else ""} | '
                     f'{c.get("n_raters", "–")} raters | |')
        L.append('')
    g = res['gate_l']
    L += ['## Gate L (leakage), H0 Δ_src ≥ +0.01', '',
          '| Test | Set / stratum | SIM_src CRT-AL | SIM_src FAISS | Δ_src [95 % CI] | Rejects | n (no source E; failed '
          'A/B) | Worst case | M9 difference [95 % CI] | Share D > 0 (CRT-AL / FAISS) | target_copy SIM_src (CRT-AL '
          'on those pairs) | Interpretable |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, r in g['strata'].items():
        m9, sh, nat, wc = r['m9_diff'], r['share_d_pos'], r['natural_other_speaker'], r.get('worst_case')
        L.append(f'| {r["id"]} | {k} | {f(r.get("a"))} | {f(r.get("b"))} | {f(r.get("est"))} '
                 f'[{f(r.get("lo"))}, {f(r.get("hi"))}] | {r["rejects"]} | {r.get("n", 0)} ({r["n_no_source_E"]}; '
                 f'{r.get("fail_a", 0)}/{r.get("fail_b", 0)}) | '
                 f'{"hi " + f(wc.get("hi")) + ", rejects " + str(wc.get("rejects")) if wc else "–"} | '
                 f'{f(m9.get("est"))} [{f(m9.get("lo"))}, {f(m9.get("hi"))}] | '
                 f'{f(sh["crt_al"]["share"])} / {f(sh["faiss"]["share"])} | '
                 f'{f(nat["sim_src_target_copy"]) + " (" + f(nat["sim_src_crt_al_same_pairs"]) + ")" if nat else "–"}'
                 f' | {r["interpretable"]} |')
    L += ['', f'All four reject: {g["all_reject"]}; at the worst case: {g["worst_case"]["all_reject"]}.', '',
          '## Sensitivity analyses', '']
    sens = res['sensitivity']
    u = sens['union_pk']
    L.append(f'- P ∪ K, Holm over {u["m"]}' + (f' (over {u["m_strict"]} whatever the CMOS shows)' if 'm_strict' in u
                                                 else '') + f': decisions that change: {fmt_changed(u["changed"])}.')
    for key, v in sens['wer_definition'].items():
        L.append(f'- Other WER definition, family {key}: decisions that change: {fmt_changed(v["changed"])}.')
    for key, v in sens['ltc_independent'].items():
        L.append(f'- LTC-2k with the independent bootstrap, family {key}: decisions that change: '
                 f'{fmt_changed(v["changed"])}.')
    gi = sens['ltc_independent_gate']
    L.append(f'- LTC-2k gate L with the independent bootstrap: L2s {gi["L2s"]}, L2c {gi["L2c"]}.')
    for key, v in sens['worst_case'].items():
        L.append(f'- Failed conversions at their worst case, family {key}: computed for '
                 f'{", ".join(v["worst_case_used"]) or "no hypothesis (equal failure counts)"}; decisions that change: '
                 f'{fmt_changed(v["changed"])}.')
    gw = g['worst_case']
    L.append(f'- Failed conversions at their worst case, gate L: computed for {", ".join(gw["used"]) or "no stratum"}; '
             f'all four reject: {gw["all_reject"]}.')
    L += ['', '## Descriptive (CIs only)', '']
    for s in SETS:
        L += [f'### Levels, {s}, 30 s: intelligibility and quality', '',
              '| System | rows (failed) | capped WER % | WER % | capped WER % base | WER % base | capped CER % | UTMOS | '
              'dB-MCD | log-F0 RMSE | V/UV error | F0 PCC |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for n, row in res['levels'][s].items():
            flag = f' [{row["flag"]}]' if row.get('flag') else ''
            L.append(f'| {n}{flag} | {row["n_rows"]} ({row["fail"]}) | ' + ' | '.join(
                ci(row.get(q), digits(q)) for q in ('wer', 'wer_u', 'wer_b', 'wer_bu', 'cer', 'utmos', 'mcd', 'lf0',
                                                      'vuv', 'f0pcc')) + ' |')
        L += ['', f'### Levels, {s}, 30 s: speaker', '',
              '| System | SIM | SECS | SIM_src | SECS_src | M9 D | share D > 0 | EER |', '|---|---|---|---|---|---|---|---|']
        for n, row in res['levels'][s].items():
            L.append(f'| {n} | ' + ' | '.join(ci(row.get(q), 3) for q in ('sim', 'secs', 'sim_src', 'secs_src', 'm9'))
                     + f' | {f((row.get("share_d_pos") or {}).get("share"))} | {f(res["eer"][s].get(n))} |')
        cov = {n: row['coverage'] for n, row in res['levels'][s].items() if row.get('coverage')}
        if cov:
            L += ['', f'### Coverage, {s}, 30 s', '', '| System | Field | Mean |', '|---|---|---|']
            for n, c in cov.items():
                for k, v in c.items():
                    L.append(f'| {n} | {k} | {f(v["overall"]) + " overall" if isinstance(v, dict) else f(v)} |')
        L.append('')
    L += ['### Budget sweep (E16)', '', '| Set | System | Endpoint | 3 s | 30 s | 180 s |', '|---|---|---|---|---|---|']
    for s, d in res['budget_sweep'].items():
        for n, qs in d.items():
            for q, bs in qs.items():
                L.append(f'| {s} | {n} | {LABEL[q]} | ' + ' | '.join(
                    ci(bs.get(b), digits(q)) + (f' (n {bs[b]["n"]})' if b in bs else '') for b in (3, 30, 180)) + ' |')
    L += ['', '### Gender pairs', '']
    for s, d in res['gender'].items():
        L += [f'{s}:', '', '| System | Pair | capped WER % | SIM | UTMOS | SIM_src |', '|---|---|---|---|---|---|']
        for n, gs in d.items():
            for gp, v in gs.items():
                L.append(f'| {n} | {gp} | ' + ' | '.join(ci(v.get(q), digits(q)) for q in ('wer', 'sim', 'utmos',
                                                                                              'sim_src')) + ' |')
        L.append('')
    L += ['## Inputs', ''] + [f'- job M part {k}: `{v}`' for k, v in res['m_done'].items()] + \
         [f'- `{x["path"]}` sha256 {x["sha256"]}' + (f' ({x["rows"]} rows)' if 'rows' in x else '')
          for x in res['inputs']]
    return '\n'.join(L) + '\n'


# ── main ─────────────────────────────────────────────────────

def analyse(sets, systems, cmos=None):
    a, fa, s3, s2 = systems['crt_al'], systems['faiss'], systems['s3'], systems['s2']
    P = family(sets, 'P', a, fa, ENDPOINTS, 'wer_capped', cmos, 'P8')
    K = family(sets, 'K', a, s3, ENDPOINTS, 'wer_capped', cmos, 'K8')
    V = family(sets, 'V', s2, fa, ENDPOINTS, 'wer')
    S = s_family(sets, systems, 'wer_capped')
    Lg = gate(sets, a, fa)
    fams = {'P': P, 'K': K, 'V': V, 'S': S}
    other_wer = {'P': family(sets, 'P', a, fa, ENDPOINTS, 'wer', cmos, 'P8'),
                 'K': family(sets, 'K', a, s3, ENDPOINTS, 'wer', cmos, 'K8'),
                 'V': family(sets, 'V', s2, fa, ENDPOINTS, 'wer_capped'),
                 'S': s_family(sets, systems, 'wer')}
    ind = stats.INDEPENDENT
    ltc_ind = {'P': family(sets, 'P', a, fa, ENDPOINTS, 'wer_capped', cmos, 'P8', mode_ltc=ind),
               'K': family(sets, 'K', a, s3, ENDPOINTS, 'wer_capped', cmos, 'K8', mode_ltc=ind),
               'V': family(sets, 'V', s2, fa, ENDPOINTS, 'wer', mode_ltc=ind),
               'S': s_family(sets, systems, 'wer_capped', mode_ltc=ind)}
    g_ind = gate(sets, a, fa, mode_ltc=ind)
    sens = {'union_pk': union_pk(P, K),
            'wer_definition': {k: {'changed': changed(fams[k], v), 'family': v} for k, v in other_wer.items()},
            'ltc_independent': {k: {'changed': changed(fams[k], v), 'family': v} for k, v in ltc_ind.items()},
            'ltc_independent_gate': {g_ind['strata'][k]['id']: g_ind['strata'][k]['rejects']
                                     for k in ('ltc2k/same', 'ltc2k/cross')},
            'worst_case': {k: worst_case_family(v) for k, v in fams.items()}}
    return {'families': fams, 'gate_l': Lg, 'outcome': outcomes(P, Lg, a, fa), 'sensitivity': sens}


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    v1 = os.path.join(here, '..', 'paper', 'journal_extension', 'results', 'v1')
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--m_dir', default=os.path.join(v1, 'm'))
    ap.add_argument('--vctk_pairs', default=os.path.join(v1, 'pairs', 'vctk2k.jsonl'))
    ap.add_argument('--ltc_pairs', default=os.path.join(v1, 'pairs', 'ltc2k.jsonl'))
    ap.add_argument('--source_e', default=os.path.join(v1, 'pairs', 'vctk2k_source_E.json'))
    ap.add_argument('--record', default=os.path.join(here, 'DEV_SELECTION_RECORD_v1.json'))
    ap.add_argument('--cmos', default=None, help='JSON {"P8": {est, lo, hi, p, n_raters}, "K8": {...}} (§5.4)')
    ap.add_argument('--out', default=os.path.join(here, 'TEST_RESULTS_v1.md'))
    a = ap.parse_args(argv)
    systems = systems_from_record(json.load(open(a.record)))
    vp, lp = read_pairs(a.vctk_pairs), read_pairs(a.ltc_pairs)
    want = 400 * int(systems['vctk_targets'])
    if len(vp) != want:
        raise SystemExit(f'ERROR: {a.vctk_pairs} has {len(vp)} pairs; the record\'s {systems["vctk_targets"]}-target '
                         f'variant has {want}')
    src_e = {k for k, v in json.load(open(a.source_e)).items() if (v or {}).get('paths')}
    inputs = [{'path': os.path.abspath(p), 'sha256': sha256(p)} for p in (a.record, a.vctk_pairs, a.ltc_pairs,
                                                                          a.source_e) + ((a.cmos,) if a.cmos else ())]
    files = [('t1', 30), ('t3', 3), ('t3', 180)]
    if systems['baselines']:
        files += [('t4', 30)] + ([('t4', 3), ('t4', 180)] if 'knnvc' in systems['baselines'] else [])
    m_done = check_done(a.m_dir, ['t1', 't2', 't3'] + (['t4'] if systems['baselines'] else []))
    sets = {}
    for s, pairs, fl in (('vctk2k', vp, files), ('ltc2k', lp, [('t2' if x == 't1' else x, b) for x, b in files])):
        R, ins = load_set(a.m_dir, s, fl, pairs)
        inputs += ins
        sets[s] = TestSet(s, pairs, R, src_e if s == 'vctk2k' else None)
    missing, short = required(sets, systems)
    if missing or short:
        raise SystemExit(f'ERROR: no rows for {missing}; not one row per pair at 30 s (set, budget, system, rows, '
                         f'pairs): {short}')
    cmos = json.load(open(a.cmos)) if a.cmos else None
    res = analyse(sets, systems, cmos)
    res.update(written=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
               script_sha256=sha256(os.path.abspath(__file__)), stats_sha256=sha256(stats.__file__),
               systems=systems, inputs=inputs, m_done=m_done,
               levels={s: levels(sets[s], systems) for s in SETS},
               gender={s: gender_table(sets[s], systems) for s in SETS},
               eer={s: eer_table(sets[s]) for s in SETS},
               budget_sweep=budget_sweep(sets, systems))
    md = render(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fo:
        fo.write(md)
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as fo:
        json.dump(res, fo, indent=1, default=float)
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
