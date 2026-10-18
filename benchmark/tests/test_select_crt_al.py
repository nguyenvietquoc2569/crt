"""Tests of benchmark/select_crt_al.py: ANALYSIS_PLAN_v1.md §4.1-§4.6 (Dev quantities, the S2 τ rule, S1u, E17,
CRT-AL with E1-E4, the one-SE choice and F1-F3, the record), §5.3 (size-check inputs), §5.4 (listening items) and
§0.3 row 20 (capped WER in the v1 rules, v0's uncapped WER in §4.2 and family V)."""
import functools
import hashlib
import json
import os
import random
import sys

import numpy as np
import pytest

import fixtures_dev as F
import select_crt_al as S
import stats

SY = S.SYSTEMS
S1, A5, S3, S2 = SY['S1'], SY['A5'], SY['S3'], SY['S2'][0.9]
W = stats.WER_KEY
PIDS = ['p1', 'p2']


def approx(x):
    return pytest.approx(x, rel=1e-12, abs=1e-12)


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ── rule logic on given point estimates ──────────────────────

class FakeDev:
    """Stands in for select_crt_al.Dev: point estimates q ({system: {quantity: value}}); a comparison A - B has
    est = q[A] - q[B], SE se[(A, B, quantity)] (default 1) and the CI est -+ 2 SE, unless hi sets its upper limit."""

    def __init__(self, q, se=None, hi=None):
        self.q, self.se, self.hi = q, se or {}, hi or {}

    def has(self, system, kind='tgt'):
        return bool(system) and system in self.q

    def ok(self, system, pid, kind='tgt'):
        return True

    def subset(self, pids, which):
        return list(pids)

    def point(self, system, pids, q, sub=None):
        return self.q[system].get(q) if self.has(system) else None

    def diff(self, a, b, pids, q):
        if not (self.has(a) and self.has(b)):
            return None
        est = self.q[a][q] - self.q[b][q]
        se = self.se.get((a, b, q), 1.0)
        return {'est': est, 'se': se, 'lo': est - 2 * se, 'hi': self.hi.get((a, b, q), est + 2 * se)}


REF = {S1: dict(wer=90.0, wer_u=120.0, sim=0.91, utmos=1.70, mcd=8.8, src_same=0.80, src_cross=0.62),
       S2: dict(wer=85.0, wer_u=110.0, sim=0.90, utmos=1.60, mcd=9.0),
       A5: dict(wer=88.0, wer_u=115.0, sim=0.91, utmos=1.70, mcd=8.7)}
CAND = dict(wer=20.0, wer_u=20.0, sim=0.85, utmos=2.2, mcd=9.0, src_same=0.78, src_cross=0.60)   # meets E1-E4


def fake(cands, se=None, hi=None, ref=REF):
    """FakeDev of the reference systems and the candidates {label: overrides of CAND}; se and hi are keyed by
    (label or system, label or system, quantity)."""
    name = lambda x: SY[x] if x in S.CANDS else x       # noqa: E731
    key = lambda d: {(name(a), name(b), w): v for (a, b, w), v in (d or {}).items()}      # noqa: E731
    return FakeDev({**{s: dict(v) for s, v in ref.items()}, **{SY[c]: dict(CAND, **v) for c, v in cands.items()}},
                   key(se), key(hi))


@pytest.mark.parametrize('flag, at, past', [
    ('E1', {'sim': 0.80}, {'sim': np.nextafter(0.80, 0)}),
    ('E2', {'utmos': 1.70}, {'utmos': np.nextafter(1.70, 0)}),
    ('E3', {'src_same': 0.80 + S.MARGIN}, {'src_same': np.nextafter(0.80 + S.MARGIN, 1)}),
    ('E3', {'src_cross': 0.62 + S.MARGIN}, {'src_cross': np.nextafter(0.62 + S.MARGIN, 1)}),
])
def test_eligibility_flag_flips_at_its_threshold(flag, at, past):
    for over, met in ((at, True), (past, False)):
        r = S.crt_al_rule(fake({'C1': over}), PIDS, S2)
        assert r['flags']['C1'][flag] is met
        assert all(v is True for k, v in r['flags']['C1'].items() if k != flag)
        assert r['eligible'] == (['C1'] if met else [])


@pytest.mark.parametrize('ref', [S2, A5])
def test_e4_needs_the_upper_limit_below_zero_against_s2_and_a5(ref):
    for hi, met in ((0.0, False), (np.nextafter(0.0, -1), True), (3.0, False)):
        r = S.crt_al_rule(fake({'C1': {}}, hi={('C1', ref, 'wer'): hi}), PIDS, S2)
        assert r['flags']['C1']['E4'] is met and r['eligible'] == (['C1'] if met else [])


def test_e4_not_computable_counts_as_not_met():
    r = S.crt_al_rule(fake({'C1': {}}, ref={S1: REF[S1], S2: REF[S2]}), PIDS, S2)       # no A5 rows
    assert r['flags']['C1']['E4'] is None and r['eligible'] == [] and r['fallback'] == 'F3'
    assert any('not computable, counted as not met: C1 E4' in t for t in r['trace'])


def test_one_se_rule_takes_the_highest_sim_within_s_k():
    cands = {'C1': {'wer': 21.0, 'sim': 0.86}, 'C2': {'wer': 20.0, 'sim': 0.85}, 'C3': {'wer': 22.5, 'sim': 0.90},
             'C4': {'wer': 20.5, 'sim': 0.84}}
    se = {('C1', 'C2', 'wer'): 1.0, ('C3', 'C2', 'wer'): 2.4, ('C4', 'C2', 'wer'): 0.3}
    r = S.crt_al_rule(fake(cands, se), PIDS, S2)
    # C1 +1.00 <= 1.00 (in T at equality); C3 +2.50 > 2.40 although its SIM is the highest; C4 +0.50 > 0.30
    assert r['c_m'] == 'C2' and r['s_k'] == {'C1': 1.0, 'C2': 0.0, 'C3': 2.4, 'C4': 0.3}
    assert r['T'] == ['C1', 'C2'] and r['choice'] == 'C1' and r['fallback'] is None
    r = S.crt_al_rule(fake(cands, {**se, ('C3', 'C2', 'wer'): 2.5}), PIDS, S2)
    assert r['T'] == ['C1', 'C2', 'C3'] and r['choice'] == 'C3'


@pytest.mark.parametrize('wers, c_m', [
    ({'C1': 20.004, 'C2': 19.996}, 'C1'),           # both 20.00: the lower candidate number
    ({'C1': 20.004, 'C2': 19.994}, 'C2'),           # 20.00 vs 19.99
    ({'C5': 20.0, 'C3': 20.0}, 'C3'),
])
def test_c_m_ties_at_two_decimals_go_to_the_lower_number(wers, c_m):
    assert S.crt_al_rule(fake({c: {'wer': w} for c, w in wers.items()}), PIDS, S2)['c_m'] == c_m


@pytest.mark.parametrize('q, choice', [
    ({'C1': (0.8504, 9.0, 2.2), 'C2': (0.8506, 9.5, 2.0)}, 'C2'),            # SIM 0.850 < 0.851
    ({'C1': (0.8504, 9.004, 2.2), 'C2': (0.8496, 8.994, 2.0)}, 'C2'),        # SIM tie: dB-MCD 9.00 > 8.99
    ({'C1': (0.8504, 9.004, 2.194), 'C2': (0.8496, 8.996, 2.204)}, 'C2'),    # dB-MCD tie: UTMOS 2.19 < 2.20
    ({'C1': (0.8504, 9.004, 2.196), 'C2': (0.8496, 8.996, 2.204)}, 'C1'),    # all tied: the lower number
    ({'C3': (0.85, None, 2.2), 'C4': (0.85, 9.9, 2.2)}, 'C4'),               # a missing value ranks last
    ({'C3': (None, 9.0, 2.2), 'C4': (0.81, 9.9, 2.0)}, 'C4'),
])
def test_sim_tie_break_chain(q, choice):
    qq = {c: {'sim': s, 'mcd': m, 'utmos': u} for c, (s, m, u) in q.items()}
    assert S.best_sim(list(qq), qq) == choice
    if all(v['sim'] for v in qq.values()):      # through steps 3-4, with every candidate in T
        r = S.crt_al_rule(fake(qq, se={(c, d, 'wer'): 5.0 for c in qq for d in qq}), PIDS, S2)
        assert sorted(r['T']) == sorted(qq) and r['choice'] == choice


def test_f1_highest_sim_among_candidates_meeting_e3_and_e4():
    cands = {'C1': {'sim': 0.79}, 'C2': {'sim': 0.795}, 'C3': {'sim': 0.83, 'src_cross': 0.70},     # C3 fails E3
             'C4': {'sim': 0.81, 'utmos': 1.5}}                                                    # C4 fails E2
    r = S.crt_al_rule(fake(cands), PIDS, S2)
    assert r['eligible'] == [] and r['fallback'] == 'F1' and r['choice'] == 'C4'
    assert r['deficits'] == {'E1': pytest.approx(0.01), 'E2': pytest.approx(-0.2)}


@pytest.mark.parametrize('cands, choice', [
    ({'C1': {'src_cross': 0.65}, 'C2': {'src_cross': 0.66}, 'C3': {'src_cross': 0.64, 'wer': 95.0}}, 'C1'),
    ({'C1': {'src_cross': 0.6504, 'sim': 0.85}, 'C2': {'src_cross': 0.6496, 'sim': 0.86}}, 'C2'),   # 0.030 tie: SIM
    ({'C1': {'src_cross': 0.6504, 'sim': 0.8504}, 'C2': {'src_cross': 0.6496, 'sim': 0.8496}}, 'C1'),  # number
    ({'C1': {'src_same': 0.85, 'src_cross': 0.60}, 'C2': {'src_cross': 0.66}}, 'C2'),   # max over the strata
    ({'C1': {'src_cross': None}, 'C2': {'src_cross': 0.70}}, 'C2'),                     # no SIM_src: ranks last
])
def test_f2_smallest_leakage_excess_then_sim(cands, choice):
    r = S.crt_al_rule(fake(cands), PIDS, S2)       # C3's WER 95 % is above S2's 85 %: it fails E4
    assert r['fallback'] == 'F2' and r['choice'] == choice
    s1 = REF[S1]
    for c, v in r['excess'].items():
        q = dict(CAND, **cands[c])
        want = None if None in (q['src_same'], q['src_cross']) else max(q['src_same'] - s1['src_same'],
                                                                      q['src_cross'] - s1['src_cross'])
        assert v == (want if want is None else pytest.approx(want))


def test_f3_no_candidate_meets_e4():
    r = S.crt_al_rule(fake({c: {'wer': 99.0} for c in S.CANDS}), PIDS, S2)
    assert r['choice'] is None and r['fallback'] == 'F3' and any(t.startswith('F3') for t in r['trace'])


def test_tolerance_sensitivity_table():
    cands = {'C1': {'wer': 20.0, 'sim': 0.82}, 'C2': {'wer': 20.8, 'sim': 0.83}, 'C3': {'wer': 21.5, 'sim': 0.84},
             'C4': {'wer': 22.5, 'sim': 0.85}}
    r = S.crt_al_rule(fake(cands), PIDS, S2)        # SE 1 throughout; CI = difference -+ 2
    assert (r['c_m'], r['T'], r['choice']) == ('C1', ['C1', 'C2'], 'C2')
    assert {k: (v['T'], v['choice']) for k, v in r['sensitivity'].items()} == {
        'fixed tolerance 0 points': (['C1'], 'C1'), 'fixed tolerance 1 points': (['C1', 'C2'], 'C2'),
        'fixed tolerance 2 points': (['C1', 'C2', 'C3'], 'C3'),
        '95 % CI of WER(Ck) − WER(C_m) contains 0': (['C1', 'C2', 'C3'], 'C3')}
    cands['C4'] = {'wer': 19.996, 'sim': 0.85}             # tied with C1 at 20.00 (C_m stays C1), CI below 0
    r = S.crt_al_rule(fake(cands, se={('C4', 'C1', 'wer'): 0.001}), PIDS, S2)
    assert r['c_m'] == 'C1' and r['T'] == ['C1', 'C2', 'C4'] and r['diff_vs_m']['wer']['C4']['hi'] < 0
    assert r['sensitivity']['95 % CI of WER(Ck) − WER(C_m) contains 0']['T'] == ['C1', 'C2', 'C3']


def test_uncapped_variant_uses_uncapped_wer_in_e4_and_steps_1_2():
    cands = {'C1': {'wer': 15.0, 'wer_u': 200.0, 'sim': 0.84}, 'C2': {'wer': 20.0, 'wer_u': 20.0, 'sim': 0.83},
             'C3': {'wer': 21.0, 'wer_u': 21.0, 'sim': 0.86}}
    se = {('C3', 'C2', 'wer_u'): 1.5}
    cap, unc = (S.crt_al_rule(fake(cands, se), PIDS, S2, wer=w) for w in ('wer', 'wer_u'))
    assert cap['eligible'] == ['C1', 'C2', 'C3'] and (cap['c_m'], cap['T'], cap['choice']) == ('C1', ['C1'], 'C1')
    assert unc['flags']['C1']['E4'] is False and unc['eligible'] == ['C2', 'C3']
    assert (unc['c_m'], unc['T'], unc['choice']) == ('C2', ['C2', 'C3'], 'C3')
    assert all(set(r['e4']['C1']) == set(r['diff_vs_m']) == {'wer', 'wer_u'} for r in (cap, unc))


def tau_dev(grid, sim1=0.91):
    """FakeDev for §4.2: grid {τ: (uncapped WER, capped WER, SIM, dB-MCD)}; a τ left out has no rows."""
    return FakeDev({S1: {'sim': sim1}, **{SY['S2'][t]: dict(zip(('wer_u', 'wer', 'sim', 'mcd'), v))
                                          for t, v in grid.items()}})


def grid_below(sim=0.85):
    return {t: (100.0, 90.0, sim, 9.0) for t in S.TAU_GRID}


def test_tau_rule_lowest_uncapped_wer_among_sim_at_least_faiss():
    g = grid_below()
    g[0.1] = (50.0, 40.0, 0.90, 9.0)        # lowest WER, SIM below FAISS
    g[0.5] = (70.0, 30.0, 0.91, 9.0)        # SIM equal to FAISS meets; lowest capped WER (not the rule's input)
    g[0.6] = (60.0, 55.0, 0.92, 9.5)
    r = S.tau_rule(tau_dev(g), PIDS)
    assert (r['tau'], r['system'], r['step'], r['missing']) == (0.6, SY['S2'][0.6], 1, [])
    assert [x['meets'] for x in r['grid']] == [t in (0.5, 0.6) for t in S.TAU_GRID]
    assert r['grid'][0]['deficit'] == pytest.approx(0.01)


@pytest.mark.parametrize('w6, m6, w7, m7, tau, step', [
    (60.004, 9.10, 59.996, 9.00, 0.7, 2),       # both 60.00 % at two decimals: the lower dB-MCD
    (59.996, 9.00, 60.004, 9.10, 0.6, 2),
    (60.004, 9.001, 59.996, 9.004, 0.6, 2),     # dB-MCD compared unrounded
    (60.0, 9.0, 60.0, 9.0, 0.6, 2),             # tied on dB-MCD too: the lower τ
    (60.006, 9.0, 60.004, 9.5, 0.7, 1),         # 60.01 vs 60.00: no tie, the WER decides
])
def test_tau_rule_wer_ties_at_two_decimals_broken_by_mcd(w6, m6, w7, m7, tau, step):
    g = grid_below()
    g[0.6], g[0.7] = (w6, 50.0, 0.92, m6), (w7, 50.0, 0.93, m7)
    r = S.tau_rule(tau_dev(g), PIDS)
    assert (r['tau'], r['step']) == (tau, step)


def test_tau_rule_step_3_smallest_sim_deficit():
    g = grid_below()
    g[0.3] = (120.0, 95.0, 0.905, 9.0)          # no τ reaches FAISS's 0.91; τ = 0.3 is closest
    r = S.tau_rule(tau_dev(g), PIDS)
    assert (r['tau'], r['step']) == (0.3, 3) and any('step 3' in n for n in r['notes'])


def test_tau_rule_runs_on_the_present_values_and_needs_faiss_sim():
    g = {0.4: (80.0, 70.0, 0.92, 9.0), 0.8: (70.0, 60.0, 0.93, 9.0)}
    r = S.tau_rule(tau_dev(g), PIDS)
    assert r['tau'] == 0.8 and len(r['missing']) == 7 and any('run on 2 of 9' in n for n in r['notes'])
    r = S.tau_rule(tau_dev(g, sim1=None), PIDS)
    assert r['tau'] is None and r['step'] is None and 'not computable' in r['notes'][0]


U1, U2 = SY['S1u']


def s1u_dev(u1, u2):
    """u: (capped WER, uncapped WER, SIM, dB-MCD) of a setting, None: no rows."""
    q = {S1: {'sim': 0.91, 'wer': 90.0, 'wer_u': 120.0}}
    q.update({s: dict(zip(('wer', 'wer_u', 'sim', 'mcd'), v), utmos=2.0) for s, v in ((U1, u1), (U2, u2)) if v})
    return FakeDev(q)


@pytest.mark.parametrize('u1, u2, setting', [
    ((30.0, 90.0, 0.93, 9.0), (31.0, 40.0, 0.94, 8.0), U1),         # both meet: the lower capped WER
    ((30.004, 30.0, 0.93, 9.0), (29.996, 30.0, 0.94, 8.5), U2),     # 30.00 both: the lower dB-MCD
    ((30.0, 30.0, 0.93, 9.0), (30.0, 30.0, 0.94, 9.0), U1),         # tied on dB-MCD: the first setting
    ((30.0, 30.0, 0.90, 9.0), (35.0, 35.0, 0.91, 9.0), U2),         # only U2 meets (SIM equal to FAISS)
    ((30.0, 30.0, 0.90, 9.0), (35.0, 35.0, 0.89, 9.0), U1),         # neither meets: the higher SIM
])
def test_s1u_rule(u1, u2, setting):
    assert S.s1u_rule(s1u_dev(u1, u2), PIDS)['setting'] == setting


def test_s1u_rule_with_one_setting_missing():
    r = S.s1u_rule(s1u_dev((30.0, 30.0, 0.95, 9.0), None), PIDS)
    assert r['setting'] == U1 and r['missing'] == [U2] and 'only' in r['how']
    assert S.s1u_rule(s1u_dev(None, None), PIDS)['setting'] is None


E17 = SY['E17']


@pytest.mark.parametrize('e, s2, sim, enters', [
    ((94.69, 200.0), (94.70, 100.0), 0.91, True),       # lower capped WER at two decimals; SIM equal to FAISS
    ((94.696, 90.0), (94.704, 100.0), 0.95, False),     # lower, but equal at two decimals (the screen's case)
    ((80.0, 80.0), (94.70, 100.0), np.nextafter(0.91, 0), False),
    ((95.0, 80.0), (94.70, 100.0), 0.95, False),        # a lower uncapped WER does not count
])
def test_e17_rule(e, s2, sim, enters):
    q = {S1: {'wer': 90.0, 'wer_u': 120.0, 'sim': 0.91}, S2: dict(zip(('wer', 'wer_u'), s2), sim=0.90),
         E17: dict(zip(('wer', 'wer_u'), e), sim=sim)}
    assert S.e17_rule(FakeDev(q), PIDS, S2)['enters'] is enters
    assert S.e17_rule(FakeDev(q), PIDS, None)['enters'] is None


# ── on synthetic rows ────────────────────────────────────────

def small(n=112, scenario='eligible'):
    pairs = F.dev_pairs(n)
    return pairs, F.system_rows(pairs, F.profiles(scenario))


def test_dev_quantities_follow_section_4_1():
    pairs, R = small()
    ids = [p['pair_id'] for p in pairs]
    F.set_wer(R, SY['C1'], ids[0], 40)          # a Whisper loop on a wer_primary pair: 40 edits on 10 words
    dev = S.Dev(R, pairs)
    c, cs = R[SY['C1']], R[SY['C1'] + S.SRC]
    info = {p['pair_id']: p for p in pairs}
    prim = [k for k in ids if info[k]['wer_primary']]
    e = {k: sum(c[k][W][x] for x in 'SDI') for k in ids}
    n = {k: c[k][W]['N'] for k in ids}
    assert e[ids[0]] > n[ids[0]] and info[ids[0]]['wer_primary']
    words = sum(n[k] for k in prim)
    assert dev.point(SY['C1'], ids, 'wer') == approx(100 * sum(min(e[k], n[k]) for k in prim) / words)
    assert dev.point(SY['C1'], ids, 'wer_u') == approx(100 * sum(e[k] for k in prim) / words)
    assert dev.point(SY['C1'], ids, 'sim') == pytest.approx(np.mean([c[k]['sim_wavlm'] for k in ids]))
    assert dev.point(SY['C1'], ids, 'mcd') == pytest.approx(np.mean([c[k]['mcd_db'] for k in ids if info[k]['P']]))
    for g, combos in (('same', ('F2F', 'M2M')), ('cross', ('F2M', 'M2F'))):
        sub = [k for k in ids if info[k]['genders'] in combos]
        assert dev.point(SY['C1'], ids, f'src_{g}') == pytest.approx(np.mean([cs[k]['sim_wavlm'] for k in sub]))
        assert dev.point(SY['C1'], ids, f'm9_{g}') == pytest.approx(
            np.mean([cs[k]['sim_wavlm'] - c[k]['sim_wavlm'] for k in sub]))
    assert dev.point('no_such_system', ids, 'sim') is None and dev.level('no_such_system', ids, 'sim') is None


def test_crt_al_rule_on_rows_is_consistent_with_its_comparisons():
    pairs, R = small()
    ids = [p['pair_id'] for p in pairs]
    r = S.crt_al_rule(S.Dev(R, pairs), ids, S2)
    assert r['eligible'] == ['C1', 'C2', 'C3'] and r['flags']['C4']['E3'] is False and r['flags']['C5']['E1'] is False
    m = min(r['eligible'], key=lambda c: (round(r['q'][c]['wer'], 2), c))
    s = {c: 0.0 if c == m else r['diff_vs_m']['wer'][c]['se'] for c in r['eligible']}
    T = [c for c in r['eligible'] if c == m or r['diff_vs_m']['wer'][c]['est'] <= s[c]]
    assert (r['c_m'], r['s_k'], r['T'], r['choice']) == (m, s, T, S.best_sim(T, r['q']))
    for c in r['candidates']:
        for w, metric in (('wer', 'wer_capped'), ('wer_u', 'wer')):
            assert r['e4'][c][w]['S2']['metric'] == metric
            assert r['e4'][c][w]['S2']['est'] == pytest.approx(r['q'][c][w] - r['ref'][S2][w])
        assert r['flags'][c]['E4'] == S.e4_met(r['e4'][c]['wer'])


VC = {'mu': 0.0, 'sigma2_a': 0.0, 'sigma2_b': 0.0, 'sigma2_c': 0.0, 'sigma2_e': 1.0, 'converged': True,
      'method': 'stub', 'reml_llf': {}, 'fit_errors': {}, 'warnings': []}


def spy_vc(monkeypatch):
    seen = []

    def vc(d, src, tgt):
        seen.append((np.asarray(d, float), list(src), list(tgt)))
        return dict(VC, **stats.design_counts(src, tgt))
    monkeypatch.setattr(stats, 'variance_components', vc)
    return seen


def linearised(R, a, b, pids, capped):
    ea = np.array([sum(R[a][k][W][x] for x in 'SDI') for k in pids], float)
    eb = np.array([sum(R[b][k][W][x] for x in 'SDI') for k in pids], float)
    n = np.array([R[a][k][W]['N'] for k in pids], float)
    if capped:
        ea, eb = np.minimum(ea, n), np.minimum(eb, n)
    return 100 * (ea - eb) / n.mean(), 100 * (ea.sum() - eb.sum()) / n.sum()


def test_size_conditions_fit_p_and_k_on_capped_wer(monkeypatch):
    pairs, R = small()
    ids = [p['pair_id'] for p in pairs]
    prim = [p['pair_id'] for p in pairs if p['wer_primary']]
    for k in prim[:3]:
        F.set_wer(R, S1, k, 60)                  # FAISS loops: capped and uncapped differ
    seen = spy_vc(monkeypatch)
    conds = S.size_conditions(S.Dev(R, pairs), ids, {'choice': 'C1'}, S2)
    assert [(c['vctk2k'], c['ltc2k']) for c in conds] == [
        ('P1', 'P2'), ('P3', 'P4'), ('P5', 'P6'), ('P7', None), ('K1', 'K2'), ('K3', 'K4'), ('K5', 'K6'), ('K7', None),
        ('L1s', 'L2s'), ('L1c', 'L2c')]
    assert all(c['vc'] is not None for c in conds) and len(seen) == len(conds)
    a = SY['C1']
    for c, (d, src, tgt) in zip(conds, seen):
        b = {'P': S1, 'K': S3, 'L': S1}[c['vctk2k'][0]]
        if c['kind'] == 'wer':
            dc, want = linearised(R, a, b, prim, capped=True)
            _, other = linearised(R, a, b, prim, capped=False)
            assert np.allclose(d, dc) and c['wer_capped'] is True
            assert c['dev_mean'] == pytest.approx(want) and c['dev_mean_uncapped'] == pytest.approx(other)
            assert src == [R[a][k]['source_speaker'] for k in prim]
        elif c['kind'].startswith('gate_'):
            sub = [p['pair_id'] for p in pairs if stats.stratum(p['genders']) == c['kind'][5:]]
            want = [R[a + S.SRC][k]['sim_wavlm'] - R[b + S.SRC][k]['sim_wavlm'] for k in sub]
            assert (c['a'], c['b']) == (a + S.SRC, b + S.SRC) and np.allclose(d, want)
            assert c['delta_dev'] == pytest.approx(np.mean(want))
        else:
            assert (c['a'], c['b'], c['delta_dev']) == (a, b, None) and 'wer_capped' not in c
    # FAISS's pairs above 100 % (the loops among them) put its uncapped WER far above its capped WER
    assert linearised(R, a, S1, prim, capped=False)[1] - conds[0]['dev_mean'] < -10


def test_size_conditions_fit_family_v_on_uncapped_wer_under_f3(monkeypatch):
    pairs, R = small(scenario='f3')
    ids = [p['pair_id'] for p in pairs]
    prim = [p['pair_id'] for p in pairs if p['wer_primary']]
    for k in prim[:3]:
        F.set_wer(R, S2, k, 80)                  # S2 loops: its capped and uncapped WER differ
    seen = spy_vc(monkeypatch)
    conds = S.size_conditions(S.Dev(R, pairs), ids, {'choice': None, 'fallback': 'F3'}, S2)
    assert [(c['vctk2k'], c['ltc2k'], c['a'], c['b']) for c in conds] == [
        ('V1', 'V2', S2, S1), ('V3', 'V4', S2, S1), ('V5', 'V6', S2, S1), ('V7', None, S2, S1)]
    d = seen[0][0]
    du, want = linearised(R, S2, S1, prim, capped=False)
    _, capped = linearised(R, S2, S1, prim, capped=True)
    assert conds[0]['wer_capped'] is False and np.allclose(d, du) and conds[0]['dev_mean'] == pytest.approx(want)
    assert conds[0]['dev_mean_capped'] == pytest.approx(capped) and abs(want - capped) > 1


def test_size_conditions_report_unfittable_conditions():
    pairs, R = small()
    ids = [p['pair_id'] for p in pairs]
    R.pop(S3)
    conds = {c['vctk2k']: c for c in S.size_conditions(S.Dev(R, pairs), ids, {'choice': 'C1'}, S2)}
    assert all(conds[f'K{i}']['vc'] is None and 'no rows' in conds[f'K{i}']['error'] for i in (1, 3, 5, 7))
    R[S1][ids[0]] = dict(R[S1][ids[0]], **{W: F.wer_counts(1, 99)})    # other reference words than C1's
    with pytest.raises(ValueError, match='reference words'):
        S.vc_inputs(S.Dev(R, pairs), SY['C1'], S1, ids, 'wer', 'wer')


@pytest.mark.parametrize('change, within, identical', [
    (None, True, True),
    (('sim_wavlm', 0.0005), True, False),
    (('sim_wavlm', 0.002), False, False),       # |ΔSIM| <= 0.001 (§8.1 item 9)
    (('utmos', 0.02), False, False),            # |ΔUTMOS| <= 0.01
    (('hyp', None), False, False),              # Whisper hypotheses must be identical
])
def test_c1_identity_check(change, within, identical):
    pairs, R = small()
    ids = [p['pair_id'] for p in pairs]
    twin = SY['C1_anchor']
    if change:
        key, dv = change
        r = R[twin][ids[5]]
        R[twin][ids[5]] = (dict(r, **{W: dict(r[W], hyp='another hypothesis')}) if key == 'hyp'
                           else dict(r, **{key: r[key] + dv}))
    ck = S.c1_check(S.Dev(R, pairs), ids)
    assert ck['compared'] and ck['n'] == len(ids) and ck['within_tolerance'] is within and ck['identical'] is identical


def test_gender_table():
    pairs, R = small()
    ids = [p['pair_id'] for p in pairs]
    for g in S.GENDERS:                         # a loop in every cell: capped and uncapped WER differ
        F.set_wer(R, SY['C2'], next(p['pair_id'] for p in pairs if p['genders'] == g and p['wer_primary']), 50)
    dev = S.Dev(R, pairs)
    tab = S.gender_table(dev, ids)
    assert list(tab) == ['S1', *S.CANDS] and all(list(v) == list(S.GENDERS) for v in tab.values())
    for g in S.GENDERS:
        sub = [p['pair_id'] for p in pairs if p['genders'] == g]
        row = tab['C2'][g]
        assert (row['n'], row['n_wer_primary']) == (len(sub), sum(p['wer_primary'] for p in pairs if p['genders'] == g))
        assert row['wer'] < row['wer_u']
        for k, metric in (('wer', 'wer_capped'), ('wer_u', 'wer'), ('sim', 'sim_wavlm'), ('utmos', 'utmos')):
            prim = [k2 for k2 in sub if dev.info[k2]['wer_primary']] if k.startswith('wer') else sub
            assert row[k] == stats.point(R[SY['C2']], prim, metric)[0]
        assert row['sim_src'] == stats.point(R[SY['C2'] + S.SRC], sub, 'sim_wavlm')[0]


def test_listening_draw_seeded_order_and_items(tmp_path):
    v, lt = F.vctk_pairs(10), F.ltc_pairs()
    pv, pl = F._jsonl(str(tmp_path / 'v.jsonl'), v), F._jsonl(str(tmp_path / 'l.jsonl'), lt)
    d1 = S.listening_draw({'vctk2k': (v, pv), 'ltc2k': (lt, pl)}, '10', str(tmp_path / 'a.json'))
    rnd = random.Random(0)
    d2 = S.listening_draw({'vctk2k': (rnd.sample(v, len(v)), pv), 'ltc2k': (rnd.sample(lt, len(lt)), pl)}, '10',
                          str(tmp_path / 'b.json'))
    assert d1['sha256'] == d2['sha256'] == sha(tmp_path / 'a.json')       # input row order does not matter
    doc = json.load(open(tmp_path / 'a.json'))
    assert d1['n_items'] == len(doc['items']) == 32 and doc['seed'] == stats.SEED and doc['vctk2k_variant'] == '10'
    for name, rows in (('vctk2k', v), ('ltc2k', lt)):
        assert sum(i['set'] == name for i in doc['items']) == 16
        for g in S.GENDERS:
            order = sorted(p['pair_id'] for p in rows if p['genders'] == g)
            random.Random(f'20260928:listening:{name}:{g}').shuffle(order)     # make_pairs.rng(seed, *tags)
            cell = [i for i in doc['items'] if (i['set'], i['genders']) == (name, g)]
            assert doc['sets'][name]['order'][g] == order and len(order) > 4
            assert [i['pair_id'] for i in cell] == order[:4] and [i['rank'] for i in cell] == [0, 1, 2, 3]
    assert doc['sets']['vctk2k']['order'] != S.listening_order(v, 'ltc2k')          # the set name seeds the stream


def test_evidence_readers(tmp_path):
    p = tmp_path / 'sacct.txt'
    p.write_text('JobID|JobName|Submit|End|State\n3010_0|d1|2026-10-21T10:00|2026-10-21T20:00|COMPLETED\n'
                 '3010_1|d1|2026-10-21T09:00|2026-10-21T21:00|FAILED\n30100|x|2026-10-01T00:00|2026-10-30T00:00|X\n')
    j = S.read_sacct([str(p)], {'D1': '3010', 'D2': '9'})
    assert (j['D1']['submit'], j['D1']['end'], j['D1']['state']) == ('2026-10-21T09:00', '2026-10-21T21:00',
                                                                    ['COMPLETED', 'FAILED'])
    assert j['D2']['submit'] is None and j['D2']['state'] is None
    e = tmp_path / 'E.json'
    e.write_text(json.dumps({'a': {'paths': ['x']}, 'b': {'paths': []}, 'c': ['y'], 'd': []}))
    assert S.source_e_speakers(str(e)) == {'a', 'c'}
    r = tmp_path / 'r.json'
    r.write_text(json.dumps({'passed': False, 'summary': 's'}))
    assert S.read_repro([str(r)])[0] | {'content': None} == {
        'path': str(r), 'sha256': sha(r), 'passed': False, 'summary': 's', 'content': None}
    ren = {'r3a_x': 'cand_x', 'a@src': 'b@src'}           # a name without @src renames its @src rows too
    assert [S.renamed(x, ren) for x in ('r3a_x', 'r3a_x@src', 'a@src', 'a', 'z')] == [
        'cand_x', 'cand_x@src', 'b@src', 'a', 'z']


def test_load_metrics_refuses_a_row_in_two_files(tmp_path):
    row = {'pair_id': 'p', 'system': 'anc_faiss', 'set': 'dev', 'budget': 30}
    a = F._jsonl(str(tmp_path / 'a.jsonl'), [row])
    b = F._jsonl(str(tmp_path / 'b.jsonl'), [dict(row, system='old')])
    rows, files = S.load_metrics([a], {})
    assert files == [{'path': a, 'sha256': sha(a), 'rows': 1}] and list(rows) == [('p', 'anc_faiss')]
    with pytest.raises(SystemExit):
        S.load_metrics([a, b], {'old': 'anc_faiss'})


# ── main(): the selection record ─────────────────────────────

def run(mp, opts, **over):
    mp.setattr(sys, 'argv', F.argv(opts, **over))
    code = 0
    try:
        S.main()
    except SystemExit as e:
        code = e.code
    out = over.get('out', opts['--out'])
    return code, json.load(open(os.path.splitext(out)[0] + '.json')), open(out).read()


def quick(mp):
    """B = 400 and a fixed variance-component fit (raising, as the real one does, without data), for runs that test
    the record rather than the statistics."""
    def vc(d, src, tgt):
        if not len(d):
            raise ValueError('no data')
        return dict(VC, **stats.design_counts(src, tgt))
    mp.setattr(stats, 'bootstrap', functools.partial(stats.bootstrap, B=400))
    mp.setattr(stats, 'B', 400)
    mp.setattr(stats, 'variance_components', vc)


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    """One full-size run with every input (B = 10,000, real fits). Failed rows: C2 on Dev rows 3 and 40, C4@src on
    row 7 (its target-side row is fine), S1@src on row 60 (40 and 60 are screen rows), E17 (not required) on row 5."""
    pairs = F.dev_pairs(1000)
    R = F.system_rows(pairs, F.profiles())
    for s, k in ((SY['C2'], 3), (SY['C2'], 40), (SY['C4'] + S.SRC, 7), (S1 + S.SRC, 60), (SY['E17'], 5)):
        F.fail(R, s, pairs[k]['pair_id'])
    paths, opts = F.write_run(tmp_path_factory.mktemp('complete'), pairs, R)
    calls = []
    rule = S.crt_al_rule
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, 'crt_al_rule', lambda dev, pids, s2, **kw: calls.append((list(pids), kw)) or rule(
            dev, pids, s2, **kw))
        code, rec, md = run(mp, opts)
    return {'pairs': pairs, 'paths': paths, 'opts': opts, 'code': code, 'rec': rec, 'md': md, 'calls': calls}


def test_complete_run_has_no_problem(complete):
    rec = complete['rec']
    assert complete['code'] == 0 and rec['status'] == 'complete' and rec['problems'] == []
    assert '**Status: complete**' in complete['md'] and f'B = {stats.B:,}' in complete['md']
    ex = {s: n for s, n in rec['inputs']['common']['excluded'].items() if n}
    assert rec['inputs']['common']['n'] == 996 and ex == {SY['C2']: 2, SY['C4'] + S.SRC: 1, S1 + S.SRC: 1}
    assert rec['e17']['n_pairs'] == 995                 # E17 is not required: its failure only leaves it out
    # C3: the lowest WER (so C_m, in every T) and the highest SIM of the eligible C1-C3
    assert rec['selection']['choice'] == 'C3' and rec['selection']['fallback'] is None


def test_950_pair_check_leaves_out_exactly_the_screen_rows(complete):
    ids = [p['pair_id'] for p in complete['pairs']]
    common = [k for i, k in enumerate(ids) if i not in (3, 7, 40, 60)]     # §4.1: target-side and @src rows
    (p_all, kw_all), (p_u, kw_u), (p_chk, kw_chk) = complete['calls']
    assert p_all == p_u == common and kw_all == {'levels': True} and kw_u == {'wer': 'wer_u'} and kw_chk == {}
    assert p_chk == [k for i, k in enumerate(ids) if i % S.SCREEN_MOD and i not in (3, 7)] and len(p_chk) == 948
    rec = complete['rec']
    assert rec['check_950']['n_pairs'] == 948 and rec['inputs']['screen_n'] == 50
    assert rec['inputs']['common']['n_screen'] == 48


def test_record_holds_the_sha256_of_every_input(complete):
    rec, p = complete['rec'], complete['paths']
    assert [(f['path'], f['sha256']) for f in rec['inputs']['files']] == [(x, sha(x)) for x in p['metrics']]
    assert rec['inputs']['pairs']['sha256'] == sha(p['pairs'])
    ev = {e['path']: e['sha256'] for e in rec['inputs']['evidence']}
    want = [p['sacct'], p['repro_ok'], p['baselines'], p['vctk5'], p['vctk10'], p['ltc']] + [
        os.path.join(os.path.dirname(p[k]), f) for k in ('vctk5', 'vctk10')
        for f in ('summary.json', 'vctk2k_source_E.json')]
    assert set(ev) == set(want) and all(ev[x] == sha(x) for x in want)
    here = os.path.dirname(os.path.abspath(S.__file__))
    assert rec['scripts'] == {f: sha(os.path.join(here, f)) for f in ('select_crt_al.py', 'stats.py')}
    assert rec['listening']['sha256'] == sha(rec['listening']['path'])
    assert rec['inputs']['files'][0]['sha256'] in complete['md']


def test_listening_items_come_from_the_variant_the_size_check_picks(complete):
    rec, p = complete['rec'], complete['paths']
    rb = rec['size']['rebuild']
    assert rb['evaluable'] and rb['use'] in ('5', '10')
    by = {(r['endpoint'], r['design']): r for r in rec['size']['rows']}
    assert rb['triggers'] == [n for n in rb['conditions']
                              if by[n, 'vctk2k_5']['pass'] is False and by[n, 'vctk2k_10']['pass'] is True]
    vpath = p['vctk10' if rb['use'] == '10' else 'vctk5']
    ls = rec['listening']
    assert ls['vctk2k_variant'] == rb['use'] and ls['sets']['vctk2k']['pairs'] == vpath and ls['n_items'] == 32
    doc = json.load(open(ls['path']))
    want = S.listening_order(S.read_jsonl(vpath), 'vctk2k')
    assert doc['sets']['vctk2k']['order'] == want and doc['sets']['ltc2k']['order'] == S.listening_order(
        S.read_jsonl(p['ltc']), 'ltc2k')
    assert [i['pair_id'] for i in doc['items'] if i['set'] == 'vctk2k'] == [k for g in S.GENDERS for k in want[g][:4]]


def test_size_check_in_the_record(complete):
    rec = complete['rec']
    conds = rec['size']['conds']
    assert [c['vctk2k'] for c in conds] == ['P1', 'P3', 'P5', 'P7', 'K1', 'K3', 'K5', 'K7', 'L1s', 'L1c']
    assert all(c['vc'] and c['vc']['method'] in c['vc']['reml_llf'] for c in conds)
    assert conds[0]['wer_capped'] is True and conds[0]['a'] == SY[rec['selection']['choice']]
    rows = {(r['endpoint'], r['design']): r for r in rec['size']['rows']}
    v5 = S.read_jsonl(complete['paths']['vctk5'])
    no_e = sorted({p['source_speaker'] for p in v5})[0]       # write_run gives every other source an E set
    same = [p for p in v5 if stats.stratum(p['genders']) == 'same']
    assert rows['L1s', 'vctk2k_5']['n_left_out_no_source_E'] == sum(p['source_speaker'] == no_e for p in same) > 0
    assert rows['P2', 'ltc2k']['n'] == len(S.read_jsonl(complete['paths']['ltc']))
    assert rec['size']['wer_primary']['vctk2k_10'] == {'counted': 160, 'summary': 160}


def test_missing_required_system_makes_the_record_incomplete(complete, tmp_path, monkeypatch):
    pairs = complete['pairs']
    R = F.system_rows(pairs, F.profiles())
    R.pop(SY['C4'])
    R.pop(SY['C4'] + S.SRC)
    _, opts = F.write_run(tmp_path, pairs, R)
    quick(monkeypatch)
    code, rec, md = run(monkeypatch, opts)
    assert code == 0 and rec['status'].startswith('INCOMPLETE') and '**Status: INCOMPLETE' in md
    assert rec['inputs']['missing'] == [SY['C4'], SY['C4'] + S.SRC]
    assert rec['problems'] == [f"2 required systems have no rows (left out of the common set): "
                               f"{[SY['C4'], SY['C4'] + S.SRC]}"]
    assert rec['selection']['candidates'] == ['C1', 'C2', 'C3', 'C5'] and rec['selection']['missing'] == ['C4']
    assert rec['inputs']['common']['n'] == 1000


@pytest.mark.parametrize('drop, problem', [
    ('vctk10_pairs', '10-target VCTK-2k list not given: the rebuild rule cannot be applied'),
    ('baselines', 'baselines not given (--baselines)'),
    ('repro', 'reproduction check not given (--repro)'),
    ('ltc_pairs', 'test pair lists not all given (--vctk_pairs, --ltc_pairs)'),
])
def test_missing_input_makes_the_record_incomplete(complete, monkeypatch, drop, problem):
    quick(monkeypatch)
    out = os.path.join(os.path.dirname(complete['paths']['out']), drop, 'RECORD.md')
    code, rec, _ = run(monkeypatch, complete['opts'], out=out, **{drop: None})
    assert code == 0 and rec['status'].startswith('INCOMPLETE') and rec['problems'] == [problem]
    if drop == 'vctk10_pairs':      # no rebuild decision: the items come from the 5-target list
        assert rec['size']['rebuild']['use'] is None and rec['listening']['vctk2k_variant'] == '5'
        assert rec['listening']['sets']['vctk2k']['pairs'] == complete['paths']['vctk5']
        assert any('could not choose' in n for n in rec['listening']['notes'])


@pytest.mark.parametrize('n, no_rows', [(200, 0), (1000, 10)])
def test_record_is_incomplete_without_the_full_dev(tmp_path, monkeypatch, n, no_rows):
    pairs = F.dev_pairs(n)
    R = F.system_rows(pairs, F.profiles())
    for rows in R.values():                 # the first no_rows pairs have no row of any system
        for p in pairs[:no_rows]:
            rows.pop(p['pair_id'])
    _, opts = F.write_run(tmp_path, pairs, R)
    quick(monkeypatch)
    code, rec, _ = run(monkeypatch, opts)
    assert code == 0 and rec['problems'] == [f'rows cover {n - no_rows} of the {n} --pairs pairs (full Dev: 1,000)']
    assert rec['inputs']['common']['n_no_rows'] == no_rows and rec['inputs']['common']['n'] == n - no_rows


def twin_off(p, tmp):
    rows = S.read_jsonl(p['metrics'][0])         # the anchor-lab file holds C1's twin
    k = next(i for i, r in enumerate(rows) if r['system'] == SY['C1_anchor'] and 'error' not in r)
    rows[k] = dict(rows[k], sim_wavlm=rows[k]['sim_wavlm'] + 0.002)
    return {'metrics': ','.join([F._jsonl(str(tmp / 'd1.jsonl'), rows)] + p['metrics'][1:])}


def bad_baselines(p, tmp):
    return {'baselines': F._json(str(tmp / 'bl.json'), [{'id': 'S5', 'system': 'kNN-VC'}])}


def no_source_e(p, tmp):
    return {'vctk_pairs': F._jsonl(str(tmp / 'v5' / 'vctk2k.jsonl'), S.read_jsonl(p['vctk5']))}


@pytest.mark.parametrize('over, problem', [
    (lambda p, tmp: {'jobs': '2634=2634,D0=3001,D1=3010,D2=9999'}, "job IDs not in the --sacct files: ['9999']"),
    (lambda p, tmp: {'jobs': '2634=2634,D0=3001,D1=3010'},
     'job IDs / sacct times of 2634, D0, D1, D2 not all given (--jobs, --sacct)'),
    (no_source_e, 'vctk2k_5: no source-E file: gate L counts assume every source speaker has an E set'),
    (bad_baselines, f"--baselines: an entry lacks one of {list(S.BASELINE_KEYS)} (§2.4, §4.6)"),
    (twin_off, f"C1 identity check: `{SY['C1']}` differs from `{SY['C1_anchor']}` beyond the §8.1 item 9 tolerance"),
])
def test_other_causes_of_an_incomplete_record(complete, tmp_path, monkeypatch, over, problem):
    quick(monkeypatch)
    code, rec, _ = run(monkeypatch, complete['opts'], out=str(tmp_path / 'RECORD.md'),
                       **over(complete['paths'], tmp_path))
    assert code == 0 and rec['status'].startswith('INCOMPLETE') and rec['problems'] == [problem]


def test_rename_and_ignored_rows(complete, tmp_path, monkeypatch):
    p = complete['paths']
    rows = [dict(r, system=r['system'].replace(SY['C1'], 'r3a_pm_rq_T1')) for r in S.read_jsonl(p['metrics'][1])]
    extra = [dict(rows[0], system='faiss_b3', budget=3), dict(rows[0], system='faiss_vctk', set='vctk2k'),
             dict(rows[0], system=S1, pair_id='dev__x_9999__y')]       # (pair_id, system) keys of their own
    d2 = F._jsonl(str(tmp_path / 'd2.jsonl'), rows + extra)
    quick(monkeypatch)
    code, rec, _ = run(monkeypatch, complete['opts'], out=str(tmp_path / 'RECORD.md'),
                       metrics=','.join([p['metrics'][0], d2, p['metrics'][2]]), rename='r3a_pm_rq_T1=' + SY['C1'])
    assert code == 0 and rec['status'] == 'complete' and rec['rename'] == {'r3a_pm_rq_T1': SY['C1']}
    assert rec['inputs']['ignored'] == {'other set or budget': 2, 'pair_id not in --pairs': 1}
    assert rec['inputs']['common'] == complete['rec']['inputs']['common']
    assert rec['selection']['choice'] == complete['rec']['selection']['choice']


@pytest.mark.parametrize('scenario, fallback, choice', [('f1', 'F1', 'C5'), ('f2', 'F2', 'C1')])
def test_fallback_records(tmp_path, monkeypatch, scenario, fallback, choice):
    pairs = F.dev_pairs(1000)
    _, opts = F.write_run(tmp_path, pairs, F.system_rows(pairs, F.profiles(scenario)))
    quick(monkeypatch)
    code, rec, md = run(monkeypatch, opts)
    sel = rec['selection']
    assert code == 0 and rec['status'] == 'complete' and (sel['fallback'], sel['choice']) == (fallback, choice)
    assert sel['eligible'] == [] and f'**Choice: {choice}** (fallback {fallback})' in md
    assert 'not applicable: no candidate is eligible' in md and f'- CRT-AL: {choice}' in md
    assert [c['vctk2k'] for c in rec['size']['conds']][-2:] == ['L1s', 'L1c']
    if fallback == 'F1':
        assert sel['deficits']['E1'] < 0 and sel['flags']['C4']['E3'] is False
    else:
        assert min(sel['excess'], key=sel['excess'].get) == choice


def test_failed_reproduction_check_stops_the_record(complete, monkeypatch):
    quick(monkeypatch)
    out = os.path.join(os.path.dirname(complete['paths']['out']), 'stopped', 'RECORD.md')
    code, rec, md = run(monkeypatch, complete['opts'], out=out,
                        repro=f"{complete['paths']['repro_ok']},{complete['paths']['repro_bad']}")
    assert code == 1 and rec['status'].startswith('STOPPED') and '**Status: STOPPED' in md
    assert [r['passed'] for r in rec['repro']] == [True, False] and rec['problems'] == []
    assert rec['selection']['choice'] == 'C3' and '## Decisions' in md      # still written in full


def test_uncapped_rerun_is_reported_but_never_changes_the_choice(tmp_path, monkeypatch):
    pairs = F.dev_pairs(1000)
    R = F.system_rows(pairs, F.profiles())
    prim = [p['pair_id'] for p in pairs if p['wer_primary']]
    for k in prim[:3]:
        F.set_wer(R, SY['C3'], k, 5000)        # Whisper loops: C3's uncapped WER rises far above S2's
    _, opts = F.write_run(tmp_path, pairs, R)
    quick(monkeypatch)
    code, rec, md = run(monkeypatch, opts)
    sel, su = rec['selection'], rec['selection_uncapped']
    assert code == 0 and sel['choice'] == 'C3' and sel['c_m'] == 'C3'
    assert su['flags']['C3']['E4'] is False and su['choice'] not in (None, 'C3')
    assert sel['flags']['C3']['E4'] is True and sel['e4']['C3']['wer_u']['S2']['hi'] > 0     # both printed
    assert f"the §4.4 rule picks {su['choice']}" in md and 'capped WER picks C3, and CRT-AL stays as chosen' in md
    assert '- CRT-AL: C3' in md and all(c['a'].startswith(SY['C3']) for c in rec['size']['conds'])


def test_f3_record_reverts_to_family_v(tmp_path, monkeypatch):
    pairs = F.dev_pairs(1000)
    _, opts = F.write_run(tmp_path, pairs, F.system_rows(pairs, F.profiles('f3')))
    quick(monkeypatch)
    code, rec, md = run(monkeypatch, opts)
    assert code == 0 and rec['status'] == 'complete'
    assert rec['selection']['fallback'] == 'F3' and rec['selection']['choice'] is None
    assert [c['vctk2k'] for c in rec['size']['conds']] == ['V1', 'V3', 'V5', 'V7']
    assert rec['size']['conds'][0]['wer_capped'] is False and 'V = S2 − S1 only' in md
    assert rec['listening']['n_items'] == 32
