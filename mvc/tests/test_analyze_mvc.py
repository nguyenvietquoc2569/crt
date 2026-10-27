"""Tests of mvc/analysis/analyze_mvc.py, the one-shot T1 statistics of mvc/ANALYSIS_PLAN_mvc_v1.md (§5.8 list): NL
never through gate_l_strata and target_copy never failing on the pairs with P; the removal of the LTC-2k 180 s
shortfall pairs; the NI p-value and its agreement with the CI rule; the Holm sizes 6, 24, 12, 12 and 6; the §7.1
outcome table; the ECAPA coverage against a direct computation; refusals on duplicate rows, a missing m.done and a wrong
sha256. A synthetic world of metrics rows with known effects stands in for T1 (no MimiFlow-VC test row exists). Where
the journal's result rows are on this Mac (paper/journal_extension/results/v1), the loaders and the plan's reference
values (§3.3, §3.6, §6.1) are checked on them as well."""
import base64
import copy
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mvc' / 'analysis'))
import analyze_mvc as am  # noqa: E402  (puts benchmark/ on sys.path)
import analyze_tests as at  # noqa: E402
import stats  # noqa: E402

MVC, Q32, COPY, CRT = am.MVC, am.Q32, am.COPY, am.CRT
V1 = ROOT / 'paper' / 'journal_extension' / 'results' / 'v1'


# ── synthetic world ──────────────────────────────────────────

def gender(i):
    return 'FM'[i % 2]


def r_block(short180=False):
    return {str(b): {'budget_s': b, 'shortfall': bool(short180 and b == 180)} for b in (3, 30, 180)}


def vctk_pairs(n=8, utts=4, per_utt=3):
    """Sources s00.. and targets t00.. are disjoint; odd utterances are shared-text (P, not wer_primary)."""
    out = []
    for i in range(n):
        for u in range(utts):
            for j in range(per_utt):
                t = (i + u + j) % n
                shared = u % 2 == 1
                out.append({'pair_id': f'vctk2k__s{i:02d}_{u:03d}__t{t:02d}', 'set': 'vctk2k',
                            'source_speaker': f's{i:02d}', 'target_speaker': f't{t:02d}',
                            'genders': f'{gender(i)}2{gender(t + 1)}', 'wer_primary': not shared,
                            'P': f'/x/t{t:02d}_{u:03d}.flac' if shared else None, 'R': r_block()})
    return out


SHORT_TARGET = '0005'


def ltc_pairs(n=8, utts=3, per_utt=3):
    """Linked: every speaker is a source and a target; the pairs of target 0005 have a 180 s shortfall."""
    out = []
    for i in range(n):
        for u in range(utts):
            for j in range(per_utt):
                t = (i + 1 + (u + j) % (n - 1)) % n
                out.append({'pair_id': f'ltc2k__{i:04d}_{u:03d}__{t:04d}', 'set': 'ltc2k',
                            'source_speaker': f'{i:04d}', 'target_speaker': f'{t:04d}',
                            'genders': f'{gender(i)}2{gender(t)}', 'wer_primary': True, 'P': None,
                            'R': r_block(f'{t:04d}' == SHORT_TARGET)})
    return out


def pack(v):
    return base64.b64encode(np.asarray(v, '<f4').tobytes()).decode()


# rate: word error rate; src: SIM_src; e / e_src: ECAPA SIM / SIM_src; ovrl: DNSMOS OVRL
BASE = {MVC: dict(rate=0.03, ins=0, sim=0.90, utmos=3.8, mcd=7.0, src=0.66, e=0.62, e_src=0.20, ovrl=3.2),
        Q32: dict(rate=0.04, ins=0, sim=0.89, utmos=3.7, mcd=7.1, src=0.66, e=0.60, e_src=0.20, ovrl=3.1),
        COPY: dict(rate=0.01, ins=0, sim=0.70, utmos=4.0, mcd=9.0, src=0.95, e=0.10, e_src=0.80, ovrl=3.3),
        CRT: dict(rate=0.20, ins=0, sim=0.80, utmos=2.6, mcd=9.4, src=0.68, e=0.15, e_src=0.70, ovrl=2.5),
        'faiss': dict(rate=0.9, ins=3, sim=0.90, utmos=1.8, mcd=8.8, src=0.70, e=0.50, e_src=0.30, ovrl=2.0),
        'knnvc': dict(rate=0.30, ins=0, sim=0.85, utmos=2.9, mcd=8.0, src=0.70, e=0.55, e_src=0.25, ovrl=2.8),
        'freevc': dict(rate=0.10, ins=0, sim=0.75, utmos=3.5, mcd=7.5, src=0.60, e=0.50, e_src=0.30, ovrl=3.0),
        'seedvc': dict(rate=0.05, ins=0, sim=0.92, utmos=3.8, mcd=7.1, src=0.74, e=0.65, e_src=0.22, ovrl=3.1),
        'facodec': dict(rate=0.05, ins=0, sim=0.88, utmos=3.6, mcd=6.8, src=0.78, e=0.40, e_src=0.35, ovrl=2.9),
        'source_copy': dict(rate=0.0, ins=0, sim=0.70, utmos=4.0, mcd=9.0, src=0.95, e=0.10, e_src=0.85, ovrl=3.3),
        'target_copy': dict(rate=0.0, ins=0, sim=0.95, utmos=4.0, mcd=0.0, src=0.66, e=0.80, e_src=0.12, ovrl=3.3),
        'mimi_copy_src': dict(rate=0.01, ins=0, sim=0.70, utmos=3.9, mcd=9.0, src=0.93, e=0.10, e_src=0.8, ovrl=3.2),
        'mimi_copy_P': dict(rate=0.01, ins=0, sim=0.93, utmos=3.9, mcd=2.0, src=0.66, e=0.75, e_src=0.12, ovrl=3.2)}
P_ONLY = ('target_copy', 'mimi_copy_P')
AT30 = tuple(BASE)
AT3 = (MVC, CRT, 'faiss', 'knnvc', 'freevc', 'seedvc', 'facodec')
AT180 = (MVC, CRT, 'faiss', 'knnvc')


def rng_for(*key):
    return np.random.default_rng(zlib.crc32('/'.join(map(str, key)).encode()))


def rows_for(pairs, system, budget=30, cfg=None, fail=(), src_shift=None):
    """metrics.py rows and @src rows of one system; at 180 s a shortfall pair gets an error row with shortfall=True
    (CRT-AL: no row, as in job M) and no @src row; a failed conversion (fail) an error row and, as lab_rows.py writes
    it, no @src row."""
    c = dict(BASE[system], **(cfg or {}))
    rng = rng_for(system, budget, pairs[0]['set'])
    tgt, src = [], []
    for p in pairs:
        if system in P_ONLY and not p['P']:
            continue
        base = {'pair_id': p['pair_id'], 'system': system, 'set': p['set'], 'budget': budget,
                'target_speaker': p['target_speaker']}
        if budget == 180 and p['R']['180']['shortfall']:
            if system != CRT:
                tgt.append(dict(base, shortfall=True, ref_s=None, error='FileNotFoundError: no wav'))
            continue
        if p['pair_id'] in fail:
            tgt.append(dict(base, error='RuntimeError: boom'))
            continue
        n = 12
        k = int(rng.binomial(n, c['rate']))
        ins = int(rng.poisson(c['ins'])) if c['ins'] else 0
        kb = min(n, k + 1)
        spk = int(p['target_speaker'][-2:]) % 8
        r = dict(base, wer_large_v3={'S': k, 'D': 0, 'I': ins, 'N': n, 'cS': 3 * k, 'cD': 0, 'cI': 2 * ins, 'cN': 60},
                 wer_base={'S': kb, 'D': 0, 'I': ins, 'N': n, 'cS': 3 * kb, 'cD': 0, 'cI': 2 * ins, 'cN': 60},
                 sim_wavlm=c['sim'] + 0.02 * rng.standard_normal(), secs_resemblyzer=0.8 + 0.01 * rng.standard_normal(),
                 utmos=c['utmos'] + 0.2 * rng.standard_normal(),
                 emb_wavlm=pack(np.eye(8)[spk] + 0.3 * rng.standard_normal(8)), ref_emb_wavlm=pack(np.eye(8)[spk]))
        if p['P']:
            r.update(mcd_db=c['mcd'] + 0.3 * rng.standard_normal(), lf0_rmse=0.2, vuv_err=0.1, f0_pcc=0.5)
        tgt.append(r)
        shift = (src_shift or {}).get(stats.stratum(p['genders']), 0.0)
        src.append(dict(base, system=system + '@src', target_speaker=p['source_speaker'],
                        sim_wavlm=c['src'] + shift + 0.02 * rng.standard_normal(),
                        secs_resemblyzer=0.6 + 0.01 * rng.standard_normal()))
    return tgt, src


def side_rows(pairs, system, kind, fail=()):
    """ECAPA (target and @src), DNSMOS or guard rows of one system at 30 s. A failed conversion (fail) gets, as
    score_ecapa.py writes it, an ECAPA error row and no @src row; dnsmos.py's error row; no guard row."""
    c = BASE[system]
    rng = rng_for(system, kind, pairs[0]['set'])
    tgt, src = [], []
    for p in pairs:
        if system in P_ONLY and not p['P']:
            continue
        base = {'pair_id': p['pair_id'], 'system': system}
        if p['pair_id'] in fail:
            if kind == 'ecapa':
                tgt.append(dict(base, set=p['set'], budget=30, error='input row: RuntimeError: boom'))
            elif kind == 'dnsmos':
                tgt.append(dict(base, error="KeyError: 'wav'"))
            continue
        if kind == 'ecapa':
            base.update(set=p['set'], budget=30)
            tgt.append(dict(base, sim_ecapa=c['e'] + 0.03 * rng.standard_normal()))
            src.append(dict(base, system=system + '@src', sim_ecapa=c['e_src'] + 0.03 * rng.standard_normal()))
        elif kind == 'dnsmos':
            o = c['ovrl'] + 0.1 * rng.standard_normal()
            tgt.append(dict(base, SIG=o + 0.3, BAK=3.9, OVRL=o, P808=o + 0.4))
        else:
            tgt.append(dict(base, hnr=12.0 + rng.standard_normal(), hnr_src=13.0, voiced_kept=0.93, f0_pcc=0.8,
                            st_sd_ratio=1.0, rms_db_diff=float(rng.standard_normal())))
    return tgt, src


def by_pid(rows):
    return {r['pair_id']: r for r in rows}


def make_world(cfg=None, fail=None, src_shift=None):
    """{set: {'pairs', 'R' (metrics rows by budget), 'E' (ECAPA), 'D' (DNSMOS), 'G' (guards)}}; cfg: {(system,
    budget): overrides}; fail: {(system, budget): pair_ids with an error row}; src_shift: {system: {stratum: +SIM_src}}."""
    out = {}
    for name, pairs in (('vctk2k', vctk_pairs()), ('ltc2k', ltc_pairs())):
        R, E, D, G = {}, {30: {}}, {30: {}}, {30: {}}
        for budget, systems in ((30, AT30), (3, AT3), (180, AT180)):
            for s in systems:
                if s in P_ONLY and name == 'ltc2k':
                    continue
                tgt, src = rows_for(pairs, s, budget, cfg=(cfg or {}).get((s, budget)),
                                    fail=(fail or {}).get((s, budget), ()), src_shift=(src_shift or {}).get(s))
                R.setdefault(budget, {})[s] = by_pid(tgt)
                R[budget][s + '@src'] = by_pid(src)
        f30 = {s: (fail or {}).get((s, 30), ()) for s in BASE}
        for s in am.ECAPA_SYSTEMS:
            if s in P_ONLY and name == 'ltc2k':
                continue
            tgt, src = side_rows(pairs, s, 'ecapa', f30[s])
            E[30][s], E[30][s + '@src'] = by_pid(tgt), by_pid(src)
        for s in am.DNSMOS_SYSTEMS:
            D[30][s] = by_pid(side_rows(pairs, s, 'dnsmos', f30[s])[0])
        for s in am.GUARD_SYSTEMS:
            G[30][s] = by_pid(side_rows(pairs, s, 'guard', f30[s])[0])
        out[name] = {'pairs': pairs, 'R': R, 'E': E, 'D': D, 'G': G}
    return out


def src_e_of(w):
    return {p['source_speaker'] for p in w['vctk2k']['pairs']}


def make_sets(w, src_e=None):
    src_e = src_e_of(w) if src_e is None else src_e
    sets = {s: am.make_set(s, w[s]['pairs'], copy.deepcopy(w[s]['R']), src_e) for s in am.SETS}
    esets = {s: am.make_set(s, w[s]['pairs'], w[s]['E'], src_e, keep=am.ECAPA_SYSTEMS) for s in am.SETS}
    dsets = {s: am.make_set(s, w[s]['pairs'], w[s]['D'], src_e, keep=am.DNSMOS_SYSTEMS) for s in am.SETS}
    gsets = {s: am.make_set(s, w[s]['pairs'], w[s]['G'], src_e, keep=am.GUARD_SYSTEMS) for s in am.SETS}
    return sets, esets, dsets, gsets


@pytest.fixture(scope='module')
def world():
    return make_world()


@pytest.fixture(scope='module')
def result(world):
    sets, esets, dsets, gsets = make_sets(world)
    return sets, am.analyse(sets, esets, dsets, gsets)


# ── frozen code and constants ────────────────────────────────

def test_frozen_journal_code_and_plan_constants():
    got = am.check_frozen()
    assert got == am.FROZEN
    assert {k: round(am.multiplier(k), 2) for k in am.POWER} == {'H1': 3.48, 'Q': 3.48, 'S': 3.92, 'NI': 3.71,
                                                                 'RB': 3.71, 'NL': 2.80, 'L': 2.80}
    assert am.ENDPOINTS == at.S_ENDPOINTS and stats.MARGIN_L == 0.01 and stats.B == 10000 and stats.SEED == 20260928


def test_check_frozen_refuses_a_changed_module(monkeypatch):
    monkeypatch.setitem(am.FROZEN, 'stats', '0' * 64)
    with pytest.raises(SystemExit, match='frozen journal code changed'):
        am.check_frozen()


# ── families ─────────────────────────────────────────────────

def test_holm_family_sizes_6_24_12_12_6(result):
    _, res = result
    F = res['families']
    assert {k: len(F[k]['hypotheses']) for k in F} == {'H1': 6, 'S': 24, 'NI': 12, 'RB': 12, 'Q': 6}
    assert {k: F[k]['m'] for k in F} == {'H1': 6, 'S': 24, 'NI': 12, 'RB': 12, 'Q': 6}
    assert F['NI']['alpha'] == 0.025 and all(F[k]['alpha'] == 0.05 for k in ('H1', 'S', 'RB', 'Q'))
    assert [h['id'] for h in F['H1']['hypotheses']] == [f'H1-{i}' for i in range(1, 7)]
    assert [(h['set'], h['quantity']) for h in F['H1']['hypotheses']] == at.S_ENDPOINTS
    assert [h['id'] for h in F['S']['hypotheses']] == [f'S-{b}-{i}' for b in am.BASELINES for i in range(1, 7)]
    assert [h['id'] for h in F['NI']['hypotheses']] == [f'NI-{b}-{i}' for b in ('seedvc', 'knnvc')
                                                        for i in range(1, 7)]
    assert [h['id'] for h in F['RB']['hypotheses']] == [f'RB-{b}-{i}' for b in (3, 180) for i in range(1, 7)]
    assert {h['id'] for h in F['S']['hypotheses'] if h.get('flag') == 'VCTK-seen'} == {'S-freevc-1', 'S-freevc-3',
                                                                                        'S-freevc-5'}
    assert all((h['sys_a'], h['sys_b']) == (Q32, MVC) for h in F['Q']['hypotheses'])
    assert res['sensitivity']['union_h1_s']['m'] == 30


def test_endpoint_subsets_metrics_and_modes(result):
    sets, res = result
    h = {x['id']: x for x in res['families']['H1']['hypotheses']}
    v = sets['vctk2k']
    assert h['H1-1']['n'] == len(v.subset('wer')) < len(v.pairs) and h['H1-3']['n'] == len(v.pairs)
    assert h['H1-1']['metric'] == 'wer_capped' and h['H1-1']['label'] == 'capped WER %'
    assert h['H1-1']['mode'] == stats.INDEPENDENT and h['H1-2']['mode'] == stats.LINKED
    # mvc_v1 is far better than CRT-AL on every synthetic endpoint
    assert all(x['reject'] and x['favours'] == MVC for x in h.values())
    assert res['outcome']['outcome'] == 'A'
    unc = res['sensitivity']['wer_uncapped']['H1']['family']['hypotheses'][0]
    assert unc['metric'] == 'wer' and unc['label'] == 'WER %'
    ind = res['sensitivity']['ltc_independent']['H1']['family']['hypotheses'][1]
    assert ind['mode'] == stats.INDEPENDENT


def test_holm_counts_an_uncomputable_hypothesis_in_m():
    def fam(ps):
        return [{'id': f'x{i}', 'p': p, 'est': 1.0, 'quantity': 'sim', 'sys_a': MVC, 'sys_b': CRT}
                for i, p in enumerate(ps)]
    one = am.holm_family('H1', fam([0.009]))
    six = am.holm_family('H1', fam([0.009] + [None] * 5))
    assert one['hypotheses'][0]['reject'] and not six['hypotheses'][0]['reject'] and six['m'] == 6
    # NI at α 0.025: p 0.003 with m 12 is rejected at 0.05 / 12 but not at 0.025 / 12
    ni = am.ni_family([{'id': f'n{i}', 'p_ni': p} for i, p in enumerate([0.003] + [0.5] * 11)])
    assert not ni['hypotheses'][0]['reject'] and stats.holm([0.003] + [0.5] * 11)[0]
    ni = am.ni_family([{'id': f'n{i}', 'p_ni': p} for i, p in enumerate([0.002] + [0.5] * 11)])
    assert ni['hypotheses'][0]['reject'] and ni['hypotheses'][0]['favours'] is None


# ── NL ───────────────────────────────────────────────────────

def test_nl_never_through_gate_l_strata(world, monkeypatch):
    sets, *_ = make_sets(world)
    v = sets['vctk2k']
    tc = v.rows('target_copy@src')
    calls = []
    real = stats.gate_l_strata

    def spy(src_a, src_b, *a, **k):
        calls.append(src_b)
        assert src_b is not tc and src_a is not tc, 'NL went through gate_l_strata'
        return real(src_a, src_b, *a, **k)
    monkeypatch.setattr(stats, 'gate_l_strata', spy)
    NL = am.nl_family(v)
    assert calls == []
    am.family_l(sets)
    assert len(calls) == 2 and all(c is v.rows('knnvc@src') or c is sets['ltc2k'].rows('knnvc@src') for c in calls)
    for g, r in NL['strata'].items():
        par = [p['pair_id'] for p in v.pairs if stats.stratum(p['genders']) == g and p['P']]
        assert r['n_pairs'] == r['n'] == len(par) and r['fail_b'] == 0 and 'worst_case' not in r
        assert r['metric'] == 'sim_wavlm' and r['mode'] == stats.INDEPENDENT
        assert r['rejects'] == (r['hi'] < 0.03)
    # why: gate_l_strata against target_copy runs over every stratum pair, so the pairs without P count as failures
    wrong = real(v.rows(MVC + '@src'), tc, v.pairs, mode=v.mode)
    assert all(wrong[g]['fail_b'] == sum(stats.stratum(p['genders']) == g and not p['P'] for p in v.pairs) > 0
               for g in stats.STRATA)


def test_target_copy_never_fails_on_the_pairs_with_p_and_no_worst_case_from_it(world):
    v0 = world['vctk2k']['pairs']
    fail = [p['pair_id'] for p in v0 if p['P']][:5]
    w = make_world(fail={(MVC, 30): fail})
    sets, *_ = make_sets(w)
    NL = am.nl_family(sets['vctk2k'])
    k = {g: sum(stats.stratum(p['genders']) == g for p in v0 if p['pair_id'] in fail) for g in stats.STRATA}
    for g, r in NL['strata'].items():
        assert r['fail_b'] == r['target_copy_failures'] == 0 and r['fail_a'] == k[g]
        if k[g]:
            wc = r['worst_case']
            assert wc['imputed_b'] == 0 and wc['imputed_a'] == k[g] and wc['worst_a'] == max(
                x['sim_wavlm'] for x in sets['vctk2k'].rows(MVC + '@src').values() if 'error' not in x)
            assert wc['rejects'] == (wc['hi'] < 0.03)
    assert NL['worst_case_all_reject'] in (True, False)
    # the summarize.py value keeps, for both terms, only the pairs where mvc_v1's row has no error (summarize.py l. 86)
    v = sets['vctk2k']
    for g in stats.STRATA:
        ok = [p['pair_id'] for p in v0 if stats.stratum(p['genders']) == g and p['pair_id'] not in fail]
        want = (np.mean([v.rows(MVC + '@src')[k]['sim_wavlm'] for k in ok])
                - np.mean([v.rows('target_copy@src')[k]['sim_wavlm'] for k in ok if v.info[k]['P']]))
        assert NL['strata'][g]['summarize_value']['est'] == pytest.approx(want)


def test_nl_leakage_row_and_summarize_value(world):
    w = make_world(src_shift={MVC: {'cross': 0.06}})
    sets, *_ = make_sets(w)
    v = sets['vctk2k']
    NL = am.nl_family(v)
    assert NL['strata']['same']['rejects'] is True and NL['strata']['cross']['rejects'] is False
    assert not NL['all_reject']
    L = am.family_l(sets)
    txt = am.leakage_text(NL, L)
    assert txt['nl'].startswith('leakage above the natural level not excluded (NL1c)') and txt['nl_in_abstract']
    # the summarize.py value: mvc_v1's mean over all the stratum's pairs minus target_copy's over the pairs with P
    for g in stats.STRATA:
        allp = [p['pair_id'] for p in v.pairs if stats.stratum(p['genders']) == g]
        par = [k for k in allp if v.info[k]['P']]
        u = NL['strata'][g]['summarize_value']
        want = (np.mean([v.rows(MVC + '@src')[k]['sim_wavlm'] for k in allp])
                - np.mean([v.rows('target_copy@src')[k]['sim_wavlm'] for k in par]))
        assert u['est'] == pytest.approx(want) and u['lo'] <= u['est'] <= u['hi'] and u['n_x'] == len(allp)
        assert NL['strata'][g]['sim_x_same_pairs']['n'] == len(par)


def test_summarize_value_keeps_target_copy_only_where_its_target_row_has_no_error(world):
    """summarize.py l. 50-52: target_copy's SIM_src counts only where target_copy's own target row has no error."""
    w = copy.deepcopy(world)
    pairs = w['vctk2k']['pairs']
    k = next(p['pair_id'] for p in pairs if p['P'] and stats.stratum(p['genders']) == 'same')
    w['vctk2k']['R'][30]['target_copy'][k] = {'pair_id': k, 'system': 'target_copy', 'error': 'boom'}
    sets, *_ = make_sets(w)
    v = sets['vctk2k']
    u = am.unpaired_nl(v, MVC, 'same')
    allp = [p['pair_id'] for p in pairs if stats.stratum(p['genders']) == 'same']
    par = [x for x in allp if v.info[x]['P'] and x != k]
    want = (np.mean([v.rows(MVC + '@src')[x]['sim_wavlm'] for x in allp])
            - np.mean([v.rows('target_copy@src')[x]['sim_wavlm'] for x in par]))
    assert u['est'] == pytest.approx(want) and u['n_target_copy'] == len(par) and u['n_x'] == len(allp)
    # the paired NL is unaffected: it reads target_copy@src, which has its row
    assert am.nl_family(v)['strata']['same']['fail_b'] == 0


# ── L ────────────────────────────────────────────────────────

def test_family_l_is_gate_l_strata_against_knnvc(result):
    sets, res = result
    L = res['l']
    assert list(L['strata']) == ['L1s', 'L1c', 'L2s', 'L2c']
    for k, r in L['strata'].items():
        assert r['rejects'] == (r['hi'] < 0.01) and r['metric'] == 'sim_wavlm'
        assert r['mode'] == (stats.INDEPENDENT if r['set'] == 'vctk2k' else stats.LINKED)
    assert L['all_reject'] is True            # mvc_v1 SIM_src 0.66 against kNN-VC 0.70
    assert res['leakage']['l'] == 'not leakier than kNN-VC (margin +0.01)'
    assert res['sensitivity']['ltc_independent_l'] == {'L2s': True, 'L2c': True}


def test_l_counts_sources_without_an_e_set(world):
    src_e = src_e_of(world) - {'s00'}
    sets, *_ = make_sets(world, src_e=src_e)
    L = am.family_l(sets)
    n0 = {g: sum(stats.stratum(p['genders']) == g and p['source_speaker'] == 's00' for p in sets['vctk2k'].pairs)
          for g in stats.STRATA}
    assert L['strata']['L1s']['n_no_source_E'] == n0['same'] and L['strata']['L1c']['n_no_source_E'] == n0['cross']
    NL = am.nl_family(sets['vctk2k'])
    assert sum(r['n_no_source_E'] for r in NL['strata'].values()) == sum(
        p['source_speaker'] == 's00' and bool(p['P']) for p in sets['vctk2k'].pairs)


# ── shortfall ────────────────────────────────────────────────

def test_shortfall_pairs_leave_every_180s_ltc_comparison(result):
    sets, res = result
    l = sets['ltc2k']
    short = {p['pair_id'] for p in l.pairs if p['target_speaker'] == SHORT_TARGET}
    assert l.shortfall == short and len(short) > 0 and sets['vctk2k'].shortfall == set()
    rb = {h['id']: h for h in res['families']['RB']['hypotheses']}
    for i in (2, 4, 6):                                                   # the LTC-2k endpoints
        h = rb[f'RB-180-{i}']
        assert h['n_pairs'] == h['n'] == len(l.pairs) - len(short)
        assert h['fail_a'] == h['fail_b'] == 0 and 'worst_case' not in h
        assert (h['budget_a'], h['budget_b']) == (180, 30)
        assert rb[f'RB-3-{i}']['n_pairs'] == len(l.pairs)                 # 3 s keeps every pair
    for h in res['descriptive']['knnvc_180s'] + [x for x in res['descriptive']['crt_3s_180s'] if x['budget_a'] == 180]:
        if h['set'] == 'ltc2k':
            assert h['n_pairs'] == len(l.pairs) - len(short) and h['fail_a'] == h['fail_b'] == 0
    bt = res['budget_table']['ltc2k'][MVC]['sim']
    assert bt[180]['n_pairs'] == len(l.pairs) - len(short) and bt[180]['fail'] == 0 and bt[30]['n_pairs'] == len(l.pairs)
    # without the removal the shortfall rows would be failed conversions and trigger a worst case
    plain = stats.compare(l.rows(MVC, 180), l.rows(MVC, 30), l.subset('all'), l.spk, 'sim_wavlm', mode=l.mode,
                          worst='min')
    assert plain['fail_a'] == len(short) and 'worst_case' in plain


def test_rbn_checks_read_the_rb_cis(result):
    _, res = result
    rb = {h['id']: h for h in res['families']['RB']['hypotheses']}
    assert [(r['id'], r['set']) for r in res['rbn']] == [(k, s) for s in am.SETS for k in ('RBN-1', 'RBN-2', 'RBN-3')]
    for r in res['rbn']:
        h = rb[r['from']]
        assert h['set'] == r['set']
        want = h['lo'] > r['bound'] if r['limit'] == 'lo' else h['hi'] < r['bound']
        assert r['passes'] == want
    assert {r['from'] for r in res['rbn'] if r['set'] == 'vctk2k'} == {'RB-3-3', 'RB-180-3', 'RB-3-1'}


# ── NI ───────────────────────────────────────────────────────

def test_ni_p_value_formula():
    reps = np.r_[np.full(9000, 0.02), np.full(1000, -0.02)]
    assert am.ni_p(reps, +1, 0.010) == (1 + 1000) / 10001        # SIM: #{Δ* <= −0.010}
    assert am.ni_p(reps, +1, 0.030) == 1 / 10001
    w = np.r_[np.full(9750, 0.5), np.full(250, 1.0)]
    assert am.ni_p(w, -1, 1.0) == 251 / 10001                       # WER: #{Δ* >= +1.0}, the boundary counts
    assert am.ni_p(None, 1, 0.01) is None


def test_ni_replicates_reproduce_stats_compare(world):
    w = make_world(fail={(MVC, 30): [p['pair_id'] for p in world['vctk2k']['pairs'][:4]]})
    sets, *_ = make_sets(w)
    v = sets['vctk2k']
    for q in ('wer', 'sim', 'utmos'):
        metric = at.Q[q][0]
        ra, rb, pids = v.quantity_rows(MVC, q), v.quantity_rows('seedvc', q), v.subset(at.Q[q][1])
        c = stats.compare(ra, rb, pids, v.spk, metric, mode=v.mode, worst=at.Q[q][3], n_ref=v.n_ref)
        est, reps = am.replicates(ra, rb, pids, v.spk, metric, v.mode)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        assert (est, lo, hi, np.std(reps, ddof=1), stats.p_two_sided(reps)) == (c['est'], c['lo'], c['hi'], c['se'],
                                                                                 c['p'])
        ok, ca, cb = am.worst_case_columns(ra, rb, pids, metric, at.Q[q][3], v.n_ref)
        est_w, reps_w = am._reps(ok, ca, cb, v.spk, metric, v.mode)
        wc = c['worst_case']
        assert len(ok) == wc['n'] and est_w == wc['est'] and np.percentile(reps_w, 2.5) == wc['lo']


def test_ni_p_agrees_with_the_ci_rule(world):
    sets, *_ = make_sets(world)
    v = sets['vctk2k']
    for q in ('wer', 'sim'):
        sign = at.Q[q][2]
        _, reps = am.replicates(v.quantity_rows(MVC, q), v.quantity_rows('seedvc', q), v.subset(at.Q[q][1]), v.spk,
                                at.Q[q][0], v.mode)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        n_checked = 0
        for d in np.quantile(-sign * reps, np.linspace(0.001, 0.999, 400)):
            delta = float(d) + 1e-12
            count = int(np.sum(sign * reps <= -delta))
            rule = am.ci_rule({'lo': lo, 'hi': hi}, sign, delta)
            if count != 250:                    # the one count where interpolated percentiles and counts can differ
                assert (am.ni_p(reps, sign, delta) <= 0.025) == rule
                n_checked += 1
        assert n_checked > 50


def test_ni_family_on_the_synthetic_world(result):
    _, res = result
    ni = {h['id']: h for h in res['families']['NI']['hypotheses']}
    for h in ni.values():
        assert h['p_ni'] is not None and h['margin'] == am.NI_MARGIN[h['quantity']]
        assert h['ci_rule'] == (h['lo'] > -h['margin'] if h['sign'] > 0 else h['hi'] < h['margin'])
        assert (h['p_ni'] <= 0.025) == h['ci_rule']
    # mvc_v1 has a far lower WER than kNN-VC and Seed-VC's level is close on SIM (0.92 against 0.90)
    assert ni['NI-knnvc-1']['reject'] and ni['NI-knnvc-1']['bound'] == 'Δ < +1'
    assert not ni['NI-seedvc-3']['reject']
    assert {h['sys_b'] for h in ni.values()} == {'seedvc', 'knnvc'}


# ── §7 ───────────────────────────────────────────────────────

def h1(spec):
    """spec per endpoint: (reject, direction): +1 favours mvc_v1, −1 CRT-AL, 0 an estimate of exactly 0, None none."""
    hs = []
    for i, ((s, q), (rej, d)) in enumerate(zip(at.S_ENDPOINTS, spec), 1):
        est = None if d is None else d * at.Q[q][2] * 0.5
        h = {'id': f'H1-{i}', 'set': s, 'quantity': q, 'sys_a': MVC, 'sys_b': CRT, 'est': est, 'reject': rej}
        h['favours'] = at.favours(h, q, (MVC, CRT)) if rej else None
        hs.append(h)
    return {'hypotheses': hs}


@pytest.mark.parametrize('spec, want', [
    ([(True, 1)] * 6, 'A'),
    ([(True, 1)] * 5 + [(False, 1)], 'A−'),
    ([(False, 1)] * 6, 'A−'),
    ([(True, -1)] + [(True, 1)] * 5, 'B'),
    ([(True, -1)] + [(False, -1)] * 5, 'B'),
    ([(True, 1)] * 5 + [(False, -1)], 'C'),
    ([(False, 1)] * 5 + [(False, -1)], 'C'),
    ([(True, 1)] * 5 + [(False, 0)], None),
    ([(True, 1)] * 5 + [(False, None)], None),
], ids=['A', 'A-minus', 'A-minus-none-rejected', 'B', 'B-over-C', 'C', 'C-none-rejected', 'zero', 'missing'])
def test_outcome_table(spec, want):
    o = am.outcome(h1(spec))
    assert o['outcome'] == want
    assert o['framing'] == am.OUTCOMES.get(want, 'no row of §7.1 applies')


def test_flags_of_the_second_models():
    def h(hid, q, b, s='vctk2k'):
        return {'id': hid, 'quantity': q, 'sys_a': MVC, 'sys_b': b, 'set': s, 'reject': True, 'favours': MVC}
    fams = {'S': {'hypotheses': [h('S-knnvc-3', 'sim', 'knnvc'), h('S-knnvc-5', 'utmos', 'knnvc'),
                                 h('S-seedvc-3', 'sim', 'seedvc')]},
            'H1': {'hypotheses': [h('H1-5', 'utmos', CRT), h('H1-3', 'sim', CRT)]}}
    ec = {'sim': [{'sys_b': 'knnvc', 'set': 'vctk2k', 'est': -0.02, 'lo': -0.03, 'hi': -0.01},
                  {'sys_b': 'seedvc', 'set': 'vctk2k', 'est': -0.02, 'lo': -0.03, 'hi': 0.001},
                  {'sys_b': CRT, 'set': 'vctk2k', 'est': -0.02, 'lo': -0.03, 'hi': -0.01}],
          'sim_src': [{'sys_b': 'knnvc', 'set': 'vctk2k', 'stratum': 'same', 'est': 0.02, 'lo': 0.01, 'hi': 0.03},
                      {'sys_b': 'knnvc', 'set': 'vctk2k', 'stratum': 'cross', 'est': 0.02, 'lo': -0.01, 'hi': 0.03}],
          'nl': {'strata': {'same': {'lo': 0.031}, 'cross': {'lo': 0.0}}}}
    dn = {'comparisons': [{'field': 'OVRL', 'sys_b': 'knnvc', 'set': 'vctk2k', 'est': -0.1, 'lo': -0.2, 'hi': -0.05},
                          {'field': 'OVRL', 'sys_b': CRT, 'set': 'vctk2k', 'est': -0.1, 'lo': -0.2, 'hi': 0.01}]}
    L = {'strata': {'L1s': {'set': 'vctk2k', 'stratum': 'same', 'rejects': True},
                    'L1c': {'set': 'vctk2k', 'stratum': 'cross', 'rejects': True}}}
    NL = {'strata': {'same': {'id': 'NL1s', 'rejects': True}, 'cross': {'id': 'NL1c', 'rejects': True}}}
    am.apply_flags(fams, L, NL, ec, dn)
    S, H = fams['S']['hypotheses'], fams['H1']['hypotheses']
    assert S[0]['second_model_flag'] == am.NOT_CONFIRMED_SPK          # ECAPA against, CI excludes 0
    assert S[1]['second_model_flag'] == am.NOT_CONFIRMED_NAT          # OVRL against, CI excludes 0
    assert 'second_model_flag' not in S[2]                             # ECAPA CI includes 0
    assert 'second_model_flag' not in H[0]                             # OVRL CI includes 0
    assert 'second_model_flag' not in H[1]                             # the ECAPA flag is family S only (§6.1)
    assert L['strata']['L1s']['second_model_flag'] == am.NOT_CONFIRMED_SPK
    assert 'second_model_flag' not in L['strata']['L1c']
    assert NL['strata']['same']['second_model_flag'] == 'not confirmed by ECAPA'
    assert 'second_model_flag' not in NL['strata']['cross']
    txt = am.leakage_text(NL, L)
    assert txt['nl'].startswith('no source-speaker leakage beyond the natural other-speaker level (+0.03, WavLM-SV)')
    assert 'NL1s: not confirmed by ECAPA' in txt['nl'] and 'L1s: ' + am.NOT_CONFIRMED_SPK in txt['l']


def test_wording_against_the_baselines():
    def s(b, st, q, rej, fav):
        return {'id': f'S-{b}-{st}-{q}', 'sys_b': b, 'set': st, 'quantity': q, 'reject': rej, 'favours': fav,
                'lo': -1, 'hi': 1}
    fams = {'S': {'hypotheses': [s('freevc', 'vctk2k', 'sim', True, MVC), s('freevc', 'ltc2k', 'sim', True, MVC),
                                 s('freevc', 'vctk2k', 'utmos', True, 'freevc'), s('seedvc', 'vctk2k', 'sim', False, None),
                                 s('knnvc', 'ltc2k', 'wer', False, None)]},
            'NI': {'hypotheses': [{'sys_b': 'seedvc', 'set': 'vctk2k', 'quantity': 'sim', 'reject': True,
                                   'bound': 'Δ > −0.01'},
                                  {'sys_b': 'knnvc', 'set': 'ltc2k', 'quantity': 'wer', 'reject': False,
                                   'bound': 'Δ < +1'}]}}
    w = [x['wording'] for x in am.wording(fams)]
    assert w == ['no claim (VCTK-seen)', 'better than freevc', 'worse than freevc',
                 'comparable to seedvc (non-inferiority, Δ > −0.01)', 'no claim; the CI is given']
    # S and NI can both reject for the same baseline, endpoint and set: both statements are given
    fams = {'S': {'hypotheses': [s('seedvc', 'vctk2k', 'wer', True, 'seedvc'), s('knnvc', 'ltc2k', 'sim', True, MVC)]},
            'NI': {'hypotheses': [{'sys_b': 'seedvc', 'set': 'vctk2k', 'quantity': 'wer', 'reject': True,
                                   'bound': 'Δ < +1'},
                                  {'sys_b': 'knnvc', 'set': 'ltc2k', 'quantity': 'sim', 'reject': True,
                                   'bound': 'Δ > −0.01'}]}}
    w = am.wording(fams)
    assert [x['wording'] for x in w] == ['worse than seedvc; comparable to seedvc (non-inferiority, Δ < +1)',
                                         'better than knnvc; comparable to knnvc (non-inferiority, Δ > −0.01)']
    assert all(x['ni'] is True for x in w)


def test_tiers_are_point_estimates(result):
    _, res = result
    t = res['tiers']
    assert t['must']['outcome_a'] and t['must']['coverage'] == res['ecapa']['coverage'][MVC]['est']
    assert t['must']['meets'] == (t['must']['coverage'] >= 0.5)
    lv = res['levels']['vctk2k'][MVC]
    assert t['should']['vctk2k']['wer'] == lv['wer']['est'] and t['should']['vctk2k']['sim'] == lv['sim']['est']
    rb = {h['id']: h for h in res['families']['RB']['hypotheses']}
    assert t['should']['ltc2k']['wer_3s_minus_30s'] == rb['RB-3-2']['est']


# ── coverage ─────────────────────────────────────────────────

def test_coverage_against_a_direct_computation(world):
    _, esets, _, _ = make_sets(world)
    e = esets['vctk2k']
    c = am.coverage(e, MVC)
    par = e.subset('par')
    x = np.array([[e.rows(n)[k]['sim_ecapa'] for n in (MVC, 'source_copy', 'target_copy')] for k in par])
    m = x.mean(0)
    assert c['n'] == len(par) and c['est'] == pytest.approx((m[0] - m[1]) / (m[2] - m[1]))
    assert (c['e_out'], c['e_src'], c['e_tgt']) == pytest.approx(tuple(m))
    # every replicate: the three weighted means from the pair weights c_src[s] c_tgt[t] of the same draws
    assert c['redraws'] == 0 and c['mode'] == stats.INDEPENDENT
    d = stats.Design([e.spk[k][0] for k in par], [e.spk[k][1] for k in par], stats.INDEPENDENT)
    wts = d.pair_weights(*d.draw(np.random.default_rng(stats.SEED), stats.B))
    mw = (wts @ x) / wts.sum(1, keepdims=True)
    reps = (mw[:, 0] - mw[:, 1]) / (mw[:, 2] - mw[:, 1])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    assert c['lo'] == pytest.approx(lo, abs=1e-12) and c['hi'] == pytest.approx(hi, abs=1e-12)
    assert c['se'] == pytest.approx(np.std(reps, ddof=1), abs=1e-12)


def test_coverage_uses_only_the_pairs_with_p_and_counts_failures(world):
    par = [p['pair_id'] for p in world['vctk2k']['pairs'] if p['P']]
    w = copy.deepcopy(world)
    for k in par[:3]:
        w['vctk2k']['E'][30][MVC][k] = {'pair_id': k, 'system': MVC, 'error': 'boom'}
    _, esets, _, _ = make_sets(w)
    c = am.coverage(esets['vctk2k'], MVC)
    assert c['n_pairs'] == len(par) and c['n'] == len(par) - 3 and c['fail'] == [3, 0, 0]


# ── row checks and refusals ──────────────────────────────────

def test_required_passes_on_a_complete_world(world):
    assert am.required(*make_sets(world)[:3]) == []


def test_required_names_what_is_missing(world):
    w = copy.deepcopy(world)
    v, l = w['vctk2k'], w['ltc2k']
    del v['R'][30]['seedvc@src']
    k = next(iter(l['R'][30][Q32]))
    del l['R'][30][Q32][k]
    par = next(p['pair_id'] for p in v['pairs'] if p['P'])
    v['R'][30]['target_copy@src'][par] = {'pair_id': par, 'system': 'target_copy@src', 'error': 'x'}
    sk = next(iter(sorted(am.shortfall_pairs(l['pairs']))))
    l['R'][180][MVC][sk] = dict(l['R'][180][MVC][sk], shortfall=False)
    del v['R'][3][MVC][next(iter(v['R'][3][MVC]))]
    del v['E'][30][MVC + '@src']
    del l['D'][30]['mimi_copy_src']
    kc = next(iter(v['R'][30][COPY]))
    del v['R'][30][COPY][kc]
    probs = am.required(*make_sets(w)[:3])
    n_v = len(v['pairs'])
    want = [f'vctk2k seedvc@src 30 s: no row on {n_v} of the {n_v} pairs whose target row has no error',
            f'ltc2k {Q32} 30 s: {len(l["pairs"]) - 1} rows', f'vctk2k {COPY} 30 s: {n_v - 1} rows',
            'vctk2k target_copy@src: rows on', 'ltc2k mvc_v1 180 s: shortfall rows on', 'vctk2k mvc_v1 3 s:',
            f'vctk2k ECAPA mvc_v1@src: no row on {n_v} of the {n_v}', 'ltc2k DNSMOS mimi_copy_src: 0 rows']
    for x in want:
        assert any(p.startswith(x) for p in probs), (x, probs)


def test_required_accepts_failures_as_lab_rows_and_score_ecapa_write_them(world):
    """§5.4, §8.3: a failed conversion has an error target row and no @src row (lab_rows.py, score_ecapa.py); it is a
    failure, never a refusal. An @src row is still required on every pair whose target row has no error."""
    vp = [p['pair_id'] for p in world['vctk2k']['pairs'] if p['P']][:2]
    lp = [p['pair_id'] for p in world['ltc2k']['pairs']][:2]
    w = make_world(fail={(MVC, 30): vp + lp, (Q32, 30): vp[:1], (COPY, 30): lp[:1], ('seedvc', 30): vp[1:],
                         (MVC, 3): lp})
    assert vp[0] not in w['vctk2k']['R'][30][MVC + '@src'] and 'error' in w['vctk2k']['R'][30][MVC][vp[0]]
    assert vp[0] not in w['vctk2k']['E'][30][MVC + '@src'] and 'error' in w['vctk2k']['E'][30][MVC][vp[0]]
    assert am.required(*make_sets(w)[:3]) == []
    # a pair whose target row has no error and no @src row is a gap, in the metrics rows and in the ECAPA rows
    ok = [k for k in w['ltc2k']['R'][30][Q32 + '@src']][0]
    del w['ltc2k']['R'][30][Q32 + '@src'][ok]
    del w['vctk2k']['E'][30][COPY + '@src'][next(iter(w['vctk2k']['E'][30][COPY + '@src']))]
    probs = am.required(*make_sets(w)[:3])
    n_v, n_l = len(w['vctk2k']['pairs']), len(w['ltc2k']['pairs'])
    assert sorted(probs) == sorted([
        f'ltc2k {Q32}@src 30 s: no row on 1 of the {n_l} pairs whose target row has no error',
        f'vctk2k ECAPA {COPY}@src: no row on 1 of the {n_v} pairs whose target row has no error']), probs


def test_required_src_rows_follow_the_source_e_sets(world):
    """VCTK-2k pairs whose source has no E set get no @src row (lab_rows.py) and need none; an @src row there means
    another source-E file."""
    src_e = src_e_of(world) - {'s00'}
    w = copy.deepcopy(world)
    drop = {p['pair_id'] for p in w['vctk2k']['pairs'] if p['source_speaker'] == 's00'}
    for R in (w['vctk2k']['R'][30], w['vctk2k']['E'][30]):
        for name, rows in R.items():
            if name.endswith('@src') and name != 'target_copy@src':
                for k in drop & set(rows):
                    del rows[k]
    sets, esets, dsets, _ = make_sets(w, src_e=src_e)
    assert am.required(sets, esets, dsets) == []
    sets, esets, dsets, _ = make_sets(world, src_e=src_e)
    probs = am.required(sets, esets, dsets)
    assert probs and all(p.endswith(f'rows on {len(drop)} pairs whose source has no E set') for p in probs)
    assert any(p.startswith(f'vctk2k {MVC}@src 30 s') for p in probs)
    assert any(p.startswith(f'vctk2k ECAPA {MVC}@src') for p in probs)


def test_merge_refuses_a_row_found_twice():
    a = {30: {'knnvc': {'p1': {'pair_id': 'p1'}}}}
    b = {30: {'knnvc': {'p2': {'pair_id': 'p2'}}}, 3: {'knnvc': {'p1': {'pair_id': 'p1'}}}}
    assert set(am.merge([('a', a), ('b', b)])[30]['knnvc']) == {'p1', 'p2'}
    with pytest.raises(SystemExit, match=r'\(p1, knnvc, 30 s\) found twice \(b\)'):
        am.merge([('a', a), ('b', {30: {'knnvc': {'p1': {'pair_id': 'p1'}}}})])
    # whole=True (the metrics and ECAPA rows): a system at one budget comes from one source, as
    # analyze_exploratory.load requires
    with pytest.raises(SystemExit, match='knnvc at 30 s comes from both a and b'):
        am.merge([('a', a), ('b', b)], whole=True)
    assert set(am.merge([('a', a), ('b', {3: b[3]})], whole=True)) == {30, 3}


def test_check_systems_keeps_the_t1_parts_to_their_systems():
    R = {30: {MVC: {}, MVC + '@src': {}, Q32: {}, COPY + '@src': {}}}
    assert am.check_systems(R, am.MVC_SYSTEMS, 'x') is R
    with pytest.raises(SystemExit, match=r"x at 30 s holds rows of \['mvc_copy_latent@src', 'mvc_v1_q32'\]"):
        am.check_systems(R, am.T3_SYSTEMS, 'x')
    with pytest.raises(SystemExit, match=r"holds rows of \['mvc_r1_nosng', 'mvc_r1_nosng@src'\]"):
        am.check_systems({3: {MVC: {}, 'mvc_r1_nosng': {}, 'mvc_r1_nosng@src': {}}}, am.T3_SYSTEMS, 'x')


def test_record_shas_finds_the_script_not_its_test():
    rec = {'analysis': {'mvc/analysis/analyze_mvc.py': 'a' * 64, 'mvc/tests/test_analyze_mvc.py': 'b' * 64},
           'files': [{'path': '/x/mvc/analysis/analyze_mvc.py', 'sha256': 'a' * 64}]}
    assert am.record_shas(rec, 'analyze_mvc.py') == ['a' * 64, 'a' * 64]
    assert am.record_shas(rec, 'test_analyze_mvc.py') == ['b' * 64]


# ── files and main ───────────────────────────────────────────

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def write_part(d, part, name, budget, R, systems):
    base = d / part / 'metrics' / f'{name}_{budget}'
    rows = [r for s in systems if s in R.get(budget, {}) for r in R[budget][s].values()]
    src = [r for s in systems if s + '@src' in R.get(budget, {}) for r in R[budget][s + '@src'].values()]
    write_jsonl(Path(f'{base}.jsonl'), rows)
    write_jsonl(Path(f'{base}_src.jsonl'), src)


# the parts of select_mvc.py's record that analyze_mvc.py reads
RECORD = {'status': 'complete', 'written': '2026-10-30 09:15:02', 'final_round': 'r1', 'mvc_v1': {'config': 'c0_w1.0'},
          'stages': {'r1/r2_3': {'d900': {'numbers': {'c0_w1.0': {
              'wer': 2.1, 'sim': 0.901, 'utmos': 3.70, 'n': 900, 'fail': 0,
              'nl': {'same': {'est': -0.01, 'se': 0.02}, 'cross': {'est': 0.012, 'se': 0.03}}}}}}},
          'operator': {'analyze_mvc': {'analyze_mvc.py': sha(am.__file__), 'test_analyze_mvc.py': sha(__file__)}}}


def record_md(rec):
    """The lines of select_mvc.render that analyze_mvc.py checks: the status line with the 'written' stamp, and the
    analyze_mvc sha256 values."""
    a = (rec.get('operator') or {}).get('analyze_mvc') or {}
    return (f'# MimiFlow-VC selection record v1 (ANALYSIS_PLAN_mvc_v1.md §4)\n\nStatus: **{rec.get("status")}**. '
            f'Written {rec.get("written")} by `mvc/analysis/select_mvc.py` (sha256 {"0" * 64}; tests {"1" * 64}).\n\n'
            '- analyze_mvc.py and its tests: ' + '; '.join(f'`{k}` {v}' for k, v in a.items()) + '\n')


def append_jsonl(path, rows):
    with open(path, 'a') as fo:
        fo.write(''.join(json.dumps(r) + '\n' for r in rows))


def write_inputs(tmp, w, mr_sha=None, journal_sha=am.JOURNAL_SELECTION_SHA256, record=None, md=None, skip=()):
    """The V1 and MR trees of the plan (§8.2) from a synthetic world, and the record (md: its .md text, by default
    the record's own lines; False: no .md); returns (v1, mr, record)."""
    v1, mr = tmp / 'v1', tmp / 'mr'
    jm = (CRT, 'faiss', 'source_copy', 'target_copy', 'mimi_copy_src', 'mimi_copy_P')
    for name in am.SETS:
        R, pairs = w[name]['R'], w[name]['pairs']
        main = 't1' if name == 'vctk2k' else 't2'
        write_part(v1 / 'm', main, name, 30, R, jm)
        for b in (3, 180):
            write_part(v1 / 'm', 't3', name, b, R, (CRT, 'faiss'))
            write_part(v1 / 'm', 't4', name, b, R, ('knnvc',))
            write_part(mr / 'm', 't3', name, b, R, (MVC,))
        write_part(v1 / 'm', 't4', name, 30, R, am.BASELINES)
        write_part(v1 / 'x1', 'm', name, 3, R, ('freevc', 'seedvc', 'facodec'))
        write_part(mr / 'm', main, name, 30, R, am.MVC_SYSTEMS)
        E = w[name]['E'][30]
        for d, systems in ((v1 / 'x2' / 'ecapa', (CRT, 'faiss', 'source_copy', 'target_copy') + am.BASELINES),
                           (mr / 'x2' / 'ecapa', am.MVC_SYSTEMS)):
            write_jsonl(d / f'{name}_30.jsonl', [r for s in systems if s in E for r in E[s].values()])
            write_jsonl(d / f'{name}_30_src.jsonl', [r for s in systems if s + '@src' in E
                                                     for r in E[s + '@src'].values()])
        D, G = w[name]['D'][30], w[name]['G'][30]
        write_jsonl(mr / 'dnsmos' / f'{name}_30_mvc.jsonl', [r for s in am.MVC_SYSTEMS for r in D[s].values()])
        write_jsonl(mr / 'dnsmos' / f'{name}_30_journal.jsonl',
                    [r for s in am.DNSMOS_SYSTEMS if s not in am.MVC_SYSTEMS for r in D[s].values()])
        write_jsonl(mr / 'guard' / f'{name}_30.jsonl', [r for s in am.GUARD_SYSTEMS for r in G[s].values()])
        write_jsonl(v1 / 'pairs' / f'{name}.jsonl', pairs)
    (v1 / 'pairs' / 'vctk2k_source_E.json').write_text(json.dumps({s: {'paths': ['/e.flac']} for s in src_e_of(w)}))
    for p in ('t1', 't2', 't3', 't4'):
        (v1 / 'm' / p / 'm.done').write_text(f'job 1 2026-10-20 10:00:00 selection sha256 {journal_sha}\n')
    for x in ('x1', 'x2'):
        (v1 / x / f'{x}.done').write_text(f'job 2 2026-10-21 10:00:00 selection sha256 {journal_sha}\n')
    rec_path = tmp / 'MVC_SELECTION_RECORD_v1.json'
    rec = record if record is not None else RECORD
    rec_path.write_text(json.dumps(rec))
    if md is not False:
        rec_path.with_suffix('.md').write_text(record_md(rec) if md is None else md)
    for p in ('t1', 't2', 't3'):
        if p in skip:
            continue
        (mr / 'm' / p / 'm.done').write_text(f'job 7 2026-11-02 10:00:00 selection sha256 '
                                             f'{mr_sha or sha(rec_path if md is False else rec_path.with_suffix(".md"))}'
                                             f'\n')
    return v1, mr, rec_path


@pytest.fixture
def synthetic_lists(monkeypatch, world, tmp_path):
    """Writes the inputs and points the list sha256 and the §1 counts at the synthetic lists."""
    def setup(**kw):
        v1, mr, rec = write_inputs(tmp_path, kw.pop('w', world), **kw)
        monkeypatch.setattr(am, 'PAIR_SHA256', {'vctk2k': sha(v1 / 'pairs' / 'vctk2k.jsonl'),
                                                'ltc2k': sha(v1 / 'pairs' / 'ltc2k.jsonl'),
                                                'source_e': sha(v1 / 'pairs' / 'vctk2k_source_E.json')})
        monkeypatch.setattr(am, 'EXPECTED', {'shortfall': len(am.shortfall_pairs(world['ltc2k']['pairs'])),
                                             'par': sum(bool(p['P']) for p in world['vctk2k']['pairs'])})
        return ['--v1', str(v1), '--mr', str(mr), '--record', str(rec), '--out', str(tmp_path / 'out' / 'R.md')]
    return setup


def test_main_end_to_end(synthetic_lists, tmp_path, capsys):
    argv = synthetic_lists()
    assert am.main(argv) == 0
    md = (tmp_path / 'out' / 'R.md').read_text()
    res = json.loads((tmp_path / 'out' / 'R.json').read_text())
    assert md.startswith('# MimiFlow-VC test results v1') and '**Outcome A**' in md
    for x in ('Family H1 (primary)', 'Family NL', 'Family L', 'Family S', 'Family NI', 'Family RB', 'RBN-3',
              'Family Q', 'Wording against the baselines', 'Power with the realised SEs', 'Sensitivity analyses',
              'ECAPA coverage', 'DNSMOS', 'Acoustic guards', 'Budget table', 'Gender pairs', 'selected on Dev'):
        assert x in md, x
    assert 'NL (§7.2): NL1s' in md                      # every family-S table carries the NL result in its caption
    assert res['record_sha256'] == sha(tmp_path / 'MVC_SELECTION_RECORD_v1.json')
    assert res['record_md_sha256'] == sha(tmp_path / 'MVC_SELECTION_RECORD_v1.md')
    assert res['dev900'] == {'wer': 2.1, 'sim': 0.901, 'utmos': 3.70, 'nl': {'same': -0.01, 'cross': 0.012}}
    assert 'mvc_v1 on Dev-900, selected on Dev: capped WER 2.10' in md
    # §4.9, §9: mvc_v1's Dev-900 value beside the H1, S, NI, Q and NL rows, and under each 30 s levels table
    lines = md.splitlines()
    dev = {'wer': '2.10', 'sim': '0.901', 'utmos': '3.700'}
    for fam in ('H1', 'S', 'NI', 'Q'):
        for h in res['families'][fam]['hypotheses']:
            row = next(x for x in lines if x.startswith(f'| {h["id"]} |') or x.startswith(f'| {h["id"]} ['))
            assert row.endswith(f' {dev[h["quantity"]]} |'), row
    assert all(not x.endswith(' 2.10 |') for x in lines if x.startswith('| RB-'))
    for g, want in (('same', '-0.010'), ('cross', '0.012')):
        row = next(x for x in lines if x.startswith(f'| {res["nl"]["strata"][g]["id"]} | {g} |'))
        assert row.endswith(f' {want} |'), row
    assert sum(x.endswith('| mvc_v1 on Dev-900, selected on Dev |') for x in lines) == 4      # H1, S, NI, Q heads
    assert md.count('mvc_v1 on Dev-900, selected on Dev: capped WER 2.10') == 3                 # summary, 2 levels
    assert res['record_checked'] == {'analyze_mvc.py': sha(am.__file__), 'test_analyze_mvc.py': sha(__file__)}
    assert res['frozen'] == am.FROZEN and len(res['script_sha256']) == 64
    # record (.json, .md), 2 lists, source E; per set: 6 journal M + 1 X1 + 3 T1 parts x 2 files, 4 ECAPA, 2 DNSMOS,
    # 1 guard
    assert len(res['inputs']) == 5 + 2 * (2 * 10 + 4 + 2 + 1)
    assert all(len(x['sha256']) == 64 for x in res['inputs'])
    assert set(res['done']) == {f'journal job M part t{i}' for i in range(1, 5)} | {
        'journal job X1', 'journal job X2'} | {f'MimiFlow-VC T1 part t{i}' for i in range(1, 4)}
    assert res['eer']['vctk2k'][MVC] is not None and 'crt_tau0.9' not in res['levels']['vctk2k']


def test_main_never_overwrites_an_earlier_output(synthetic_lists, tmp_path):
    argv = synthetic_lists()
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'R.json').write_text('{}')
    with pytest.raises(SystemExit, match='runs once'):
        am.main(argv)


def test_main_refuses_a_missing_m_done(synthetic_lists):
    with pytest.raises(SystemExit, match='part t2 of job M has not finished'):
        am.main(synthetic_lists(skip=('t2',)))


def test_main_refuses_m_done_under_another_record(synthetic_lists):
    with pytest.raises(SystemExit, match='MimiFlow-VC T1 parts .* name the selection sha256'):
        am.main(synthetic_lists(mr_sha='b' * 64))


def test_main_refuses_journal_rows_under_another_selection(synthetic_lists):
    with pytest.raises(SystemExit, match='journal job M parts .* name the selection sha256'):
        am.main(synthetic_lists(journal_sha='c' * 64))


def test_main_refuses_a_wrong_pair_list_sha256(synthetic_lists, monkeypatch):
    argv = synthetic_lists()
    monkeypatch.setitem(am.PAIR_SHA256, 'ltc2k', 'd' * 64)
    with pytest.raises(SystemExit, match=r'ltc2k\.jsonl \(ltc2k\) has sha256'):
        am.main(argv)


def test_main_refuses_a_record_without_or_with_another_script_sha256(synthetic_lists):
    with pytest.raises(SystemExit, match='gives no sha256 for analyze_mvc.py'):
        am.main(synthetic_lists(record={'status': 'complete'}))
    rec = dict(RECORD, operator={'analyze_mvc': {'analyze_mvc.py': 'e' * 64, 'test_analyze_mvc.py': sha(__file__)}})
    with pytest.raises(SystemExit, match='analyze_mvc.py has sha256'):
        am.main(synthetic_lists(record=rec))


def test_main_refuses_a_draft_record(synthetic_lists):
    with pytest.raises(SystemExit, match="status 'DRAFT: not the selection record', not complete"):
        am.main(synthetic_lists(record=dict(RECORD, status='DRAFT: not the selection record')))


def test_main_accepts_m_done_naming_the_record_json(synthetic_lists, tmp_path, monkeypatch):
    """m.done may name the .md's sha256 (select_mvc.py's RECORD_SHA256) or the .json's; refusals come later only."""
    argv = synthetic_lists()
    rec = tmp_path / 'MVC_SELECTION_RECORD_v1.json'
    for p in ('t1', 't2', 't3'):
        (tmp_path / 'mr' / 'm' / p / 'm.done').write_text(f'job 7 2026-11-02 selection sha256 {sha(rec)}\n')
    monkeypatch.setattr(am, 'required', lambda *x: ['stop here'])
    with pytest.raises(SystemExit, match='stop here'):
        am.main(argv)


def test_dev900_from_the_record():
    assert am.dev900(RECORD) == {'wer': 2.1, 'sim': 0.901, 'utmos': 3.70, 'nl': {'same': -0.01, 'cross': 0.012}}
    with pytest.raises(SystemExit, match='gives no Dev-900 numbers of mvc_v1'):
        am.dev900({'status': 'complete'})
    with pytest.raises(SystemExit, match='gives no Dev-900 numbers of mvc_v1'):
        am.dev900(dict(RECORD, mvc_v1={'config': 'c1_w0.5'}))
    rec = copy.deepcopy(RECORD)
    rec['stages']['r1/r2_3']['d900']['numbers']['c0_w1.0']['nl']['cross']['est'] = None
    with pytest.raises(SystemExit, match=r"Dev-900 numbers of mvc_v1 lack \['nl cross'\]"):
        am.dev900(rec)


def test_main_refuses_a_record_without_dev900_numbers(synthetic_lists):
    rec = {k: v for k, v in RECORD.items() if k != 'stages'}
    with pytest.raises(SystemExit, match='gives no Dev-900 numbers of mvc_v1'):
        am.main(synthetic_lists(record=rec))


def test_main_refuses_a_record_md_from_another_run(synthetic_lists):
    """§4.8: the .md beside the .json is from the same select_mvc.py run (status, 'written', analyze_mvc sha256)."""
    with pytest.raises(SystemExit, match=r'MVC_SELECTION_RECORD_v1\.md missing'):
        am.main(synthetic_lists(md=False))
    with pytest.raises(SystemExit, match='not from the same select_mvc.py run'):
        am.main(synthetic_lists(md=record_md(dict(RECORD, written='2026-10-30 09:15:01'))))
    with pytest.raises(SystemExit, match='not from the same select_mvc.py run'):
        am.main(synthetic_lists(md=record_md(dict(RECORD, status='DRAFT: not the selection record'))))
    other = dict(RECORD, operator={'analyze_mvc': {'analyze_mvc.py': sha(am.__file__), 'test_analyze_mvc.py': 'f' * 64}})
    with pytest.raises(SystemExit, match=r'does not name the sha256 \[.*\] of test_analyze_mvc.py'):
        am.main(synthetic_lists(md=record_md(other)))


def test_main_refuses_a_system_from_two_sources(synthetic_lists, world, tmp_path):
    """As analyze_exploratory.load: kNN-VC at 3 s from job M (t4) and from job X1 stops the run."""
    argv = synthetic_lists()
    append_jsonl(tmp_path / 'v1' / 'x1' / 'm' / 'metrics' / 'vctk2k_3.jsonl', world['vctk2k']['R'][3]['knnvc'].values())
    with pytest.raises(SystemExit, match='knnvc at 3 s comes from both journal job M and journal job X1'):
        am.main(argv)


def renamed(rows, system, **kw):
    return [dict(r, system=system + ('@src' if r['system'].endswith('@src') else ''), **kw) for r in rows]


@pytest.mark.parametrize('case', ['nosng_t1', 'knnvc_t1', 'target_copy_t1', 'crt_t3_shortfall', 'q32_t3',
                                  'ecapa_knnvc'])
def test_main_refuses_rows_of_systems_outside_the_t1_parts(case, synthetic_lists, world, tmp_path):
    """§8.2, §9: MR/m/t1 and t2 hold mvc_v1, mvc_v1_q32 and mvc_copy_latent, t3 mvc_v1 only, each with its @src twin;
    the MR ECAPA rows the three systems. Anything else (a configuration R2 did not choose, a journal system, rows for
    the shortfall pairs) stops the run instead of being dropped or merged."""
    argv = synthetic_lists()
    mr, v, l = tmp_path / 'mr', world['vctk2k'], world['ltc2k']
    if case == 'nosng_t1':
        path, rows, want = mr / 'm' / 't1' / 'metrics' / 'vctk2k_30_src.jsonl', renamed(
            v['R'][30][MVC + '@src'].values(), 'mvc_r1_nosng'), r"vctk2k MimiFlow-VC T1 part t1 at 30 s holds rows of " \
                                                               r"\['mvc_r1_nosng@src'\]"
    elif case == 'knnvc_t1':
        path, rows, want = mr / 'm' / 't2' / 'metrics' / 'ltc2k_30.jsonl', l['R'][30]['knnvc'].values(), \
            r"ltc2k MimiFlow-VC T1 part t2 at 30 s holds rows of \['knnvc'\]"
    elif case == 'target_copy_t1':
        rows = renamed([r for k, r in v['R'][30][MVC].items() if not v['R'][30].get('target_copy', {}).get(k)],
                       'target_copy')
        path, want = mr / 'm' / 't1' / 'metrics' / 'vctk2k_30.jsonl', r"holds rows of \['target_copy'\]"
    elif case == 'crt_t3_shortfall':
        rows = [dict({k: x for k, x in r.items() if k != 'error'}, system=CRT, sim_wavlm=0.9)
                for r in l['R'][180][MVC].values() if r.get('shortfall')]
        path, want = mr / 'm' / 't3' / 'metrics' / 'ltc2k_180.jsonl', \
            r"ltc2k MimiFlow-VC T1 part t3 at 180 s holds rows of \['cand_pm18_rq'\]"
    elif case == 'q32_t3':
        path, rows, want = mr / 'm' / 't3' / 'metrics' / 'vctk2k_3.jsonl', renamed(v['R'][3][MVC].values(), Q32), \
            r"vctk2k MimiFlow-VC T1 part t3 at 3 s holds rows of \['mvc_v1_q32'\]"
    else:
        path, rows, want = mr / 'x2' / 'ecapa' / 'ltc2k_30.jsonl', l['E'][30]['knnvc'].values(), \
            r"MimiFlow-VC ECAPA .*ltc2k_30\.jsonl at 30 s holds rows of \['knnvc'\]"
    assert rows
    append_jsonl(path, rows)
    with pytest.raises(SystemExit, match=want):
        am.main(argv)


def test_main_refuses_missing_rows(synthetic_lists, world):
    w = copy.deepcopy(world)
    del w['vctk2k']['R'][30][Q32 + '@src']
    with pytest.raises(SystemExit, match=f'vctk2k {Q32}@src 30 s: no row on {len(w["vctk2k"]["pairs"])} of the'):
        am.main(synthetic_lists(w=w))


def test_main_end_to_end_with_failed_conversions(synthetic_lists, world, tmp_path):
    """Failed T1 conversions as run_mvc.py, lab_rows.py, metrics.py, score_ecapa.py and dnsmos.py leave them (an error
    target row, no @src row, an ECAPA error row, a DNSMOS error row) are failures: counted, left out of every
    comparison, imputed at their worst case in the sensitivity analysis, never a refusal (§5.4, §8.3)."""
    vp = next(p['pair_id'] for p in world['vctk2k']['pairs'] if p['P'] and stats.stratum(p['genders']) == 'cross')
    lp = next(p['pair_id'] for p in world['ltc2k']['pairs'] if p['target_speaker'] != SHORT_TARGET)
    w = make_world(fail={(MVC, 30): [vp, lp], (Q32, 30): [vp], (MVC, 3): [lp], (MVC, 180): [lp]})
    assert am.main(synthetic_lists(w=w)) == 0
    res = json.loads((tmp_path / 'out' / 'R.json').read_text())
    md = (tmp_path / 'out' / 'R.md').read_text()
    h1 = {h['id']: h for h in res['families']['H1']['hypotheses']}
    assert [h1[f'H1-{i}']['fail_a'] for i in range(1, 7)] == [0, 1, 1, 1, 1, 1]   # vp is a P pair, not wer_primary
    assert all(h1[f'H1-{i}']['fail_b'] == 0 and 'worst_case' in h1[f'H1-{i}'] for i in range(2, 7))
    assert set(res['sensitivity']['worst_case']['H1']['worst_case_used']) == {f'H1-{i}' for i in range(2, 7)}
    nl = res['nl']['strata']
    assert nl['cross']['fail_a'] == 1 and nl['cross']['fail_b'] == 0 and 'rejects' in nl['cross']['worst_case']
    assert nl['same']['fail_a'] == 0 and 'worst_case' not in nl['same']
    assert res['l']['strata']['L1c']['fail_a'] == 1 and res['l']['strata']['L2s']['fail_a'] + \
        res['l']['strata']['L2c']['fail_a'] == 1
    assert res['ecapa']['nl']['strata']['cross']['fail_a'] == 1
    assert res['ecapa']['coverage'][MVC]['fail'] == [1, 0, 0]
    q = {h['id']: h for h in res['families']['Q']['hypotheses']}
    assert q['Q-3']['fail_a'] == 1 and q['Q-3']['fail_b'] == 1                    # both failed on vp: no worst case
    assert 'worst_case' not in q['Q-3'] and q['Q-4']['fail_b'] == 1
    rb = {h['id']: h for h in res['families']['RB']['hypotheses']}
    assert rb['RB-3-4']['fail_a'] == rb['RB-3-4']['fail_b'] == 1 and rb['RB-180-4']['fail_a'] == 1
    assert rb['RB-180-4']['n_pairs'] == len(w['ltc2k']['pairs']) - len(am.shortfall_pairs(w['ltc2k']['pairs']))
    assert res['dnsmos']['levels']['vctk2k'][MVC]['OVRL']['fail'] == 1
    assert res['levels']['vctk2k'][MVC]['fail'] == 1 and '**Outcome' in md


# ── the journal's own rows, where present ────────────────────

journal = pytest.mark.skipif(not (V1 / 'm' / 't1' / 'metrics' / 'vctk2k_30.jsonl').exists(),
                             reason='journal result rows not on this machine')


@pytest.fixture(scope='module')
def real_vctk():
    pairs = at.read_pairs(V1 / 'pairs' / 'vctk2k.jsonl')
    src_e = {k for k, v in json.load(open(V1 / 'pairs' / 'vctk2k_source_E.json')).items() if (v or {}).get('paths')}
    R, _ = at.load_set(str(V1 / 'm'), 'vctk2k', [('t1', 30), ('t4', 30)], pairs)
    E, _ = am.load_ecapa([('journal X2', str(V1 / 'x2' / 'ecapa'), None)], 'vctk2k', pairs)
    return (am.make_set('vctk2k', pairs, am.merge([('journal job M', R)]), src_e),
            am.make_set('vctk2k', pairs, E, src_e, keep=am.ECAPA_SYSTEMS))


@journal
def test_real_journal_layout_lists_and_done_files():
    for k, name in (('vctk2k', 'vctk2k.jsonl'), ('ltc2k', 'ltc2k.jsonl'), ('source_e', 'vctk2k_source_E.json')):
        am.check_sha(str(V1 / 'pairs' / name), am.PAIR_SHA256[k], k)
    ltc = at.read_pairs(V1 / 'pairs' / 'ltc2k.jsonl')
    short = am.shortfall_pairs(ltc)
    assert len(short) == am.EXPECTED['shortfall'] == 101
    assert {p['target_speaker'] for p in ltc if p['pair_id'] in short} == {'3729', '6930'}
    assert sum(bool(p['P']) for p in at.read_pairs(V1 / 'pairs' / 'vctk2k.jsonl')) == am.EXPECTED['par']
    lines = am.check_parts(str(V1 / 'm'), ['t1', 't2', 't3', 't4'], am.JOURNAL_SELECTION_SHA256, 'journal job M')
    assert am.ax.x1_done(str(V1 / 'x1'), lines).endswith(am.JOURNAL_SELECTION_SHA256)
    assert am.done_line(str(V1 / 'x2' / 'x2.done'), am.JOURNAL_SELECTION_SHA256, 'job X2')
    # job M wrote the 101 shortfall rows of kNN-VC at 180 s as error rows with shortfall=True
    R, _ = at.load_set(str(V1 / 'm'), 'ltc2k', [('t4', 180)], ltc)
    ts = am.make_set('ltc2k', ltc, R)
    assert {k for k, r in ts.rows('knnvc', 180).items() if r.get('shortfall')} == short == ts.shortfall
    assert len(am.endpoint_pids(ts, 'sim', (180, 30))) == 1699


@journal
def test_real_rows_reproduce_the_plans_nl_reference_values(real_vctk):
    """§3.3: paired NL on VCTK-2k, kNN-VC +0.049 / +0.055, FreeVC +0.070 / +0.100, Seed-VC +0.051 / +0.065, FACodec
    +0.095 / +0.096, CRT-AL +0.001 / −0.014; Seed-VC's summarize.py value +0.039 / +0.055; target_copy never fails."""
    ts, _ = real_vctk
    want = {'knnvc': (0.049, 0.055), 'freevc': (0.070, 0.100), 'seedvc': (0.051, 0.065), 'facodec': (0.095, 0.096),
            CRT: (0.001, -0.014)}
    for x, (s, c) in want.items():
        NL = am.nl_family(ts, x=x, beside=(x == 'seedvc'))
        r = NL['strata']
        assert (round(r['same']['est'], 3), round(r['cross']['est'], 3)) == (s, c), x
        assert (r['same']['n'], r['cross']['n']) == (491, 489) and r['same']['fail_b'] == r['cross']['fail_b'] == 0
        if x == 'seedvc':
            assert (round(r['same']['summarize_value']['est'], 3), round(r['cross']['summarize_value']['est'], 3)) \
                == (0.039, 0.055)
    assert len(ts.rows('target_copy@src')) == 980


@journal
def test_real_rows_reproduce_the_plans_coverage_and_ecapa_nl(real_vctk):
    """§3.6: coverage on the 980 pairs, CRT-AL 0.070, kNN-VC 0.83, FreeVC 0.75, Seed-VC 0.94, FACodec 0.56, FAISS 0.72;
    §6.1: the baselines' ECAPA NL +0.035 to +0.18, CRT-AL's −0.048 / −0.027."""
    _, e = real_vctk
    cov = {x: am.coverage(e, x)['est'] for x in (CRT, 'knnvc', 'freevc', 'seedvc', 'facodec', 'faiss')}
    assert {x: round(v, 2) for x, v in cov.items() if x != 'faiss'} == {CRT: 0.07, 'knnvc': 0.83, 'freevc': 0.75,
                                                                        'seedvc': 0.94, 'facodec': 0.56}
    assert abs(cov['faiss'] - 0.72) < 0.006          # 0.7146: the plan's 0.72 rounds 0.715 a second time
    assert round(am.coverage(e, CRT)['est'], 3) == 0.070
    nl = {x: am.nl_family(e, x=x, key=am.ECAPA_KEY, beside=False)['strata'] for x in (CRT,) + am.BASELINES}
    assert (round(nl[CRT]['same']['est'], 3), round(nl[CRT]['cross']['est'], 3)) == (-0.048, -0.027)
    vals = [r['est'] for x in am.BASELINES for r in nl[x].values()]
    assert round(min(vals), 3) == 0.035 and round(max(vals), 2) == 0.18
