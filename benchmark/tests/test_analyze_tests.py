"""Tests of benchmark/analyze_tests.py, the one-shot test-set statistics of ANALYSIS_PLAN_v1.md §5 and §7, on synthetic
metrics rows with known effects: the systems taken from the record, the endpoint subsets (wer_primary for WER, pairs
with P for dB-MCD), capped WER in P / K / S and uncapped in V, the bootstrap mode per set, Holm per family with the
CMOS pending, in or dropped, gate L with its source-E exclusions and the natural other-speaker check, the §7 outcome,
the sensitivities (P ∪ K, the other WER definition, LTC-2k independent, worst case) and the input checks of main.
The synthetic VCTK-2k-like list has disjoint source and target speakers; the LTC-2k-like list is linked."""
import base64
import copy
import json
import zlib

import numpy as np
import pytest

import analyze_tests as at
import stats

RECORD = {'status': 'complete', 'selection': {'choice': 'C2'}, 'tau_rule': {'tau': 0.9},
          'baselines': [{'id': 'S5'}, {'id': 'S6'}], 'size': {'rebuild': {'use': '5'}}}
SYSTEMS = at.systems_from_record(RECORD)
A, F, S3, S2 = 'cand_pm18_rq', 'faiss', 'crt_tau0.4', 'crt_tau0.9'


# ── synthetic world ──────────────────────────────────────────

def gender(i):
    return 'FM'[i % 2]


def vctk_pairs(n_src=6, n_tgt=6, utts=4, per_utt=3):
    """Sources s00.. and targets t00.. are disjoint; odd utterances are shared-text (P, not wer_primary)."""
    out = []
    for i in range(n_src):
        for u in range(utts):
            for j in range(per_utt):
                t = (i + u + j) % n_tgt
                shared = u % 2 == 1
                out.append({'pair_id': f'vctk2k__s{i:02d}_{u:03d}__t{t:02d}', 'set': 'vctk2k',
                            'source_speaker': f's{i:02d}', 'target_speaker': f't{t:02d}',
                            'genders': f'{gender(i)}2{gender(t + 1)}', 'wer_primary': not shared,
                            'P': f'/x/t{t:02d}_{u:03d}.flac' if shared else None})
    return out


def ltc_pairs(n_spk=6, utts=3, per_utt=3):
    out = []
    for i in range(n_spk):
        for u in range(utts):
            for j in range(per_utt):
                t = (i + 1 + (u + j) % (n_spk - 1)) % n_spk
                out.append({'pair_id': f'ltc2k__{i:04d}_{u:03d}__{t:04d}', 'set': 'ltc2k',
                            'source_speaker': f'{i:04d}', 'target_speaker': f'{t:04d}',
                            'genders': f'{gender(i)}2{gender(t)}', 'wer_primary': True, 'P': None})
    return out


def pack(v):
    return base64.b64encode(np.asarray(v, np.float32).tobytes()).decode()


BASE = {'faiss': dict(rate=0.9, ins=3, sim=0.90, utmos=1.8, mcd=8.8, src=0.70),
        'crt_tau0.4': dict(rate=0.8, ins=2, sim=0.89, utmos=1.7, mcd=9.0, src=0.70),
        'crt_tau0.9': dict(rate=0.88, ins=3, sim=0.90, utmos=1.8, mcd=8.85, src=0.70),
        'cand_pm18_rq': dict(rate=0.2, ins=0, sim=0.80, utmos=2.6, mcd=9.4, src=0.68),
        'knnvc': dict(rate=0.3, ins=0, sim=0.85, utmos=2.9, mcd=8.0, src=0.65),
        'freevc': dict(rate=0.1, ins=0, sim=0.75, utmos=3.5, mcd=7.5, src=0.60),
        'source_copy': dict(rate=0.0, ins=0, sim=0.70, utmos=4.0, mcd=9.0, src=0.95),
        'target_copy': dict(rate=0.0, ins=0, sim=0.95, utmos=4.0, mcd=0.0, src=0.66)}


def rows_for(pairs, system, budget=30, seed=0, cfg=None, fail=(), src_shift=None):
    """metrics.py rows (and @src rows) of one system; cfg overrides BASE; fail: pair_ids with an error row;
    src_shift: {stratum: added SIM_src}."""
    c = dict(BASE[system], **(cfg or {}))
    rng = np.random.default_rng(zlib.crc32(f'{system}/{budget}/{seed}'.encode()))
    tgt, src = [], []
    for p in pairs:
        if system == 'target_copy' and not p['P']:
            continue
        base = {'pair_id': p['pair_id'], 'system': system, 'set': p['set'], 'budget': budget,
                'target_speaker': p['target_speaker']}
        if p['pair_id'] in fail:
            tgt.append(dict(base, error='RuntimeError: boom'))
            src.append(dict(base, system=system + '@src', error='RuntimeError: boom'))
            continue
        n = 12
        k = int(rng.binomial(n, c['rate']))
        ins = int(rng.poisson(c['ins'])) if c['ins'] else 0
        kb = min(n, k + 1)
        r = dict(base, wer_large_v3={'hyp': f'h{k}', 'S': k, 'D': 0, 'I': ins, 'N': n, 'cS': 3 * k, 'cD': 0,
                                     'cI': 2 * ins, 'cN': 60},
                 wer_base={'hyp': f'b{kb}', 'S': kb, 'D': 0, 'I': ins, 'N': n, 'cS': 3 * kb, 'cD': 0, 'cI': 2 * ins,
                           'cN': 60},
                 sim_wavlm=c['sim'] + 0.02 * rng.standard_normal(), secs_resemblyzer=0.8 + 0.01 * rng.standard_normal(),
                 utmos=c['utmos'] + 0.2 * rng.standard_normal(),
                 emb_wavlm=pack(np.eye(8)[int(p['target_speaker'][-2:]) % 8] + 0.3 * rng.standard_normal(8)),
                 ref_emb_wavlm=pack(np.eye(8)[int(p['target_speaker'][-2:]) % 8]))
        if p['P']:
            r.update(mcd_db=c['mcd'] + 0.3 * rng.standard_normal(), lf0_rmse=0.2, vuv_err=0.1, f0_pcc=0.5)
        if system == 'crt_tau0.4':
            r['override_coverage'] = {'per_cb': {'cb1': 0.3, 'cb2': 0.2}, 'overall': 0.25}
        if system == 'cand_pm18_rq':
            r.update(latent_dist_to_faiss=2.0, orphan_share=float('nan') if p['P'] else 0.01)
        tgt.append(r)
        g = stats.stratum(p['genders'])
        shift = (src_shift or {}).get(g, 0.0)
        src.append(dict(base, system=system + '@src', target_speaker=p['source_speaker'],
                        sim_wavlm=c['src'] + shift + 0.02 * rng.standard_normal(),
                        secs_resemblyzer=0.6 + 0.01 * rng.standard_normal()))
    return tgt, src


def build(pairs, systems=('faiss', 'crt_tau0.4', 'crt_tau0.9', 'cand_pm18_rq', 'knnvc', 'freevc', 'source_copy',
                          'target_copy'), budgets=(30,), cfg=None, fail=None, src_shift=None):
    R = {}
    for b in budgets:
        for s in systems:
            if b != 30 and s not in ('faiss', 'crt_tau0.4', 'crt_tau0.9', 'cand_pm18_rq', 'knnvc'):
                continue
            tgt, src = rows_for(pairs, s, b, cfg=(cfg or {}).get(s), fail=(fail or {}).get(s, ()),
                                src_shift=(src_shift or {}).get(s))
            R.setdefault(b, {})[s] = {r['pair_id']: r for r in tgt}
            if src:
                R[b][s + '@src'] = {r['pair_id']: r for r in src}
    return R


def make_sets(cfg=None, fail=None, src_shift=None, src_e=None, budgets=(30,)):
    vp, lp = vctk_pairs(), ltc_pairs()
    sets = {'vctk2k': at.TestSet('vctk2k', vp, build(vp, budgets=budgets, cfg=cfg, fail=fail, src_shift=src_shift),
                                 src_e if src_e is not None else {p['source_speaker'] for p in vp}),
            'ltc2k': at.TestSet('ltc2k', lp, build(lp, budgets=budgets, cfg=cfg, fail=fail, src_shift=src_shift))}
    return sets


@pytest.fixture(scope='module')
def base_result():
    sets = make_sets()
    return sets, at.analyse(sets, SYSTEMS)


# ── systems from the record ──────────────────────────────────

def test_systems_from_the_record():
    assert SYSTEMS['crt_al'] == 'cand_pm18_rq' and SYSTEMS['s2'] == 'crt_tau0.9' and SYSTEMS['s3'] == 'crt_tau0.4'
    assert SYSTEMS['baselines'] == ['knnvc', 'freevc'] and SYSTEMS['vctk_targets'] == '5'
    c1 = at.systems_from_record(dict(RECORD, selection={'choice': 'C1'}, tau_rule={'tau': 0.4}))
    assert c1['crt_al'] == 'anc_R3_s0_pm_rq_T1' and c1['s2'] == c1['s3'] == 'crt_tau0.4'


@pytest.mark.parametrize('bad', [dict(RECORD, status='INCOMPLETE'), dict(RECORD, selection={'choice': None})],
                         ids=['incomplete', 'F3'])
def test_record_refusals(bad):
    with pytest.raises(SystemExit):
        at.systems_from_record(bad)


# ── families ─────────────────────────────────────────────────

def test_family_p_endpoints_subsets_and_directions(base_result):
    sets, res = base_result
    P = res['families']['P']
    ids = [h['id'] for h in P['hypotheses']]
    assert ids == ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
    h = {x['id']: x for x in P['hypotheses']}
    v, l = sets['vctk2k'], sets['ltc2k']
    assert h['P1']['n'] == len(v.subset('wer')) < len(v.pairs)               # wer_primary pairs only
    assert h['P2']['n'] == len(l.pairs)                                        # every LTC pair is wer_primary
    assert h['P7']['n'] == len(v.subset('par')) and h['P7']['set'] == 'vctk2k'  # dB-MCD on the pairs with P
    assert h['P1']['metric'] == 'wer_capped' and h['P1']['label'] == 'capped WER %'
    assert h['P1']['mode'] == stats.INDEPENDENT and h['P2']['mode'] == stats.LINKED
    assert h['P1']['reject'] and h['P1']['favours'] == A                      # far lower WER
    assert h['P5']['reject'] and h['P5']['favours'] == A                      # higher UTMOS
    assert h['P3']['reject'] and h['P3']['favours'] == F                      # lower SIM favours FAISS
    assert h['P7']['reject'] and h['P7']['favours'] == F                      # higher dB-MCD favours FAISS
    assert P['cmos']['status'] == 'pending' and all('reject_strict' in x for x in P['hypotheses'])


def test_capped_wer_in_p_uncapped_in_v(base_result):
    sets, res = base_result
    P1 = res['families']['P']['hypotheses'][0]
    V1 = res['families']['V']['hypotheses'][0]
    v = sets['vctk2k']
    assert V1['metric'] == 'wer'
    want_capped = stats.corpus_wer([v.rows(F)[k] for k in v.subset('wer')], capped=True)['wer']
    want_unc = stats.corpus_wer([v.rows(F)[k] for k in v.subset('wer')])['wer']
    assert P1['b'] == pytest.approx(want_capped) and V1['b'] == pytest.approx(want_unc) and want_unc > want_capped


def test_family_k_and_v_systems(base_result):
    _, res = base_result
    K, V = res['families']['K'], res['families']['V']
    assert (K['a'], K['b']) == (A, S3) and (V['a'], V['b']) == (S2, F)
    assert [h['id'] for h in V['hypotheses']] == [f'V{i}' for i in range(1, 8)] and 'cmos' not in V
    assert V['holm'] == 'over 7'


def test_s_family_per_baseline_and_vctk_seen_flag(base_result):
    _, res = base_result
    S = res['families']['S']
    assert len(S['hypotheses']) == 12 and S['holm'] == 'over 12'
    flagged = {(h['sys_b'], h['set']) for h in S['hypotheses'] if h.get('flag') == 'VCTK-seen'}
    assert flagged == {('freevc', 'vctk2k')}
    assert {h['quantity'] for h in S['hypotheses']} == {'wer', 'sim', 'utmos'}


def test_holm_is_per_family(base_result):
    _, res = base_result
    for key in ('P', 'K', 'V', 'S'):
        fam = res['families'][key]
        assert [h['reject'] for h in fam['hypotheses']] == stats.holm([h['p'] for h in fam['hypotheses']])


# ── CMOS (P8, K8) ────────────────────────────────────────────

def test_cmos_in_family_dropped_or_pending():
    sets = make_sets()
    hyps_only = at.family(sets, 'P', A, F, at.ENDPOINTS, 'wer_capped', None, 'P8')
    assert hyps_only['holm'].startswith('over 7 if P8 is dropped')
    good = {'P8': {'est': 0.8, 'lo': 0.4, 'hi': 1.2, 'p': 0.0002, 'n_raters': 18}}
    inn = at.family(sets, 'P', A, F, at.ENDPOINTS, 'wer_capped', good, 'P8')
    assert inn['cmos']['status'] == 'in family' and inn['holm'] == 'over 8' and inn['cmos']['reject']
    few = {'P8': dict(good['P8'], n_raters=12)}
    out = at.family(sets, 'P', A, F, at.ENDPOINTS, 'wer_capped', few, 'P8')
    assert out['cmos']['status'] == 'dropped' and out['holm'] == 'over 7 (P8 dropped)' and out['cmos']['p'] is None


def test_pending_cmos_is_the_strictest_case():
    p = [0.004, 0.008, 0.2, 0.8, 0.9, 0.95, 0.99]
    hyps = [{'p': x} for x in p]
    seven, _ = at.apply_holm(hyps)
    eight, _ = at.apply_holm(hyps, [{'p': None}])
    assert seven[:2] == [True, True] and eight[:2] == [True, False]   # 0.008 <= 0.05/6 but not <= 0.05/7


# ── gate L ───────────────────────────────────────────────────

def test_gate_rejects_without_leakage(base_result):
    _, res = base_result
    L = res['gate_l']
    assert set(L['strata']) == {'vctk2k/same', 'vctk2k/cross', 'ltc2k/same', 'ltc2k/cross'}
    assert [L['strata'][k]['id'] for k in ('vctk2k/same', 'vctk2k/cross', 'ltc2k/same', 'ltc2k/cross')] == \
        ['L1s', 'L1c', 'L2s', 'L2c']
    assert L['all_reject'] is True
    for r in L['strata'].values():
        assert r['hi'] < 0.01 and r['rejects'] is True and r['m9_diff']['est'] is not None


def test_gate_fails_with_cross_gender_leakage_and_frames_at_most_b():
    sets = make_sets(src_shift={A: {'cross': 0.08}})
    res = at.analyse(sets, SYSTEMS)
    L = res['gate_l']['strata']
    assert L['vctk2k/cross']['rejects'] is False and L['ltc2k/cross']['rejects'] is False
    assert L['vctk2k/same']['rejects'] is True
    for o in res['outcome']['variants'].values():
        assert o['leakage_not_excluded'] and set(o['gate_l_failed']) == {'L1c', 'L2c'}
        assert o['pattern'] == 'B' and o['framing'] == 'B'


def test_gate_leaves_out_sources_without_e_set():
    vp = vctk_pairs()
    keep = {p['source_speaker'] for p in vp} - {'s00'}
    sets = make_sets(src_e=keep)
    L = at.gate(sets, A, F)['strata']
    left = sum(1 for p in vp if p['source_speaker'] == 's00')
    assert L['vctk2k/same']['n_no_source_E'] + L['vctk2k/cross']['n_no_source_E'] == left
    assert L['ltc2k/same']['n_no_source_E'] == 0


def test_gate_pass_below_the_natural_level_is_not_interpretable():
    sets = make_sets(cfg={A: {'src': 0.55}, F: {'src': 0.56}, 'target_copy': {'src': 0.66}})
    L = at.gate(sets, A, F)['strata']
    r = L['vctk2k/same']
    assert r['rejects'] is True and r['a'] < r['natural_other_speaker']['sim_src_target_copy']
    assert r['interpretable'] is False and L['ltc2k/same']['interpretable'] is True
    o = at.outcome(at.family(sets, 'P', A, F, at.ENDPOINTS, 'wer_capped', None, 'P8'), at.gate(sets, A, F), A, F)
    assert 'L1s' in o['not_interpretable']


# ── outcome ──────────────────────────────────────────────────

def fake_family(effects, cmos=None):
    hyps = [{'id': f'P{i}', 'quantity': q, 'reject': r, 'favours': (A if fav == 'a' else F) if r else None}
            for i, (q, r, fav) in enumerate(effects, 1)]
    return {'hypotheses': hyps, 'cmos': cmos or {'id': 'P8', 'status': 'pending', 'p': None}}


GATE_OK = {'all_reject': True, 'strata': {k: {'id': k, 'rejects': True, 'interpretable': True}
                                          for k in ('L1s', 'L1c', 'L2s', 'L2c')}}


@pytest.mark.parametrize('effects, want', [
    ([('wer', True, 'a'), ('sim', False, None)], 'A'),
    ([('wer', True, 'a'), ('sim', True, 'b')], 'B'),
    ([('wer', False, None), ('sim', False, None)], 'C'),
    ([('wer', False, None), ('sim', True, 'b')], 'C'),
])
def test_outcome_patterns(effects, want):
    o = at.outcome(fake_family(effects), GATE_OK, A, F)
    assert o['pattern'] == want and o['framing'] == want and not o['leakage_not_excluded']
    assert at.outcomes(fake_family(effects), GATE_OK, A, F)['p8_pending']


def test_outcome_a_without_the_gate_is_framed_b():
    gate = copy.deepcopy(GATE_OK)
    gate['all_reject'] = False
    gate['strata']['L2c']['rejects'] = False
    o = at.outcome(fake_family([('wer', True, 'a')]), gate, A, F)
    assert o['pattern'] == 'A' and o['framing'] == 'B' and o['gate_l_failed'] == ['L2c']


def test_outcome_counts_a_significant_cmos():
    o = at.outcome(fake_family([('wer', False, None)], {'id': 'P8', 'status': 'in family', 'p': 0.001,
                                                        'est': -0.5, 'reject': True, 'favours': F}), GATE_OK, A, F)
    assert o['pattern'] == 'C' and o['favour_faiss'] == ['P8']
    o = at.outcome(fake_family([('wer', False, None)], {'id': 'P8', 'status': 'in family', 'p': 0.001,
                                                        'est': 0.5, 'reject': True, 'favours': A}), GATE_OK, A, F)
    assert o['pattern'] == 'A' and o['favour_crt_al'] == ['P8']
    assert not at.outcomes(fake_family([('wer', False, None)], {'id': 'P8', 'status': 'in family', 'p': 0.001,
                                                                'reject': True, 'favours': A}), GATE_OK, A, F)['p8_pending']


@pytest.mark.parametrize('q, est, want', [('wer', -1.0, 'a'), ('wer', 1.0, 'b'), ('sim', 0.01, 'a'),
                                          ('sim', -0.01, 'b'), ('utmos', 0.1, 'a'), ('mcd', 0.2, 'b'),
                                          ('mcd', -0.2, 'a'), ('sim', 0.0, None)])
def test_favours(q, est, want):
    assert at.favours({'est': est}, q, ('a', 'b')) == want


# ── sensitivities ────────────────────────────────────────────

def test_sensitivity_blocks(base_result):
    _, res = base_result
    sens = res['sensitivity']
    assert sens['union_pk']['m'] == 14 and sens['union_pk']['m_strict'] == 16 and sens['union_pk']['decisions_strict']
    assert set(sens['wer_definition']) == {'P', 'K', 'V', 'S'}
    assert sens['wer_definition']['P']['family']['hypotheses'][0]['metric'] == 'wer'
    assert sens['wer_definition']['V']['family']['hypotheses'][0]['metric'] == 'wer_capped'
    ltc = sens['ltc_independent']['P']['family']['hypotheses']
    assert ltc[1]['mode'] == stats.INDEPENDENT and ltc[0]['mode'] == stats.INDEPENDENT    # VCTK-2k always independent
    assert res['families']['P']['hypotheses'][1]['mode'] == stats.LINKED
    assert set(sens['ltc_independent_gate']) == {'L2s', 'L2c'}


def test_failed_conversions_are_left_out_and_get_a_worst_case():
    vp = vctk_pairs()
    bad = [p['pair_id'] for p in vp if p['wer_primary']][:3]
    sets = make_sets(fail={A: bad})
    res = at.analyse(sets, SYSTEMS)
    P1 = res['families']['P']['hypotheses'][0]
    assert P1['fail_a'] == 3 and P1['fail_b'] == 0 and P1['n'] == len(sets['vctk2k'].subset('wer')) - 3
    wc = P1['worst_case']
    assert wc['imputed_a'] == 3 and wc['est'] > P1['est']         # every word of a failed pair deleted
    assert 'P1' in res['sensitivity']['worst_case']['P']['worst_case_used']


def test_changed_compares_reject_and_direction():
    fam = {'hypotheses': [{'id': 'X1', 'reject': True, 'favours': A}, {'id': 'X2', 'reject': False, 'favours': None},
                          {'id': 'X3', 'reject': True, 'favours': A}]}
    other = {'hypotheses': [{'id': 'X1', 'reject': True, 'favours': A}, {'id': 'X2', 'reject': True, 'favours': F},
                            {'id': 'X3', 'reject': True, 'favours': F}]}
    assert at.changed(fam, other) == {'holm': ['X2', 'X3']}


def test_analysis_is_deterministic():
    a = at.analyse(make_sets(), SYSTEMS)
    b = at.analyse(make_sets(), SYSTEMS)
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


# ── inputs and main ──────────────────────────────────────────

def write_m(root, name, part, budget, R):
    d = root / part / 'metrics'
    d.mkdir(parents=True, exist_ok=True)
    tgt = [r for s, rs in R.items() if not s.endswith('@src') for r in rs.values()]
    src = [r for s, rs in R.items() if s.endswith('@src') for r in rs.values()]
    (d / f'{name}_{budget}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tgt))
    (d / f'{name}_{budget}_src.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in src))


SEL = 'a' * 64


def full_inputs(tmp_path, n_src=20, utts=20, per_utt=5, drop=None, no_done=(), done_sha=None, cut=None):
    vp, lp = vctk_pairs(n_src=n_src, n_tgt=20, utts=utts, per_utt=per_utt), ltc_pairs()
    done_sha = done_sha or {}
    root = tmp_path / 'm'
    for name, pairs, main_part in (('vctk2k', vp, 't1'), ('ltc2k', lp, 't2')):
        R = build(pairs, budgets=(30, 3, 180))
        core = ('faiss', 'crt_tau0.4', 'crt_tau0.9', 'cand_pm18_rq', 'source_copy', 'target_copy')
        pick = lambda b, names: {s: (dict(list(v.items())[:-1]) if s == cut else v) for s, v in R[b].items()
                                 if s.split('@')[0] in names and s.split('@')[0] != drop}
        write_m(root, name, main_part, 30, pick(30, core))
        write_m(root, name, 't4', 30, pick(30, ('knnvc', 'freevc')))
        for b in (3, 180):
            write_m(root, name, 't3', b, pick(b, ('faiss', 'crt_tau0.4', 'crt_tau0.9', 'cand_pm18_rq')))
            write_m(root, name, 't4', b, pick(b, ('knnvc',)))
    for part in ('t1', 't2', 't3', 't4'):
        if part not in (no_done or ()):
            (root / part / 'm.done').write_text(f'job 1 2026-10-20 10:00:00 selection sha256 {done_sha.get(part, SEL)}\n')
    paths = {}
    for name, pairs in (('vctk', vp), ('ltc', lp)):
        paths[name] = tmp_path / f'{name}.jsonl'
        paths[name].write_text(''.join(json.dumps(p) + '\n' for p in pairs))
    paths['src_e'] = tmp_path / 'src_e.json'
    paths['src_e'].write_text(json.dumps({p['source_speaker']: {'paths': ['/x.flac']} for p in vp}))
    paths['record'] = tmp_path / 'record.json'
    paths['record'].write_text(json.dumps(RECORD))
    return root, paths


def argv(root, paths, out):
    return ['--m_dir', str(root), '--vctk_pairs', str(paths['vctk']), '--ltc_pairs', str(paths['ltc']),
            '--source_e', str(paths['src_e']), '--record', str(paths['record']), '--out', str(out)]


def test_main_end_to_end(tmp_path, capsys):
    root, paths = full_inputs(tmp_path)
    out = tmp_path / 'TEST_RESULTS_v1.md'
    assert at.main(argv(root, paths, out)) == 0
    md, res = out.read_text(), json.loads((tmp_path / 'TEST_RESULTS_v1.json').read_text())
    assert '## Outcome (§7)' in md and '## Family P (primary)' in md and '## Gate L' in md
    assert res['outcome']['p8_pending'] is True
    assert {v['pattern'] for v in res['outcome']['variants'].values()} == {'B'}
    assert set(res['outcome']['variants']) == {at.VARIANT_DROPPED, at.VARIANT_STRICT}
    assert 'whatever the CMOS shows' in md and 'possibly anti-conservative' in md and 'VCTK-seen' in md
    lv = res['levels']['vctk2k']
    assert lv['faiss']['wer_b']['est'] is not None and lv['faiss']['secs_src']['est'] is not None
    assert lv['crt_tau0.4']['coverage']['override_coverage']['overall'] == pytest.approx(0.25)
    assert lv['cand_pm18_rq']['coverage']['orphan_share'] == pytest.approx(0.01)       # NaN rows left out
    assert res['m_done'] == {p: f'job 1 2026-10-20 10:00:00 selection sha256 {SEL}' for p in ('t1', 't2', 't3', 't4')}
    assert len(res['inputs']) == 4 + 2 * 2 * 6                   # record, 2 lists, source E; 6 files x 2 per set
    assert all(len(x['sha256']) == 64 for x in res['inputs'])
    assert set(res['budget_sweep']['vctk2k']['knnvc']['wer']) == {'3', '30', '180'} or \
        set(res['budget_sweep']['vctk2k']['knnvc']['wer']) == {3, 30, 180}
    assert res['eer']['vctk2k']['faiss'] is not None
    assert res['levels']['ltc2k']['faiss'].get('mcd') is None and 'mcd' in res['levels']['vctk2k']['faiss']


def test_main_refuses_a_missing_system(tmp_path):
    root, paths = full_inputs(tmp_path, drop='crt_tau0.4')
    with pytest.raises(SystemExit, match='no rows for'):
        at.main(argv(root, paths, tmp_path / 'r.md'))


def test_main_refuses_a_wrong_vctk_variant(tmp_path):
    root, paths = full_inputs(tmp_path, utts=10)
    with pytest.raises(SystemExit, match='5-target variant has 2000'):
        at.main(argv(root, paths, tmp_path / 'r.md'))


def test_load_set_refusals(tmp_path):
    pairs = ltc_pairs()
    R = build(pairs, systems=('faiss',))
    write_m(tmp_path, 'ltc2k', 't2', 30, R[30])
    with pytest.raises(SystemExit, match='missing'):
        at.load_set(str(tmp_path), 'ltc2k', [('t2', 30), ('t3', 3)], pairs)
    write_m(tmp_path, 'ltc2k', 't4', 30, R[30])                    # the same rows twice
    with pytest.raises(SystemExit, match='occurs in two files'):
        at.load_set(str(tmp_path), 'ltc2k', [('t2', 30), ('t4', 30)], pairs)
    with pytest.raises(SystemExit, match='is not in the ltc2k list'):
        at.load_set(str(tmp_path), 'ltc2k', [('t2', 30)], pairs[1:])
    write_m(tmp_path, 'ltc2k', 't3', 180, R[30])                   # budget 30 rows in a 180 s file
    with pytest.raises(SystemExit, match='budget'):
        at.load_set(str(tmp_path), 'ltc2k', [('t3', 180)], pairs)


# ── review additions: Holm sizes, strict outcome, gate worst case, CMOS keys, completeness ──

def holm_case(ps, c=None):
    hyps = [{'id': f'H{i}', 'p': p} for i, p in enumerate(ps, 1)]
    return at.holm_flags(hyps, c)


def test_holm_flags_size_depends_on_the_cmos_status():
    ps = [0.004, 0.008, 0.2, 0.3, 0.4, 0.5, 0.6]             # 0.008 <= 0.05/6 (m = 7) but > 0.05/7 (m = 8)
    flags, strict, cf, label = holm_case(ps)
    assert flags[:2] == [True, True] and strict is None and label == 'over 7'
    flags, strict, cf, label = holm_case(ps, {'id': 'P8', 'status': 'pending', 'p': None})
    assert flags[:2] == [True, True] and strict[:2] == [True, False] and label.startswith('over 7 if P8 is dropped')
    flags, strict, cf, label = holm_case(ps, {'id': 'P8', 'status': 'in family', 'p': 0.9})
    assert flags[:2] == [True, False] and cf is False and label == 'over 8'
    flags, strict, cf, label = holm_case(ps, {'id': 'P8', 'status': 'in family', 'p': 0.001})
    assert flags[:2] == [True, True] and cf is True                     # a small P8 takes the 0.05/8 slot
    flags, strict, cf, label = holm_case(ps, {'id': 'P8', 'status': 'dropped', 'p': None})
    assert flags[:2] == [True, True] and label == 'over 7 (P8 dropped)'


def test_outcome_differs_between_the_holm_variants():
    fam = fake_family([('wer', True, 'a'), ('sim', True, 'b')])
    fam['hypotheses'][0].update(reject_strict=True, favours_strict=A)
    fam['hypotheses'][1].update(reject_strict=False, favours_strict=None)
    o = at.outcomes(fam, GATE_OK, A, F)
    assert o['p8_pending'] and o['variants'][at.VARIANT_DROPPED]['pattern'] == 'B'
    assert o['variants'][at.VARIANT_STRICT]['pattern'] == 'A'


def test_gate_worst_case_is_computed_and_reported():
    vp = vctk_pairs()
    cross = [p['pair_id'] for p in vp if stats.stratum(p['genders']) == 'cross'][:4]
    sets = make_sets(fail={A: cross})
    res = at.analyse(sets, SYSTEMS)
    g = res['gate_l']
    r = g['strata']['vctk2k/cross']
    assert r['fail_a'] == 4 and r['worst_case']['rejects'] in (True, False)
    assert g['worst_case']['used'] == ['L1c'] and 'L1c' in g['worst_case']['rejects']


def test_cmos_file_without_a_block_leaves_it_pending():
    sets = make_sets()
    only_p8 = {'P8': {'est': 0.3, 'lo': 0.1, 'hi': 0.5, 'p': 0.01, 'n_raters': 16}}
    res = at.analyse(sets, SYSTEMS, only_p8)
    assert res['families']['P']['cmos']['status'] == 'in family'
    assert res['families']['K']['cmos']['status'] == 'pending'
    assert res['families']['K']['holm'].startswith('over 7 if K8 is dropped')
    u = res['sensitivity']['union_pk']
    assert u['m'] == 15 and u['m_strict'] == 16
    assert not res['outcome']['p8_pending'] and set(res['outcome']['variants']) == {'final'}


def test_main_refuses_unfinished_or_mixed_parts_of_m(tmp_path):
    root, paths = full_inputs(tmp_path, no_done=('t3',))
    with pytest.raises(SystemExit, match='has not finished'):
        at.main(argv(root, paths, tmp_path / 'r.md'))
    root, paths = full_inputs(tmp_path / 'b', done_sha={'t4': 'b' * 64})
    with pytest.raises(SystemExit, match='different selection.env'):
        at.main(argv(root, paths, tmp_path / 'r.md'))


def test_main_refuses_a_system_short_of_rows(tmp_path):
    root, paths = full_inputs(tmp_path, cut='faiss')
    with pytest.raises(SystemExit, match='not one row per pair'):
        at.main(argv(root, paths, tmp_path / 'r.md'))


def test_main_creates_the_output_folder(tmp_path):
    root, paths = full_inputs(tmp_path)
    out = tmp_path / 'new' / 'dir' / 'R.md'
    assert at.main(argv(root, paths, out)) == 0 and out.exists() and (out.parent / 'R.json').exists()


def test_natural_level_also_on_the_same_pairs():
    # CRT-AL's SIM_src is 0.60 on the pairs with P and 0.70 elsewhere: above the natural level 0.62 over all the
    # gate's pairs (about 0.65), below it on the pairs where target_copy exists
    sets = make_sets(cfg={A: {'src': 0.60}, F: {'src': 0.67}, 'target_copy': {'src': 0.62}})
    v = sets['vctk2k']
    for k, r in v.R[30][A + '@src'].items():
        if not v.info[k]['P']:
            r['sim_wavlm'] += 0.1
    L = at.gate(sets, A, F)['strata']
    r = L['vctk2k/same']
    nat = r['natural_other_speaker']
    assert r['rejects'] is True and nat['n_same_pairs'] == nat['n'] > 0
    assert r['a'] > nat['sim_src_target_copy'] > nat['sim_src_crt_al_same_pairs']
    assert r['interpretable'] is False


@pytest.mark.parametrize('n, status', [(15, 'in family'), (14, 'dropped'), (20, 'in family'), (0, 'dropped')])
def test_cmos_needs_at_least_15_screened_raters(n, status):
    c = at.cmos_entry({'P8': {'est': 0.2, 'p': 0.03, 'n_raters': n}}, 'P8')
    assert c['status'] == status and (c['p'] is None) == (status == 'dropped')
