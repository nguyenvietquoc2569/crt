"""Tests of benchmark/listening_analysis.py on a simulated rater panel built from listening_stimuli.trials(): the
pre-registered screening rules (headphone check, gold items, attention items, hidden identical pairs), MOS / SMOS per
condition, the CMOS preference for CRT-AL whatever the A/B order, the p-value, cmos.json for analyze_tests.py, the
pilot's rater count and the 60-minute halves rule."""
import json

import numpy as np
import pytest

import listening_analysis as la
import listening_stimuli as ls
from test_listening_stimuli import world

MOS_MEAN = {'nat': 4.6, 'nat_mimi': 4.0, 'faiss': 2.0, 'crt04': 2.2, 'crtal': 3.1}
SMOS_MEAN = {'nat': 4.5, 'nat_mimi': 4.3, 'faiss': 3.6, 'crt04': 3.5, 'crtal': 3.2}
PREF = {'faiss': 1.0, 'crt04': 0.8}


@pytest.fixture(scope='module')
def setup():
    items, pairs, rows = world()
    pages, clips, key_trials = ls.trials(ls.plan(items, pairs, rows, ls.CONDITIONS))
    return pages, {'trials': key_trials, 'clips': clips}


def clip(v, lo, hi):
    return int(min(hi, max(lo, round(v))))


def rater(pages, key, rid, seed, pref=PREF, noise=0.6, block_min=(5, 20, 20, 15), cmos_noise=None, extra_min=10,
          **bad):
    """A results file as the page writes it; bad: gold_low (n gold items rated 1), attention_wrong (bool),
    hidden_big (n hidden pairs rated +3), headphone (bool)."""
    rng = np.random.default_rng(seed)
    parity = int(rid[-2:]) % 2
    trials, clips = key['trials'], key['clips']
    resp = {'practice': [], 'mos': [], 'smos': [], 'cmos': []}
    gold_low = bad.get('gold_low', 0)
    for t in pages['mos']:
        kt = trials[t['tid']]
        if 'attention' in t:
            r = t['attention'] if not bad.get('attention_wrong') else (t['attention'] % 5) + 1
        elif kt['kind'] == 'gold' and gold_low > 0:
            r, gold_low = 1, gold_low - 1
        else:
            r = clip(MOS_MEAN[clips[t['stim']]['condition']] + noise * rng.standard_normal(), 1, 5)
        resp['mos'].append({'tid': t['tid'], 'stim': t['stim'], 'rating': r, 'plays': 1, 'rt_ms': 3000})
    for t in pages['smos']:
        if 'attention' in t:
            r = t['attention'] if not bad.get('attention_wrong') else (t['attention'] % 5) + 1
        else:
            r = clip(SMOS_MEAN[clips[t['stim']]['condition']] + noise * rng.standard_normal(), 1, 5)
        resp['smos'].append({'tid': t['tid'], 'stim': t['stim'], 'ref': t['ref'], 'rating': r})
    hidden_big = bad.get('hidden_big', 0)
    for k, t in enumerate(pages['cmos']):
        x_first = (k + parity) % 2 == 0
        a, b = (t['x'], t['y']) if x_first else (t['y'], t['x'])
        kt = trials[t['tid']]
        if 'attention' in t:
            r = t['attention'] if not bad.get('attention_wrong') else t['attention'] + 1
        elif kt['kind'] == 'hidden':
            if hidden_big > 0:
                r, hidden_big = 3, hidden_big - 1
            else:
                r = clip(0.3 * rng.standard_normal(), -3, 3)
        else:
            p = pref[kt['y']] + (noise if cmos_noise is None else cmos_noise) * rng.standard_normal()   # for CRT-AL (x)
            r = clip(p if b == t['x'] else -p, -3, 3)
        resp['cmos'].append({'tid': t['tid'], 'a': a, 'b': b, 'rating': r})
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 10, 21, 2, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=sum(block_min) + extra_min)
    return {'version': 1, 'rater_id': rid, 'parity': parity, 'half': None,
            'started': t0.isoformat().replace('+00:00', 'Z'), 'finished': t1.isoformat().replace('+00:00', 'Z'),
            'headphone': {'attempts': [], 'passed': bad.get('headphone', True)}, 'responses': resp,
            'block_ms': dict(zip(('practice', 'mos', 'smos', 'cmos'), (m * 60000 for m in block_min)))}


def panel(pages, key, n=16, **kw):
    return {f'R{i:02d}': rater(pages, key, f'R{i:02d}', i, **kw) for i in range(1, n + 1)}


# ── screening ────────────────────────────────────────────────

def test_screening_rules(setup):
    pages, key = setup
    res = panel(pages, key, n=4)
    res['R05'] = rater(pages, key, 'R05', 5, gold_low=2)
    res['R06'] = rater(pages, key, 'R06', 6, gold_low=1)
    res['R07'] = rater(pages, key, 'R07', 7, attention_wrong=True)
    res['R08'] = rater(pages, key, 'R08', 8, hidden_big=5)
    res['R09'] = rater(pages, key, 'R09', 9, hidden_big=4)
    res['R10'] = rater(pages, key, 'R10', 10, headphone=False)
    sc = la.screen(res, key)
    assert sc['R01'] == [] and sc['R06'] == [] and sc['R09'] == []          # one low gold, four big hidden: kept
    assert 'gold items' in sc['R05'][0] and 'attention item' in sc['R07'][0]
    assert 'hidden identical pairs' in sc['R08'][0] and 'headphone' in sc['R10'][0]
    assert len(sc['R07']) == 4                                             # every failed attention item listed


# ── final ────────────────────────────────────────────────────

def test_final_levels_and_cmos(setup):
    pages, key = setup
    o = la.final(panel(pages, key), key)
    assert o['n_kept'] == 16
    for c, m in MOS_MEAN.items():
        x = o['mos'][c]
        assert x['lo'] <= x['est'] <= x['hi'] and abs(x['est'] - m) < 0.35 and x['n_items'] == 32
    assert o['mos']['nat']['est'] > o['mos']['crtal']['est'] > o['mos']['faiss']['est']
    for k, want in (('P8', PREF['faiss']), ('K8', PREF['crt04'])):
        assert abs(o[k]['est'] - want) < 0.3 and o[k]['p'] < 0.01 and o[k]['n_raters'] == 16
        assert o[k]['lo'] > 0
    assert abs(o['hidden']['est']) < 0.2
    assert o['mos_codec_diffs']['crtal-faiss']['est'] > 0.5


def test_cmos_sign_does_not_depend_on_the_ab_order(setup):
    pages, key = setup
    even = {f'R{i:02d}': rater(pages, key, f'R{i:02d}', i, noise=0.0) for i in (2, 4, 6)}
    odd = {f'R{i:02d}': rater(pages, key, f'R{i:02d}', i, noise=0.0) for i in (1, 3, 5)}
    for res in (even, odd):
        v = la.cmos_values(res, key, sorted(res), 'P8')
        assert {x[2] for x in v} == {1.0}                                   # +1 for CRT-AL in every pair


def test_no_effect_gives_a_large_p(setup):
    pages, key = setup
    o = la.final(panel(pages, key, pref={'faiss': 0.0, 'crt04': 0.0}), key)
    assert o['P8']['p'] > 0.05 and o['P8']['lo'] < 0 < o['P8']['hi']


def test_excluded_raters_are_not_counted(setup):
    pages, key = setup
    res = panel(pages, key, n=15)
    res['R16'] = rater(pages, key, 'R16', 16, attention_wrong=True)
    o = la.final(res, key)
    assert o['n_kept'] == 15 and o['P8']['n_raters'] == 15 and 'R16' not in o['kept']


def test_a_cmos_trial_that_played_other_systems_is_an_error(setup):
    pages, key = setup
    res = panel(pages, key, n=1)
    t = next(x for x in res['R01']['responses']['cmos'] if key['trials'][x['tid']]['kind'] == 'P8')
    other = next(c for c, v in key['clips'].items() if v['condition'] == 'nat')
    t['b'] = other
    with pytest.raises(SystemExit, match='the key says'):
        la.cmos_values(res, key, ['R01'], 'P8')


def test_crossed_bootstrap_is_deterministic_and_centred():
    vals = [(f'r{r}', f'i{i}', float(r + i)) for r in range(5) for i in range(6)]
    a, ra = la.crossed(vals, B_=2000)
    b, rb = la.crossed(vals, B_=2000)
    assert a == b == pytest.approx(np.mean([v[2] for v in vals])) and np.array_equal(ra, rb)
    assert abs(np.mean(ra) - a) < 0.2


# ── pilot ────────────────────────────────────────────────────

def test_pilot_rater_count_and_halves(setup):
    pages, key = setup
    calm = la.pilot(panel(pages, key, n=5, noise=0.3, block_min=(5, 15, 15, 10)), key)
    noisy = la.pilot(panel(pages, key, n=5, noise=0.3, cmos_noise=2.5, block_min=(5, 30, 30, 20)), key)
    assert len(noisy['kept']) == 5
    assert calm['P8']['raters'] == 15 and calm['P8']['mde_at_raters'] <= 0.2
    assert noisy['P8']['raters'] >= calm['P8']['raters']
    assert calm['halves'] is False and noisy['halves'] is True             # 55 min wall clock; 95 min > 60 min
    assert calm['session_min']['median'] == pytest.approx(55.0) and calm['session_min']['basis'] == 'wall clock'
    assert calm['session_min']['trial_screens_median'] == pytest.approx(45.0)


# ── main ─────────────────────────────────────────────────────

def test_main_writes_cmos_json_for_analyze_tests(setup, tmp_path):
    pages, key = setup
    (tmp_path / 'key.json').write_text(json.dumps(key))
    d = tmp_path / 'results'
    d.mkdir()
    for rid, r in panel(pages, key).items():
        (d / f'listening_{rid}.json').write_text(json.dumps(r))
    out = tmp_path / 'out' / 'LISTENING_RESULTS_v1.md'
    assert la.main(['final', '--key', str(tmp_path / 'key.json'), '--results', str(d), '--out', str(out)]) == 0
    cmos = json.loads((out.parent / 'cmos.json').read_text())
    assert set(cmos) == {'P8', 'K8'} and cmos['P8']['n_raters'] == 16 and cmos['P8']['p'] < 0.01
    import analyze_tests as at
    assert at.cmos_entry(cmos, 'P8')['status'] == 'in family'
    assert 'P8: CRT-AL vs FAISS' in out.read_text()


def test_duplicate_rater_ids_are_refused(setup, tmp_path):
    pages, key = setup
    r = rater(pages, key, 'R01', 1)
    for n in (1, 2):
        (tmp_path / f'f{n}.json').write_text(json.dumps(r))
    with pytest.raises(SystemExit, match='twice'):
        la.load_results(la.result_files(str(tmp_path)))
