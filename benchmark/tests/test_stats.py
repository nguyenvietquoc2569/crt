"""Tests of benchmark/stats.py: ANALYSIS_PLAN_v1.md §5.2 (crossed bootstrap, p, Holm, gate L, failed conversions),
§5.3 (variance components, size check), §3.1 M3c and §0.3 row 20 (capped WER)."""
import json
import math

import numpy as np
import pytest

import stats
from fixtures_dev import dev_pairs, ltc_pairs, vctk_pairs, wer_counts

W = stats.WER_KEY


def approx(x):
    return pytest.approx(x, rel=1e-12, abs=1e-12)


def wrow(pid, e, n, **kw):
    return {'pair_id': pid, W: wer_counts(e, n), 'wer_primary': True, **kw}


def err(pid):
    return {'pair_id': pid, 'error': 'RuntimeError: synthetic'}


def ring(n_pairs, n_spk):
    """pair_id -> (source, target) over a speaker pool, source != target."""
    return {f'p{i:02d}': (f's{i % n_spk}', f's{(i + 1 + i // n_spk % (n_spk - 1)) % n_spk}') for i in range(n_pairs)}


def ref_bootstrap(src, tgt, mode, cols, den=(), B=stats.B, seed=stats.SEED):
    """§5.2 from its definition: explicit pair weights c_src[source] * c_tgt[target] per replicate (no cell sums),
    invalid replicates redrawn in order from the same generator. Returns the replicate sums (B x k+1), the redraws."""
    linked = mode == stats.LINKED
    ps = sorted(set(src) | set(tgt)) if linked else sorted(set(src))
    pt = ps if linked else sorted(set(tgt))
    si, ti = [ps.index(s) for s in src], [pt.index(t) for t in tgt]
    X = np.column_stack([np.asarray(cols, float).reshape(len(src), -1), np.ones(len(src))])
    rng = np.random.default_rng(seed)

    def counts(m, n):
        return np.array([np.bincount(r, minlength=n) for r in rng.integers(0, n, (m, n))], float)

    def sums(m):
        cs = counts(m, len(ps))
        ct = cs if linked else counts(m, len(pt))
        return (cs[:, si] * ct[:, ti]) @ X

    def ok(s):
        return s[-1] > 0 and all(s[j] > 0 for j in den)
    S, redraws = sums(B), 0
    for b in range(B):
        while not ok(S[b]):
            redraws += 1
            S[b] = sums(1)[0]
    return S, redraws


def ci(reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {'lo': lo, 'hi': hi, 'se': np.std(reps, ddof=1)}


def assert_close(out, **want):
    for k, v in want.items():
        assert out[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


# ── corpus WER and CER, capped (M3c) and uncapped ────────────

def test_corpus_wer_hand_computation():
    rows = [wrow('a', 2, 10), wrow('b', 0, 5), wrow('c', 7, 4, wer_primary=False)]     # c: 7 edits on 4 words
    assert stats.corpus_wer(rows) == pytest.approx({'n_utts': 3, 'wer': 100 * 9 / 19, 'cer': 100 * 27 / 95})
    assert stats.corpus_wer(rows, primary_only=True) == pytest.approx(
        {'n_utts': 2, 'wer': 100 * 2 / 15, 'cer': 100 * 6 / 75})
    # capped: c counts min(7, 4) words and min(21, 20) characters
    assert stats.corpus_wer(rows, capped=True) == pytest.approx(
        {'n_utts': 3, 'wer': 100 * 6 / 19, 'cer': 100 * 26 / 95})


def test_capped_wer_caps_each_pair_not_the_total():
    rows = [wrow('a', 12, 4), wrow('b', 0, 16)]         # 300 % and 0 %
    assert stats.corpus_wer(rows)['wer'] == 60.0
    assert stats.corpus_wer(rows, capped=True)['wer'] == 20.0      # min(12, 4) / 20; capping the sum would give 60
    by = {r['pair_id']: r for r in rows}
    assert stats.point(by, ['a', 'b'], 'wer_capped') == (20.0, 2) and stats.point(by, ['a', 'b'], 'wer') == (60.0, 2)
    assert stats.point(by, ['a'], 'cer_capped') == (100.0, 1) and stats.point(by, ['a'], 'cer') == (180.0, 1)


@pytest.mark.parametrize('e', [0, 3, 10])       # at most 100 % on every pair
def test_capped_equals_uncapped_without_pair_over_100(e):
    rows = [wrow('a', e, 10), wrow('b', 1, 7), wrow('c', 7, 7)]
    assert stats.corpus_wer(rows, capped=True) == stats.corpus_wer(rows)
    by = {r['pair_id']: r for r in rows}
    for m in ('wer', 'cer'):
        assert stats.point(by, list(by), m + '_capped') == stats.point(by, list(by), m)


def test_reimplementation_equals_metrics_py(tmp_path):
    import metrics          # librosa loads lazily: the import is cheap
    rng = np.random.default_rng(6)
    rows = [wrow(f'p{i}', int(rng.integers(0, 30)), int(rng.integers(1, 15)), wer_primary=bool(i % 2), system='s')
            for i in range(40)]
    for po in (False, True):
        assert stats.corpus_wer(rows, primary_only=po) == metrics.corpus_wer(rows, primary_only=po)
    f = tmp_path / 'm.jsonl'
    f.write_text(''.join(json.dumps(r) + '\n' for r in rows + [dict(rows[0], v=1)]) + '\n')
    assert stats.load_results(str(f)) == metrics.load_results(str(f))


def test_corpus_wer_raises_on_error_rows_and_missing_counts():
    with pytest.raises(ValueError):
        stats.corpus_wer([wrow('a', 1, 5), err('b')])
    with pytest.raises(ValueError):
        stats.corpus_wer([{'pair_id': 'a', 'wer_primary': True}])
    with pytest.raises(ValueError):
        stats.corpus_wer([{'pair_id': 'a', W: wer_counts(1, 5)}], primary_only=True)
    assert math.isnan(stats.corpus_wer([wrow('a', 0, 0)])['wer'])


def test_compare_capped_equals_uncapped_until_a_pair_exceeds_100():
    spk = ring(12, 4)
    a = {p: wrow(p, i % 4, 10) for i, p in enumerate(spk)}
    b = {p: wrow(p, 2, 10) for p in spk}
    keys = ('est', 'lo', 'hi', 'se', 'p', 'redraws', 'a', 'b', 'wilcoxon')
    u, c = (stats.compare(a, b, list(spk), spk, m) for m in ('wer', 'wer_capped'))
    assert all(u[k] == c[k] for k in keys)
    a['p01'] = wrow('p01', 60, 10)                       # a Whisper loop: 600 %
    u, c = (stats.compare(a, b, list(spk), spk, m) for m in ('wer', 'wer_capped'))
    e_a = sum(i % 4 for i in range(12) if i != 1)
    assert u['est'] == approx(100 * (e_a + 60) / 120 - 20) and c['est'] == approx(100 * (e_a + 10) / 120 - 20)
    assert c['hi'] < u['hi']


# ── crossed bootstrap ────────────────────────────────────────

def test_pair_weights_are_products_of_draw_counts():
    d = stats.Design(['a', 'a', 'b'], ['b', 'c', 'a'], stats.LINKED)         # one pool; a count serves both roles
    assert d.srcs == d.tgts == ['a', 'b', 'c']
    c = np.array([[2.0, 0, 1], [1, 3, 0]])
    assert d.pair_weights(c, c).tolist() == [[0, 2, 0], [3, 0, 3]]
    d = stats.Design(['a', 'a', 'b'], ['a', 'c', 'a'], stats.INDEPENDENT)    # source a and target a are distinct
    assert (d.srcs, d.tgts) == (['a', 'b'], ['a', 'c'])
    assert d.pair_weights(np.array([[0.0, 2]]), np.array([[1.0, 3]])).tolist() == [[0, 0, 2]]
    with pytest.raises(ValueError):
        stats.Design(['a'], ['b'], 'crossed')


@pytest.mark.parametrize('mode', [stats.LINKED, stats.INDEPENDENT])
def test_draw_takes_pool_size_counts_from_the_generator(mode):
    d = stats.Design(['a', 'b', 'c', 'a'], ['b', 'c', 'd', 'd'], mode)
    cs, ct = d.draw(np.random.default_rng(5), 7)
    rng = np.random.default_rng(5)

    def counts(n):
        return np.array([np.bincount(r, minlength=n) for r in rng.integers(0, n, (7, n))], float)
    if mode == stats.LINKED:
        assert d.srcs == ['a', 'b', 'c', 'd'] and ct is cs and np.array_equal(cs, counts(4))
    else:       # all source draws of the batch, then all target draws
        assert (d.srcs, d.tgts) == (['a', 'b', 'c'], ['b', 'c', 'd'])
        assert np.array_equal(cs, counts(3)) and np.array_equal(ct, counts(3))
    assert (cs.sum(1) == len(d.srcs)).all() and (ct.sum(1) == len(d.tgts)).all()


def test_linked_draw_weights_both_roles_of_a_speaker_alike():
    src, tgt = ['a', 'b', 'a', 'c'], ['b', 'a', 'c', 'a']
    for mode, same in ((stats.LINKED, True), (stats.INDEPENDENT, False)):
        d = stats.Design(src, tgt, mode)
        w = d.pair_weights(*d.draw(np.random.default_rng(0), 200))
        assert np.array_equal(w[:, 0], w[:, 1]) is same           # a->b and b->a: c_a c_b both, when linked


class Scripted(stats.Design):
    """A Design whose draws are given, one (c_src, c_tgt) batch per draw() call."""

    def __init__(self, src, tgt, mode, script):
        super().__init__(src, tgt, mode)
        self.script = [tuple(np.array(x, float) for x in b) for b in script]

    def draw(self, rng, m):
        cs, ct = self.script.pop(0)
        assert len(cs) == m
        return cs, ct


def test_zero_weight_replicates_are_redrawn_in_order_and_counted():
    # pairs a->b (3 errors on 0 words) and c->d (1 on 5); independent: sources a, c; targets b, d
    d = Scripted(['a', 'c'], ['b', 'd'], stats.INDEPENDENT, [
        ([[2, 0], [1, 1], [0, 2]], [[2, 0], [1, 1], [2, 0]]),   # weights (4, 0): no word; (1, 1); (0, 0): no pair
        ([[2, 0]], [[0, 2]]),                                    # replicate 0, redraw 1: (0, 0)
        ([[1, 1]], [[0, 2]]),                                    # replicate 0, redraw 2: (0, 2)
        ([[0, 2]], [[0, 2]])])                                   # replicate 2, redraw 3: (0, 4)
    est, reps, redraws = stats.bootstrap(d, [[3, 0], [1, 5]], lambda S: 100 * S[:, 0] / S[:, 1], den=(1,), B=3)
    assert est == 80.0 and reps.tolist() == [20.0, 80.0, 20.0] and redraws == 3 and not d.script


def test_full_sample_without_reference_words_raises():
    with pytest.raises(ValueError):
        stats.bootstrap(stats.Design(['a'], ['b']), [[2.0, 0.0]], lambda S: S[:, 0] / S[:, 1], den=(1,))


@pytest.mark.parametrize('mode', [stats.LINKED, stats.INDEPENDENT])
def test_bootstrap_sums_equal_the_definition(mode):
    spk = ring(9, 5)
    src, tgt = [v[0] for v in spk.values()], [v[1] for v in spk.values()]
    cols = np.column_stack([np.arange(9.0), [0, 0, 5, 0, 0, 0, 0, 0, 3]])        # words on two pairs only
    got = []
    stats.bootstrap(stats.Design(src, tgt, mode), cols, lambda S: got.append(S.copy()) or S[:, 0], den=(1,), B=800)
    S, redraws = ref_bootstrap(src, tgt, mode, cols, den=(1,), B=800)
    assert np.array_equal(got[1], S)
    _, _, red = stats.bootstrap(stats.Design(src, tgt, mode), cols, lambda S: S[:, 0], den=(1,), B=800)
    assert red == redraws > 0


def test_replicates_reproducible_and_shared_by_endpoints_on_the_same_pairs():
    spk = ring(40, 6)
    d = stats.Design([v[0] for v in spk.values()], [v[1] for v in spk.values()])
    x = np.random.default_rng(1).normal(size=40)
    mean = lambda S: S[:, 0] / S[:, -1]     # noqa: E731
    _, r1, _ = stats.bootstrap(d, x, mean, B=500)
    _, r2, _ = stats.bootstrap(d, x, mean, B=500)
    _, r3, _ = stats.bootstrap(d, 2 * x + 3, mean, B=500)
    _, r4, _ = stats.bootstrap(d, x, mean, B=500, seed=stats.SEED + 1)
    assert np.array_equal(r1, r2) and np.allclose(r3, 2 * r1 + 3) and not np.array_equal(r1, r4)


@pytest.mark.parametrize('mode', [stats.LINKED, stats.INDEPENDENT])
def test_compare_and_level_equal_the_definition(mode):
    spk = ring(16, 5)
    rng = np.random.default_rng(2)
    n = rng.integers(4, 15, 16)
    n[3] = 0                                            # a pair without reference words
    ea, eb = rng.integers(0, 30, 16), rng.integers(0, 12, 16)
    a = {p: wrow(p, int(ea[i]), int(n[i]), sim_wavlm=0.8 + 0.01 * i) for i, p in enumerate(spk)}
    b = {p: wrow(p, int(eb[i]), int(n[i]), sim_wavlm=0.85 - 0.002 * i) for i, p in enumerate(spk)}
    ids = list(spk)
    src, tgt = [spk[p][0] for p in ids], [spk[p][1] for p in ids]
    c = stats.compare(a, b, ids, spk, 'wer', mode=mode)
    S, red = ref_bootstrap(src, tgt, mode, np.column_stack([ea, n, eb, n]), den=(1, 3))
    reps = 100 * (S[:, 0] / S[:, 1] - S[:, 2] / S[:, 3])
    assert c['est'] == approx(100 * (ea.sum() - eb.sum()) / n.sum()) and (c['n'], c['mode']) == (16, mode)
    assert_close(c, **ci(reps), p=stats.p_two_sided(reps), redraws=red)
    c = stats.compare(a, b, ids, spk, 'sim_wavlm', mode=mode)
    d = np.array([a[p]['sim_wavlm'] - b[p]['sim_wavlm'] for p in ids])
    S, red = ref_bootstrap(src, tgt, mode, d)
    assert c['est'] == pytest.approx(d.mean())
    assert_close(c, **ci(S[:, 0] / S[:, 1]), p=stats.p_two_sided(S[:, 0] / S[:, 1]), redraws=red)
    lv = stats.level(a, ids, spk, 'wer_capped', mode=mode)
    S, red = ref_bootstrap(src, tgt, mode, np.column_stack([np.minimum(ea, n), n]), den=(1,))
    assert lv['est'] == approx(100 * np.minimum(ea, n).sum() / n.sum())
    assert_close(lv, **ci(100 * S[:, 0] / S[:, 1]), redraws=red)


# ── p, Holm, gate L ──────────────────────────────────────────

@pytest.mark.parametrize('reps, p', [
    (np.r_[np.full(9990, -1.0), np.full(10, 1.0)], 2 * 11 / 10001),
    (np.full(10000, 3.0), 2 / 10001),
    ([-3, -2, -1, 1, 2, 3, 4, 5, 6, 7], 2 * 4 / 11),
    ([-1, 0, 0, 2, 3], 1.0),                            # zeros count on both sides: 2 min(4, 5) / 6 > 1
    ([0.0, 1, 2, 3, 4, 5, 6], 2 * 2 / 8),
])
def test_p_two_sided_formula(reps, p):
    assert stats.p_two_sided(reps) == approx(p)


@pytest.mark.parametrize('pvals, rejected', [
    ([0.01, 0.04, 0.03, 0.005], [True, False, False, True]),     # 0.005 <= .0125, 0.01 <= .0167, 0.03 > .025 stops
    ([0.01, 0.02, 0.03], [True, True, True]),
    ([0.025, 0.05], [True, True]),                               # p_(i) <= alpha / (m - i + 1) includes equality
    ([0.001, None, 0.02], [True, False, True]),                  # None counts in m = 3 and is never rejected
    ([0.03, None], [False, False]),                              # with m = 1 (None dropped) 0.03 would be rejected
    ([None], [False]),
    ([], []),
])
def test_holm(pvals, rejected):
    assert stats.holm(pvals) == rejected


def test_holm_agrees_with_statsmodels():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = list(rng.uniform(0, 0.08, rng.integers(1, 10)))
        assert stats.holm(p) == multipletests(p, alpha=0.05, method='holm')[0].tolist()


def test_gate_l_rejects_only_below_the_margin():
    assert stats.gate_l({'hi': np.nextafter(0.01, 0)}) is True
    assert stats.gate_l({'hi': 0.01}) is False
    assert stats.gate_l({'hi': np.nextafter(0.01, 1)}) is False
    assert stats.gate_l({'hi': -0.5}) is True
    assert stats.gate_l({}) is None and stats.gate_l({'hi': None}) is None


def test_gate_l_strata_splits_strata_filters_source_e_and_takes_worst_max():
    pairs = dev_pairs(48, 2, 2)
    ids = [p['pair_id'] for p in pairs]
    b = {p: {'pair_id': p, 'sim_wavlm': 0.70 + 0.001 * i} for i, p in enumerate(ids)}
    a = {p: {'pair_id': p, 'sim_wavlm': b[p]['sim_wavlm'] + (-0.05 if stats.stratum(pp['genders']) == 'same'
                                                             else 0.05) + 0.002 * (i % 3)}
         for i, (p, pp) in enumerate(zip(ids, pairs))}
    gl = stats.gate_l_strata(a, b, pairs)
    for g, rejects in (('same', True), ('cross', False)):
        sub = [p['pair_id'] for p in pairs if stats.stratum(p['genders']) == g]
        assert gl[g]['n'] == len(sub) and gl[g]['n_no_source_E'] == 0 and gl[g]['rejects'] is rejects
        assert gl[g]['est'] == pytest.approx(np.mean([a[p]['sim_wavlm'] - b[p]['sim_wavlm'] for p in sub]))
    gl = stats.gate_l_strata(a, b, pairs, src_e={'f0', 'm1'})
    for g in stats.STRATA:
        sub = [p for p in pairs if stats.stratum(p['genders']) == g]
        keep = [p for p in sub if p['source_speaker'] in ('f0', 'm1')]
        assert (gl[g]['n'], gl[g]['n_no_source_E']) == (len(keep), len(sub) - len(keep))
    x = next(p['pair_id'] for p in pairs if stats.stratum(p['genders']) == 'same')
    gl = stats.gate_l_strata(dict(a, **{x: err(x)}), b, pairs)
    wc = gl['same']['worst_case']
    assert gl['same']['fail_a'] == 1 and 'worst_case' not in gl['cross']
    assert wc['worst_a'] == max(r['sim_wavlm'] for r in a.values() if r['pair_id'] != x) and wc['imputed_a'] == 1


# ── failed conversions ───────────────────────────────────────

def test_compare_wer_failures_and_worst_case_e_equals_n():
    spk = ring(8, 4)
    a = {p: wrow(p, 1, 10) for p in spk}
    b = {p: wrow(p, 3, 10) for p in spk}
    a['p00'] = err('p00')                   # A failed on p00 and has no row for p01; B failed on p02
    del a['p01']
    b['p02'] = err('p02')
    c = stats.compare(a, b, list(spk), spk, 'wer')
    assert (c['fail_a'], c['fail_b'], c['n']) == (2, 1, 5) and c['est'] == approx(10.0 - 30.0)
    wc = c['worst_case']
    # failed pairs: every reference word deleted (E = N, N from the other system's row)
    assert (wc['imputed_a'], wc['imputed_b'], wc['unimputable'], wc['n']) == (2, 1, 0, 8)
    assert wc['est'] == approx(100 * (10 + 10 + 1 + 5) / 80 - 100 * (3 + 3 + 10 + 15) / 80)
    a['p03'] = wrow('p03', 25, 10)         # 250 %: capped at 10 in the primary and the worst case
    cc = stats.compare(a, b, list(spk), spk, 'wer_capped')
    assert cc['est'] == approx(100 * 14 / 50 - 30.0)
    assert cc['worst_case']['est'] == approx(100 * (10 + 10 + 1 + 14) / 80 - 100 * 31 / 80)


def test_worst_case_when_both_failed_needs_n_ref():
    spk = ring(6, 3)
    a = {p: wrow(p, 2, 10) for p in spk}
    b = {p: wrow(p, 4, 10) for p in spk}
    a['p00'], b['p00'], a['p01'] = err('p00'), err('p00'), err('p01')
    wc = stats.compare(a, b, list(spk), spk, 'wer')['worst_case']
    assert (wc['imputed_a'], wc['imputed_b'], wc['unimputable'], wc['n']) == (1, 0, 1, 5)
    wc = stats.compare(a, b, list(spk), spk, 'wer', n_ref={'p00': 8, 'p01': 99})['worst_case']    # p01: B's N wins
    assert (wc['imputed_a'], wc['imputed_b'], wc['unimputable'], wc['n']) == (2, 1, 0, 6)
    assert wc['est'] == approx(100 * (8 + 10 + 4 * 2) / 58 - 100 * (8 + 4 * 5) / 58)


@pytest.mark.parametrize('worst, pick', [('min', min), ('max', max)])
def test_compare_continuous_worst_case(worst, pick):
    spk = ring(10, 5)
    a = {p: {'pair_id': p, 'sim_wavlm': 0.6 + 0.02 * i} for i, p in enumerate(spk)}
    b = {p: {'pair_id': p, 'sim_wavlm': 0.9 - 0.01 * i} for i, p in enumerate(spk)}
    a['p00'], b['p09'] = err('p00'), err('p09')
    a['p05'] = err('p05')
    b['p03'] = {'pair_id': 'p03', 'sim_wavlm': float('nan')}       # no error, not finite: undefined, not imputed
    ids = list(spk)[:8]                                              # p09 lies outside: B's worst is over all its rows
    c = stats.compare(a, b, ids, spk, 'sim_wavlm', worst=worst)
    assert (c['fail_a'], c['fail_b'], c['undefined'], c['n']) == (2, 0, 1, 5)
    wc = c['worst_case']
    wa = pick(r['sim_wavlm'] for r in a.values() if 'error' not in r)
    wb = pick(r['sim_wavlm'] for r in b.values() if 'error' not in r and np.isfinite(r['sim_wavlm']))
    assert (wc['worst_a'], wc['worst_b'], wc['imputed_a'], wc['imputed_b'], wc['n']) == (wa, wb, 2, 0, 7)
    keep = [p for p in ids if p != 'p03']
    va = [wa if 'error' in a[p] else a[p]['sim_wavlm'] for p in keep]
    assert wc['est'] == pytest.approx(np.mean(va) - np.mean([b[p]['sim_wavlm'] for p in keep]))
    assert stats.compare(a, b, ids, spk, 'sim_wavlm')['worst_case'] is None         # no worst given
    eq = stats.compare(a, dict(b, p01=err('p01'), p02=err('p02')), ids, spk, 'sim_wavlm', worst=worst)
    assert eq['fail_a'] == eq['fail_b'] == 2 and 'worst_case' not in eq          # equal failure counts


def test_margin_rows_and_stratum():
    m9 = stats.margin_rows({'x': {'sim_wavlm': 0.9}, 'y': {'sim_wavlm': 0.8}, 'z': err('z')},
                           {'x': {'sim_wavlm': 0.7}, 'z': {'sim_wavlm': 0.5}, 'w': {'sim_wavlm': float('nan')}})
    assert m9['x'] == {'m9': pytest.approx(-0.2)} and all('error' in m9[k] for k in 'yzw')
    assert [stats.stratum(g) for g in ('F2F', 'M2M', 'F2M', 'M2F', 'X')] == ['same', 'same', 'cross', 'cross', None]


def test_wilcoxon_signed_rank_and_rank_biserial():
    from scipy.stats import wilcoxon
    d = np.array([1.5, -0.5, 2.0, 0.0, 3.0, -1.0, 2.5, np.nan])
    w = stats.wilcoxon_rb(d)
    assert (w['n'], w['w_plus'], w['w_minus']) == (6, 18, 3) and w['r_rb'] == approx(15 / 21)
    assert w['p'] == approx(wilcoxon(d[np.isfinite(d) & (d != 0)]).pvalue)
    assert stats.wilcoxon_rb([1, 2, 3])['r_rb'] == 1.0 and stats.wilcoxon_rb([0, 0])['p'] is None


def test_load_results_keeps_the_last_row(tmp_path):
    f1, f2 = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    f1.write_text(json.dumps({'pair_id': 'p', 'system': 's', 'v': 1}) + '\n\n'
                  + json.dumps({'pair_id': 'q', 'system': 's', 'v': 2}) + '\n')
    f2.write_text(json.dumps({'pair_id': 'p', 'system': 's', 'v': 3}) + '\n')
    assert sorted(r['v'] for r in stats.load_results([str(f1), str(f2)])) == [2, 3]
    assert len(stats.load_results(str(f1))) == 2


# ── §5.3: variance components and size check ─────────────────

@pytest.mark.parametrize('capped', [False, True])
def test_wer_linearised_mean_is_the_corpus_wer_difference(capped):
    rng = np.random.default_rng(4)
    ea, eb, nw = rng.integers(0, 25, 60), rng.integers(0, 9, 60), rng.integers(4, 20, 60)
    assert (ea > nw).any()
    a = {i: wrow(i, int(ea[i]), int(nw[i])) for i in range(60)}
    b = {i: wrow(i, int(eb[i]), int(nw[i])) for i in range(60)}
    m = 'wer_capped' if capped else 'wer'
    want = stats.point(a, list(a), m)[0] - stats.point(b, list(b), m)[0]
    d = stats.wer_linearised(ea, eb, nw, capped=capped)
    assert d.mean() == pytest.approx(want, rel=1e-12)
    cap = np.minimum if capped else lambda e, n: e
    assert np.allclose(d, 100 * (cap(ea, nw) - cap(eb, nw)) / nw.mean())


def crossed(seed, n_src=30, k=3, reps=2, comps=(4.0, 1.0, 0.25, 0.0625)):
    """d_p = 1 + a_source + b_target + c_cell + e_p: n_src sources, k targets each, reps pairs per cell."""
    rs = np.random.default_rng(seed)
    sa, sb, sc, se = (math.sqrt(v) for v in comps)
    a, b = rs.normal(0, sa, n_src), rs.normal(0, sb, n_src)
    src = np.repeat(np.arange(n_src), k * reps)
    tgt = np.array([(i + 1 + j * (n_src // k)) % n_src for i in range(n_src) for j in range(k) for _ in range(reps)])
    cells = {}
    c = np.array([cells.setdefault((i, j), rs.normal(0, sc)) for i, j in zip(src, tgt)])
    return 1 + a[src] + b[tgt] + c + rs.normal(0, se, len(src)), [f's{i}' for i in src], [f't{j}' for j in tgt]


def test_variance_components_recover_simulated_crossed_components():
    true = (4.0, 1.0, 0.25, 0.0625)
    d, src, tgt = crossed(3, comps=true)
    vc = stats.variance_components(d, src, tgt)
    est = [vc[k] for k in ('sigma2_a', 'sigma2_b', 'sigma2_c', 'sigma2_e')]
    assert all(0.5 * v < e < 2 * v for e, v in zip(est, true)), est
    assert abs(vc['mu'] - 1) < 1.5 and isinstance(vc['converged'], bool)
    assert (vc['n'], vc['n_s'], vc['n_t'], vc['n_c']) == (180, 30, 30, 90)
    assert set(vc['reml_llf']) == {'/'.join(m) for m in stats.VC_METHODS} and not vc['fit_errors']
    assert vc['reml_llf'][vc['method']] == max(vc['reml_llf'].values())


def test_variance_components_list_failed_optimisers(monkeypatch):
    from statsmodels.regression.mixed_linear_model import MixedLM
    fit = MixedLM.fit

    def failing(self, *a, **kw):
        if kw.get('method') == ['powell']:
            raise np.linalg.LinAlgError('Singular matrix')
        return fit(self, *a, **kw)
    monkeypatch.setattr(MixedLM, 'fit', failing)
    d, src, tgt = crossed(3)
    vc = stats.variance_components(d, src, tgt)
    assert vc['fit_errors'] == {'powell': 'LinAlgError: Singular matrix'}
    assert set(vc['reml_llf']) == {'bfgs/lbfgs/cg', 'nm'} and vc['method'] in vc['reml_llf']

    def always(self, *a, **kw):
        raise ValueError('bad start')
    monkeypatch.setattr(MixedLM, 'fit', always)
    with pytest.raises(ValueError, match='no REML fit'):
        stats.variance_components(d, src, tgt)


def test_variance_components_degenerate_inputs():
    s20, t20 = [f's{i % 20}' for i in range(200)], [f's{(7 * i + 3) % 20}' for i in range(200)]
    rz = np.random.default_rng(0)
    vz = stats.variance_components(np.r_[np.zeros(190), rz.normal(0, 30, 10)], s20, t20)    # boundary fit
    assert vz['fit_errors'] and all(':' in m for m in vz['fit_errors'].values())
    assert set(vz['reml_llf']) | set(vz['fit_errors']) == {'/'.join(m) for m in stats.VC_METHODS}
    assert vz['method'] in vz['reml_llf'] and vz['reml_llf'][vz['method']] == max(vz['reml_llf'].values())
    with pytest.raises(ValueError, match='no REML fit'):                                     # every d_p equal
        stats.variance_components(np.zeros(200), s20, t20)


def test_predict_se_formula():
    vc = {'sigma2_a': 4.0, 'sigma2_b': 1.0, 'sigma2_c': 2.0, 'sigma2_e': 50.0}
    assert stats.predict_se(vc, {'n_s': 20, 'n_t': 20, 'n_c': 100, 'n': 1000}) == approx(math.sqrt(0.32))


def test_endpoint_pairs():
    v = vctk_pairs(5)
    assert stats.endpoint_pairs(v, 'wer') == ([p for p in v if p['wer_primary']], 0)
    assert stats.endpoint_pairs(v, 'mcd') == ([p for p in v if p['P']], 0)
    assert stats.endpoint_pairs(v, 'sim') == (v, 0) and stats.endpoint_pairs(v, 'utmos') == (v, 0)
    same = [p for p in v if p['genders'] in ('F2F', 'M2M')]
    keep, out = stats.endpoint_pairs(v, 'gate_same', {'sf0', 'sm2'})
    assert keep == [p for p in same if p['source_speaker'] in ('sf0', 'sm2')] and out == len(same) - len(keep) > 0


def counts(pairs):
    return {'n': len(pairs), 'n_s': len({p['source_speaker'] for p in pairs}),
            'n_t': len({p['target_speaker'] for p in pairs}),
            'n_c': len({(p['source_speaker'], p['target_speaker']) for p in pairs})}


VC_W = {'sigma2_a': 0.0, 'sigma2_b': 0.0, 'sigma2_c': 0.0, 'sigma2_e': 40.0}
VC_G = {'sigma2_a': 0.0, 'sigma2_b': 0.0, 'sigma2_c': 0.0, 'sigma2_e': 0.0005}
CONDS = [{'vctk2k': 'P1', 'ltc2k': 'P2', 'kind': 'wer', 'vc': VC_W},
         {'vctk2k': 'L1s', 'ltc2k': 'L2s', 'kind': 'gate_same', 'vc': VC_G, 'delta_dev': -0.01},
         {'vctk2k': 'L1c', 'ltc2k': 'L2c', 'kind': 'gate_cross', 'vc': VC_G, 'delta_dev': 0.02}]
SRC_E = {'vctk2k_5': {'sf0', 'sf1'}, 'vctk2k_10': {'sf0', 'sf1'}}


def test_size_check_mde_and_rebuild_rule_fails_with_5_passes_with_10():
    v5, v10, ltc = vctk_pairs(5), vctk_pairs(10), ltc_pairs()
    sc = stats.size_check(CONDS, {'vctk2k_5': v5, 'vctk2k_10': v10, 'ltc2k': ltc}, SRC_E)
    by = {(r['endpoint'], r['design']): r for r in sc['rows']}
    assert set(by) == {(n, d) for n in ('P1', 'L1s', 'L1c') for d in ('vctk2k_5', 'vctk2k_10')} | {
        ('P2', 'ltc2k'), ('L2s', 'ltc2k'), ('L2c', 'ltc2k')}
    for dn, pairs in (('vctk2k_5', v5), ('vctk2k_10', v10), ('ltc2k', ltc)):
        r = by['P2' if dn == 'ltc2k' else 'P1', dn]
        n = counts([p for p in pairs if p['wer_primary']])
        se = math.sqrt(40 / n['n'])
        assert {k: r[k] for k in n} == n and r['se'] == approx(se)
        assert r['mde'] == approx(3.53 * se) and r['threshold'] == 2.0 and r['pass'] is (3.53 * se <= 2.0)
    assert by['P1', 'vctk2k_5']['pass'] is False and by['P1', 'vctk2k_10']['pass'] is True     # 2.50, then 1.77
    assert sc['rebuild'] == {'conditions': ['P1', 'L1s', 'L1c'], 'use': '10', 'triggers': ['P1'], 'evaluable': True,
                             'underpowered': ['L1c'], 'underpowered_design': 'vctk2k_10'}


def test_size_check_gate_power_on_pairs_with_a_source_e_set():
    v5 = vctk_pairs(5)
    sc = stats.size_check(CONDS, {'vctk2k_5': v5, 'vctk2k_10': None, 'ltc2k': ltc_pairs()}, SRC_E)
    by = {(r['endpoint'], r['design']): r for r in sc['rows']}
    same = [p for p in v5 if stats.stratum(p['genders']) == 'same']
    keep = [p for p in same if p['source_speaker'] in SRC_E['vctk2k_5']]
    r = by['L1s', 'vctk2k_5']
    se = math.sqrt(0.0005 / len(keep))
    assert r['n'] == len(keep) and r['n_left_out_no_source_E'] == len(same) - len(keep) > 0
    assert r['se'] == approx(se) and r['lhs'] == approx(2.80 * se) and r['rhs'] == approx(0.02)
    assert 2.80 * se <= 0.02 and r['pass'] is True and 'note' not in r
    rc = by['L1c', 'vctk2k_5']          # Δ_dev,g = 0.02 >= 0.01: no power under any design
    assert rc['pass'] is False and 'no power' in rc['note'] and rc['rhs'] == approx(-0.01)
    lt = by['L2s', 'ltc2k']             # LTC-2k: every source counts, never filtered
    assert lt['n'] == sum(stats.stratum(p['genders']) == 'same' for p in ltc_pairs()) and 'note' not in lt
    sc2 = stats.size_check(CONDS, {'vctk2k_5': v5, 'vctk2k_10': None, 'ltc2k': ltc_pairs()},
                           dict(SRC_E, ltc2k={'lf0'}))
    assert sc2['rows'] == sc['rows']
    assert sc['rebuild'] == {'conditions': ['P1', 'L1s', 'L1c'], 'use': None, 'triggers': [], 'evaluable': False,
                             'underpowered': ['P1', 'L1c'], 'underpowered_design': 'vctk2k_5'}


def test_size_check_without_trigger_keeps_5_targets_and_lists_underpowered():
    vc_tiny = dict(VC_W, sigma2_e=1.0)                  # P1 passes with 5 targets already
    conds = [dict(CONDS[0], vc=vc_tiny), CONDS[1], CONDS[2],
             {'vctk2k': 'P7', 'ltc2k': None, 'kind': 'mcd', 'vc': dict(VC_W, sigma2_e=4.0)}]   # fails with both
    sc = stats.size_check(conds, {'vctk2k_5': vctk_pairs(5), 'vctk2k_10': vctk_pairs(10), 'ltc2k': None})
    rb = sc['rebuild']
    assert rb['use'] == '5' and rb['triggers'] == [] and rb['underpowered'] == ['L1c', 'P7']
    by = {(r['endpoint'], r['design']): r for r in sc['rows']}
    assert by['P7', 'vctk2k_5']['pass'] is False and by['P7', 'vctk2k_10']['pass'] is False
    assert 'no source-E file' in by['L1s', 'vctk2k_5']['note']               # src_e not given: every source counts
    assert by['L1s', 'vctk2k_5']['n'] == sum(stats.stratum(p['genders']) == 'same' for p in vctk_pairs(5))


def test_rebuild_rule_ignores_a_condition_not_evaluable_with_5_targets():
    conds = [CONDS[1]]          # L1s passes with 10 targets; with 5 no source has an E set, so it has no pairs
    sc = stats.size_check(conds, {'vctk2k_5': vctk_pairs(5), 'vctk2k_10': vctk_pairs(10), 'ltc2k': None},
                          {'vctk2k_5': set(), 'vctk2k_10': SRC_E['vctk2k_10']})
    by = {(r['endpoint'], r['design']): r for r in sc['rows']}
    assert by['L1s', 'vctk2k_5']['n'] == 0 and by['L1s', 'vctk2k_5']['pass'] is None
    assert by['L1s', 'vctk2k_10']['pass'] is True
    assert sc['rebuild']['use'] == '5' and sc['rebuild']['triggers'] == [] and sc['rebuild']['underpowered'] == ['L1s']


def test_gate_has_no_power_at_delta_dev_of_the_margin_even_without_variance():
    zero = dict(VC_G, sigma2_e=0.0)
    conds = [dict(CONDS[1], vc=zero, delta_dev=0.01), dict(CONDS[2], vc=zero, delta_dev=np.nextafter(0.01, 0))]
    sc = stats.size_check(conds, {'vctk2k_5': vctk_pairs(5), 'vctk2k_10': None, 'ltc2k': None}, SRC_E)
    s, c = sc['rows']
    assert s['lhs'] == 0.0 and s['rhs'] == 0.0 and s['pass'] is False and 'no power' in s['note']
    assert c['pass'] is True and 'note' not in c


def test_size_check_se_uses_every_component_and_skips_conditions_without_fit():
    vc = {'sigma2_a': 2.0, 'sigma2_b': 1.0, 'sigma2_c': 3.0, 'sigma2_e': 40.0}
    v10 = vctk_pairs(10)
    conds = [{'vctk2k': 'K3', 'ltc2k': 'K4', 'kind': 'sim', 'vc': vc},
             {'vctk2k': 'K5', 'ltc2k': 'K6', 'kind': 'utmos', 'vc': None}]
    sc = stats.size_check(conds, {'vctk2k_5': vctk_pairs(5), 'vctk2k_10': v10, 'ltc2k': None})
    r = {(r['endpoint'], r['design']): r for r in sc['rows']}
    n = counts(v10)
    assert r['K3', 'vctk2k_10']['se'] == approx(math.sqrt(2 / n['n_s'] + 1 / n['n_t'] + 3 / n['n_c'] + 40 / n['n']))
    assert r['K3', 'vctk2k_10']['threshold'] == 0.010
    assert r['K5', 'vctk2k_10']['se'] is None and r['K5', 'vctk2k_10']['pass'] is None
    assert sc['rebuild']['underpowered'] == ['K3', 'K5']
