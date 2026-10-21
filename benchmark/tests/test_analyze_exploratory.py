"""Tests of benchmark/analyze_exploratory.py, the exploratory addition X1 (DEVIATIONS_v1.md item 5): CRT-AL against
each baseline at the 3 s budget on the family-S endpoints, with analyze_tests.py's statistics. Synthetic metrics rows
come from the world of test_analyze_tests.py; Seed-VC and FACodec get their own effects here."""
import json

import pytest

import analyze_exploratory as ax
import analyze_tests as at
import test_analyze_tests as tat

RECORD = dict(tat.RECORD, baselines=[{'id': 'S5'}, {'id': 'S6'}, {'id': 'S7'}, {'id': 'S8'}])
SYSTEMS = at.systems_from_record(RECORD)
BASELINES = ['knnvc', 'freevc', 'seedvc', 'facodec']
X1_BASE = {'seedvc': dict(rate=0.05, ins=0, sim=0.92, utmos=3.8, mcd=7.1, src=0.74),
           'facodec': dict(rate=0.05, ins=0, sim=0.88, utmos=3.6, mcd=6.8, src=0.78)}
AT3 = ('faiss', 'crt_tau0.4', 'crt_tau0.9', 'cand_pm18_rq')


@pytest.fixture(autouse=True)
def x1_systems(monkeypatch):
    for k, v in X1_BASE.items():
        monkeypatch.setitem(tat.BASE, k, v)


def rows(pairs, systems, budget=3, cfg=None):
    R = {}
    for s in systems:
        tgt, src = tat.rows_for(pairs, s, budget, cfg=(cfg or {}).get(s))
        R[s] = {r['pair_id']: r for r in tgt}
        R[s + '@src'] = {r['pair_id']: r for r in src}
    return R


def make_sets(extra30=None, cfg=None):
    vp, lp = tat.vctk_pairs(), tat.ltc_pairs()
    out = {}
    for name, pairs in (('vctk2k', vp), ('ltc2k', lp)):
        R = {3: rows(pairs, AT3 + tuple(BASELINES), cfg=cfg)}
        if extra30:
            R[30] = rows(pairs, extra30[0], 30, cfg=extra30[1])
        out[name] = at.TestSet(name, pairs, R, {p['source_speaker'] for p in vp} if name == 'vctk2k' else None)
    return out


def test_family_shape_and_labels():
    res = ax.analyse(make_sets(), SYSTEMS, BASELINES)
    hyps = res['x1']['hypotheses']
    assert [h['id'] for h in hyps] == [f'X1-{b}-{i}' for b in BASELINES for i in range(1, 7)]
    assert [(h['set'], h['quantity']) for h in hyps[:6]] == at.S_ENDPOINTS
    assert all(h['sys_a'] == 'cand_pm18_rq' for h in hyps)
    assert {h['id'] for h in hyps if h.get('flag') == 'VCTK-seen'} == {'X1-freevc-1', 'X1-freevc-3', 'X1-freevc-5'}
    assert 'guide only' in res['x1']['holm'] and res['x1']['holm'].startswith('over 24')
    assert hyps[0]['label'] == 'capped WER %'
    assert res['sensitivity']['wer_definition']['family']['hypotheses'][0]['label'] == 'WER %'


def test_directions_follow_the_synthetic_effects():
    res = ax.analyse(make_sets(), SYSTEMS, BASELINES)
    h = {x['id']: x for x in res['x1']['hypotheses']}
    # CRT-AL: rate 0.2, SIM 0.80, UTMOS 2.6; Seed-VC: rate 0.05, SIM 0.92, UTMOS 3.8
    for i in (1, 2):
        assert h[f'X1-seedvc-{i}']['est'] > 0 and h[f'X1-seedvc-{i}']['favours'] in ('seedvc', None)
    for i in (3, 4, 5, 6):
        assert h[f'X1-seedvc-{i}']['est'] < 0
    assert h['X1-seedvc-5']['reject'] and h['X1-seedvc-5']['favours'] == 'seedvc'
    # FreeVC's synthetic SIM (0.75) is below CRT-AL's (0.80): the difference favours CRT-AL
    assert h['X1-freevc-3']['est'] > 0


def test_only_the_3s_rows_count():
    """Rows at 30 s, where CRT-AL would be perfect and Seed-VC poor, must not move any 3 s comparison."""
    plain = ax.analyse(make_sets(), SYSTEMS, BASELINES)
    flipped = ax.analyse(make_sets(extra30=(('cand_pm18_rq', 'seedvc'),
                                            {'cand_pm18_rq': dict(rate=0.0, sim=0.99, utmos=4.5),
                                             'seedvc': dict(rate=0.9, sim=0.5, utmos=1.0)})), SYSTEMS, BASELINES)
    for a, b in zip(plain['x1']['hypotheses'], flipped['x1']['hypotheses']):
        assert a['est'] == b['est'] and a['p'] == b['p']
    assert set(plain['levels']['vctk2k']) == set(AT3) | set(BASELINES)


def test_required_reports_short_and_missing_systems():
    sets = make_sets()
    assert ax.required(sets, 'cand_pm18_rq', BASELINES) == []
    del sets['ltc2k'].R[3]['seedvc@src']
    k = next(iter(sets['vctk2k'].R[3]['facodec']))
    del sets['vctk2k'].R[3]['facodec'][k]
    probs = ax.required(sets, 'cand_pm18_rq', BASELINES)
    assert any(p.startswith('vctk2k facodec:') for p in probs) and 'ltc2k seedvc@src: no rows' in probs


# ── files and main ───────────────────────────────────────────

def full_inputs(tmp_path, x1_sha=tat.SEL, x1_extra=(), drop_x1=None, no_x1_done=False):
    vp, lp = tat.vctk_pairs(), tat.ltc_pairs()
    m, x1 = tmp_path / 'm', tmp_path / 'x1'
    for name, pairs in (('vctk2k', vp), ('ltc2k', lp)):
        R = rows(pairs, AT3 + tuple(BASELINES))
        pick = lambda names: {s: v for s, v in R.items() if s.split('@')[0] in names}
        tat.write_m(m, name, 't3', 3, pick(AT3))
        tat.write_m(m, name, 't4', 3, pick(('knnvc',)))
        tat.write_m(x1, name, 'm', 3, pick(tuple(s for s in ax.X1_SYSTEMS if s != drop_x1) + tuple(x1_extra)))
    for part in ('t1', 't2', 't3', 't4'):
        (m / part).mkdir(parents=True, exist_ok=True)
        (m / part / 'm.done').write_text(f'job 1 2026-10-20 10:00:00 selection sha256 {tat.SEL}\n')
    if not no_x1_done:
        (x1 / 'x1.done').write_text(f'job 9 2026-10-21 12:00:00 selection sha256 {x1_sha}\n')
    paths = {}
    for name, pairs in (('vctk', vp), ('ltc', lp)):
        paths[name] = tmp_path / f'{name}.jsonl'
        paths[name].write_text(''.join(json.dumps(p) + '\n' for p in pairs))
    paths['src_e'] = tmp_path / 'src_e.json'
    paths['src_e'].write_text(json.dumps({p['source_speaker']: {'paths': ['/x.flac']} for p in vp}))
    paths['record'] = tmp_path / 'record.json'
    paths['record'].write_text(json.dumps(RECORD))
    return m, x1, paths


def argv(m, x1, paths, out):
    return ['--m_dir', str(m), '--x1_dir', str(x1), '--vctk_pairs', str(paths['vctk']), '--ltc_pairs',
            str(paths['ltc']), '--source_e', str(paths['src_e']), '--record', str(paths['record']), '--out', str(out)]


def test_main_end_to_end(tmp_path, capsys):
    m, x1, paths = full_inputs(tmp_path)
    out = tmp_path / 'EXPLORATORY_X1_v1.md'
    assert ax.main(argv(m, x1, paths, out)) == 0
    md, res = out.read_text(), json.loads((tmp_path / 'EXPLORATORY_X1_v1.json').read_text())
    assert md.startswith('# Exploratory addition X1') and '**Exploratory.**' in md and 'guide only' in md
    assert 'X1-facodec-6' in md and 'VCTK-seen' in md and '### Levels, ltc2k, 3 s (descriptive)' in md
    assert len(res['x1']['hypotheses']) == 24 and res['baselines'] == BASELINES
    assert res['x1_done'] == f'job 9 2026-10-21 12:00:00 selection sha256 {tat.SEL}'
    assert len(res['inputs']) == 4 + 2 * 3 * 2          # record, 2 lists, source E; t3, t4, x1 files x 2 per set
    assert all(len(x['sha256']) == 64 for x in res['inputs'])
    assert res['eer']['vctk2k']['seedvc'] is not None
    assert len(res['analyze_tests_sha256']) == 64 and len(res['script_sha256']) == 64


def test_main_refuses_x1_under_another_selection(tmp_path):
    m, x1, paths = full_inputs(tmp_path, x1_sha='b' * 64)
    with pytest.raises(SystemExit, match='not written under the selection.env'):
        ax.main(argv(m, x1, paths, tmp_path / 'r.md'))


def test_main_refuses_unfinished_x1(tmp_path):
    m, x1, paths = full_inputs(tmp_path, no_x1_done=True)
    with pytest.raises(SystemExit, match='job X1 has not finished'):
        ax.main(argv(m, x1, paths, tmp_path / 'r.md'))


def test_main_refuses_a_system_from_both_jobs(tmp_path):
    m, x1, paths = full_inputs(tmp_path, x1_extra=('knnvc',))
    with pytest.raises(SystemExit, match='knnvc at 3 s comes from both'):
        ax.main(argv(m, x1, paths, tmp_path / 'r.md'))


def test_main_refuses_a_missing_baseline(tmp_path):
    m, x1, paths = full_inputs(tmp_path, drop_x1='seedvc')
    with pytest.raises(SystemExit, match='seedvc: 0 rows'):
        ax.main(argv(m, x1, paths, tmp_path / 'r.md'))


def test_main_refuses_a_record_without_the_baselines(tmp_path):
    m, x1, paths = full_inputs(tmp_path)
    paths['record'].write_text(json.dumps(tat.RECORD))         # S5, S6 only
    with pytest.raises(SystemExit, match="does not list"):
        ax.main(argv(m, x1, paths, tmp_path / 'r.md'))
