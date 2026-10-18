"""Tests of benchmark/lab_rows.py (ANALYSIS_PLAN_v1.md §8.1 item 5, §3.1 M8): the merge that renames decode-lab rows
lab_<system> -> <tag>_<system>, sim_src_wavs from the pair list or a source-E file, the @src rows (crt_anchor_lab.py's
transformation), and the refusals (rows not named lab_*, duplicates, @src inputs, output collisions)."""
import copy
import json
import os
import subprocess
import sys

import pytest

import crt_anchor_lab
import lab_rows
import metrics
from fixtures_rows import (conv_row, dev_pairs, lab_row, ltc_pairs, pair_row, read_jsonl, stub_anchor_lab,
                           vctk_pairs, write_jsonl)

CAND = ('pm_rq_T1', 'pm18_rq', 'gv1_pm_cont_T1', 'gv1_shrink0.85_cont', 'gv0.5_shrink0.75_cont')   # C1-C5
FRONT = ('us32_w0.2_0.2_0.1', 'us32_w0.5_0.3_0.3', 'lmbr_k2_g1_T1')                              # S1u, E17


def make_lab(out_dir, faiss_rows, systems):
    """A crt_decode_lab.py --out_dir: manifest.jsonl with a lab_<system> row per pair and system, pair-major."""
    rows = [lab_row(f, s, out_dir) for f in faiss_rows for s in systems]
    write_jsonl(os.path.join(out_dir, 'manifest.jsonl'), rows)
    return rows


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['lab_rows.py', *map(str, argv)])
    lab_rows.main()


def refused(monkeypatch, *argv):
    """The ERROR message of a refused invocation."""
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *argv)
    assert isinstance(e.value.code, str) and e.value.code.startswith('ERROR'), e.value.code
    return e.value.code


@pytest.fixture
def dev(tmp_path):
    """The Dev-like pair list on disk, run_conversions faiss rows, and two decode-lab out_dirs (cand, front) as
    job_d0.slurm step 7 writes them."""
    pairs = dev_pairs()
    path = write_jsonl(tmp_path / 'pairs' / 'dev.jsonl', pairs)
    faiss = [conv_row(p, 'faiss', '/rc') for p in pairs]
    cand = make_lab(str(tmp_path / 'lab' / 'cand'), faiss, CAND)
    front = make_lab(str(tmp_path / 'lab' / 'front'), faiss, FRONT)
    return {'pairs': pairs, 'path': path, 'faiss': faiss, 'cand': cand, 'front': front, 'lab': tmp_path / 'lab'}


def round4_merge(dirs_tags):
    """The inline merge step of run_round4.slurm (the rename lab_rows.merge reproduces), apart from file output."""
    rows = []
    for d, tag in dirs_tags:
        f = f'{d}/manifest.jsonl'
        if os.path.exists(f):
            for line in open(f):
                r = json.loads(line)
                r['system'] = r['system'].replace('lab_', f'{tag}_', 1)
                rows.append(r)
    return rows


def spk_rows_of(pairs):
    """crt_anchor_lab.py main's spk_rows: the first pair row of every target speaker."""
    out = {}
    for r in pairs:
        out.setdefault(r['target_speaker'], r)
    return out


# ── merge ────────────────────────────────────────────────────

def test_merge_renames_lab_rows_by_tag(dev):
    labs = [(str(dev['lab'] / 'cand'), 'cand'), (str(dev['lab'] / 'front'), 'front')]
    rows, report = lab_rows.merge(labs, [])
    want = [dict(r, system='cand_' + r['system'][4:]) for r in dev['cand']] + \
           [dict(r, system='front_' + r['system'][4:]) for r in dev['front']]
    assert rows == want                                       # names changed, every other field and the order kept
    assert {r['system'] for r in rows} == {'cand_' + s for s in CAND} | {'front_' + s for s in FRONT}
    assert report == [f'cand: 120 rows, 5 systems, 24 pairs <- {labs[0][0]}/manifest.jsonl',
                      f'front: 72 rows, 3 systems, 24 pairs <- {labs[1][0]}/manifest.jsonl']


def test_merge_equals_run_round4_rename(dev, tmp_path):
    """The rename is run_round4.slurm's: only the first lab_ is replaced, a missing manifest is skipped."""
    d = str(tmp_path / 'odd')
    write_jsonl(f'{d}/manifest.jsonl', [dict(dev['faiss'][0], system='lab_knn_lab_rq')])
    labs = [(str(dev['lab'] / 'cand'), 'cand'), (d, 'r4p'), (str(tmp_path / 'nowhere'), 'x'),
            (str(dev['lab'] / 'front'), 'front')]
    rows, report = lab_rows.merge(labs, [])
    assert rows == round4_merge(labs)
    assert rows[120]['system'] == 'r4p_knn_lab_rq'
    assert report[2] == f'x: {tmp_path}/nowhere/manifest.jsonl missing, skipped'


def test_merge_keeps_manifest_system_names(dev, tmp_path):
    copies = [conv_row(p, 'mimi_copy_src', '/mc') for p in dev['pairs']] + \
             [conv_row(p, 'mimi_copy_P', '/mc') for p in dev['pairs'] if p['P']]
    natural = [dict(f, system='source_copy', wav=f['source_wav']) for f in dev['faiss']]
    f1, f2 = write_jsonl(tmp_path / 'mc.jsonl', copies), write_jsonl(tmp_path / 'natural.jsonl', natural)
    rows, report = lab_rows.merge([(str(dev['lab'] / 'cand'), 'cand')], [f1, f2])
    assert rows[120:] == copies + natural
    assert report[1:] == [f'36 rows, 2 systems, 24 pairs (pairs per system 12-24) <- {f1}',
                          f'24 rows, 1 systems, 24 pairs <- {f2}']


def test_rows_not_named_lab_are_refused(dev, tmp_path):
    d = str(tmp_path / 'anc')
    write_jsonl(f'{d}/manifest.jsonl', [dict(dev['faiss'][0], system='anc_R3_s0_pm_rq_T1')])
    with pytest.raises(SystemExit) as e:
        lab_rows.merge([(d, 'cand')], [])
    assert str(e.value).startswith(f'ERROR: {d}/manifest.jsonl has rows not named lab_*')


@pytest.mark.parametrize('case', ['same lab twice', 'renamed lab row in a manifest', 'repeated manifest row'])
def test_duplicates_are_refused(dev, tmp_path, monkeypatch, case):
    cand = str(dev['lab'] / 'cand')
    if case == 'same lab twice':
        labs, mans = [(cand, 'cand'), (cand, 'cand')], []
    elif case == 'renamed lab row in a manifest':
        labs = [(cand, 'cand')]
        row = dev['cand'][7]
        mans = [write_jsonl(tmp_path / 'm.jsonl', [dict(row, system='cand_' + row['system'][4:])])]
    else:
        labs, mans = [], [write_jsonl(tmp_path / 'm.jsonl', dev['faiss'] + dev['faiss'][3:4])]
    want = 'ERROR: 120 (pair_id, system)' if case == 'same lab twice' else 'ERROR: 1 (pair_id, system)'
    with pytest.raises(SystemExit) as e:
        lab_rows.merge(labs, mans)
    assert str(e.value).startswith(want + ' occur more than once in the inputs, e.g. (')
    out = tmp_path / 'out' / 'manifest.jsonl'
    msg = refused(monkeypatch, '--pairs', dev['path'], *sum((['--lab', d, t] for d, t in labs), []),
                  *sum((['--manifest', f] for f in mans), []), '--out', out)
    assert msg.startswith(want) and not os.path.exists(tmp_path / 'out')


@pytest.mark.parametrize('where', ['manifest', 'lab'])
def test_src_rows_in_the_inputs_are_refused(dev, tmp_path, where):
    src = lab_rows.src_rows([dict(dev['faiss'][0], sim_src_wavs=['/e.flac'])])
    if where == 'manifest':
        labs, mans = [], [write_jsonl(tmp_path / 'm_src.jsonl', src)]
    else:
        d = str(tmp_path / 'lab_src')
        write_jsonl(f'{d}/manifest.jsonl', [dict(src[0], system='lab_pm_rq_T1@src')])
        labs, mans = [(d, 'cand')], []
    with pytest.raises(SystemExit) as e:
        lab_rows.merge(labs, mans)
    assert str(e.value).startswith('ERROR: the inputs hold @src rows')


def test_refusal_exits_with_status_1_and_writes_nothing(dev, tmp_path):
    """The command line as job_d0.slurm runs it: an @src input is an ERROR on stderr, exit status 1, no output."""
    f = write_jsonl(tmp_path / 'm_src.jsonl', lab_rows.src_rows([dict(dev['faiss'][0], sim_src_wavs=['/e.flac'])]))
    out = tmp_path / 'rows' / 'conv.jsonl'
    p = subprocess.run([sys.executable, lab_rows.__file__, '--pairs', dev['path'], '--manifest', f, '--out', out],
                       capture_output=True, text=True)
    assert p.returncode == 1 and p.stderr.startswith('ERROR: the inputs hold @src rows'), p.stderr
    assert not os.path.exists(out) and not os.path.exists(tmp_path / 'rows' / 'conv_src.jsonl')


def test_no_input_is_refused(dev, tmp_path, monkeypatch):
    assert refused(monkeypatch, '--pairs', dev['path'], '--out', tmp_path / 'o.jsonl') == \
        'ERROR: give at least one --lab or --manifest'


# ── outputs ──────────────────────────────────────────────────

@pytest.mark.parametrize('case', ['out == out_src', 'out == --manifest', 'out == --lab manifest', 'out == --pairs',
                                  'out == --source_e', 'out_src == --manifest', 'default out_src == --manifest',
                                  'out spelled with ..', 'out a symlink to --manifest'])
def test_output_collisions_are_refused(dev, tmp_path, monkeypatch, case):
    man = write_jsonl(tmp_path / 'rc' / 'manifest.jsonl', dev['faiss'])
    src_e = tmp_path / 'source_E.json'
    src_e.write_text(json.dumps({'p225': ['/e.flac']}))
    lab_man = str(dev['lab'] / 'cand' / 'manifest.jsonl')
    other = str(tmp_path / 'out' / 'merged.jsonl')
    out, out_src = {
        'out == out_src': (other, other), 'out == --manifest': (man, ''), 'out == --lab manifest': (lab_man, ''),
        'out == --pairs': (dev['path'], ''), 'out == --source_e': (str(src_e), other + '2'),
        'out_src == --manifest': (other, man), 'default out_src == --manifest': (str(tmp_path / 'rc' / 'manifest'), ''),
        'out spelled with ..': (str(tmp_path / 'rc' / '..' / 'rc' / 'manifest.jsonl'), ''),
        'out a symlink to --manifest': (str(tmp_path / 'link.jsonl'), '')}[case]
    if case == 'default out_src == --manifest':      # --out rc/manifest -> rc/manifest_src, here an input
        man = write_jsonl(tmp_path / 'rc' / 'manifest_src', dev['faiss'][:2])
    if case == 'out a symlink to --manifest':
        os.symlink(man, out)
    inputs = [dev['path'], man, lab_man, str(src_e)]
    before = {f: open(f).read() for f in inputs}
    argv = ['--pairs', dev['path'], '--lab', dev['lab'] / 'cand', 'cand', '--manifest', man, '--out', out]
    argv += ['--out_src', out_src] if out_src else []
    argv += ['--source_e', src_e] if case == 'out == --source_e' else []
    assert 'must differ from each other and from every input' in refused(monkeypatch, *argv)
    assert {f: open(f).read() for f in inputs} == before
    assert not os.path.exists(other) and not os.path.exists(tmp_path / 'out')


@pytest.mark.parametrize('out, out_src', [('m.jsonl', 'm_src.jsonl'), ('rows/conv.jsonl', 'rows/conv_src.jsonl'),
                                          ('merged', 'merged_src')])
def test_default_out_src_name(dev, tmp_path, monkeypatch, out, out_src):
    run_main(monkeypatch, '--pairs', dev['path'], '--lab', dev['lab'] / 'cand', 'cand', '--out', tmp_path / out)
    assert len(read_jsonl(tmp_path / out)) == 120 and len(read_jsonl(tmp_path / out_src)) == 120


def test_missing_lab_manifest_is_skipped(dev, tmp_path, monkeypatch, capsys):
    out = tmp_path / 'lab' / 'manifest.jsonl'
    run_main(monkeypatch, '--pairs', dev['path'], '--lab', tmp_path / 'nowhere', 'cand', '--lab', dev['lab'] / 'front',
             'front', '--out', out)
    log = capsys.readouterr().out
    assert f'cand: {tmp_path}/nowhere/manifest.jsonl missing, skipped' in log
    assert [r['system'] for r in read_jsonl(out)] == ['front_' + r['system'][4:] for r in dev['front']]
    assert len(read_jsonl(tmp_path / 'lab' / 'manifest_src.jsonl')) == 72


# ── sim_src_wavs ─────────────────────────────────────────────

def test_source_e_sets_from_the_first_pair_row_of_each_target():
    pairs = dev_pairs()
    rows_226 = [i for i, r in enumerate(pairs) if r['target_speaker'] == 'p226']
    for i in rows_226[1:]:                                   # every later row of target p226 has another E
        pairs[i] = dict(pairs[i], E={'paths': ['/other/e.flac'], 'durs': [3.0], 'total_s': 3.0})
    e = lab_rows.source_e_sets(pairs)
    assert e == {s: r['E']['paths'] for s, r in spk_rows_of(pairs).items()}
    assert e['p226'] != ['/other/e.flac'] and set(e) == {'p225', 'p226', 'p227', 'p228'}


def test_source_e_file_in_both_formats(tmp_path):
    """make_pairs.py's {spk: {paths, durs, total_s}} and the plain {spk: [paths]} are read as the paths."""
    f = tmp_path / 'vctk2k_source_E.json'
    f.write_text(json.dumps({'p301': {'paths': ['/a.flac', '/b.flac'], 'durs': [3.0, 4.0], 'total_s': 7.0},
                             'p302': ['/c.flac'], 'p303': []}, indent=1))
    assert lab_rows.source_e_sets(vctk_pairs(), str(f)) == {'p301': ['/a.flac', '/b.flac'], 'p302': ['/c.flac'],
                                                            'p303': []}


@pytest.mark.parametrize('content', [['/a.flac'], {'p301': '/a.flac'}, {'p301': {'durs': [3.0]}},
                                     {'p301': ['/a.flac', 3]}, {'p301': {'paths': '/a.flac'}}])
def test_bad_source_e_file_is_refused(tmp_path, content):
    f = tmp_path / 'source_E.json'
    f.write_text(json.dumps(content))
    with pytest.raises(SystemExit) as e:
        lab_rows.source_e_sets(vctk_pairs(), str(f))
    assert str(e.value).startswith(f'ERROR: {f}')


def test_add_sim_src_and_the_speakers_without_e():
    pairs = vctk_pairs()
    rows = [conv_row(p, 'faiss', '/rc') for p in pairs]
    no_e = lab_rows.add_sim_src(rows, pairs, {'p301': ['/a.flac'], 'p302': []})     # p302 empty, p303 absent
    assert no_e == ['p302', 'p303']
    assert [r['sim_src_wavs'] for r in rows] == [['/a.flac'] if r['source_speaker'] == 'p301' else None for r in rows]
    assert lab_rows.add_sim_src(rows, pairs, {'p301': ['/a.flac'], 'p302': []}) == ['p302', 'p303']   # idempotent


@pytest.mark.parametrize('case', ['pair_id not in --pairs', 'other source speaker', 'other sim_src_wavs'])
def test_add_sim_src_refuses_rows_that_do_not_fit(case):
    pairs = dev_pairs()
    row = conv_row(pairs[0], 'faiss', '/rc')
    if case == 'pair_id not in --pairs':
        row['pair_id'] = 'dev__p999_001__p225'
    elif case == 'other source speaker':
        row['source_speaker'] = 'p226'
    else:
        row['sim_src_wavs'] = ['/other.flac']
    with pytest.raises(SystemExit) as e:
        lab_rows.add_sim_src([row], pairs, lab_rows.source_e_sets(pairs))
    assert str(e.value).startswith(f'ERROR: {row["pair_id"]} faiss: ')


def test_add_sim_src_keeps_an_equal_existing_value():
    pairs = dev_pairs()
    e = lab_rows.source_e_sets(pairs)
    row = conv_row(pairs[0], 'anc_faiss', '/anc', sim_src_wavs=list(e[pairs[0]['source_speaker']]))
    assert lab_rows.add_sim_src([row], pairs, e) == [] and row['sim_src_wavs'] == e[pairs[0]['source_speaker']]
    row = conv_row(pairs[0], 'anc_faiss', '/anc', sim_src_wavs=None)            # null is filled in
    assert lab_rows.add_sim_src([row], pairs, e) == [] and row['sim_src_wavs'] == e[pairs[0]['source_speaker']]


# ── @src rows ────────────────────────────────────────────────

def test_src_row_transformation():
    pairs = dev_pairs()
    row = conv_row(pairs[0], 'cand_pm_rq_T1', '/lab', sim_src_wavs=['/s/e0.flac'])
    (s,) = lab_rows.src_rows([row])
    assert s == dict(row, system='cand_pm_rq_T1@src', sim_ref_wavs=['/s/e0.flac'],
                     target_speaker=row['source_speaker'], true_target_speaker=row['target_speaker'])
    assert list(s) == list(row) + ['true_target_speaker']


def _inputs(kind, tmp_path):
    """(pair rows, source-E file or '', merged rows) for the equivalence test."""
    if kind == 'dev':
        pairs, se = dev_pairs(), ''
    else:
        pairs = vctk_pairs()
        se = str(tmp_path / 'vctk2k_source_E.json')
        with open(se, 'w') as f:
            json.dump({'p301': {'paths': ['/v/p301/p301_900.flac'], 'durs': [3.0], 'total_s': 3.0},
                       'p302': [], 'p999': ['/v/p999.flac']}, f)
    faiss = [conv_row(p, 'faiss', '/rc') for p in pairs]
    rows = [dict(lab_row(f, s, '/lab/cand'), system='cand_' + s) for f in faiss for s in CAND[:2]]
    rows += [dict(faiss[2], system='crt_tau0.4', error='ValueError: x'), dict(faiss[5], system='crt_argmax')]
    if kind == 'vctk, without --source_e':
        se = ''
    return pairs, se, rows


@pytest.mark.parametrize('kind', ['dev', 'vctk, with --source_e', 'vctk, without --source_e'])
def test_src_rows_equal_crt_anchor_lab_src_rows(tmp_path, kind):
    """Field by field (values and order) the @src rows crt_anchor_lab.py writes from the same rows, and the same
    sim_src_wavs as crt_anchor_lab.main (source_e_of over its spk_rows, or load_source_e of the same file)."""
    pairs, se, rows = _inputs(kind, tmp_path)
    lab_rows.add_sim_src(rows, pairs, lab_rows.source_e_sets(pairs, se))
    src_e = crt_anchor_lab.load_source_e(se) if se else None
    assert [r['sim_src_wavs'] for r in rows] == \
        [crt_anchor_lab.source_e_of(r['source_speaker'], spk_rows_of(pairs), src_e) for r in rows]
    ours, theirs = lab_rows.src_rows(copy.deepcopy(rows)), crt_anchor_lab.src_rows(copy.deepcopy(rows))
    assert [json.dumps(r) for r in ours] == [json.dumps(r) for r in theirs]
    n_with_e = sum('error' not in r and r['sim_src_wavs'] is not None for r in rows)
    assert len(ours) == n_with_e == {'dev': 49, 'vctk, with --source_e': 8, 'vctk, without --source_e': 0}[kind]


# ── main, as job_d0.slurm step 8 runs it ─────────────────────

def test_lab_out_dirs_to_manifest_and_src(dev, monkeypatch, capsys):
    out = dev['lab'] / 'manifest.jsonl'
    run_main(monkeypatch, '--pairs', dev['path'], '--lab', dev['lab'] / 'cand', 'cand', '--lab', dev['lab'] / 'front',
             'front', '--out', out)
    rows, src = read_jsonl(out), read_jsonl(dev['lab'] / 'manifest_src.jsonl')
    e = lab_rows.source_e_sets(dev['pairs'])
    assert len(rows) == len(src) == 192
    assert [{k: v for k, v in r.items() if k != 'sim_src_wavs'} for r in rows] == \
        lab_rows.merge([(str(dev['lab'] / 'cand'), 'cand'), (str(dev['lab'] / 'front'), 'front')], [])[0]
    assert all(r['sim_src_wavs'] == e[r['source_speaker']] for r in rows)
    assert src == crt_anchor_lab.src_rows(rows)
    log = capsys.readouterr().out
    assert '192 @src rows -> ' in log
    assert 'left out: 0 rows (0 pairs) without a source E set (source speakers [])' in log
    monkeypatch.setattr(metrics, 'text_of', lambda p: p)       # no transcripts here
    for r in rows + src:                                         # metrics.py's own check on load
        metrics.check_row(metrics.as_metric_row(r), r.get('R'))


def test_conversion_rows_controls_and_copies(dev, tmp_path, monkeypatch, capsys):
    """The second D0 call: run_conversions rows (with an error row), the Mimi copies and the natural controls."""
    rc_rows = [conv_row(p, s, '/rc', tau=t) for p in dev['pairs'] for s, t in (('faiss', None), ('crt_tau0.4', 0.4))]
    rc_rows[3] = dict(rc_rows[3], error='ValueError: boom', shortfall=False)
    mc = [conv_row(p, 'mimi_copy_src', '/mc') for p in dev['pairs']] + \
         [conv_row(p, 'mimi_copy_P', '/mc') for p in dev['pairs'] if p['P']]
    natural = []
    for r in rc_rows:
        if r['system'] == 'faiss' and not r.get('error'):
            natural.append(dict(r, system='source_copy', wav=r['source_wav']))
            if r.get('parallel_wav'):
                natural.append(dict(r, system='target_copy', wav=r['parallel_wav'], copy_control=True))
    files = [write_jsonl(tmp_path / n, rs) for n, rs in (('rc.jsonl', rc_rows), ('mc.jsonl', mc),
                                                         ('natural.jsonl', natural))]
    out = tmp_path / 'rows' / 'conv.jsonl'
    run_main(monkeypatch, '--pairs', dev['path'], *sum((['--manifest', f] for f in files), []), '--out', out)
    rows, src = read_jsonl(out), read_jsonl(tmp_path / 'rows' / 'conv_src.jsonl')
    assert [{k: v for k, v in r.items() if k != 'sim_src_wavs'} for r in rows] == rc_rows + mc + natural
    assert len(src) == len(rows) - 1 and not any('error' in s for s in src)
    assert {s['system'] for s in src} == {x + '@src' for x in ('faiss', 'crt_tau0.4', 'mimi_copy_src', 'mimi_copy_P',
                                                               'source_copy', 'target_copy')}
    assert 'left out: 0 rows (0 pairs) without a source E set (source speakers []), 1 error rows' in \
        capsys.readouterr().out


def test_pairs_without_a_source_e_are_left_out_and_counted(tmp_path, monkeypatch, capsys):
    """VCTK-2k: the source-E file has p301 (make_pairs format), p302 an empty list, p303 not at all."""
    pairs = vctk_pairs()
    path = write_jsonl(tmp_path / 'vctk2k.jsonl', pairs)
    se = tmp_path / 'vctk2k_source_E.json'
    se.write_text(json.dumps({'p301': {'paths': ['/v/p301/e0.flac', '/v/p301/e1.flac'], 'durs': [3.0, 3.0],
                                       'total_s': 6.0}, 'p302': []}))
    faiss = [conv_row(p, 'faiss', '/rc') for p in pairs]
    make_lab(str(tmp_path / 'cand'), faiss, CAND[:2])
    rows_in = read_jsonl(tmp_path / 'cand' / 'manifest.jsonl')
    rows_in[0]['error'] = 'RuntimeError: x'               # a p301 row: an error row is left out too
    write_jsonl(tmp_path / 'cand' / 'manifest.jsonl', rows_in)
    out = tmp_path / 'm.jsonl'
    run_main(monkeypatch, '--pairs', path, '--source_e', se, '--lab', tmp_path / 'cand', 'cand', '--out', out)
    rows, src = read_jsonl(out), read_jsonl(tmp_path / 'm_src.jsonl')
    assert [r['sim_src_wavs'] for r in rows] == \
        [['/v/p301/e0.flac', '/v/p301/e1.flac'] if r['source_speaker'] == 'p301' else None for r in rows]
    assert len(src) == 7 and {s['target_speaker'] for s in src} == {'p301'}
    assert all(s['sim_ref_wavs'] == ['/v/p301/e0.flac', '/v/p301/e1.flac'] for s in src)
    assert 'left out: 16 rows (8 pairs) without a source E set (source speakers [\'p302\', \'p303\']), 1 error rows' \
        in capsys.readouterr().out


def test_rows_of_another_pair_list_are_refused(dev, tmp_path, monkeypatch):
    other = write_jsonl(tmp_path / 'ltc2k.jsonl', [pair_row('ltc2k', '1089', '908', 7, parallel=False)])
    msg = refused(monkeypatch, '--pairs', other, '--lab', dev['lab'] / 'cand', 'cand', '--out', tmp_path / 'o.jsonl')
    assert msg == f'ERROR: {dev["pairs"][0]["pair_id"]} cand_pm_rq_T1: not a --pairs row with source speaker p225'
    assert not os.path.exists(tmp_path / 'o.jsonl')


@pytest.mark.parametrize('kind', ['vctk2k 30 s, --source_e', 'ltc2k 180 s, shortfall rows', 'dev 30 s'])
def test_anchor_lab_manifests_through_lab_rows(tmp_path, monkeypatch, kind):
    """job_m.slurm passes the crt_anchor_lab manifests of the test sets through lab_rows (with the same source-E file):
    their sim_src_wavs are accepted as they are, and the @src rows are crt_anchor_lab's own manifest_src rows."""
    stub_anchor_lab(monkeypatch, crt_anchor_lab)
    extra = []
    if kind.startswith('vctk2k'):
        pairs, set_, budget = vctk_pairs(), 'vctk2k', '30'
        se = tmp_path / 'vctk2k_source_E.json'
        se.write_text(json.dumps({'p301': {'paths': ['/v/p301/e0.flac'], 'durs': [3.0], 'total_s': 3.0},
                                  'p302': ['/v/p302/e0.flac', '/v/p302/e1.flac']}))
        extra = ['--source_e', str(se)]
    elif kind.startswith('ltc2k'):
        pairs, set_, budget = ltc_pairs(), 'ltc2k', '180'
    else:
        pairs, set_, budget = dev_pairs(), 'dev', '30'
    path = write_jsonl(tmp_path / f'{set_}.jsonl', pairs)
    run = tmp_path / 'anc' / set_ / budget
    monkeypatch.setattr(sys, 'argv', ['crt_anchor_lab.py', '--pairs', path, '--set', set_, '--budget', budget,
                                      '--out_dir', str(tmp_path / 'anc'), '--device', 'cpu', '--mode', 'R1+R3',
                                      '--systems', 'src0_faiss,R3_s0_pm_rq_T1', *extra])
    crt_anchor_lab.main()
    anc, anc_src = read_jsonl(run / 'manifest.jsonl'), read_jsonl(run / 'manifest_src.jsonl')
    assert any('error' in r for r in anc) == kind.startswith('ltc2k')
    out = tmp_path / 'rows' / f'{set_}_{budget}.jsonl'
    run_main(monkeypatch, '--pairs', path, *extra, '--manifest', run / 'manifest.jsonl', '--out', out)
    assert read_jsonl(out) == anc
    assert read_jsonl(tmp_path / 'rows' / f'{set_}_{budget}_src.jsonl') == anc_src and len(anc_src) > 0
