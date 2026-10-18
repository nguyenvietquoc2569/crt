"""Tests of benchmark/run_conversions.py: the Mimi Q = 32 copy-synthesis controls mimi_copy_src / mimi_copy_P
(ANALYSIS_PLAN_v1.md §2.5, §8.1 item 6) through parse_systems and main (fixtures_rows.FakePipe stands in for Mimi and
CRT), and the pure helpers load_pairs, parse_shard and base_row."""
import json
import os
import sys
import types

import numpy as np
import pytest

import metrics
import run_conversions as rc
from fixtures_rows import (FRAME, SR, FakePipe, conv_row, dev_pairs, fake_audio, read_jsonl, stub_pipeline,
                           vctk_pairs, write_jsonl)

COPIES = 'faiss,crt_tau:0.4,crt_argmax,mimi_copy_src,mimi_copy_P'
DEFAULT = 'faiss,crt_tau:0.4,crt_argmax'
BASE_KEYS = ['pair_id', 'set', 'budget', 'system', 'tau', 'wav', 'ref_text', 'parallel_wav', 'sim_ref_wavs',
             'source_wav', 'R', 'wer_primary', 'source_is_shared_text', 'genders', 'source_speaker', 'target_speaker',
             'dump']
GATE_KEYS = {'override_coverage', 'change_coverage', 'n_changed', 'override_counts', 'change_counts'}


# ── parse_systems ────────────────────────────────────────────

@pytest.mark.parametrize('spec, want', [
    (DEFAULT, [('faiss', None), ('crt_tau0.4', 0.4), ('crt_argmax', None)]),
    ('mimi_copy_src,mimi_copy_P', [('mimi_copy_src', None), ('mimi_copy_P', None)]),
    ('crt_tau:0.3,0.4,mimi_copy_P,crt_argmax,mimi_copy_src',
     [('crt_tau0.3', 0.3), ('crt_tau0.4', 0.4), ('mimi_copy_P', None), ('crt_argmax', None), ('mimi_copy_src', None)]),
    ('faiss,crt_tau:0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,crt_argmax',             # job D0 step 2
     [('faiss', None)] + [(f'crt_tau{t / 10:g}', t / 10) for t in range(1, 10)] + [('crt_argmax', None)]),
    ('crt_tau:.5,0', [('crt_tau0.5', 0.5), ('crt_tau0', 0.0)]),
])
def test_parse_systems(spec, want):
    assert rc.parse_systems(spec) == want


@pytest.mark.parametrize('spec', ['mimi_copy_src,mimi_copy_src', 'mimi_copy_P,faiss,mimi_copy_P', 'faiss,faiss',
                                  'crt_tau:0.4,0.40', 'crt_tau:0.4,mimi_copy_src,crt_tau:0.4'])
def test_parse_systems_refuses_a_duplicate(spec):
    with pytest.raises(SystemExit) as e:
        rc.parse_systems(spec)
    assert e.value.code == f'ERROR: duplicate system in {spec!r}'


@pytest.mark.parametrize('spec, tok', [('mimi_copy_p', 'mimi_copy_p'), ('faiss,mimi_copy', 'mimi_copy'),
                                       ('crt_tau:0.3,mimi_copy_P,0.4', '0.4'), ('faiss,mimi_copy_src,0.4', '0.4'),
                                       ('faiss,', ''), ('crt_tau:1.0', '1.0'), ('crt_tau0.4', 'crt_tau0.4'),
                                       ('copy_src', 'copy_src')])
def test_parse_systems_refuses_an_unknown_token(spec, tok):
    """A copy name closes a τ list, as faiss does."""
    with pytest.raises(SystemExit) as e:
        rc.parse_systems(spec)
    assert e.value.code == f'ERROR: bad --systems token {tok!r}'


# ── parse_shard, load_pairs, base_row ────────────────────────

@pytest.mark.parametrize('spec, want', [('0/1', (0, 1)), ('0/20', (0, 20)), ('19/20', (19, 20)), ('3/400', (3, 400))])
def test_parse_shard(spec, want):
    assert rc.parse_shard(spec) == want


@pytest.mark.parametrize('spec', ['1/1', '20/20', '-1/2', '1/0', '0/0', 'a/b', '0/2/3', '', '0 /2', '1', '1/2 '])
def test_parse_shard_refuses(spec):
    with pytest.raises(SystemExit) as e:
        rc.parse_shard(spec)
    assert e.value.code == f'ERROR: --shard {spec!r} is not i/n with 0 <= i < n'


def selector(path, set_='dev', **kw):
    a = types.SimpleNamespace(pairs=str(path), set=set_, compat_spie=False, pair_ids='', max_pairs=0, shard='')
    a.__dict__.update(kw)
    return a


@pytest.fixture
def pairs_file(tmp_path):
    pairs = dev_pairs()
    path = write_jsonl(tmp_path / 'dev.jsonl', pairs)
    with open(path, 'a') as f:
        f.write('\n  \n')                                                  # blank lines are skipped
    return path, pairs


def test_load_pairs_selection(pairs_file):
    path, pairs = pairs_file
    ids = [p['pair_id'] for p in pairs]

    def sel(**kw):
        return [r['pair_id'] for r in rc.load_pairs(selector(path, **kw))]
    assert rc.load_pairs(selector(path)) == pairs
    assert sel(shard=(0, 20)) == ids[0::20] == [ids[0], ids[20]]           # D0's --shard 0/20: rows k % 20 == 0
    assert sel(shard=(3, 5)) == ids[3::5]
    assert sel(max_pairs=7) == ids[:7]
    assert sel(max_pairs=7, shard=(1, 3)) == ids[1:7:3]                    # first N rows, then the shard
    assert sel(max_pairs=100) == ids
    want = ','.join([ids[9], ids[2], ids[17]])
    assert sel(pair_ids=want) == [ids[2], ids[9], ids[17]]                 # in file order
    assert sel(pair_ids=want, max_pairs=2) == [ids[2], ids[9]]             # pair_ids, then max_pairs, then the shard
    assert sel(pair_ids=want, shard=(1, 2)) == [ids[9]]
    assert sel(pair_ids=want, max_pairs=2, shard=(1, 2)) == [ids[9]]


@pytest.mark.parametrize('case', ['unknown pair_id', 'other set', 'duplicate pair_id'])
def test_load_pairs_refuses(tmp_path, pairs_file, case):
    path, pairs = pairs_file
    if case == 'unknown pair_id':
        a, msg = selector(path, pair_ids=f'{pairs[1]["pair_id"]},dev__x'), "ERROR: pair_ids not in --pairs: ['dev__x']"
    elif case == 'other set':
        a, msg = selector(path, 'vctk2k'), f"ERROR: {path} has rows of set ['dev'], not 'vctk2k'"
        mixed = write_jsonl(tmp_path / 'mixed.jsonl', pairs[:2] + vctk_pairs()[:1])
        with pytest.raises(SystemExit) as e:
            rc.load_pairs(selector(mixed))
        assert e.value.code == f"ERROR: {mixed} has rows of set ['vctk2k'], not 'dev'"
    else:
        a = selector(write_jsonl(tmp_path / 'dup.jsonl', pairs + pairs[3:4]))
        msg = 'ERROR: duplicate pair_id in --pairs'
    with pytest.raises(SystemExit) as e:
        rc.load_pairs(a)
    assert e.value.code == msg


def test_load_pairs_compat_spie(tmp_path):
    ev = [{'source_speaker': 'p225', 'target_speaker': 'p226', 'source_file': '/s/p225_001.wav',
           'target_files': ['/t/p226_001.wav', '/t/p226_002.wav']},
          {'source_speaker': 'p227', 'target_speaker': 'p228', 'source_file': '/s/p227_003.wav',
           'target_files': ['/t/p228_004.wav']}]
    f = tmp_path / 'eval_pairs.json'
    f.write_text(json.dumps(ev))
    rows = rc.load_pairs(selector(f, 'conf100', compat_spie=True, shard=(1, 2)))
    assert rows == [{'pair_id': 'p227_to_p228', 'set': 'conf100', 'source_speaker': 'p227', 'target_speaker': 'p228',
                     'source_path': '/s/p227_003.wav', 'ref_paths': ['/t/p228_004.wav']}]


@pytest.mark.parametrize('budget', ['3', '30', '180'])
def test_base_row_fields(budget):
    p = dev_pairs()[0]
    a = types.SimpleNamespace(out_dir='/out', set='dev', budget=budget, compat_spie=False)
    r = rc.base_row(a, p, 'mimi_copy_P', None)
    assert list(r) == BASE_KEYS
    assert r == {'pair_id': p['pair_id'], 'set': 'dev', 'budget': int(budget), 'system': 'mimi_copy_P', 'tau': None,
                 'wav': f'/out/dev/{budget}/mimi_copy_P/{p["pair_id"]}.wav', 'ref_text': p['source_text'],
                 'parallel_wav': p['P'], 'sim_ref_wavs': p['E']['paths'], 'source_wav': p['source_path'],
                 'R': {budget: p['R'][budget]}, 'wer_primary': p['wer_primary'],
                 'source_is_shared_text': p['source_is_shared_text'], 'genders': p['genders'],
                 'source_speaker': p['source_speaker'], 'target_speaker': p['target_speaker'],
                 'dump': f'/out/dumps/dev/{budget}/{p["pair_id"]}.npz'}
    assert (r['wav'], r['dump']) == rc.paths_for(a, p['pair_id'], 'mimi_copy_P')
    assert r == {k: v for k, v in conv_row(p, 'mimi_copy_P', '/out', budget).items() if k in r}   # the fixture's rows


def test_base_row_compat_spie():
    row = {'pair_id': 'p225_to_p226', 'set': 'conf100', 'source_speaker': 'p225', 'target_speaker': 'p226',
           'source_path': '/s.wav', 'ref_paths': ['/t1.wav']}
    a = types.SimpleNamespace(out_dir='/out', set='conf100', budget='full', compat_spie=True)
    r = rc.base_row(a, row, 'crt_tau0.4', 0.4)
    assert r == {'pair_id': 'p225_to_p226', 'set': 'conf100', 'budget': 'full', 'system': 'crt_tau0.4', 'tau': 0.4,
                 'wav': '/out/conf100/full/crt_tau0.4/p225_to_p226.wav', 'source_wav': '/s.wav',
                 'ref_paths': ['/t1.wav'], 'source_speaker': 'p225', 'target_speaker': 'p226',
                 'dump': '/out/dumps/conf100/full/p225_to_p226.npz'}


# ── the copies through main ──────────────────────────────────

@pytest.fixture
def six(tmp_path):
    """Six Dev pairs, the first three with P; a dummy weights / script file."""
    pairs = dev_pairs()[:6]
    assert [bool(p['P']) for p in pairs] == [True] * 3 + [False] * 3
    dummy = tmp_path / 'weights.pt'
    dummy.write_text('stub')
    return types.SimpleNamespace(pairs=pairs, path=write_jsonl(tmp_path / 'dev.jsonl', pairs), dummy=dummy,
                                 tmp=tmp_path)


def run_rc(monkeypatch, s, out, systems, *extra):
    """run_conversions.main(): 0 if it returns, else the SystemExit code."""
    monkeypatch.setattr(sys, 'argv', ['run_conversions.py', '--pairs', s.path, '--set', 'dev', '--systems', systems,
                                      '--out_dir', str(out), '--device', 'cpu', '--ckpt', str(s.dummy), '--src_ckpt',
                                      str(s.dummy), '--crt_script', str(s.dummy), *extra])
    try:
        rc.main()
    except SystemExit as e:
        return e.code
    return 0


def by_key(out):
    return {(r['pair_id'], r['system']): r for r in read_jsonl(os.path.join(out, 'dev', '30', 'manifest.jsonl'))}


def copy_wav(tmp, path):
    """The WAV a Mimi copy of `path` must be: FakePipe decode(encode(recording)), written by write_wav."""
    out = os.path.join(tmp, 'expected', os.path.basename(path) + '.wav')
    rc.write_wav(out, FakePipe.decode(FakePipe.encode(fake_audio(path))), SR)
    return open(out, 'rb').read()


def test_copy_rows(six, monkeypatch, capsys):
    log = stub_pipeline(monkeypatch, rc)
    out = six.tmp / 'mc'
    assert run_rc(monkeypatch, six, out, COPIES) == 0
    assert '0 errors' in capsys.readouterr().out
    rows = by_key(out)
    assert len(rows) == 6 * 4 + 3 and not any('error' in r for r in rows.values())
    monkeypatch.setattr(metrics, 'text_of', lambda p: p)
    drop = {'system', 'wav', 'conv_time_s'}
    for p in six.pairs:
        pid = p['pair_id']
        f, c = rows[(pid, 'faiss')], rows[(pid, 'mimi_copy_src')]
        assert {k: v for k, v in c.items() if k not in drop} == {k: v for k, v in f.items() if k not in drop}
        assert c['wav'] == f'{out}/dev/30/mimi_copy_src/{pid}.wav' and c['tau'] is None and not GATE_KEYS & set(c)
        assert c['n_frames'] == len(fake_audio(p['source_path'])) // FRAME == f['n_frames']
        assert open(c['wav'], 'rb').read() == copy_wav(six.tmp, p['source_path'])
        metrics.check_row(metrics.as_metric_row(c), c['R'])
        cp = rows.get((pid, 'mimi_copy_P'))
        if p['P']:
            assert {k: v for k, v in cp.items() if k not in drop | {'n_frames'}} == \
                {k: v for k, v in f.items() if k not in drop | {'n_frames'}}
            assert (cp['parallel_wav'], cp['ref_text'], cp['sim_ref_wavs']) == (p['P'], p['source_text'],
                                                                               p['E']['paths'])
            assert cp['n_frames'] == len(fake_audio(p['P'])) // FRAME
            assert open(cp['wav'], 'rb').read() == copy_wav(six.tmp, p['P'])
            assert 'encode_P' in cp['conv_time_s'] and cp['tau'] is None and not GATE_KEYS & set(cp)
            metrics.check_row(metrics.as_metric_row(cp), cp['R'])
        else:                                             # no P: no row, not an error row, no file
            assert cp is None and not os.path.exists(f'{out}/dev/30/mimi_copy_P/{pid}.wav')
        for s in ('crt_tau0.4', 'crt_argmax'):
            assert GATE_KEYS <= set(rows[(pid, s)]) and rows[(pid, s)]['n_frames'] == f['n_frames']
        assert os.path.exists(f['dump'])
    assert sorted(x for x in log.loads if x in {p['P'] for p in six.pairs}) == sorted(p['P'] for p in six.pairs[:3])
    assert any(rows[(p['pair_id'], 'mimi_copy_P')]['n_frames'] != rows[(p['pair_id'], 'faiss')]['n_frames']
               for p in six.pairs[:3])                    # the P copies' own lengths, not the source's


def test_pairs_without_p_count_as_complete_on_resume(six, monkeypatch, capsys):
    log = stub_pipeline(monkeypatch, rc)
    out = six.tmp / 'mc'
    assert run_rc(monkeypatch, six, out, COPIES) == 0
    before = read_jsonl(out / 'dev' / '30' / 'manifest.jsonl')
    capsys.readouterr()
    assert run_rc(monkeypatch, six, out, COPIES) == 0
    assert '0 pairs to convert (6 already complete)' in capsys.readouterr().out
    assert read_jsonl(out / 'dev' / '30' / 'manifest.jsonl') == before
    lost = by_key(out)[(six.pairs[1]['pair_id'], 'mimi_copy_P')]
    os.remove(lost['wav'])
    del log.loads[:]
    assert run_rc(monkeypatch, six, out, COPIES) == 0
    assert '1 pairs to convert (5 already complete)' in capsys.readouterr().out
    assert log.loads == [six.pairs[1]['source_path'], six.pairs[1]['P']]
    again = by_key(out)
    assert {k: r for k, r in again.items() if k != (lost['pair_id'], 'mimi_copy_P')} == \
        {(r['pair_id'], r['system']): r for r in before if (r['pair_id'], r['system']) != (lost['pair_id'],
                                                                                           'mimi_copy_P')}
    assert os.path.exists(lost['wav'])


def test_p_is_read_only_for_its_copy(six, monkeypatch):
    """Without mimi_copy_P no P is read, so an unreadable P cannot fail the other systems."""
    log = stub_pipeline(monkeypatch, rc, fail={p['P'] for p in six.pairs if p['P']})
    out = six.tmp / 'rc'
    assert run_rc(monkeypatch, six, out, DEFAULT + ',mimi_copy_src') == 0
    assert not any('error' in r for r in by_key(out).values())
    assert not {p['P'] for p in six.pairs} & set(log.loads)


def test_a_finished_p_copy_is_not_read_again(six, monkeypatch):
    """Resume where only another system is missing: the finished mimi_copy_P does not make the pair read P again
    (P unreadable since the first run)."""
    stub_pipeline(monkeypatch, rc)
    out = six.tmp / 'mc'
    assert run_rc(monkeypatch, six, out, 'faiss,mimi_copy_P') == 0
    log = stub_pipeline(monkeypatch, rc, fail={p['P'] for p in six.pairs if p['P']})
    assert run_rc(monkeypatch, six, out, 'faiss,mimi_copy_P,crt_argmax') == 0
    rows = by_key(out)
    assert len(rows) == 6 * 2 + 3 and not any('error' in r for r in rows.values())
    assert not {p['P'] for p in six.pairs} & set(log.loads)


def test_error_rows_of_a_failed_pair(six, monkeypatch, capsys):
    """A failed pair gets an error row for every system it needed: mimi_copy_P only where the pair has P. The P of
    pair 0 is unreadable, and so is the source utterance of pairs 3-5 (no P), as in pairs/dev.jsonl one source
    utterance per target."""
    with_p = six.pairs[0]
    without_p = [p for p in six.pairs if p['source_path'] == six.pairs[4]['source_path']]
    assert [p['P'] for p in without_p] == [None] * 3
    stub_pipeline(monkeypatch, rc, fail={with_p['P'], six.pairs[4]['source_path']})
    out = six.tmp / 'mc'
    assert run_rc(monkeypatch, six, out, COPIES) == 1
    assert '4 errors' in capsys.readouterr().out
    rows = by_key(out)
    assert sorted(s for (pid, s), r in rows.items() if pid == with_p['pair_id'] and 'error' in r) == \
        ['crt_argmax', 'crt_tau0.4', 'faiss', 'mimi_copy_P', 'mimi_copy_src']
    for p in without_p:
        assert sorted(s for (pid, s), r in rows.items() if pid == p['pair_id']) == \
            ['crt_argmax', 'crt_tau0.4', 'faiss', 'mimi_copy_src']
        assert all('error' in rows[(p['pair_id'], s)] for s in ('crt_argmax', 'crt_tau0.4', 'faiss', 'mimi_copy_src'))
    assert sum('error' in r for r in rows.values()) == 5 + 3 * 4 and len(rows) == 27
    assert all(not os.path.exists(r['wav']) and r['shortfall'] is False for r in rows.values() if 'error' in r)


def test_adding_the_copies_leaves_the_other_systems_unchanged(six, monkeypatch):
    stub_pipeline(monkeypatch, rc)
    a, b = six.tmp / 'a', six.tmp / 'b'
    assert run_rc(monkeypatch, six, a, DEFAULT) == 0
    assert run_rc(monkeypatch, six, b, COPIES) == 0
    ra, rb = by_key(a), by_key(b)
    assert set(ra) == {k for k in rb if k[1] in ('faiss', 'crt_tau0.4', 'crt_argmax')}

    def same(r, out):
        return {k: (os.path.relpath(v, out) if k in ('wav', 'dump') else v) for k, v in r.items()
                if k not in ('run_id', 'conv_time_s')}
    for k in ra:
        assert same(ra[k], a) == same(rb[k], b)
        assert open(ra[k]['wav'], 'rb').read() == open(rb[k]['wav'], 'rb').read()
    for p in six.pairs:
        name = f'dumps/dev/30/{p["pair_id"]}.npz'
        with np.load(a / name) as za, np.load(b / name) as zb:
            assert za.files == zb.files
            for f in za.files:
                np.testing.assert_array_equal(za[f], zb[f], err_msg=f)


def test_copies_only_run(six, monkeypatch):
    stub_pipeline(monkeypatch, rc)
    out = six.tmp / 'mc'
    assert run_rc(monkeypatch, six, out, 'mimi_copy_src,mimi_copy_P') == 0
    rows = by_key(out)
    assert sorted(k[1] for k in rows) == ['mimi_copy_P'] * 3 + ['mimi_copy_src'] * 6
    assert all(os.path.exists(r['dump']) and not GATE_KEYS & set(r) for r in rows.values())
