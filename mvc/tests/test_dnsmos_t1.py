"""Tests of mvc/analysis/dnsmos_t1.py with a stand-in for dnsmos.py that, like the real one, writes rows, a summary file
of means and prints the means: every batch is scored from the copied WAVs; nothing of a metric value is printed; the
summary and the WAVs are gone afterwards; the moved nvquoc paths are fetched from ceai; a finished batch is skipped; a
WAV missing on ceai or a changed dnsmos.py stops the run."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import dnsmos_t1 as dt  # noqa: E402

MARK = '0.42424'
STUB = f'''import json, os, sys
a = sys.argv[2:]
rows, out = a[a.index('--rows') + 1], a[a.index('--out') + 1]
assert a[a.index('--norm_lufs') + 1] == '-26'
res = []
for l in open(rows):
    r = json.loads(l)
    if 'wav' in r and os.path.exists(r['wav']):
        res.append(dict(pair_id=r['pair_id'], system=r['system'], SIG={MARK}1, BAK={MARK}2, OVRL={MARK}3, P808={MARK}4))
    else:
        res.append(dict(pair_id=r['pair_id'], system=r['system'], error='FileNotFoundError'))
open(out, 'w').write(''.join(json.dumps(r) + chr(10) for r in res))
open(out[:-6] + '.summary.json', 'w').write(json.dumps(dict(OVRL={MARK}9)))
print(json.dumps(dict(OVRL={MARK}9)))
'''
PAIRS = {'vctk2k': ['v_a', 'v_b'], 'ltc2k': ['l_a', 'l_b']}


def setup(tmp_path, drop=None):
    mr, v1, remote = tmp_path / 'mr', tmp_path / 'v1', tmp_path / 'remote'

    def rows(path, systems, prefix, s, part):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = []
        for n in systems:
            for pid in PAIRS[s]:
                r = dict(pair_id=pid, system=n, utmos=float(MARK + '5'), sim=float(MARK + '6'))
                if (n, pid) != ('mvc_v1', 'v_b'):                      # a failed conversion: no wav
                    wav = f'{prefix}/mimi_vc/{part}/{s}/{n}/{pid}.wav'
                    r['wav'] = wav
                    real = remote / dt.remote_path(wav).lstrip('/')
                    if (n, pid) != drop:
                        real.parent.mkdir(parents=True, exist_ok=True)
                        real.write_bytes(b'RIFF')
                out.append(r)
        out.append(dict(pair_id=PAIRS[s][0], system='other_system', wav='/x.wav'))
        path.write_text(''.join(json.dumps(r) + '\n' for r in out))

    for s in dt.SETS:
        p = dt.PART[s]
        rows(mr / 'm' / p / 'metrics' / f'{s}_30.jsonl', dt.MVC_SYSTEMS, '/datastore/ceai/crt', s, 'test')
        rows(v1 / 'm' / p / 'metrics' / f'{s}_30.jsonl', dt.JOURNAL_SYSTEMS, '/datastore/nvquoc/crt', s, p)
        rows(v1 / 'm' / 't4' / 'metrics' / f'{s}_30.jsonl', dt.BASELINES, '/datastore/ceai/crt', s, 't4')
    models = tmp_path / 'models'
    models.mkdir()
    for k in dt.MODEL_SHA256:
        (models / k).write_bytes(b'onnx')
    stub = tmp_path / 'stub_dnsmos.py'
    stub.write_text(STUB)
    return mr, v1, remote, models, stub


@pytest.fixture
def env(tmp_path, monkeypatch):
    def make(drop=None):
        mr, v1, remote, models, stub = setup(tmp_path, drop)
        monkeypatch.setattr(dt, 'MODEL_SHA256', {k: dt.sha256(models / k) for k in dt.MODEL_SHA256})
        fetched = []

        def fetch(paths, wav_root, remote_host):
            fetched.extend(paths)
            for p in paths:
                src = remote / p.lstrip('/')
                if src.exists():
                    Path(wav_root + p).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, wav_root + p)
            return 0

        monkeypatch.setattr(dt, 'fetch', fetch)
        argv = ['--mr', str(mr), '--v1', str(v1), '--models', str(models), '--python', f'{sys.executable} {stub}',
                '--workers', '2']
        return mr, argv, fetched
    return make


def test_every_batch_is_scored_and_no_value_or_wav_is_left(env, capsys):
    mr, argv, fetched = env()
    assert dt.main(argv) == 0
    out = capsys.readouterr().out
    assert MARK not in out and '20 batches, 40 rows' in out
    files = sorted(os.listdir(mr / 'dnsmos'))
    assert files == ['input'] + sorted(f'{s}_30_{n}.jsonl' for s in dt.SETS
                                       for n in dt.MVC_SYSTEMS + dt.JOURNAL_SYSTEMS + dt.BASELINES)
    assert not list((mr / 'dnsmos').glob('*.summary.json'))
    assert not [p for p in (mr / 'wav').rglob('*') if p.is_file()]
    assert fetched and all(p.startswith('/datastore/ceai/crt/') for p in fetched)
    assert any('/vctk2k/cand_pm18_rq/' in p for p in fetched)            # a journal row named nvquoc
    rows = [json.loads(x) for x in open(mr / 'dnsmos' / 'vctk2k_30_mvc_v1.jsonl')]
    assert [r['pair_id'] for r in rows if 'error' in r] == ['v_b'] and len(rows) == 2
    inp = [json.loads(x) for x in open(mr / 'dnsmos' / 'input' / 'vctk2k_30_cand_pm18_rq.jsonl')]
    assert all(set(r) == {'pair_id', 'system', 'wav'} and r['wav'].startswith(str(mr / 'wav') + '/datastore/ceai/crt/')
               for r in inp)
    assert 'vctk2k mvc_v1: 2 rows, 1 WAVs, 1 error rows' in out


def test_a_finished_batch_is_skipped(env, capsys):
    mr, argv, fetched = env()
    dt.main(argv)
    fetched.clear()
    assert dt.main(argv) == 0 and not fetched
    assert capsys.readouterr().out.count('already scored, skipped') == 20


def test_a_wav_missing_on_ceai_stops_before_scoring(env):
    mr, argv, _ = env(drop=('seedvc', 'l_a'))
    with pytest.raises(SystemExit, match='ltc2k seedvc: 1 WAVs not copied'):
        dt.main(argv)
    assert not (mr / 'dnsmos' / 'ltc2k_30_seedvc.jsonl').exists()


def test_a_changed_dnsmos_py_stops(env, monkeypatch):
    _, argv, fetched = env()
    monkeypatch.setattr(dt, 'DNSMOS_SHA256', '0' * 64)
    with pytest.raises(SystemExit, match='frozen sha256'):
        dt.main(argv)
    assert not fetched


def test_the_systems_are_analyze_mvc_s():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
    import analyze_mvc as am
    assert set(am.DNSMOS_SYSTEMS) == set(dt.MVC_SYSTEMS + dt.JOURNAL_SYSTEMS + dt.BASELINES)
    assert dt.sha256(dt.DNSMOS_PY) == dt.DNSMOS_SHA256
