"""Tests of benchmark/run_baselines.py (the baseline conversion harness, ANALYSIS_PLAN_v1 §2.4, §1.3) with stand-in
adapters: a module put into sys.modules as baselines.<name> (in-process runs of main()), or loaded by a
sitecustomize.py on PYTHONPATH (the multi-system run, whose children are fresh interpreters). No model and no torch:
the stand-in mixes the source with numpy noise, so the harness's per-pair seeding shows in the output bytes.

The pairs are synthetic: 24 kHz PCM_16 WAVs under a temporary directory, rows with the fields make_pairs.py writes,
R entries cut as make_pairs.build_refs cuts them (each budget lists whole utterances until >= budget, keep_s cuts the
last one), transcripts next to the R files for metrics.check_row. Target t1 has two pairs (one cached reference),
t3 has too little audio for 3 s (a shortfall entry).
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import types

import librosa
import numpy as np
import pytest
import soundfile as sf
import soxr  # noqa: F401  (imported up front: run_baselines records the versions of the modules already imported)

import make_pairs
import metrics
import run_baselines as rb
import run_conversions as rc

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_BASELINES = os.path.join(BENCH, 'run_baselines.py')
FILE_SR = 24000
SOURCES = {'s1_001': 1.3, 's1_002': 0.9, 's2_001': 1.1, 's2_002': 1.6, 's3_001': 1.0}
TARGET_R = {'t1': [1.25, 1.1, 0.9, 9.0, 9.5, 10.0], 't2': [1.0, 1.5, 0.75, 10.0, 10.0, 10.0], 't3': [1.0, 0.8, 0.9]}
IDS = ['dev__s1_001__t1', 'dev__s2_001__t1', 'dev__s1_002__t2', 'dev__s3_001__t3', 'dev__s2_002__t2']
OK = [IDS[0], IDS[1], IDS[2], IDS[4]]          # pairs whose reference fills the 3 s and 30 s budgets
SHORT = IDS[3]                                 # t3: 2.7 s of reference audio
EXTRA = {'ref_s', 'src_s', 'out_s', 'conv_time_s', 'run_id', 'conv_cfg_sha'}
TIMES = {'load_src', 'ref_load', 'ref_cached', 'convert', 'write_wav'}


def seed_of(pair_id):
    return int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16)


def write_tone(path, seconds, seed):
    n = int(round(seconds * FILE_SR))
    t = np.arange(n) / FILE_SR
    x = 0.3 * np.sin(2 * np.pi * (110 + 23 * seed) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x += 0.01 * np.random.default_rng(seed).standard_normal(n)
    sf.write(path, x.astype(np.float32), FILE_SR, subtype='PCM_16')
    return path


def r_entry(paths, budget):
    """An R[budget] entry as make_pairs.build_refs writes it."""
    sel, keep, acc = [], [], 0.0
    for p in paths:
        if acc >= budget:
            break
        d = round(sf.info(p).duration, 3)
        sel.append(p)
        keep.append(round(min(d, budget - acc), 3))
        acc += d
    return {'budget_s': budget, 'paths': sel, 'durs': [round(sf.info(p).duration, 3) for p in sel], 'keep_s': keep,
            'total_s': round(sum(keep), 3), 'uncut_total_s': round(acc, 3), 'shortfall': acc < budget}


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp('pairs')
    seed = iter(range(1000))
    src = {u: write_tone(str(root / f'{u}.wav'), d, next(seed)) for u, d in SOURCES.items()}
    refs, e_sets = {}, {}
    for t, durs in TARGET_R.items():
        paths = []
        for i, d in enumerate(durs):
            p = write_tone(str(root / f'{t}_r{i:02d}.wav'), d, next(seed))
            with open(p[:-4] + '.normalized.txt', 'w') as f:
                f.write(f'target {t} reference sentence number {i}')
            paths.append(p)
        refs[t] = {b: r_entry(paths, int(b)) for b in ('3', '30')}
        e = [write_tone(str(root / f'{t}_e{i}.wav'), 0.8, next(seed)) for i in range(2)]
        e_sets[t] = {'paths': e, 'durs': [0.8, 0.8], 'total_s': 1.6}
    p_path = write_tone(str(root / 't2_p_s2_002.wav'), 1.2, next(seed))
    rows = []
    for pid in IDS:
        _, u, t = pid.split('__')
        shared = u == 's2_002'
        rows.append({'pair_id': pid, 'set': 'dev', 'source_speaker': u.split('_')[0], 'source_gender': 'F',
                     'source_utt': u, 'source_path': src[u], 'source_text': f'Source sentence of {u}.',
                     'source_dur': SOURCES[u], 'source_is_shared_text': shared,
                     'shared_sentence_id': 's01' if shared else None, 'wer_primary': not shared,
                     'target_speaker': t, 'target_gender': 'M', 'genders': 'F2M',
                     'P': p_path if shared else None, 'P_text': 'Source sentence of s2_002.' if shared else None,
                     'P_dur': 1.2 if shared else None, 'R': refs[t], 'E': e_sets[t]})
    assert [r['R']['3']['shortfall'] for r in rows] == [False, False, False, True, False]
    assert all(r['R'][b]['total_s'] == int(b) for r in rows if r['pair_id'] != SHORT for b in ('3', '30'))
    pairs = root / 'dev.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    rev = root / 'dev_reversed.jsonl'
    rev.write_text(''.join(json.dumps(r) + '\n' for r in rows[::-1]))
    return types.SimpleNamespace(root=root, pairs=str(pairs), reversed=str(rev), rows={r['pair_id']: r for r in rows})


# ── The stand-in adapter ─────────────────────────────────────

def bad_output(kind, wav, sr):
    if kind == 'nan':
        wav[3] = np.nan
    elif kind == 'inf':
        wav[0] = np.inf
    elif kind == '2d':
        wav = wav[None]
    elif kind == 'empty':
        wav = wav[:0]
    elif kind == 'int16':
        wav = (wav * 32767).astype(np.int16)
    elif kind == 'sr':
        sr += 1
    return wav, sr


class Adapter(types.ModuleType):
    """baselines.<name> with the contract of baselines/__init__.py; logs every call."""

    def __init__(self, name, ckpt, sr=16000, sr_out=None, bad=None, fail_calls=(), lacks=()):
        super().__init__(f'baselines.{name}')
        self.name, self.ckpt, self.SR, self.sr_out = name, ckpt, sr, sr_out or sr
        self.bad, self.fail_calls, self.lacks = bad, set(fail_calls), lacks
        self.log, self.cfg = [], None

    def calls(self, event):
        return [x for e, x in self.log if e == event]

    def load(self, device, **cfg):
        self.log.append(('load', {'device': device, 'cfg': cfg}))
        self.cfg = dict({'topk': 4}, **cfg)

    def convert(self, src, ref, sr):
        torch = sys.modules.get('torch')
        self.log.append(('convert', {'src': src.copy(), 'ref': ref, 'sr': sr, 'np': np.random.random(),
                                     'py': random.random(), 'torch_seed': getattr(torch, 'seed_value', None)}))
        if len(self.calls('convert')) in self.fail_calls:
            raise RuntimeError('stand-in failure')
        n = int(round(len(src) * self.sr_out / sr))
        wav = (0.25 * np.resize(src, n) + 0.05 * np.random.standard_normal(n)).astype(np.float32)
        return bad_output(self.bad, wav, self.sr_out) if self.bad else (wav, self.sr_out)

    def describe(self):
        d = {'repo': f'stub://{self.name}', 'commit': 'c0ffee0', 'checkpoints': [str(self.ckpt)],
             'sr_out': self.sr_out, 'settings': dict(self.cfg), 'notes': 'stand-in'}
        return {k: v for k, v in d.items() if k not in self.lacks}


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    """install(**kw) puts a fresh stand-in into sys.modules (sys.modules is restored after the test)."""
    def install(name='knnvc', **kw):
        ckpt = tmp_path / f'{name}_weights.bin'
        if not ckpt.exists():
            ckpt.write_bytes(f'weights of {name}'.encode() * 100)
        ad = Adapter(name, ckpt, **kw)
        monkeypatch.setitem(sys.modules, f'baselines.{name}', ad)
        return ad
    return install


class FakeTorch(types.ModuleType):
    """torch as the harness uses it: cuda.is_available / get_device_name, manual_seed."""

    def __init__(self, cuda=False):
        super().__init__('torch')
        self.__version__ = 'fake'
        self.seeds, self.seed_value = [], None
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda: 'Fake GPU 2g.35gb')

    def manual_seed(self, s):
        self.seeds.append(s)
        self.seed_value = s


def harness(monkeypatch, *args):
    """run_baselines.main() in this process; returns (exit status, SystemExit message)."""
    monkeypatch.setattr(sys, 'argv', [RUN_BASELINES, *map(str, args)])
    try:
        rb.main()
    except SystemExit as e:
        return (e.code, '') if isinstance(e.code, int) else (0 if e.code is None else 1, str(e.code))
    return 0, ''


def base_args(data, out, ids=OK, budget='3', *extra):
    return ['--pairs', data.pairs, '--set', 'dev', '--budget', budget, '--systems', 'knnvc', '--out_dir', out,
            '--device', 'cpu', '--pair_ids', ','.join(ids), *extra]


def run_dir(out, budget='3', system='knnvc'):
    return os.path.join(out, 'dev', budget, system)


def manifest(out, budget='3', system='knnvc', name='manifest.jsonl'):
    p = os.path.join(run_dir(out, budget, system), name)
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def rows_by_id(out, **kw):
    return {r['pair_id']: r for r in manifest(out, **kw)}


def runs(out, budget='3', system='knnvc'):
    return [json.loads(line) for line in open(os.path.join(run_dir(out, budget, system), 'runs.jsonl'))]


def wav_bytes(out, budget='3', system='knnvc'):
    return {pid: open(r['wav'], 'rb').read() for pid, r in rows_by_id(out, budget=budget, system=system).items()
            if 'error' not in r}


# ── The reference ────────────────────────────────────────────

@pytest.mark.parametrize('sr, budget', [(16000, '3'), (22050, '3'), (16000, '30')])
def test_reference_is_make_pairs_load_reference_at_the_adapter_rate(data, adapter, monkeypatch, tmp_path, sr, budget):
    ad = adapter(sr=sr)
    out = str(tmp_path / 'out')
    assert harness(monkeypatch, *base_args(data, out, OK, budget)) == (0, '')
    conv = ad.calls('convert')
    assert len(conv) == 4
    for pid, c in zip(OK, conv):
        row = data.rows[pid]
        want = make_pairs.load_reference(row['R'][budget], sr)
        assert c['sr'] == sr and c['ref'].dtype == np.float32 and np.array_equal(c['ref'], want)
        assert len(c['ref']) <= int(budget) * sr and not c['ref'].flags.writeable
        src = librosa.load(row['source_path'], sr=sr, mono=True)[0].astype(np.float32)
        assert c['src'].dtype == np.float32 and np.array_equal(c['src'], src)
    # one array per target reference: the two t1 pairs share it, t2 has its own
    assert conv[0]['ref'] is conv[1]['ref'] and conv[2]['ref'] is conv[3]['ref'] and conv[0]['ref'] is not conv[2]['ref']
    rows = rows_by_id(out, budget=budget)
    assert [rows[p]['conv_time_s']['ref_cached'] for p in OK] == [False, True, False, True]
    assert rows[OK[1]]['conv_time_s']['ref_load'] == 0.0 and rows[OK[3]]['conv_time_s']['ref_load'] == 0.0
    assert {r['ref_s'] for r in rows.values()} == {round(len(conv[0]['ref']) / sr, 3)}
    assert abs(rows[OK[0]]['ref_s'] - int(budget)) < 1e-3


def test_reference_cache_key_is_that_of_run_conversions(data):
    """run_conversions.get_reference's key: (target_speaker, budget, R paths, keep_s)."""
    args = argparse.Namespace(budget='3')
    a, b, c = data.rows[OK[0]], data.rows[OK[1]], data.rows[OK[2]]
    cache = {}
    wa, _, hit_a = rb.get_reference(a, args, 16000, cache)
    wb, tb, hit_b = rb.get_reference(b, args, 16000, cache)
    wc, _, hit_c = rb.get_reference(c, args, 16000, cache)
    assert (hit_a, hit_b, hit_c) == (False, True, False) and wa is wb and wa is not wc and tb == 0.0
    e = a['R']['3']
    assert list(cache) == [('t1', '3', tuple(e['paths']), tuple(e['keep_s'])),
                           ('t2', '3', tuple(c['R']['3']['paths']), tuple(c['R']['3']['keep_s']))]
    other = dict(a, R={'3': dict(e, keep_s=e['keep_s'][:-1] + [0.5])})   # same target, another cut: not shared
    assert rb.get_reference(other, args, 16000, cache)[2] is False


# ── Seeding ──────────────────────────────────────────────────

def test_seed_all_seeds_random_numpy_and_torch(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    rb.seed_all('dev__p225_014__p239')
    s = seed_of('dev__p225_014__p239')
    assert torch.seeds == [s] and s < 2 ** 32
    assert np.random.random() == np.random.RandomState(s).random_sample()
    assert random.random() == random.Random(s).random()


def test_seed_all_without_torch_imports_no_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    rb.seed_all('x')
    assert 'torch' not in sys.modules


def test_each_pair_is_seeded_from_its_pair_id_before_convert(data, adapter, monkeypatch, tmp_path):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    ad = adapter()
    assert harness(monkeypatch, *base_args(data, str(tmp_path / 'out')))[0] == 0
    for pid, c in zip(OK, ad.calls('convert')):
        s = seed_of(pid)
        assert c['torch_seed'] == s
        assert c['np'] == np.random.RandomState(s).random_sample() and c['py'] == random.Random(s).random()
    assert torch.seeds == [seed_of(p) for p in OK]


def test_outputs_do_not_depend_on_shards_order_or_resume(data, adapter, monkeypatch, tmp_path):
    out = {k: str(tmp_path / k) for k in ('one', 'shards', 'reversed', 'resumed')}
    adapter()
    assert harness(monkeypatch, *base_args(data, out['one']))[0] == 0
    ref = wav_bytes(out['one'])
    assert sorted(ref) == sorted(OK) and len(set(ref.values())) == 4
    for i in range(2):
        adapter()
        assert harness(monkeypatch, *base_args(data, out['shards']), '--shard', f'{i}/2')[0] == 0
    shards = {}
    for i in range(2):
        rows = rows_by_id(out['shards'], name=f'manifest.shard{i}of2.jsonl')
        assert sorted(rows) == sorted(OK[i::2])
        shards.update({p: open(r['wav'], 'rb').read() for p, r in rows.items()})
    assert shards == ref
    adapter()
    args = base_args(data, out['reversed'])
    args[1] = data.reversed
    assert harness(monkeypatch, *args)[0] == 0
    assert wav_bytes(out['reversed']) == ref
    adapter(fail_calls={1, 3})
    assert harness(monkeypatch, *base_args(data, out['resumed']))[0] == 1
    adapter()
    assert harness(monkeypatch, *base_args(data, out['resumed']))[0] == 0
    assert wav_bytes(out['resumed']) == ref


# ── Resume ───────────────────────────────────────────────────

def test_resume_retries_only_unfinished_rows(data, adapter, monkeypatch, tmp_path, capsys):
    out = str(tmp_path / 'out')
    adapter(fail_calls={2})
    assert harness(monkeypatch, *base_args(data, out)) == (1, '')
    rows = rows_by_id(out)
    assert sorted(rows) == sorted(OK)
    bad = rows[OK[1]]
    assert bad['error'] == 'RuntimeError: stand-in failure' and bad['shortfall'] is False
    assert bad['wav'] == os.path.join(run_dir(out), f'{OK[1]}.wav') and not os.path.exists(bad['wav'])
    done = {p: (os.stat(rows[p]['wav']).st_mtime_ns, open(rows[p]['wav'], 'rb').read()) for p in OK if p != OK[1]}

    ad = adapter()
    assert harness(monkeypatch, *base_args(data, out)) == (0, '')
    assert len(ad.calls('convert')) == 1 and len(ad.calls('load')) == 1
    lines = manifest(out)
    assert len(lines) == 4 and all('error' not in r for r in lines)   # compacted to the last row per pair
    for p, (mtime, b) in done.items():
        assert os.stat(rows[p]['wav']).st_mtime_ns == mtime and open(rows[p]['wav'], 'rb').read() == b
    rr = runs(out)
    assert [r['event'] for r in rr] == ['start', 'end', 'start', 'end']
    assert (rr[1]['n_errors'], rr[1]['n_pairs_done'], rr[3]['n_errors'], rr[3]['n_pairs_done']) == (1, 3, 0, 1)
    assert rr[0]['conv_cfg_sha'] == rr[2]['conv_cfg_sha']
    new = rows_by_id(out)
    assert new[OK[1]]['run_id'] == rr[2]['run_id'] and new[OK[0]]['run_id'] == rr[0]['run_id']   # (the same id when
    # both in-process runs fall in one second: run_id is time, host and pid)
    assert {r['conv_cfg_sha'] for r in new.values()} == {rr[0]['conv_cfg_sha']}

    capsys.readouterr()
    ad = adapter()
    before = open(os.path.join(run_dir(out), 'manifest.jsonl')).read()
    assert harness(monkeypatch, *base_args(data, out)) == (0, '')
    assert ad.log == [] and 'all 4 pairs complete' in capsys.readouterr().out   # the adapter is not even loaded
    assert open(os.path.join(run_dir(out), 'manifest.jsonl')).read() == before

    os.remove(new[OK[2]]['wav'])   # a row whose WAV is gone is converted again
    ad = adapter()
    assert harness(monkeypatch, *base_args(data, out)) == (0, '')
    assert len(ad.calls('convert')) == 1 and os.path.exists(new[OK[2]]['wav'])


# ── Error rows ───────────────────────────────────────────────

@pytest.mark.parametrize('kind', ['nan', 'inf', '2d', 'empty', 'int16', 'sr'])
def test_invalid_output_gives_error_rows_and_exit_1(data, adapter, monkeypatch, tmp_path, kind):
    out = str(tmp_path / 'out')
    stale = os.path.join(run_dir(out), f'{OK[0]}.wav')   # left by a killed run
    os.makedirs(os.path.dirname(stale))
    with open(stale, 'wb') as f:
        f.write(b'stale')
    adapter(sr=22050, sr_out=24000, bad=kind)
    assert harness(monkeypatch, *base_args(data, out)) == (1, '')
    rows = manifest(out)
    assert len(rows) == 4
    for r in rows:
        assert r['error'].startswith('ValueError: knnvc.convert returned') and 'at 24000 Hz' in r['error']
        assert r['shortfall'] is False and not os.path.exists(r['wav'])
    assert runs(out)[-1]['n_errors'] == 4 and runs(out)[-1]['n_pairs_done'] == 0
    adapter(sr=22050, sr_out=24000)                       # fixed: the error rows are retried
    assert harness(monkeypatch, *base_args(data, out)) == (0, '')
    assert all('error' not in r and sf.info(r['wav']).samplerate == 24000 for r in manifest(out))


def test_shortfall_gives_a_shortfall_row_not_an_error(data, adapter, monkeypatch, tmp_path):
    out = str(tmp_path / 'out')
    ad = adapter()
    assert harness(monkeypatch, *base_args(data, out, [OK[0], SHORT]), '--record') == (0, '')
    rows = rows_by_id(out)
    r = rows[SHORT]
    assert r['error'].startswith('ValueError: reference shortfall: 2.7 s available for budget 3 s')
    assert r['shortfall'] is True and not os.path.exists(r['wav']) and 'error' not in rows[OK[0]]
    assert len(ad.calls('convert')) == 1
    end = runs(out)[-1]
    assert (end['n_shortfall'], end['n_errors'], end['n_pairs_done']) == (1, 0, 1)
    rec = json.load(open(os.path.join(out, 'baselines_record.json')))['knnvc']
    assert (rec['n_rows'], rec['n_ok'], rec['n_error'], rec['n_shortfall']) == (2, 1, 0, 1)
    ad = adapter()                                        # retried on every run, still a shortfall, still exit 0
    assert harness(monkeypatch, *base_args(data, out, [OK[0], SHORT])) == (0, '')
    assert ad.calls('convert') == [] and rows_by_id(out)[SHORT]['shortfall'] is True


@pytest.mark.parametrize('ids, fail, code', [(OK, (), 0), ([SHORT], (), 0), (OK, (3,), 1), ([OK[0], SHORT], (1,), 1)],
                         ids=['ok', 'shortfall_only', 'error', 'error_and_shortfall'])
def test_exit_status(data, adapter, monkeypatch, tmp_path, ids, fail, code):
    adapter(fail_calls=fail)
    assert harness(monkeypatch, *base_args(data, str(tmp_path / 'out'), ids))[0] == code


# ── Manifest rows ────────────────────────────────────────────

def test_manifest_rows_are_run_conversions_base_rows_plus_the_baseline_fields(data, adapter, monkeypatch, tmp_path):
    out = str(tmp_path / 'out')
    adapter(sr=22050, sr_out=24000)
    assert harness(monkeypatch, *base_args(data, out, IDS)) == (0, '')
    ns = argparse.Namespace(out_dir=out, set='dev', budget='3', compat_spie=False)
    rows = rows_by_id(out)
    assert sorted(rows) == sorted(IDS)
    sha = {r['conv_cfg_sha'] for r in rows.values()}
    assert len(sha) == 1 and len(sha.pop()) == 16
    for pid, r in rows.items():
        base = rc.base_row(ns, data.rows[pid], 'knnvc', None)
        assert base['dump'] is not None
        base['dump'] = None                                   # a baseline has no codes to dump
        if pid == SHORT:
            assert set(r) == set(base) | {'error', 'shortfall', 'run_id', 'conv_cfg_sha'}
        else:
            assert set(r) == set(base) | EXTRA
            assert set(r['conv_time_s']) == TIMES and isinstance(r['conv_time_s']['ref_cached'], bool)
            info = sf.info(r['wav'])
            assert (info.samplerate, info.subtype, info.channels) == (24000, 'PCM_16', 1)
            assert r['out_s'] == round(info.frames / 24000, 3)
            assert r['src_s'] == round(len(librosa.load(data.rows[pid]['source_path'], sr=22050)[0]) / 22050, 3)
            m = metrics.as_metric_row(r)
            metrics.check_row(m, r['R'])                      # metrics.py's reference rule accepts the row
        assert {k: r[k] for k in base} == base
    r = rows[IDS[4]]
    assert r['tau'] is None and r['budget'] == 3 and r['system'] == 'knnvc' and r['set'] == 'dev'
    assert r['R'] == {'3': data.rows[IDS[4]]['R']['3']} and r['parallel_wav'] == data.rows[IDS[4]]['P']
    assert r['sim_ref_wavs'] == data.rows[IDS[4]]['E']['paths'] and r['ref_text'] == data.rows[IDS[4]]['source_text']
    assert r['wav'] == os.path.join(out, 'dev', '3', 'knnvc', f'{IDS[4]}.wav')
    for x in ('n_frames', 'n_ref_frames', 'same_codes_as', 'override_coverage'):
        assert x not in r


def test_out_dir_is_made_absolute_and_shards_get_their_own_manifest(data, adapter, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    adapter()
    assert harness(monkeypatch, *base_args(data, 'rel_out'), '--shard', '1/2')[0] == 0
    d = tmp_path / 'rel_out' / 'dev' / '3' / 'knnvc'
    assert sorted(os.listdir(d)) == sorted([f'{p}.wav' for p in OK[1::2]] + ['manifest.shard1of2.jsonl',
                                                                             'runs.jsonl'])
    rows = [json.loads(x) for x in open(d / 'manifest.shard1of2.jsonl')]
    assert [r['pair_id'] for r in rows] == OK[1::2]
    for r in rows:
        assert os.path.isabs(r['wav']) and os.path.realpath(r['wav']) == os.path.realpath(d / f'{r["pair_id"]}.wav')


# ── conv_cfg: refusal, record ────────────────────────────────

def test_manifest_written_under_another_conv_cfg_is_refused(data, adapter, monkeypatch, tmp_path):
    out = str(tmp_path / 'out')
    adapter()
    assert harness(monkeypatch, *base_args(data, out))[0] == 0
    os.remove(rows_by_id(out)[OK[3]]['wav'])                  # work left to do
    man = open(os.path.join(run_dir(out), 'manifest.jsonl')).read()
    runs_before = open(os.path.join(run_dir(out), 'runs.jsonl')).read()
    for change in ('cfg', 'checkpoint'):
        ad = adapter()
        args = base_args(data, out)
        if change == 'cfg':
            args += ['--cfg', '{"knnvc": {"topk": 8}}']
        else:
            ad.ckpt.write_bytes(b'other weights')            # describe() hashes the checkpoints
        code, msg = harness(monkeypatch, *args)
        assert code == 1 and 'holds rows of conv_cfg' in msg and 'use another --out_dir' in msg, msg
        assert ad.calls('convert') == []
        assert open(os.path.join(run_dir(out), 'manifest.jsonl')).read() == man
        assert open(os.path.join(run_dir(out), 'runs.jsonl')).read() == runs_before


def test_cfg_reaches_load_and_the_conv_cfg(data, adapter, monkeypatch, tmp_path):
    out = str(tmp_path / 'out')
    ad = adapter()
    args = base_args(data, out) + ['--cfg', '{"knnvc": {"topk": 8}, "seedvc": {"diffusion_steps": 10}}', '--record']
    assert harness(monkeypatch, *args)[0] == 0
    assert [(x['device'], x['cfg']) for x in ad.calls('load')] == [('cpu', {'topk': 8})]
    rec = json.load(open(os.path.join(out, 'baselines_record.json')))['knnvc']
    assert rec['cfg'] == {'topk': 8} and rec['describe']['settings'] == {'topk': 8}


def test_record_entry_and_its_merge(data, adapter, monkeypatch, tmp_path):
    out = str(tmp_path / 'out')
    os.makedirs(out)
    path = os.path.join(out, 'baselines_record.json')
    with open(path, 'w') as f:
        json.dump({'freevc': {'conv_cfg_sha': 'feedbeef', 'kept': True}}, f)
    ad = adapter()
    assert harness(monkeypatch, *base_args(data, out), '--record') == (0, '')
    rec = json.load(open(path))
    assert rec['freevc'] == {'conv_cfg_sha': 'feedbeef', 'kept': True}
    e = rec['knnvc']
    rows = rows_by_id(out)
    assert e['conv_cfg_sha'] == {r['conv_cfg_sha'] for r in rows.values()}.pop() == runs(out)[0]['conv_cfg_sha']
    assert e['describe']['checkpoints'] == {str(ad.ckpt): hashlib.sha256(ad.ckpt.read_bytes()).hexdigest()}
    assert {k: e['describe'][k] for k in ('repo', 'commit', 'sr_out', 'settings', 'notes')} == {
        'repo': 'stub://knnvc', 'commit': 'c0ffee0', 'sr_out': 16000, 'settings': {'topk': 4}, 'notes': 'stand-in'}
    assert (e['system'], e['sr_in'], e['cfg'], e['device_type'], e['reference'], e['wav_subtype']) == (
        'knnvc', 16000, {}, 'cpu', 'make_pairs.load_reference', 'PCM_16')
    assert e['seed'] == 'int(sha256(pair_id)[:8], 16)'
    assert e['versions']['numpy'] == np.__version__ and e['versions']['librosa'] == librosa.__version__
    assert ('torch' in e['versions']) == ('torch' in sys.modules)   # only modules already imported
    assert e['env']['executable'] == sys.executable and e['env']['prefix'] == sys.prefix
    assert set(e['env']['vars']) == set(rb.ENV_VARS) and e['cwd'] == os.getcwd() and e['gpu'] is None
    assert (e['set'], e['budget'], e['pairs'], e['manifest']) == (
        'dev', 3, data.pairs, os.path.join(run_dir(out), 'manifest.jsonl'))
    assert (e['n_rows'], e['n_ok'], e['n_error'], e['n_shortfall']) == (4, 4, 0, 0)
    assert e['run_id'] == runs(out)[0]['run_id'] and isinstance(e['rtf_median'], float)
    assert isinstance(e['command'], str) and e['command']
    assert [f for f in os.listdir(out) if f.startswith('baselines_record.json.tmp')] == []

    # another conv_cfg is refused before anything is converted or written, even into a new manifest
    record = open(path).read()
    ad = adapter()
    code, msg = harness(monkeypatch, *base_args(data, out, OK, '30'), '--record', '--cfg', '{"knnvc": {"topk": 8}}')
    assert code == 1 and 'records knnvc under conv_cfg' in msg and 'use another --out_dir' in msg, msg
    assert open(path).read() == record and ad.calls('convert') == []
    assert not os.path.exists(os.path.join(run_dir(out, '30'), 'manifest.jsonl'))
    assert not os.path.exists(os.path.join(run_dir(out, '30'), 'runs.jsonl'))

    # the same conv_cfg replaces the entry (budget 30 now) and keeps the other system's
    ad = adapter()
    assert harness(monkeypatch, *base_args(data, out, OK, '30'), '--record') == (0, '')
    rec = json.load(open(path))
    assert rec['knnvc']['budget'] == 30 and rec['knnvc']['conv_cfg_sha'] == e['conv_cfg_sha'] and 'freevc' in rec

    # --record with nothing to convert still loads, describes and records
    ad = adapter()
    assert harness(monkeypatch, *base_args(data, out), '--record') == (0, '')
    assert len(ad.calls('load')) == 1 and ad.calls('convert') == []
    rec = json.load(open(path))['knnvc']
    assert rec['budget'] == 3 and rec['n_ok'] == 4 and rec['run_id'] == runs(out)[-1]['run_id']


def test_describe_must_have_the_contract_keys(data, adapter, monkeypatch, tmp_path):
    adapter(lacks=('settings', 'commit'))
    code, msg = harness(monkeypatch, *base_args(data, str(tmp_path / 'out')))
    assert code == 1 and "baselines.knnvc.describe() lacks ['commit', 'settings']" in msg


# ── Device ───────────────────────────────────────────────────

def test_device_cuda_without_cuda_is_refused(data, adapter, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch(cuda=False))
    ad = adapter()
    args = base_args(data, str(tmp_path / 'out'))
    args[args.index('--device') + 1] = 'cuda'
    code, msg = harness(monkeypatch, *args)
    assert code == 1 and 'CUDA is not available' in msg and ad.log == []


@pytest.mark.parametrize('cuda', [False, True])
def test_device_auto(data, adapter, monkeypatch, tmp_path, cuda):
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch(cuda=cuda))
    ad = adapter()
    out = str(tmp_path / 'out')
    args = base_args(data, out)
    args[args.index('--device') + 1] = 'auto'
    assert harness(monkeypatch, *args, '--record')[0] == 0
    dev = 'cuda' if cuda else 'cpu'
    assert ad.calls('load')[0]['device'] == dev
    start = runs(out)[0]
    assert start['conv_cfg']['device_type'] == dev and start['conv_cfg']['versions']['torch'] == 'fake'
    assert start['gpu'] == ('Fake GPU 2g.35gb' if cuda else None)
    assert json.load(open(os.path.join(out, 'baselines_record.json')))['knnvc']['gpu'] == start['gpu']


# ── Arguments ────────────────────────────────────────────────

@pytest.mark.parametrize('change, msg', [
    (['--systems', 'faiss'], 'give distinct names of knnvc,freevc,seedvc,facodec'),
    (['--systems', 'knnvc,knnvc'], 'give distinct names'),
    (['--systems', ''], 'give distinct names'),
    (['--cfg', '[1]'], 'is not a JSON object'),
    (['--cfg', 'nonsense'], 'is not a JSON object'),
    (['--cfg', '{"bogus": {}}'], 'is not a JSON object'),
    (['--cfg', '{"knnvc": 4}'], 'is not a JSON object'),
    (['--budget', '60'], 'not in 3, 30, 180'),
    (['--shard', '2/2'], 'is not i/n with 0 <= i < n'),
    (['--set', 'ltc2k'], "has rows of set ['dev'], not 'ltc2k'"),
], ids=lambda x: x if isinstance(x, str) else ' '.join(x))
def test_bad_arguments_are_refused(data, adapter, monkeypatch, tmp_path, change, msg):
    ad = adapter()
    args = base_args(data, str(tmp_path / 'out'))
    for k, v in zip(change[::2], change[1::2]):
        if k in args:
            args[args.index(k) + 1] = v
        else:
            args += [k, v]
    code, got = harness(monkeypatch, *args)
    assert code == 1 and msg in got and ad.log == [], got


def test_abbreviated_options_are_rejected(data, monkeypatch, tmp_path):
    args = base_args(data, str(tmp_path / 'out'))
    args[args.index('--systems')] = '--sys'                   # child_argv rewrites --systems by its full spelling
    assert harness(monkeypatch, *args)[0] == 2


@pytest.mark.parametrize('form', [['--systems', 'knnvc,freevc'], ['--systems=knnvc,freevc']])
def test_child_argv_runs_this_command_for_one_system(monkeypatch, form):
    monkeypatch.setattr(sys, 'argv', ['bench/run_baselines.py', '--pairs', 'p.jsonl', *form, '--record',
                                      '--cfg', '{"freevc": {}}'])
    assert rb.child_argv('freevc') == [sys.executable, os.path.abspath('bench/run_baselines.py'), '--pairs', 'p.jsonl',
                                       '--record', '--cfg', '{"freevc": {}}', '--systems', 'freevc']


# ── Several systems: one interpreter each ────────────────────

STUB_FILE = r'''
"""Stand-in adapter loaded by sitecustomize.py; logs to STUB_LOG."""
import json, os
import numpy as np
NAME = os.path.basename(__file__)[:-3]
SR = {'knnvc': 16000, 'facodec': 16000, 'freevc': 16000, 'seedvc': 22050}[NAME]
CKPT = os.path.join(os.environ['STUB_ADAPTER_DIR'], NAME + '.bin')
_n = [0]


def _log(**kw):
    with open(os.environ['STUB_LOG'], 'a') as f:
        f.write(json.dumps(dict(kw, system=NAME, pid=os.getpid())) + '\n')


def load(device, **cfg):
    _log(event='load', device=device, cfg=cfg)


def convert(src, ref, sr):
    _n[0] += 1
    _log(event='convert', n=_n[0])
    if os.environ.get('STUB_FAIL') == NAME and _n[0] == 2:
        raise RuntimeError('stand-in failure')
    return (0.25 * src + 0.05 * np.random.standard_normal(len(src))).astype(np.float32), SR


def describe():
    return {'repo': 'stub://' + NAME, 'commit': 'c0ffee0', 'checkpoints': [CKPT], 'sr_out': SR, 'settings': {}}
'''

SITECUSTOMIZE = r'''
import importlib.util, os, sys
d = os.environ.get('STUB_ADAPTER_DIR')
for f in sorted(os.listdir(d)) if d else ():
    if f.endswith('.py'):
        spec = importlib.util.spec_from_file_location('baselines.' + f[:-3], os.path.join(d, f))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
'''


def test_several_systems_run_one_process_each_and_record_their_own_command(data, tmp_path):
    site, stubs, out, log = tmp_path / 'site', tmp_path / 'stubs', str(tmp_path / 'out'), tmp_path / 'log.jsonl'
    site.mkdir()
    stubs.mkdir()
    (site / 'sitecustomize.py').write_text(SITECUSTOMIZE)
    for name in ('facodec', 'knnvc'):
        (stubs / f'{name}.py').write_text(STUB_FILE)
        (stubs / f'{name}.bin').write_bytes(name.encode() * 50)
    env = dict(os.environ, PYTHONPATH=str(site), STUB_ADAPTER_DIR=str(stubs), STUB_LOG=str(log), STUB_FAIL='facodec')
    cmd = [sys.executable, RUN_BASELINES, '--pairs', data.pairs, '--set', 'dev', '--budget', '3', '--out_dir', out,
           '--device', 'cpu', '--record', '--pair_ids', ','.join(OK)]
    p = subprocess.run(cmd + ['--systems', 'facodec,knnvc'], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 1 and 'ERROR: run_baselines failed for facodec' in p.stderr, p.stdout + p.stderr
    events = [json.loads(x) for x in open(log)]
    pids = {e['system']: e['pid'] for e in events if e['event'] == 'load'}
    assert len(pids) == 2 and len(set(pids.values())) == 2      # a fresh interpreter per system, in --systems order
    assert [e['system'] for e in events if e['event'] == 'load'] == ['facodec', 'knnvc']
    assert all(e['pid'] == pids[e['system']] for e in events)
    fac, knn = rows_by_id(out, system='facodec'), rows_by_id(out, system='knnvc')
    assert sum('error' in r for r in fac.values()) == 1 and all('error' not in r for r in knn.values())
    assert len(fac) == len(knn) == 4
    rec = json.load(open(os.path.join(out, 'baselines_record.json')))
    assert sorted(rec) == ['facodec', 'knnvc'] and rec['facodec']['n_error'] == 1 and rec['knnvc']['n_ok'] == 4
    for name in rec:
        c = rec[name]['command']
        assert c.endswith(f'--systems {name}') and 'facodec,knnvc' not in c and c.startswith(sys.executable)
        assert rec[name]['env']['executable'] == sys.executable
        assert rec[name]['conv_cfg_sha'] == {r['conv_cfg_sha'] for r in rows_by_id(out, system=name).values()}.pop()

    # fixed and run again (the --systems=... spelling): only facodec's error row is converted, both are re-recorded
    log.write_text('')
    env.pop('STUB_FAIL')
    sha = {k: v['conv_cfg_sha'] for k, v in rec.items()}
    p = subprocess.run(cmd + ['--systems=knnvc,facodec'], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    events = [json.loads(x) for x in open(log)]
    assert [(e['system'], e['event']) for e in events] == [('knnvc', 'load'), ('facodec', 'load'),
                                                          ('facodec', 'convert')]
    rec2 = json.load(open(os.path.join(out, 'baselines_record.json')))
    assert {k: v['conv_cfg_sha'] for k, v in rec2.items()} == sha and rec2['facodec']['n_ok'] == 4
    assert all(rec2[k]['run_id'] != rec[k]['run_id'] for k in rec)   # each invocation its own run_id
