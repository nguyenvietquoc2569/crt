"""Tests of mvc/probe_campplus.py with speaker-dependent synthetic audio, a spectral stand-in for CAM++ and FakeMimi:
the --path_map rewrite, the Dev-list file set, segmenting (concatenated pieces, per-file pieces, the cap), the all-pairs
trials and their EER, the gate, and a full run through main() with its JSON, cache reuse and cache invalidation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import probe_campplus as pc  # noqa: E402

PERIOD = 960                  # 40 ms at 24 kHz: divides Mimi's 1920-sample frame and every segment length used here
MAC_DEV = Path(__file__).resolve().parents[2] / 'paper/journal_extension/results/v1/pairs/dev.jsonl'
OLD = '/datastore/nvquoc/crt'


def voice(spk, n, seed, sr=em.SR):
    """n samples of a periodic 'voice' (6 speaker-specific harmonics of 25 Hz) plus utterance-specific noise."""
    g = np.random.default_rng(list(spk.encode()))
    k, ph = g.choice(np.arange(2, 150), 6, replace=False), g.uniform(0, 2 * np.pi, 6)
    t = np.arange(n) * em.SR / sr
    x = sum(np.sin(2 * np.pi * kk * t / PERIOD + p) for kk, p in zip(k, ph)) / 6
    return (0.5 * x + 0.02 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


class SpecCam:
    """CAM++ stand-in: the mean magnitude spectrum of PERIOD-sample frames (192 bins); counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, wav, sr):
        assert sr == em.SR
        self.calls += 1
        n = len(wav) // PERIOD * PERIOD
        frames = np.asarray(wav[:n], dtype=np.float64).reshape(-1, PERIOD)
        return np.abs(np.fft.rfft(frames, axis=1)).mean(0)[:192].astype(np.float32)


def test_path_map_rewrites_the_prefix_only():
    maps = pc.parse_map([f'{OLD}=/datastore/ceai/crt', '/a=/b'])
    assert maps == [(OLD, '/datastore/ceai/crt'), ('/a', '/b')]
    assert pc.map_path(f'{OLD}/mimi_vc/x.flac', maps) == '/datastore/ceai/crt/mimi_vc/x.flac'
    assert pc.map_path(f'/x{OLD}/y', maps) == f'/x{OLD}/y'
    assert pc.map_path('/a/b', maps) == '/b/b' and pc.map_path('/c', maps) == '/c'
    assert pc.map_path('/a/b', []) == '/a/b'
    for bad in ('no_separator', '=/new'):
        with pytest.raises(SystemExit, match='OLD=NEW'):
            pc.parse_map([bad])


def test_vctk_files_take_sources_and_E_paths_by_speaker():
    def p(spk, i):
        return f'{OLD}/mimi_vc/benchmark/vctk/wav48_silence_trimmed/{spk}/{spk}_{i:03d}_mic1.flac'
    rows = [dict(source_speaker='p1', source_path=p('p1', 1), target_speaker='p2', E={'paths': [p('p2', 5), p('p2', 6)]}),
            dict(source_speaker='p2', source_path=p('p2', 5), target_speaker='p1', E={'paths': [p('p1', 7)]}),
            dict(source_speaker='p1', source_path=p('p1', 1), target_speaker='p3', E={'paths': [p('p3', 2)]})]
    f = pc.vctk_files(rows, pc.parse_map([f'{OLD}=/new']))
    assert list(f) == ['p1', 'p2', 'p3']
    assert [Path(x).name for x in f['p1']] == ['p1_001_mic1.flac', 'p1_007_mic1.flac']
    assert [Path(x).name for x in f['p2']] == ['p2_005_mic1.flac', 'p2_006_mic1.flac']        # deduplicated
    assert all(x.startswith('/new/mimi_vc/') for v in f.values() for x in v)


@pytest.mark.skipif(not MAC_DEV.exists(), reason='no Mac copy of the Dev pair list')
def test_vctk_files_on_the_mac_dev_list():
    rows = [json.loads(l) for l in open(MAC_DEV)]
    f = pc.vctk_files(rows, pc.parse_map([f'{OLD}=/datastore/ceai/crt']))
    assert len(f) == 20 and sum(len(v) for v in f.values()) == 389
    assert all(x.startswith('/datastore/ceai/crt/') and Path(x).parent.name == s for s, v in f.items() for x in v)


def test_spread_is_even_ordered_and_capped():
    assert pc.spread(3, 5) == [0, 1, 2] and pc.spread(30, 30) == list(range(30))
    s = pc.spread(81, 20)
    assert len(s) == 20 and s[0] == 0 and s[-1] == 80 and all(b > a for a, b in zip(s, s[1:]))
    assert pc.spread(7, 1) == [0] and pc.spread(0, 20) == []


def test_concat_segments_follow_the_file_order_and_drop_the_tail():
    audio = {'a1': np.arange(0, 7), 'a2': np.arange(7, 15), 'a3': np.arange(15, 26), 'b1': np.arange(100, 109)}
    segs = list(pc.concat_segments({'a': ['a1', 'a2', 'a3'], 'b': ['b1']}, audio.__getitem__, 10 / em.SR, 20))
    assert [(i, s) for i, s, _ in segs] == [('a_c000', 'a'), ('a_c001', 'a')]           # b is shorter than a segment
    assert np.array_equal(segs[0][2], np.arange(10)) and np.array_equal(segs[1][2], np.arange(10, 20))
    assert segs[0][2].dtype == np.float32
    long = {'x': np.zeros(81 * 10)}
    capped = list(pc.concat_segments({'s': ['x']}, long.__getitem__, 10 / em.SR, 20))
    assert len(capped) == 20 and capped[0][0] == 's_c000' and capped[-1][0] == 's_c080'


def test_file_segments_cut_each_file_and_skip_short_ones():
    audio = {'d/p1_001_mic1.flac': np.arange(25.0), 'd/p1_002_mic1.flac': np.arange(5.0),
             'd/p1_003_mic1.flac': np.arange(100.0, 112.0)}
    files = {'p1': sorted(audio)}
    segs = list(pc.file_segments(files, audio.__getitem__, 10 / em.SR, 30))
    assert [i for i, _, _ in segs] == ['p1_001_mic1_0', 'p1_001_mic1_1', 'p1_003_mic1_0']
    assert np.array_equal(segs[1][2], np.arange(10.0, 20.0)) and np.array_equal(segs[2][2], np.arange(100.0, 110.0))
    assert [i for i, _, _ in pc.file_segments(files, audio.__getitem__, 10 / em.SR, 2)] == ['p1_001_mic1_0',
                                                                                          'p1_003_mic1_0']


def test_libri_files_group_by_speaker_in_id_order(tmp_path):
    for rel in ('84/121550/84_121550_000002_000000', '84/121123/84_121123_000008_000000',
                '84/121123/84_121123_000001_000003', '174/50561/174_50561_000004_000000'):
        p = tmp_path / 'dev-clean' / f'{rel}.wav'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'')
    f = pc.libri_files(tmp_path)
    assert list(f) == ['174', '84']
    assert [Path(x).stem for x in f['84']] == ['84_121123_000001_000003', '84_121123_000008_000000',
                                               '84_121550_000002_000000']
    with pytest.raises(SystemExit, match='no wav files'):
        pc.libri_files(tmp_path, 'dev-other')


def test_embed_batches_and_averages_the_stored_frames():
    sizes = []

    def mimi(ws):
        sizes.append(len(ws))
        return [(np.tile(np.arange(em.n_frames(len(w)), dtype=np.float32)[:, None], (1, 512)), None) for w in ws]

    segs = [(f's{i}', 'a' if i < 3 else 'b', np.zeros(5 * em.HOP, np.float32)) for i in range(5)]
    ids, spk, C, M = pc.embed(iter(segs), SpecCam(), mimi, batch=2)
    assert ids == [f's{i}' for i in range(5)] and spk == ['a'] * 3 + ['b'] * 2 and sizes == [2, 2, 1]
    assert C.shape == (5, 192) and M.shape == (5, 512)
    assert np.allclose(M, 1.0)                     # frames 0, 1, 2: the last TRIM of the 5 frames are not used


def test_trials_cover_every_pair_and_eer_separates():
    rng = np.random.default_rng(0)
    spk = [f's{i % 8}' for i in range(80)]
    centres = rng.standard_normal((8, 32))
    good = pc.l2(np.stack([centres[i % 8] + 0.3 * rng.standard_normal(32) for i in range(80)]))
    r = pc.score(good, spk)
    assert r['n_trials'] == 80 * 79 // 2 and r['n_genuine'] == 8 * (10 * 9 // 2) and r['eer'] < 0.02
    bad = pc.score(pc.l2(rng.standard_normal((80, 32))), spk)
    assert 0.35 < bad['eer'] < 0.65
    g, imp = pc.trials(pc.l2(np.array([[1.0, 0], [1, 0.1], [0, 1]])), ['a', 'a', 'b'])
    assert len(g) == 1 and len(imp) == 2 and g[0] > imp.max()


def test_decide_applies_both_gates():
    ok = {'dev-clean': {'passed': True}, 'vctk-dev': {'passed': True}}
    assert pc.decide(ok)[0] is True
    assert pc.decide({**ok, 'vctk-dev': {'passed': False}})[0] is False
    assert pc.decide({'dev-clean': {'passed': True}})[0] is None
    assert pc.GATES == {'dev-clean': 0.05, 'vctk-dev': 0.08}


def make_sets(root):
    """dev-clean: 4 speakers x 2 chapters x 3 utterances (24 kHz wav); VCTK: 4 speakers x 5 files (48 kHz flac, one
    shorter than a segment) under root/ceai/crt, and a pair list naming them with the old /datastore/nvquoc prefix."""
    import soundfile as sf
    raw = root / 'LibriTTS_R'
    for s in ('84', '174', '251', '422'):
        for c in ('1001', '1002'):
            for u in range(3):
                p = raw / 'dev-clean' / s / c / f'{s}_{c}_{u:06d}_000000.wav'
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, voice(s, PERIOD * (10 + 15 * u), int(s) + int(c) + u), em.SR)
    vk = 'mimi_vc/benchmark/vctk/wav48_silence_trimmed'
    spks = ['p225', 'p226', 'p227', 'p228']
    for s in spks:
        for j, sec in enumerate((0.32, 0.88, 1.2, 1.6, 1.92)):
            p = root / 'ceai/crt' / vk / s / f'{s}_{j:03d}_mic1.flac'
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, voice(s, round(48000 * sec), j, sr=48000), 48000)
    rows = []
    for s in spks:
        for t in spks:
            if t != s:
                rows.append(dict(pair_id=f'dev__{s}__{t}', source_speaker=s, target_speaker=t,
                                 source_path=f'{OLD}/{vk}/{s}/{s}_000_mic1.flac',
                                 E={'paths': [f'{OLD}/{vk}/{t}/{t}_{j:03d}_mic1.flac' for j in range(1, 5)]}))
    pairs = root / 'dev.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return raw, pairs


def test_main_end_to_end_with_cache(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('soundfile')
    pytest.importorskip('librosa')
    from test_encode_mimi import FakeMimi
    raw, pairs = make_sets(tmp_path)
    cam, enc = SpecCam(), em.MimiEncoder('cpu', model=FakeMimi())
    monkeypatch.setattr(pc.se, 'CamPlus', lambda device: cam)
    monkeypatch.setattr(pc.em, 'MimiEncoder', lambda device: enc)
    out = tmp_path / 'results'
    argv = ['--raw', str(raw), '--pairs', str(pairs), '--path_map', f'{OLD}={tmp_path}/ceai/crt', '--out', str(out),
            '--seg_libri', '0.8', '--max_libri', '5', '--seg_vctk', '0.4', '--max_vctk', '10', '--device', 'cpu']
    assert pc.main(argv) == 0
    res = json.loads((out / 'p4_campplus.json').read_text())
    lib, vc = res['sets']['dev-clean'], res['sets']['vctk-dev']
    assert lib['n_speakers'] == 4 and lib['n_files'] == 24 and lib['n_segments'] == 20       # 7 pieces each, cap 5
    assert lib['n_trials'] == 20 * 19 // 2 and lib['n_genuine'] == 4 * 10
    assert vc['n_speakers'] == 4 and vc['n_files'] == 20 and vc['n_segments'] == 40          # 0+2+3+4+4, cap 10
    assert vc['n_trials'] == 40 * 39 // 2 and vc['n_genuine'] == 4 * 45 and vc['segments_per_speaker'] == [10, 10]
    for r in (lib, vc):
        assert r['eer']['campplus'] < 0.05 and r['eer']['mimi_lat'] < 0.05 and r['passed']
    assert res['keep_campplus'] is True and res['decision'].startswith('keep CAM++')
    assert {p.name for p in out.iterdir()} == {'p4_campplus.json', 'p4_dev-clean.npz', 'p4_vctk-dev.npz'}
    assert cam.calls == 60
    assert pc.main(argv) == 0 and cam.calls == 60                                              # both sets cached
    assert json.loads((out / 'p4_campplus.json').read_text())['sets'] == res['sets']
    argv[argv.index('--max_libri') + 1] = '3'
    assert pc.main(argv) == 0 and cam.calls == 72                                              # dev-clean redone
    assert json.loads((out / 'p4_campplus.json').read_text())['sets']['dev-clean']['n_segments'] == 12


def test_unknown_set_is_an_error(tmp_path):
    a = pc.argparse.Namespace(sets='dev-other', path_map=[], out=str(tmp_path))
    with pytest.raises(SystemExit, match='unknown set'):
        pc.probe(a, None, None)


def test_a_speaker_blind_embedder_fails_the_gate(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('soundfile')
    pytest.importorskip('librosa')
    from test_encode_mimi import FakeMimi
    raw, pairs = make_sets(tmp_path)
    rng = np.random.default_rng(0)
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    monkeypatch.setattr(pc.se, 'CamPlus', lambda device: lambda wav, sr: rng.standard_normal(192).astype(np.float32))
    monkeypatch.setattr(pc.em, 'MimiEncoder', lambda device: enc)
    pc.main(['--raw', str(raw), '--pairs', str(pairs), '--path_map', f'{OLD}={tmp_path}/ceai/crt',
             '--out', str(tmp_path / 'r'), '--seg_libri', '0.8', '--seg_vctk', '0.4', '--device', 'cpu'])
    res = json.loads((tmp_path / 'r' / 'p4_campplus.json').read_text())
    for r in res['sets'].values():
        assert r['eer']['campplus'] > 0.25 and not r['passed'] and r['eer']['mimi_lat'] < 0.05
    assert res['keep_campplus'] is False and 'Mimi-AAM' in res['decision']
