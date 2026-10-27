"""Tests of mvc/run_mvc.py (with stub journal modules) and mvc/vc_adapter.py (a real tiny checkpoint, a fake Mimi)."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mvc  # noqa: E402
import train as tr  # noqa: E402
import vc_adapter  # noqa: E402


class StubRC:
    def __init__(self, rows):
        self.rows = rows

    def load_pairs(self, a):
        rows = self.rows
        if a.shard:
            rows = rows[a.shard[0]::a.shard[1]]
        return rows

    @staticmethod
    def load_manifest(path):
        out = {}
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l)
                out[(r['pair_id'], r['system'])] = r
        return out


class StubRB:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    @staticmethod
    def describe(adapter, name):
        return adapter.describe()

    @staticmethod
    def base_row(a, row, name):
        return dict(pair_id=row['pair_id'], system=name, wav=os.path.join(a.out_dir, a.set, a.budget, name, row['pair_id'] + '.wav'))

    def convert_pair(self, adapter, sr_out, row, a, name, cache):
        self.calls.append(row['pair_id'])
        if row['pair_id'] in self.fail:
            raise RuntimeError('boom')
        r = self.base_row(a, row, name)
        Path(r['wav']).write_bytes(b'RIFF')
        return r


class StubAdapter:
    SR = 24000

    def __init__(self, tag='x'):
        self.tag = tag

    def describe(self):
        return dict(repo='r', commit=self.tag, checkpoints=[], sr_out=24000, settings={})


def args(tmp_path, **kw):
    a = run_mvc.parse(['--journal', '.', '--pairs', 'p', '--set', 'dev', '--name', 'mvc', '--out_dir', str(tmp_path)])
    a.shard = kw.pop('shard', '')
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_run_converts_resumes_and_records_errors(tmp_path):
    rows = [dict(pair_id=f'p{i}') for i in range(6)]
    rb = StubRB(fail={'p3'})
    assert run_mvc.run(args(tmp_path), StubAdapter(), rb, StubRC(rows), log=lambda m: None) == 1
    man = tmp_path / 'dev' / '30' / 'mvc' / 'manifest.jsonl'
    got = {json.loads(l)['pair_id']: json.loads(l) for l in open(man)}
    assert set(got) == {f'p{i}' for i in range(6)} and 'error' in got['p3'] and 'error' not in got['p0']
    rb2 = StubRB()
    assert run_mvc.run(args(tmp_path), StubAdapter(), rb2, StubRC(rows), log=lambda m: None) == 0
    assert rb2.calls == ['p3']                                          # only the failed pair is redone
    with pytest.raises(SystemExit, match='conv_cfg'):
        rows2 = rows + [dict(pair_id='p9')]
        run_mvc.run(args(tmp_path), StubAdapter('other'), StubRB(), StubRC(rows2), log=lambda m: None)


def test_run_shard_manifest_name(tmp_path):
    rows = [dict(pair_id=f'p{i}') for i in range(10)]
    rb = StubRB()
    run_mvc.run(args(tmp_path, shard=(0, 5)), StubAdapter(), rb, StubRC(rows), log=lambda m: None)
    assert rb.calls == ['p0', 'p5'] and (tmp_path / 'dev' / '30' / 'mvc' / 'manifest.shard0of5.jsonl').exists()


class FakeMimiIO:
    encodes = 0
    decoder_ckpts = []

    def __init__(self, device, decoder_ckpt=None):
        FakeMimiIO.decoder_ckpts.append(decoder_ckpt)

    def encode(self, wav):
        FakeMimiIO.encodes += 1
        T = -(-len(wav) // 1920)
        rng = np.random.default_rng(len(wav))
        return rng.standard_normal((T, 512)).astype(np.float32), rng.integers(0, 2048, (T, 32))

    def decode_latent(self, x):
        return np.zeros(len(x) * 1920, np.float32)

    def quantize(self, x):
        return np.zeros((len(x), 32), np.int64)

    def decode_codes(self, c):
        return np.zeros(len(c) * 1920, np.float32)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    a = tr.parse(['--train_dirs', 'x', '--val_dirs', 'x', '--out', str(tmp_path), '--d', '32', '--layers', '1',
                  '--heads', '2', '--content_dim', '8', '--device', 'cpu'])
    m = tr.build(a)
    torch.save(dict(ema={k: v.float() for k, v in m.state_dict().items()}, step=7, args=vars(a)), tmp_path / 'ema_last.pt')
    np.savez(tmp_path / 'stats.npz', mean=np.zeros(512, np.float32), std=np.ones(512, np.float32))
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=FakeMimiIO))
    return tmp_path


def test_adapter_converts_with_cache_and_describes(run_dir):
    ad = vc_adapter.MimiVC()
    with pytest.raises(ValueError, match='unknown'):
        ad.load('cpu', run_dir=str(run_dir), nonsense=1)
    ad = vc_adapter.MimiVC()
    ad.load('cpu', run_dir=str(run_dir), steps=2)
    ref = np.zeros(24000 * 45, np.float32)
    ref.flags.writeable = False
    FakeMimiIO.encodes = 0
    src = np.zeros(24000 * 2 + 100, np.float32)
    w1, sr = ad.convert(src, ref, 24000)
    w2, _ = ad.convert(src, ref, 24000)
    assert sr == 24000 and w1.shape == src.shape and w1.dtype == np.float32
    assert FakeMimiIO.encodes == 2 + 2              # ref: its 30 s and 15 s chunks, once; src: once per call
    d = ad.describe()
    assert set(d) >= {'repo', 'commit', 'checkpoints', 'sr_out', 'settings'} and d['settings']['train_step'] == 7
    with pytest.raises(ValueError, match='24000'):
        ad.convert(src, ref, 16000)


def test_copy_modes_need_no_checkpoint(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=FakeMimiIO))
    ad = vc_adapter.MimiVC()
    ad.load('cpu', mode='copy_latent')
    src = np.zeros(24000 + 5, np.float32)
    w, sr = ad.convert(src, np.zeros(24000 * 5, np.float32), 24000)
    assert w.shape == src.shape and sr == 24000 and ad.describe()['checkpoints'] == []
    ad2 = vc_adapter.MimiVC()
    ad2.load('cpu', mode='copy_codes')
    assert ad2.convert(src, np.zeros(10, np.float32), 24000)[0].shape == src.shape


class FakeCam:
    def __init__(self, device):
        self.calls = []

    def __call__(self, wav, sr=24000):
        self.calls.append(len(wav))
        return np.full(192, len(wav) / sr, np.float32)


def test_adapter_v1_chunks_and_windowed_speaker_vector(tmp_path, monkeypatch):
    a = tr.parse(['--train_dirs', 'x', '--val_dirs', 'x', '--out', str(tmp_path), '--d', '32', '--layers', '1',
                  '--heads', '2', '--content_dim', '8', '--device', 'cpu', '--sampler', 'prompt', '--pos', 'segment_random',
                  '--spk', '192'])
    m = tr.build(a)
    torch.save(dict(ema={k: v.float() for k, v in m.state_dict().items()}, step=3, args=vars(a)), tmp_path / 'ema_last.pt')
    np.savez(tmp_path / 'stats.npz', mean=np.zeros(512, np.float32), std=np.ones(512, np.float32))
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=FakeMimiIO))
    cam = FakeCam('cpu')
    monkeypatch.setitem(sys.modules, 'spk_embed', types.SimpleNamespace(CamPlus=lambda device: cam))
    ad = vc_adapter.MimiVC()
    ad.load('cpu', run_dir=str(tmp_path), steps=2)
    ref = np.zeros(24000 * 180, np.float32)
    chunks, g = ad.ref_features(ref)
    assert len(chunks) == 3 and sum(len(c[0]) for c in chunks) == 2250 and max(len(c[0]) for c in chunks) <= 750
    assert len(cam.calls) == 18 and np.isclose(np.linalg.norm(g), 1)      # 18 windows of 10 s
    w, sr = ad.convert(np.zeros(24000 * 2, np.float32), ref, 24000)
    assert w.shape == (48000,)
    ad.ref_cache = {}
    short = np.zeros(24000 * 3, np.float32)
    chunks, g = ad.ref_features(short)
    assert len(chunks) == 1 and len(cam.calls) == 19                       # one window at 3 s


def test_latent_floors():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((200, 512)).astype(np.float32) * 2 + 5
    src = ref[[3, 7, 11]] * 1.01
    out = vc_adapter.knn_latent(src, ref, k=1)
    assert np.allclose(out, ref[[3, 7, 11]])
    src2 = rng.standard_normal((400, 512)).astype(np.float32)
    mapped = vc_adapter.ot_latent(src2, ref)
    assert np.allclose(mapped.mean(0), ref.mean(0), atol=1e-3)
    assert np.allclose(np.cov(mapped.T), np.cov(ref.T), atol=0.6)


def test_floor_modes_run_through_the_adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=FakeMimiIO))
    for mode in ('knn_latent', 'ot_latent'):
        ad = vc_adapter.MimiVC()
        ad.load('cpu', mode=mode)
        w, sr = ad.convert(np.zeros(24000 * 2, np.float32), np.zeros(24000 * 40, np.float32), 24000)
        assert w.shape == (48000,) and np.all(np.isfinite(w))


def test_naturalness_knobs_have_neutral_defaults_and_are_validated():
    """solver, temp, cfg_rescale, sharpen, gv, stats_path, decoder_ckpt and init_t exist with defaults that change
    nothing."""
    import vc_adapter as va
    with pytest.raises(ValueError, match='unknown settings'):
        va.MimiVC().load('cpu', run_dir='x', warp=1)
    d = va.DEFAULTS
    assert (d['solver'], d['temp'], d['cfg_rescale'], d['sharpen'], d['gv'], d['stats_path'], d['decoder_ckpt'],
            d['init_t'], d['match_rms'], d['relax'], d['relax_k'], d['cfg_branch'], d['neg_src'], d['neg_src_until']) == \
        ('euler', 1.0, 0.0, 0.0, 0.0, None, None, 0.0, False, 0.0, 2, 'std', 0.0, 1.0)


class LoudFakeMimiIO(FakeMimiIO):
    def decode_latent(self, x):
        return np.full(len(x) * 1920, 0.5, np.float32)


def test_match_rms_scales_the_output_to_the_source_level(run_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=LoudFakeMimiIO))
    src = np.random.default_rng(0).standard_normal(24000 * 2).astype(np.float32) * 0.05
    ref = np.zeros(24000 * 10, np.float32)
    plain, levelled = vc_adapter.MimiVC(), vc_adapter.MimiVC()
    plain.load('cpu', run_dir=str(run_dir), steps=2)
    levelled.load('cpu', run_dir=str(run_dir), steps=2, match_rms=True)
    w0, _ = plain.convert(src, ref, 24000)
    w1, _ = levelled.convert(src, ref, 24000)
    rms = lambda w: float(np.sqrt(np.mean(w ** 2)))  # noqa: E731
    assert abs(rms(w0) - 0.5) < 1e-6 and abs(rms(w1) - rms(src)) < 1e-5


def test_mimi_io_relax_uses_the_stock_decoder_and_damps_the_round_trip():
    import mimi_io
    torch_ = pytest.importorskip('torch')
    io = object.__new__(mimi_io.MimiIO)
    io.torch = torch_
    io.m = types.SimpleNamespace(upsample='U', decoder_transformer='TR', decoder='STOCK')
    io.dec = dict(upsample='u2', decoder_transformer='tr2', decoder='FINE')            # a fine-tuned decoder is loaded
    seen = []
    io.decoder_input = lambda x: np.asarray(x)
    io._decode_input = lambda emb, d=None: (seen.append(d['decoder']), np.repeat(emb.mean(1), 1920))[1]
    io.encode = lambda w: (np.repeat(w.reshape(-1, 1920).mean(1, keepdims=True), 512, 1) * 0 + 3.0, None)   # E(D(x)) = 3
    x = np.random.default_rng(0).standard_normal((5, 512)).astype(np.float32)
    assert np.allclose(io.relax(x, 1.0, 1), 3.0) and set(seen) == {'STOCK'}
    two = io.relax(x, 0.25, 2)
    one = x + 0.25 * (3.0 - x)
    assert np.allclose(two, one + 0.25 * (3.0 - one)) and two.shape == x.shape
    assert np.array_equal(io.relax(x, 0.0, 2), x)


def test_adapter_relax_setting_calls_the_round_trip(run_dir, monkeypatch):
    calls = []

    class RelaxIO(FakeMimiIO):
        def relax(self, x, alpha, k=2):
            calls.append((alpha, k))
            return np.asarray(x) + 1.0
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=RelaxIO))
    src, ref = np.zeros(24000 * 2, np.float32), np.zeros(24000 * 10, np.float32)
    plain, relaxed = vc_adapter.MimiVC(), vc_adapter.MimiVC()
    plain.load('cpu', run_dir=str(run_dir), steps=2)
    relaxed.load('cpu', run_dir=str(run_dir), steps=2, relax=0.15, relax_k=2)
    np.random.seed(0)
    a = plain.convert_latent(src, ref)
    np.random.seed(0)
    b = relaxed.convert_latent(src, ref)
    assert calls == [(0.15, 2)] and np.allclose(b, a + 1.0)
    with pytest.raises(ValueError, match='relax'):
        vc_adapter.MimiVC().load('cpu', run_dir=str(run_dir), relax=1.5)


def test_adapter_source_negative_guidance(tmp_path, monkeypatch):
    """neg_src: default 0 (unchanged), negative values refused; with neg_src > 0 the source is encoded once more as
    the negative prompt and its windowed CAM++ vector is computed."""
    assert vc_adapter.DEFAULTS['neg_src'] == 0.0
    with pytest.raises(ValueError, match='neg_src'):
        vc_adapter.MimiVC().load('cpu', run_dir='x', neg_src=-1)
    a = tr.parse(['--train_dirs', 'x', '--val_dirs', 'x', '--out', str(tmp_path), '--d', '32', '--layers', '1',
                  '--heads', '2', '--content_dim', '8', '--device', 'cpu', '--sampler', 'prompt', '--pos', 'segment_random',
                  '--spk', '192'])
    m = tr.build(a)
    torch.save(dict(ema={k: v.float() for k, v in m.state_dict().items()}, step=3, args=vars(a)), tmp_path / 'ema_last.pt')
    np.savez(tmp_path / 'stats.npz', mean=np.zeros(512, np.float32), std=np.ones(512, np.float32))
    monkeypatch.setitem(sys.modules, 'mimi_io', types.SimpleNamespace(MimiIO=FakeMimiIO))
    cam = FakeCam('cpu')
    monkeypatch.setitem(sys.modules, 'spk_embed', types.SimpleNamespace(CamPlus=lambda device: cam))
    ref = np.zeros(24000 * 30, np.float32)
    out = {}
    for w in (0.0, 1.0):
        ad = vc_adapter.MimiVC()
        ad.load('cpu', run_dir=str(tmp_path), steps=2, neg_src=w, seed_from_pair=False)
        n0 = len(cam.calls)
        wav, sr = ad.convert(np.zeros(24000 * 2, np.float32), ref, 24000)
        out[w] = (wav, len(cam.calls) - n0)
    assert out[0.0][0].shape == out[1.0][0].shape == (48000,)
    assert out[1.0][1] == out[0.0][1] + 1                                  # one more CAM++ call: the 2 s source
