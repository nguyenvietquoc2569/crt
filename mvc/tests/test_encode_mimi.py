"""Tests of mvc/encode_mimi.py with a small causal stand-in for Mimi (no download): frame counts, batching, shard
planning, atomic shard files, resume, the batched-vs-single check and the corpus walk."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402

torch = pytest.importorskip('torch')


def _fake_semantic_quantizer():
    """Stand-in for Mimi's semantic RVQ: an input projection and one 2048 x 256 codebook."""
    from types import SimpleNamespace
    g = torch.Generator().manual_seed(1)
    cb = SimpleNamespace(embed_sum=torch.randn(2048, 256, generator=g), cluster_usage=torch.ones(2048), epsilon=1e-5)
    torch.manual_seed(1)
    return SimpleNamespace(input_proj=torch.nn.Conv1d(512, 256, 1, bias=False), layers=[SimpleNamespace(codebook=cb)])


class FakeMimi(torch.nn.Module):
    """Causal like Mimi: one frame per 1920 samples (last frame padded), frame t sees samples < 1920 (t + 1) only."""

    def __init__(self, leak=False, lookahead=False):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(512, em.HOP, generator=g) * 0.01)
        self.leak = leak
        self.lookahead = lookahead
        self.quantizer = FakeMimi.Q()
        self.quantizer.semantic_residual_vector_quantizer = _fake_semantic_quantizer()

    def encoder(self, x):
        B, _, L = x.shape
        T = -(-L // em.HOP)
        x = torch.nn.functional.pad(x, (0, T * em.HOP - L))
        h = torch.einsum('bth,dh->bdt', x.reshape(B, T, em.HOP), self.w)
        if self.leak:                                        # a non-causal stand-in: every frame sees the batch's end
            h = h + h[:, :, -1:]
        if self.lookahead:                                   # like Mimi's last frame: it sees one frame past the end,
            nxt = torch.nn.functional.pad(h[:, :, 1:] + 0.3, (0, 1))   # where a batch holds the response to zeros
            h = h + 0.5 * nxt
        return h

    def encoder_transformer(self, h, return_dict=False):
        return (torch.cumsum(h, 1) / torch.arange(1, h.shape[1] + 1)[None, :, None],)

    def downsample(self, h):
        return h

    class Q:
        @staticmethod
        def encode(lat):
            return (lat[:, :32].transpose(0, 1) * 100).round().long() % 2048

    quantizer = Q()


def test_frame_count_and_batches():
    assert em.n_frames(1920) == 1 and em.n_frames(1921) == 2 and em.n_frames(24000) == 13
    lens = [50000, 10000, 48000, 12000, 30000]
    bs = em.batches(lens, max_batch_samples=100000, max_batch=2)
    assert sorted(i for b in bs for i in b) == list(range(5)) and all(len(b) <= 2 for b in bs)
    for b in bs:
        assert len(b) * max(lens[i] for i in b) <= 100000 or len(b) == 1


def test_shard_plan_covers_in_order():
    frames = [100, 200, 300, 50, 400, 10]
    plan = em.shard_plan(list(range(6)), frames, 450)
    assert [i for s in plan for i in s] == list(range(6))
    assert all(sum(frames[i] for i in s) >= 450 for s in plan[:-1])


def test_encoder_is_padding_independent():
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(30000).astype(np.float32), rng.standard_normal(9000).astype(np.float32)
    (la, ca), (lb, cb) = enc([a, b])
    (la1, ca1), = enc([a])
    (lb1, cb1), = enc([b])
    assert la.shape == (em.n_frames(30000), 512) and lb.shape == (em.n_frames(9000), 512) and ca.shape[1] == 32
    assert np.allclose(lb, lb1, atol=1e-5) and np.array_equal(cb, cb1) and np.allclose(la, la1, atol=1e-5)


def make_corpus(root, n_spk=3, n_utt=4, seed=0):
    rng = np.random.default_rng(seed)
    audio = {}
    for s in range(n_spk):
        for u in range(n_utt):
            d = Path(root) / 'train-x' / f'{100 + s}' / '7'
            d.mkdir(parents=True, exist_ok=True)
            p = d / f'{100 + s}_7_000{u}_00000{u}.wav'
            p.write_bytes(b'')
            (d / f'{p.stem}.normalized.txt').write_text(f'utterance {s} {u}')
            audio[str(p)] = rng.standard_normal(int(24000 * (0.6 + 0.4 * u))).astype(np.float32)
    return audio


def test_encode_subset_writes_shards_and_resumes(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    logs = []
    n = em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=logs.append,
                         reader=lambda p: audio[p])
    files = sorted((tmp_path / 'out' / 'train-x').glob('shard_*.json'))
    assert len(files) == n >= 2 and any('batch check' in m for m in logs)
    seen = []
    for f in files:
        rows, lat, cod = em.load_shard(str(f)[:-5])
        assert lat.dtype == np.float16 and cod.dtype == np.int16 and lat.shape[0] == cod.shape[0] == sum(r['frames'] for r in rows)
        for r in rows:
            wav = audio[str(next(Path(tmp_path / 'raw').rglob(r['id'] + '.wav')))]
            assert r['frames'] == em.stored_frames(len(wav)) == em.n_frames(len(wav)) - em.TRIM
            assert r['text'].startswith('utterance')
            single, = enc([wav])
            assert np.allclose(lat[r['offset']:r['offset'] + r['frames']], single[0][:r['frames']], atol=2e-3)
            seen.append(r['id'])
    assert len(seen) == 12 and len(set(seen)) == 12
    stamp = files[0].stat().st_mtime_ns
    logs.clear()
    em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=logs.append,
                     reader=lambda p: audio[p])
    assert files[0].stat().st_mtime_ns == stamp and not any('shard' in m and 'utts' in m for m in logs)


def test_trailing_padding_dependence_is_trimmed_and_accepted(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi(lookahead=True))
    logs = []
    em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=logs.append,
                     reader=lambda p: audio[p])
    checks = [m for m in logs if 'batch check' in m]
    assert checks and any('first differing frame None' not in m for m in checks)     # the tail did differ


def test_batch_check_stops_a_padding_dependent_encoder(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi(leak=True))
    with pytest.raises(SystemExit, match='differs'):
        em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=lambda m: None,
                         reader=lambda p: audio[p], fallback=False)


def test_fallback_encodes_one_by_one_exactly(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi(leak=True))
    logs = []
    em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=logs.append,
                     reader=lambda p: audio[p])
    assert any('falling back' in m for m in logs)
    for f in sorted((tmp_path / 'out' / 'train-x').glob('shard_*.json')):
        rows, lat, cod = em.load_shard(str(f)[:-5])
        for r in rows:
            wav = audio[str(next(Path(tmp_path / 'raw').rglob(r['id'] + '.wav')))]
            single, = enc([wav])
            assert np.allclose(lat[r['offset']:r['offset'] + r['frames']], single[0][:r['frames']], atol=2e-3)


def test_missing_subset_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match='no wav files'):
        em.encode_subset(None, tmp_path, 'nothing', tmp_path / 'out')


def test_write_shard_is_atomic(tmp_path):
    base = tmp_path / 'shard_0000'
    em.write_shard(base, [{'id': 'a', 'offset': 0, 'frames': 2}], [np.ones((2, 512))], [np.ones((2, 32))])
    assert not list(tmp_path.glob('*.tmp')) and json.load(open(str(base) + '.json'))[0]['id'] == 'a'
