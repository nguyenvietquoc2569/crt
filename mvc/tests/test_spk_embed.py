"""Tests of mvc/spk_embed.py with a stand-in network: fbank features at 16 kHz, per-utterance mean removal, one
embedding per shard row, skip when present."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchaudio')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import spk_embed as se  # noqa: E402
from test_encode_mimi import FakeMimi, make_corpus  # noqa: E402


class MeanNet(torch.nn.Module):
    """[1, T, 80] fbank -> [1, 192]: records its input; the output is the frame-mean projected."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(80, 192)
        self.seen = None

    def forward(self, f):
        self.seen = f
        return self.lin(f.mean(1))


def test_features_are_16k_fbank_with_zero_mean():
    net = MeanNet()
    cp = se.CamPlus('cpu', model=net)
    w = np.random.default_rng(0).standard_normal(24000 * 2).astype(np.float32) * 0.1
    e = cp(w)
    assert e.shape == (192,) and e.dtype == np.float32
    f = net.seen[0]
    assert f.shape[1] == 80 and 190 <= f.shape[0] <= 200                 # 2 s at 16 kHz, 10 ms hop
    assert torch.allclose(f.mean(0), torch.zeros(80), atol=1e-4)


def test_embed_shard_rows_and_skip(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=lambda m: None,
                     reader=lambda p: audio[p])
    cp = se.CamPlus('cpu', model=MeanNet())
    for js in sorted((tmp_path / 'out' / 'train-x').glob('shard_*.json')):
        base = str(js)[:-5]
        assert se.embed_shard(cp, base, tmp_path / 'raw', reader=audio.__getitem__, log=lambda m: None)
        rows, _, _ = em.load_shard(base)
        assert np.load(base + '.spk.npy').shape == (len(rows), 192)
        assert not se.embed_shard(cp, base, tmp_path / 'raw', reader=audio.__getitem__, log=lambda m: None)
