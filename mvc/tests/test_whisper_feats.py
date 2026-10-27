"""Tests of mvc/whisper_feats.py with a stand-in encoder: 4:1 pooling onto Mimi frames, the 30 s limit, and the shard
pass (perturbed draws with encode_perturbed's seeds, frame-aligned, skipped when present)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('librosa')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import encode_perturbed as ep  # noqa: E402
import whisper_feats as wf  # noqa: E402
from test_encode_mimi import FakeMimi, make_corpus  # noqa: E402


class FakeFE:
    """Whisper's feature extractor stand-in: 'mel' = the 16 kHz samples padded to 30 s, framed at 10 ms."""

    def __call__(self, w, sampling_rate, return_tensors):
        x = np.zeros(30 * 16000, np.float32)
        x[:len(w)] = w
        from types import SimpleNamespace
        return SimpleNamespace(input_features=torch.as_tensor(x.reshape(3000, 160).T[None]))


class FakeEnc(torch.nn.Module):
    """[1, 160, 3000] -> [1, 1500, 768]: frame j = (j, mean of its 20 ms) so pooling can be checked."""

    def forward(self, mel):
        x = mel[0].T.reshape(1500, 320).mean(1)
        out = torch.zeros(1, 1500, 768)
        out[0, :, 0] = torch.arange(1500).float()
        out[0, :, 1] = x
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=out)


def test_pooling_onto_mimi_frames():
    f = wf.WhisperFeat('cpu', model=FakeEnc(), fe=FakeFE())
    w = np.zeros(int(24000 * 1.0), np.float32)
    F = f(w)
    assert F.shape == (em.n_frames(len(w)), 768)
    assert np.allclose(F[:, 0], np.arange(len(F)) * 4 + 1.5)          # mean of Whisper frames 4t..4t+3
    with pytest.raises(ValueError, match='30 s'):
        f(np.zeros(24000 * 31, np.float32))


def flip(w, sr, rng):
    return (-w + 0.01 * rng.standard_normal(len(w))).astype(np.float32)


def test_shard_pass(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=lambda m: None,
                     reader=lambda p: audio[p])
    f = wf.WhisperFeat('cpu', model=FakeEnc(), fe=FakeFE())
    js = sorted((tmp_path / 'out' / 'train-x').glob('shard_*.json'))[0]
    base = str(js)[:-5]
    assert wf.feats_shard(f, base, tmp_path / 'raw', 1, 7, reader=audio.__getitem__, perturb_fn=flip, log=lambda m: None)
    rows, _, cod = em.load_shard(base)
    F = np.load(base + '.pwhisper0.npy')
    assert F.shape == (len(cod), 768) and F.dtype == np.float16
    r = rows[0]
    w = flip(audio[ep.wav_path(tmp_path / 'raw', r)], em.SR, np.random.default_rng(ep.utt_seed(7, r['id'], 0)))
    assert np.allclose(F[r['offset']:r['offset'] + r['frames']], f(w)[:r['frames']], atol=1e-2)
    assert not wf.feats_shard(f, base, tmp_path / 'raw', 1, 7, reader=audio.__getitem__, perturb_fn=flip, log=lambda m: None)
