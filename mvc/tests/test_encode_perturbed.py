"""Tests of mvc/encode_perturbed.py with the causal stand-in for Mimi: perturbed codes are frame-aligned with the clean
shard, draws are seeded per utterance and k, existing outputs are skipped, and a frame shortfall stops the job."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import encode_perturbed as ep  # noqa: E402
from test_encode_mimi import FakeMimi, make_corpus  # noqa: E402

torch = pytest.importorskip('torch')


def flip(w, sr, rng):
    """A perturbation stand-in: sign flip plus a little seeded noise (keeps the length)."""
    return (-w + 0.01 * rng.standard_normal(len(w))).astype(np.float32)


def setup(tmp_path):
    audio = make_corpus(tmp_path / 'raw')
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    em.encode_subset(enc, tmp_path / 'raw', 'train-x', tmp_path / 'out', shard_frames=40, log=lambda m: None,
                     reader=lambda p: audio[p])
    return audio, enc, sorted((tmp_path / 'out' / 'train-x').glob('shard_*.json'))


def test_perturbed_codes_are_aligned_and_seeded(tmp_path):
    audio, enc, shards = setup(tmp_path)
    reader = audio.__getitem__
    logs = []
    for js in shards:
        assert ep.encode_shard(enc, str(js)[:-5], tmp_path / 'raw', 2, 7, reader=reader, perturb_fn=flip, log=logs.append)
    assert len(logs) == len(shards) and all('CB0 agreement' in m for m in logs)
    for js in shards:
        base = str(js)[:-5]
        rows, lat, cod = em.load_shard(base)
        p0, p1 = np.load(base + '.pcodes0.npy'), np.load(base + '.pcodes1.npy')
        assert p0.shape == cod.shape == p1.shape and p0.dtype == np.int16
        for r in rows:
            w = audio[ep.wav_path(tmp_path / 'raw', r)]
            exp = enc([flip(w, em.SR, np.random.default_rng(ep.utt_seed(7, r['id'], 0)))])[0][1][:r['frames']]
            assert np.array_equal(p0[r['offset']:r['offset'] + r['frames']], exp)
        assert not np.array_equal(p0, p1)                       # independent draws per k
        si, sd = np.load(base + '.psoft0.idx.npy'), np.load(base + '.psoft0.d2.npy')
        assert si.shape == sd.shape == (len(cod), 8) and si.dtype == np.int16 and sd.dtype == np.float16
        assert np.all(np.diff(sd.astype(np.float32), axis=1) >= -1e-2)          # nearest first
    assert ep.utt_seed(7, 'a', 0) == ep.utt_seed(7, 'a', 0) != ep.utt_seed(7, 'a', 1)


def test_existing_outputs_are_skipped(tmp_path):
    audio, enc, shards = setup(tmp_path)
    base = str(shards[0])[:-5]
    assert ep.encode_shard(enc, base, tmp_path / 'raw', 1, 0, reader=audio.__getitem__, perturb_fn=flip, log=lambda m: None)
    assert not ep.encode_shard(enc, base, tmp_path / 'raw', 1, 0, reader=audio.__getitem__, perturb_fn=flip, log=lambda m: None)


def test_shortened_audio_is_refused(tmp_path):
    audio, enc, shards = setup(tmp_path)
    cut = lambda w, sr, rng: w[: len(w) // 3]   # noqa: E731
    with pytest.raises(SystemExit, match='frames'):
        ep.encode_shard(enc, str(shards[0])[:-5], tmp_path / 'raw', 1, 0, reader=audio.__getitem__, perturb_fn=cut,
                        log=lambda m: None)
