"""Tests of mvc/probe_speaker_info.py: a stream that encodes the speaker is identified, a random one is at chance, the
EER behaves, and the feature builder reads shards and perturbed codes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import probe_speaker_info as ps  # noqa: E402


def test_evaluate_separates_speaker_and_random_features():
    rng = np.random.default_rng(0)
    spk = [f's{i % 10}' for i in range(400)]
    centres = rng.standard_normal((10, 64))
    X = np.stack([centres[i % 10] + 0.3 * rng.standard_normal(64) for i in range(400)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    good = ps.evaluate(X, spk)
    R = rng.standard_normal((400, 64))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    bad = ps.evaluate(R, spk)
    assert good['acc'] > 0.95 and good['eer'] < 0.05 and good['chance'] == 0.1
    assert bad['acc'] < 0.25 and 0.35 < bad['eer'] < 0.65


def test_eer_extremes():
    assert ps.eer(np.array([1.0, 0.9]), np.array([0.1, 0.0])) == 0.0
    assert ps.eer(np.array([0.0, 0.1]), np.array([1.0, 0.9])) == pytest.approx(1.0, abs=0.5)


def test_features_from_shards(tmp_path):
    rng = np.random.default_rng(1)
    rows, lats, cods, off = [], [], [], 0
    for s in range(4):
        for u in range(6):
            T = 30
            cod = rng.integers(0, 2048, (T, 32))
            cod[:, 0] = s * 100 + rng.integers(0, 5, T)            # CB0 encodes the speaker strongly
            rows.append(dict(id=f'{s}_{u}', speaker=str(s), chapter=str(u % 2), text=f'book {s} words {u}', offset=off,
                             frames=T))
            lats.append(rng.standard_normal((T, 512)) + s)
            cods.append(cod)
            off += T
    base = tmp_path / 'shard_0000'
    em.write_shard(base, rows, lats, cods)
    pc = np.concatenate(cods).astype(np.int16)
    pc[:, 0] = rng.integers(0, 2048, len(pc))                      # perturbed CB0: speaker gone
    np.save(str(base) + '.pcodes0.npy', pc)
    F, spk, chap = ps.features([str(tmp_path)], K=1)
    assert set(F) == {'cb0', 'cb1', 'lat', 'pcb0_0', 'text'} and len(spk) == len(chap) == 24
    res = {k: ps.evaluate(v, spk, chapters=chap) for k, v in F.items()}
    assert res['cb0']['acc'] == 1.0 and res['pcb0_0']['acc'] < 0.6 and res['cb0']['n_test'] == 12


def test_chapter_split_keeps_chapters_apart_and_drops_single_chapter_speakers():
    rng = np.random.default_rng(0)
    spk = ['a'] * 6 + ['b'] * 6 + ['c'] * 3
    chap = ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'b1', 'b1', 'b1', 'b2', 'b2', 'b2', 'c1', 'c1', 'c1']
    X = rng.standard_normal((15, 8))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = ps.evaluate(X, spk, chapters=chap)
    assert r['n_speakers'] == 2 and r['chance'] == 0.5 and r['n_test'] == 7
