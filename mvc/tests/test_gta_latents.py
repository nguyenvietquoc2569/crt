"""Tests of mvc/gta_latents.py: the plan (every k-th utterance, exclusions, references from the same speaker never
holding the source), frame-aligned npz chunks, and resuming."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gta_latents as gl  # noqa: E402


def rows(n_spk=3, n_utt=8):
    return [dict(id=f'{s}_c_{u}', speaker=str(s), chapter='c', subset='train-clean-100', text='', samples=24000 * 5,
                 frames=60 + u) for s in range(n_spk) for u in range(n_utt)]


def test_plan_respects_every_and_exclusions():
    allp = gl.plan(rows(), 1, 12)
    some = gl.plan(rows(), 3, 12, exclude={'0_c_0', '1_c_1'})
    assert len(some) < len(allp) and not {'0_c_0', '1_c_1'} & {u['id'] for u, _ in some}
    for u, ref in allp:
        assert ref and all(r['speaker'] == u['speaker'] and r['id'] != u['id'] for r in ref)


class Fake:
    def __init__(self):
        self.calls = 0

    def convert_latent(self, src, ref):
        self.calls += 1
        return np.full((len(src) // 1000, 512), self.calls, np.float32)     # frames encoded by the source length


def test_run_writes_aligned_chunks_and_resumes(tmp_path):
    items = gl.plan(rows(), 2, 12)
    reader = lambda p: np.zeros(1000 * next(r['frames'] for r in rows() if r['id'] == Path(p).stem), np.float32)  # noqa: E731
    fake = Fake()
    n = gl.run(fake, items, '/raw', tmp_path, chunk=3, reader=reader, log=lambda m: None)
    assert n == len(items) and len(list(tmp_path.glob('gta_*.npz'))) == -(-len(items) // 3)
    z = np.load(sorted(tmp_path.glob('gta_*.npz'))[0])
    assert list(z['frames']) == [u['frames'] for u, _ in items[:3]]
    assert float(z['lat'][z['offsets'][1], 0]) == 2.0                       # second utterance starts at its offset
    assert gl.run(fake, items, '/raw', tmp_path, chunk=3, reader=reader, log=lambda m: None) == 0
    bad = lambda p: np.zeros(1000, np.float32)  # noqa: E731
    with pytest.raises(SystemExit, match='frames'):
        gl.run(Fake(), gl.plan(rows(), 1, 12)[:1], '/raw', tmp_path / 'x', reader=bad, log=lambda m: None)


class Extra:
    """A converter whose output has `extra` frames more than the shard stores (the shards drop the last TRIM)."""
    def __init__(self, extra):
        self.extra = extra

    def convert_latent(self, src, ref):
        n = len(src) // 1000 + self.extra
        return np.arange(n * 512, dtype=np.float32).reshape(n, 512)


def test_trim_frames_are_cut_and_other_mismatches_refused(tmp_path):
    import encode_mimi as em
    items = gl.plan(rows(), 1, 12)[:2]
    reader = lambda p: np.zeros(1000 * next(r['frames'] for r in rows() if r['id'] == Path(p).stem), np.float32)  # noqa: E731
    assert gl.run(Extra(em.TRIM), items, '/raw', tmp_path / 'ok', reader=reader, log=lambda m: None) == 2
    z = np.load(sorted((tmp_path / 'ok').glob('gta_*.npz'))[0])
    assert list(z['frames']) == [u['frames'] for u, _ in items] and z['lat'].shape[0] == sum(z['frames'])
    for extra in (em.TRIM + 1, -1):
        with pytest.raises(SystemExit, match='frames'):
            gl.run(Extra(extra), items, '/raw', tmp_path / f'bad{extra}', reader=reader, log=lambda m: None)
