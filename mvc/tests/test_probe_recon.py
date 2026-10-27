"""Tests of mvc/probe_recon.py: reference and enrolment sets exclude the utterance and each other, rows pass the
journal's check_row rules, and a rerun resumes."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_recon as pr  # noqa: E402


def rows(n_spk=3, n_utt=8):
    return [dict(id=f'{s}_c_{u}', speaker=str(s), chapter='c', subset='dev-clean', text=f'text {s} {u}',
                 samples=24000 * 6, frames=73) for s in range(n_spk) for u in range(n_utt)]


def test_plan_sets_are_disjoint():
    items = pr.plan(rows(), 12, ref_s=15)
    assert len(items) == 12 and len({u['id'] for u, _, _ in items}) == 12
    for u, ref, enrol in items:
        ids_r, ids_e = {r['id'] for r in ref}, {r['id'] for r in enrol}
        assert u['id'] not in ids_r | ids_e and not ids_r & ids_e
        assert all(r['speaker'] == u['speaker'] for r in ref + enrol)
        assert sum(r['samples'] for r in ref) / 24000 >= 15 and len(enrol) <= 10


class Echo:
    def convert(self, src, ref, sr):
        return src.astype(np.float32), sr


def test_run_writes_valid_rows_and_resumes(tmp_path):
    items = pr.plan(rows(), 6, ref_s=10)
    reader = lambda p: np.zeros(24000, np.float32)  # noqa: E731
    p = pr.run(Echo(), items, '/raw', tmp_path, 'rec', reader=reader, ref_s=10, log=lambda m: None)
    rs = [json.loads(l) for l in open(p)]
    assert len(rs) == 6
    for r in rs:
        assert r['wav'] not in r['sim_ref_wavs'] and r['source_wav'] not in r['sim_ref_wavs'] and r['parallel_wav'] is None
        assert Path(r['wav']).exists() and r['wer_primary'] and r['ref_text'].startswith('text')
    pr.run(Echo(), items, '/raw', tmp_path, 'rec', reader=reader, ref_s=10, log=lambda m: None)
    assert len(open(p).readlines()) == 6


def test_plan_max_s_keeps_short_utterances_only():
    rs = rows()
    for r in rs:
        if r['id'].endswith(('_1', '_3', '_5')):
            r['samples'] = 24000 * 12
    items = pr.plan(rs, 12, ref_s=15, max_s=10)
    assert items and all(u['samples'] <= 24000 * 10 for u, _, _ in items)
    assert any(r['samples'] > 24000 * 10 for _, ref, _ in items for r in ref)        # references may be long
