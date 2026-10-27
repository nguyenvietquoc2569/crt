"""Tests of mvc/acoustic_guard.py: a harmonic tone is periodic and voiced, white noise is not, and an output equal to its
source gives the identity values."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('parselmouth')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acoustic_guard as ag  # noqa: E402

SR = 16000


def tone(f0=150.0, s=1.5, vibrato=0.0):
    t = np.arange(int(SR * s)) / SR
    f = f0 * (1 + vibrato * np.sin(2 * np.pi * 3 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return sum(np.sin(k * ph) / k for k in range(1, 8)) * 0.1


def test_tone_is_periodic_and_noise_is_not():
    f, h = ag.f0_hnr(tone())
    assert ag.mean_hnr(h) > 20 and np.mean(f > 0) > 0.9
    noise = np.random.default_rng(0).standard_normal(int(SR * 1.5)) * 0.1
    f2, h2 = ag.f0_hnr(noise)
    assert ag.mean_hnr(h2) < 3 or np.mean(f2 > 0) < 0.2


def test_identity_and_flattened_pitch():
    x = tone(vibrato=0.05)
    g = ag.compare(x, x)
    assert g['voiced_kept'] > 0.99 and g['f0_pcc'] > 0.99 and abs(g['st_sd_ratio'] - 1) < 1e-6 and abs(g['rms_db_diff']) < 1e-9
    flat = ag.compare(tone(vibrato=0.0), x)
    assert flat['st_sd_ratio'] < 0.3                     # a monotone output is caught
    louder = ag.compare(2 * x, x)
    assert abs(louder['rms_db_diff'] - 20 * np.log10(2)) < 1e-6


def test_cli_writes_rows_and_summary(tmp_path):
    import soundfile as sf
    a, b = tmp_path / 'a.wav', tmp_path / 'b.wav'
    sf.write(a, tone(vibrato=0.05), SR)
    sf.write(b, tone(vibrato=0.05), SR)
    rows = tmp_path / 'rows.jsonl'
    rows.write_text('\n'.join(json.dumps(dict(pair_id=f'p{i}', system='s', wav=str(a), source_wav=str(b))) for i in range(3))
                    + '\n' + json.dumps(dict(pair_id='bad', system='s', wav=str(tmp_path / 'missing.wav'), source_wav=str(b))) + '\n')
    out = tmp_path / 'g' / 'guard.jsonl'
    assert ag.main(['--rows', str(rows), '--out', str(out), '--workers', '2']) == 0
    res = [json.loads(l) for l in open(out)]
    assert len(res) == 4 and 'error' in res[-1]
    s = json.loads((tmp_path / 'g' / 'guard.summary.json').read_text())
    assert s['n'] == 4 and s['n_error'] == 1 and s['voiced_kept'] > 0.99
