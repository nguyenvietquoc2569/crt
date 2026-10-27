"""Tests of mvc/dnsmos.py. The calibration and windowing run always; scoring needs the DNSMOS ONNX files in the folder
named by DNSMOS_DIR (skipped otherwise): clean harmonic speech-like audio must outscore the same audio buried in noise."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dnsmos  # noqa: E402

MODELS = os.environ.get('DNSMOS_DIR', '')
have = MODELS and (Path(MODELS) / 'sig_bak_ovr.onnx').exists() and (Path(MODELS) / 'model_v8.onnx').exists()


def test_calibration_matches_the_reference_polynomials():
    assert abs(dnsmos.P_OVR(3.0) - (-0.06766283 * 9 + 1.11546468 * 3 + 0.04602535)) < 1e-9
    assert dnsmos.summarize([dict(SIG=3.0, BAK=4.0, OVRL=2.0, P808=3.5), dict(error='x')])['n_error'] == 1


def voiced(s=3.0, seed=0):
    t = np.arange(int(16000 * s)) / 16000
    f = 140 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))
    ph = 2 * np.pi * np.cumsum(f) / 16000
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    return (sum(np.sin(k * ph) / k for k in range(1, 20)) * env * 0.05).astype(np.float32)


@pytest.mark.skipif(not have, reason='DNSMOS_DIR with the ONNX models not set')
def test_noise_lowers_the_scores(tmp_path):
    m = dnsmos.DNSMOS(MODELS)
    clean = m.score(voiced())
    noisy = m.score(voiced() + np.random.default_rng(0).standard_normal(48000).astype(np.float32) * 0.05)
    assert set(clean) == set(dnsmos.FIELDS) and clean['BAK'] > noisy['BAK'] and clean['OVRL'] > noisy['OVRL']
    import soundfile as sf
    sf.write(tmp_path / 'a.wav', voiced(), 24000)
    (tmp_path / 'rows.jsonl').write_text(json.dumps(dict(pair_id='p', wav=str(tmp_path / 'a.wav'))) + '\n')
    assert dnsmos.main(['--models', MODELS, '--rows', str(tmp_path / 'rows.jsonl'), '--out', str(tmp_path / 'o.jsonl')]) == 0
    assert json.loads((tmp_path / 'o.summary.json').read_text())['n'] == 1


def test_lufs_normalisation_removes_gain_differences():
    pytest.importorskip('pyloudnorm')
    x = voiced()
    a, b = dnsmos.normalize_lufs(x, -26.0), dnsmos.normalize_lufs(0.3 * x, -26.0)
    assert np.allclose(a, b, atol=1e-5)
    import pyloudnorm
    assert abs(pyloudnorm.Meter(16000).integrated_loudness(a.astype(np.float64)) + 26.0) < 0.05
