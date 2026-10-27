"""Tests of mvc/perturb.py on a synthetic voiced signal: F0 and formant shifts move as asked, EQ is stable and
roughly unit-gain at 0 dB, length and peak are preserved, and draws are reproducible."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perturb as pt  # noqa: E402

pytest.importorskip('parselmouth')
SR = 24000


def vowel(f0=120.0, sec=1.2, formants=(700, 1200, 2500)):
    """A glottal pulse train through three resonators: a crude /a/."""
    n = int(SR * sec)
    x = np.zeros(n)
    x[::int(SR / f0)] = 1.0
    for f in formants:
        b, a = pt.biquad('peak', f, 18.0, 6.0, SR)
        x = pt.lfilter(b, a, x)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def median_f0(w):
    import parselmouth
    p = parselmouth.Sound(w.astype(np.float64), SR).to_pitch(0.01, 75, 600).selected_array['frequency']
    return float(np.median(p[p > 0]))


def first_formant(w, lo=400.0, hi=1100.0):
    """Peak of the smoothed (200 Hz moving average) magnitude spectrum between lo and hi Hz."""
    s = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    f = np.fft.rfftfreq(len(w), 1 / SR)
    k = int(200 / (f[1] - f[0]))
    sm = np.convolve(s, np.ones(k) / k, mode='same')
    band = (f >= lo) & (f <= hi)
    return float(f[band][np.argmax(sm[band])])


def test_change_gender_moves_pitch_and_formants():
    w = vowel()
    up = pt.change_gender(w, SR, 1.0, 1.5, 1.0)
    assert len(up) == len(w)
    assert median_f0(up) == pytest.approx(median_f0(w) * 1.5, rel=0.08)
    hi = pt.change_gender(w, SR, 1.3, 1.0, 1.0)
    assert first_formant(w) == pytest.approx(700, abs=80)
    assert first_formant(hi) > first_formant(w) * 1.15


def test_biquads_are_stable_and_flat_at_zero_gain():
    for kind in ('low', 'high', 'peak'):
        b, a = pt.biquad(kind, 1000.0, 0.0, 0.7, SR)
        assert np.allclose(b, a, atol=1e-9)
        b, a = pt.biquad(kind, 1000.0, 12.0, 0.7, SR)
        assert np.all(np.abs(np.roots(a)) < 1)


def test_perturb_keeps_length_and_peak_and_is_reproducible():
    w = vowel()
    y1, p1 = pt.perturb(w, SR, np.random.default_rng(3), return_params=True)
    y2, p2 = pt.perturb(w, SR, np.random.default_rng(3), return_params=True)
    assert y1.shape == w.shape and y1.dtype == np.float32 and p1 == p2 and p1['stage'] == 'full'
    assert np.corrcoef(y1, y2)[0, 1] > 0.95                # Praat's PSOLA is not bit-reproducible
    assert np.abs(y1).max() <= np.abs(w).max() + 1e-6 and np.all(np.isfinite(y1))
    for k, lo, hi in (('formant', 1.12, 1.4), ('pitch', 1.25, 2.0), ('range', 1.0, 1.5)):  # noqa: B007
        r = max(p1[k], 1 / p1[k])
        assert lo - 1e-9 <= r <= hi + 1e-9
    draws = [pt._ratio(np.random.default_rng(i), 1.4, 1.12) for i in range(500)]
    assert min(max(d, 1 / d) for d in draws) >= 1.12 and 0.35 < np.mean([d > 1 for d in draws]) < 0.65
    y3, p3 = pt.perturb(w, SR, np.random.default_rng(4), return_params=True)
    assert p3 != p1


def test_perturb_survives_unvoiced_input():
    noise = (0.1 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    y = pt.perturb(noise, SR, np.random.default_rng(0))
    assert y.shape == noise.shape and np.all(np.isfinite(y))


def test_praat_refusals_fall_back(monkeypatch, tmp_path):
    import parselmouth
    w = vowel()
    log = tmp_path / 'calls.txt'                           # change_gender runs in a forked child: count through a file

    def flaky(wav, sr, f, p, r, fmin=75.0, fmax=600.0):
        with open(log, 'a') as fh:
            fh.write(f'{p}\n')
        if p != 1.0:
            raise parselmouth.PraatError('When rounding up the real value --undefined--')
        return np.asarray(wav, dtype=np.float64)
    monkeypatch.setattr(pt, 'change_gender', flaky)
    y, prm = pt.perturb(w, SR, np.random.default_rng(0), return_params=True)
    assert prm['stage'] == 'formant_only' and prm['timeouts'] == [] and len(log.read_text().split()) == 2
    assert y.shape == w.shape

    def broken(*a, **k):
        raise parselmouth.PraatError('nope')
    monkeypatch.setattr(pt, 'change_gender', broken)
    y, prm = pt.perturb(w, SR, np.random.default_rng(0), return_params=True)
    assert prm['stage'] == 'none' and np.all(np.isfinite(y)) and y.shape == w.shape


def test_guarded_returns_raises_and_kills_a_hung_call():
    import time
    import parselmouth
    x = np.arange(300000, dtype=np.float64)                 # larger than a pipe buffer
    assert np.array_equal(pt.guarded(lambda a: a * 2, (x,), 30), x * 2)

    def praat_error():
        raise parselmouth.PraatError('refused')

    def bug():
        raise ValueError('a bug')
    with pytest.raises(parselmouth.PraatError, match='refused'):
        pt.guarded(praat_error, (), 30)
    with pytest.raises(RuntimeError, match='ValueError: a bug'):
        pt.guarded(bug, (), 30)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        pt.guarded(time.sleep, (60,), 1.0)
    assert time.monotonic() - t0 < 10


def test_a_hung_praat_call_falls_back_to_the_next_stage(monkeypatch, capsys):
    import time
    w = vowel()

    def hangs_with_pitch(wav, sr, f, p, r, fmin=75.0, fmax=600.0):
        if p != 1.0:
            time.sleep(60)
        return np.asarray(wav, dtype=np.float64)
    monkeypatch.setattr(pt, 'change_gender', hangs_with_pitch)
    y, stage, timeouts = pt.safe_change_gender(w, SR, 1.2, 1.5, 1.2, timeout=1.0)
    assert stage == 'formant_only' and timeouts == ['full'] and y.shape == w.shape
    assert 'timed out' in capsys.readouterr().err

    def always_hangs(*a, **k):
        time.sleep(60)
    monkeypatch.setattr(pt, 'change_gender', always_hangs)
    y, stage, timeouts = pt.safe_change_gender(w, SR, 1.2, 1.5, 1.2, timeout=0.5)
    assert stage == 'none' and timeouts == ['full', 'formant_only'] and np.allclose(y, w)
