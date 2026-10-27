"""Information perturbation of speech (NANSY, Choi et al. 2021), to strip speaker cues from content features.

perturb(wav, sr, rng) applies, as NANSY's g(x) for linguistic features:
  - formant shift by a ratio fs ~ U(fs_min, fs_max), inverted (1/fs) with probability 1/2,
  - pitch randomisation: the median F0 moved by ps ~ U(ps_min, ps_max) (inverted with p 1/2) and the pitch range
    scaled by pr ~ U(1, pr_max) (inverted with p 1/2),
both through Praat's "Change gender" (PSOLA), then
  - a random parametric equaliser: a low shelf, a high shelf and n_peak peaking filters with gains ~ U(-g, g) dB.
The output has the input's length and peak at most the input's peak (never clipped). The lower bounds fs_min and
ps_min keep every draw away from identity: with NANSY's U(1, r) about a quarter of the formant draws lie within 10 %
of 1, so the perturbed content would still carry the speaker's own cues (design review of 25 Oct 2026).
"""
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter

PRESET = dict(fs_min=1.12, fs_max=1.4, ps_min=1.25, ps_max=2.0, pr_max=1.5, eq_gain_db=12.0, n_peak=8)


def _ratio(rng, hi, lo=1.0):
    r = rng.uniform(lo, hi)
    return r if rng.random() < 0.5 else 1.0 / r


def biquad(kind, f0, gain_db, q, sr):
    """RBJ audio-EQ-cookbook biquad coefficients (b, a) for 'low', 'high' (shelves) or 'peak'."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if kind == 'peak':
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ('low', 'high'):
        s = 1 if kind == 'low' else -1
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - s * (A - 1) * cw + sa), s * 2 * A * ((A - 1) - s * (A + 1) * cw), A * ((A + 1) - s * (A - 1) * cw - sa)]
        a = [(A + 1) + s * (A - 1) * cw + sa, -s * 2 * ((A - 1) + s * (A + 1) * cw), (A + 1) + s * (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    b, a = np.array(b, float), np.array(a, float)
    return b / a[0], a / a[0]


def random_eq(wav, sr, rng, gain_db=12.0, n_peak=8):
    y = wav.astype(np.float64)
    nyq = sr / 2
    specs = [('low', 60.0, 0.7), ('high', min(10000.0, 0.8 * nyq), 0.7)]
    centres = np.geomspace(150.0, min(8000.0, 0.7 * nyq), n_peak)
    specs += [('peak', float(f * rng.uniform(0.85, 1.15)), float(rng.uniform(0.5, 4.0))) for f in centres]
    for kind, f0, q in specs:
        b, a = biquad(kind, f0, rng.uniform(-gain_db, gain_db), q, sr)
        y = lfilter(b, a, y)
    return y


def change_gender(wav, sr, formant_ratio, pitch_ratio, range_ratio, fmin=75.0, fmax=600.0):
    """Praat "Change gender": formants scaled by formant_ratio, median F0 by pitch_ratio, F0 range by range_ratio."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array['frequency']
    voiced = f0[f0 > 0]
    median = float(np.median(voiced)) if len(voiced) >= 5 else 0.0
    new_median = median * pitch_ratio if median > 0 and pitch_ratio != 1.0 else 0.0
    out = call(snd, 'Change gender', fmin, fmax, formant_ratio, new_median, range_ratio, 1.0)
    y = np.asarray(out.values[0], dtype=np.float64)
    return y[:len(wav)] if len(y) >= len(wav) else np.pad(y, (0, len(wav) - len(y)))


def guarded(fn, args, timeout):
    """fn(*args) -> float64 array, computed in a forked child that is killed after `timeout` seconds (TimeoutError).
    Praat's 'Change gender' can loop without end inside C++ on rare utterances, where no Python signal reaches it
    (jobs 5639 and 5640, 25 Oct 2026: one worker at 100 % CPU for over 2 h). An error in the child is raised again
    here, as parselmouth.PraatError when it was one, else RuntimeError."""
    import select
    import signal
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                            # child: compute, send, exit without cleanup handlers
        os.close(r)
        try:
            y = np.ascontiguousarray(fn(*args), dtype=np.float64)
            msg = b'A' + y.tobytes()
        except BaseException as e:                         # noqa: BLE001 -- reported to the parent
            msg = b'E' + f'{type(e).__name__}\n{e}'.encode()
        try:
            view = memoryview(msg)
            while view:
                view = view[os.write(w, view):]
        finally:
            os._exit(0)
    os.close(w)
    chunks, deadline = [], time.monotonic() + timeout
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([r], [], [], left)[0]:
                os.kill(pid, signal.SIGKILL)
                raise TimeoutError(f'{getattr(fn, "__name__", fn)} took more than {timeout:.0f} s')
            b = os.read(r, 1 << 20)
            if not b:
                break
            chunks.append(b)
    finally:
        os.close(r)
        os.waitpid(pid, 0)
    msg = b''.join(chunks)
    if msg[:1] == b'A':
        return np.frombuffer(msg[1:], dtype=np.float64).copy()
    name, _, text = msg[1:].decode('utf-8', 'replace').partition('\n')
    if name == 'PraatError':
        import parselmouth
        raise parselmouth.PraatError(text)
    raise RuntimeError(f'{name}: {text}' if msg else 'the child exited without a result')


def safe_change_gender(wav, sr, formant, pitch, rng_ratio, timeout=None):
    """change_gender with fallbacks for the inputs Praat refuses (e.g. 'rounding up the real value --undefined--' on
    utterances whose pitch analysis is degenerate) or never finishes (guarded(), default limit 20 s + 4 s per second
    of audio): first the formant shift alone (pitch kept), then no PSOLA at all.
    Returns (signal, stage, timeouts) with stage 'full', 'formant_only' or 'none' and the stages that timed out."""
    import parselmouth
    limit = timeout if timeout is not None else 20.0 + 4.0 * len(wav) / sr
    timeouts = []
    for stage, args in (('full', (formant, pitch, rng_ratio)), ('formant_only', (formant, 1.0, 1.0))):
        try:
            return guarded(change_gender, (wav, sr) + args, limit), stage, timeouts
        except parselmouth.PraatError:
            continue
        except TimeoutError:
            timeouts.append(stage)
            print(f'perturb: Praat {stage} timed out after {limit:.0f} s on {len(wav) / sr:.1f} s of audio '
                  f'(formant {formant:.3f}, pitch {pitch:.3f}, range {rng_ratio:.3f})', file=sys.stderr, flush=True)
    return np.asarray(wav, dtype=np.float64), 'none', timeouts


def perturb(wav, sr, rng, fs_min=1.12, fs_max=1.4, ps_min=1.25, ps_max=2.0, pr_max=1.5, eq_gain_db=12.0, n_peak=8,
            return_params=False):
    params = dict(formant=_ratio(rng, fs_max, fs_min), pitch=_ratio(rng, ps_max, ps_min), range=_ratio(rng, pr_max))
    y, params['stage'], params['timeouts'] = safe_change_gender(wav, sr, params['formant'], params['pitch'],
                                                                 params['range'])
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y)
    y = random_eq(y, sr, rng, eq_gain_db, n_peak)
    peak_in, peak = float(np.abs(wav).max()) or 1.0, float(np.abs(y).max())
    if peak > 0:
        y = y * (min(peak_in, 0.99) / peak)
    y = y.astype(np.float32)
    return (y, params) if return_params else y
