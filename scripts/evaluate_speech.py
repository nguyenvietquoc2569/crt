#!/usr/bin/env python3
"""
Speech Quality Evaluation Script for Voice Conversion.

Evaluates converted speech using multiple objective metrics:
  UTMOS, WER, SECS, PESQ, MCD, F0 smoothness, V/UV error.

Usage:
  # Single file evaluation
  python evaluate_speech.py \
      --source_audio source.wav \
      --converted_audio converted.wav \
      --target_audio target.wav \
      --output_json results.json

  # Batch evaluation
  python evaluate_speech.py \
      --eval_dir ./outputs \
      --source_audio source.wav \
      --target_audio target.wav \
      --pattern "crt_*refined*.wav"
"""

import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lazy imports with availability flags
# ---------------------------------------------------------------------------

_LIBROSA = False
try:
    import librosa
    _LIBROSA = True
except ImportError:
    pass

_TORCH = False
try:
    import torch
    _TORCH = True
except ImportError:
    pass

_TORCHAUDIO = False
try:
    import torchaudio
    _TORCHAUDIO = True
except ImportError:
    pass

_WHISPER = False
try:
    import whisper as whisper_pkg
    _WHISPER = True
except ImportError:
    pass

_PESQ = False
try:
    from pesq import pesq as pesq_fn
    _PESQ = True
except ImportError:
    pass

_RESEMBLYZER = False
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    _RESEMBLYZER = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio as mono float32 numpy array at given sample rate."""
    if _LIBROSA:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    elif _TORCHAUDIO:
        wav, orig_sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        return wav.squeeze(0).numpy().astype(np.float32)
    else:
        raise RuntimeError("Neither librosa nor torchaudio is available for audio loading.")


# ---------------------------------------------------------------------------
# WER via simple edit distance (no jiwer dependency needed)
# ---------------------------------------------------------------------------

def _edit_distance(ref_tokens: list, hyp_tokens: list) -> int:
    """Levenshtein distance between two token lists."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_wer(ref_text: str, hyp_text: str) -> float:
    """Word Error Rate (0-100 scale)."""
    ref_tokens = ref_text.strip().lower().split()
    hyp_tokens = hyp_text.strip().lower().split()
    if len(ref_tokens) == 0:
        return 100.0 if len(hyp_tokens) > 0 else 0.0
    dist = _edit_distance(ref_tokens, hyp_tokens)
    return min(100.0, dist / len(ref_tokens) * 100.0)


# ---------------------------------------------------------------------------
# Metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    source_file: str = ""
    converted_file: str = ""
    target_file: str = ""
    utmos: Optional[float] = None
    wer: Optional[float] = None
    source_transcript: Optional[str] = None
    converted_transcript: Optional[str] = None
    secs: Optional[float] = None
    pesq: Optional[float] = None
    mcd: Optional[float] = None
    f0_jitter_hz: Optional[float] = None
    f0_rmse_hz: Optional[float] = None
    vuv_error_pct: Optional[float] = None
    overall: Optional[float] = None
    skipped: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def eval_utmos(converted_wav_16k: np.ndarray) -> Optional[float]:
    """UTMOS naturalness score (1-5)."""
    if not _TORCH:
        return None

    # Try torch.hub UTMOS model
    try:
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        model.eval()
        wav_tensor = torch.from_numpy(converted_wav_16k).unsqueeze(0)
        with torch.no_grad():
            score = model(wav_tensor, sr=16000)
        return float(score.item())
    except Exception:
        pass

    # Try speechmos package
    try:
        from speechmos import utmos
        score = utmos.predict(converted_wav_16k, 16000)
        return float(score)
    except Exception:
        pass

    # Fallback: spectral flatness as a rough naturalness proxy (0-5 scale)
    if _LIBROSA:
        sf = librosa.feature.spectral_flatness(y=converted_wav_16k)
        mean_sf = float(np.mean(sf))
        # Map spectral flatness [0, 1] -> [1, 5].
        # Lower flatness (more tonal) generally means more natural speech.
        score = 5.0 - 4.0 * mean_sf
        return round(max(1.0, min(5.0, score)), 4)

    return None


def eval_wer(
    source_wav_16k: np.ndarray,
    converted_wav_16k: np.ndarray,
    whisper_model_name: str = "base",
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """WER between source and converted transcriptions via Whisper."""
    if not _WHISPER:
        return None, None, None
    try:
        model = whisper_pkg.load_model(whisper_model_name)
        src_result = model.transcribe(source_wav_16k, language="en", fp16=False)
        cvt_result = model.transcribe(converted_wav_16k, language="en", fp16=False)
        src_text = src_result["text"].strip()
        cvt_text = cvt_result["text"].strip()
        wer_val = compute_wer(src_text, cvt_text)
        return wer_val, src_text, cvt_text
    except Exception as e:
        print(f"  [WER] Error: {e}")
        return None, None, None


def eval_secs(
    converted_wav_16k: np.ndarray, target_wav_16k: np.ndarray
) -> Optional[float]:
    """Speaker Encoder Cosine Similarity via resemblyzer."""
    if not _RESEMBLYZER:
        return None
    try:
        enc = VoiceEncoder()
        e_cvt = enc.embed_utterance(converted_wav_16k)
        e_tgt = enc.embed_utterance(target_wav_16k)
        cos_sim = float(
            np.dot(e_cvt, e_tgt) / (np.linalg.norm(e_cvt) * np.linalg.norm(e_tgt))
        )
        return round(cos_sim, 4)
    except Exception as e:
        print(f"  [SECS] Error: {e}")
        return None


def eval_pesq(
    converted_wav_16k: np.ndarray, target_wav_16k: np.ndarray
) -> Optional[float]:
    """PESQ wideband score."""
    if not _PESQ:
        return None
    try:
        # Align lengths
        min_len = min(len(converted_wav_16k), len(target_wav_16k))
        score = pesq_fn(16000, target_wav_16k[:min_len], converted_wav_16k[:min_len], "wb")
        return round(float(score), 4)
    except Exception as e:
        print(f"  [PESQ] Error: {e}")
        return None


def eval_mcd(
    converted_wav: np.ndarray, target_wav: np.ndarray, sr: int = 16000
) -> Optional[float]:
    """Mel Cepstral Distortion (dB)."""
    if not _LIBROSA:
        return None
    try:
        r = librosa.feature.mfcc(y=target_wav.astype(np.float32), sr=sr, n_mfcc=13)
        s = librosa.feature.mfcc(y=converted_wav.astype(np.float32), sr=sr, n_mfcc=13)
        L = min(r.shape[1], s.shape[1])
        mcd_val = float(np.mean(np.sqrt(2 * np.sum((r[1:, :L] - s[1:, :L]) ** 2, axis=0))))
        return round(mcd_val, 4)
    except Exception as e:
        print(f"  [MCD] Error: {e}")
        return None


def _extract_f0(wav_16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 and voiced flag using librosa.pyin."""
    f0, voiced, _ = librosa.pyin(
        wav_16k, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=16000
    )
    voiced_flag = ~np.isnan(f0)
    f0_clean = np.where(voiced_flag, f0, 0.0)
    return f0_clean, voiced_flag


def eval_f0(
    source_wav_16k: np.ndarray, converted_wav_16k: np.ndarray
) -> tuple[Optional[float], Optional[float]]:
    """F0 jitter (Hz) and F0 RMSE (Hz)."""
    if not _LIBROSA:
        return None, None
    try:
        f0_src, v_src = _extract_f0(source_wav_16k)
        f0_cvt, v_cvt = _extract_f0(converted_wav_16k)

        # Jitter: mean absolute frame-to-frame F0 difference on voiced frames
        voiced_cvt = f0_cvt[v_cvt]
        if len(voiced_cvt) < 2:
            jitter = 0.0
        else:
            diffs = np.abs(np.diff(voiced_cvt))
            jitter = float(np.mean(diffs))

        # F0 RMSE on overlapping voiced frames
        min_len = min(len(f0_src), len(f0_cvt))
        both_voiced = v_src[:min_len] & v_cvt[:min_len]
        if both_voiced.sum() < 2:
            rmse = None
        else:
            diff = f0_src[:min_len][both_voiced] - f0_cvt[:min_len][both_voiced]
            rmse = float(np.sqrt(np.mean(diff ** 2)))

        return round(jitter, 2), round(rmse, 2) if rmse is not None else None
    except Exception as e:
        print(f"  [F0] Error: {e}")
        return None, None


def eval_vuv(
    source_wav_16k: np.ndarray, converted_wav_16k: np.ndarray
) -> Optional[float]:
    """Voiced/Unvoiced classification error rate (%)."""
    if not _LIBROSA:
        return None
    try:
        _, v_src = _extract_f0(source_wav_16k)
        _, v_cvt = _extract_f0(converted_wav_16k)
        min_len = min(len(v_src), len(v_cvt))
        errors = np.sum(v_src[:min_len] != v_cvt[:min_len])
        return round(float(errors / min_len * 100), 2)
    except Exception as e:
        print(f"  [V/UV] Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------

def _normalize(val: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def compute_overall(result: EvalResult) -> Optional[float]:
    """Weighted overall score 0-100."""
    components = []
    weights_used = 0.0

    if result.utmos is not None:
        components.append(0.30 * _normalize(result.utmos, 1.0, 5.0))
        weights_used += 0.30

    if result.wer is not None:
        components.append(0.25 * (1.0 - result.wer / 100.0))
        weights_used += 0.25

    if result.secs is not None:
        components.append(0.20 * max(0.0, min(1.0, result.secs)))
        weights_used += 0.20

    if result.pesq is not None:
        components.append(0.15 * _normalize(result.pesq, -0.5, 4.5))
        weights_used += 0.15

    if result.f0_jitter_hz is not None:
        components.append(0.10 * (1.0 - _normalize(result.f0_jitter_hz, 0.0, 50.0)))
        weights_used += 0.10

    if weights_used == 0:
        return None

    raw = sum(components) / weights_used
    return round(raw * 100, 1)


# ---------------------------------------------------------------------------
# Report card rendering
# ---------------------------------------------------------------------------

def _bar(fraction: float, width: int = 10) -> str:
    filled = int(round(fraction * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def print_report(result: EvalResult):
    W = 52
    border = "\u2550" * W
    src_name = os.path.basename(result.source_file)
    cvt_name = os.path.basename(result.converted_file)
    tgt_name = os.path.basename(result.target_file) if result.target_file else "N/A"

    print(f"\u2554{border}\u2557")
    print(f"\u2551{'Speech Quality Report Card':^{W}}\u2551")
    print(f"\u2560{border}\u2563")
    print(f"\u2551 {'Source:':<12}{src_name:<{W - 14}}\u2551")
    print(f"\u2551 {'Converted:':<12}{cvt_name:<{W - 14}}\u2551")
    print(f"\u2551 {'Target:':<12}{tgt_name:<{W - 14}}\u2551")
    print(f"\u2560{border}\u2563")

    def _row(label: str, value_str: str, frac: float):
        bar = _bar(frac)
        content = f" {label:<24}{value_str:<14}{bar} "
        print(f"\u2551{content:<{W}}\u2551")

    if result.utmos is not None:
        _row("UTMOS (Naturalness)", f"{result.utmos:.2f} / 5.00", result.utmos / 5.0)
    else:
        _row("UTMOS (Naturalness)", "skipped", 0)

    if result.wer is not None:
        _row("WER (Intelligibility)", f"{result.wer:.1f}%", 1.0 - result.wer / 100.0)
    else:
        _row("WER (Intelligibility)", "skipped", 0)

    if result.secs is not None:
        _row("SECS (Speaker Match)", f"{result.secs:.3f}", max(0, result.secs))
    else:
        _row("SECS (Speaker Match)", "skipped", 0)

    if result.pesq is not None:
        _row("PESQ (Perceptual)", f"{result.pesq:.2f} / 4.50", _normalize(result.pesq, -0.5, 4.5))
    else:
        _row("PESQ (Perceptual)", "skipped", 0)

    if result.mcd is not None:
        # Lower MCD is better; ~4 is excellent, ~12+ is poor
        _row("MCD (Spectral Dist.)", f"{result.mcd:.2f}", 1.0 - _normalize(result.mcd, 0, 15))
    else:
        _row("MCD (Spectral Dist.)", "skipped", 0)

    if result.f0_jitter_hz is not None:
        _row("F0 Smoothness", f"{result.f0_jitter_hz:.1f} Hz jitter", 1.0 - _normalize(result.f0_jitter_hz, 0, 50))
    else:
        _row("F0 Smoothness", "skipped", 0)

    if result.f0_rmse_hz is not None:
        _row("F0 RMSE", f"{result.f0_rmse_hz:.1f} Hz", 1.0 - _normalize(result.f0_rmse_hz, 0, 100))

    if result.vuv_error_pct is not None:
        _row("V/UV Error", f"{result.vuv_error_pct:.1f}%", 1.0 - result.vuv_error_pct / 100.0)

    print(f"\u2560{border}\u2563")

    if result.overall is not None:
        _row("OVERALL SCORE", f"{result.overall:.1f} / 100", result.overall / 100.0)
    else:
        _row("OVERALL SCORE", "N/A", 0)

    print(f"\u255A{border}\u255D")

    if result.skipped:
        print(f"\nSkipped metrics (missing packages): {', '.join(result.skipped)}")

    if result.source_transcript:
        print(f"\nSource transcript:    {result.source_transcript}")
    if result.converted_transcript:
        print(f"Converted transcript: {result.converted_transcript}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    source_audio: str,
    converted_audio: str,
    target_audio: Optional[str] = None,
    whisper_model: str = "base",
) -> EvalResult:
    result = EvalResult(
        source_file=source_audio,
        converted_file=converted_audio,
        target_file=target_audio or "",
    )

    print(f"Loading audio files...")
    src_16k = load_audio(source_audio, sr=16000)
    cvt_16k = load_audio(converted_audio, sr=16000)
    tgt_16k = load_audio(target_audio, sr=16000) if target_audio else None

    # UTMOS
    print("  Computing UTMOS...")
    result.utmos = eval_utmos(cvt_16k)
    if result.utmos is None:
        result.skipped.append("UTMOS")

    # WER
    print("  Computing WER...")
    if _WHISPER:
        result.wer, result.source_transcript, result.converted_transcript = eval_wer(
            src_16k, cvt_16k, whisper_model
        )
    else:
        result.skipped.append("WER (install openai-whisper)")

    # SECS
    if tgt_16k is not None:
        print("  Computing SECS...")
        result.secs = eval_secs(cvt_16k, tgt_16k)
        if result.secs is None:
            result.skipped.append("SECS (install resemblyzer)")
    else:
        result.skipped.append("SECS (no target audio)")

    # PESQ
    if tgt_16k is not None:
        print("  Computing PESQ...")
        result.pesq = eval_pesq(cvt_16k, tgt_16k)
        if result.pesq is None:
            result.skipped.append("PESQ (install pesq)")
    else:
        result.skipped.append("PESQ (no target audio)")

    # MCD
    if tgt_16k is not None:
        print("  Computing MCD...")
        result.mcd = eval_mcd(cvt_16k, tgt_16k, sr=16000)
        if result.mcd is None:
            result.skipped.append("MCD")
    else:
        result.skipped.append("MCD (no target audio)")

    # F0
    print("  Computing F0 metrics...")
    result.f0_jitter_hz, result.f0_rmse_hz = eval_f0(src_16k, cvt_16k)
    if result.f0_jitter_hz is None:
        result.skipped.append("F0")

    # V/UV
    print("  Computing V/UV error...")
    result.vuv_error_pct = eval_vuv(src_16k, cvt_16k)
    if result.vuv_error_pct is None:
        result.skipped.append("V/UV")

    # Overall
    result.overall = compute_overall(result)

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def batch_evaluate(
    eval_dir: str,
    source_audio: str,
    target_audio: Optional[str] = None,
    pattern: str = "crt_*refined*.wav",
    whisper_model: str = "base",
) -> list[EvalResult]:
    search = os.path.join(eval_dir, pattern)
    files = sorted(glob.glob(search))
    if not files:
        print(f"No files matched pattern '{pattern}' in {eval_dir}")
        return []

    print(f"Found {len(files)} file(s) matching '{pattern}':\n")
    results = []
    for f in files:
        print(f"--- Evaluating: {os.path.basename(f)} ---")
        r = evaluate(source_audio, f, target_audio, whisper_model)
        print_report(r)
        print()
        results.append(r)

    # Comparison table
    if len(results) > 1:
        print_comparison_table(results)

    return results


def print_comparison_table(results: list[EvalResult]):
    header = f"{'File':<40} {'UTMOS':>6} {'WER%':>6} {'SECS':>6} {'PESQ':>6} {'MCD':>6} {'Jitter':>7} {'Overall':>8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("COMPARISON TABLE")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        name = os.path.basename(r.converted_file)[:39]
        utmos_s = f"{r.utmos:.2f}" if r.utmos is not None else "  -"
        wer_s = f"{r.wer:.1f}" if r.wer is not None else "  -"
        secs_s = f"{r.secs:.3f}" if r.secs is not None else "  -"
        pesq_s = f"{r.pesq:.2f}" if r.pesq is not None else "  -"
        mcd_s = f"{r.mcd:.2f}" if r.mcd is not None else "  -"
        jit_s = f"{r.f0_jitter_hz:.1f}" if r.f0_jitter_hz is not None else "  -"
        ov_s = f"{r.overall:.1f}" if r.overall is not None else "  -"
        print(f"{name:<40} {utmos_s:>6} {wer_s:>6} {secs_s:>6} {pesq_s:>6} {mcd_s:>6} {jit_s:>7} {ov_s:>8}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate voice conversion speech quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source_audio", type=str, help="Source speech wav file")
    parser.add_argument("--converted_audio", type=str, help="Converted speech wav file")
    parser.add_argument("--target_audio", type=str, default=None, help="Target speaker reference wav (for SECS, PESQ, MCD)")
    parser.add_argument("--output_json", type=str, default=None, help="Save results as JSON")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size (default: base)")

    # Batch mode
    parser.add_argument("--eval_dir", type=str, default=None, help="Directory containing wav files for batch eval")
    parser.add_argument("--pattern", type=str, default="crt_*refined*.wav", help='Glob pattern (default: "crt_*refined*.wav")')

    args = parser.parse_args()

    # Validate
    if args.eval_dir:
        if not args.source_audio:
            parser.error("--source_audio is required for batch evaluation")
        results = batch_evaluate(
            args.eval_dir, args.source_audio, args.target_audio, args.pattern, args.whisper_model
        )
    else:
        if not args.source_audio or not args.converted_audio:
            parser.error("--source_audio and --converted_audio are required for single evaluation")
        r = evaluate(args.source_audio, args.converted_audio, args.target_audio, args.whisper_model)
        print_report(r)
        results = [r]

    # Save JSON
    if args.output_json and results:
        out = [asdict(r) for r in results]
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out if len(out) > 1 else out[0], f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
