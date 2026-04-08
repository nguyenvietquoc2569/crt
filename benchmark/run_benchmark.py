#!/usr/bin/env python3
"""
Benchmark: Compare CRT v5.6.5, kNN-VC, and FreeVC on the same source-target pair.
Computes SECS and MCD for all methods using identical audio inputs.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
print('[DEBUG] Starting imports...', flush=True)
import argparse
import time
print('[DEBUG] Importing numpy...', flush=True)
import numpy as np
print('[DEBUG] Importing librosa...', flush=True)
import librosa
print('[DEBUG] Importing soundfile...', flush=True)
import soundfile as sf
from pathlib import Path
print('[DEBUG] All imports done.', flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source_audio', default='/datastore/eaic/nvquoc/mimi_vc/source.mp3')
    p.add_argument('--target_dir', default='/datastore/eaic/nvquoc/mimi_vc/target_audio')
    p.add_argument('--crt_output_dir', default='/datastore/eaic/nvquoc/mimi_vc/v5.6.5-secs-high/output')
    p.add_argument('--output_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/results')
    p.add_argument('--knnvc_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/knn-vc')
    p.add_argument('--freevc_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/FreeVC')
    p.add_argument('--sr', type=int, default=24000)
    p.add_argument('--skip_knnvc', action='store_true')
    p.add_argument('--skip_freevc', action='store_true')
    p.add_argument('--crt_step', type=int, default=210000)
    p.add_argument('--timeout', type=int, default=300, help='Timeout per method in seconds')
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────

def compute_all_secs(wav_paths, names, spk_enc, tgt_emb):
    """Compute SECS for all wav files."""
    from resemblyzer import preprocess_wav
    results = {}
    for i, (p, name) in enumerate(zip(wav_paths, names)):
        t0 = time.time()
        w = preprocess_wav(p)
        ea = spk_enc.embed_utterance(w)
        s = float(np.dot(ea, tgt_emb) / (np.linalg.norm(ea) * np.linalg.norm(tgt_emb)))
        results[name] = s
        print(f'    [{i+1}/{len(wav_paths)}] {name}: SECS={s:.4f} ({time.time()-t0:.1f}s)', flush=True)
    return results


def compute_secs_single(wav_path, spk_enc, tgt_emb):
    """Compute SECS for a single wav file."""
    from resemblyzer import preprocess_wav
    w = preprocess_wav(wav_path)
    ea = spk_enc.embed_utterance(w)
    return float(np.dot(ea, tgt_emb) / (np.linalg.norm(ea) * np.linalg.norm(tgt_emb)))


def compute_mcd(ref, syn, sr):
    r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=sr, n_mfcc=13)
    s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=sr, n_mfcc=13)
    L = min(r.shape[1], s.shape[1])
    return float(np.mean(np.sqrt(2 * np.sum((r[1:, :L] - s[1:, :L]) ** 2, axis=0))))


# ── WER (Whisper ASR) ────────────────────────────────────────

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print('  Loading Whisper (base) for WER...', flush=True)
        _whisper_model = whisper.load_model('base', device='cuda')
    return _whisper_model

def compute_wer(ref_text, wav_path):
    """Compute Word Error Rate between reference text and ASR transcription."""
    model = _get_whisper()
    # Load audio via librosa to avoid ffmpeg dependency
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    import whisper
    audio = whisper.pad_or_trim(wav)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language='en', fp16=True)
    result = whisper.decode(model, mel, options)
    hyp_text = result.text.strip()

    # Normalize: lowercase, remove punctuation
    import re
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.split()

    ref_words = normalize(ref_text)
    hyp_words = normalize(hyp_text)

    if not ref_words:
        return 0.0, hyp_text

    # Levenshtein distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    wer = dp[n][m] / n * 100
    return wer, hyp_text


def get_source_transcript(source_path):
    """Transcribe source audio to get reference text for WER computation."""
    model = _get_whisper()
    wav, sr = librosa.load(source_path, sr=16000, mono=True)
    import whisper
    audio = whisper.pad_or_trim(wav)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language='en', fp16=True)
    result = whisper.decode(model, mel, options)
    text = result.text.strip()
    print(f'  Source transcript: "{text}"', flush=True)
    return text


# ── UTMOS (Neural MOS Predictor) ─────────────────────────────

_utmos_model = None

def _get_utmos():
    global _utmos_model
    if _utmos_model is None:
        import torch
        print('  Loading UTMOS predictor...', flush=True)
        _utmos_model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong',
                                       trust_repo=True)
        _utmos_model = _utmos_model.cuda().eval()
    return _utmos_model

def compute_utmos(wav_path):
    """Compute UTMOS (predicted MOS 1-5) for a wav file."""
    import torch
    model = _get_utmos()
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).cuda()
    with torch.no_grad():
        score = model(wav_tensor, sr).item()
    return score


# ── kNN-VC ────────────────────────────────────────────────────

def run_knnvc(source_path, target_dir, output_path, knnvc_dir, timeout=300):
    print('  Running kNN-VC...', flush=True)
    t0 = time.time()
    import torch, torchaudio

    sys.path.insert(0, knnvc_dir)
    os.chdir(knnvc_dir)

    from hubconf import knn_vc
    print('  Loading kNN-VC model...', flush=True)
    model = knn_vc(pretrained=True, progress=True, prematched=False, device='cuda')

    # Prepare source as 16kHz wav file (kNN-VC API expects paths or tensors)
    src_wav_16k = os.path.join(os.path.dirname(output_path), 'knnvc_src_16k.wav')
    w, sr = torchaudio.load(source_path)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    torchaudio.save(src_wav_16k, w, 16000)

    # Prepare target: first file, max 30s, as 16kHz wav
    tgt_wav_16k = os.path.join(os.path.dirname(output_path), 'knnvc_tgt_16k.wav')
    tgt_path = None
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(target_dir).glob(ext)):
            tgt_path = str(f)
            break
        if tgt_path:
            break
    if not tgt_path:
        print('  ERROR: No target audio found')
        return None

    w, sr = torchaudio.load(tgt_path)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    max_samples = 30 * 16000
    if w.shape[1] > max_samples:
        w = w[:, :max_samples]
    torchaudio.save(tgt_wav_16k, w, 16000)
    print(f'  Source: {os.path.getsize(src_wav_16k)//1000}KB, Target: {w.shape[1]/16000:.1f}s', flush=True)

    # kNN-VC API: get_features(path) -> get_matching_set([path]) -> match -> vocode
    print('  Computing features...', flush=True)
    with torch.no_grad():
        query_seq = model.get_features(src_wav_16k)
        matching_set = model.get_matching_set([tgt_wav_16k])
        print(f'  Query: {query_seq.shape}, Matching: {matching_set.shape}', flush=True)
        out_wav = model.match(query_seq, matching_set, topk=4)
    out_np = out_wav.squeeze().cpu().numpy()

    sf.write(output_path, out_np, 16000)
    elapsed = time.time() - t0
    print(f'  kNN-VC done in {elapsed:.1f}s', flush=True)
    return output_path


# ── FreeVC ────────────────────────────────────────────────────

def run_freevc(source_path, target_dir, output_path, freevc_dir, timeout=300):
    """Run FreeVC in a subprocess to avoid torch module conflicts with kNN-VC."""
    print('  Running FreeVC (subprocess)...', flush=True)
    t0 = time.time()
    import subprocess

    # Find target audio
    tgt_path = None
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(target_dir).glob(ext)):
            tgt_path = str(f)
            break
        if tgt_path:
            break
    if not tgt_path:
        print('  ERROR: No target audio found')
        return None

    ckpt_path = os.path.join(freevc_dir, 'checkpoints', 'freevc.pth')
    config_path = os.path.join(freevc_dir, 'configs', 'freevc.json')
    spk_enc_path = os.path.join(freevc_dir, 'speaker_encoder', 'ckpt', 'pretrained_bak_5805000.pt')

    for p in [ckpt_path, config_path, spk_enc_path]:
        if not os.path.exists(p):
            print(f'  ERROR: Missing {p}')
            return None

    # Write a standalone FreeVC inference script and run it in a subprocess
    # This avoids torch internal conflicts (_to_dlpack etc.) from kNN-VC's torch.hub
    worker_script = os.path.join(os.path.dirname(output_path), '_freevc_worker.py')
    with open(worker_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""FreeVC worker — runs in isolated subprocess to avoid torch conflicts."""
import os, sys
os.chdir({freevc_dir!r})
sys.path.insert(0, {freevc_dir!r})

import torch
import librosa
import numpy as np
import soundfile as sf

import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from wavlm import WavLM, WavLMConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  FreeVC device: {{device}}', flush=True)

# Load config + model
hps = utils.get_hparams_from_file({config_path!r})
print('  Loading FreeVC model...', flush=True)
net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device).eval()
utils.load_checkpoint({ckpt_path!r}, net_g, None)

# Speaker encoder
smodel = SpeakerEncoder({spk_enc_path!r}, device=device)

# WavLM — load from checkpoint file directly
print('  Loading WavLM...', flush=True)
wavlm_ckpt = torch.load({os.path.join(freevc_dir, 'wavlm', 'WavLM-Large.pt')!r}, map_location='cpu')
wavlm_cfg = WavLMConfig(wavlm_ckpt['cfg'])
cmodel = WavLM(wavlm_cfg).to(device)
cmodel.load_state_dict(wavlm_ckpt['model'])
cmodel.eval()

# Load audio
target_sr = hps.data.sampling_rate
src, _ = librosa.load({source_path!r}, sr=target_sr)
tgt, _ = librosa.load({tgt_path!r}, sr=target_sr)

# Speaker embedding from target
tgt16 = librosa.resample(tgt, orig_sr=target_sr, target_sr=16000)
g_tgt = smodel.embed_utterance(tgt16)
g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)

# WavLM features from source — resample to 16kHz, use extract_features API
src16 = librosa.resample(src, orig_sr=target_sr, target_sr=16000)
src_tensor = torch.from_numpy(src16).unsqueeze(0).to(device)
with torch.no_grad():
    c = cmodel.extract_features(src_tensor)[0]
    c = c.transpose(1, 2)

# Inference
print('  Running FreeVC inference...', flush=True)
with torch.no_grad():
    audio = net_g.infer(c, g=g_tgt)
    audio = audio[0][0].data.cpu().float().numpy()

sf.write({output_path!r}, audio, target_sr)
print('  FreeVC subprocess done.', flush=True)
''')

    # Run in subprocess with inherited env (CUDA_VISIBLE_DEVICES etc.)
    result = subprocess.run(
        [sys.executable, worker_script],
        capture_output=False,  # let output stream to parent
        timeout=timeout,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(f'  FreeVC subprocess exited with code {result.returncode}', flush=True)
        return None

    if os.path.exists(output_path):
        elapsed = time.time() - t0
        print(f'  FreeVC done in {elapsed:.1f}s', flush=True)
        return output_path
    else:
        print('  FreeVC: output file not created', flush=True)
        return None


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    orig_dir = os.getcwd()

    print('=' * 70)
    print(' Benchmark: CRT v5.6.5 vs kNN-VC vs FreeVC')
    print('=' * 70)
    print(f'  Source: {args.source_audio}')
    print(f'  Target: {args.target_dir}')
    print(f'  CRT outputs: {args.crt_output_dir}')
    sys.stdout.flush()

    # Load source audio
    print('  Loading source audio...', flush=True)
    src_wav, _ = librosa.load(args.source_audio, sr=args.sr, mono=True)
    print(f'  Source: {len(src_wav)/args.sr:.1f}s', flush=True)

    # SECS: use resemblyzer but precompute target embedding once
    print('  Initializing CUDA...', flush=True)
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {dev}', flush=True)

    # Load resemblyzer on CPU (LSTM + cuDNN has compatibility issues, CPU is fast enough)
    print('  Loading resemblyzer VoiceEncoder (cpu)...', flush=True)
    from resemblyzer import VoiceEncoder, preprocess_wav
    spk_enc = VoiceEncoder(device='cpu')
    print('  VoiceEncoder loaded.', flush=True)

    # Target embedding (short clip, fast)
    print('  Computing target speaker embedding (3s clip)...', flush=True)
    t0_emb = time.time()
    tgt_path = None
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_path = str(f)
            break
        if tgt_path:
            break
    tgt_wav = preprocess_wav(tgt_path)
    tgt_wav = tgt_wav[:16000 * 3]  # only 3 seconds
    tgt_emb = spk_enc.embed_utterance(tgt_wav)
    print(f'  Target embedding done in {time.time()-t0_emb:.1f}s', flush=True)

    # Transcribe source for WER computation
    print('  Transcribing source audio (Whisper)...', flush=True)
    t0_w = time.time()
    try:
        src_transcript = get_source_transcript(args.source_audio)
        print(f'  Source transcription done in {time.time()-t0_w:.1f}s', flush=True)
    except Exception as e:
        print(f'  Whisper transcription failed: {e}. WER will be skipped.', flush=True)
        src_transcript = None

    results = {}

    # ── kNN-VC (run first — need GPU inference + 1 SECS computation) ──
    if not args.skip_knnvc:
        print('\n--- kNN-VC ---', flush=True)
        knnvc_out = os.path.join(args.output_dir, 'knnvc_output.wav')
        try:
            knnvc_path = run_knnvc(args.source_audio, args.target_dir, knnvc_out, args.knnvc_dir, args.timeout)
            os.chdir(orig_dir)
            if knnvc_path and os.path.exists(knnvc_path):
                print('  Computing kNN-VC metrics...', flush=True)
                t0_m = time.time()
                s = compute_secs_single(knnvc_path, spk_enc, tgt_emb)
                wav, _ = librosa.load(knnvc_path, sr=args.sr, mono=True)
                m = compute_mcd(src_wav, wav, args.sr)
                r = {'secs': s, 'mcd': m}
                # WER
                if src_transcript:
                    try:
                        wer, hyp = compute_wer(src_transcript, knnvc_path)
                        r['wer'] = wer
                        print(f'    WER={wer:.1f}% hyp="{hyp}"', flush=True)
                    except Exception as e:
                        print(f'    WER failed: {e}', flush=True)
                # UTMOS
                try:
                    mos = compute_utmos(knnvc_path)
                    r['utmos'] = mos
                    print(f'    UTMOS={mos:.3f}', flush=True)
                except Exception as e:
                    print(f'    UTMOS failed: {e}', flush=True)
                results['kNN-VC'] = r
                print(f'  {"kNN-VC":<30} SECS={s:.4f}  MCD={m:.2f}  ({time.time()-t0_m:.0f}s)', flush=True)
        except Exception as e:
            os.chdir(orig_dir)
            print(f'  kNN-VC FAILED: {e}', flush=True)
    else:
        print('\n--- kNN-VC: SKIPPED ---', flush=True)

    # ── FreeVC (run second — need GPU inference + 1 SECS computation) ──
    if not args.skip_freevc:
        print('\n--- FreeVC ---', flush=True)
        freevc_out = os.path.join(args.output_dir, 'freevc_output.wav')
        try:
            freevc_path = run_freevc(args.source_audio, args.target_dir, freevc_out, args.freevc_dir, args.timeout)
            os.chdir(orig_dir)
            if freevc_path and os.path.exists(freevc_path):
                print('  Computing FreeVC metrics...', flush=True)
                t0_m = time.time()
                s = compute_secs_single(freevc_path, spk_enc, tgt_emb)
                wav, _ = librosa.load(freevc_path, sr=args.sr, mono=True)
                m = compute_mcd(src_wav, wav, args.sr)
                r = {'secs': s, 'mcd': m}
                if src_transcript:
                    try:
                        wer, hyp = compute_wer(src_transcript, freevc_path)
                        r['wer'] = wer
                        print(f'    WER={wer:.1f}% hyp="{hyp}"', flush=True)
                    except Exception as e:
                        print(f'    WER failed: {e}', flush=True)
                try:
                    mos = compute_utmos(freevc_path)
                    r['utmos'] = mos
                    print(f'    UTMOS={mos:.3f}', flush=True)
                except Exception as e:
                    print(f'    UTMOS failed: {e}', flush=True)
                results['FreeVC'] = r
                print(f'  {"FreeVC":<30} SECS={s:.4f}  MCD={m:.2f}  ({time.time()-t0_m:.0f}s)', flush=True)
        except Exception as e:
            os.chdir(orig_dir)
            print(f'  FreeVC FAILED: {e}', flush=True)
    else:
        print('\n--- FreeVC: SKIPPED ---', flush=True)

    # ── CRT variants (last — 11 SECS computations, slowest) ──
    print('\n--- CRT v5.6.5 Results ---', flush=True)
    step = args.crt_step
    import glob
    available = glob.glob(os.path.join(args.crt_output_dir, f'*-{step}-step.wav'))
    print(f'  Found {len(available)} files for step {step}', flush=True)
    crt_variants = [
        ('CRT-FAISS-baseline', f'faiss_baseline-{step}-step.wav'),
        ('CRT-argmax', f'crt_argmax-{step}-step.wav'),
        ('CRT-conf>0.3', f'crt_conf_0.3-{step}-step.wav'),
        ('CRT-conf>0.4', f'crt_conf_0.4-{step}-step.wav'),
        ('CRT-conf>0.5', f'crt_conf_0.5-{step}-step.wav'),
        ('CRT-conf>0.6', f'crt_conf_0.6-{step}-step.wav'),
        ('CRT-conf>0.7', f'crt_conf_0.7-{step}-step.wav'),
        ('CRT-conf>0.8', f'crt_conf_0.8-{step}-step.wav'),
        ('CRT-conf>0.9', f'crt_conf_0.9-{step}-step.wav'),
        ('CRT-reranker', f'crt_reranker-{step}-step.wav'),
        ('CRT-reranker+conf', f'crt_reranker_conf-{step}-step.wav'),
    ]

    valid_variants = []
    for name, filename in crt_variants:
        path = os.path.join(args.crt_output_dir, filename)
        if os.path.exists(path):
            valid_variants.append((name, path))
        else:
            print(f'  {name:<30} NOT FOUND', flush=True)

    if valid_variants:
        print(f'  Computing SECS for {len(valid_variants)} variants...', flush=True)
        t0_secs = time.time()
        secs_results = compute_all_secs(
            [p for _, p in valid_variants],
            [n for n, _ in valid_variants],
            spk_enc, tgt_emb)
        print(f'  All SECS done in {time.time()-t0_secs:.1f}s', flush=True)

        print(f'  Computing MCD + WER + UTMOS...', flush=True)
        for name, path in valid_variants:
            wav, _ = librosa.load(path, sr=args.sr, mono=True)
            m = compute_mcd(src_wav, wav, args.sr)
            s = secs_results[name]
            r = {'secs': s, 'mcd': m}
            # WER for all CRT variants
            if src_transcript:
                try:
                    wer, hyp = compute_wer(src_transcript, path)
                    r['wer'] = wer
                except Exception:
                    pass
            # UTMOS for all CRT variants
            try:
                mos = compute_utmos(path)
                r['utmos'] = mos
            except Exception:
                pass
            results[name] = r
            wer_str = f"  WER={r.get('wer', -1):.1f}%" if 'wer' in r else ''
            mos_str = f"  UTMOS={r.get('utmos', -1):.3f}" if 'utmos' in r else ''
            print(f'  {name:<30} SECS={s:.4f}  MCD={m:.2f}{wer_str}{mos_str}', flush=True)

    # ── Summary ──
    has_wer = any('wer' in r for r in results.values())
    has_utmos = any('utmos' in r for r in results.values())
    print(f'\n{"=" * 90}')
    print(f' BENCHMARK RESULTS (sorted by MCD)')
    print(f'{"=" * 90}')
    hdr = f'{"Method":<30} {"SECS":>8} {"MCD":>8}'
    if has_wer:
        hdr += f' {"WER%":>8}'
    if has_utmos:
        hdr += f' {"UTMOS":>8}'
    print(hdr)
    print('-' * (54 + (9 if has_wer else 0) + (9 if has_utmos else 0)))
    for name, r in sorted(results.items(), key=lambda x: x[1]['mcd']):
        line = f'{name:<30} {r["secs"]:>8.4f} {r["mcd"]:>8.2f}'
        if has_wer:
            line += f' {r.get("wer", -1):>7.1f}%' if 'wer' in r else f' {"N/A":>8}'
        if has_utmos:
            line += f' {r.get("utmos", -1):>8.3f}' if 'utmos' in r else f' {"N/A":>8}'
        print(line)
    print(f'{"=" * 90}')

    # Save
    import json
    with open(os.path.join(args.output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output_dir}/benchmark_results.json')


if __name__ == '__main__':
    main()
