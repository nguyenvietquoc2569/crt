#!/usr/bin/env python3
"""
Latency Benchmark: Measure real-time factor (RTF) for CRT, kNN-VC, and FreeVC.
Reports per-stage timing, total latency, RTF, and GPU memory usage.
Runs each method multiple times for stable measurements.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
import argparse
import time
import numpy as np
import librosa
import soundfile as sf
import torch
import gc
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source_audio', default='/datastore/eaic/nvquoc/mimi_vc/source.mp3')
    p.add_argument('--target_dir', default='/datastore/eaic/nvquoc/mimi_vc/target_audio')
    p.add_argument('--crt_ckpt', default='/datastore/eaic/nvquoc/mimi_vc/v5.6.5-secs-high/checkpoints/crt_xlarge_best.pt')
    p.add_argument('--knnvc_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/knn-vc')
    p.add_argument('--freevc_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/FreeVC')
    p.add_argument('--data_processed', default='/datastore/eaic/nvquoc/mimi_vc/data_storage/processed_merged_v3')
    p.add_argument('--n_runs', type=int, default=5, help='Number of runs per method for averaging')
    p.add_argument('--warmup_runs', type=int, default=2, help='Warmup runs (excluded from timing)')
    p.add_argument('--output_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/results_latency')
    p.add_argument('--skip_knnvc', action='store_true')
    p.add_argument('--skip_freevc', action='store_true')
    p.add_argument('--skip_crt', action='store_true')
    return p.parse_args()


def gpu_mem():
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0


def gpu_peak():
    """Return peak GPU memory in MB since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


# ── CRT Pipeline ─────────────────────────────────────────────

def benchmark_crt(args, device):
    """Benchmark full CRT pipeline with per-stage timing."""
    print('\n' + '='*60)
    print(' CRT Pipeline Latency Benchmark')
    print('='*60, flush=True)

    # Load source audio
    src_wav = librosa.load(args.source_audio, sr=24000, mono=True)[0].astype(np.float32)
    audio_duration = len(src_wav) / 24000
    print(f'  Source audio: {audio_duration:.1f}s', flush=True)

    # Stage 0: Load models (not timed — one-time cost)
    print('  Loading Mimi...', flush=True)
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16
    mimi = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR = feat_ext.sampling_rate

    # Load CRT model
    print('  Loading CRT...', flush=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(args.crt_ckpt)))
    # Import CRT architecture from v5.6.5
    crt_script = os.path.join(os.path.dirname(os.path.dirname(args.crt_ckpt)), 'train_crt.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location('crt_module', crt_script)
    crt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crt_mod)

    ckpt = torch.load(args.crt_ckpt, weights_only=False, map_location='cpu')
    cfg = crt_mod.MODEL_CONFIGS[ckpt.get('model_size', 'xlarge')]
    crt_model = crt_mod.CRT(n_cb=18, cb_size=2048, spk_emb_dim=256, **cfg).to(device).eval()
    crt_model.load_state_dict(ckpt['model_state'], strict=False)

    # CB0 embeddings for FAISS
    import faiss
    q = mimi.quantizer
    if hasattr(q, 'semantic_residual_vector_quantizer'):
        layer = q.semantic_residual_vector_quantizer.layers[0]
    else:
        layer = q.layers[0]
    cb0_emb = None
    for attr in ['codebook.embed', 'codebook.embed_sum', 'codebook.embedding.weight']:
        obj = layer
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            w = obj.detach().cpu().float()
            if w.dim() == 3:
                w = w.squeeze(0)
            if w.shape[0] >= 2048:
                cb0_emb = w.numpy()
                break
        except AttributeError:
            continue

    # Speaker embedding
    from resemblyzer import VoiceEncoder
    spk_enc = VoiceEncoder(device='cpu')
    tgt_wavs = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_wavs.append(librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32))
    tgt_concat = np.concatenate(tgt_wavs)
    tgt16 = librosa.resample(tgt_concat, orig_sr=SR, target_sr=16000)
    tgt_spk_emb = torch.tensor(spk_enc.embed_utterance(tgt16), dtype=torch.float32)

    # Pre-encode target for FAISS
    CHUNK = int(30 * SR)
    tgt_codes = torch.cat([
        mimi.encode(
            feat_ext(raw_audio=tgt_concat[i:i+CHUNK], sampling_rate=SR, return_tensors='pt')
            ['input_values'].to(device=device, dtype=dtype),
            None
        ).audio_codes.cpu()
        for i in range(0, len(tgt_concat), CHUNK)
        if len(tgt_concat[i:i+CHUNK]) > SR
    ], dim=2)
    tgt_full = tgt_codes[0].numpy().T
    tgt_sem = tgt_codes[0, 0].numpy()

    # Build target FAISS index
    tgt_emb = cb0_emb[tgt_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb)
    tgt_index = faiss.IndexFlatIP(cb0_emb.shape[1])
    tgt_index.add(tgt_emb)

    print(f'  Models loaded. Running {args.warmup_runs} warmup + {args.n_runs} timed runs...', flush=True)

    stage_times = {'encode': [], 'faiss': [], 'crt_forward': [], 'confidence': [], 'decode': [], 'total': []}

    for run in range(args.warmup_runs + args.n_runs):
        is_warmup = run < args.warmup_runs
        reset_gpu_stats()
        torch.cuda.synchronize()

        # Stage 1: Mimi Encode
        t0 = time.perf_counter()
        inp = feat_ext(raw_audio=src_wav, sampling_rate=SR, return_tensors='pt')
        inp = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None) for k, v in inp.items()}
        with torch.no_grad():
            src_codes = mimi.encode(inp['input_values'], inp.get('padding_mask')).audio_codes.cpu()
        torch.cuda.synchronize()
        t_encode = time.perf_counter() - t0

        T_src = src_codes.shape[2]
        src_sem_np = src_codes[0, 0].numpy()

        # Stage 2: FAISS Retrieval
        t0 = time.perf_counter()
        src_emb_q = cb0_emb[src_sem_np].astype(np.float32).copy()
        faiss.normalize_L2(src_emb_q)
        dists_k, indices_k = tgt_index.search(src_emb_q, 8)

        selected = np.zeros(T_src, dtype=np.int64)
        selected[0] = indices_k[0, 0]
        for t in range(1, T_src):
            cands = indices_k[t]
            combined = 0.5 * dists_k[t] + 0.5 / (1 + np.abs(cands - selected[t-1]) / 12.5)
            selected[t] = cands[np.argmax(combined)]

        faiss_codes = src_codes.clone()
        for t in range(T_src):
            faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])
        t_faiss = time.perf_counter() - t0

        # Stage 3: CRT Forward Pass
        ret_cb0_np = np.array([tgt_full[selected[t]][0] for t in range(T_src)])
        ret_cb_all = np.array([tgt_full[selected[t]][1:19] for t in range(T_src)])
        cb0_d = torch.tensor(ret_cb0_np, dtype=torch.long).unsqueeze(0).to(device)
        cb_input_d = torch.tensor(ret_cb_all.T, dtype=torch.long).unsqueeze(0).to(device)
        spk_d = tgt_spk_emb.unsqueeze(0).to(device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _ = crt_model(cb0_d, cb_input_d, spk_d)
        torch.cuda.synchronize()
        t_crt = time.perf_counter() - t0

        # Stage 4: Confidence Gating
        t0 = time.perf_counter()
        probs = logits.float().softmax(-1)
        crt_tokens = probs.argmax(-1)[0].cpu()
        crt_confidence = probs.max(-1).values[0].cpu()
        faiss_tokens = torch.tensor(ret_cb_all.T, dtype=torch.long)
        sel = faiss_tokens.clone()
        mask = crt_confidence > 0.4
        sel[mask] = crt_tokens[mask]
        out_codes = faiss_codes.clone()
        out_codes[0, 1:19] = sel
        t_conf = time.perf_counter() - t0

        # Stage 5: Mimi Decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wav_out = mimi.decode(out_codes.to(device)).audio_values.cpu()
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t0

        t_total = t_encode + t_faiss + t_crt + t_conf + t_decode

        if not is_warmup:
            stage_times['encode'].append(t_encode * 1000)
            stage_times['faiss'].append(t_faiss * 1000)
            stage_times['crt_forward'].append(t_crt * 1000)
            stage_times['confidence'].append(t_conf * 1000)
            stage_times['decode'].append(t_decode * 1000)
            stage_times['total'].append(t_total * 1000)

        label = 'WARMUP' if is_warmup else f'Run {run - args.warmup_runs + 1}'
        print(f'    [{label}] encode={t_encode*1000:.0f}ms faiss={t_faiss*1000:.0f}ms '
              f'crt={t_crt*1000:.0f}ms conf={t_conf*1000:.0f}ms decode={t_decode*1000:.0f}ms '
              f'total={t_total*1000:.0f}ms', flush=True)

    peak_mem = gpu_peak()

    print(f'\n  CRT Pipeline Results ({args.n_runs} runs):')
    print(f'  {"Stage":<20} {"Mean (ms)":>10} {"Std (ms)":>10} {"% Total":>10}')
    print(f'  {"-"*52}')
    mean_total = np.mean(stage_times['total'])
    for stage in ['encode', 'faiss', 'crt_forward', 'confidence', 'decode', 'total']:
        m = np.mean(stage_times[stage])
        s = np.std(stage_times[stage])
        pct = m / mean_total * 100 if stage != 'total' else 100
        print(f'  {stage:<20} {m:>10.1f} {s:>10.1f} {pct:>9.1f}%')
    rtf = mean_total / 1000 / audio_duration
    print(f'  Audio duration: {audio_duration:.1f}s')
    print(f'  RTF: {rtf:.4f}x ({1/rtf:.1f}x faster than real-time)')
    print(f'  Peak GPU memory: {peak_mem:.0f} MB')

    return {
        'method': 'CRT (v5.6.5, 707M)',
        'stages': {k: {'mean_ms': float(np.mean(v)), 'std_ms': float(np.std(v))} for k, v in stage_times.items()},
        'audio_duration_s': audio_duration,
        'rtf': rtf,
        'peak_gpu_mb': peak_mem,
    }


# ── kNN-VC Pipeline ──────────────────────────────────────────

def benchmark_knnvc(args, device):
    print('\n' + '='*60)
    print(' kNN-VC Pipeline Latency Benchmark')
    print('='*60, flush=True)

    src_wav = librosa.load(args.source_audio, sr=24000, mono=True)[0].astype(np.float32)
    audio_duration = len(src_wav) / 24000

    import torchaudio
    sys.path.insert(0, args.knnvc_dir)
    orig_dir = os.getcwd()
    os.chdir(args.knnvc_dir)
    from hubconf import knn_vc

    print('  Loading kNN-VC model...', flush=True)
    model = knn_vc(pretrained=True, progress=True, prematched=False, device='cuda')

    # Prepare files
    src_16k = os.path.join(args.output_dir, '_knnvc_bench_src.wav')
    tgt_16k = os.path.join(args.output_dir, '_knnvc_bench_tgt.wav')
    w, sr = torchaudio.load(args.source_audio)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    torchaudio.save(src_16k, w, 16000)

    tgt_path = None
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_path = str(f)
            break
        if tgt_path:
            break
    w, sr = torchaudio.load(tgt_path)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    w = w[:, :30*16000]
    torchaudio.save(tgt_16k, w, 16000)

    print(f'  Running {args.warmup_runs} warmup + {args.n_runs} timed runs...', flush=True)

    stage_times = {'wavlm_features': [], 'matching_set': [], 'knn_match': [], 'vocoder': [], 'total': []}

    for run in range(args.warmup_runs + args.n_runs):
        is_warmup = run < args.warmup_runs
        reset_gpu_stats()
        torch.cuda.synchronize()

        # WavLM feature extraction (source)
        t0 = time.perf_counter()
        with torch.no_grad():
            query_seq = model.get_features(src_16k)
        torch.cuda.synchronize()
        t_feat = time.perf_counter() - t0

        # Matching set (target)
        t0 = time.perf_counter()
        with torch.no_grad():
            matching_set = model.get_matching_set([tgt_16k])
        torch.cuda.synchronize()
        t_match_set = time.perf_counter() - t0

        # kNN matching + vocoding (combined in the API)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_wav = model.match(query_seq, matching_set, topk=4)
        torch.cuda.synchronize()
        t_match_voc = time.perf_counter() - t0

        t_total = t_feat + t_match_set + t_match_voc

        if not is_warmup:
            stage_times['wavlm_features'].append(t_feat * 1000)
            stage_times['matching_set'].append(t_match_set * 1000)
            stage_times['knn_match'].append(t_match_voc * 1000)
            stage_times['total'].append(t_total * 1000)

        label = 'WARMUP' if is_warmup else f'Run {run - args.warmup_runs + 1}'
        print(f'    [{label}] features={t_feat*1000:.0f}ms match_set={t_match_set*1000:.0f}ms '
              f'match+voc={t_match_voc*1000:.0f}ms total={t_total*1000:.0f}ms', flush=True)

    os.chdir(orig_dir)
    peak_mem = gpu_peak()

    print(f'\n  kNN-VC Results ({args.n_runs} runs):')
    print(f'  {"Stage":<20} {"Mean (ms)":>10} {"Std (ms)":>10} {"% Total":>10}')
    print(f'  {"-"*52}')
    mean_total = np.mean(stage_times['total'])
    for stage in ['wavlm_features', 'matching_set', 'knn_match', 'total']:
        m = np.mean(stage_times[stage])
        s = np.std(stage_times[stage])
        pct = m / mean_total * 100 if stage != 'total' else 100
        print(f'  {stage:<20} {m:>10.1f} {s:>10.1f} {pct:>9.1f}%')
    rtf = mean_total / 1000 / audio_duration
    print(f'  Audio duration: {audio_duration:.1f}s')
    print(f'  RTF: {rtf:.4f}x ({1/rtf:.1f}x faster than real-time)')
    print(f'  Peak GPU memory: {peak_mem:.0f} MB')

    return {
        'method': 'kNN-VC',
        'stages': {k: {'mean_ms': float(np.mean(v)), 'std_ms': float(np.std(v))} for k, v in stage_times.items()},
        'audio_duration_s': audio_duration,
        'rtf': rtf,
        'peak_gpu_mb': peak_mem,
    }


# ── FreeVC Pipeline ──────────────────────────────────────────

def benchmark_freevc(args, device):
    """Run FreeVC in subprocess for clean torch state, parse timing from output."""
    print('\n' + '='*60)
    print(' FreeVC Pipeline Latency Benchmark')
    print('='*60, flush=True)

    src_wav = librosa.load(args.source_audio, sr=24000, mono=True)[0].astype(np.float32)
    audio_duration = len(src_wav) / 24000

    tgt_path = None
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_path = str(f)
            break
        if tgt_path:
            break

    freevc_dir = args.freevc_dir
    ckpt_path = os.path.join(freevc_dir, 'checkpoints', 'freevc.pth')
    config_path = os.path.join(freevc_dir, 'configs', 'freevc.json')
    spk_enc_path = os.path.join(freevc_dir, 'speaker_encoder', 'ckpt', 'pretrained_bak_5805000.pt')
    output_path = os.path.join(args.output_dir, '_freevc_bench_out.wav')

    n_total = args.warmup_runs + args.n_runs

    worker_script = os.path.join(args.output_dir, '_freevc_latency_worker.py')
    with open(worker_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
import os, sys, time, json
import numpy as np
os.chdir({freevc_dir!r})
sys.path.insert(0, {freevc_dir!r})

import torch
import librosa

import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from wavlm import WavLM, WavLMConfig

device = 'cuda'

# Load models
hps = utils.get_hparams_from_file({config_path!r})
net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device).eval()
utils.load_checkpoint({ckpt_path!r}, net_g, None)

smodel = SpeakerEncoder({spk_enc_path!r}, device=device)

wavlm_ckpt = torch.load({os.path.join(freevc_dir, 'wavlm', 'WavLM-Large.pt')!r}, map_location='cpu')
wavlm_cfg = WavLMConfig(wavlm_ckpt['cfg'])
cmodel = WavLM(wavlm_cfg).to(device)
cmodel.load_state_dict(wavlm_ckpt['model'])
cmodel.eval()

target_sr = hps.data.sampling_rate
src, _ = librosa.load({args.source_audio!r}, sr=target_sr)
tgt, _ = librosa.load({tgt_path!r}, sr=target_sr)
tgt16 = librosa.resample(tgt, orig_sr=target_sr, target_sr=16000)
src16 = librosa.resample(src, orig_sr=target_sr, target_sr=16000)

results = []
for run in range({n_total}):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Speaker embedding
    t0 = time.perf_counter()
    g_tgt = smodel.embed_utterance(tgt16)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
    torch.cuda.synchronize()
    t_spk = time.perf_counter() - t0

    # WavLM features
    t0 = time.perf_counter()
    src_tensor = torch.from_numpy(src16).unsqueeze(0).to(device)
    with torch.no_grad():
        c = cmodel.extract_features(src_tensor)[0]
        c = c.transpose(1, 2)
    torch.cuda.synchronize()
    t_wavlm = time.perf_counter() - t0

    # VITS inference
    t0 = time.perf_counter()
    with torch.no_grad():
        audio = net_g.infer(c, g=g_tgt)
        audio = audio[0][0].data.cpu().float().numpy()
    torch.cuda.synchronize()
    t_vits = time.perf_counter() - t0

    t_total = t_spk + t_wavlm + t_vits
    peak = torch.cuda.max_memory_allocated() / 1e6

    results.append({{
        'spk_enc': t_spk * 1000,
        'wavlm': t_wavlm * 1000,
        'vits_decode': t_vits * 1000,
        'total': t_total * 1000,
        'peak_gpu_mb': peak,
    }})
    print(f'    [Run {{run+1}}] spk={{t_spk*1000:.0f}}ms wavlm={{t_wavlm*1000:.0f}}ms '
          f'vits={{t_vits*1000:.0f}}ms total={{t_total*1000:.0f}}ms', flush=True)

with open({os.path.join(args.output_dir, '_freevc_latency.json')!r}, 'w') as f:
    json.dump(results, f)
print('FreeVC latency benchmark done.', flush=True)
''')

    import subprocess
    result = subprocess.run(
        [sys.executable, worker_script],
        capture_output=False, timeout=600,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(f'  FreeVC benchmark failed (exit code {result.returncode})', flush=True)
        return None

    import json
    latency_file = os.path.join(args.output_dir, '_freevc_latency.json')
    with open(latency_file) as f:
        all_runs = json.load(f)

    # Skip warmup runs
    timed_runs = all_runs[args.warmup_runs:]

    print(f'\n  FreeVC Results ({len(timed_runs)} runs):')
    print(f'  {"Stage":<20} {"Mean (ms)":>10} {"Std (ms)":>10} {"% Total":>10}')
    print(f'  {"-"*52}')
    mean_total = np.mean([r['total'] for r in timed_runs])
    for stage in ['spk_enc', 'wavlm', 'vits_decode', 'total']:
        vals = [r[stage] for r in timed_runs]
        m, s = np.mean(vals), np.std(vals)
        pct = m / mean_total * 100 if stage != 'total' else 100
        print(f'  {stage:<20} {m:>10.1f} {s:>10.1f} {pct:>9.1f}%')
    rtf = mean_total / 1000 / audio_duration
    peak_mem = np.mean([r['peak_gpu_mb'] for r in timed_runs])
    print(f'  Audio duration: {audio_duration:.1f}s')
    print(f'  RTF: {rtf:.4f}x ({1/rtf:.1f}x faster than real-time)')
    print(f'  Peak GPU memory: {peak_mem:.0f} MB')

    return {
        'method': 'FreeVC',
        'stages': {k: {'mean_ms': float(np.mean([r[k] for r in timed_runs])),
                       'std_ms': float(np.std([r[k] for r in timed_runs]))}
                   for k in ['spk_enc', 'wavlm', 'vits_decode', 'total']},
        'audio_duration_s': audio_duration,
        'rtf': rtf,
        'peak_gpu_mb': peak_mem,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('='*60)
    print(' Real-Time Capability Benchmark')
    print(f' Device: {device}')
    if device == 'cuda':
        print(f' GPU: {torch.cuda.get_device_name()}')
    print(f' Runs: {args.warmup_runs} warmup + {args.n_runs} timed')
    print('='*60, flush=True)

    all_results = {}

    # kNN-VC first (needs clean GPU state)
    if not args.skip_knnvc and os.path.exists(args.knnvc_dir):
        try:
            all_results['knnvc'] = benchmark_knnvc(args, device)
        except Exception as e:
            print(f'  kNN-VC benchmark failed: {e}', flush=True)
        # Clean up
        reset_gpu_stats()
        for mod_name in list(sys.modules.keys()):
            if 'hubconf' in mod_name or 'hifigan' in mod_name:
                del sys.modules[mod_name]

    # FreeVC (subprocess, clean state)
    if not args.skip_freevc and os.path.exists(os.path.join(args.freevc_dir, 'checkpoints', 'freevc.pth')):
        try:
            all_results['freevc'] = benchmark_freevc(args, device)
        except Exception as e:
            print(f'  FreeVC benchmark failed: {e}', flush=True)

    # CRT (needs Mimi + CRT model)
    if not args.skip_crt and os.path.exists(args.crt_ckpt):
        try:
            all_results['crt'] = benchmark_crt(args, device)
        except Exception as e:
            print(f'  CRT benchmark failed: {e}', flush=True)
            import traceback
            traceback.print_exc()

    # ── Summary Table ──
    print(f'\n{"="*70}')
    print(f' REAL-TIME CAPABILITY SUMMARY')
    print(f'{"="*70}')
    print(f'{"Method":<25} {"Total (ms)":>12} {"RTF":>10} {"Speedup":>12} {"GPU MB":>10}')
    print(f'{"-"*70}')
    for key in ['crt', 'knnvc', 'freevc']:
        if key in all_results:
            r = all_results[key]
            total = r['stages']['total']['mean_ms']
            print(f'{r["method"]:<25} {total:>11.1f} {r["rtf"]:>10.4f} '
                  f'{1/r["rtf"]:>10.1f}x {r["peak_gpu_mb"]:>10.0f}')
    print(f'{"="*70}')

    # Save
    import json
    with open(os.path.join(args.output_dir, 'latency_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {args.output_dir}/latency_results.json')


if __name__ == '__main__':
    main()
