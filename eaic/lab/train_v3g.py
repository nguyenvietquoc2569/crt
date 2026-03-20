#!/usr/bin/env python3
"""
Mimi-VC V3g: Multi-Speaker Voice Conversion Training
V3f + autoregressive codebook conditioning: CB_i is conditioned on CB_1..CB_{i-1}
via teacher forcing during training and greedy-then-sample during inference.

Usage:
  python train_v3f.py --target_audio /path/to/target.wav --source_audio /path/to/source.wav
  python train_v3f.py --target_dir /path/to/target_wavs/ --source_audio /path/to/source.wav
  python train_v3f.py --phase pretrain  # pretrain only
  python train_v3f.py --phase finetune  # finetune only (requires pretrained checkpoint)
  python train_v3f.py --phase infer     # inference only (requires finetuned checkpoint)

Authors: Quoc V. Nguyen (UIT, VNU-HCM), Harry Nguyen (Kyndryl US), Lung Duc Vu (UIT, VNU-HCM)
"""

import argparse
import os
import sys
import io
import time
import math
import random
import gc
from collections import defaultdict, Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Slurm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ======================== CLI ========================

def parse_args():
    p = argparse.ArgumentParser(description="Mimi-VC V3g Training")
    p.add_argument('--data_raw', type=str, default='/datastore/eaic/nvquoc/mimi_vc/data_storage/librispeech',
                   help='Raw data cache directory (LibriSpeech download)')
    p.add_argument('--data_processed', type=str, default='/datastore/eaic/nvquoc/mimi_vc/data_storage/processed',
                   help='Processed data directory (encoded speakers, FAISS indices, etc.)')
    p.add_argument('--ckpt_dir', type=str, default='/datastore/eaic/nvquoc/mimi_vc/checkpoints_v3g',
                   help='Checkpoint directory')
    p.add_argument('--output_dir', type=str, default='/datastore/eaic/nvquoc/mimi_vc/output_v3g',
                   help='Output directory for results')
    p.add_argument('--target_audio', type=str, default=None,
                   help='Path to target speaker audio file')
    p.add_argument('--target_dir', type=str, default=None,
                   help='Path to directory of target speaker audio files')
    p.add_argument('--source_audio', type=str, default=None,
                   help='Path to source audio file for conversion')
    p.add_argument('--phase', type=str, default='all',
                   choices=['all', 'pretrain', 'finetune', 'infer'],
                   help='Which phase to run')
    p.add_argument('--prepare_only', action='store_true',
                   help='Download and preprocess data only, then exit (no training)')
    p.add_argument('--download_only', action='store_true',
                   help='Download LibriSpeech to data_raw only (no GPU needed, run on login node)')
    # Training hyperparams
    p.add_argument('--batch_size', type=int, default=48, help='Pre-training batch size')
    p.add_argument('--ft_batch', type=int, default=32, help='Fine-tuning batch size')
    p.add_argument('--pretrain_steps', type=int, default=30000)
    p.add_argument('--ft_steps', type=int, default=3000)
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal loss gamma (0=CE, 2=strong focus on rare tokens)')
    p.add_argument('--ft_focal_gamma', type=float, default=1.5,
                   help='Focal loss gamma for fine-tuning')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--ft_lr', type=float, default=5e-5)
    p.add_argument('--n_speakers', type=int, default=2338,
                   help='Number of speakers to encode (default: all 2338)')
    p.add_argument('--utts_per_speaker', type=int, default=10)
    p.add_argument('--use_train360', action='store_true', default=True,
                   help='Also load LibriSpeech train-clean-360 (921 more speakers)')
    p.add_argument('--use_train500', action='store_true', default=True,
                   help='Also load LibriSpeech train-other-500 (1166 more speakers)')
    # Model architecture
    p.add_argument('--d_model', type=int, default=768,
                   help='Transformer hidden dim (default 768; was 512)')
    p.add_argument('--n_heads', type=int, default=12,
                   help='Attention heads (default 12; was 8)')
    p.add_argument('--n_enc_layers', type=int, default=8,
                   help='Encoder layers (default 8; was 6)')
    p.add_argument('--n_dec_layers', type=int, default=8,
                   help='Decoder layers (default 8; was 6)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dropout', type=float, default=0.05,
                   help='Transformer dropout (default 0.05; was 0.15)')
    p.add_argument('--spk_noise_std', type=float, default=0.1,
                   help='Std of noise added to speaker embeddings (default 0.1; was 0.3)')
    p.add_argument('--cb0_noise_rate', type=float, default=0.30,
                   help='Fraction of CB0 tokens span-masked during training')
    return p.parse_args()


# ======================== Setup ========================

def setup(args):
    """Load Mimi model and configure environment."""
    os.makedirs(args.data_raw, exist_ok=True)
    os.makedirs(args.data_processed, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print(f'Loading Mimi weights (dtype={dtype})...', flush=True)
    t_load = time.time()
    try:
        mimi = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype)
        print(f'  weights loaded ({time.time()-t_load:.1f}s), moving to {device}...', flush=True)
        mimi = mimi.to(device).eval()
        print(f'  Mimi ready on {device} ({time.time()-t_load:.1f}s total)', flush=True)
    except Exception as e:
        print(f'  ERROR loading Mimi: {e}', flush=True)
        raise

    print('Loading feature extractor...', flush=True)
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    print('  feature extractor ready', flush=True)

    SR = feat_ext.sampling_rate
    N_CB = mimi.config.num_quantizers
    CB_SIZE = mimi.config.codebook_size
    CB_DIM = mimi.config.codebook_dim
    FRAME_RATE = mimi.config.frame_rate
    N_LEARN = 6

    print(f'Mimi config: SR={SR}, {N_CB} codebooks x {CB_SIZE}, dim={CB_DIM}, {FRAME_RATE}Hz', flush=True)

    cfg = {
        'device': device, 'mimi': mimi, 'feat_ext': feat_ext,
        'SR': SR, 'N_CB': N_CB, 'CB_SIZE': CB_SIZE, 'CB_DIM': CB_DIM,
        'FRAME_RATE': FRAME_RATE, 'N_LEARN': N_LEARN,
    }
    return cfg


# ======================== Core Functions ========================

def load_audio(path, target_sr=24000):
    wav, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    return wav.astype(np.float32)


def make_encode_fn(cfg):
    mimi, feat_ext, device, SR = cfg['mimi'], cfg['feat_ext'], cfg['device'], cfg['SR']
    mimi_dtype = next(mimi.parameters()).dtype
    def encode(wav_np):
        inputs = feat_ext(raw_audio=wav_np, sampling_rate=SR, return_tensors='pt')
        inputs = {k: v.to(device=device, dtype=mimi_dtype if v.is_floating_point() else None)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = mimi.encode(inputs['input_values'], inputs.get('padding_mask'))
        return out.audio_codes.cpu()
    return encode


def make_decode_fn(cfg):
    mimi, device = cfg['mimi'], cfg['device']
    def decode(codes):
        with torch.no_grad():
            out = mimi.decode(codes.to(device))
        return out.audio_values.cpu().squeeze().numpy().astype(np.float32)
    return decode


def get_codebook_embeddings(mimi, cb_idx=0):
    CB_SIZE = mimi.config.codebook_size
    q = mimi.quantizer
    if hasattr(q, 'semantic_residual_vector_quantizer'):
        n_sem = q.num_semantic_quantizers
        if cb_idx < n_sem:
            layer = q.semantic_residual_vector_quantizer.layers[cb_idx]
        else:
            layer = q.acoustic_residual_vector_quantizer.layers[cb_idx - n_sem]
    elif hasattr(q, 'layers'):
        layer = q.layers[cb_idx]
    else:
        raise RuntimeError(f'Unknown quantizer: {type(q).__name__}')
    for attr in ['codebook.embed', 'codebook.embed_sum', 'codebook.embedding.weight']:
        obj = layer
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            w = obj.detach().cpu().float()
            if w.dim() == 3: w = w.squeeze(0)
            if w.shape[0] >= CB_SIZE: return w
        except AttributeError:
            continue
    raise RuntimeError('Cannot find codebook embeddings')


# ======================== Data Preparation ========================

def prepare_data(cfg, args):
    """Load LibriSpeech, encode speakers, build FAISS indices, compute token frequencies.

    Downloads raw data to args.data_raw, saves processed data to args.data_processed.
    On subsequent runs, loads from cache if available.
    """
    import pickle
    from datasets import load_dataset

    encode = make_encode_fn(cfg)
    SR = cfg['SR']
    N_CB = cfg['N_CB']
    N_LEARN = cfg['N_LEARN']
    CB_SIZE = cfg['CB_SIZE']
    FRAME_RATE = cfg['FRAME_RATE']

    print('Extracting CB0 embeddings...', flush=True)
    cb0_emb = get_codebook_embeddings(cfg['mimi'], 0).numpy()
    print(f'CB0 embeddings: {cb0_emb.shape}', flush=True)

    # --- Check for processed cache ---
    cache_path = os.path.join(args.data_processed, 'prepared_data.pt')
    embeddings_cache = os.path.join(args.data_processed, 'speaker_embeddings.pkl')

    if os.path.exists(cache_path) and os.path.exists(embeddings_cache):
        print(f'\nLoading cached processed data from {args.data_processed}...')
        cached = torch.load(cache_path, weights_only=False, map_location='cpu')
        with open(embeddings_cache, 'rb') as f:
            emb_data = pickle.load(f)

        speaker_data = cached['speaker_data']
        cb_class_weights = cached['cb_class_weights']
        speaker_embeddings = emb_data['speaker_embeddings']

        # Rebuild FAISS indices (can't serialize them reliably)
        print('Rebuilding FAISS indices from cached codes...')
        K_FAISS = 8
        ALPHA_FAISS = 0.5
        speaker_faiss = {}
        for spk_id, sdata in speaker_data.items():
            codes = sdata['codes']
            sem_tokens = codes[0].numpy()
            full_codes = codes.numpy().T
            emb = cb0_emb[sem_tokens].astype(np.float32).copy()
            faiss.normalize_L2(emb)
            idx = faiss.IndexFlatIP(cb0_emb.shape[1])
            idx.add(emb)
            speaker_faiss[spk_id] = {'index': idx, 'full_codes': full_codes}

        print(f'Loaded {len(speaker_data)} speakers, {len(speaker_faiss)} FAISS indices')
        print(f'Class weights shape: {cb_class_weights.shape}')

        from resemblyzer import VoiceEncoder
        spk_enc = VoiceEncoder(device='cpu')  # avoid VRAM contention with other jobs

        # Load raw dataset reference (needed for ds in some phases)
        print('Loading LibriSpeech dataset reference...', flush=True)
        from datasets import Audio as HFAudio, concatenate_datasets
        _splits = []
        _splits.append(load_dataset('librispeech_asr', 'clean', split='train.100',
                                    trust_remote_code=True, cache_dir=args.data_raw))
        if args.use_train360:
            _splits.append(load_dataset('librispeech_asr', 'clean', split='train.360',
                                        trust_remote_code=True, cache_dir=args.data_raw))
        if args.use_train500:
            _splits.append(load_dataset('librispeech_asr', 'other', split='train.500',
                                        trust_remote_code=True, cache_dir=args.data_raw))
        ds = concatenate_datasets(_splits) if len(_splits) > 1 else _splits[0]
        ds = ds.cast_column('audio', HFAudio(decode=False))
        print(f'  Dataset: {len(ds)} utterances', flush=True)

        data = {
            'speaker_data': speaker_data,
            'speaker_faiss': speaker_faiss,
            'speaker_embeddings': speaker_embeddings,
            'cb0_emb': cb0_emb,
            'cb_class_weights': cb_class_weights,
            'spk_enc': spk_enc,
            'K_FAISS': K_FAISS,
            'ALPHA_FAISS': ALPHA_FAISS,
            'SPK_EMB_DIM': 256,
            'ds': ds,
        }
        return data

    def decode_audio(row):
        """Decode audio bytes with soundfile, avoiding torchcodec dependency."""
        audio = row['audio']
        raw = audio.get('bytes') or open(audio['path'], 'rb').read()
        wav, orig_sr = sf.read(io.BytesIO(raw))
        wav = np.squeeze(wav).astype(np.float32)
        return wav, orig_sr

    # --- Load LibriSpeech from data_raw cache ---
    from datasets import Audio as HFAudio, concatenate_datasets
    all_splits = []
    print(f'\nLoading LibriSpeech train-clean-100...', flush=True)
    ds100 = load_dataset('librispeech_asr', 'clean', split='train.100',
                         trust_remote_code=True, cache_dir=args.data_raw)
    print(f'  train-clean-100: {len(ds100)} utterances', flush=True)
    all_splits.append(ds100)
    if args.use_train360:
        print(f'Loading LibriSpeech train-clean-360...', flush=True)
        ds360 = load_dataset('librispeech_asr', 'clean', split='train.360',
                             trust_remote_code=True, cache_dir=args.data_raw)
        print(f'  train-clean-360: {len(ds360)} utterances', flush=True)
        all_splits.append(ds360)
    if args.use_train500:
        print(f'Loading LibriSpeech train-other-500...', flush=True)
        ds500 = load_dataset('librispeech_asr', 'other', split='train.500',
                             trust_remote_code=True, cache_dir=args.data_raw)
        print(f'  train-other-500: {len(ds500)} utterances', flush=True)
        all_splits.append(ds500)
    ds = concatenate_datasets(all_splits) if len(all_splits) > 1 else all_splits[0]
    print(f'  Total: {len(ds)} utterances', flush=True)
    ds = ds.cast_column('audio', HFAudio(decode=False))

    speaker_utts = defaultdict(list)
    for i, row in enumerate(ds):
        speaker_utts[row['speaker_id']].append(i)
    speakers = sorted(speaker_utts.keys())
    print(f'Total speakers: {len(speakers)}', flush=True)

    # --- Encode speakers (batched Mimi + prefetch audio in thread pool) ---
    MAX_AUDIO_SEC = 15
    PREFETCH = min(8, os.cpu_count() or 4)  # sliding window of prefetched speakers

    speakers_sorted = sorted(speakers, key=lambda s: len(speaker_utts[s]), reverse=True)
    selected_speakers = speakers_sorted[:args.n_speakers]

    def load_speaker_audio(spk_id):
        """Decode + resample all utterances for one speaker. CPU-only, thread-safe."""
        wavs = []
        for idx in speaker_utts[spk_id][:args.utts_per_speaker]:
            try:
                row = ds[idx]
                wav, orig_sr = decode_audio(row)
                if orig_sr != SR:
                    wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=SR)
                wav = wav[:int(MAX_AUDIO_SEC * SR)]
                if len(wav) >= SR:
                    wavs.append(wav)
            except Exception as e:
                print(f'    [warn] skip utt {idx}: {e}', flush=True)
        return wavs

    print(f'\nEncoding {len(selected_speakers)} speakers x {args.utts_per_speaker} utterances'
          f' (prefetch={PREFETCH} workers, I/O overlapped with GPU)...', flush=True)
    # Load Resemblyzer encoder once (CPU) — used inline during encoding loop
    from resemblyzer import VoiceEncoder
    print('Loading Resemblyzer speaker encoder...', flush=True)
    spk_enc = VoiceEncoder(device='cpu')
    SPK_EMB_DIM = 256

    # Resume from incremental checkpoint if available
    partial_cache = os.path.join(args.data_processed, 'prepared_data_partial.pt')
    speaker_data = {}
    speaker_embeddings = {}
    speaker_raw_audio = {}
    if os.path.exists(partial_cache):
        print(f'Resuming from partial checkpoint: {partial_cache}', flush=True)
        partial = torch.load(partial_cache, weights_only=False, map_location='cpu')
        speaker_data = partial['speaker_data']
        speaker_embeddings = partial.get('speaker_embeddings', {})
        print(f'  Loaded {len(speaker_data)} already-encoded speakers, '
              f'{len(speaker_embeddings)} embeddings', flush=True)
    done_set = set(speaker_data.keys())
    SAVE_EVERY = 100  # save partial checkpoint every N speakers

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        # Sliding window: keep PREFETCH futures ahead of current position
        future_queue = deque()
        next_submit = 0

        def submit_next():
            nonlocal next_submit
            if next_submit < len(selected_speakers):
                spk = selected_speakers[next_submit]
                future_queue.append((spk, pool.submit(load_speaker_audio, spk)))
                next_submit += 1

        # Fill initial window
        for _ in range(min(PREFETCH, len(selected_speakers))):
            submit_next()

        for si, spk_id in enumerate(selected_speakers):
            submit_next()  # keep window full
            _, fut = future_queue.popleft()

            # Skip already-encoded speakers (resume support)
            if spk_id in done_set:
                fut.cancel()
                continue

            wavs = fut.result()

            if not wavs:
                if (si + 1) % 10 == 0:
                    print(f'  {si+1}/{len(selected_speakers)} speakers encoded | {time.time()-t0:.0f}s', flush=True)
                continue

            # Encode each utterance individually — no padding overhead
            code_tensors = []
            valid_wavs = []
            for wav in wavs:
                codes = encode(wav)  # [1, N_CB, T]
                if codes.shape[2] > 0:
                    code_tensors.append(codes[0])  # [N_CB, T]
                    valid_wavs.append(wav)
            del wavs  # free prefetched audio immediately

            if not code_tensors:
                continue

            cat_codes = torch.cat(code_tensors, dim=1)
            del code_tensors  # free per-utterance tensors

            speaker_data[spk_id] = {'codes': cat_codes}
            raw_audio = np.concatenate(valid_wavs)
            del valid_wavs  # free raw wav list

            # Resemblyzer embedding — compute while audio is in memory, then free
            wav16k = librosa.resample(raw_audio, orig_sr=SR, target_sr=16000)
            del raw_audio
            speaker_embeddings[spk_id] = spk_enc.embed_utterance(wav16k)
            del wav16k

            if (si + 1) % 10 == 0:
                print(f'  {si+1}/{len(selected_speakers)} speakers encoded | {time.time()-t0:.0f}s', flush=True)

            # Incremental save every SAVE_EVERY speakers
            if len(speaker_data) % SAVE_EVERY == 0:
                print(f'  [checkpoint] saving {len(speaker_data)} speakers to {partial_cache}...', flush=True)
                torch.save({'speaker_data': speaker_data,
                            'speaker_embeddings': speaker_embeddings}, partial_cache)
                gc.collect()

    total_frames_all = sum(v['codes'].shape[1] for v in speaker_data.values())
    print(f'\nDone: {len(speaker_data)} speakers, {total_frames_all} frames '
          f'({total_frames_all/FRAME_RATE:.0f}s) in {time.time()-t0:.0f}s')

    # --- Build per-speaker FAISS indices ---
    print('\nBuilding per-speaker FAISS indices...')
    K_FAISS = 8
    ALPHA_FAISS = 0.5
    speaker_faiss = {}

    for spk_id, sdata in speaker_data.items():
        codes = sdata['codes']
        sem_tokens = codes[0].numpy()
        full_codes = codes.numpy().T

        emb = cb0_emb[sem_tokens].astype(np.float32).copy()
        faiss.normalize_L2(emb)
        idx = faiss.IndexFlatIP(cb0_emb.shape[1])
        idx.add(emb)

        speaker_faiss[spk_id] = {
            'index': idx,
            'full_codes': full_codes,
        }

    print(f'Built FAISS indices for {len(speaker_faiss)} speakers')

    # --- Compute per-codebook token frequencies for focal loss class weights ---
    print('\nComputing token frequencies for class-balanced focal loss...')
    cb_token_counts = torch.zeros(N_LEARN, CB_SIZE, dtype=torch.long)
    for spk_id, sdata in speaker_data.items():
        codes = sdata['codes']
        for cb_i in range(N_LEARN):
            tokens = codes[1 + cb_i]
            cb_token_counts[cb_i] += torch.bincount(tokens, minlength=CB_SIZE)

    cb_class_weights = []
    for cb_i in range(N_LEARN):
        freq = cb_token_counts[cb_i].float()
        freq = freq / freq.sum()
        w = 1.0 / (freq.sqrt() + 1e-6)
        w = w / w.mean()
        w = w.clamp(max=20.0)
        cb_class_weights.append(w)
        used = (cb_token_counts[cb_i] > 0).sum().item()
        print(f'  CB{cb_i+1}: {used} tokens appear in training data')

    cb_class_weights = torch.stack(cb_class_weights)
    print(f'Class weights shape: {cb_class_weights.shape}')

    speaker_ids = sorted(speaker_data.keys())
    spk_emb_matrix = np.stack([speaker_embeddings[s] for s in speaker_ids])
    print(f'Speaker embeddings: {spk_emb_matrix.shape}')

    # --- Save final processed data ---
    print(f'\nSaving processed data to {args.data_processed}...')
    torch.save({
        'speaker_data': speaker_data,
        'cb_class_weights': cb_class_weights,
    }, cache_path)
    # Remove partial checkpoint now that final save succeeded
    if os.path.exists(partial_cache):
        os.remove(partial_cache)
        print(f'Removed partial checkpoint: {partial_cache}', flush=True)

    with open(embeddings_cache, 'wb') as f:
        pickle.dump({'speaker_embeddings': speaker_embeddings}, f)

    print(f'Saved: {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)')
    print(f'Saved: {embeddings_cache} ({os.path.getsize(embeddings_cache)/1e6:.1f} MB)')

    del speaker_raw_audio
    gc.collect()

    data = {
        'speaker_data': speaker_data,
        'speaker_faiss': speaker_faiss,
        'speaker_embeddings': speaker_embeddings,
        'cb0_emb': cb0_emb,
        'cb_class_weights': cb_class_weights,
        'spk_enc': spk_enc,
        'K_FAISS': K_FAISS,
        'ALPHA_FAISS': ALPHA_FAISS,
        'SPK_EMB_DIM': SPK_EMB_DIM,
        'ds': ds,
    }
    return data


# ======================== Model ========================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        T = x.shape[2]
        return self.cos_cached[:, :, :T, :], self.sin_cached[:, :, :T, :]


def apply_rope(q, k, cos, sin):
    def rotate(x, c, s):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)
    return rotate(q, cos, sin), rotate(k, cos, sin)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x, B, T):
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self._reshape(self.q_proj(h), B, T)
        k = self._reshape(self.k_proj(h), B, T)
        v = self._reshape(self.v_proj(h), B, T)
        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.dropout(self.o_proj(out))
        x = x + self.ffn(self.norm2(x))
        return x


class FiLMLayer(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class AcousticTranslatorV3g(nn.Module):
    """
    V3g: V3f + autoregressive codebook conditioning.

    Key fix over V3f: each codebook CB_i is conditioned on CB_1..CB_{i-1} via
    teacher forcing during training. This respects Mimi's RVQ structure where
    CB_{i+1} encodes the residual after CB_i. In V3f all CBs were predicted
    independently from the same CB0 input, causing a plateau at ~5-10% accuracy.

    Architecture:
      - CB1: decoder(sem_enc + cb_type_1) + FiLM(spk)
      - CB2: decoder(sem_enc + acou_embed(CB1) + cb_type_2) + FiLM(spk)
      - CB3: decoder(sem_enc + acou_embed(CB1) + acou_embed(CB2) + cb_type_3) + FiLM(spk)
      - ...
    Teacher forcing during training, greedy AR during inference.
    """
    def __init__(self, n_vocab=2048, n_cb=6, d_model=512, n_heads=8,
                 n_enc_layers=6, n_dec_layers=6, spk_emb_dim=256,
                 dropout=0.15, cb0_noise_rate=0.30, spk_noise_std=0.3):
        super().__init__()
        self.n_cb = n_cb
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.cb0_noise_rate = cb0_noise_rate
        self.spk_noise_std = spk_noise_std
        ff_dim = d_model * 4

        self.sem_embed = nn.Embedding(n_vocab, d_model)
        self.cb_embed = nn.Embedding(n_cb, d_model)
        # NEW: one embedding table per CB to embed previous CB's tokens as conditioning
        self.acou_embeds = nn.ModuleList([
            nn.Embedding(n_vocab, d_model) for _ in range(n_cb)
        ])

        self.spk_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.film_layers = nn.ModuleList([
            FiLMLayer(d_model, d_model) for _ in range(n_dec_layers)
        ])

        self.encoder = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.decoder = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, n_vocab) for _ in range(n_cb)
        ])

    def _augment_cb0(self, sem_tokens):
        if not self.training or self.cb0_noise_rate <= 0:
            return sem_tokens
        B, T = sem_tokens.shape
        result = sem_tokens.clone()
        target_corrupt = int(T * self.cb0_noise_rate)
        for b in range(B):
            corrupted = 0
            while corrupted < target_corrupt:
                span_len = random.randint(3, min(10, target_corrupt - corrupted + 3))
                span_len = min(span_len, T)
                start = random.randint(0, T - span_len)
                result[b, start:start+span_len] = torch.randint(
                    0, self.n_vocab, (span_len,), device=sem_tokens.device)
                corrupted += span_len
        return result

    def _augment_spk(self, spk_emb):
        if not self.training or self.spk_noise_std <= 0:
            return spk_emb
        return spk_emb + torch.randn_like(spk_emb) * self.spk_noise_std

    def encode_semantic(self, sem_tokens):
        sem_tokens = self._augment_cb0(sem_tokens)
        x = self.sem_embed(sem_tokens)
        for layer in self.encoder:
            x = layer(x)
        return self.enc_norm(x)

    def forward(self, sem_tokens, tgt_spk_emb, prev_acoustic=None):
        """
        sem_tokens:   [B, T]        — CB0 semantic tokens
        tgt_spk_emb:  [B, spk_dim] — target speaker embedding
        prev_acoustic:[B, n_cb, T] — teacher-forced acoustic tokens for AR conditioning
                                     (use ground-truth during training, None for inference)

        Each CB_i is conditioned on CB_0..CB_{i-1} via accumulated acoustic embeddings.
        This respects Mimi's RVQ structure where CB_{i+1} encodes the residual of CB_i.
        """
        B, T = sem_tokens.shape
        sem_enc = self.encode_semantic(sem_tokens)          # [B, T, D]
        spk_cond = self.spk_proj(self._augment_spk(tgt_spk_emb))  # [B, D]

        # h_in accumulates: sem_enc + sum of previous CBs' acoustic embeddings
        h_in = sem_enc  # [B, T, D]
        all_logits = []

        for cb_i in range(self.n_cb):
            # CB type indicator + accumulated acoustic context
            cb_id = torch.full((B,), cb_i, dtype=torch.long, device=sem_tokens.device)
            h = h_in + self.cb_embed(cb_id).unsqueeze(1)   # [B, T, D]

            for layer, film in zip(self.decoder, self.film_layers):
                h = layer(h)
                h = film(h, spk_cond)
            h = self.dec_norm(h)

            logits_i = self.output_heads[cb_i](h)          # [B, T, V]
            all_logits.append(logits_i)

            # Feed current CB tokens into next CB's input context
            if cb_i < self.n_cb - 1:
                if prev_acoustic is not None:
                    # Teacher forcing: use ground-truth tokens
                    cb_tokens = prev_acoustic[:, cb_i]      # [B, T]
                else:
                    # Inference: greedy from current logits
                    cb_tokens = logits_i.detach().argmax(-1)  # [B, T]
                h_in = h_in + self.acou_embeds[cb_i](cb_tokens)  # [B, T, D]

        return torch.stack(all_logits, dim=1)  # [B, n_cb, T, V]

    def compute_loss(self, logits, target_tokens, cb_weights, class_weights=None, focal_gamma=2.0):
        """Focal loss with class-frequency weighting.

        Focal: -alpha_t * (1 - p_t)^gamma * log(p_t)
        - Common tokens (p_t~0.9): weight ~ 0.01 -> tiny gradient
        - Rare tokens (p_t~0.01): weight ~ 0.98 -> large gradient
        - Forces model to learn the long tail of the token distribution
        """
        B, n_cb, T, V = logits.shape
        total_loss = 0

        for cb_i in range(n_cb):
            cb_logits = logits[:, cb_i].reshape(-1, V)
            cb_targets = target_tokens[:, cb_i].reshape(-1)

            log_probs = F.log_softmax(cb_logits.float(), dim=-1)
            target_log_probs = log_probs.gather(1, cb_targets.unsqueeze(1)).squeeze(1)
            target_probs = target_log_probs.exp()

            # Focal weight (detached — standard practice)
            focal_weight = (1.0 - target_probs.detach()) ** focal_gamma

            # Class frequency weight
            if class_weights is not None:
                alpha_t = class_weights[cb_i].to(cb_logits.device)
                alpha_t = alpha_t.gather(0, cb_targets)
                focal_weight = focal_weight * alpha_t

            loss = -(focal_weight * target_log_probs).mean()
            total_loss += cb_weights[cb_i] * loss

        return total_loss / n_cb

    @torch.no_grad()
    def generate(self, sem_tokens, tgt_spk_emb, temperature=1.0, top_p=0.92):
        """AR nucleus sampling: CB_i sampled, then used to condition CB_{i+1}.

        Nucleus sampling keeps the smallest set of tokens whose cumulative
        probability exceeds top_p. Preserves diversity in the distribution tail.
        AR conditioning: sampled CB_i tokens feed into CB_{i+1}'s input context.
        """
        self.eval()
        B, T = sem_tokens.shape
        sem_enc = self.encode_semantic(sem_tokens)
        spk_cond = self.spk_proj(tgt_spk_emb)

        h_in = sem_enc  # [B, T, D]
        tokens = torch.zeros(B, self.n_cb, T, dtype=torch.long, device=sem_tokens.device)

        def nucleus_sample(logits_i):
            # logits_i: [B, T, V]
            logits_scaled = logits_i / temperature
            probs = F.softmax(logits_scaled.float(), dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum_probs - sorted_probs) >= top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Sample position by position (multinomial over vocab)
            flat_probs = sorted_probs.reshape(-1, sorted_probs.shape[-1])
            flat_sampled = torch.multinomial(flat_probs, 1).squeeze(-1)  # [B*T]
            flat_indices = sorted_indices.reshape(-1, sorted_indices.shape[-1])
            sampled_tokens = flat_indices.gather(1, flat_sampled.unsqueeze(1)).squeeze(1)
            return sampled_tokens.reshape(B, T)  # [B, T]

        for cb_i in range(self.n_cb):
            cb_id = torch.full((B,), cb_i, dtype=torch.long, device=sem_tokens.device)
            h = h_in + self.cb_embed(cb_id).unsqueeze(1)

            for layer, film in zip(self.decoder, self.film_layers):
                h = layer(h)
                h = film(h, spk_cond)
            h = self.dec_norm(h)

            logits_i = self.output_heads[cb_i](h)           # [B, T, V]
            sampled_i = nucleus_sample(logits_i)             # [B, T]
            tokens[:, cb_i] = sampled_i

            if cb_i < self.n_cb - 1:
                h_in = h_in + self.acou_embeds[cb_i](sampled_i)

        return tokens


# ======================== Pre-training ========================

def pretrain(cfg, data, args):
    """Mixed pre-training: self-reconstruction + cross-speaker FAISS distillation."""
    device = cfg['device']
    N_LEARN = cfg['N_LEARN']
    CB_SIZE = cfg['CB_SIZE']

    D_MODEL = args.d_model
    N_HEADS = args.n_heads
    SEQ_LEN = 128
    SAVE_EVERY = 5000
    CROSS_SPEAKER_RATIO_MAX = 0.20   # V3g: 20% cross-speaker (was 50% in V3f)
    CROSS_SPEAKER_WARMUP = 5000      # ramp from 0 → 0.20 over first 5k steps

    cb_weights = torch.tensor([2.0**(-(i)/2) for i in range(N_LEARN)], device=device)
    cb_weights = cb_weights / cb_weights.sum() * N_LEARN

    print(f'\n{"="*70}')
    print(f'PRE-TRAINING V3g (focal loss + AR codebook conditioning)')
    print(f'{"="*70}')
    print(f'Steps: {args.pretrain_steps}, Batch: {args.batch_size}, Seq: {SEQ_LEN}')
    print(f'Model: d_model={D_MODEL}, n_heads={N_HEADS}, enc={args.n_enc_layers}, dec={args.n_dec_layers}')
    print(f'Cross-speaker ratio: 0→{CROSS_SPEAKER_RATIO_MAX*100:.0f}% over {CROSS_SPEAKER_WARMUP} steps')
    print(f'LOSS: focal loss (gamma={args.focal_gamma}) + class-frequency weighting')
    print(f'AUGMENTATION: CB0 span masking {args.cb0_noise_rate*100:.0f}%, speaker noise std={args.spk_noise_std}')
    print(f'REGULARIZATION: dropout={args.dropout}')

    model = AcousticTranslatorV3g(
        n_vocab=CB_SIZE, n_cb=N_LEARN, d_model=D_MODEL, n_heads=N_HEADS,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        spk_emb_dim=data['SPK_EMB_DIM'],
        dropout=args.dropout, cb0_noise_rate=args.cb0_noise_rate, spk_noise_std=args.spk_noise_std
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_params/1e6:.1f}M', flush=True)

    use_compile = hasattr(torch, 'compile') and device == 'cuda'
    if use_compile:
        try:
            model = torch.compile(model)
            print('torch.compile: ENABLED')
        except Exception:
            use_compile = False

    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.float16

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < 1000:
            return step / 1000
        progress = (step - 1000) / (args.pretrain_steps - 1000)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Resume
    start_step = 0
    resume_path = os.path.join(args.ckpt_dir, 'v3g_pretrain_latest.pt')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, weights_only=False, map_location=device)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        clean = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()}
        raw_model.load_state_dict(clean)
        opt.load_state_dict(ckpt['opt_state'])
        start_step = ckpt['step']
        for _ in range(start_step):
            sched.step()
        print(f'\n>> RESUMED from step {start_step}')
        if start_step >= args.pretrain_steps:
            print(f'   Pre-training already complete.')
            return model, D_MODEL, N_HEADS, cb_weights

    # Data accessors
    speaker_data = data['speaker_data']
    speaker_faiss = data['speaker_faiss']
    speaker_embeddings = data['speaker_embeddings']
    cb0_emb = data['cb0_emb']
    cb_class_weights = data['cb_class_weights'].to(device)
    K_FAISS = data['K_FAISS']
    ALPHA_FAISS = data['ALPHA_FAISS']
    spk_list = list(speaker_data.keys())

    def faiss_retrieve_tokens(src_sem_tokens, tgt_spk_id):
        T = len(src_sem_tokens)
        sf = speaker_faiss[tgt_spk_id]
        src_emb = cb0_emb[src_sem_tokens].astype(np.float32).copy()
        faiss.normalize_L2(src_emb)
        dists_k, indices_k = sf['index'].search(src_emb, K_FAISS)

        selected = np.zeros(T, dtype=np.int64)
        selected[0] = indices_k[0, 0]
        for t in range(1, T):
            cands = indices_k[t]
            combined = ALPHA_FAISS * dists_k[t] + (1 - ALPHA_FAISS) / (
                1 + np.abs(cands - selected[t-1]).astype(np.float32))
            selected[t] = cands[np.argmax(combined)]

        full_codes = sf['full_codes']
        retrieved = full_codes[selected]
        return torch.tensor(retrieved[:, 1:1+N_LEARN].T, dtype=torch.long)

    def sample_mixed_batch(batch_size, seq_len, cross_speaker_ratio):
        n_cross = int(batch_size * cross_speaker_ratio)
        n_self = batch_size - n_cross

        sem_batch, tgt_acou_batch, spk_batch = [], [], []

        for _ in range(n_self):
            spk = random.choice(spk_list)
            codes = speaker_data[spk]['codes']
            T = codes.shape[1]
            start = random.randint(0, max(0, T - seq_len))
            crop = codes[:, start:start+seq_len]
            if crop.shape[1] < seq_len:
                crop = F.pad(crop, (0, seq_len - crop.shape[1]))
            sem_batch.append(crop[0])
            tgt_acou_batch.append(crop[1:1+N_LEARN])
            spk_batch.append(torch.tensor(speaker_embeddings[spk], dtype=torch.float32))

        for _ in range(n_cross):
            src_spk = random.choice(spk_list)
            tgt_spk = random.choice(spk_list)
            while tgt_spk == src_spk:
                tgt_spk = random.choice(spk_list)

            codes = speaker_data[src_spk]['codes']
            T = codes.shape[1]
            start = random.randint(0, max(0, T - seq_len))
            crop = codes[:, start:start+seq_len]
            if crop.shape[1] < seq_len:
                crop = F.pad(crop, (0, seq_len - crop.shape[1]))

            sem_tokens = crop[0]
            pseudo_acou = faiss_retrieve_tokens(sem_tokens.numpy(), tgt_spk)

            sem_batch.append(sem_tokens)
            tgt_acou_batch.append(pseudo_acou)
            spk_batch.append(torch.tensor(speaker_embeddings[tgt_spk], dtype=torch.float32))

        return (
            torch.stack(sem_batch).to(device),
            torch.stack(tgt_acou_batch).to(device),
            torch.stack(spk_batch).to(device),
        )

    # Training loop
    all_losses = []
    per_cb_accs = {i: [] for i in range(N_LEARN)}
    model.train()
    t0 = time.time()
    log_interval = 100
    best_loss = float('inf')

    best_path = os.path.join(args.ckpt_dir, 'v3g_pretrain_best.pt')
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, weights_only=False, map_location='cpu')
        best_loss = best_ckpt.get('loss', float('inf'))
        del best_ckpt
        print(f'Existing best loss: {best_loss:.4f}')

    for step in range(start_step, args.pretrain_steps):
        # Linearly ramp cross-speaker ratio from 0 → max over warmup steps
        cross_ratio = min(CROSS_SPEAKER_RATIO_MAX,
                          CROSS_SPEAKER_RATIO_MAX * step / max(1, CROSS_SPEAKER_WARMUP))
        sem_b, tgt_acou_b, spk_b = sample_mixed_batch(
            args.batch_size, SEQ_LEN, cross_ratio)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Teacher forcing: pass ground-truth acoustic tokens so CB_i sees CB_1..CB_{i-1}
            logits = model(sem_b, spk_b, prev_acoustic=tgt_acou_b)
            loss = model.compute_loss(logits, tgt_acou_b, cb_weights,
                                      cb_class_weights, args.focal_gamma)

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                for cb_i in range(N_LEARN):
                    pred_tokens = logits[:, cb_i].float().argmax(-1).reshape(-1)
                    gt_tokens = tgt_acou_b[:, cb_i].reshape(-1)
                    acc = (pred_tokens == gt_tokens).float().mean().item()
                    per_cb_accs[cb_i].append(acc * 100)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        all_losses.append(loss.item())

        if (step + 1) % log_interval == 0:
            cb_str = ' '.join([f'CB{i+1}:{per_cb_accs[i][-1]:.0f}%' for i in range(N_LEARN)])
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            sps = steps_done / elapsed
            eta = (args.pretrain_steps - step - 1) / sps
            print(f'  Step {step+1:5d}/{args.pretrain_steps} | Loss:{loss.item():.4f} | '
                  f'{cb_str} | {sps:.1f} step/s | ETA {eta:.0f}s', flush=True)

        if (step + 1) % SAVE_EVERY == 0:
            avg_loss = np.mean(all_losses[-SAVE_EVERY:])
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            ckpt_data = {
                'step': step + 1,
                'model_state': raw_model.state_dict(),
                'opt_state': opt.state_dict(),
                'loss': avg_loss,
            }
            torch.save(ckpt_data, os.path.join(args.ckpt_dir, 'v3g_pretrain_latest.pt'))
            tag = ''
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt_data, best_path)
                tag = ' ** BEST'
            print(f'  >> Saved step {step+1} (avg loss: {avg_loss:.4f}){tag}', flush=True)

    pretrain_time = time.time() - t0

    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, weights_only=False, map_location=device)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_model.load_state_dict(best_ckpt['model_state'])
        print(f'\nLoaded best checkpoint (loss: {best_loss:.4f})')
    model.eval()

    # Plot
    if all_losses:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
        axes[0].plot(all_losses, linewidth=0.5, color='#7030A0')
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Pre-training ({args.pretrain_steps} steps, {pretrain_time/60:.0f}min)')
        axes[0].grid(True, alpha=0.3)
        colors = plt.cm.tab10(np.linspace(0, 1, N_LEARN))
        for i in range(N_LEARN):
            axes[1].plot(per_cb_accs[i], label=f'CB{i+1}', color=colors[i])
        axes[1].set_xlabel('Log point'); axes[1].set_ylabel('Accuracy %')
        axes[1].set_title('Per-CB Accuracy')
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, 'v3g_pretrain_curves.png'), dpi=150)
        plt.close()
        print(f'Pre-training done: {pretrain_time:.0f}s ({pretrain_time/60:.1f}min)')

    return model, D_MODEL, N_HEADS, cb_weights


# ======================== Target Data ========================

def load_target(cfg, data, args):
    """Load and encode target speaker audio."""
    encode = make_encode_fn(cfg)
    SR = cfg['SR']
    FRAME_RATE = cfg['FRAME_RATE']
    N_LEARN = cfg['N_LEARN']
    cb0_emb = data['cb0_emb']

    # Collect target audio files
    target_wavs = []
    if args.target_audio:
        wav = load_audio(args.target_audio, SR)
        print(f'Target: {args.target_audio} ({len(wav)/SR:.1f}s)')
        target_wavs.append(wav)
    elif args.target_dir:
        for f in sorted(Path(args.target_dir).glob('*.wav')):
            wav = load_audio(str(f), SR)
            print(f'Target: {f.name} ({len(wav)/SR:.1f}s)')
            target_wavs.append(wav)
        # Also try .flac, .mp3
        for ext in ['*.flac', '*.mp3']:
            for f in sorted(Path(args.target_dir).glob(ext)):
                wav = load_audio(str(f), SR)
                print(f'Target: {f.name} ({len(wav)/SR:.1f}s)')
                target_wavs.append(wav)
    else:
        raise ValueError('Must provide --target_audio or --target_dir')

    target_wav_concat = np.concatenate(target_wavs)
    print(f'Total target: {len(target_wav_concat)/SR:.1f}s')

    # Encode
    CHUNK_SEC = 30
    chunk_samples = int(CHUNK_SEC * SR)
    tgt_chunks = []
    for wav in target_wavs:
        for i in range(0, len(wav), chunk_samples):
            chunk = wav[i:i+chunk_samples]
            if len(chunk) > SR:
                tgt_chunks.append(chunk)

    tgt_codes_list = [encode(c) for c in tgt_chunks]
    all_tgt_codes = torch.cat(tgt_codes_list, dim=2)
    T_tgt = all_tgt_codes.shape[2]
    print(f'Target: {T_tgt} frames ({T_tgt/FRAME_RATE:.1f}s)')

    # Speaker embedding
    tgt_wav_16k = librosa.resample(target_wav_concat, orig_sr=SR, target_sr=16000)
    tgt_spk_emb = data['spk_enc'].embed_utterance(tgt_wav_16k)
    tgt_spk_emb_t = torch.tensor(tgt_spk_emb, dtype=torch.float32).to(cfg['device'])

    # FAISS index for retrieval baseline + remaining CBs
    all_sem = all_tgt_codes[0, 0].numpy()
    all_full = all_tgt_codes[0].numpy().T
    tgt_emb = cb0_emb[all_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb)
    faiss_index = faiss.IndexFlatIP(cb0_emb.shape[1])
    faiss_index.add(tgt_emb)
    print(f'Target FAISS index: {faiss_index.ntotal} vectors')

    return {
        'all_tgt_codes': all_tgt_codes,
        'T_tgt': T_tgt,
        'tgt_spk_emb': tgt_spk_emb,
        'tgt_spk_emb_t': tgt_spk_emb_t,
        'target_wav_concat': target_wav_concat,
        'faiss_index': faiss_index,
        'all_full': all_full,
    }


# ======================== Fine-tuning ========================

def finetune(cfg, data, target, model, D_MODEL, N_HEADS, cb_weights, args):
    """Fine-tune on target speaker with self-reconstruction."""
    device = cfg['device']
    N_LEARN = cfg['N_LEARN']
    CB_SIZE = cfg['CB_SIZE']
    SPK_EMB_DIM = data['SPK_EMB_DIM']
    SEQ_LEN = 128

    all_tgt_codes = target['all_tgt_codes']
    T_tgt = target['T_tgt']
    tgt_spk_emb = target['tgt_spk_emb']
    tgt_spk_emb_t = target['tgt_spk_emb_t']
    cb_class_weights = data['cb_class_weights'].to(device)

    print(f'\n{"="*70}')
    print(f'FINE-TUNING V3g ON TARGET SPEAKER')
    print(f'{"="*70}')
    print(f'Target data: {T_tgt} frames ({T_tgt/cfg["FRAME_RATE"]:.1f}s)')
    print(f'Steps: {args.ft_steps}, Batch: {args.ft_batch}, LR: {args.ft_lr}')
    print(f'Loss: focal loss (gamma={args.ft_focal_gamma}) + class-frequency weighting')

    # Check existing checkpoint
    ft_ckpt_path = os.path.join(args.ckpt_dir, 'v3g_finetuned.pt')
    if os.path.exists(ft_ckpt_path):
        ft_ckpt = torch.load(ft_ckpt_path, weights_only=False, map_location=device)
        saved_emb = ft_ckpt.get('tgt_spk_emb')
        if saved_emb is not None and np.allclose(saved_emb, tgt_spk_emb, atol=1e-5):
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            clean = {k.replace('_orig_mod.', ''): v for k, v in ft_ckpt['model_state'].items()}
            raw_model.load_state_dict(clean)
            model.eval()
            print(f'Loaded fine-tuned checkpoint (same target). Skipping.')
            del ft_ckpt
            return model
        else:
            print(f'Checkpoint exists but for different target. Retraining...')
        del ft_ckpt

    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.float16

    ft_sem = all_tgt_codes[0, 0]
    ft_acou = all_tgt_codes[0, 1:1+N_LEARN]

    ft_opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=0.01)
    ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, args.ft_steps)

    model.train()
    t0 = time.time()
    ft_log = max(1, min(50, args.ft_steps // 20))
    ft_losses = []
    ft_accs = {i: [] for i in range(N_LEARN)}

    for step in range(args.ft_steps):
        starts = torch.randint(0, max(1, T_tgt - SEQ_LEN), (args.ft_batch,))
        sem_b = torch.stack([ft_sem[s:s+SEQ_LEN] for s in starts]).to(device)
        acou_b = torch.stack([ft_acou[:, s:s+SEQ_LEN] for s in starts]).to(device)
        spk_b = tgt_spk_emb_t.unsqueeze(0).expand(args.ft_batch, -1)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Teacher forcing: CB_i sees GT CB_1..CB_{i-1}
            logits = model(sem_b, spk_b, prev_acoustic=acou_b)
            loss = model.compute_loss(logits, acou_b, cb_weights,
                                      cb_class_weights, args.ft_focal_gamma)

        if (step + 1) % ft_log == 0:
            with torch.no_grad():
                for cb_i in range(N_LEARN):
                    pred_tokens = logits[:, cb_i].float().argmax(-1).reshape(-1)
                    gt_tokens = acou_b[:, cb_i].reshape(-1)
                    acc = (pred_tokens == gt_tokens).float().mean().item()
                    ft_accs[cb_i].append(acc * 100)

        ft_opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(ft_opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(ft_opt)
        scaler.update()
        ft_sched.step()
        ft_losses.append(loss.item())

        if (step + 1) % ft_log == 0:
            cb_str = ' '.join([f'CB{i+1}:{ft_accs[i][-1]:.0f}%' for i in range(N_LEARN)])
            print(f'  FT {step+1:4d}/{args.ft_steps} | Loss:{loss.item():.4f} | '
                  f'{cb_str} | {time.time()-t0:.0f}s', flush=True)

    ft_time = time.time() - t0
    model.eval()

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'model_state': raw_model.state_dict(),
        'config': {'n_vocab': CB_SIZE, 'n_cb': N_LEARN, 'd_model': D_MODEL,
                   'n_heads': N_HEADS, 'spk_emb_dim': SPK_EMB_DIM},
        'tgt_spk_emb': tgt_spk_emb,
    }, ft_ckpt_path)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ft_losses, linewidth=0.8, color='#C55A11')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title(f'Fine-tuning ({args.ft_steps} steps, {ft_time:.0f}s)')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'v3g_finetune_curve.png'), dpi=150)
    plt.close()

    print(f'\nFine-tuning done ({ft_time:.0f}s)')
    for i in range(N_LEARN):
        print(f'  CB{i+1}: {ft_accs[i][-1]:.1f}%')

    return model


# ======================== Inference + Diagnostics ========================

def inference(cfg, data, target, model, args):
    """Run voice conversion and print diagnostics."""
    device = cfg['device']
    SR = cfg['SR']
    N_CB = cfg['N_CB']
    N_LEARN = cfg['N_LEARN']
    cb0_emb = data['cb0_emb']
    decode = make_decode_fn(cfg)
    encode = make_encode_fn(cfg)

    tgt_spk_emb_t = target['tgt_spk_emb_t']
    faiss_index = target['faiss_index']
    all_full = target['all_full']
    target_wav_concat = target['target_wav_concat']

    if args.source_audio is None:
        print('No --source_audio provided. Skipping inference.')
        return

    # Load source
    source_wav = load_audio(args.source_audio, SR)
    print(f'\nSource: {args.source_audio} ({len(source_wav)/SR:.1f}s)')
    src_codes = encode(source_wav)
    T_src = src_codes.shape[2]
    print(f'Source: {T_src} frames')

    print(f'\n{"="*70}')
    print(f'VOICE CONVERSION - V3g')
    print(f'{"="*70}')

    # 1. FAISS+Smooth baseline
    print('\n1. FAISS+Smooth (retrieval baseline)...')
    src_sem_np = src_codes[0, 0].numpy()
    src_emb_q = cb0_emb[src_sem_np].astype(np.float32).copy()
    faiss.normalize_L2(src_emb_q)
    K = 8; ALPHA = 0.5
    dists_k, indices_k = faiss_index.search(src_emb_q, K)

    selected = np.zeros(T_src, dtype=np.int64)
    selected[0] = indices_k[0, 0]
    for t in range(1, T_src):
        cands = indices_k[t]
        combined = ALPHA * dists_k[t] + (1-ALPHA) / (
            1 + np.abs(cands - selected[t-1]).astype(np.float32))
        selected[t] = cands[np.argmax(combined)]

    ret_codes = src_codes.clone()
    for t in range(T_src):
        ret_codes[0, :, t] = torch.tensor(all_full[selected[t]])
    ret_wav = decode(ret_codes)

    # 2. V3f with different top_p values
    print('\n2. V3g nucleus sampling...')
    src_sem_d = src_codes[0, 0:1].to(device)
    spk_d = tgt_spk_emb_t.unsqueeze(0)

    v3_results = {}
    for top_p in [0.85, 0.92, 0.98]:
        torch.manual_seed(42)
        pred = model.generate(src_sem_d, spk_d, temperature=1.0, top_p=top_p)
        vc = src_codes.clone()
        vc[0, 1:1+N_LEARN] = pred[0].cpu()
        for t in range(T_src):
            vc[0, 1+N_LEARN:, t] = torch.tensor(all_full[selected[t]][1+N_LEARN:])
        w = decode(vc)
        v3_results[top_p] = (vc, w)
        print(f'  V3g top_p={top_p}: done')

    # 3. V3f + keep source CB0
    print('\n3. V3g (keep source CB0)...')
    v3_keepcb0 = src_codes.clone()
    v3_keepcb0[0, 1:1+N_LEARN] = v3_results[0.92][0][0, 1:1+N_LEARN]
    for t in range(T_src):
        v3_keepcb0[0, 1+N_LEARN:, t] = torch.tensor(all_full[selected[t]][1+N_LEARN:])
    v3_keepcb0_wav = decode(v3_keepcb0)

    # === DIAGNOSTICS ===
    print(f'\n{"="*70}')
    print(f'DIAGNOSTICS - V3g')
    print(f'{"="*70}')

    # Token distribution
    print('\n--- Token distribution: V3g on source audio ---')
    v3_pred_codes = v3_results[0.92][0][0, 1:1+N_LEARN, :].numpy()
    for cb_i in range(N_LEARN):
        pred_tokens = v3_pred_codes[cb_i]
        n_unique = len(set(pred_tokens.tolist()))
        c = Counter(pred_tokens.tolist())
        probs = np.array(list(c.values())) / len(pred_tokens)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        print(f'  CB{cb_i+1}: {n_unique:4d} unique tokens, entropy={entropy:.2f}')

    print(f'\n  V3e had: 37-41 unique tokens, entropy ~4.9 (diversity loss counterproductive)')
    print(f'  V3c had: 40 unique tokens, entropy ~4.89 (self-recon mode collapse)')
    print(f'  Ground truth: ~160 unique tokens, entropy ~7.1')
    print(f'  V3g target: 80-160 unique tokens, entropy 6-7')

    # Per-CB change rate
    print(f'\n--- Per-codebook change rate (vs source) ---')
    N_SHOW = min(10, N_CB)
    src_t = src_codes[0, :, :].numpy()
    print(f'{"Method":<25}', end='')
    for cb in range(N_SHOW):
        print(f' CB{cb:>2}', end='')
    print()
    print('-' * (25 + N_SHOW * 5))

    for name, codes in [
        ('FAISS+Smooth', ret_codes),
        ('V3g (p=0.92)', v3_results[0.92][0]),
        ('V3f keep CB0', v3_keepcb0),
    ]:
        c = codes[0, :, :].numpy()
        print(f'{name:<25}', end='')
        for cb in range(N_SHOW):
            rate = np.mean(c[cb] != src_t[cb]) * 100
            print(f' {rate:3.0f}%', end='')
        print()

    # Objective metrics
    from resemblyzer import VoiceEncoder
    spk_enc = data['spk_enc']

    def compute_secs(a, b):
        a16 = librosa.resample(a.astype(np.float32), orig_sr=SR, target_sr=16000)
        b16 = librosa.resample(b.astype(np.float32), orig_sr=SR, target_sr=16000)
        ea = spk_enc.embed_utterance(a16)
        eb = spk_enc.embed_utterance(b16)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

    def compute_mcd(ref, syn):
        r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=SR, n_mfcc=13)
        s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=SR, n_mfcc=13)
        L = min(r.shape[1], s.shape[1])
        if L == 0: return None
        diff = r[1:, :L] - s[1:, :L]
        return float(np.mean(np.sqrt(2.0 * np.sum(diff**2, axis=0))))

    tgt_ref_wav = target_wav_concat[:int(SR*15)]
    src_tgt_secs = compute_secs(source_wav, tgt_ref_wav)

    print(f'\n--- Objective metrics ---')
    print(f'  Source vs Target baseline: SECS={src_tgt_secs:.4f}')
    print(f'\n{"Method":<30} {"SECS(->tgt)":>12} {"SECS(->src)":>12} {"MCD":>8}')
    print('-'*65)

    eval_methods = [
        ('FAISS+Smooth (retrieval)', ret_wav),
        ('V3g (p=0.85)', v3_results[0.85][1]),
        ('V3g (p=0.92)', v3_results[0.92][1]),
        ('V3g (p=0.98)', v3_results[0.98][1]),
        ('V3f keep CB0 (p=0.92)', v3_keepcb0_wav),
    ]

    for name, wav in eval_methods:
        secs_tgt = compute_secs(wav, tgt_ref_wav)
        secs_src = compute_secs(wav, source_wav)
        mcd = compute_mcd(source_wav, wav)
        print(f'{name:<30} {secs_tgt:>12.4f} {secs_src:>12.4f} {mcd:>8.2f}')

    # Save audio
    save_items = [
        ('faiss_smooth.wav', ret_wav),
        ('v3g_p085.wav', v3_results[0.85][1]),
        ('v3g_p092.wav', v3_results[0.92][1]),
        ('v3g_p098.wav', v3_results[0.98][1]),
        ('v3g_keepcb0.wav', v3_keepcb0_wav),
    ]
    for fname, wav in save_items:
        out_path = os.path.join(args.output_dir, fname)
        sf.write(out_path, wav, SR)
        print(f'Saved: {out_path}')

    # Spectrograms
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), constrained_layout=True)
    for ax, (lbl, w) in zip(axes, [
        ('Source', source_wav),
        ('FAISS+Smooth', ret_wav),
        ('V3g (p=0.92)', v3_results[0.92][1]),
        ('V3f keep CB0', v3_keepcb0_wav),
        ('Target ref', target_wav_concat[:int(SR*15)]),
    ]):
        mel = librosa.feature.melspectrogram(y=w, sr=SR, n_mels=80, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=SR, x_axis='time', y_axis='mel', ax=ax, fmax=8000)
        ax.set_title(lbl, fontsize=10)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Mel Spectrograms - V3g', fontsize=13)
    plt.savefig(os.path.join(args.output_dir, 'spectrograms_v3g.png'), dpi=150)
    plt.close()

    print(f'\nAll results saved to: {args.output_dir}')


# ======================== Main ========================

def main():
    args = parse_args()
    cfg = setup(args)

    if args.download_only:
        print('=== download_only mode: fetching LibriSpeech to data_raw (no GPU needed) ===')
        from datasets import load_dataset
        os.makedirs(args.data_raw, exist_ok=True)
        print(f'[1/3] Downloading train-clean-100 ...', flush=True)
        ds100 = load_dataset('librispeech_asr', 'clean', split='train.100',
                             trust_remote_code=True, cache_dir=args.data_raw)
        print(f'  train-clean-100: {len(ds100)} utterances', flush=True)
        if args.use_train360:
            print(f'[2/3] Downloading train-clean-360 (~30GB) ...', flush=True)
            ds360 = load_dataset('librispeech_asr', 'clean', split='train.360',
                                 trust_remote_code=True, cache_dir=args.data_raw)
            print(f'  train-clean-360: {len(ds360)} utterances', flush=True)
        if args.use_train500:
            print(f'[3/3] Downloading train-other-500 (~50GB) ...', flush=True)
            ds500 = load_dataset('librispeech_asr', 'other', split='train.500',
                                 trust_remote_code=True, cache_dir=args.data_raw)
            print(f'  train-other-500: {len(ds500)} utterances', flush=True)
        print('Download complete. Exiting.', flush=True)
        return

    if args.prepare_only:
        print('=== prepare_only mode: encoding data with GPU (needs data_raw populated first) ===')
        prepare_data(cfg, args)
        print('=== Data preparation complete. Exiting. ===')
        return

    if args.phase in ('all', 'pretrain', 'finetune'):
        data = prepare_data(cfg, args)

    if args.phase in ('all', 'pretrain'):
        model, D_MODEL, N_HEADS, cb_weights = pretrain(cfg, data, args)

    if args.phase in ('all', 'finetune'):
        if args.phase == 'finetune':
            # Load pretrained model
            D_MODEL, N_HEADS = args.d_model, args.n_heads
            cb_weights = torch.tensor(
                [2.0**(-(i)/2) for i in range(cfg['N_LEARN'])], device=cfg['device'])
            cb_weights = cb_weights / cb_weights.sum() * cfg['N_LEARN']
            model = AcousticTranslatorV3g(
                n_vocab=cfg['CB_SIZE'], n_cb=cfg['N_LEARN'], d_model=D_MODEL, n_heads=N_HEADS,
                n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
                spk_emb_dim=data['SPK_EMB_DIM'],
            ).to(cfg['device'])
            best_path = os.path.join(args.ckpt_dir, 'v3g_pretrain_best.pt')
            ckpt = torch.load(best_path, weights_only=False, map_location=cfg['device'])
            model.load_state_dict(ckpt['model_state'])
            print(f'Loaded pretrained: {best_path}')

        target = load_target(cfg, data, args)
        model = finetune(cfg, data, target, model, D_MODEL, N_HEADS, cb_weights, args)

    if args.phase in ('all', 'infer'):
        if args.phase == 'infer':
            data = prepare_data(cfg, args)
            D_MODEL, N_HEADS = args.d_model, args.n_heads
            model = AcousticTranslatorV3g(
                n_vocab=cfg['CB_SIZE'], n_cb=cfg['N_LEARN'], d_model=D_MODEL, n_heads=N_HEADS,
                n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
                spk_emb_dim=data['SPK_EMB_DIM'],
            ).to(cfg['device'])
            ft_path = os.path.join(args.ckpt_dir, 'v3g_finetuned.pt')
            ckpt = torch.load(ft_path, weights_only=False, map_location=cfg['device'])
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            print(f'Loaded finetuned: {ft_path}')
            target = load_target(cfg, data, args)

        inference(cfg, data, target, model, args)

    print('\nDone.')


if __name__ == '__main__':
    main()
