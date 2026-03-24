#!/usr/bin/env python3
"""
CRT v5.7.1 — Continuous Latent Refinement

Operates in Mimi's quantizer.decode() output space [B, 512, T] instead of
discrete codebook tokens. The model predicts smooth corrections to FAISS-
retrieved continuous embeddings, bypassing quantization entirely at inference.

Architecture: Conformer backbone (same as v5.6.1), but input/output are
continuous 512-dim embeddings instead of discrete token embeddings.
Residual connection: output = projection(x) + input (predict correction only).

Decoder bypass at inference:
    refined_emb -> mimi.upsample -> mimi.decoder_transformer -> mimi.decoder -> audio
"""

import argparse
import os
import sys
import math
import random
import time
import gc
import pickle
import queue
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


# ── Sizes ──────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'small':  dict(d_model=128, n_heads=4,  n_layers=4,  conv_kernel=15),
    'medium': dict(d_model=256, n_heads=4,  n_layers=6,  conv_kernel=31),
    'large':  dict(d_model=512, n_heads=8,  n_layers=8,  conv_kernel=31),
    'xlarge': dict(d_model=1280, n_heads=16, n_layers=16, conv_kernel=31),
}

DEFAULT_STEPS = {
    'small': 10000, 'medium': 10000, 'large': 8000, 'xlarge': 60000,
}


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_processed', default='/datastore/eaic/nvquoc/mimi_vc/data_storage/processed_merged_latents')
    p.add_argument('--ckpt_dir',       default='/datastore/eaic/nvquoc/mimi_vc/v5.7.1-CRT-continuous-latent/checkpoints')
    p.add_argument('--output_dir',     default='/datastore/eaic/nvquoc/mimi_vc/v5.7.1-CRT-continuous-latent/output')
    p.add_argument('--model_size',     choices=list(MODEL_CONFIGS), default='xlarge')
    p.add_argument('--steps',          type=int, default=0,    help='0 = use size default')
    p.add_argument('--batch_size',     type=int, default=128)
    p.add_argument('--seq_len',        type=int, default=128)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--latent_dim',     type=int, default=512,
                   help='Mimi latent dim (fixed at 512)')
    p.add_argument('--corrupt_ratio',  type=float, default=0.30,
                   help='Fraction of frames to corrupt with noise')
    p.add_argument('--faiss_frac',     type=float, default=0.90,
                   help='Of corrupt samples, fraction using FAISS (vs random noise)')
    p.add_argument('--secs_weight',    type=float, default=0.1,
                   help='Weight for speaker similarity loss (0=disabled)')
    p.add_argument('--curriculum',     action='store_true',
                   help='Phase corrupt_ratio: 0.30→0.50→0.70 over three equal thirds')
    p.add_argument('--seed',           type=int, default=42)
    # Inference / refinement
    p.add_argument('--target_dir',     default=None)
    p.add_argument('--source_audio',   default=None)
    p.add_argument('--phase',          choices=['train', 'refine', 'all'], default='train')
    return p.parse_args()


# ── RoPE ───────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos()[None, None], persistent=False)
        self.register_buffer('sin_cached', freqs.sin()[None, None], persistent=False)

    def forward(self, x):
        T = x.shape[2]
        return self.cos_cached[:, :, :T, :], self.sin_cached[:, :, :T, :]


def apply_rope(q, k, cos, sin):
    h = q.shape[-1] // 2
    def rot(x):
        return torch.cat([x[..., :h] * cos - x[..., h:] * sin,
                          x[..., h:] * cos + x[..., :h] * sin], dim=-1)
    return rot(q), rot(k)


# ── FiLM ───────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x, cond):
        g, b = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + g) + b


# ── Conformer block ────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    Standard Conformer: FFN(½) → MHSA+RoPE → DepthwiseConv → FFN(½) → LN

    Better than pure transformer for speech sequences:
    the depthwise conv captures local frame-to-frame continuity
    (vocal tract smoothness) that global attention alone misses.
    """
    def __init__(self, d_model, n_heads, ff_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        ff_dim   = d_model * ff_mult
        head_dim = d_model // n_heads
        self.n_heads  = n_heads
        self.head_dim = head_dim

        # FFN 1 (pre, half step)
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

        # Multi-head self-attention + RoPE
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv    = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope   = RotaryEmbedding(head_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Depthwise conv (GLU gating + depthwise + LayerNorm)
        self.conv_norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model * 2)           # pointwise expand + gate
        self.dw  = nn.Conv1d(d_model, d_model, conv_kernel,
                             padding=conv_kernel // 2, groups=d_model)
        self.ln_conv = nn.LayerNorm(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.conv_drop = nn.Dropout(dropout)

        # FFN 2 (post, half step)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # FFN 1 (½ step)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        # MHSA
        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out  = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # Depthwise conv (GLU)
        h = self.conv_norm(x)
        h, gate = self.pw1(h).chunk(2, dim=-1)
        h = h * F.silu(gate)
        h = self.dw(h.transpose(1, 2))[:, :, :T].transpose(1, 2)  # trim causal pad
        h = self.pw2(self.ln_conv(h))
        x = x + self.conv_drop(h)

        # FFN 2 (½ step)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


# ── CRTLatent Model ───────────────────────────────────────────

class CRTLatent(nn.Module):
    """
    Continuous Latent Refinement Transformer.

    Input:  corrupted continuous embeddings [B, T, 512] + speaker_emb [B, 256]
    Output: refined continuous embeddings [B, T, 512] (with residual from input)

    Operates in Mimi's quantizer.decode() output space (range ±42).
    Predicts corrections via residual connection — FAISS embedding is the base signal.
    """
    def __init__(self, latent_dim=512, d_model=1280, n_heads=16,
                 n_layers=16, conv_kernel=31, spk_emb_dim=256, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Input: continuous embedding [512] -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Speaker conditioning (same as CRT)
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.film = FiLMLayer(d_model, d_model)

        # Conformer backbone
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output: d_model -> continuous embedding [512]
        self.output_proj = nn.Linear(d_model, latent_dim)

        # Speaker identity head (for SECS loss)
        self.spk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, spk_emb_dim),
        )

    def forward(self, corrupted_emb, spk_emb):
        """
        corrupted_emb: [B, T, 512] — FAISS-corrupted continuous embeddings
        spk_emb: [B, 256] — target speaker embedding
        Returns: (refined_emb [B, T, 512], spk_pred [B, 256])
        """
        x = self.input_proj(corrupted_emb)    # [B, T, d_model]
        spk = self.spk_proj(spk_emb)
        x = self.film(x, spk)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Residual: predict correction, not full embedding
        refined = self.output_proj(x) + corrupted_emb  # [B, T, 512]
        spk_pred = F.normalize(self.spk_head(x.mean(dim=1)), dim=-1)
        return refined, spk_pred

    def compute_loss(self, refined, target, spk_pred, spk_emb, secs_weight=0.1):
        recon_loss = F.l1_loss(refined, target)
        spk_tgt = F.normalize(spk_emb.float(), dim=-1)
        secs_loss = (1.0 - (spk_pred * spk_tgt).sum(dim=-1)).mean()
        total = recon_loss + secs_weight * secs_loss
        return total, recon_loss, secs_loss


# ── Data ───────────────────────────────────────────────────────

def load_data(args):
    """Load processed speaker data with continuous embeddings."""
    cache_path = os.path.join(args.data_processed, 'prepared_data.pt')
    emb_path   = os.path.join(args.data_processed, 'speaker_embeddings.pkl')

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f'Preprocessed latent cache not found: {cache_path}\n'
            f'Run scripts/preprocess_latents.py first to build it.'
        )

    print(f'Loading preprocessed latent data from {args.data_processed}...', flush=True)
    cached = torch.load(cache_path, weights_only=False, map_location='cpu')
    with open(emb_path, 'rb') as f:
        emb_data = pickle.load(f)

    speaker_data       = cached['speaker_data']
    speaker_embeddings = emb_data['speaker_embeddings']

    # Validate that embeddings exist
    sample_key = next(iter(speaker_data))
    if 'embeddings' not in speaker_data[sample_key]:
        raise RuntimeError(
            f'Speaker data missing "embeddings" key. '
            f'Run scripts/preprocess_latents.py first.'
        )
    print(f'  {len(speaker_data)} speakers loaded', flush=True)
    sample_emb = speaker_data[sample_key]['embeddings']
    print(f'  Embedding shape: {sample_emb.shape}, dtype: {sample_emb.dtype}', flush=True)

    # Sanity check: codes and embeddings must have matching T dimension
    mismatches = 0
    for spk_id, sdata in speaker_data.items():
        T_codes = sdata['codes'].shape[1]
        T_embs = sdata['embeddings'].shape[1]
        if T_codes != T_embs:
            mismatches += 1
            if mismatches <= 3:
                print(f'  [WARN] {spk_id}: codes T={T_codes} != embeddings T={T_embs}', flush=True)
            # Fix by trimming to minimum
            min_T = min(T_codes, T_embs)
            sdata['codes'] = sdata['codes'][:, :min_T]
            sdata['embeddings'] = sdata['embeddings'][:, :min_T]
    if mismatches:
        print(f'  Fixed {mismatches} speakers with T mismatch (trimmed to min)', flush=True)

    # Extract CB0 codebook embeddings for FAISS
    print('Extracting CB0 embeddings from Mimi...', flush=True)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained('kyutai/mimi')
    CB_SIZE = mimi.config.codebook_size
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
            if w.shape[0] >= CB_SIZE:
                cb0_emb = w.numpy()
                break
        except AttributeError:
            continue
    if cb0_emb is None:
        raise RuntimeError('Cannot extract CB0 embeddings from Mimi quantizer')
    del mimi
    gc.collect()
    print(f'  CB0 embeddings: {cb0_emb.shape}', flush=True)

    # Build per-speaker FAISS indices (for cross-speaker corruption)
    faiss.omp_set_num_threads(1)
    print('Building FAISS indices for corruption...', flush=True)
    speaker_faiss = {}
    for spk_id, sdata in speaker_data.items():
        codes = sdata['codes']
        sem   = codes[0].numpy()
        full  = codes.numpy().T
        emb   = cb0_emb[sem].astype(np.float32).copy()
        faiss.normalize_L2(emb)
        idx = faiss.IndexFlatIP(cb0_emb.shape[1])
        idx.add(emb)
        speaker_faiss[spk_id] = {
            'index': idx,
            'full_codes': full,
            'embeddings': sdata['embeddings'],  # [512, T] fp16
        }
    print(f'  {len(speaker_faiss)} FAISS indices built', flush=True)

    return {
        'speaker_data':       speaker_data,
        'speaker_embeddings': speaker_embeddings,
        'speaker_faiss':      speaker_faiss,
        'cb0_emb':            cb0_emb,
        'CB_SIZE':            CB_SIZE,
        'FRAME_RATE':         12.5,
    }


# ── Corruption ─────────────────────────────────────────────────

def corrupt_faiss_latent(cb0_np, tgt_spk_id, speaker_faiss, cb0_emb, frame_rate=12.5):
    """Retrieve continuous embeddings from a different speaker via FAISS."""
    T = len(cb0_np)
    sfd = speaker_faiss[tgt_spk_id]
    emb = cb0_emb[cb0_np].astype(np.float32).copy()
    faiss.normalize_L2(emb)
    dists, indices = sfd['index'].search(emb, 8)

    selected = np.zeros(T, dtype=np.int64)
    selected[0] = indices[0, 0]
    for t in range(1, T):
        cands = indices[t]
        combined = 0.5 * dists[t] + 0.5 / (1 + np.abs(cands - selected[t-1]) / frame_rate)
        selected[t] = cands[np.argmax(combined)]

    # Return continuous embeddings from target speaker
    spk_embeddings = sfd['embeddings']  # [512, T_spk] fp16
    T_emb = spk_embeddings.shape[1]
    selected = np.clip(selected, 0, T_emb - 1)  # safety clamp
    return spk_embeddings[:, selected].float()  # [512, T]


def corrupt_random_latent(embeddings, corrupt_ratio, rng):
    """Add noise to corrupt continuous embeddings via span masking."""
    C, T = embeddings.shape
    result = embeddings.clone()
    n_corrupt = int(T * corrupt_ratio)
    done = 0
    while done < n_corrupt:
        span = rng.randint(3, min(10, max(4, n_corrupt - done + 3)))
        start = rng.randint(0, max(0, T - span))
        end = min(start + span, T)
        # Replace with noise matching latent statistics
        result[:, start:end] = torch.randn(C, end - start) * embeddings.std() + embeddings.mean()
        done += end - start
    return result


# ── Batch sampler ──────────────────────────────────────────────

def sample_batch(data, args, device):
    speaker_data       = data['speaker_data']
    speaker_embeddings = data['speaker_embeddings']
    speaker_faiss      = data['speaker_faiss']
    cb0_emb            = data['cb0_emb']
    spk_list           = list(speaker_data.keys())
    SEQ_LEN            = args.seq_len
    faiss_frac         = args.faiss_frac
    corrupt_ratio      = args.corrupt_ratio

    def make_one(rng):
        spk = rng.choice(spk_list)
        sdata = speaker_data[spk]
        codes = sdata['codes']
        embs = sdata['embeddings']  # [512, T] fp16
        T = embs.shape[1]

        start = rng.randint(0, max(0, T - SEQ_LEN))
        crop_codes = codes[:, start:start + SEQ_LEN]
        crop_embs = embs[:, start:start + SEQ_LEN].float()  # cast fp16 -> fp32

        if crop_embs.shape[1] < SEQ_LEN:
            crop_embs = F.pad(crop_embs, (0, SEQ_LEN - crop_embs.shape[1]))
            crop_codes = F.pad(crop_codes, (0, SEQ_LEN - crop_codes.shape[1]))

        cb0 = crop_codes[0].numpy()
        target_emb = crop_embs  # [512, SEQ_LEN] — ground truth

        use_faiss = rng.random() < faiss_frac
        if use_faiss:
            tgt_spk = rng.choice(spk_list)
            while tgt_spk == spk:
                tgt_spk = rng.choice(spk_list)
            try:
                corrupted = corrupt_faiss_latent(cb0, tgt_spk, speaker_faiss, cb0_emb)
            except Exception:
                corrupted = corrupt_random_latent(target_emb, corrupt_ratio, rng)
        else:
            corrupted = corrupt_random_latent(target_emb, corrupt_ratio, rng)

        spk_emb = torch.tensor(speaker_embeddings[spk], dtype=torch.float32)
        return corrupted.T, target_emb.T, spk_emb
        # [T, 512], [T, 512], [256]

    rngs = [random.Random() for _ in range(args.batch_size)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(make_one, rngs))

    corrupted_batch, target_batch, spk_batch = zip(*results)
    return (
        torch.stack(list(corrupted_batch)).to(device),  # [B, T, 512]
        torch.stack(list(target_batch)).to(device),      # [B, T, 512]
        torch.stack(list(spk_batch)).to(device),          # [B, 256]
    )


# ── GPU stats ──────────────────────────────────────────────────

def _phys_gpu_idx():
    cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_vis:
        try:
            return int(cuda_vis.split(',')[0])
        except ValueError:
            pass
    return torch.cuda.current_device()


def _gpu_stats():
    if not torch.cuda.is_available():
        return ''
    try:
        mem_used  = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        phys_idx  = _phys_gpu_idx()
        try:
            import pynvml
            pynvml.nvmlInit()
            h     = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
            util  = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            r = subprocess.run(
                ['nvidia-smi', f'--id={phys_idx}',
                 '--query-gpu=utilization.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2)
            line  = r.stdout.strip().splitlines()[0]
            parts = line.split(', ')
            util  = int(parts[0])
            power = float(parts[1])
        return f'GPU:{util}% {mem_used:.1f}/{mem_total:.0f}GB {power:.0f}W'
    except Exception:
        return ''


# ── Curriculum ─────────────────────────────────────────────────

_CURRICULUM_PHASES = [
    0.30,  # phase 1: easy
    0.50,  # phase 2: medium
    0.70,  # phase 3: hard
]

def curriculum_ratio(step, total_steps):
    """Return corrupt_ratio for the current training step."""
    phase = min(2, int(step / total_steps * 3))
    return _CURRICULUM_PHASES[phase]


# ── Prefetch queue ─────────────────────────────────────────────

class PrefetchQueue:
    """Background thread that runs sample_batch() continuously so the GPU
    never stalls waiting for CPU FAISS work."""

    def __init__(self, data, args, device, maxsize=2):
        self.q     = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._t    = threading.Thread(
            target=self._worker, args=(data, args, device), daemon=True)
        self._t.start()

    def _worker(self, data, args, device):
        while not self._stop.is_set():
            try:
                batch = sample_batch(data, args, device)
                self.q.put(batch, timeout=0.5)
            except queue.Full:
                pass
            except Exception as e:
                import traceback
                print(f'\n[PrefetchQueue ERROR] {e}', flush=True)
                traceback.print_exc()
                # Put a sentinel so training loop doesn't hang forever
                self._stop.set()
                break

    def get(self):
        return self.q.get()

    def close(self):
        self._stop.set()
        self._t.join()


# ── Training ───────────────────────────────────────────────────

def train(data, args, device):
    cfg    = MODEL_CONFIGS[args.model_size]
    steps  = args.steps if args.steps > 0 else DEFAULT_STEPS[args.model_size]

    model = CRTLatent(
        latent_dim=args.latent_dim, spk_emb_dim=256, dropout=args.dropout,
        **cfg
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*70}')
    print(f'CRTLatent [{args.model_size.upper()}]')
    print(f'  d_model={cfg["d_model"]}, n_heads={cfg["n_heads"]}, '
          f'n_layers={cfg["n_layers"]}, conv_kernel={cfg["conv_kernel"]}')
    print(f'  latent_dim={args.latent_dim}')
    print(f'  Params: {n_params/1e6:.1f}M')
    print(f'  Steps: {steps} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'  Corrupt: {args.corrupt_ratio*100:.0f}% | '
          f'FAISS frac: {args.faiss_frac*100:.0f}%')
    print(f'  SECS weight: {args.secs_weight}')
    print(f'{"="*70}', flush=True)

    use_amp = device == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        warmup = 500
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ckpt_path      = os.path.join(args.ckpt_dir, f'crt_{args.model_size}.pt')
    best_ckpt_path = os.path.join(args.ckpt_dir, f'crt_{args.model_size}_best.pt')
    start_step     = 0
    best_loss      = float('inf')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        missing, unexpected = raw_model.load_state_dict(ckpt['model_state'], strict=False)
        if missing:
            print(f'  New keys (randomly init): {missing}', flush=True)
        if unexpected:
            print(f'  Dropped keys (ignored): {unexpected}', flush=True)
        if not missing and not unexpected:
            opt.load_state_dict(ckpt['opt_state'])
        else:
            print('  Optimizer reset (architecture changed)', flush=True)
        start_step = ckpt['step']
        best_loss  = ckpt.get('loss', float('inf'))
        for _ in range(start_step):
            sched.step()
        print(f'  Resumed from step {start_step} (loss {best_loss:.4f})', flush=True)
        if start_step >= steps:
            print('  Training complete, loading best checkpoint...', flush=True)
            raw_model.load_state_dict(
                torch.load(best_ckpt_path, weights_only=False, map_location=device)['model_state'],
                strict=False)
            return model

    model.train()
    t0         = time.time()
    log_every  = 200
    save_every  = 2000
    infer_every = 5000
    losses     = []
    l1_losses  = []
    secs_losses = []

    # Quick sanity test: run one sample_batch before starting queue
    print('Testing sample_batch...', flush=True)
    try:
        test_batch = sample_batch(data, args, device)
        print(f'  OK: shapes = {[t.shape for t in test_batch]}', flush=True)
        del test_batch
    except Exception as e:
        import traceback
        print(f'  FAILED: {e}', flush=True)
        traceback.print_exc()
        raise

    prefetch = PrefetchQueue(data, args, device, maxsize=4)
    _pending_save = None
    print(f'Training loop starting from step {start_step}...', flush=True)
    for step in range(start_step, steps):
        if args.curriculum:
            args.corrupt_ratio = curriculum_ratio(step, steps)
        corrupted_b, target_b, spk_b = prefetch.get()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            refined, spk_pred = model(corrupted_b, spk_b)
            loss, recon_loss, secs_loss = model.compute_loss(
                refined, target_b, spk_pred, spk_b, args.secs_weight)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        l1_losses.append(recon_loss.item())
        secs_losses.append(secs_loss.item())

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                l1 = F.l1_loss(refined, target_b).item()
                cos = F.cosine_similarity(
                    refined.reshape(-1, args.latent_dim),
                    target_b.reshape(-1, args.latent_dim), dim=-1).mean().item()
            sps = (step + 1 - start_step) / (time.time() - t0)
            eta = (steps - step - 1) / sps
            avg     = np.mean(losses[-log_every:])
            avg_l1  = np.mean(l1_losses[-log_every:])
            avg_sec = np.mean(secs_losses[-log_every:])
            gpu_str = _gpu_stats()
            phase_str = f' | corrupt:{args.corrupt_ratio:.0%}' if args.curriculum else ''
            print(f'  [xlarge] {step+1:6d}/{steps} | '
                  f'Loss:{avg:.4f} L1:{avg_l1:.4f} SECS:{avg_sec:.3f} CosSim:{cos:.3f} | '
                  f'{sps:.1f}step/s | ETA {eta:.0f}s'
                  + phase_str
                  + (f' | {gpu_str}' if gpu_str else ''), flush=True)

        if (step + 1) % save_every == 0:
            avg = np.mean(losses[-save_every:])
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            state = {
                'step': step + 1, 'loss': avg,
                'model_state': raw_model.state_dict(),
                'opt_state': opt.state_dict(),
                'model_size': args.model_size,
                'config': cfg,
            }
            tag = ' ** BEST' if avg < best_loss else ''
            if avg < best_loss:
                best_loss = avg
            paths = [ckpt_path, best_ckpt_path] if tag else [ckpt_path]
            if _pending_save is not None:
                _pending_save.join()
            def _do_save(s, ps):
                for p in ps:
                    torch.save(s, p)
            _pending_save = threading.Thread(target=_do_save, args=(state, paths), daemon=True)
            _pending_save.start()
            print(f'  >> Saving [{args.model_size}] step {step+1} (avg loss {avg:.4f}){tag}',
                  flush=True)

        if (step + 1) % infer_every == 0 and args.source_audio and args.target_dir:
            print(f'  >> Running inference at step {step+1}...', flush=True)
            model.eval()
            try:
                refine(data, model, args, device,
                       suffix=f'-{step+1}-step')
            except Exception as e:
                print(f'  >> Inference failed: {e}', flush=True)
            model.train()

    if _pending_save is not None:
        _pending_save.join()

    # Load best
    if os.path.exists(best_ckpt_path):
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_model.load_state_dict(
            torch.load(best_ckpt_path, weights_only=False, map_location=device)['model_state'])
    prefetch.close()
    model.eval()

    elapsed = time.time() - t0
    print(f'\nTraining done [{args.model_size}]: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    return model


# ── Refinement inference ───────────────────────────────────────

def load_mimi(device):
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    mimi     = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR       = feat_ext.sampling_rate
    return mimi, feat_ext, SR, dtype


def refine(data, model, args, device, suffix=''):
    """Apply trained CRTLatent to refine FAISS VC output.

    Pipeline:
      source -> encode -> CB0 -> FAISS -> target embeddings -> CRTLatent -> refined embeddings
      refined embeddings -> mimi.upsample -> decoder_transformer -> decoder -> audio

    Generates 3 outputs:
      1. faiss_baseline.wav      — FAISS with discrete decode (same as v5.6.1)
      2. faiss_latent_bypass.wav — FAISS embeddings decoded via bypass
      3. crt_latent_refined.wav  — CRT-refined embeddings decoded via bypass
    """
    if not args.source_audio or not args.target_dir:
        print('Skipping refinement: --source_audio and --target_dir required.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    cb0_emb    = data['cb0_emb']
    FRAME_RATE = data['FRAME_RATE']

    mimi, feat_ext, SR, dtype = load_mimi(device)

    def encode(wav):
        inp = feat_ext(raw_audio=wav, sampling_rate=SR, return_tensors='pt')
        inp = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None)
               for k, v in inp.items()}
        with torch.no_grad():
            return mimi.encode(inp['input_values'], inp.get('padding_mask')).audio_codes.cpu()

    def decode(codes):
        with torch.no_grad():
            return mimi.decode(codes.to(device)).audio_values.cpu().squeeze().numpy().astype(np.float32)

    # Source
    src_wav   = librosa.load(args.source_audio, sr=SR, mono=True)[0].astype(np.float32)
    src_codes = encode(src_wav)
    T_src     = src_codes.shape[2]
    print(f'Source: {T_src} frames ({T_src/FRAME_RATE:.1f}s)')

    # Target
    tgt_wavs = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_wavs.append(librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32))
    tgt_concat = np.concatenate(tgt_wavs)

    CHUNK = int(30 * SR)
    tgt_codes = torch.cat([
        encode(tgt_concat[i:i + CHUNK])
        for i in range(0, len(tgt_concat), CHUNK)
        if len(tgt_concat[i:i + CHUNK]) > SR
    ], dim=2)

    # Get target continuous embeddings
    with torch.no_grad():
        tgt_embs = mimi.quantizer.decode(tgt_codes.to(device)).cpu()  # [1, 512, T_tgt]
    tgt_embs_np = tgt_embs[0]  # [512, T_tgt]

    tgt_full = tgt_codes[0].numpy().T   # [T_tgt, 32]
    tgt_sem  = tgt_codes[0, 0].numpy()

    # Target FAISS index (from CB0)
    tgt_emb_faiss = cb0_emb[tgt_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb_faiss)
    tgt_index = faiss.IndexFlatIP(cb0_emb.shape[1])
    tgt_index.add(tgt_emb_faiss)

    # Target speaker embedding
    from resemblyzer import VoiceEncoder
    spk_enc = VoiceEncoder(device='cpu')
    tgt16   = librosa.resample(tgt_concat, orig_sr=SR, target_sr=16000)
    tgt_spk_emb = torch.tensor(spk_enc.embed_utterance(tgt16), dtype=torch.float32)

    # FAISS retrieval: source CB0 → target frames (with continuity)
    src_sem_np = src_codes[0, 0].numpy()
    src_emb_q  = cb0_emb[src_sem_np].astype(np.float32).copy()
    faiss.normalize_L2(src_emb_q)
    dists_k, indices_k = tgt_index.search(src_emb_q, 8)

    selected = np.zeros(T_src, dtype=np.int64)
    selected[0] = indices_k[0, 0]
    for t in range(1, T_src):
        cands    = indices_k[t]
        combined = 0.5 * dists_k[t] + 0.5 / (1 + np.abs(cands - selected[t-1]) / FRAME_RATE)
        selected[t] = cands[np.argmax(combined)]

    # 1. FAISS baseline: all CBs from target speaker frames (discrete decode)
    faiss_codes = src_codes.clone()
    for t in range(T_src):
        faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])
    faiss_wav = decode(faiss_codes)

    # 2. FAISS latent bypass: FAISS-selected embeddings decoded via bypass
    corrupted_emb = tgt_embs_np[:, selected]  # [512, T_src]
    with torch.no_grad():
        lat_c = corrupted_emb.unsqueeze(0).to(device, dtype=dtype)  # [1, 512, T]
        up_c = mimi.upsample(lat_c)
        dec_c = mimi.decoder_transformer(up_c.transpose(1, 2))[0].transpose(1, 2)
        audio_c = mimi.decoder(dec_c)
    faiss_latent_wav = audio_c.cpu().squeeze().numpy().astype(np.float32)

    # 3. CRT Latent refinement
    corrupted_input = corrupted_emb.T.unsqueeze(0).float().to(device)  # [1, T, 512]
    spk_d = tgt_spk_emb.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            refined_emb, _ = model(corrupted_input, spk_d)
        refined_latent = refined_emb[0].T.to(dtype=dtype)  # [512, T]

    # Decode via bypass (no quantization)
    with torch.no_grad():
        lat = refined_latent.unsqueeze(0).to(device)  # [1, 512, T]
        up = mimi.upsample(lat)
        dec = mimi.decoder_transformer(up.transpose(1, 2))[0].transpose(1, 2)
        audio = mimi.decoder(dec)
    refined_wav = audio.cpu().squeeze().numpy().astype(np.float32)

    # Metrics
    def secs(a, b):
        a16 = librosa.resample(a.astype(np.float32), orig_sr=SR, target_sr=16000)
        b16 = librosa.resample(b.astype(np.float32), orig_sr=SR, target_sr=16000)
        ea  = spk_enc.embed_utterance(a16)
        eb  = spk_enc.embed_utterance(b16)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

    def mcd(ref, syn):
        r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=SR, n_mfcc=13)
        s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=SR, n_mfcc=13)
        L = min(r.shape[1], s.shape[1])
        return float(np.mean(np.sqrt(2 * np.sum((r[1:, :L] - s[1:, :L]) ** 2, axis=0))))

    tgt_ref = tgt_concat[:int(SR * 15)]
    print(f'\n--- Refinement results [{args.model_size}] ---')
    print(f'{"Method":<35} {"SECS(->tgt)":>12} {"MCD":>8}')
    print('-' * 57)
    for name, wav in [('FAISS baseline (discrete)', faiss_wav),
                      ('FAISS latent (bypass)', faiss_latent_wav),
                      ('CRT latent refined (bypass)', refined_wav)]:
        print(f'{name:<35} {secs(wav, tgt_ref):>12.4f} {mcd(src_wav, wav):>8.2f}')

    if not suffix:
        sf.write(os.path.join(args.output_dir, 'faiss_baseline.wav'), faiss_wav, SR)
        sf.write(os.path.join(args.output_dir, 'faiss_latent_bypass.wav'), faiss_latent_wav, SR)
    sf.write(os.path.join(args.output_dir, f'crt_latent_refined{suffix}.wav'), refined_wav, SR)
    print(f'Saved to: {args.output_dir}/')


# ── Main ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    data  = load_data(args)
    model = None

    if args.phase in ('train', 'all'):
        model = train(data, args, device)

    if args.phase in ('refine', 'all'):
        if model is None:
            cfg  = MODEL_CONFIGS[args.model_size]
            model = CRTLatent(
                latent_dim=args.latent_dim, spk_emb_dim=256,
                dropout=args.dropout, **cfg
            ).to(device)
            best_path = os.path.join(args.ckpt_dir, f'crt_{args.model_size}_best.pt')
            ckpt = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            print(f'Loaded CRTLatent [{args.model_size}] from {best_path}')
        refine(data, model, args, device)

    print('\nDone.')


if __name__ == '__main__':
    main()
