#!/usr/bin/env python3
"""
CRT v10.0 — CB0 Refinement for Speaker Identity

Key insight: CB0 (semantic/WavLM-distilled) carries source speaker identity
that fights against voice conversion. Previous versions passed CB0 through
unchanged from FAISS. v10.0 predicts CB0-CB9 (10 codebooks), allowing the
model to strip source speaker identity from CB0 while preserving content.

Architecture changes from v9.0:
- CRT predicts CB0-CB9 (n_learn=10) instead of CB1-CB18
- CB0 is both input anchor AND prediction target
- FAISS CB0 feeds into the model as before, but the CB0 head can override it
- Confidence-gated CB0 refinement at inference (same as other CBs)
- Progressive training: CB0→CB0-2→CB0-5→CB0-9
- EWC inherited from v9.0 to protect early CB quality

Expected improvement: SECS should increase since CB0 no longer locks in
source speaker identity. The model can learn to adjust semantic tokens
toward the target speaker's distribution.
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
    'small': 10000, 'medium': 10000, 'large': 120000, 'xlarge': 240000,
}


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_processed', default='/datastore/eaic/nvquoc/mimi_vc/data_storage/processed')
    p.add_argument('--ckpt_dir',       default='/datastore/eaic/nvquoc/mimi_vc/v10.0-cb0-refine/checkpoints')
    p.add_argument('--output_dir',     default='/datastore/eaic/nvquoc/mimi_vc/v10.0-cb0-refine/output')
    p.add_argument('--model_size',     choices=list(MODEL_CONFIGS), default='large')
    p.add_argument('--steps',          type=int, default=0,    help='0 = use size default')
    p.add_argument('--batch_size',     type=int, default=32)
    p.add_argument('--seq_len',        type=int, default=128)
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--n_learn',        type=int, default=10,
                   help='Number of CBs to predict (CB0 to CB{n_learn-1})')
    p.add_argument('--corrupt_ratio',  type=float, default=0.7,
                   help='Fraction of tokens to corrupt per sample')
    p.add_argument('--corrupt_ratio_hi', type=float, default=0.3,
                   help='Fraction of higher CBs to corrupt')
    p.add_argument('--faiss_frac',     type=float, default=0.5,
                   help='Of corrupt samples, fraction using FAISS (vs random)')
    p.add_argument('--focal_gamma',    type=float, default=1.0)
    p.add_argument('--secs_weight',    type=float, default=0.3,
                   help='Weight for speaker similarity loss (higher to push SECS)')
    p.add_argument('--curriculum',     action='store_true')
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--target_dir',     default=None)
    p.add_argument('--source_audio',   default=None)
    p.add_argument('--phase',          choices=['train', 'refine', 'all'], default='train')
    p.add_argument('--confidence_threshold', type=float, default=0.5)
    p.add_argument('--rerank_topk',    type=int, default=0)
    p.add_argument('--rvq_dropout',    type=float, default=0.15)
    p.add_argument('--progressive',    action='store_true')
    p.add_argument('--ewc_lambda',    type=float, default=5000.0)
    p.add_argument('--ewc_samples',   type=int, default=200)
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
    def __init__(self, d_model, n_heads, ff_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        ff_dim   = d_model * ff_mult
        head_dim = d_model // n_heads
        self.n_heads  = n_heads
        self.head_dim = head_dim

        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv    = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope   = RotaryEmbedding(head_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.conv_norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model * 2)
        self.dw  = nn.Conv1d(d_model, d_model, conv_kernel,
                             padding=conv_kernel // 2, groups=d_model)
        self.ln_conv = nn.LayerNorm(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.conv_drop = nn.Dropout(dropout)

        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out  = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        h = self.conv_norm(x)
        h, gate = self.pw1(h).chunk(2, dim=-1)
        h = h * F.silu(gate)
        h = self.dw(h.transpose(1, 2))[:, :, :T].transpose(1, 2)
        h = self.pw2(self.ln_conv(h))
        x = x + self.conv_drop(h)

        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


# ── CRT Model (v10.0 — CB0 included) ─────────────────────────

class CRT(nn.Module):
    """
    Codebook Refinement Transformer v10.0.

    Key change: predicts CB0 through CB{n_cb-1} (not CB1-N).
    Input:  FAISS CB0 (anchor) + FAISS CB1-{n_cb-1} (noisy) + speaker_emb
    Output: refined CB logits [B, n_cb, T, cb_size] where index 0 = CB0

    CB0 prediction allows the model to strip source speaker identity
    from the semantic codebook while preserving linguistic content.
    """
    def __init__(self, n_cb=10, cb_size=2048, d_model=512, n_heads=8,
                 n_layers=8, conv_kernel=31, spk_emb_dim=256, dropout=0.1,
                 rvq_dropout_prob=0.0):
        super().__init__()
        self.n_cb    = n_cb
        self.d_model = d_model
        self.rvq_dropout_prob = rvq_dropout_prob
        self.active_cbs = n_cb

        # Embed FAISS CB0 as anchor + each noisy CB (CB0 through CB{n_cb-1})
        # All are from FAISS retrieval (cross-speaker)
        self.cb_embeds = nn.ModuleList([
            nn.Embedding(cb_size, d_model) for _ in range(n_cb)
        ])

        # Speaker conditioning via FiLM
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

        # Per-CB output heads (index 0 = CB0, index 1 = CB1, etc.)
        self.heads = nn.ModuleList([nn.Linear(d_model, cb_size) for _ in range(n_cb)])

        # Speaker identity head
        self.spk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, spk_emb_dim),
        )

    def forward(self, cb_noisy, spk_emb):
        """
        cb_noisy: [B, n_cb, T]  — FAISS CB0 through CB{n_cb-1} (all from cross-speaker retrieval)
        spk_emb:  [B, 256]      — target speaker embedding
        Returns:  (logits [B, n_cb, T, V], spk_pred [B, 256])
        """
        # Sum all input CB embeddings
        x = self.cb_embeds[0](cb_noisy[:, 0])            # [B, T, D]
        for i in range(1, self.n_cb):
            x = x + self.cb_embeds[i](cb_noisy[:, i])

        spk = self.spk_proj(spk_emb)
        x = self.film(x, spk)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        logits = torch.stack([h(x) for h in self.heads], dim=1)  # [B, n_cb, T, V]

        # RVQ dropout
        if self.training and self.rvq_dropout_prob > 0:
            drop_mask = torch.rand(logits.shape[0], logits.shape[1], 1, 1,
                                   device=logits.device) < self.rvq_dropout_prob
            logits = logits.masked_fill(drop_mask, 0.0)

        spk_pred = F.normalize(self.spk_head(x.mean(dim=1)), dim=-1)
        return logits, spk_pred

    def compute_loss(self, logits, targets, spk_pred, spk_emb,
                     focal_gamma=1.0, secs_weight=0.1):
        B, n_cb, T, V = logits.shape
        active = min(self.active_cbs, n_cb)
        total = 0.0
        computed = 0
        for i in range(active):
            if self.training and logits[:, i].abs().max() < 1e-6:
                continue
            log_p = F.log_softmax(logits[:, i].float().reshape(-1, V), dim=-1)
            tgt_i = targets[:, i].reshape(-1)
            lp_t  = log_p.gather(1, tgt_i.unsqueeze(1)).squeeze(1)
            if focal_gamma > 0:
                w = (1.0 - lp_t.exp().detach()) ** focal_gamma
            else:
                w = torch.ones_like(lp_t)
            total += -(w * lp_t).mean()
            computed += 1
        ce_loss = total / max(1, computed)
        spk_tgt_norm = F.normalize(spk_emb.float(), dim=-1)
        secs_loss = (1.0 - (spk_pred * spk_tgt_norm).sum(dim=-1)).mean()
        return ce_loss + secs_weight * secs_loss, ce_loss, secs_loss


# ── Data ───────────────────────────────────────────────────────

def load_data(args):
    cache_path = os.path.join(args.data_processed, 'prepared_data.pt')
    emb_path   = os.path.join(args.data_processed, 'speaker_embeddings.pkl')

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'Processed cache not found: {cache_path}')

    print(f'Loading processed cache from {args.data_processed}...', flush=True)
    cached = torch.load(cache_path, weights_only=False, map_location='cpu')
    with open(emb_path, 'rb') as f:
        emb_data = pickle.load(f)

    speaker_data       = cached['speaker_data']
    speaker_embeddings = emb_data['speaker_embeddings']
    print(f'  {len(speaker_data)} speakers loaded', flush=True)

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

    # Build per-speaker FAISS indices
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
        speaker_faiss[spk_id] = {'index': idx, 'full_codes': full}
    print(f'  {len(speaker_faiss)} FAISS indices built', flush=True)

    return {
        'speaker_data':       speaker_data,
        'speaker_embeddings': speaker_embeddings,
        'speaker_faiss':      speaker_faiss,
        'cb0_emb':            cb0_emb,
        'CB_SIZE':            CB_SIZE,
        'N_LEARN':            args.n_learn,
        'FRAME_RATE':         12.5,
    }


# ── Corruption ─────────────────────────────────────────────────

def corrupt_random(cb_slice, corrupt_ratio, cb_size=2048):
    n_cb, T = cb_slice.shape
    result = cb_slice.clone()
    n_corrupt = int(T * corrupt_ratio)
    for cb_i in range(n_cb):
        done = 0
        while done < n_corrupt:
            span = random.randint(3, min(10, n_corrupt - done + 3))
            span = min(span, T)
            start = random.randint(0, T - span)
            result[cb_i, start:start + span] = torch.randint(0, cb_size, (span,))
            done += span
    return result


def _random_split_corrupt(cb_clean, n_heavy, ratio_heavy, ratio_light, cb_size):
    result = corrupt_random(cb_clean[:n_heavy], ratio_heavy, cb_size)
    if cb_clean.shape[0] > n_heavy:
        rest = corrupt_random(cb_clean[n_heavy:], ratio_light, cb_size)
        result = torch.cat([result, rest], dim=0)
    return result


def corrupt_faiss(cb0_np, tgt_spk_id, speaker_faiss, cb0_emb, N_LEARN, frame_rate=12.5):
    """Retrieve CB0-CB{N_LEARN-1} from a different speaker via FAISS."""
    T   = len(cb0_np)
    sfd = speaker_faiss[tgt_spk_id]
    q   = cb0_emb[cb0_np].astype(np.float32).copy()
    faiss.normalize_L2(q)
    dists, indices = sfd['index'].search(q, 8)

    selected = np.zeros(T, dtype=np.int64)
    selected[0] = indices[0, 0]
    for t in range(1, T):
        cands    = indices[t]
        combined = 0.5 * dists[t] + 0.5 / (1 + np.abs(cands - selected[t-1]) / frame_rate)
        selected[t] = cands[np.argmax(combined)]

    full = sfd['full_codes'][selected]
    # CB0 through CB{N_LEARN-1} from the cross-speaker retrieval
    return torch.tensor(full[:, :N_LEARN].T, dtype=torch.long)  # [N_LEARN, T]


# ── Batch sampler ──────────────────────────────────────────────

def sample_batch(data, args, device):
    speaker_data       = data['speaker_data']
    speaker_embeddings = data['speaker_embeddings']
    speaker_faiss      = data['speaker_faiss']
    cb0_emb            = data['cb0_emb']
    N_LEARN            = data['N_LEARN']
    CB_SIZE            = data['CB_SIZE']
    spk_list           = list(speaker_data.keys())
    SEQ_LEN            = args.seq_len
    N_HEAVY            = min(6, N_LEARN)

    def make_one(rng):
        spk   = rng.choice(spk_list)
        codes = speaker_data[spk]['codes']
        T     = codes.shape[1]
        start = rng.randint(0, max(0, T - SEQ_LEN))
        crop  = codes[:, start:start + SEQ_LEN]
        if crop.shape[1] < SEQ_LEN:
            crop = F.pad(crop, (0, SEQ_LEN - crop.shape[1]))

        # Clean target: CB0 through CB{N_LEARN-1}
        cb_clean = crop[:N_LEARN]  # [N_LEARN, T] — includes CB0

        # CB0 from this speaker (for FAISS query)
        cb0_np = crop[0].numpy()

        use_faiss = (rng.random() < args.faiss_frac)
        if use_faiss:
            tgt = rng.choice(spk_list)
            while tgt == spk:
                tgt = rng.choice(spk_list)
            try:
                cb_noisy = corrupt_faiss(
                    cb0_np, tgt, speaker_faiss, cb0_emb, N_LEARN)
            except Exception:
                cb_noisy = _random_split_corrupt(
                    cb_clean, N_HEAVY, args.corrupt_ratio, args.corrupt_ratio_hi, CB_SIZE)
        else:
            cb_noisy = _random_split_corrupt(
                cb_clean, N_HEAVY, args.corrupt_ratio, args.corrupt_ratio_hi, CB_SIZE)

        spk_emb = torch.tensor(speaker_embeddings[spk], dtype=torch.float32)
        return cb_noisy, cb_clean, spk_emb

    rngs = [random.Random() for _ in range(args.batch_size)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(make_one, rngs))

    noisy_batch, clean_batch, spk_batch = zip(*results)
    return (
        torch.stack(list(noisy_batch)).to(device),   # [B, N_LEARN, T]
        torch.stack(list(clean_batch)).to(device),   # [B, N_LEARN, T]
        torch.stack(list(spk_batch)).to(device),     # [B, 256]
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
        mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
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


# ── Progressive schedule (v10.0: CB0-based) ──────────────────

_PROGRESSIVE_PHASES = [
    # (corrupt_ratio, corrupt_ratio_hi, active_cbs)
    (0.30, 0.15, 1),    # Phase 1: CB0 only
    (0.30, 0.15, 3),    # Phase 2: CB0-2
    (0.40, 0.20, 6),    # Phase 3: CB0-5
    (0.50, 0.25, 10),   # Phase 4: CB0-9 all
    (0.70, 0.30, 10),   # Phase 5: CB0-9 + hard corruption
]

def progressive_ratios(step, total_steps, n_cb):
    n_phases = len(_PROGRESSIVE_PHASES)
    phase = min(n_phases - 1, int(step / total_steps * n_phases))
    cr, crh, active = _PROGRESSIVE_PHASES[phase]
    return cr, crh, min(active, n_cb), phase

def curriculum_ratios(step, total_steps):
    _CURRICULUM_PHASES = [
        (0.30, 0.15),
        (0.50, 0.20),
        (0.70, 0.30),
    ]
    phase = min(2, int(step / total_steps * 3))
    return _CURRICULUM_PHASES[phase]


# ── EWC (Elastic Weight Consolidation) ────────────────────────

class EWC:
    def __init__(self):
        self.snapshots = []

    def compute_fisher(self, model, data, args, device, n_samples=200):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}

        for _ in range(n_samples):
            noisy_b, clean_b, spk_b = sample_batch(data, args, device)
            model.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
                logits, spk_pred = model(noisy_b, spk_b)
                loss, _, _ = model.compute_loss(
                    logits, clean_b, spk_pred, spk_b,
                    args.focal_gamma, args.secs_weight)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= n_samples
            fisher[n] = fisher[n].clamp(max=fisher[n].quantile(0.99).item())

        param_snapshot = {n: p.detach().clone()
                          for n, p in model.named_parameters() if n in fisher}

        self.snapshots.append({'fisher': fisher, 'params': param_snapshot})
        model.train()
        n_params = sum(f.numel() for f in fisher.values())
        avg_f = np.mean([f.mean().item() for f in fisher.values()])
        print(f'  >> EWC snapshot #{len(self.snapshots)}: {n_params/1e6:.1f}M params, '
              f'avg Fisher={avg_f:.6f}', flush=True)

    def penalty(self, model):
        if not self.snapshots:
            return 0.0
        total = 0.0
        for snap in self.snapshots:
            fisher = snap['fisher']
            params_star = snap['params']
            for n, p in model.named_parameters():
                if n in fisher:
                    total = total + (fisher[n] * (p - params_star[n]) ** 2).sum()
        return total

    def state_dict(self):
        return [{'fisher': {n: f.cpu() for n, f in s['fisher'].items()},
                 'params': {n: p.cpu() for n, p in s['params'].items()}}
                for s in self.snapshots]

    def load_state_dict(self, state, device):
        self.snapshots = [
            {'fisher': {n: f.to(device) for n, f in s['fisher'].items()},
             'params': {n: p.to(device) for n, p in s['params'].items()}}
            for s in state
        ]


# ── Prefetch queue ─────────────────────────────────────────────

class PrefetchQueue:
    def __init__(self, data, args, device, maxsize=2):
        self.q     = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._t    = threading.Thread(
            target=self._worker, args=(data, args, device), daemon=True)
        self._t.start()

    def _worker(self, data, args, device):
        while not self._stop.is_set():
            try:
                self.q.put(sample_batch(data, args, device), timeout=0.5)
            except queue.Full:
                pass

    def get(self):
        return self.q.get()

    def close(self):
        self._stop.set()
        self._t.join()


# ── Training ───────────────────────────────────────────────────

def train(data, args, device):
    cfg    = MODEL_CONFIGS[args.model_size]
    steps  = args.steps if args.steps > 0 else DEFAULT_STEPS[args.model_size]
    N_LEARN = data['N_LEARN']
    CB_SIZE = data['CB_SIZE']

    model = CRT(
        n_cb=N_LEARN, cb_size=CB_SIZE, spk_emb_dim=256, dropout=args.dropout,
        rvq_dropout_prob=args.rvq_dropout, **cfg
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*70}')
    print(f'CRT v10.0 [{args.model_size.upper()}] — CB0 Refinement')
    print(f'  d_model={cfg["d_model"]}, n_heads={cfg["n_heads"]}, '
          f'n_layers={cfg["n_layers"]}, conv_kernel={cfg["conv_kernel"]}')
    print(f'  Params: {n_params/1e6:.1f}M')
    print(f'  Predicts: CB0-CB{N_LEARN-1} ({N_LEARN} codebooks, CB0 INCLUDED)')
    print(f'  Steps: {steps} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'  Corrupt: {args.corrupt_ratio*100:.0f}% | '
          f'FAISS frac: {args.faiss_frac*100:.0f}%')
    print(f'  Focal gamma: {args.focal_gamma} | SECS weight: {args.secs_weight}')
    print(f'  RVQ dropout: {args.rvq_dropout:.0%} | Progressive: {args.progressive}')
    print(f'  EWC lambda: {args.ewc_lambda} | EWC samples: {args.ewc_samples}')
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
    prev_phase     = -1

    ewc = EWC()

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        missing, unexpected = raw_model.load_state_dict(ckpt['model_state'], strict=False)
        if missing:
            print(f'  New keys (randomly init): {missing}', flush=True)
        if unexpected:
            print(f'  Dropped keys (ignored): {unexpected}', flush=True)
        if not missing and not unexpected and 'opt_state' in ckpt:
            opt.load_state_dict(ckpt['opt_state'])
        else:
            print('  Optimizer reset (warm-start or architecture changed)', flush=True)
        start_step = ckpt['step']
        best_loss  = ckpt.get('loss', float('inf'))
        prev_phase = ckpt.get('phase_idx', -1)
        if 'ewc_state' in ckpt:
            ewc.load_state_dict(ckpt['ewc_state'], device)
            print(f'  Restored {len(ewc.snapshots)} EWC snapshots', flush=True)
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
    ce_losses  = []
    secs_losses = []
    cb_accs    = {i: [] for i in range(N_LEARN)}

    prefetch = PrefetchQueue(data, args, device, maxsize=4)
    _pending_save = None
    ewc_losses = []
    curr_phase = prev_phase
    for step in range(start_step, steps):
        if args.progressive:
            args.corrupt_ratio, args.corrupt_ratio_hi, active, curr_phase = \
                progressive_ratios(step, steps, N_LEARN)
            model.active_cbs = active

            if curr_phase != prev_phase:
                if prev_phase >= 0 and args.ewc_lambda > 0:
                    print(f'  >> Computing EWC Fisher for phase {prev_phase+1} '
                          f'({args.ewc_samples} batches)...', flush=True)
                    ewc.compute_fisher(model, data, args, device, n_samples=args.ewc_samples)
                prev_phase = curr_phase
        elif args.curriculum:
            args.corrupt_ratio, args.corrupt_ratio_hi = curriculum_ratios(step, steps)

        noisy_b, clean_b, spk_b = prefetch.get()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            logits, spk_pred = model(noisy_b, spk_b)
            loss, ce_loss, secs_loss = model.compute_loss(
                logits, clean_b, spk_pred, spk_b,
                args.focal_gamma, args.secs_weight)

        ewc_loss_val = 0.0
        if ewc.snapshots and args.ewc_lambda > 0:
            ewc_pen = ewc.penalty(model)
            loss = loss + (args.ewc_lambda / 2.0) * ewc_pen
            ewc_loss_val = ewc_pen.item()

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        ce_losses.append(ce_loss.item())
        secs_losses.append(secs_loss.item())
        ewc_losses.append(ewc_loss_val)

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                for i in range(N_LEARN):
                    pred = logits[:, i].float().argmax(-1).reshape(-1)
                    gt   = clean_b[:, i].reshape(-1)
                    cb_accs[i].append((pred == gt).float().mean().item() * 100)
            sps = (step + 1 - start_step) / (time.time() - t0)
            eta = (steps - step - 1) / sps
            avg     = np.mean(losses[-log_every:])
            avg_ce  = np.mean(ce_losses[-log_every:])
            avg_sec = np.mean(secs_losses[-log_every:])
            active = getattr(model, 'active_cbs', N_LEARN)
            cb_str  = ' '.join(f'CB{i}:{cb_accs[i][-1]:.0f}%' for i in range(min(active, N_LEARN)))
            gpu_str = _gpu_stats()
            avg_ewc = np.mean(ewc_losses[-log_every:]) if ewc_losses else 0.0
            if args.progressive:
                phase_str = f' | phase:{curr_phase+1}/{len(_PROGRESSIVE_PHASES)} active:{active}/{N_LEARN} corrupt:{args.corrupt_ratio:.0%}'
                if avg_ewc > 0:
                    phase_str += f' ewc:{avg_ewc:.4f}'
            elif args.curriculum:
                phase_str = f' | corrupt:{args.corrupt_ratio:.0%}'
            else:
                phase_str = ''
            print(f'  [{args.model_size}] {step+1:5d}/{steps} | '
                  f'Loss:{avg:.4f} CE:{avg_ce:.4f} SECS:{avg_sec:.3f} | '
                  f'{cb_str} | {sps:.1f}step/s | ETA {eta:.0f}s'
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
                'phase_idx': curr_phase if args.progressive else -1,
                'ewc_state': ewc.state_dict(),
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
                refine(data, model, args, device, suffix=f'-{step+1}-step')
            except Exception as e:
                print(f'  >> Inference failed: {e}', flush=True)
            model.train()

    if _pending_save is not None:
        _pending_save.join()

    if os.path.exists(best_ckpt_path):
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_model.load_state_dict(
            torch.load(best_ckpt_path, weights_only=False, map_location=device)['model_state'])
    prefetch.close()
    model.eval()

    elapsed = time.time() - t0
    print(f'\nTraining done [{args.model_size}]: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    for i in range(N_LEARN):
        if cb_accs[i]:
            print(f'  CB{i}: {cb_accs[i][-1]:.1f}%')
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
    """Apply trained CRT to refine FAISS VC output.

    v10.0: CRT predicts CB0-CB{N_LEARN-1}. CB0 is now refined too,
    allowing the model to adjust semantic tokens for better speaker identity.
    """
    if not args.source_audio or not args.target_dir:
        print('Skipping refinement: --source_audio and --target_dir required.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    N_LEARN    = data['N_LEARN']
    cb0_emb    = data['cb0_emb']
    FRAME_RATE = data['FRAME_RATE']

    print(f'FAISS→CRT mode: CRT predicts CB0-CB{N_LEARN-1} (CB0 INCLUDED).', flush=True)
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
    tgt_full = tgt_codes[0].numpy().T   # [T_tgt, 32]
    tgt_sem  = tgt_codes[0, 0].numpy()

    # Target FAISS index
    tgt_emb = cb0_emb[tgt_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb)
    tgt_index = faiss.IndexFlatIP(cb0_emb.shape[1])
    tgt_index.add(tgt_emb)

    # Target speaker embedding
    from resemblyzer import VoiceEncoder
    spk_enc = VoiceEncoder(device='cpu')
    tgt16   = librosa.resample(tgt_concat, orig_sr=SR, target_sr=16000)
    tgt_spk_emb = torch.tensor(spk_enc.embed_utterance(tgt16), dtype=torch.float32)

    # FAISS retrieval
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

    # FAISS baseline: all CBs from target speaker frames
    faiss_codes = src_codes.clone()
    for t in range(T_src):
        faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])
    faiss_wav = decode(faiss_codes)

    # CRT inputs: CB0 through CB{N_LEARN-1} all from FAISS retrieval
    ret_cb_all = np.array([tgt_full[selected[t]][:N_LEARN] for t in range(T_src)])
    cb_input_d = torch.tensor(ret_cb_all.T, dtype=torch.long).unsqueeze(0).to(device)  # [1, N_LEARN, T]
    spk_d      = tgt_spk_emb.unsqueeze(0).to(device)

    # CRT refinement
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            logits, _spk_pred = model(cb_input_d, spk_d)
        probs = logits.float().softmax(-1)                     # [1, N_LEARN, T, 2048]
        crt_tokens = probs.argmax(-1)[0].cpu()                 # [N_LEARN, T]
        crt_confidence = probs.max(-1).values[0].cpu()         # [N_LEARN, T]

    faiss_tokens = torch.tensor(ret_cb_all.T, dtype=torch.long)  # [N_LEARN, T]

    # --- Variant 1: CRT argmax (all tokens replaced including CB0) ---
    argmax_codes = faiss_codes.clone()
    argmax_codes[0, :N_LEARN] = crt_tokens

    # --- Variant 2: Confidence-based selective refinement ---
    conf_variants = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sel = faiss_tokens.clone()
        mask = crt_confidence > thresh
        sel[mask] = crt_tokens[mask]
        codes = faiss_codes.clone()
        codes[0, :N_LEARN] = sel
        override_pct = mask.float().mean().item() * 100
        # Also track CB0 override separately
        cb0_override = mask[0].float().mean().item() * 100
        conf_variants[thresh] = (codes, override_pct, cb0_override)

    # --- Variant 3: FAISS re-ranker ---
    reranked = faiss_tokens.clone()
    for t in range(T_src):
        cand_frames = indices_k[t]
        for cb_i in range(N_LEARN):
            cand_tokens = [int(tgt_full[f][cb_i]) for f in cand_frames]
            cand_probs = [probs[0, cb_i, t, tok].item() for tok in cand_tokens]
            reranked[cb_i, t] = cand_tokens[int(np.argmax(cand_probs))]
    rerank_codes = faiss_codes.clone()
    rerank_codes[0, :N_LEARN] = reranked

    # --- Variant 4: Re-ranker + confidence threshold ---
    rerank_conf = faiss_tokens.clone()
    thresh_rr = args.confidence_threshold
    for t in range(T_src):
        cand_frames = indices_k[t]
        for cb_i in range(N_LEARN):
            cand_tokens = [int(tgt_full[f][cb_i]) for f in cand_frames]
            cand_probs = [probs[0, cb_i, t, tok].item() for tok in cand_tokens]
            best_prob = max(cand_probs)
            if best_prob > thresh_rr:
                rerank_conf[cb_i, t] = cand_tokens[int(np.argmax(cand_probs))]
    rerank_conf_codes = faiss_codes.clone()
    rerank_conf_codes[0, :N_LEARN] = rerank_conf

    # --- Variant 5: CB0-only refinement (keep CB1+ from FAISS) ---
    cb0_only_codes = faiss_codes.clone()
    for thresh in [0.3, 0.5]:
        codes = faiss_codes.clone()
        mask0 = crt_confidence[0] > thresh
        cb0_refined = faiss_tokens[0].clone()
        cb0_refined[mask0] = crt_tokens[0][mask0]
        codes[0, 0] = cb0_refined
        pct = mask0.float().mean().item() * 100
        conf_variants[f'cb0only_{thresh}'] = (codes, pct, pct)

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

    # --- Print comparison table ---
    print(f'\n{"="*75}')
    print(f' Refinement results [{args.model_size}]{suffix}  (v10.0 — CB0 included)')
    print(f'{"="*75}')
    print(f'{"Method":<45} {"SECS":>6} {"MCD":>7} {"Override%":>10} {"CB0%":>6}')
    print('-' * 75)

    results = []

    # FAISS baseline
    faiss_wav = decode(faiss_codes)
    s, m = secs(faiss_wav, tgt_ref), mcd(src_wav, faiss_wav)
    print(f'{"FAISS baseline (all 32 CB)":<45} {s:>6.4f} {m:>7.2f} {"100%":>10} {"N/A":>6}')
    results.append(('faiss_baseline', faiss_wav, s, m))

    # CRT argmax
    argmax_wav = decode(argmax_codes)
    s, m = secs(argmax_wav, tgt_ref), mcd(src_wav, argmax_wav)
    print(f'{"CRT argmax (CB0-9 all replaced)":<45} {s:>6.4f} {m:>7.2f} {"100%":>10} {"100%":>6}')
    results.append(('crt_argmax', argmax_wav, s, m))

    # Confidence sweep
    for thresh in sorted(k for k in conf_variants.keys() if isinstance(k, float)):
        codes, pct, cb0_pct = conf_variants[thresh]
        wav = decode(codes)
        s, m = secs(wav, tgt_ref), mcd(src_wav, wav)
        print(f'{f"CRT confidence>{thresh:.1f}":<45} {s:>6.4f} {m:>7.2f} {f"{pct:.1f}%":>10} {f"{cb0_pct:.0f}%":>6}')
        results.append((f'crt_conf_{thresh:.1f}', wav, s, m))

    # CB0-only variants
    for key in sorted(k for k in conf_variants.keys() if isinstance(k, str)):
        codes, pct, cb0_pct = conf_variants[key]
        wav = decode(codes)
        s, m = secs(wav, tgt_ref), mcd(src_wav, wav)
        print(f'{f"CRT {key}":<45} {s:>6.4f} {m:>7.2f} {f"{pct:.1f}%":>10} {f"{cb0_pct:.0f}%":>6}')
        results.append((f'crt_{key}', wav, s, m))

    # Re-ranker
    rerank_wav = decode(rerank_codes)
    s, m = secs(rerank_wav, tgt_ref), mcd(src_wav, rerank_wav)
    print(f'{"CRT re-ranker (top-8)":<45} {s:>6.4f} {m:>7.2f} {"100%":>10} {"100%":>6}')
    results.append(('crt_reranker', rerank_wav, s, m))

    # Re-ranker + confidence
    rerank_conf_wav = decode(rerank_conf_codes)
    s, m = secs(rerank_conf_wav, tgt_ref), mcd(src_wav, rerank_conf_wav)
    print(f'{f"CRT re-ranker + conf>{thresh_rr:.1f}":<45} {s:>6.4f} {m:>7.2f}')
    results.append(('crt_reranker_conf', rerank_conf_wav, s, m))

    print(f'{"="*75}')

    # Save all variants
    for name, wav, _, _ in results:
        path = os.path.join(args.output_dir, f'{name}{suffix}.wav')
        sf.write(path, wav, SR)
    print(f'Saved {len(results)} variants to {args.output_dir}/')


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
            model = CRT(n_cb=data['N_LEARN'], cb_size=data['CB_SIZE'],
                        spk_emb_dim=256, dropout=args.dropout, **cfg).to(device)
            best_path = os.path.join(args.ckpt_dir, f'crt_{args.model_size}_best.pt')
            ckpt = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            print(f'Loaded CRT [{args.model_size}] from {best_path}')
        refine(data, model, args, device)

    print('\nDone.')


if __name__ == '__main__':
    main()
