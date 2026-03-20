#!/usr/bin/env python3
"""
CRT v5 — Codebook Refinement Transformer

Pipeline:
  source_CB0 → FAISS → target CB0-18 (already target speaker)
  (target_CB0, FAISS_CB1-18, spk_emb) → CRT → refined CB1-18
  [refined CB1-18, CB19-31_faiss] → Mimi decode → clean audio

Training uses cross-speaker FAISS corruption (faiss_frac=0.9) so the
inference distribution matches training exactly. CRT smooths frame-boundary
discontinuities introduced by frame-level FAISS retrieval.

Architecture: Conformer backbone (attention + depthwise-conv),
non-autoregressive, FiLM speaker conditioning.

Sizes:
  small  : d=128, h=4,  L=4   ~2M params
  medium : d=256, h=4,  L=6   ~8M params
  large  : d=512, h=8,  L=8   ~28M params
  xlarge : d=768, h=12, L=10  ~70M params
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
    'xlarge': dict(d_model=768, n_heads=12, n_layers=14, conv_kernel=31),
}

DEFAULT_STEPS = {
    'small': 10000, 'medium': 10000, 'large': 8000, 'xlarge': 30000,
}


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_processed', default='/datastore/eaic/nvquoc/mimi_vc/data_storage/processed')
    p.add_argument('--ckpt_dir',       default='/datastore/eaic/nvquoc/mimi_vc/v5.3.1-CRT-parallel/checkpoints')
    p.add_argument('--output_dir',     default='/datastore/eaic/nvquoc/mimi_vc/v5.3.1-CRT-parallel/output')
    p.add_argument('--model_size',     choices=list(MODEL_CONFIGS), default='medium')
    p.add_argument('--steps',          type=int, default=0,    help='0 = use size default')
    p.add_argument('--batch_size',     type=int, default=32)
    p.add_argument('--seq_len',        type=int, default=128)
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--n_learn',        type=int, default=18,
                   help='Number of CBs to predict (CB1 to CB{n_learn})')
    p.add_argument('--corrupt_ratio',  type=float, default=0.7,
                   help='Fraction of CB1-6 tokens to corrupt per sample')
    p.add_argument('--corrupt_ratio_hi', type=float, default=0.3,
                   help='Fraction of CB7+ tokens to corrupt (lower: FAISS already good)')
    p.add_argument('--faiss_frac',     type=float, default=0.5,
                   help='Of corrupt samples, fraction using FAISS (vs random)')
    p.add_argument('--focal_gamma',    type=float, default=1.0,
                   help='Focal loss gamma (0=CE, 1=mild focus on hard tokens)')
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


# ── CRT Model ──────────────────────────────────────────────────

class CRT(nn.Module):
    """
    Codebook Refinement Transformer.

    Input:  CB0 (anchor from FAISS) + FAISS CB1-N (noisy cross-speaker) + speaker_emb
    Output: refined CB logits  [B, n_cb, T, cb_size]
    """
    def __init__(self, n_cb=6, cb_size=2048, d_model=256, n_heads=4,
                 n_layers=6, conv_kernel=31, spk_emb_dim=256, dropout=0.1):
        super().__init__()
        self.n_cb    = n_cb
        self.d_model = d_model

        # Embed CB0 and each noisy CB independently, sum per frame
        self.cb0_embed = nn.Embedding(cb_size, d_model)
        self.cb_embeds = nn.ModuleList([
            nn.Embedding(cb_size, d_model) for _ in range(n_cb)
        ])

        # Speaker conditioning via FiLM (proven stable, near-identity init)
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

        # Per-CB output heads
        self.heads = nn.ModuleList([nn.Linear(d_model, cb_size) for _ in range(n_cb)])

    def forward(self, cb0, cb_noisy, spk_emb):
        """
        cb0:      [B, T]         — target CB0 (from FAISS)
        cb_noisy: [B, n_cb, T]   — first-pass CB1-N predictions (noisy)
        spk_emb:  [B, 256]       — target speaker embedding
        Returns:  [B, n_cb, T, cb_size]
        """
        x = self.cb0_embed(cb0)                           # [B, T, D]
        for i, emb in enumerate(self.cb_embeds):
            x = x + emb(cb_noisy[:, i])                  # add each noisy CB hint

        spk = self.spk_proj(spk_emb)
        x = self.film(x, spk)                            # speaker conditioning

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return torch.stack([h(x) for h in self.heads], dim=1)  # [B, n_cb, T, V]

    def compute_loss(self, logits, targets, focal_gamma=1.0):
        B, n_cb, T, V = logits.shape
        total = 0.0
        for i in range(n_cb):
            log_p = F.log_softmax(logits[:, i].float().reshape(-1, V), dim=-1)
            tgt_i = targets[:, i].reshape(-1)
            lp_t  = log_p.gather(1, tgt_i.unsqueeze(1)).squeeze(1)
            if focal_gamma > 0:
                w = (1.0 - lp_t.exp().detach()) ** focal_gamma
            else:
                w = torch.ones_like(lp_t)
            total += -(w * lp_t).mean()
        return total / n_cb


# ── Data ───────────────────────────────────────────────────────

def load_data(args):
    """Load processed speaker data from V4 cache (no re-encoding needed)."""
    cache_path = os.path.join(args.data_processed, 'prepared_data.pt')
    emb_path   = os.path.join(args.data_processed, 'speaker_embeddings.pkl')

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f'V4 processed cache not found: {cache_path}\n'
            f'Run V4 training first to build the cache.'
        )

    print(f'Loading V4 processed cache from {args.data_processed}...', flush=True)
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

    # Build per-speaker FAISS indices (for cross-speaker corruption)
    # omp_set_num_threads(1): each of the 8 ThreadPoolExecutor workers runs its
    # own single-threaded FAISS search — true parallelism instead of all workers
    # fighting over the same OMP thread pool.
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

def corrupt_random(cb1_6, corrupt_ratio, cb_size=2048):
    """Span-mask random tokens in CB1-6 (fast, general)."""
    n_cb, T = cb1_6.shape
    result = cb1_6.clone()
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
    """Random corruption with split ratio: first n_heavy CBs get ratio_heavy,
    remainder get ratio_light. Used when n_learn > 6 to avoid over-corrupting
    CB7-12 which are already well-approximated by FAISS."""
    result = corrupt_random(cb_clean[:n_heavy], ratio_heavy, cb_size)
    if cb_clean.shape[0] > n_heavy:
        rest = corrupt_random(cb_clean[n_heavy:], ratio_light, cb_size)
        result = torch.cat([result, rest], dim=0)
    return result


def corrupt_faiss(cb0_np, tgt_spk_id, speaker_faiss, cb0_emb, N_LEARN, frame_rate=12.5):
    """Retrieve CB1-6 from a different speaker via FAISS (realistic VC artifacts)."""
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
    return torch.tensor(full[:, 1:1 + N_LEARN].T, dtype=torch.long)  # [N_LEARN, T]


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
        """Build one training sample. Uses a private rng for thread safety."""
        spk   = rng.choice(spk_list)
        codes = speaker_data[spk]['codes']
        T     = codes.shape[1]
        start = rng.randint(0, max(0, T - SEQ_LEN))
        crop  = codes[:, start:start + SEQ_LEN]
        if crop.shape[1] < SEQ_LEN:
            crop = F.pad(crop, (0, SEQ_LEN - crop.shape[1]))

        cb0_clean = crop[0]
        cb_clean  = crop[1:1 + N_LEARN]

        use_faiss = (rng.random() < args.faiss_frac)
        if use_faiss:
            tgt = rng.choice(spk_list)
            while tgt == spk:
                tgt = rng.choice(spk_list)
            try:
                cb_noisy = corrupt_faiss(
                    cb0_clean.numpy(), tgt, speaker_faiss, cb0_emb, N_LEARN)
            except Exception:
                cb_noisy = _random_split_corrupt(
                    cb_clean, N_HEAVY, args.corrupt_ratio, args.corrupt_ratio_hi, CB_SIZE)
        else:
            cb_noisy = _random_split_corrupt(
                cb_clean, N_HEAVY, args.corrupt_ratio, args.corrupt_ratio_hi, CB_SIZE)

        spk_emb = torch.tensor(speaker_embeddings[spk], dtype=torch.float32)
        return cb0_clean, cb_noisy, cb_clean, spk_emb

    rngs = [random.Random() for _ in range(args.batch_size)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(make_one, rngs))

    cb0_batch, noisy_batch, clean_batch, spk_batch = zip(*results)
    return (
        torch.stack(list(cb0_batch)).to(device),
        torch.stack(list(noisy_batch)).to(device),
        torch.stack(list(clean_batch)).to(device),
        torch.stack(list(spk_batch)).to(device),
    )


# ── GPU stats ──────────────────────────────────────────────────

def _phys_gpu_idx():
    """Return the physical GPU index for pynvml/nvidia-smi queries.
    CUDA_VISIBLE_DEVICES remaps physical GPUs to logical index 0 for PyTorch,
    so we must read the env var to get the actual physical index."""
    cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_vis:
        try:
            return int(cuda_vis.split(',')[0])
        except ValueError:
            pass
    return torch.cuda.current_device()


def _gpu_stats():
    """Return formatted GPU stat string: util%, memory used/total, power watts.
    Uses torch.cuda for memory (fast) and pynvml/nvidia-smi for util+power.
    Never raises — returns empty string on any error."""
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


# ── Prefetch queue ─────────────────────────────────────────────

class PrefetchQueue:
    """Background thread that runs sample_batch() continuously so the GPU
    never stalls waiting for CPU FAISS work. The queue holds up to maxsize
    pre-built batches; the training loop just calls .get()."""

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
        n_cb=N_LEARN, cb_size=CB_SIZE, spk_emb_dim=256, dropout=args.dropout, **cfg
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*70}')
    print(f'CRT [{args.model_size.upper()}]')
    print(f'  d_model={cfg["d_model"]}, n_heads={cfg["n_heads"]}, '
          f'n_layers={cfg["n_layers"]}, conv_kernel={cfg["conv_kernel"]}')
    print(f'  Params: {n_params/1e6:.1f}M')
    print(f'  Steps: {steps} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'  Corrupt: {args.corrupt_ratio*100:.0f}% | '
          f'FAISS frac: {args.faiss_frac*100:.0f}%')
    print(f'  Focal gamma: {args.focal_gamma}')
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
        raw_model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['opt_state'])
        start_step = ckpt['step']
        best_loss  = ckpt.get('loss', float('inf'))
        for _ in range(start_step):
            sched.step()
        print(f'  Resumed from step {start_step} (loss {best_loss:.4f})', flush=True)
        if start_step >= steps:
            print('  Training complete, loading best checkpoint...', flush=True)
            raw_model.load_state_dict(
                torch.load(best_ckpt_path, weights_only=False, map_location=device)['model_state'])
            return model

    model.train()
    t0         = time.time()
    log_every  = 200
    save_every = 2000
    losses     = []
    cb_accs    = {i: [] for i in range(N_LEARN)}

    prefetch = PrefetchQueue(data, args, device, maxsize=4)
    _pending_save = None
    for step in range(start_step, steps):
        cb0_b, noisy_b, clean_b, spk_b = prefetch.get()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            logits = model(cb0_b, noisy_b, spk_b)
            loss   = model.compute_loss(logits, clean_b, args.focal_gamma)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                for i in range(N_LEARN):
                    pred = logits[:, i].float().argmax(-1).reshape(-1)
                    gt   = clean_b[:, i].reshape(-1)
                    cb_accs[i].append((pred == gt).float().mean().item() * 100)
            sps = (step + 1 - start_step) / (time.time() - t0)
            eta = (steps - step - 1) / sps
            avg = np.mean(losses[-log_every:])
            cb_str  = ' '.join(f'CB{i+1}:{cb_accs[i][-1]:.0f}%' for i in range(N_LEARN))
            gpu_str = _gpu_stats()
            print(f'  [{args.model_size}] {step+1:5d}/{steps} | '
                  f'Loss:{avg:.4f} | {cb_str} | {sps:.1f}step/s | ETA {eta:.0f}s'
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
            # Save in background so GPU is not blocked by disk I/O
            if _pending_save is not None:
                _pending_save.join()
            def _do_save(s, ps):
                for p in ps:
                    torch.save(s, p)
            _pending_save = threading.Thread(target=_do_save, args=(state, paths), daemon=True)
            _pending_save.start()
            print(f'  >> Saving [{args.model_size}] step {step+1} (avg loss {avg:.4f}){tag}',
                  flush=True)

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
    for i in range(N_LEARN):
        if cb_accs[i]:
            print(f'  CB{i+1}: {cb_accs[i][-1]:.1f}%')
    return model


# ── Refinement inference ───────────────────────────────────────

def load_mimi(device):
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    mimi     = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR       = feat_ext.sampling_rate
    return mimi, feat_ext, SR, dtype


def refine(data, model, args, device):
    """Apply trained CRT to refine FAISS VC output.

    Pipeline: source → FAISS → target CB0-{N_LEARN} → CRT → refined CB1-{N_LEARN}
    Training uses cross-speaker FAISS corruption (faiss_frac=0.9) so inference
    distribution matches training. CRT smooths frame-boundary discontinuities.
    """
    if not args.source_audio or not args.target_dir:
        print('Skipping refinement: --source_audio and --target_dir required.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    N_LEARN    = data['N_LEARN']
    cb0_emb    = data['cb0_emb']
    FRAME_RATE = data['FRAME_RATE']

    print(f'FAISS→CRT mode: CB1-{N_LEARN} from FAISS (faiss_frac=0.9 matches inference).', flush=True)
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

    # FAISS baseline: all CBs from target speaker frames
    faiss_codes = src_codes.clone()
    for t in range(T_src):
        faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])
    faiss_wav = decode(faiss_codes)

    # CRT inputs: CB0 and CB1-N_LEARN all from FAISS (same distribution as training)
    ret_cb0_np  = np.array([tgt_full[selected[t]][0]          for t in range(T_src)])
    ret_cb_all  = np.array([tgt_full[selected[t]][1:1+N_LEARN] for t in range(T_src)])
    cb0_d      = torch.tensor(ret_cb0_np, dtype=torch.long).unsqueeze(0).to(device)
    cb_input_d = torch.tensor(ret_cb_all.T, dtype=torch.long).unsqueeze(0).to(device)
    spk_d      = tgt_spk_emb.unsqueeze(0).to(device)

    # CRT refinement
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            logits = model(cb0_d, cb_input_d, spk_d)
        refined = logits.float().argmax(-1)[0].cpu()   # [N_LEARN, T]

    refined_codes = faiss_codes.clone()
    refined_codes[0, 1:1 + N_LEARN] = refined
    refined_wav = decode(refined_codes)

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
    print(f'{"Method":<30} {"SECS(->tgt)":>12} {"MCD":>8}')
    print('-' * 52)
    for name, wav in [('FAISS baseline (CB1-18)', faiss_wav),
                      (f'CRT-{args.model_size}', refined_wav)]:
        print(f'{name:<30} {secs(wav, tgt_ref):>12.4f} {mcd(src_wav, wav):>8.2f}')

    sf.write(os.path.join(args.output_dir, 'faiss_baseline.wav'), faiss_wav, SR)
    sf.write(os.path.join(args.output_dir, f'crt_{args.model_size}_refined.wav'), refined_wav, SR)
    print(f'Saved to: {args.output_dir}')


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
