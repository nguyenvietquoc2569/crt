#!/usr/bin/env python3
"""
v7.1 — Autoregressive Voice Conversion with Prefix Conditioning

Architecture: CausalConformerVC
  - Prepend target speaker reference tokens as PREFIX (all 19 CBs summed)
  - Add [SEP] token
  - Source CB0 tokens follow; model generates CB1-18 autoregressively
    (one FRAME at a time, 18 output heads per frame)
  - Causal masking: each generated frame sees prefix + all previous frames
  - Prefix region uses bidirectional attention within itself

Key differences from v7.0 (cross-attention):
  - AR generation instead of parallel prediction
  - CausalConformerBlock with prefix-aware causal mask + left-only conv padding
  - Teacher forcing during training (shifted-right CB1-18)
  - KV-cached inference for efficiency
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
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


# -- Sizes ----------------------------------------------------------------

MODEL_CONFIGS = {
    'small':  dict(d_model=256,  n_heads=4,  n_layers=6,  conv_kernel=31),
    'medium': dict(d_model=512,  n_heads=8,  n_layers=8,  conv_kernel=31),
    'large':  dict(d_model=768,  n_heads=12, n_layers=12, conv_kernel=31),
    'xlarge': dict(d_model=1024, n_heads=16, n_layers=14, conv_kernel=31),
}


# -- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_processed', required=True)
    p.add_argument('--ckpt_dir',       default='/datastore/eaic/nvquoc/mimi_vc/v7.1-AR-voice-conversion/checkpoints')
    p.add_argument('--output_dir',     default='/datastore/eaic/nvquoc/mimi_vc/v7.1-AR-voice-conversion/output')
    p.add_argument('--model_size',     choices=list(MODEL_CONFIGS), default='xlarge')
    p.add_argument('--steps',          type=int, default=120000)
    p.add_argument('--batch_size',     type=int, default=32)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--n_cb',           type=int, default=18,
                   help='Number of acoustic CBs to predict (CB1..CB{n_cb})')
    p.add_argument('--t_ref',          type=int, default=128,
                   help='Reference sequence length (target speaker, ~10s)')
    p.add_argument('--t_src',          type=int, default=64,
                   help='Source sequence length (~5s)')
    p.add_argument('--cross_speaker_ratio', type=float, default=0.9,
                   help='Fraction of samples using cross-speaker source CB0')
    p.add_argument('--focal_gamma',    type=float, default=1.0)
    p.add_argument('--secs_weight',    type=float, default=0.1)
    p.add_argument('--temperature',    type=float, default=0.7,
                   help='Temperature for AR sampling at inference')
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--target_dir',     default=None)
    p.add_argument('--source_audio',   default=None)
    p.add_argument('--phase',          choices=['train', 'infer', 'all'], default='train')
    return p.parse_args()


# -- RoPE -----------------------------------------------------------------

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


# -- FiLM -----------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x, cond):
        g, b = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + g) + b


# -- Prefix-aware causal mask ---------------------------------------------

def _prefix_causal_mask(T_total, prefix_len, device):
    """Attention mask where True = cannot attend.

    Prefix positions (0..prefix_len-1) attend to all prefix positions
    (bidirectional within prefix).
    Source/generation positions attend causally: to entire prefix + all
    previous positions up to and including themselves.
    """
    mask = torch.ones(T_total, T_total, dtype=torch.bool, device=device)
    # Prefix sees all prefix (bidirectional)
    mask[:prefix_len, :prefix_len] = False
    # Each generation position sees prefix + all previous + self
    for t in range(prefix_len, T_total):
        mask[t, :t + 1] = False
    return mask


# -- CausalConformerBlock --------------------------------------------------

class CausalConformerBlock(nn.Module):
    """
    Conformer block with causal self-attention and causal depthwise conv.

    Differences from standard ConformerBlock:
    1. Self-attention uses prefix-aware causal mask
    2. Depthwise conv uses LEFT-ONLY padding (no future leak)
    """
    def __init__(self, d_model, n_heads, ff_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        ff_dim   = d_model * ff_mult
        head_dim = d_model // n_heads
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.conv_kernel = conv_kernel

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

        # Depthwise conv (GLU gating + causal depthwise + LayerNorm)
        self.conv_norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model * 2)
        # No padding in Conv1d — we handle causal padding manually
        self.dw  = nn.Conv1d(d_model, d_model, conv_kernel,
                             padding=0, groups=d_model)
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

    def forward(self, x, mask=None):
        """
        x:    [B, T, D]
        mask: [T, T] bool where True = cannot attend (computed once, shared across layers)
        """
        # FFN 1 (1/2 step)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        # Causal self-attention with RoPE
        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out  = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # Causal depthwise conv (LEFT-ONLY padding)
        h = self.conv_norm(x)
        h, gate = self.pw1(h).chunk(2, dim=-1)
        h = h * F.silu(gate)
        # Causal: pad left by (kernel_size - 1), no right padding
        h = F.pad(h.transpose(1, 2), (self.conv_kernel - 1, 0))
        h = self.dw(h).transpose(1, 2)   # [B, T, D]
        h = self.pw2(self.ln_conv(h))
        x = x + self.conv_drop(h)

        # FFN 2 (1/2 step)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


# -- CausalConformerVC model -----------------------------------------------

class CausalConformerVC(nn.Module):
    """
    Autoregressive voice conversion with prefix conditioning.

    Input sequence: [prefix_ref | SEP | source_with_teacher_forcing]
    Prefix: target speaker reference (all 19 CBs summed per frame)
    Source: CB0 (always known) + shifted-right CB1-18 (teacher forced)
    Output: CB1-18 logits per source frame + speaker embedding prediction
    """
    def __init__(self, n_cb=18, cb_size=2048, d_model=1024, n_heads=16,
                 n_layers=14, conv_kernel=31, spk_emb_dim=256, dropout=0.1):
        super().__init__()
        self.n_cb    = n_cb
        self.d_model = d_model
        self.cb_size = cb_size

        # 19 codebook embeddings (CB0 through CB18)
        self.cb_embeds = nn.ModuleList([
            nn.Embedding(cb_size, d_model) for _ in range(n_cb + 1)
        ])
        self.embed_scale = 1.0 / math.sqrt(n_cb + 1)

        # SEP token (learnable)
        self.sep_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Speaker conditioning
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.film = FiLMLayer(d_model, d_model)

        # Causal Conformer layers
        self.layers = nn.ModuleList([
            CausalConformerBlock(d_model, n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output: 18 heads for CB1-18
        self.heads = nn.ModuleList([nn.Linear(d_model, cb_size) for _ in range(n_cb)])

        # Speaker identity head
        self.spk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, spk_emb_dim))

    def forward(self, prefix_codes, src_cb0, target_cb, spk_emb):
        """
        prefix_codes: [B, 19, T_ref]  -- target speaker reference (all CBs)
        src_cb0:      [B, T_src]       -- source content CB0
        target_cb:    [B, 18, T_src]   -- ground truth CB1-18 (for teacher forcing)
        spk_emb:      [B, 256]         -- target speaker embedding
        Returns: (logits [B, N_CB, T_src, V], spk_pred [B, 256])
        """
        B = src_cb0.shape[0]
        T_ref = prefix_codes.shape[2]
        T_src = src_cb0.shape[1]
        N = self.n_cb

        # Embed prefix: sum all 19 codebook embeddings per frame
        x_prefix = sum(self.cb_embeds[i](prefix_codes[:, i]) for i in range(N + 1))
        x_prefix = x_prefix * self.embed_scale  # [B, T_ref, D]

        # SEP token
        sep = self.sep_embed.expand(B, -1, -1)  # [B, 1, D]

        # Embed source: CB0 always known + teacher-forced CB1-18 (shifted right)
        x_src = self.cb_embeds[0](src_cb0)  # [B, T_src, D]
        for i in range(N):
            # Shift right by 1: pad start with 0, drop last
            shifted = F.pad(target_cb[:, i, :-1], (1, 0), value=0)
            x_src = x_src + self.cb_embeds[i + 1](shifted)
        x_src = x_src * self.embed_scale  # [B, T_src, D]

        # Concatenate: [prefix | SEP | source]
        x = torch.cat([x_prefix, sep, x_src], dim=1)  # [B, T_ref+1+T_src, D]

        # Speaker conditioning
        spk = self.spk_proj(spk_emb)
        x = self.film(x, spk)

        # Build causal mask once, pass to all layers
        T_total = T_ref + 1 + T_src
        prefix_len = T_ref + 1  # prefix + SEP
        mask = _prefix_causal_mask(T_total, prefix_len, x.device)

        # Causal Conformer
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)

        # Output: only predict in source region
        x_out = x[:, prefix_len:]  # [B, T_src, D]
        logits = torch.stack([h(x_out) for h in self.heads], dim=1)  # [B, N_CB, T_src, V]
        spk_pred = F.normalize(self.spk_head(x_out.mean(dim=1)), dim=-1)  # [B, 256]
        return logits, spk_pred

    def forward_cached(self, x_full, prefix_len):
        """Run full forward on concatenated sequence, return logits at last position.

        Used during AR inference: x_full = [prefix | SEP | frames_so_far]
        Returns logits for the last frame only.
        """
        B, T_total, D = x_full.shape
        mask = _prefix_causal_mask(T_total, prefix_len, x_full.device)

        h = x_full
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.norm(h)

        # Only need last position
        h_last = h[:, -1:]  # [B, 1, D]
        logits = torch.stack([head(h_last) for head in self.heads], dim=1)  # [B, N_CB, 1, V]
        return logits

    def compute_loss(self, logits, targets, spk_pred, spk_emb,
                     focal_gamma=1.0, secs_weight=0.1):
        """Focal CE over N_CB codebooks + SECS speaker similarity loss."""
        B, n_cb, T, V = logits.shape
        ce_total = 0.0
        for i in range(n_cb):
            log_p = F.log_softmax(logits[:, i].float().reshape(-1, V), dim=-1)
            tgt_i = targets[:, i].reshape(-1)
            lp_t  = log_p.gather(1, tgt_i.unsqueeze(1)).squeeze(1)
            if focal_gamma > 0:
                w = (1.0 - lp_t.exp().detach()) ** focal_gamma
            else:
                w = torch.ones_like(lp_t)
            ce_total += -(w * lp_t).mean()
        ce_loss = ce_total / n_cb

        spk_tgt = F.normalize(spk_emb.float(), dim=-1)
        secs_loss = (1.0 - (spk_pred * spk_tgt).sum(dim=-1)).mean()
        total = ce_loss + secs_weight * secs_loss
        return total, ce_loss, secs_loss


# -- Data ------------------------------------------------------------------

def load_data(args):
    """Load processed speaker data (prepared_data.pt + speaker_embeddings.pkl). No FAISS."""
    cache_path = os.path.join(args.data_processed, 'prepared_data.pt')
    emb_path   = os.path.join(args.data_processed, 'speaker_embeddings.pkl')

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f'Processed cache not found: {cache_path}\n'
            f'Run preprocessing first to build the cache.')

    print(f'Loading processed cache from {args.data_processed}...', flush=True)
    cached = torch.load(cache_path, weights_only=False, map_location='cpu')
    with open(emb_path, 'rb') as f:
        emb_data = pickle.load(f)

    speaker_data       = cached['speaker_data']
    speaker_embeddings = emb_data['speaker_embeddings']
    spk_list = list(speaker_data.keys())
    print(f'  {len(speaker_data)} speakers loaded', flush=True)

    total_frames = sum(sd['codes'].shape[1] for sd in speaker_data.values())
    print(f'  {total_frames} total frames ({total_frames / 12.5 / 3600:.1f}h)', flush=True)

    # Extract CB0 embeddings from Mimi (needed for FAISS baseline at inference)
    cb0_emb = None
    if args.phase in ('infer', 'all'):
        print('Extracting CB0 embeddings from Mimi...', flush=True)
        from transformers import MimiModel
        _mimi = MimiModel.from_pretrained('kyutai/mimi').eval()
        q = _mimi.quantizer
        if hasattr(q, 'semantic_residual_vector_quantizer'):
            layer = q.semantic_residual_vector_quantizer.layers[0]
        else:
            layer = q.layers[0]
        for attr in ['codebook.embed', 'codebook.embed_sum', 'codebook.embedding.weight']:
            obj = layer
            try:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
                w = obj.detach().cpu().float()
                if w.dim() == 3:
                    w = w.squeeze(0)
                if w.shape[0] >= _mimi.config.codebook_size:
                    cb0_emb = w.numpy()
                    break
            except AttributeError:
                continue
        print(f'  CB0 embeddings: {cb0_emb.shape}', flush=True)
        del _mimi

    return {
        'speaker_data':       speaker_data,
        'speaker_embeddings': speaker_embeddings,
        'spk_list':           spk_list,
        'N_CB':               args.n_cb,
        'FRAME_RATE':         12.5,
        'cb0_emb':            cb0_emb,
    }


# -- Batch sampler (cross-speaker, returns prefix + src + target) ----------

def sample_batch(data, args, device):
    speaker_data       = data['speaker_data']
    speaker_embeddings = data['speaker_embeddings']
    spk_list           = data['spk_list']
    N_CB               = data['N_CB']
    T_REF              = args.t_ref
    T_SRC              = args.t_src

    def make_one(rng):
        # 1. Pick target speaker
        tgt_spk = rng.choice(spk_list)
        tgt_codes = speaker_data[tgt_spk]['codes']  # [32, T]
        T_total = tgt_codes.shape[1]

        # Need at least T_REF + T_SRC frames
        min_len = T_REF + T_SRC
        if T_total < min_len:
            tgt_codes = F.pad(tgt_codes, (0, min_len - T_total))
            T_total = min_len

        # 2. Reference from first half, target from second half (non-overlapping)
        half = T_total // 2
        ref_start = rng.randint(0, max(0, half - T_REF))
        ref_crop  = tgt_codes[:, ref_start:ref_start + T_REF]  # [32, T_REF]
        if ref_crop.shape[1] < T_REF:
            ref_crop = F.pad(ref_crop, (0, T_REF - ref_crop.shape[1]))

        tgt_start = rng.randint(half, max(half, T_total - T_SRC))
        tgt_crop  = tgt_codes[:, tgt_start:tgt_start + T_SRC]  # [32, T_SRC]
        if tgt_crop.shape[1] < T_SRC:
            tgt_crop = F.pad(tgt_crop, (0, T_SRC - tgt_crop.shape[1]))

        # 3. Source CB0: cross-speaker or same-speaker
        if rng.random() < args.cross_speaker_ratio:
            src_spk = rng.choice(spk_list)
            while src_spk == tgt_spk and len(spk_list) > 1:
                src_spk = rng.choice(spk_list)
            src_codes = speaker_data[src_spk]['codes']
            T_s = src_codes.shape[1]
            if T_s < T_SRC:
                src_codes = F.pad(src_codes, (0, T_SRC - T_s))
                T_s = T_SRC
            s_start = rng.randint(0, max(0, T_s - T_SRC))
            src_cb0 = src_codes[0, s_start:s_start + T_SRC]
        else:
            src_cb0 = tgt_crop[0]

        if src_cb0.shape[0] < T_SRC:
            src_cb0 = F.pad(src_cb0, (0, T_SRC - src_cb0.shape[0]))

        # 4. Prefix codes: first 19 CBs (CB0..CB18) from reference
        prefix_codes = ref_crop[:N_CB + 1]  # [19, T_REF]

        # 5. Target CB1-{n_cb} from target crop
        target_cb = tgt_crop[1:1 + N_CB]  # [N_CB, T_SRC]

        # 6. Speaker embedding
        spk_emb = torch.tensor(speaker_embeddings[tgt_spk], dtype=torch.float32)

        return prefix_codes, src_cb0, target_cb, spk_emb

    rngs = [random.Random() for _ in range(args.batch_size)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(make_one, rngs))

    prefix_batch, src_batch, target_batch, spk_batch = zip(*results)
    return (
        torch.stack(list(prefix_batch)).to(device),   # [B, 19, T_REF]
        torch.stack(list(src_batch)).to(device),       # [B, T_SRC]
        torch.stack(list(target_batch)).to(device),    # [B, N_CB, T_SRC]
        torch.stack(list(spk_batch)).to(device),       # [B, 256]
    )


# -- Prefetch queue --------------------------------------------------------

class PrefetchQueue:
    def __init__(self, data, args, device, maxsize=4):
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


# -- GPU stats -------------------------------------------------------------

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


# -- Training --------------------------------------------------------------

def train(data, args, device):
    cfg   = MODEL_CONFIGS[args.model_size]
    steps = args.steps
    N_CB  = data['N_CB']

    model = CausalConformerVC(
        n_cb=N_CB, cb_size=2048,
        spk_emb_dim=256, dropout=args.dropout, **cfg
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*70}', flush=True)
    print(f'CausalConformerVC [{args.model_size.upper()}]', flush=True)
    print(f'  d_model={cfg["d_model"]}, n_heads={cfg["n_heads"]}, '
          f'n_layers={cfg["n_layers"]}, conv_kernel={cfg["conv_kernel"]}', flush=True)
    print(f'  Params: {n_params/1e6:.1f}M', flush=True)
    print(f'  Steps: {steps} | Batch: {args.batch_size} | LR: {args.lr}', flush=True)
    print(f'  T_ref={args.t_ref} T_src={args.t_src} | '
          f'cross_speaker: {args.cross_speaker_ratio:.0%}', flush=True)
    print(f'  Focal gamma: {args.focal_gamma} | SECS weight: {args.secs_weight}', flush=True)
    print(f'  AR mode: teacher forcing (train) | temperature={args.temperature} (infer)', flush=True)
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

    ckpt_path      = os.path.join(args.ckpt_dir, f'ar_vc_{args.model_size}.pt')
    best_ckpt_path = os.path.join(args.ckpt_dir, f'ar_vc_{args.model_size}_best.pt')
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
    t0          = time.time()
    log_every   = 200
    save_every  = 2000
    infer_every = 5000
    losses      = []
    ce_losses   = []
    secs_losses = []

    if N_CB <= 18:
        REPR_CBS = [0, 5, 11, 17]
    else:
        REPR_CBS = [0, 5, 11, 17, 23, min(30, N_CB - 1)]
    REPR_CBS = [i for i in REPR_CBS if i < N_CB]
    cb_accs  = {i: [] for i in REPR_CBS}

    prefetch = PrefetchQueue(data, args, device, maxsize=4)
    _pending_save = None

    for step in range(start_step, steps):
        prefix_b, src_b, target_b, spk_b = prefetch.get()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            logits, spk_pred = model(prefix_b, src_b, target_b, spk_b)
            loss, ce_loss, secs_loss = model.compute_loss(
                logits, target_b, spk_pred, spk_b,
                args.focal_gamma, args.secs_weight)

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

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                for i in REPR_CBS:
                    pred = logits[:, i].float().argmax(-1).reshape(-1)
                    gt   = target_b[:, i].reshape(-1)
                    cb_accs[i].append((pred == gt).float().mean().item() * 100)
            sps = (step + 1 - start_step) / (time.time() - t0)
            eta = (steps - step - 1) / sps
            avg     = np.mean(losses[-log_every:])
            avg_ce  = np.mean(ce_losses[-log_every:])
            avg_sec = np.mean(secs_losses[-log_every:])
            cb_str  = ' '.join(
                f'CB{i+1}:{cb_accs[i][-1]:.0f}%' for i in REPR_CBS)
            gpu_str = _gpu_stats()
            print(f'[{args.model_size}] {step+1:6d}/{steps} | '
                  f'Loss:{avg:.4f} CE:{avg_ce:.4f} SECS:{avg_sec:.3f} | '
                  f'{cb_str} | {sps:.1f} step/s | ETA {eta:.0f}s'
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
                voice_convert(data, model, args, device, suffix=f'-{step+1}-step')
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
    print(f'\nTraining done [{args.model_size}]: {elapsed:.0f}s ({elapsed/60:.1f}min)', flush=True)
    for i in REPR_CBS:
        if cb_accs[i]:
            print(f'  CB{i+1}: {cb_accs[i][-1]:.1f}%', flush=True)
    return model


# -- Mimi helpers ----------------------------------------------------------

def load_mimi(device):
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    mimi     = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR       = feat_ext.sampling_rate
    return mimi, feat_ext, SR, dtype


# -- Inference: AR voice conversion ----------------------------------------

def voice_convert(data, model, args, device, suffix=''):
    """AR voice conversion inference.

    Pipeline:
    1. Encode source -> CB0
    2. Encode target reference -> all 32 CBs (use first 19 as prefix)
    3. AR generate CB1-18 frame by frame with temperature sampling
    4. Assemble full codes (src CB0 + predicted CB1-18 + FAISS CB19-31)
    5. Decode with Mimi
    """
    if not args.source_audio or not args.target_dir:
        print('Skipping inference: --source_audio and --target_dir required.', flush=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    N_CB       = data['N_CB']
    FRAME_RATE = data['FRAME_RATE']

    mimi, feat_ext, SR, dtype = load_mimi(device)

    def encode(wav):
        CHUNK = int(30 * SR)
        if isinstance(wav, list):
            wav = np.concatenate(wav)
        codes_list = []
        for i in range(0, len(wav), CHUNK):
            chunk = wav[i:i + CHUNK]
            if len(chunk) < SR:
                continue
            inp = feat_ext(raw_audio=chunk, sampling_rate=SR, return_tensors='pt')
            inp = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None)
                   for k, v in inp.items()}
            with torch.no_grad():
                codes = mimi.encode(inp['input_values'], inp.get('padding_mask')).audio_codes.cpu()
            codes_list.append(codes)
        return torch.cat(codes_list, dim=2)

    def decode(codes):
        with torch.no_grad():
            return mimi.decode(codes.to(device)).audio_values.cpu().squeeze().numpy().astype(np.float32)

    # Source
    src_wav   = librosa.load(args.source_audio, sr=SR, mono=True)[0].astype(np.float32)
    src_codes = encode(src_wav)
    src_cb0   = src_codes[0, 0]  # [T_src]
    T_src     = src_cb0.shape[0]
    print(f'Source: {T_src} frames ({T_src / FRAME_RATE:.1f}s)', flush=True)

    # Target reference
    tgt_wavs = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_wavs.append(librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32))
    tgt_concat = np.concatenate(tgt_wavs)
    tgt_codes = encode(tgt_concat)  # [1, 32, T_tgt]
    ref_all = tgt_codes[0]  # [32, T_tgt]
    # Use first 19 CBs as prefix, truncate to reasonable length
    MAX_REF = min(750, ref_all.shape[1])
    ref_codes = ref_all[:N_CB + 1, :MAX_REF]  # [19, T_ref]
    T_ref = ref_codes.shape[1]
    print(f'Target reference: {T_ref} frames ({T_ref / FRAME_RATE:.1f}s)', flush=True)

    # Target speaker embedding
    from resemblyzer import VoiceEncoder
    spk_enc = VoiceEncoder(device='cpu')
    tgt16   = librosa.resample(tgt_concat, orig_sr=SR, target_sr=16000)
    spk_emb = torch.tensor(spk_enc.embed_utterance(tgt16), dtype=torch.float32)

    # FAISS retrieval for baseline + CB19-31 passthrough
    import faiss
    cb0_emb = data['cb0_emb']
    tgt_full = tgt_codes[0].numpy().T  # [T_tgt, 32]
    tgt_sem  = tgt_codes[0, 0].numpy()
    tgt_emb  = cb0_emb[tgt_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb)
    tgt_index = faiss.IndexFlatIP(cb0_emb.shape[1])
    tgt_index.add(tgt_emb)

    src_sem_np = src_codes[0, 0].numpy()
    src_emb_q  = cb0_emb[src_sem_np].astype(np.float32).copy()
    faiss.normalize_L2(src_emb_q)
    dists_k, indices_k = tgt_index.search(src_emb_q, 8)

    selected = np.zeros(T_src, dtype=np.int64)
    selected[0] = indices_k[0, 0]
    for t in range(1, T_src):
        cands    = indices_k[t]
        combined = 0.5 * dists_k[t] + 0.5 / (1 + np.abs(cands - selected[t - 1]) / FRAME_RATE)
        selected[t] = cands[np.argmax(combined)]

    # FAISS baseline: all 32 CBs from target
    faiss_codes = src_codes.clone()
    for t in range(T_src):
        faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])
    faiss_wav = decode(faiss_codes)

    # AR generation frame by frame
    model.eval()
    temperature = args.temperature
    with torch.no_grad():
        # Build prefix embedding
        ref_d = ref_codes.unsqueeze(0).to(device)  # [1, 19, T_ref]
        spk_d = spk_emb.unsqueeze(0).to(device)    # [1, 256]
        src_d = src_cb0.to(device)                  # [T_src]

        # Embed prefix: sum all 19 CBs
        x_prefix = sum(model.cb_embeds[i](ref_d[:, i]) for i in range(N_CB + 1))
        x_prefix = x_prefix * model.embed_scale  # [1, T_ref, D]

        # SEP
        sep = model.sep_embed  # [1, 1, D]

        # Speaker conditioning on prefix+sep
        spk = model.spk_proj(spk_d)
        prefix_conditioned = model.film(torch.cat([x_prefix, sep], dim=1), spk)  # [1, T_ref+1, D]

        prefix_len = T_ref + 1
        cur = prefix_conditioned  # [1, prefix_len, D]

        predicted = []
        print(f'AR generation: {T_src} frames...', flush=True)
        t_start = time.time()

        for t in range(T_src):
            # Current frame input: CB0 + previous predicted CB1-18
            frame_emb = model.cb_embeds[0](src_d[t:t + 1].unsqueeze(0))  # [1, 1, D]
            if predicted:
                for i in range(N_CB):
                    frame_emb = frame_emb + model.cb_embeds[i + 1](predicted[-1][:, i:i + 1])
            frame_emb = frame_emb * model.embed_scale

            # Apply speaker conditioning to new frame
            frame_emb = model.film(frame_emb, spk)

            # Append to sequence
            cur = torch.cat([cur, frame_emb], dim=1)  # [1, prefix_len+t+1, D]

            # Forward through all layers (with causal mask)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
                logits = model.forward_cached(cur, prefix_len)  # [1, N_CB, 1, V]

            # Temperature sampling for each codebook
            frame_logits = logits[:, :, 0].float()  # [1, N_CB, V]
            frame_logits = frame_logits / temperature
            probs = F.softmax(frame_logits, dim=-1)  # [1, N_CB, V]
            frame_tokens = torch.multinomial(
                probs.reshape(-1, model.cb_size), num_samples=1
            ).reshape(1, N_CB)  # [1, N_CB]
            predicted.append(frame_tokens)

            if (t + 1) % 100 == 0:
                elapsed = time.time() - t_start
                fps = (t + 1) / elapsed
                print(f'  Frame {t+1}/{T_src} ({fps:.1f} frames/s)', flush=True)

        # Assemble output: [1, N_CB, T_src]
        pred_cbs = torch.stack(predicted, dim=-1).cpu()  # [1, N_CB, T_src]

    gen_time = time.time() - t_start
    print(f'AR generation done: {T_src} frames in {gen_time:.1f}s ({T_src/gen_time:.1f} frames/s)', flush=True)

    # Assemble full codes: src CB0 + predicted CB1-18 + FAISS CB19-31
    full_codes = faiss_codes.clone()
    full_codes[0, 0] = src_cb0
    full_codes[0, 1:1 + N_CB] = pred_cbs[0]

    # Decode
    vc_wav = decode(full_codes)

    # Metrics
    def secs_metric(a, b):
        a16 = librosa.resample(a.astype(np.float32), orig_sr=SR, target_sr=16000)
        b16 = librosa.resample(b.astype(np.float32), orig_sr=SR, target_sr=16000)
        ea  = spk_enc.embed_utterance(a16)
        eb  = spk_enc.embed_utterance(b16)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))

    def mcd_metric(ref, syn):
        r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=SR, n_mfcc=13)
        s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=SR, n_mfcc=13)
        L = min(r.shape[1], s.shape[1])
        return float(np.mean(np.sqrt(2 * np.sum((r[1:, :L] - s[1:, :L]) ** 2, axis=0))))

    tgt_ref = tgt_concat[:int(SR * 15)]
    print(f'\n--- AR Voice Conversion results [{args.model_size}] ---', flush=True)
    print(f'{"Method":<45} {"SECS(->tgt)":>12} {"MCD":>8}', flush=True)
    print('-' * 67, flush=True)

    fs = secs_metric(faiss_wav, tgt_ref)
    fm = mcd_metric(src_wav, faiss_wav)
    print(f'{"FAISS baseline (all 32 CB)":<45} {fs:>12.4f} {fm:>8.2f}', flush=True)

    s = secs_metric(vc_wav, tgt_ref)
    m = mcd_metric(src_wav, vc_wav)
    print(f'{f"AR-VC T={temperature}":<45} {s:>12.4f} {m:>8.2f}', flush=True)

    # Save
    if not suffix:
        sf.write(os.path.join(args.output_dir, 'faiss_baseline.wav'), faiss_wav, SR)
    out_path = os.path.join(args.output_dir, f'ar_vc_{args.model_size}{suffix}.wav')
    sf.write(out_path, vc_wav, SR)
    print(f'Saved to: {out_path}', flush=True)

    del mimi
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


# -- Main ------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}', flush=True)
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB', flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    data  = load_data(args)
    model = None

    if args.phase in ('train', 'all'):
        model = train(data, args, device)

    if args.phase in ('infer', 'all'):
        if model is None:
            cfg = MODEL_CONFIGS[args.model_size]
            model = CausalConformerVC(
                n_cb=data['N_CB'], cb_size=2048,
                spk_emb_dim=256, dropout=args.dropout, **cfg
            ).to(device)
            best_path = os.path.join(args.ckpt_dir, f'ar_vc_{args.model_size}_best.pt')
            ckpt = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            print(f'Loaded CausalConformerVC [{args.model_size}] from {best_path}', flush=True)
        voice_convert(data, model, args, device)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
