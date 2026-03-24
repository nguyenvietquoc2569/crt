#!/usr/bin/env python3
"""
v6.0 — Stage 1 AR (Autoregressive) semantic model for voice conversion.

Given source CB0 tokens + target speaker embedding, generates target CB0
tokens autoregressively.  Adjusts prosody, rhythm, and phoneme realization
to match the target speaker's speaking style.

Architecture: Causal Transformer Decoder with FiLM speaker conditioning.
  d_model=768, n_heads=12, n_layers=12 → ~150M params
  Vocabulary: 2049 (2048 CB0 tokens + 1 separator token)
  Max sequence: 512 (256 source + 1 sep + 255 target)
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
    'small':  dict(d_model=256,  n_heads=4,  n_layers=4),
    'medium': dict(d_model=512,  n_heads=8,  n_layers=8),
    'large':  dict(d_model=768,  n_heads=12, n_layers=12),
    'xlarge': dict(d_model=1024, n_heads=16, n_layers=16),
}

DEFAULT_STEPS = {
    'small': 20000, 'medium': 60000, 'large': 120000, 'xlarge': 200000,
}


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_processed', required=True)
    p.add_argument('--ckpt_dir',       required=True)
    p.add_argument('--output_dir',     required=True)
    p.add_argument('--model_size',     choices=list(MODEL_CONFIGS), default='large')
    p.add_argument('--steps',          type=int, default=0, help='0 = use size default')
    p.add_argument('--batch_size',     type=int, default=64)
    p.add_argument('--seq_len',        type=int, default=128)
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--temperature',    type=float, default=0.8)
    p.add_argument('--top_k',          type=int, default=50)
    p.add_argument('--same_speaker_ratio', type=float, default=0.5)
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--target_dir',     default=None)
    p.add_argument('--source_audio',   default=None)
    p.add_argument('--phase',          choices=['train', 'infer', 'all'], default='train')
    return p.parse_args()


# ── FiLM ───────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x, cond):
        g, b = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + g) + b


# ── Causal Transformer Block ──────────────────────────────────

def _causal_mask(seq_len, device):
    """Upper-triangular True mask for causal attention."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


class CausalTransformerBlock(nn.Module):
    """Standard transformer decoder block with causal masking."""
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        ff_dim = d_model * ff_mult

        # Pre-norm attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv       = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj     = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)

        # Pre-norm FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, D = x.shape

        # Self-attention with causal mask
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(_causal_mask(T, x.device).unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out  = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── AR Codec LM ───────────────────────────────────────────────

class ARCodecLM(nn.Module):
    """
    Autoregressive CB0 voice converter.
    Input:  source_CB0 [B, T_src] + target_speaker_emb [B, 256]
    Output: target_CB0 tokens [B, T_src] (generated autoregressively)
    """
    def __init__(self, vocab_size=2048, d_model=768, n_heads=12,
                 n_layers=12, spk_emb_dim=256, dropout=0.1, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.sep_token = vocab_size  # special separator token (index 2048)

        # Token embedding (vocab_size + 1 for separator)
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Speaker conditioning via FiLM (per layer)
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.films = nn.ModuleList([
            FiLMLayer(d_model, d_model) for _ in range(n_layers)])

        # Causal Transformer decoder
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        # Output head — predict over original vocab (not sep)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, spk_emb):
        """
        input_ids: [B, L] — concatenated [src_CB0, <SEP>, tgt_CB0[:-1]]
        spk_emb:   [B, 256]
        Returns:   logits [B, L, vocab_size]
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        spk = self.spk_proj(spk_emb)
        for i, layer in enumerate(self.layers):
            x = self.films[i](x, spk)
            x = layer(x)
        x = self.norm(x)
        return self.head(x)  # [B, L, vocab_size]

    @torch.no_grad()
    def generate(self, src_cb0, spk_emb, temperature=0.8, top_k=50):
        """
        Autoregressive generation.
        src_cb0: [B, T_src] — source CB0 tokens
        spk_emb: [B, 256]
        Returns: target_cb0 [B, T_src]
        """
        B, T = src_cb0.shape
        # Start with source + separator
        sep = torch.full((B, 1), self.sep_token, dtype=torch.long, device=src_cb0.device)
        prefix = torch.cat([src_cb0, sep], dim=1)  # [B, T+1]

        generated = []
        for t in range(T):
            logits = self.forward(prefix, spk_emb)  # [B, L, V]
            next_logits = logits[:, -1]  # [B, V]

            # Temperature + top-k sampling
            next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, topk_idx = next_logits.topk(top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, topk_idx, topk_vals)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [B, 1]
            generated.append(next_token)
            prefix = torch.cat([prefix, next_token], dim=1)

        return torch.cat(generated, dim=1)  # [B, T]


# ── Data ───────────────────────────────────────────────────────

def load_data(args):
    """Load processed speaker data and build FAISS indices for cross-speaker alignment."""
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

    # Build per-speaker FAISS indices for cross-speaker alignment
    faiss.omp_set_num_threads(1)
    print('Building FAISS indices...', flush=True)
    speaker_faiss = {}
    for spk_id, sdata in speaker_data.items():
        codes = sdata['codes']
        sem   = codes[0].numpy()
        full  = codes.numpy().T  # [T, 32]
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
        'FRAME_RATE':         12.5,
    }


# ── Pair construction ──────────────────────────────────────────

def make_pair(rng, speaker_data, speaker_faiss, speaker_embeddings,
              cb0_emb, seq_len, same_speaker_ratio):
    """Create a (source_CB0, target_CB0, target_spk_emb) training triple."""
    spk_list = list(speaker_data.keys())

    # Pick target speaker and segment
    tgt_spk = rng.choice(spk_list)
    tgt_codes = speaker_data[tgt_spk]['codes']
    T = tgt_codes.shape[1]
    start = rng.randint(0, max(0, T - seq_len))
    tgt_cb0 = tgt_codes[0, start:start + seq_len]
    if tgt_cb0.shape[0] < seq_len:
        tgt_cb0 = F.pad(tgt_cb0, (0, seq_len - tgt_cb0.shape[0]))

    # Same-speaker pair (identity preservation) or cross-speaker pair
    if rng.random() < same_speaker_ratio:
        # Same speaker, different segment → learns to reproduce speaker style
        src_spk = tgt_spk
        src_codes = speaker_data[src_spk]['codes']
        src_T = src_codes.shape[1]
        src_start = rng.randint(0, max(0, src_T - seq_len))
        src_cb0 = src_codes[0, src_start:src_start + seq_len]
        if src_cb0.shape[0] < seq_len:
            src_cb0 = F.pad(src_cb0, (0, seq_len - src_cb0.shape[0]))
    else:
        # Cross-speaker: FAISS-align source frames to target content
        src_spk = rng.choice(spk_list)
        while src_spk == tgt_spk:
            src_spk = rng.choice(spk_list)

        src_faiss = speaker_faiss[src_spk]
        tgt_emb = cb0_emb[tgt_cb0.numpy()].astype(np.float32).copy()
        faiss.normalize_L2(tgt_emb)
        dists, indices = src_faiss['index'].search(tgt_emb, 1)
        src_cb0 = torch.tensor(
            [src_faiss['full_codes'][idx[0]][0] for idx in indices],
            dtype=torch.long)

    # Target speaker embedding
    spk_emb = torch.tensor(speaker_embeddings[tgt_spk], dtype=torch.float32)

    return src_cb0, tgt_cb0, spk_emb


# ── Batch sampler ──────────────────────────────────────────────

def sample_batch(data, args, device):
    speaker_data       = data['speaker_data']
    speaker_embeddings = data['speaker_embeddings']
    speaker_faiss      = data['speaker_faiss']
    cb0_emb            = data['cb0_emb']
    SEQ_LEN            = args.seq_len

    def make_one(rng):
        return make_pair(rng, speaker_data, speaker_faiss, speaker_embeddings,
                         cb0_emb, SEQ_LEN, args.same_speaker_ratio)

    rngs = [random.Random() for _ in range(args.batch_size)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(make_one, rngs))

    src_batch, tgt_batch, spk_batch = zip(*results)
    return (
        torch.stack(list(src_batch)).to(device),
        torch.stack(list(tgt_batch)).to(device),
        torch.stack(list(spk_batch)).to(device),
    )


# ── Prefetch queue ─────────────────────────────────────────────

class PrefetchQueue:
    """Background thread that runs sample_batch() continuously so the GPU
    never stalls waiting for CPU FAISS work."""

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


# ── Training ───────────────────────────────────────────────────

def train(data, args, device):
    cfg    = MODEL_CONFIGS[args.model_size]
    steps  = args.steps if args.steps > 0 else DEFAULT_STEPS[args.model_size]
    CB_SIZE = data['CB_SIZE']

    model = ARCodecLM(
        vocab_size=CB_SIZE,
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        spk_emb_dim=256,
        dropout=args.dropout,
        max_len=args.seq_len * 2 + 2,  # src + sep + tgt
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*70}')
    print(f'ARCodecLM [{args.model_size.upper()}]')
    print(f'  d_model={cfg["d_model"]}, n_heads={cfg["n_heads"]}, '
          f'n_layers={cfg["n_layers"]}')
    print(f'  Params: {n_params/1e6:.1f}M')
    print(f'  Vocab: {CB_SIZE}+1sep={CB_SIZE+1} | Max seq: {args.seq_len*2+1}')
    print(f'  Steps: {steps} | Batch: {args.batch_size} | LR: {args.lr}')
    print(f'  Same-speaker ratio: {args.same_speaker_ratio:.0%}')
    print(f'{"="*70}', flush=True)

    use_amp = device == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        warmup = 1000
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ckpt_path      = os.path.join(args.ckpt_dir, f'ar_{args.model_size}.pt')
    best_ckpt_path = os.path.join(args.ckpt_dir, f'ar_{args.model_size}_best.pt')
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
    save_every = 2000
    infer_every = 20000
    losses     = []
    accs       = []

    prefetch = PrefetchQueue(data, args, device, maxsize=4)
    _pending_save = None

    for step in range(start_step, steps):
        src_cb0, tgt_cb0, spk_emb = prefetch.get()
        B = src_cb0.shape[0]
        V = model.vocab_size

        # Teacher forcing: [src_CB0, <SEP>, tgt_CB0[:-1]]
        sep = torch.full((B, 1), model.sep_token, dtype=torch.long, device=device)
        input_ids = torch.cat([src_cb0, sep, tgt_cb0[:, :-1]], dim=1)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            logits = model(input_ids, spk_emb)  # [B, L, V]

            # Loss: only on target portion (after separator)
            tgt_start = src_cb0.shape[1] + 1  # after source + separator
            tgt_logits = logits[:, tgt_start:]  # [B, T_tgt, V]
            loss = F.cross_entropy(tgt_logits.reshape(-1, V), tgt_cb0.reshape(-1))

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
                pred = tgt_logits.float().argmax(-1).reshape(-1)
                gt   = tgt_cb0.reshape(-1)
                acc  = (pred == gt).float().mean().item() * 100
                accs.append(acc)
            sps = (step + 1 - start_step) / (time.time() - t0)
            eta = (steps - step - 1) / sps
            avg = np.mean(losses[-log_every:])
            gpu_str = _gpu_stats()
            print(f'  [{args.model_size}] {step+1:5d}/{steps} | '
                  f'Loss:{avg:.4f} | Acc:{acc:.1f}% | '
                  f'{sps:.1f} step/s | ETA {eta:.0f}s'
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
                run_inference(data, model, args, device, suffix=f'-{step+1}-step')
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
    if accs:
        print(f'  Final accuracy: {accs[-1]:.1f}%')
    return model


# ── Inference ──────────────────────────────────────────────────

def load_mimi(device):
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    mimi     = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR       = feat_ext.sampling_rate
    return mimi, feat_ext, SR, dtype


def run_inference(data, model, args, device, suffix=''):
    """AR generation: source audio → CB0 → AR generate target CB0 → decode."""
    if not args.source_audio or not args.target_dir:
        print('Skipping inference: --source_audio and --target_dir required.')
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
    src_codes = encode(src_wav)  # [1, 32, T]
    src_cb0   = src_codes[0, 0]  # [T]
    T_src     = src_cb0.shape[0]
    print(f'Source: {T_src} frames ({T_src/FRAME_RATE:.1f}s)')

    # Target speaker embedding
    tgt_wavs = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_wavs.append(librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32))
    tgt_concat = np.concatenate(tgt_wavs)

    from resemblyzer import VoiceEncoder
    spk_enc = VoiceEncoder(device='cpu')
    tgt16   = librosa.resample(tgt_concat, orig_sr=SR, target_sr=16000)
    tgt_spk_emb = torch.tensor(spk_enc.embed_utterance(tgt16), dtype=torch.float32)

    # AR generate target CB0
    model.eval()
    tgt_cb0 = model.generate(
        src_cb0.unsqueeze(0).to(device),
        tgt_spk_emb.unsqueeze(0).to(device),
        temperature=args.temperature,
        top_k=args.top_k)  # [1, T]

    # For now, combine with source CB1-31 (will use NAR later)
    output_codes = src_codes.clone()
    output_codes[0, 0] = tgt_cb0[0].cpu()  # replace CB0 with generated
    ar_wav = decode(output_codes)

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
    print(f'\n--- AR inference results [{args.model_size}] ---')
    print(f'{"Method":<30} {"SECS(->tgt)":>12} {"MCD":>8}')
    print('-' * 52)
    for name, wav in [(f'AR-{args.model_size} (CB0 only)', ar_wav)]:
        print(f'{name:<30} {secs(wav, tgt_ref):>12.4f} {mcd(src_wav, wav):>8.2f}')

    sf.write(os.path.join(args.output_dir, f'ar_cb0_generated{suffix}.wav'), ar_wav, SR)
    print(f'Saved to: {args.output_dir}/ar_cb0_generated{suffix}.wav')


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

    if args.phase in ('infer', 'all'):
        if model is None:
            cfg = MODEL_CONFIGS[args.model_size]
            model = ARCodecLM(
                vocab_size=data['CB_SIZE'],
                d_model=cfg['d_model'],
                n_heads=cfg['n_heads'],
                n_layers=cfg['n_layers'],
                spk_emb_dim=256,
                dropout=args.dropout,
                max_len=args.seq_len * 2 + 2,
            ).to(device)
            best_path = os.path.join(args.ckpt_dir, f'ar_{args.model_size}_best.pt')
            ckpt = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            print(f'Loaded AR [{args.model_size}] from {best_path}')
        run_inference(data, model, args, device)

    print('\nDone.')


if __name__ == '__main__':
    main()
