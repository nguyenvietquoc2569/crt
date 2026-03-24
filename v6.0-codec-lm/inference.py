#!/usr/bin/env python3
"""
v6.0 Combined inference: AR (Stage 1) + NAR (Stage 2)

Two-stage codec language model pipeline for voice conversion:
  Stage 1 (AR):  source CB0 + spk_emb -> target CB0
  Stage 2 (NAR): target CB0 + spk_emb -> target CB1-31
  Combine all 32 CBs -> Mimi decode -> target audio

Generates multiple outputs for comparison:
  - Source audio (original, re-encoded)
  - FAISS baseline (nearest-neighbor retrieval)
  - AR-only (CB0 converted, CB1-31 from source)
  - NAR-only (CB0 from source, CB1-31 generated)
  - Full pipeline (AR + NAR)

Usage:
  python inference.py \\
      --source_audio source.mp3 \\
      --target_dir target_audio/ \\
      --ar_ckpt checkpoints_ar/ar_large_best.pt \\
      --nar_ckpt checkpoints_nar/nar_xlarge_best.pt \\
      --data_processed /path/to/processed_merged \\
      --output_dir output/
"""

import argparse
import os
import sys
import gc
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Import model classes from training scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ar import ARCodecLM, MODEL_CONFIGS as AR_CONFIGS
from train_nar import NARAcousticLM, MODEL_CONFIGS as NAR_CONFIGS


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='v6.0 Combined AR+NAR inference')
    p.add_argument('--source_audio',   required=True, help='Path to source audio file')
    p.add_argument('--target_dir',     required=True, help='Directory with target speaker audio')
    p.add_argument('--ar_ckpt',        required=True, help='Path to AR model checkpoint')
    p.add_argument('--nar_ckpt',       required=True, help='Path to NAR model checkpoint')
    p.add_argument('--data_processed', required=True, help='Path to processed data (for FAISS baseline)')
    p.add_argument('--output_dir',     default='output', help='Output directory for wav files')
    p.add_argument('--ar_size',        default='large',  choices=list(AR_CONFIGS))
    p.add_argument('--nar_size',       default='xlarge', choices=list(NAR_CONFIGS))
    p.add_argument('--temperature',    type=float, default=0.8,  help='AR sampling temperature')
    p.add_argument('--top_k',          type=int,   default=50,   help='AR top-k sampling')
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


# ── Mimi helpers ───────────────────────────────────────────────

def load_mimi(device):
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    mimi     = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR       = feat_ext.sampling_rate
    return mimi, feat_ext, SR, dtype


def encode_audio(mimi, feat_ext, wav, SR, device, dtype):
    """Encode a single waveform into Mimi codes [1, 32, T]."""
    CHUNK = int(30 * SR)
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
    if not codes_list:
        raise ValueError(f'Audio too short to encode (< {SR} samples)')
    return torch.cat(codes_list, dim=2)


def decode_audio(mimi, codes, device):
    """Decode Mimi codes [1, 32, T] to waveform."""
    with torch.no_grad():
        return mimi.decode(codes.to(device)).audio_values.cpu().squeeze().numpy().astype(np.float32)


# ── Speaker embedding ─────────────────────────────────────────

def get_speaker_embedding(wav, SR):
    """Compute speaker embedding using resemblyzer."""
    from resemblyzer import VoiceEncoder
    spk_enc = VoiceEncoder(device='cpu')
    wav16 = librosa.resample(wav.astype(np.float32), orig_sr=SR, target_sr=16000)
    emb = spk_enc.embed_utterance(wav16)
    return torch.tensor(emb, dtype=torch.float32), spk_enc


# ── FAISS baseline ─────────────────────────────────────────────

def faiss_baseline(src_codes, tgt_codes, cb0_emb, frame_rate=12.5):
    """Nearest-neighbor retrieval baseline using FAISS on CB0."""
    import faiss

    T_src = src_codes.shape[2]
    tgt_sem  = tgt_codes[0, 0].numpy()
    tgt_full = tgt_codes[0].numpy().T  # [T_tgt, 32]
    tgt_emb  = cb0_emb[tgt_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb)
    index = faiss.IndexFlatIP(cb0_emb.shape[1])
    index.add(tgt_emb)

    src_sem = src_codes[0, 0].numpy()
    src_emb = cb0_emb[src_sem].astype(np.float32).copy()
    faiss.normalize_L2(src_emb)
    dists_k, indices_k = index.search(src_emb, 8)

    # Temporal smoothing
    selected = np.zeros(T_src, dtype=np.int64)
    selected[0] = indices_k[0, 0]
    for t in range(1, T_src):
        cands    = indices_k[t]
        combined = (0.5 * dists_k[t]
                    + 0.5 / (1 + np.abs(cands - selected[t - 1]) / frame_rate))
        selected[t] = cands[np.argmax(combined)]

    faiss_codes = src_codes.clone()
    for t in range(T_src):
        faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])
    return faiss_codes


def extract_cb0_embeddings():
    """Extract CB0 codebook embeddings from Mimi quantizer."""
    from transformers import MimiModel
    mimi_q = MimiModel.from_pretrained('kyutai/mimi')
    CB_SIZE = mimi_q.config.codebook_size
    q = mimi_q.quantizer
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

    del mimi_q
    gc.collect()

    if cb0_emb is None:
        raise RuntimeError('Cannot extract CB0 embeddings from Mimi quantizer')
    return cb0_emb


# ── Metrics ────────────────────────────────────────────────────

def compute_secs(wav_a, wav_b, SR, spk_enc):
    """Speaker embedding cosine similarity."""
    a16 = librosa.resample(wav_a.astype(np.float32), orig_sr=SR, target_sr=16000)
    b16 = librosa.resample(wav_b.astype(np.float32), orig_sr=SR, target_sr=16000)
    ea = spk_enc.embed_utterance(a16)
    eb = spk_enc.embed_utterance(b16)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


def compute_mcd(ref, syn, SR):
    """Mel cepstral distortion."""
    r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=SR, n_mfcc=13)
    s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=SR, n_mfcc=13)
    L = min(r.shape[1], s.shape[1])
    return float(np.mean(np.sqrt(2 * np.sum((r[1:, :L] - s[1:, :L]) ** 2, axis=0))))


# ── Main ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')
    os.makedirs(args.output_dir, exist_ok=True)

    FRAME_RATE = 12.5

    # ── 1. Load Mimi codec ────────────────────────────────────
    print('\n[1/6] Loading Mimi codec...', flush=True)
    mimi, feat_ext, SR, dtype = load_mimi(device)

    # ── 2. Encode source audio ────────────────────────────────
    print('[2/6] Encoding source audio...', flush=True)
    src_wav   = librosa.load(args.source_audio, sr=SR, mono=True)[0].astype(np.float32)
    src_codes = encode_audio(mimi, feat_ext, src_wav, SR, device, dtype)  # [1, 32, T]
    T_src     = src_codes.shape[2]
    print(f'  Source: {T_src} frames ({T_src / FRAME_RATE:.1f}s)')

    # ── 3. Get target speaker embedding ───────────────────────
    print('[3/6] Computing target speaker embedding...', flush=True)
    tgt_wavs = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        for f in sorted(Path(args.target_dir).glob(ext)):
            tgt_wavs.append(librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32))
    if not tgt_wavs:
        raise FileNotFoundError(f'No audio files found in {args.target_dir}')
    tgt_concat = np.concatenate(tgt_wavs)
    tgt_codes  = encode_audio(mimi, feat_ext, tgt_concat, SR, device, dtype)

    spk_emb, spk_enc = get_speaker_embedding(tgt_concat, SR)
    print(f'  Target: {tgt_codes.shape[2]} frames ({tgt_codes.shape[2] / FRAME_RATE:.1f}s)')

    # ── 4. Load AR model + generate target CB0 ────────────────
    print('[4/6] AR Stage 1: source CB0 -> target CB0...', flush=True)
    ar_ckpt = torch.load(args.ar_ckpt, weights_only=False, map_location=device)
    ar_cfg  = AR_CONFIGS[args.ar_size]
    ar_model = ARCodecLM(
        vocab_size=2048,
        d_model=ar_cfg['d_model'],
        n_heads=ar_cfg['n_heads'],
        n_layers=ar_cfg['n_layers'],
        spk_emb_dim=256,
        dropout=0.0,
        max_len=T_src * 2 + 2,
    ).to(device)
    ar_model.load_state_dict(ar_ckpt['model_state'])
    ar_model.eval()
    print(f'  AR model: {args.ar_size} ({sum(p.numel() for p in ar_model.parameters())/1e6:.1f}M params)')

    src_cb0 = src_codes[0, 0].unsqueeze(0).to(device)  # [1, T]
    spk_d   = spk_emb.unsqueeze(0).to(device)           # [1, 256]

    ar_target_cb0 = ar_model.generate(
        src_cb0, spk_d,
        temperature=args.temperature,
        top_k=args.top_k)  # [1, T]
    print(f'  Generated {ar_target_cb0.shape[1]} target CB0 tokens')
    del ar_model, ar_ckpt
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── 5. Load NAR model + generate CB1-31 ───────────────────
    print('[5/6] NAR Stage 2: target CB0 -> CB1-31...', flush=True)
    nar_ckpt = torch.load(args.nar_ckpt, weights_only=False, map_location=device)
    nar_cfg  = NAR_CONFIGS[args.nar_size]
    nar_model = NARAcousticLM(
        n_cb=31, vocab_size=2048, spk_emb_dim=256, dropout=0.0, **nar_cfg
    ).to(device)
    nar_model.load_state_dict(nar_ckpt['model_state'])
    nar_model.eval()
    print(f'  NAR model: {args.nar_size} ({sum(p.numel() for p in nar_model.parameters())/1e6:.1f}M params)')

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
            # Full pipeline: AR CB0 -> NAR CB1-31
            logits_full, _ = nar_model(ar_target_cb0, spk_d)
            pred_full = logits_full.float().argmax(-1)[0].cpu()  # [31, T]

            # NAR-only: source CB0 -> NAR CB1-31
            logits_nar_only, _ = nar_model(src_cb0, spk_d)
            pred_nar_only = logits_nar_only.float().argmax(-1)[0].cpu()  # [31, T]

    del nar_model, nar_ckpt
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── 6. Assemble codes + decode all variants ───────────────
    print('[6/6] Decoding all variants...', flush=True)

    src_cb0_cpu = src_codes[0, 0]
    ar_cb0_cpu  = ar_target_cb0[0].cpu()

    # Source re-encoded (round-trip through Mimi)
    src_reenc_wav = decode_audio(mimi, src_codes, device)

    # AR-only: target CB0 + source CB1-31
    ar_only_codes = src_codes.clone()
    ar_only_codes[0, 0] = ar_cb0_cpu
    ar_only_wav = decode_audio(mimi, ar_only_codes, device)

    # NAR-only: source CB0 + generated CB1-31
    nar_only_codes = torch.zeros(1, 32, T_src, dtype=torch.long)
    nar_only_codes[0, 0]  = src_cb0_cpu
    nar_only_codes[0, 1:] = pred_nar_only
    nar_only_wav = decode_audio(mimi, nar_only_codes, device)

    # Full pipeline: AR CB0 + NAR CB1-31
    full_codes = torch.zeros(1, 32, T_src, dtype=torch.long)
    full_codes[0, 0]  = ar_cb0_cpu
    full_codes[0, 1:] = pred_full
    full_wav = decode_audio(mimi, full_codes, device)

    # FAISS baseline
    has_faiss = False
    try:
        print('  Computing FAISS baseline...', flush=True)
        cb0_emb = extract_cb0_embeddings()
        faiss_codes = faiss_baseline(src_codes, tgt_codes, cb0_emb, FRAME_RATE)
        faiss_wav = decode_audio(mimi, faiss_codes, device)
        has_faiss = True
    except Exception as e:
        print(f'  FAISS baseline skipped: {e}', flush=True)

    # ── Save all wavs ─────────────────────────────────────────
    outputs = {
        'source_reencoded':  src_reenc_wav,
        'ar_only_cb0':       ar_only_wav,
        'nar_only_cb1_31':   nar_only_wav,
        'full_ar_nar':       full_wav,
    }
    if has_faiss:
        outputs['faiss_baseline'] = faiss_wav

    for name, wav in outputs.items():
        path = os.path.join(args.output_dir, f'{name}.wav')
        sf.write(path, wav, SR)
        print(f'  Saved: {path}')

    # ── Metrics ───────────────────────────────────────────────
    tgt_ref = tgt_concat[:int(SR * 15)]

    print(f'\n{"="*70}')
    print(f'  v6.0 Inference Results')
    print(f'{"="*70}')
    print(f'{"Method":<35} {"SECS(->tgt)":>12} {"MCD(vs src)":>12}')
    print('-' * 61)

    for name, wav in outputs.items():
        s = compute_secs(wav, tgt_ref, SR, spk_enc)
        m = compute_mcd(src_wav, wav, SR)
        print(f'{name:<35} {s:>12.4f} {m:>12.2f}')

    # Source -> target baseline
    src_secs = compute_secs(src_wav, tgt_ref, SR, spk_enc)
    print(f'{"source (no conversion)":<35} {src_secs:>12.4f} {"--":>12}')

    print(f'\nAll outputs saved to: {args.output_dir}/')
    print('Done.')


if __name__ == '__main__':
    main()
