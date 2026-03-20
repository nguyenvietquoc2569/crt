"""
╔══════════════════════════════════════════════════════════════════════╗
║      MIMI TUTORIAL — Step 5: Inside the Encoder — Embeddings       ║
║                                                                      ║
║  Goal: Crack open the encoder and see the CONTINUOUS embeddings     ║
║        BEFORE they become discrete tokens. This is where the        ║
║        "information" lives.                                          ║
║                                                                      ║
║  Run:  python tutorials/mimi_basics/step05_embeddings.py             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor

# ══════════════════════════════════════════════════════════════════════
# 5.1 — Setup
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 5.1: Loading model...")
print("=" * 70)

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

audio_data, sr = sf.read("tutorials/mimi_basics/test_audio.wav")
inputs = feature_extractor(raw_audio=audio_data, sampling_rate=sr, return_tensors="pt")
input_values = inputs["input_values"].to(device)

print(f"  Audio: {len(audio_data)} samples, {sr} Hz, {len(audio_data)/sr:.2f}s")
print(f"  Input tensor: {input_values.shape}")

# ══════════════════════════════════════════════════════════════════════
# 5.2 — Walk Through the Encoder Step by Step
# ══════════════════════════════════════════════════════════════════════
#
# We manually call each sub-module of the encoder to see the shape
# at every stage. This is how Mimi compresses 24kHz audio to 12.5Hz tokens.

print("\n" + "=" * 70)
print("STEP 5.2: Walking through the encoder layer by layer")
print("=" * 70)

with torch.no_grad():
    x = input_values  # [1, 1, 72000]
    print(f"\n  Input audio:               {x.shape}")
    print(f"  → {x.shape[2]} samples at {sr} Hz = {x.shape[2]/sr:.2f}s\n")

    # Stage 1: Convolutional encoder
    print("  --- CONVOLUTIONAL ENCODER ---")
    conv_out = model.encoder(x)
    print(f"  After conv encoder:        {conv_out.shape}")
    print(f"  → {conv_out.shape[2]} frames at {conv_out.shape[2] / (x.shape[2]/sr):.1f} Hz")
    print(f"  → Hidden dim = {conv_out.shape[1]}")
    print(f"  → Compression: {x.shape[2]} → {conv_out.shape[2]} ({x.shape[2]//conv_out.shape[2]}x downsampling)")

    # Stage 2: Encoder transformer (adds temporal context)
    print("\n  --- ENCODER TRANSFORMER (8 layers) ---")
    # Transformer expects [batch, seq_len, hidden] but conv gives [batch, hidden, seq_len]
    transformer_input = conv_out.transpose(1, 2)  # [B, T, H]
    print(f"  Transformer input:         {transformer_input.shape} (transposed to [B, T, H])")

    transformer_out = model.encoder_transformer(transformer_input)
    # transformer_out is a tuple, first element is hidden states
    contextualized = transformer_out[0]
    print(f"  Transformer output:        {contextualized.shape}")
    print(f"  → Same shape! Transformer only adds CONTEXT, doesn't change dimensions")

    # Transpose back to [B, H, T] for the downsampler
    contextualized = contextualized.transpose(1, 2)  # [B, H, T]

    # Stage 3: Learned downsampler
    print("\n  --- LEARNED DOWNSAMPLER ---")
    downsampled = model.downsample(contextualized)
    print(f"  After downsampler:         {downsampled.shape}")
    print(f"  → {downsampled.shape[2]} frames at 12.5 Hz")
    print(f"  → Downsampling factor: {conv_out.shape[2]} → {downsampled.shape[2]} ({conv_out.shape[2]//downsampled.shape[2]}x)")

    # Stage 4: Quantization
    print("\n  --- SPLIT VECTOR QUANTIZER ---")
    codes = model.quantizer.encode(downsampled, num_quantizers=8)
    codes = codes.transpose(0, 1)  # [B, num_cb, T]
    print(f"  Quantized codes:           {codes.shape}")
    print(f"  → 8 integer tokens per frame, each in [0, 2047]")

print(f"""
  COMPLETE PIPELINE SUMMARY:
  ─────────────────────────────────────────────────────────────
  Input audio:     {x.shape}           ({x.shape[2]} samples, 24kHz)
       ↓ Conv encoder (stride ~{x.shape[2]//conv_out.shape[2]})
  Conv output:     {conv_out.shape}         ({conv_out.shape[2]} frames, ~{conv_out.shape[2] / (x.shape[2]/sr):.0f} Hz)
       ↓ 8-layer Transformer (causal, RoPE)
  Contextualized:  {contextualized.shape}         (same shape, added context)
       ↓ Learned downsampler (stride {conv_out.shape[2]//downsampled.shape[2]})
  Embeddings:      {downsampled.shape}          ({downsampled.shape[2]} frames, 12.5 Hz)
       ↓ Split RVQ (1 semantic + 7 acoustic)
  Tokens:          {codes.shape}          (8 integers per frame)
  ─────────────────────────────────────────────────────────────
""")

# ══════════════════════════════════════════════════════════════════════
# 5.3 — Examine the Continuous Embeddings
# ══════════════════════════════════════════════════════════════════════
#
# The "downsampled" tensor is the CONTINUOUS representation that gets
# quantized. This is the richest representation of the audio.

print("=" * 70)
print("STEP 5.3: Examining continuous embeddings (pre-quantization)")
print("=" * 70)

emb = downsampled[0].cpu()  # [hidden_dim, num_frames]

print(f"""
  Continuous embeddings shape: {emb.shape}
    → {emb.shape[0]} dimensions per frame
    → {emb.shape[1]} frames (one per 80ms)

  Statistics per dimension (across all frames):
    Mean:   {emb.mean().item():.4f}
    Std:    {emb.std().item():.4f}
    Min:    {emb.min().item():.4f}
    Max:    {emb.max().item():.4f}

  These embeddings contain ALL information about the audio:
    - Semantic content (phonemes, words)
    - Acoustic details (voice timbre, pitch)
    - Temporal patterns (rhythm, prosody)

  Quantization LOSES some of this information by rounding
  each embedding to the nearest codebook vector.
""")

# Show frame-by-frame statistics
print("  Per-frame embedding norms (L2):")
norms = torch.norm(emb, dim=0)
for i in range(min(10, emb.shape[1])):
    bar = "█" * int(norms[i].item() / norms.max().item() * 30)
    print(f"    Frame {i:2d} ({i*80:>4d}ms): norm={norms[i].item():6.2f}  {bar}")
print(f"    ... ({emb.shape[1] - 10} more frames)")

# ══════════════════════════════════════════════════════════════════════
# 5.4 — Quantization Error Analysis
# ══════════════════════════════════════════════════════════════════════
#
# How much information is lost during quantization?

print("\n" + "=" * 70)
print("STEP 5.4: Quantization error — what is lost?")
print("=" * 70)

with torch.no_grad():
    # Get the quantized embeddings (what the decoder actually sees)
    quantized = model.quantizer.decode(codes)  # [B, H, T]
    quantized_cpu = quantized[0].cpu()

# Compare continuous vs quantized
diff = emb - quantized_cpu
error_per_frame = torch.norm(diff, dim=0)
signal_per_frame = torch.norm(emb, dim=0)

# Per-frame SNR
frame_snr = 20 * torch.log10(signal_per_frame / (error_per_frame + 1e-10))

print(f"""
  Continuous embedding:  {emb.shape}
  Quantized embedding:   {quantized_cpu.shape}
  Difference (error):    {diff.shape}

  Average error norm:    {error_per_frame.mean().item():.4f}
  Average signal norm:   {signal_per_frame.mean().item():.4f}
  Average SNR:           {frame_snr.mean().item():.1f} dB

  This tells us: {100 * (1 - error_per_frame.mean() / signal_per_frame.mean()).item():.1f}%
  of the embedding information is preserved after quantization.
""")

# ══════════════════════════════════════════════════════════════════════
# 5.5 — Exploring Codebook Vectors
# ══════════════════════════════════════════════════════════════════════
#
# Each codebook is a lookup table of 2048 vectors.
# Let's examine what these vectors look like.

print("=" * 70)
print("STEP 5.5: Codebook vector analysis")
print("=" * 70)

# Get codebook parameters
semantic_q = model.quantizer.semantic_residual_vector_quantizer
acoustic_q = model.quantizer.acoustic_residual_vector_quantizer

print("\nSemantic codebook (Codebook 0):")
for name, param in semantic_q.named_parameters():
    if "embed" in name.lower():
        print(f"  {name}: {param.shape}")
        cb_vectors = param.detach().cpu()
        print(f"    → {cb_vectors.shape[0]} vectors of dimension {cb_vectors.shape[1]}")
        print(f"    → Vector norm range: [{torch.norm(cb_vectors, dim=1).min():.3f}, {torch.norm(cb_vectors, dim=1).max():.3f}]")
        print(f"    → Mean vector norm: {torch.norm(cb_vectors, dim=1).mean():.3f}")

print("\nAcoustic codebooks (Codebooks 1-7):")
for name, param in acoustic_q.named_parameters():
    if "embed" in name.lower():
        cb_vectors = param.detach().cpu()
        # Extract layer index from name
        print(f"  {name}: {cb_vectors.shape}")

# ══════════════════════════════════════════════════════════════════════
# 5.6 — Codebook Usage Patterns
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 5.6: Codebook utilization analysis")
print("=" * 70)

print("""
Each codebook has 2048 entries, but not all may be used equally.
A well-trained codebook uses many entries; a poorly-trained one
may only use a few (called "codebook collapse").
""")

codes_cpu = codes[0].cpu()
for cb in range(codes_cpu.shape[0]):
    tokens = codes_cpu[cb]
    unique = tokens.unique()
    counts = torch.bincount(tokens.long(), minlength=2048)
    nonzero = (counts > 0).sum().item()
    most_common_idx = counts.argmax().item()
    most_common_count = counts[most_common_idx].item()

    label = "Semantic" if cb == 0 else f"Acoustic {cb}"
    bar = "█" * int(nonzero / 2048 * 40)
    print(f"  CB{cb} ({label:10s}): {nonzero:4d}/2048 used ({nonzero/2048*100:4.1f}%) {bar}")
    print(f"     Most common: token {most_common_idx} appears {most_common_count}/{tokens.shape[0]} times ({most_common_count/tokens.shape[0]*100:.0f}%)")

print("""
  NOTE: Low utilization here is because our test audio is synthetic
  (simple sine waves). Real speech would use many more codebook entries.
  This is actually an important metric for evaluating codec quality.
""")

# ══════════════════════════════════════════════════════════════════════
# 5.7 — Cosine Similarity Between Frames
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 5.7: Frame similarity in embedding space")
print("=" * 70)

# Calculate cosine similarity between all pairs of frames
emb_normalized = emb / (torch.norm(emb, dim=0, keepdim=True) + 1e-10)
similarity_matrix = emb_normalized.T @ emb_normalized  # [T, T]

print(f"""
  Cosine similarity matrix: {similarity_matrix.shape}
  (each entry [i,j] = cosine similarity between frame i and frame j)

  Diagonal (self-similarity): always 1.0
  Off-diagonal: similarity between different time steps

  First 10×10 corner (frames 0-9):
""")

sim = similarity_matrix[:10, :10]
# Header
print("        ", end="")
for j in range(10):
    print(f"  F{j:>2d}  ", end="")
print()

for i in range(10):
    print(f"  F{i:>2d}  ", end="")
    for j in range(10):
        val = sim[i, j].item()
        if i == j:
            print(f" [1.00]", end="")
        elif val > 0.8:
            print(f"  {val:.2f} ", end="")
        else:
            print(f"  {val:.2f} ", end="")
    print()

print(f"""
  Interpretation:
    - High similarity between frames → similar audio content at those times
    - Our synthetic audio is periodic → expect repeating similarity patterns
    - For real speech: frames within a phoneme would be similar,
      frames across different phonemes would be different
""")

# Save embeddings for step 6
torch.save({
    "continuous_embeddings": emb,
    "quantized_embeddings": quantized_cpu,
    "codes": codes_cpu,
    "similarity_matrix": similarity_matrix,
}, "tutorials/mimi_basics/embeddings_analysis.pt")

print("=" * 70)
print("DONE! Next: python tutorials/mimi_basics/step06_real_speech.py")
print("=" * 70)
