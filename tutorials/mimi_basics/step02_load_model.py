"""
╔══════════════════════════════════════════════════════════════════════╗
║            MIMI TUTORIAL — Step 2: Load Model & Inspect             ║
║                                                                      ║
║  Goal: Download Mimi from HuggingFace, load it, and explore         ║
║        the architecture layer by layer.                              ║
║                                                                      ║
║  Run:  python tutorials/mimi_basics/step02_load_model.py             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
from transformers import MimiModel, AutoFeatureExtractor

# ══════════════════════════════════════════════════════════════════════
# 2.1 — Download and Load the Model
# ══════════════════════════════════════════════════════════════════════
#
# The first time you run this, it downloads ~400MB from HuggingFace.
# After that, it loads from cache (~2 seconds).

print("=" * 70)
print("STEP 2.1: Loading Mimi model from HuggingFace...")
print("=" * 70)

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

# Move to MPS (Apple Silicon GPU) if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set to inference mode (no dropout, no gradient tracking)

print(f"\nDevice: {device}")
print(f"Model loaded successfully!")

# ══════════════════════════════════════════════════════════════════════
# 2.2 — Model Configuration
# ══════════════════════════════════════════════════════════════════════
#
# The config tells us everything about Mimi's architecture.

print("\n" + "=" * 70)
print("STEP 2.2: Model Configuration")
print("=" * 70)

config = model.config

print(f"""
Key parameters:
  audio_channels:          {config.audio_channels}          (mono audio)
  sampling_rate:           {config.sampling_rate}       (24,000 Hz input)
  frame_rate:              {config.frame_rate}         (12.5 Hz output token rate)
  hidden_size:             {config.hidden_size}          (internal embedding dimension)
  num_filters:             {config.num_filters}           (base conv filter count)
  num_residual_layers:     {config.num_residual_layers}           (per downsampling block)
  upsampling_ratios:       {config.upsampling_ratios}    (decoder upsample strides)
  kernel_size:             {config.kernel_size}           (conv kernel size)

  Quantizer:
    codebook_size:         {config.codebook_size}        (entries per codebook = 2048)
    num_quantizers:        {config.num_quantizers}           (total codebooks = 8)
    num_semantic_quantizers: {config.num_semantic_quantizers}  (semantic codebooks = 1)
    → acoustic codebooks:  {config.num_quantizers - config.num_semantic_quantizers}           (= 8 - 1 = 7)

  Transformer (encoder & decoder):
    num_hidden_layers:     {config.num_hidden_layers}           (transformer layers = 8)
    num_attention_heads:   {config.num_attention_heads}           (attention heads = 8)
    intermediate_size:     {config.intermediate_size}        (FFN hidden size)

  Compression ratio:       {config.sampling_rate} / {config.frame_rate} = {int(config.sampling_rate / config.frame_rate)} samples per frame
""")

# ══════════════════════════════════════════════════════════════════════
# 2.3 — Architecture Overview (Layer by Layer)
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 2.3: Architecture — Top-Level Modules")
print("=" * 70)

print("\nMimi consists of these top-level components:\n")
for name, module in model.named_children():
    param_count = sum(p.numel() for p in module.parameters())
    print(f"  {name:30s}  {param_count:>12,} parameters")

total_params = sum(p.numel() for p in model.parameters())
print(f"  {'TOTAL':30s}  {total_params:>12,} parameters")

# ══════════════════════════════════════════════════════════════════════
# 2.4 — Encoder Architecture
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2.4: Encoder Layers (audio → embeddings)")
print("=" * 70)

print("""
The encoder is a sequence of:
  1. Initial Conv1d (feature extraction, no downsampling)
  2. Downsampling blocks: [ResnetBlock × N] + ELU + strided Conv1d
  3. Each block doubles the number of filters

Downsampling strides (reversed upsampling_ratios): """, end="")
reversed_ratios = list(reversed(config.upsampling_ratios))
print(reversed_ratios)
print(f"  Total stride: {reversed_ratios[0]} × {reversed_ratios[1]} × {reversed_ratios[2]} × {reversed_ratios[3]} = ", end="")
total_stride = 1
for r in reversed_ratios:
    total_stride *= r
print(total_stride)
print(f"  With downsampler: {total_stride} × 6 = {total_stride * 6} (= samples per token frame)\n")

print("Encoder layers:")
for name, module in model.encoder.named_modules():
    if name == "":
        continue
    # Only show top-level layers (depth 1-2)
    depth = name.count(".")
    if depth <= 1:
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        module_type = type(module).__name__
        indent = "  " * (depth + 1)
        if param_count > 0:
            print(f"{indent}{name}: {module_type} ({param_count:,} params)")
        else:
            print(f"{indent}{name}: {module_type}")

# ══════════════════════════════════════════════════════════════════════
# 2.5 — Quantizer Architecture
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2.5: Quantizer (embeddings → tokens)")
print("=" * 70)

quantizer = model.quantizer
print(f"""
Quantizer type: {type(quantizer).__name__}

Structure:
  ├── semantic_residual_vector_quantizer ({config.num_semantic_quantizers} codebook)
  │     └── Each codebook: {config.codebook_size} entries of dimension {config.hidden_size}
  │     └── Trained with WavLM distillation (captures MEANING)
  │
  └── acoustic_residual_vector_quantizer ({config.num_quantizers - config.num_semantic_quantizers} codebooks)
        └── Each codebook: {config.codebook_size} entries of dimension {config.hidden_size}
        └── Standard RVQ (captures SOUND QUALITY)
""")

# Show codebook shapes
for name, param in quantizer.named_parameters():
    if "embed" in name.lower() or "codebook" in name.lower():
        print(f"  {name}: {param.shape}")

# ══════════════════════════════════════════════════════════════════════
# 2.6 — Feature Extractor (Preprocessor)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2.6: Feature Extractor (audio preprocessor)")
print("=" * 70)

print(f"""
The feature extractor handles audio preprocessing before the model:

  sampling_rate:    {feature_extractor.sampling_rate} Hz
  feature_size:     {feature_extractor.feature_size}
  padding_value:    {feature_extractor.padding_value}

  What it does:
    1. Ensures audio is at {feature_extractor.sampling_rate} Hz
    2. Converts numpy/list → PyTorch tensor
    3. Creates padding mask for batched inputs
    4. Normalizes audio amplitude
""")

# ══════════════════════════════════════════════════════════════════════
# 2.7 — Quick Sanity Check
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 2.7: Quick sanity check — can the model process audio?")
print("=" * 70)

# Create a tiny test signal (0.1 seconds = 2400 samples)
test_signal = torch.randn(1, 1, 2400).to(device)

with torch.no_grad():
    # Encode
    encoder_out = model.encode(test_signal)
    codes = encoder_out.audio_codes
    print(f"\n  Input audio shape:  {test_signal.shape}  (batch=1, channels=1, samples=2400)")
    print(f"  Output codes shape: {codes.shape}  (batch, codebooks, frames)")
    print(f"  Codebook values range: [{codes.min().item()}, {codes.max().item()}] (should be in [0, 2047])")

    # Decode
    decoded = model.decode(codes)
    audio_out = decoded.audio_values
    print(f"  Decoded audio shape: {audio_out.shape}")
    print(f"\n  ✓ Model works! Audio → {codes.shape[2]} token frames → Audio")

print("\n" + "=" * 70)
print("DONE! Next: python tutorials/mimi_basics/step03_encode.py")
print("=" * 70)
