"""
╔══════════════════════════════════════════════════════════════════════╗
║            MIMI TUTORIAL — Step 3: Encode Audio to Tokens           ║
║                                                                      ║
║  Goal: Take real audio, encode it to discrete tokens, and           ║
║        understand what each codebook captures.                       ║
║                                                                      ║
║  Run:  python tutorials/mimi_basics/step03_encode.py                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor

# ══════════════════════════════════════════════════════════════════════
# 3.1 — Setup
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 3.1: Loading model and audio...")
print("=" * 70)

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

# Load our test audio
audio_data, sr = sf.read("tutorials/mimi_basics/test_audio.wav")
print(f"  Audio loaded: {len(audio_data)} samples, {sr} Hz, {len(audio_data)/sr:.2f}s")

# ══════════════════════════════════════════════════════════════════════
# 3.2 — Preprocess Audio
# ══════════════════════════════════════════════════════════════════════
#
# The feature_extractor converts raw audio arrays into model-ready tensors.

print("\n" + "=" * 70)
print("STEP 3.2: Preprocessing audio with feature extractor...")
print("=" * 70)

inputs = feature_extractor(
    raw_audio=audio_data,
    sampling_rate=sr,
    return_tensors="pt",
)

# Move to device
input_values = inputs["input_values"].to(device)
padding_mask = inputs.get("padding_mask")
if padding_mask is not None:
    padding_mask = padding_mask.to(device)

print(f"""
  Feature extractor output:
    input_values shape: {input_values.shape}
      → [batch={input_values.shape[0]}, channels={input_values.shape[1]}, samples={input_values.shape[2]}]
    dtype: {input_values.dtype}
    value range: [{input_values.min():.4f}, {input_values.max():.4f}]
    padding_mask: {padding_mask.shape if padding_mask is not None else 'None'}
""")

# ══════════════════════════════════════════════════════════════════════
# 3.3 — Encode: Audio → Tokens
# ══════════════════════════════════════════════════════════════════════
#
# This is the core operation: compress audio into discrete tokens.
#
# IMPORTANT: Mimi has 32 codebooks available, but PersonaPlex uses only 8.
# We use num_quantizers=8 to match the PersonaPlex configuration:
#   - Codebook 0: Semantic (WavLM-distilled)
#   - Codebooks 1-7: Acoustic (RVQ)

print("=" * 70)
print("STEP 3.3: Encoding audio → tokens (num_quantizers=8)")
print("=" * 70)

with torch.no_grad():
    # Encode with 8 codebooks (matching PersonaPlex)
    encoder_output = model.encode(
        input_values,
        padding_mask=padding_mask,
        num_quantizers=8,
    )
    codes = encoder_output.audio_codes

print(f"""
  Encoding result:
    codes shape: {codes.shape}
      → [batch={codes.shape[0]}, codebooks={codes.shape[1]}, frames={codes.shape[2]}]

  Interpretation:
    - {codes.shape[1]} codebooks (1 semantic + {codes.shape[1]-1} acoustic)
    - {codes.shape[2]} frames at 12.5 Hz
    - Each frame = 80ms of audio
    - Total represented audio: {codes.shape[2]} × 80ms = {codes.shape[2] * 80}ms = {codes.shape[2] * 80 / 1000:.2f}s
    - Original audio: {len(audio_data)/sr:.2f}s

  Token value range: [{codes.min().item()}, {codes.max().item()}] (expected: [0, 2047])
""")

# ══════════════════════════════════════════════════════════════════════
# 3.4 — Inspect Individual Codebooks
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 3.4: Inspecting individual codebook tokens")
print("=" * 70)

codes_cpu = codes[0].cpu()  # [codebooks, frames] — remove batch dim

for cb_idx in range(codes_cpu.shape[0]):
    tokens = codes_cpu[cb_idx]
    unique_count = tokens.unique().shape[0]
    label = "SEMANTIC (WavLM-distilled)" if cb_idx == 0 else f"ACOUSTIC (RVQ level {cb_idx})"

    print(f"\n  Codebook {cb_idx} — {label}:")
    print(f"    Tokens (first 20): {tokens[:20].tolist()}")
    print(f"    Unique values used: {unique_count} / 2048 ({unique_count/2048*100:.1f}% codebook utilization)")
    print(f"    Min={tokens.min().item()}, Max={tokens.max().item()}, "
          f"Mean={tokens.float().mean().item():.1f}")

# ══════════════════════════════════════════════════════════════════════
# 3.5 — Token Sequence Visualization
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 3.5: Token sequence visualization")
print("=" * 70)

print(f"\nFirst 15 frames (15 × 80ms = 1.2 seconds) across all {codes_cpu.shape[0]} codebooks:\n")

# Header
header = f"{'Frame':>6} │ {'Time':>7} │"
for cb in range(min(8, codes_cpu.shape[0])):
    label = "Sem" if cb == 0 else f"Ac{cb}"
    header += f" {label:>5} │"
print(header)
print("─" * len(header))

# Data rows
for frame in range(min(15, codes_cpu.shape[1])):
    time_ms = frame * 80
    row = f"{frame:>6} │ {time_ms:>5}ms │"
    for cb in range(min(8, codes_cpu.shape[0])):
        row += f" {codes_cpu[cb, frame].item():>5} │"
    print(row)

print(f"\n  ... ({codes_cpu.shape[1] - 15} more frames)")

# ══════════════════════════════════════════════════════════════════════
# 3.6 — Compression Statistics
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 3.6: Compression statistics")
print("=" * 70)

num_audio_samples = input_values.shape[2]
num_frames = codes.shape[2]
num_codebooks = codes.shape[1]
bits_per_token = 11  # log2(2048) = 11 bits

original_bits = num_audio_samples * 16  # 16-bit PCM
compressed_bits = num_frames * num_codebooks * bits_per_token

print(f"""
  Original audio:
    Samples:    {num_audio_samples:,}
    Bit depth:  16 bits
    Total bits: {original_bits:,} ({original_bits/8/1024:.1f} KB)
    Bitrate:    {sr * 16 / 1000:.0f} kbps

  Mimi tokens ({num_codebooks} codebooks):
    Frames:     {num_frames}
    Tokens:     {num_frames} × {num_codebooks} = {num_frames * num_codebooks} integers
    Bits:       {compressed_bits:,} ({compressed_bits/8/1024:.1f} KB)
    Bitrate:    {num_codebooks * bits_per_token * 12.5 / 1000:.2f} kbps

  Compression ratio: {original_bits / compressed_bits:.1f}x
  Size reduction:    {(1 - compressed_bits / original_bits) * 100:.1f}%

  For comparison:
    MP3 (128kbps):  ~{128 * len(audio_data) / sr / 8 / 1024:.1f} KB
    Mimi (8 CB):    ~{compressed_bits / 8 / 1024:.1f} KB
""")

# ══════════════════════════════════════════════════════════════════════
# 3.7 — Save tokens for next steps
# ══════════════════════════════════════════════════════════════════════

save_path = "tutorials/mimi_basics/encoded_tokens.pt"
torch.save({
    "codes": codes.cpu(),
    "num_quantizers": 8,
    "audio_samples": num_audio_samples,
    "sample_rate": sr,
    "original_audio": torch.from_numpy(audio_data).float(),
}, save_path)
print(f"Saved encoded tokens to: {save_path}")

print("\n" + "=" * 70)
print("DONE! Next: python tutorials/mimi_basics/step04_decode.py")
print("=" * 70)
