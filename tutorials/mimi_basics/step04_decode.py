"""
╔══════════════════════════════════════════════════════════════════════╗
║            MIMI TUTORIAL — Step 4: Decode Tokens → Audio            ║
║                                                                      ║
║  Goal: Take the encoded tokens from Step 3, decode them back to     ║
║        audio, and compare with the original.                         ║
║                                                                      ║
║  Run:  python tutorials/mimi_basics/step04_decode.py                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor

# ══════════════════════════════════════════════════════════════════════
# 4.1 — Load model and saved tokens
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 4.1: Loading model and saved tokens...")
print("=" * 70)

model = MimiModel.from_pretrained("kyutai/mimi")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

saved = torch.load("tutorials/mimi_basics/encoded_tokens.pt", weights_only=True)
codes = saved["codes"].to(device)  # [1, 8, 38]
original_audio = saved["original_audio"]  # [72000]
sr = saved["sample_rate"]

print(f"  Loaded tokens: {codes.shape} (batch={codes.shape[0]}, codebooks={codes.shape[1]}, frames={codes.shape[2]})")
print(f"  Original audio: {original_audio.shape[0]} samples, {sr} Hz")

# ══════════════════════════════════════════════════════════════════════
# 4.2 — Decode: Tokens → Audio
# ══════════════════════════════════════════════════════════════════════
#
# The decoder reverses the encoding:
#   tokens → codebook lookup → add all codebook vectors → upsample → audio

print("\n" + "=" * 70)
print("STEP 4.2: Decoding tokens → audio")
print("=" * 70)

with torch.no_grad():
    decoder_output = model.decode(codes)
    reconstructed = decoder_output.audio_values  # [1, 1, num_samples]

reconstructed_np = reconstructed[0, 0].cpu().numpy()
original_np = original_audio.numpy()

print(f"""
  Decoded audio shape: {reconstructed.shape}
    → [batch={reconstructed.shape[0]}, channels={reconstructed.shape[1]}, samples={reconstructed.shape[2]}]
    → Duration: {reconstructed.shape[2]/sr:.3f}s
    → Original duration: {len(original_np)/sr:.3f}s
""")

# ══════════════════════════════════════════════════════════════════════
# 4.3 — Compare Original vs Reconstructed
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 4.3: Quality comparison — original vs reconstructed")
print("=" * 70)

# Align lengths for comparison
min_len = min(len(original_np), len(reconstructed_np))
orig_aligned = original_np[:min_len]
recon_aligned = reconstructed_np[:min_len]

# Mean Squared Error
mse = np.mean((orig_aligned - recon_aligned) ** 2)

# Signal-to-Noise Ratio
signal_power = np.mean(orig_aligned ** 2)
noise_power = np.mean((orig_aligned - recon_aligned) ** 2)
snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

# Correlation
correlation = np.corrcoef(orig_aligned, recon_aligned)[0, 1]

# Peak values
orig_peak = np.max(np.abs(orig_aligned))
recon_peak = np.max(np.abs(recon_aligned))

print(f"""
  Length comparison:
    Original:      {len(original_np):,} samples ({len(original_np)/sr:.3f}s)
    Reconstructed: {len(reconstructed_np):,} samples ({len(reconstructed_np)/sr:.3f}s)
    Difference:    {abs(len(original_np) - len(reconstructed_np))} samples ({abs(len(original_np) - len(reconstructed_np))/sr*1000:.1f}ms)

  Quality metrics:
    MSE:           {mse:.6f}
    SNR:           {snr_db:.1f} dB
    Correlation:   {correlation:.4f}  (1.0 = perfect match)
    Peak original: {orig_peak:.4f}
    Peak recon:    {recon_peak:.4f}

  Interpretation:
    - SNR > 20 dB: good reconstruction
    - SNR > 30 dB: excellent reconstruction
    - Correlation > 0.9: waveforms are very similar
    - Note: Our test audio is synthetic (simple sine waves), so
      reconstruction may differ from real speech results.
""")

# ══════════════════════════════════════════════════════════════════════
# 4.4 — Save Reconstructed Audio
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 4.4: Saving audio files for listening comparison")
print("=" * 70)

# Normalize to prevent clipping
recon_normalized = reconstructed_np / (np.max(np.abs(reconstructed_np)) + 1e-8) * 0.9

sf.write("tutorials/mimi_basics/reconstructed_8cb.wav",
         recon_normalized.astype(np.float32), sr)
print(f"  Saved: tutorials/mimi_basics/reconstructed_8cb.wav")
print(f"  Compare with: tutorials/mimi_basics/test_audio.wav")

# ══════════════════════════════════════════════════════════════════════
# 4.5 — Progressive Decode: See What Each Codebook Adds
# ══════════════════════════════════════════════════════════════════════
#
# This is the KEY experiment: decode with 1 codebook, 2 codebooks, ..., 8.
# You can HEAR what each codebook adds to the audio quality.

print("\n" + "=" * 70)
print("STEP 4.5: Progressive decode — what does each codebook add?")
print("=" * 70)

print("""
We decode with increasing numbers of codebooks (1, 2, 3, ..., 8).
Each additional codebook adds finer acoustic detail.
""")

progressive_results = []

for n_cb in range(1, 9):
    with torch.no_grad():
        # Encode with all 8, but decode with only n_cb
        # We need to re-encode to get the right number of codes
        partial_codes = codes[:, :n_cb, :]  # Take first n_cb codebooks
        decoded = model.decode(partial_codes)
        partial_audio = decoded.audio_values[0, 0].cpu().numpy()

    # Calculate SNR against original
    min_l = min(len(original_np), len(partial_audio))
    o = original_np[:min_l]
    r = partial_audio[:min_l]
    sig_pow = np.mean(o ** 2)
    noise_pow = np.mean((o - r) ** 2)
    snr = 10 * np.log10(sig_pow / (noise_pow + 1e-10))
    corr = np.corrcoef(o, r)[0, 1] if len(o) > 1 else 0.0

    cb_label = "semantic" if n_cb == 1 else f"semantic + {n_cb-1} acoustic"
    print(f"  {n_cb} codebook(s) ({cb_label:30s}): SNR={snr:6.1f} dB, Correlation={corr:.4f}")

    # Save each version
    r_norm = partial_audio / (np.max(np.abs(partial_audio)) + 1e-8) * 0.9
    sf.write(f"tutorials/mimi_basics/reconstructed_{n_cb}cb.wav",
             r_norm.astype(np.float32), sr)
    progressive_results.append({"n_cb": n_cb, "snr": snr, "corr": corr})

print(f"""
  Saved 8 WAV files: reconstructed_1cb.wav through reconstructed_8cb.wav
  Listen to them in order to HEAR what each codebook adds:

    1cb: Only semantic tokens → robotic, captures speech rhythm/content
    2cb: + first acoustic    → pitch and basic voice quality appear
    3cb: + second acoustic   → clearer voice, better formants
    ...
    8cb: All codebooks       → near-original quality

  This demonstrates the HIERARCHICAL nature of RVQ:
    Codebook 1 = WHAT is said (semantic content)
    Codebooks 2-8 = HOW it sounds (progressive acoustic refinement)
""")

# ══════════════════════════════════════════════════════════════════════
# 4.6 — Token Manipulation: What Happens If We Change Tokens?
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 4.6: Token manipulation experiments")
print("=" * 70)

print("\nExperiment A: Replace all SEMANTIC tokens with zeros...")
codes_no_semantic = codes.clone()
codes_no_semantic[:, 0, :] = 0  # Zero out semantic codebook
with torch.no_grad():
    decoded_no_sem = model.decode(codes_no_semantic)
    audio_no_sem = decoded_no_sem.audio_values[0, 0].cpu().numpy()
audio_no_sem = audio_no_sem / (np.max(np.abs(audio_no_sem)) + 1e-8) * 0.9
sf.write("tutorials/mimi_basics/experiment_no_semantic.wav",
         audio_no_sem.astype(np.float32), sr)
print(f"  Saved: experiment_no_semantic.wav")
print(f"  → Audio has acoustic texture but may lose semantic structure")

print("\nExperiment B: Keep ONLY semantic tokens (zero acoustic)...")
codes_only_semantic = codes.clone()
codes_only_semantic[:, 1:, :] = 0  # Zero out all acoustic codebooks
with torch.no_grad():
    decoded_sem_only = model.decode(codes_only_semantic)
    audio_sem_only = decoded_sem_only.audio_values[0, 0].cpu().numpy()
audio_sem_only = audio_sem_only / (np.max(np.abs(audio_sem_only)) + 1e-8) * 0.9
sf.write("tutorials/mimi_basics/experiment_semantic_only.wav",
         audio_sem_only.astype(np.float32), sr)
print(f"  Saved: experiment_semantic_only.wav")
print(f"  → Audio has speech structure but sounds flat/robotic")

print("\nExperiment C: Shuffle semantic tokens (scramble word order)...")
codes_shuffled = codes.clone()
perm = torch.randperm(codes_shuffled.shape[2])
codes_shuffled[:, 0, :] = codes_shuffled[:, 0, perm]  # Shuffle temporal order
with torch.no_grad():
    decoded_shuffled = model.decode(codes_shuffled)
    audio_shuffled = decoded_shuffled.audio_values[0, 0].cpu().numpy()
audio_shuffled = audio_shuffled / (np.max(np.abs(audio_shuffled)) + 1e-8) * 0.9
sf.write("tutorials/mimi_basics/experiment_shuffled_semantic.wav",
         audio_shuffled.astype(np.float32), sr)
print(f"  Saved: experiment_shuffled_semantic.wav")
print(f"  → Semantic content is scrambled; speech becomes unintelligible")

print("\n" + "=" * 70)
print("DONE! Next: python tutorials/mimi_basics/step05_embeddings.py")
print("=" * 70)
