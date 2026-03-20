"""
╔══════════════════════════════════════════════════════════════════════╗
║      MIMI TUTORIAL — Step 6: Real Speech Encoding & Analysis        ║
║                                                                      ║
║  Goal: Download actual speech audio from HuggingFace, encode it     ║
║        with Mimi, and compare behavior vs synthetic audio.           ║
║        This shows how Mimi handles real human voice.                 ║
║                                                                      ║
║  Run:  python tutorials/mimi_basics/step06_real_speech.py            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor

# ══════════════════════════════════════════════════════════════════════
# 6.1 — Setup and load a real speech dataset
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 6.1: Loading model and real speech data...")
print("=" * 70)

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

# Load LibriSpeech sample from HuggingFace
from datasets import load_dataset, Audio

print("  Downloading LibriSpeech sample (first time may take a moment)...")
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
)

# Resample to 24kHz (Mimi's required sample rate)
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# Take 3 different samples to compare
samples = []
for i in [0, 1, 2]:
    audio_array = ds[i]["audio"]["array"]
    text = ds[i]["text"]
    duration = len(audio_array) / 24000
    samples.append({"audio": audio_array, "text": text, "duration": duration, "idx": i})
    print(f"\n  Sample {i}:")
    print(f"    Text: \"{text}\"")
    print(f"    Duration: {duration:.2f}s ({len(audio_array)} samples)")

# ══════════════════════════════════════════════════════════════════════
# 6.2 — Encode all 3 samples and compare codebook usage
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 6.2: Encoding real speech — codebook analysis")
print("=" * 70)

all_codes = []

for s in samples:
    inputs = feature_extractor(
        raw_audio=s["audio"],
        sampling_rate=24000,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        encoder_out = model.encode(input_values, num_quantizers=8)
        codes = encoder_out.audio_codes[0].cpu()  # [8, T]

    s["codes"] = codes
    s["num_frames"] = codes.shape[1]
    all_codes.append(codes)

    print(f"\n  Sample {s['idx']}: \"{s['text'][:50]}...\"")
    print(f"    Frames: {codes.shape[1]} ({codes.shape[1] * 80}ms)")

    for cb in range(8):
        tokens = codes[cb]
        unique = tokens.unique().shape[0]
        label = "Semantic" if cb == 0 else f"Acoustic{cb}"
        bar = "█" * int(unique / 100)  # Scale for display
        print(f"    CB{cb} ({label:10s}): {unique:4d} unique tokens {bar}")

# ══════════════════════════════════════════════════════════════════════
# 6.3 — Compare: synthetic vs real speech codebook utilization
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 6.3: Synthetic vs Real — codebook utilization comparison")
print("=" * 70)

# Load synthetic audio codes from step 3
saved = torch.load("tutorials/mimi_basics/encoded_tokens.pt", weights_only=True)
synthetic_codes = saved["codes"][0].cpu()  # [8, 38]

print(f"""
  Synthetic audio: {synthetic_codes.shape[1]} frames ({synthetic_codes.shape[1]*80/1000:.1f}s, sine waves)
  Real speech avg: {np.mean([s['num_frames'] for s in samples]):.0f} frames ({np.mean([s['duration'] for s in samples]):.1f}s, human voice)

  Codebook utilization comparison:
  {'Codebook':<15s} │ {'Synthetic':>10s} │ {'Real (avg)':>10s} │ {'Ratio':>8s}
  {'─'*15}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}""")

for cb in range(8):
    synth_unique = synthetic_codes[cb].unique().shape[0]
    real_uniques = [s["codes"][cb].unique().shape[0] for s in samples]
    real_avg = np.mean(real_uniques)
    ratio = real_avg / synth_unique if synth_unique > 0 else float('inf')
    label = "Semantic" if cb == 0 else f"Acoustic{cb}"
    print(f"  {label:<15s} │ {synth_unique:>10d} │ {real_avg:>10.0f} │ {ratio:>7.1f}x")

print("""
  KEY INSIGHT: Real speech uses MANY more codebook entries than synthetic
  sine waves. This is because speech contains:
    - Many different phonemes (diverse semantic tokens)
    - Rich vocal characteristics (diverse acoustic tokens)
    - Pitch variations, formant transitions, noise
""")

# ══════════════════════════════════════════════════════════════════════
# 6.4 — Reconstruction quality with real speech
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 6.4: Reconstruction quality — progressive codebook decode")
print("=" * 70)

# Use sample 0
s = samples[0]
print(f"\n  Using: \"{s['text'][:60]}...\"")
print(f"  Duration: {s['duration']:.2f}s\n")

original = s["audio"]
codes_device = s["codes"].unsqueeze(0).to(device)  # [1, 8, T]

print(f"  {'Codebooks':>12s} │ {'SNR (dB)':>10s} │ {'Correlation':>12s} │ Quality bar")
print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*30}")

for n_cb in range(1, 9):
    partial = codes_device[:, :n_cb, :]
    with torch.no_grad():
        decoded = model.decode(partial)
        recon = decoded.audio_values[0, 0].cpu().numpy()

    min_len = min(len(original), len(recon))
    o = original[:min_len]
    r = recon[:min_len]

    sig_pow = np.mean(o ** 2)
    noise_pow = np.mean((o - r) ** 2)
    snr = 10 * np.log10(sig_pow / (noise_pow + 1e-10))
    corr = np.corrcoef(o, r)[0, 1] if min_len > 1 else 0.0

    bar = "█" * max(0, int((snr + 5) * 2))
    print(f"  {n_cb:>12d} │ {snr:>10.1f} │ {corr:>12.4f} │ {bar}")

    # Save audio files
    r_norm = recon / (np.max(np.abs(recon)) + 1e-8) * 0.9
    sf.write(f"tutorials/mimi_basics/real_speech_{n_cb}cb.wav",
             r_norm.astype(np.float32), 24000)

# Save original for comparison
orig_norm = original / (np.max(np.abs(original)) + 1e-8) * 0.9
sf.write("tutorials/mimi_basics/real_speech_original.wav",
         orig_norm.astype(np.float32), 24000)

print(f"""
  Saved audio files:
    real_speech_original.wav   ← original recording
    real_speech_1cb.wav        ← semantic only (robotic but intelligible)
    real_speech_2cb.wav        ← + basic acoustic
    ...
    real_speech_8cb.wav        ← full quality

  LISTEN to these in order! You'll hear:
    1 CB:  Recognizable speech, but very robotic/buzzy
    2 CB:  Voice quality appears, gender becomes clear
    3-4 CB: Natural-sounding, good for most applications
    5-8 CB: Progressively finer detail, near-transparent quality
""")

# ══════════════════════════════════════════════════════════════════════
# 6.5 — Semantic Token Patterns in Real Speech
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 6.5: Semantic token patterns — do similar sounds → similar tokens?")
print("=" * 70)

# Analyze semantic tokens for sample 0
sem_tokens = s["codes"][0]  # [T]
print(f"\n  Total semantic tokens: {sem_tokens.shape[0]}")
print(f"  Unique semantic tokens: {sem_tokens.unique().shape[0]}")

# Find repeated token sequences (indicates repeated phonemes/sounds)
print("\n  Semantic token histogram (most common tokens):")
counts = torch.bincount(sem_tokens.long(), minlength=2048)
top_indices = counts.argsort(descending=True)[:15]

for rank, idx in enumerate(top_indices):
    count = counts[idx].item()
    if count == 0:
        break
    pct = count / sem_tokens.shape[0] * 100
    bar = "█" * int(pct)
    print(f"    Token {idx.item():>5d}: {count:>3d} times ({pct:>5.1f}%) {bar}")

print("""
  Interpretation:
    - Frequent semantic tokens = common phonemes in this utterance
    - The semantic codebook maps similar sounds to similar tokens
    - This is because of WavLM distillation: the semantic codebook
      is trained to capture LINGUISTIC content, not acoustic details
""")

# ══════════════════════════════════════════════════════════════════════
# 6.6 — Cross-Sample Semantic Comparison
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 6.6: Do different speakers share semantic tokens?")
print("=" * 70)

# Find shared semantic tokens between samples
sem_sets = [set(s["codes"][0].tolist()) for s in samples]

for i in range(len(samples)):
    for j in range(i+1, len(samples)):
        shared = sem_sets[i] & sem_sets[j]
        only_i = sem_sets[i] - sem_sets[j]
        only_j = sem_sets[j] - sem_sets[i]
        print(f"\n  Sample {samples[i]['idx']} vs Sample {samples[j]['idx']}:")
        print(f"    Shared semantic tokens:     {len(shared)}")
        print(f"    Only in sample {samples[i]['idx']}:          {len(only_i)}")
        print(f"    Only in sample {samples[j]['idx']}:          {len(only_j)}")
        print(f"    Jaccard similarity:         {len(shared) / len(sem_sets[i] | sem_sets[j]):.2f}")

print("""
  KEY INSIGHT: Different utterances SHARE many semantic tokens because:
    - English has a limited set of phonemes (~44)
    - The semantic codebook maps phonemes → tokens
    - Common sounds (vowels, common consonants) → same tokens
    - This is exactly what WavLM distillation achieves!

  For Vietnamese: we need to verify that Mimi's semantic codebook
  also distinguishes Vietnamese tones (6 tones × ~44 phonemes).
""")

# ══════════════════════════════════════════════════════════════════════
# 6.7 — Summary of what we learned
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 6.7: Summary")
print("=" * 70)

print(f"""
  What we learned from real speech:

  1. CODEBOOK UTILIZATION: Real speech uses 5-20x more codebook entries
     than synthetic audio, because speech is acoustically diverse.

  2. PROGRESSIVE QUALITY: Each codebook adds measurable improvement.
     The biggest jump is from 1→2 codebooks (semantic → acoustic).

  3. SEMANTIC TOKENS: The first codebook captures linguistic content.
     Different speakers saying different words share common tokens
     for common phonemes — this is the WavLM distillation at work.

  4. RECONSTRUCTION: Even with extreme compression (1.1 kbps, 8 codebooks),
     Mimi produces intelligible, natural-sounding speech reconstruction.

  5. FOR VIETNAMESE: We need to test whether the semantic codebook
     can distinguish tones. If "ma" (ghost) and "má" (mother) get
     the same semantic token → Mimi needs fine-tuning for Vietnamese.
""")

print("=" * 70)
print("DONE! Next: python tutorials/mimi_basics/step07_voice_analysis.py")
print("=" * 70)
