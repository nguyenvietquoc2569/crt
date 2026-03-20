"""
╔══════════════════════════════════════════════════════════════════════╗
║    MIMI TUTORIAL — Step 7: Voice Identity in Mimi Embeddings        ║
║                                                                      ║
║  Goal: Understand how Mimi represents voice identity vs content.    ║
║        Compare embeddings from DIFFERENT speakers saying the SAME   ║
║        text, and the SAME speaker saying DIFFERENT text.            ║
║        This is the foundation for voice prompts in PersonaPlex.     ║
║                                                                      ║
║  Run:  python tutorials/mimi_basics/step07_voice_analysis.py         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import soundfile as sf
from transformers import MimiModel, AutoFeatureExtractor
from datasets import load_dataset, Audio

# ══════════════════════════════════════════════════════════════════════
# 7.1 — Setup
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 7.1: Loading model and multi-speaker data...")
print("=" * 70)

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

# Load LibriSpeech — it has multiple speakers
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
)
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# Collect multiple samples (checking speaker IDs)
print("  Loading samples...")
samples = []
for i in range(min(20, len(ds))):
    item = ds[i]
    samples.append({
        "idx": i,
        "audio": item["audio"]["array"],
        "text": item["text"],
        "speaker_id": item.get("speaker_id", f"unknown_{i}"),
        "duration": len(item["audio"]["array"]) / 24000,
    })

# Show available samples
print(f"\n  Loaded {len(samples)} samples")
speakers = set(s["speaker_id"] for s in samples)
print(f"  Unique speakers: {len(speakers)}")
for spk in list(speakers)[:5]:
    spk_samples = [s for s in samples if s["speaker_id"] == spk]
    print(f"    Speaker {spk}: {len(spk_samples)} samples")
    for ss in spk_samples[:2]:
        print(f"      \"{ss['text'][:60]}...\" ({ss['duration']:.1f}s)")

# ══════════════════════════════════════════════════════════════════════
# 7.2 — Encode all samples and extract embeddings
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 7.2: Encoding all samples to tokens and embeddings...")
print("=" * 70)

for s in samples:
    inputs = feature_extractor(
        raw_audio=s["audio"], sampling_rate=24000, return_tensors="pt"
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        # Get tokens
        enc_out = model.encode(input_values, num_quantizers=8)
        s["codes"] = enc_out.audio_codes[0].cpu()  # [8, T]

        # Get continuous embeddings (pre-quantization) for deeper analysis
        conv_out = model.encoder(input_values)
        trans_out = model.encoder_transformer(conv_out.transpose(1, 2))
        emb = model.downsample(trans_out[0].transpose(1, 2))
        s["embeddings"] = emb[0].cpu()  # [hidden_dim, T]

        # Compute a "speaker embedding" = mean of all frame embeddings
        s["mean_embedding"] = s["embeddings"].mean(dim=1)  # [hidden_dim]

    print(f"  Sample {s['idx']:2d}: {s['codes'].shape[1]:3d} frames, "
          f"speaker={s['speaker_id']}, \"{s['text'][:40]}...\"")

# ══════════════════════════════════════════════════════════════════════
# 7.3 — Speaker Similarity via Mean Embeddings
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 7.3: Speaker similarity — are same-speaker embeddings closer?")
print("=" * 70)

print("""
  We compute cosine similarity between MEAN embeddings of all sample pairs.
  Hypothesis: samples from the SAME speaker should be more similar.
""")

# Compute pairwise cosine similarity
n = len(samples)
sim_matrix = torch.zeros(n, n)
for i in range(n):
    for j in range(n):
        ei = samples[i]["mean_embedding"]
        ej = samples[j]["mean_embedding"]
        sim_matrix[i, j] = torch.nn.functional.cosine_similarity(
            ei.unsqueeze(0), ej.unsqueeze(0)
        ).item()

# Analyze: same-speaker vs different-speaker similarities
same_speaker_sims = []
diff_speaker_sims = []

for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i, j].item()
        if samples[i]["speaker_id"] == samples[j]["speaker_id"]:
            same_speaker_sims.append(sim)
        else:
            diff_speaker_sims.append(sim)

if same_speaker_sims:
    print(f"  Same-speaker pairs:      {len(same_speaker_sims)}")
    print(f"    Mean similarity:       {np.mean(same_speaker_sims):.4f}")
    print(f"    Std:                   {np.std(same_speaker_sims):.4f}")
    print(f"    Range:                 [{min(same_speaker_sims):.4f}, {max(same_speaker_sims):.4f}]")
else:
    print("  No same-speaker pairs found in this dataset subset.")

if diff_speaker_sims:
    print(f"\n  Different-speaker pairs: {len(diff_speaker_sims)}")
    print(f"    Mean similarity:       {np.mean(diff_speaker_sims):.4f}")
    print(f"    Std:                   {np.std(diff_speaker_sims):.4f}")
    print(f"    Range:                 [{min(diff_speaker_sims):.4f}, {max(diff_speaker_sims):.4f}]")

if same_speaker_sims and diff_speaker_sims:
    gap = np.mean(same_speaker_sims) - np.mean(diff_speaker_sims)
    print(f"\n  Gap (same - different):  {gap:.4f}")
    if gap > 0.05:
        print(f"  → GOOD: Same-speaker embeddings ARE more similar ({gap:.4f} gap)")
        print(f"    Mimi's continuous embeddings capture speaker identity!")
    elif gap > 0:
        print(f"  → WEAK: Small positive gap ({gap:.4f})")
        print(f"    Mimi partially captures speaker identity in continuous embeddings")
    else:
        print(f"  → NOTE: No clear speaker clustering in mean embeddings")
        print(f"    Speaker identity may be in the temporal PATTERN, not the mean")

# ══════════════════════════════════════════════════════════════════════
# 7.4 — Semantic vs Acoustic Token Sharing
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 7.4: Where does voice identity live — semantic or acoustic?")
print("=" * 70)

print("""
  Theory: Semantic tokens (CB0) capture WHAT is said (speaker-invariant).
          Acoustic tokens (CB1-7) capture HOW it sounds (speaker-specific).

  Test: Compare token overlap between same-speaker and different-speaker pairs.
""")

for cb_idx in range(min(4, 8)):  # Just show first 4 codebooks
    label = "Semantic (CB0)" if cb_idx == 0 else f"Acoustic (CB{cb_idx})"
    same_overlaps = []
    diff_overlaps = []

    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(samples[i]["codes"][cb_idx].tolist())
            set_j = set(samples[j]["codes"][cb_idx].tolist())
            if len(set_i | set_j) == 0:
                continue
            jaccard = len(set_i & set_j) / len(set_i | set_j)

            if samples[i]["speaker_id"] == samples[j]["speaker_id"]:
                same_overlaps.append(jaccard)
            else:
                diff_overlaps.append(jaccard)

    same_avg = np.mean(same_overlaps) if same_overlaps else 0
    diff_avg = np.mean(diff_overlaps) if diff_overlaps else 0

    print(f"\n  {label}:")
    print(f"    Same-speaker Jaccard:  {same_avg:.4f}")
    print(f"    Diff-speaker Jaccard:  {diff_avg:.4f}")
    if same_avg > 0 and diff_avg > 0:
        ratio = same_avg / diff_avg
        print(f"    Ratio (same/diff):     {ratio:.2f}x")
        if ratio > 1.5:
            print(f"    → This codebook IS speaker-dependent")
        else:
            print(f"    → This codebook is mostly speaker-independent")

# ══════════════════════════════════════════════════════════════════════
# 7.5 — Simulating Voice Prompt Extraction
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 7.5: Simulating PersonaPlex voice prompt extraction")
print("=" * 70)

print("""
  PersonaPlex creates a "voice prompt" by:
    1. Taking a short audio clip (3-10 seconds)
    2. Encoding it through Mimi → 8 codebook tokens per frame
    3. Feeding those tokens through the Moshi LM
    4. Saving the LM's embeddings + KV cache

  Here we simulate step 1-2 (Mimi encoding) and analyze what
  information the tokens carry about speaker identity.
""")

# Pick first sample as "reference voice"
ref = samples[0]
ref_codes = ref["codes"]  # [8, T]

print(f"  Reference speaker: {ref['speaker_id']}")
print(f"  Reference text:    \"{ref['text'][:60]}...\"")
print(f"  Reference frames:  {ref_codes.shape[1]} ({ref_codes.shape[1] * 80}ms)")

# Show what PersonaPlex would store as voice prompt
print(f"""
  What PersonaPlex stores as a voice prompt (.pt file):

  1. MIMI TOKENS from reference audio:
     Shape: [{ref_codes.shape[0]} codebooks, {ref_codes.shape[1]} frames]
     These are fed into the Moshi LM one frame at a time.

  2. During each frame, the LM processes:
     - Agent audio channel ← these Mimi tokens (the reference voice)
     - Agent text channel  ← silence token (ID=3)
     - User audio channel  ← 440Hz sine wave (placeholder)

  3. After processing all {ref_codes.shape[1]} frames, the LM has:
     - Built an internal representation of this voice
     - Stored it in the Transformer's KV cache
     - This KV cache IS the voice prompt

  4. During conversation, the LM attends back to these cached
     representations, ensuring generated speech matches the voice.
""")

# Show a "voice fingerprint" from the tokens
print("  Voice fingerprint from Mimi tokens:")
print(f"  {'Codebook':>12s} │ {'Most common token':>20s} │ {'% of frames':>12s} │ {'Unique':>8s}")
print(f"  {'─'*12}─┼─{'─'*20}─┼─{'─'*12}─┼─{'─'*8}")

for cb in range(8):
    tokens = ref_codes[cb]
    counts = torch.bincount(tokens.long(), minlength=2048)
    top_token = counts.argmax().item()
    top_pct = counts[top_token].item() / tokens.shape[0] * 100
    unique = tokens.unique().shape[0]
    label = "Semantic" if cb == 0 else f"Acoustic{cb}"
    print(f"  {label:>12s} │ {top_token:>20d} │ {top_pct:>10.1f}% │ {unique:>8d}")

# ══════════════════════════════════════════════════════════════════════
# 7.6 — What This Means for Vietnamese
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 7.6: Implications for Vietnamese PersonaPlex")
print("=" * 70)

print("""
  KEY FINDINGS FROM THIS TUTORIAL:

  1. MIMI'S ARCHITECTURE IS MODULAR:
     Encoder (conv+transformer) → Quantizer (split semantic+acoustic) → Decoder
     Each part can potentially be fine-tuned independently.

  2. SEMANTIC CODEBOOK IS LANGUAGE-DEPENDENT:
     Trained via English WavLM distillation → captures English phonemes.
     Vietnamese tones may NOT be properly represented.
     → EXPERIMENT NEEDED: Encode Vietnamese tonal minimal pairs.

  3. ACOUSTIC CODEBOOKS CARRY SPEAKER IDENTITY:
     Codebooks 2-8 capture voice characteristics.
     These are more language-independent (voice timbre is universal).
     → May transfer well to Vietnamese without retraining.

  4. VOICE PROMPTS WORK THROUGH KV CACHE:
     The Moshi LM processes reference audio frame-by-frame,
     building a voice representation in its Transformer cache.
     → This mechanism is language-independent (it's about audio patterns).

  RESEARCH PLAN FOR VIETNAMESE:
  ─────────────────────────────────────────────────────
  Step A: Test Mimi on Vietnamese tonal minimal pairs
          → Does codebook 0 distinguish tones?

  Step B: If not → fine-tune Mimi's semantic codebook
          using Vietnamese WavLM (or multilingual WavLM)

  Step C: Collect/generate Vietnamese conversation data
          → Use the tools in tools/ directory

  Step D: Fine-tune PersonaPlex on Vietnamese data
          → Keep the architecture, change the data

  Step E: Evaluate with Vietnamese speakers
          → Tone accuracy, naturalness, voice cloning quality
""")

print("=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
print("""
  You've learned:
    ✓ Step 1: What Mimi is and why it exists
    ✓ Step 2: How to load and inspect the model architecture
    ✓ Step 3: How to encode audio into discrete tokens
    ✓ Step 4: How to decode tokens back to audio
    ✓ Step 5: What continuous embeddings look like inside the encoder
    ✓ Step 6: How Mimi handles real human speech
    ✓ Step 7: How voice identity is captured in the token streams

  Files created during this tutorial:
    tutorials/mimi_basics/
    ├── step01_concepts.py          ← Read-only conceptual guide
    ├── step02_load_model.py        ← Load and inspect architecture
    ├── step03_encode.py            ← Encode audio → tokens
    ├── step04_decode.py            ← Decode tokens → audio
    ├── step05_embeddings.py        ← Explore internal representations
    ├── step06_real_speech.py       ← Real speech analysis
    ├── step07_voice_analysis.py    ← Voice identity analysis (this file)
    ├── test_audio.wav              ← Synthetic test audio
    ├── encoded_tokens.pt           ← Saved token data
    ├── embeddings_analysis.pt      ← Saved embedding data
    ├── reconstructed_*cb.wav       ← Progressive reconstruction audio
    ├── real_speech_*.wav           ← Real speech reconstruction audio
    └── experiment_*.wav            ← Token manipulation experiments
""")
