"""
╔══════════════════════════════════════════════════════════════════════╗
║              MIMI TUTORIAL — Step 1: Core Concepts                  ║
║                                                                      ║
║  This file is NOT runnable code. It is a visual textbook explaining  ║
║  what Mimi is, why it exists, and how it works conceptually.         ║
║  Read this FIRST before running the other steps.                     ║
╚══════════════════════════════════════════════════════════════════════╝

========================================================================
 WHAT IS MIMI?
========================================================================

Mimi is a NEURAL AUDIO CODEC made by Kyutai Labs (Paris, France).
Think of it like MP3/AAC but powered by a neural network.

  Traditional codec:  WAV → [FFT + psychoacoustic rules] → compressed bits → [inverse] → WAV
  Neural codec (Mimi): WAV → [Neural Network Encoder] → discrete tokens → [Neural Network Decoder] → WAV

Key difference: instead of hand-designed compression rules, a neural network
LEARNS how to compress audio by training on millions of hours of speech.

========================================================================
 WHY DOES MIMI EXIST?
========================================================================

The goal: make it possible for Large Language Models (LLMs) to "speak".

LLMs work with TOKENS (discrete integers). Text is easy to tokenize:
  "Hello world" → [15496, 995]

But audio is a continuous waveform — a stream of floating-point samples
at 24,000 per second. An LLM can't process this directly.

Mimi's job: convert audio into a small number of integer tokens
that an LLM can predict, just like it predicts text tokens.

  24,000 Hz audio  ──Mimi──►  12.5 tokens/second (per codebook)

That's a 1,920x compression ratio! (24000 / 12.5 = 1920)

========================================================================
 THE THREE COMPONENTS
========================================================================

Mimi has three parts:

  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │ ENCODER  │ ──► │  QUANTIZER   │ ──► │ DECODER  │
  │ (audio   │     │  (continuous │     │ (tokens  │
  │  → embed)│     │   → tokens)  │     │  → audio)│
  └──────────┘     └──────────────┘     └──────────┘

1. ENCODER: Compresses raw audio into continuous embeddings
   - Input:  [1, 1, 72000]  = 3 seconds of 24kHz mono audio
   - Output: [1, 512, 37]   = 37 frames of 512-dim vectors at 12.5 Hz

2. QUANTIZER: Converts continuous embeddings to discrete tokens
   - Input:  [1, 512, 37]   = continuous embeddings
   - Output: [1, 8, 37]     = 8 integer tokens per frame

3. DECODER: Reconstructs audio from tokens (via embeddings)
   - Input:  [1, 8, 37]     = discrete tokens
   - Output: [1, 1, 72000]  = reconstructed audio

========================================================================
 THE ENCODER (Details)
========================================================================

The encoder is a "SEANet" architecture — a stack of:

  1. Convolutional layers that progressively downsample the audio
  2. A Transformer that adds temporal context

  24,000 Hz raw audio
      │
      ▼  Conv1d (no downsampling, feature extraction)
  24,000 Hz features
      │
      ▼  Residual blocks + Conv1d stride=8
  3,000 Hz features
      │
      ▼  Residual blocks + Conv1d stride=5
  600 Hz features
      │
      ▼  Residual blocks + Conv1d stride=4
  150 Hz features
      │
      ▼  Residual blocks + Conv1d stride=2
  75 Hz features                                     ← "hidden_size" dim
      │
      ▼  8-layer Transformer (causal, RoPE)
  75 Hz contextualized features
      │
      ▼  Learned Downsampler stride=6
  12.5 Hz embeddings                                 ← ready for quantization

  Total downsampling: 8 × 5 × 4 × 2 × 6 = 1,920x

========================================================================
 THE QUANTIZER (The interesting part!)
========================================================================

The quantizer converts continuous vectors into discrete integer tokens.
This is called "Vector Quantization" (VQ).

--- Basic VQ ---

Imagine you have a "codebook" — a lookup table of 2,048 vectors:

  Codebook = {
      0: [0.12, -0.34, 0.56, ...],    # entry 0
      1: [-0.23, 0.45, 0.78, ...],     # entry 1
      2: [0.89, -0.12, 0.34, ...],     # entry 2
      ...
      2047: [0.01, -0.99, 0.55, ...],  # entry 2047
  }

To quantize an embedding vector, find the NEAREST codebook entry:

  embedding = [0.11, -0.33, 0.57, ...]
  nearest = argmin(distance(embedding, codebook[i]) for i in 0..2047)
  token = 42  ← the index of the closest entry

This is like "rounding" in a high-dimensional space.

--- Residual VQ (RVQ) ---

One codebook can't capture all the detail. Solution: use MULTIPLE codebooks
in sequence, each one capturing the "residual" (leftover error).

  embedding             = [0.11, -0.33, 0.57, ...]
  codebook1_entry       = [0.12, -0.34, 0.56, ...]      ← token_1 = 42
  residual_1            = embedding - codebook1_entry
                        = [-0.01, 0.01, 0.01, ...]

  codebook2_entry       = [-0.01, 0.01, 0.00, ...]      ← token_2 = 789
  residual_2            = residual_1 - codebook2_entry
                        = [0.00, 0.00, 0.01, ...]

  codebook3_entry       = [0.00, 0.00, 0.01, ...]       ← token_3 = 156
  ...and so on for 8 codebooks

  Result: 8 integer tokens [42, 789, 156, ...] represent the embedding.

  To reconstruct: just ADD all codebook entries:
  reconstructed = codebook1[42] + codebook2[789] + codebook3[156] + ...

--- Mimi's SPLIT Quantizer ---

Mimi adds a twist: it SPLITS the quantizer into two parallel paths:

  Embedding ──┬──► Semantic VQ (1 codebook)  → token for WHAT is said
              │
              └──► Acoustic RVQ (7 codebooks) → tokens for HOW it sounds

  Codebook 1 (Semantic):
    - Trained to match WavLM embeddings (a speech understanding model)
    - Captures: phonemes, words, linguistic content
    - Invariant to: speaker voice, recording quality

  Codebooks 2-8 (Acoustic):
    - Standard RVQ on the residual
    - Capture: voice timbre, pitch, formants, acoustic texture
    - Each subsequent codebook captures finer detail

  KEY: Both paths quantize the SAME input embedding (parallel, not serial).
  The decode is additive: semantic_decoded + acoustic_decoded = full audio.

========================================================================
 THE DECODER
========================================================================

Mirror image of the encoder:

  12.5 Hz token embeddings
      │
      ▼  Quantizer decode (tokens → continuous embeddings)
  12.5 Hz embeddings
      │
      ▼  Learned Upsampler stride=6
  75 Hz embeddings
      │
      ▼  8-layer Transformer (causal)
  75 Hz contextualized embeddings
      │
      ▼  Transposed Conv stride=2
  150 Hz features
      │
      ▼  Transposed Conv stride=4
  600 Hz features
      │
      ▼  Transposed Conv stride=5
  3,000 Hz features
      │
      ▼  Transposed Conv stride=8
  24,000 Hz reconstructed audio

========================================================================
 KEY NUMBERS TO REMEMBER
========================================================================

  Audio sample rate:     24,000 Hz
  Token frame rate:      12.5 Hz  (= 1 token set per 80ms)
  Samples per frame:     1,920    (= 24000 / 12.5)
  Number of codebooks:   8        (1 semantic + 7 acoustic)
  Codebook size:         2,048    (each token is in 0..2047)
  Bitrate:               ~1.1 kbps (extremely low!)
  Model parameters:      96.2M
  Latency:               80ms     (one frame)

========================================================================
 ANALOGY: Mimi as a "Translator"
========================================================================

Think of Mimi as a translator between two languages:

  AUDIO LANGUAGE                    TOKEN LANGUAGE
  (continuous waveform)             (discrete integers)

  [0.1, -0.3, 0.5, 0.2, ...]  ←→  [[42, 789, 156, 23, 901, 45, 333, 12]]
  24,000 numbers per second         8 numbers per 80ms

  An LLM "speaks" the TOKEN LANGUAGE.
  Mimi translates TOKEN → AUDIO so humans can hear it.
  Mimi translates AUDIO → TOKEN so the LLM can understand.

========================================================================
 NEXT: Go run step02_load_model.py to see this in action!
========================================================================
"""

if __name__ == "__main__":
    print("This file is a conceptual guide. Read the docstring above.")
    print("Next: python tutorials/mimi_basics/step02_load_model.py")
