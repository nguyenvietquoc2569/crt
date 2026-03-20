# Mimi Neural Audio Codec — Step-by-Step Tutorial

A hands-on tutorial for understanding the Mimi neural audio codec,
the foundational component of NVIDIA PersonaPlex and Kyutai Moshi.

## Prerequisites

```bash
# From the project root (/Users/nguyenquoc/duplex)
source .venv/bin/activate
```

Dependencies (already installed in `.venv`):
- torch, torchaudio, transformers, soundfile, numpy, matplotlib
- datasets, torchcodec (for Step 6-7)
- ffmpeg (system package via `brew install ffmpeg`)

## Tutorial Steps

Run each step in order. Each script prints detailed explanations
and generates output files you can inspect.

### Step 1: Core Concepts (read-only)

```bash
# Just read the file — it's a visual textbook, not runnable code
cat tutorials/mimi_basics/step01_concepts.py
```

What you'll learn:
- What is a neural audio codec and why Mimi exists
- The three components: Encoder, Quantizer, Decoder
- How Vector Quantization (VQ) and Residual VQ (RVQ) work
- Mimi's Split Quantizer: semantic vs acoustic codebooks
- Key numbers: 24kHz input, 12.5Hz tokens, 8 codebooks, 2048 entries

### Step 2: Load Model & Inspect Architecture

```bash
python tutorials/mimi_basics/step02_load_model.py
```

What you'll learn:
- How to load Mimi from HuggingFace (`kyutai/mimi`, 96.2M params)
- Model configuration: hidden_size=512, 8 codebooks, 2048 entries each
- Architecture breakdown: encoder (12.6M), transformers (50.4M), quantizer (524K)
- Quick sanity check: can it process audio?

### Step 3: Encode Audio to Tokens

```bash
python tutorials/mimi_basics/step03_encode.py
```

What you'll learn:
- How to use the feature extractor to preprocess audio
- Encoding: 72,000 audio samples → 38 frames of 8 tokens each
- What each codebook's tokens look like (token value ranges, uniqueness)
- Compression stats: 344x compression, 1.1 kbps bitrate

Output: `encoded_tokens.pt` (saved tokens for Step 4)

### Step 4: Decode Tokens Back to Audio

```bash
python tutorials/mimi_basics/step04_decode.py
```

What you'll learn:
- Decoding tokens back to audio waveform
- Quality metrics: SNR, correlation between original and reconstructed
- **Progressive decode**: hear what each codebook adds (1cb → 8cb)
- Token manipulation: what happens when you zero-out or shuffle codebooks

Output files (listen to these!):
- `reconstructed_1cb.wav` through `reconstructed_8cb.wav` — progressive quality
- `experiment_no_semantic.wav` — acoustic without semantic
- `experiment_semantic_only.wav` — semantic without acoustic
- `experiment_shuffled_semantic.wav` — scrambled semantic tokens

### Step 5: Inside the Encoder — Continuous Embeddings

```bash
python tutorials/mimi_basics/step05_embeddings.py
```

What you'll learn:
- Walking through the encoder layer by layer (conv → transformer → downsampler → RVQ)
- Shape at every stage: [1,1,72000] → [1,512,75] → [1,512,38] → [1,8,38]
- Continuous embeddings before quantization (512-dim vectors at 12.5Hz)
- Quantization error analysis
- Codebook utilization patterns
- Frame similarity in embedding space

### Step 6: Real Human Speech

```bash
python tutorials/mimi_basics/step06_real_speech.py
```

What you'll learn:
- Encoding real speech from LibriSpeech dataset
- Real speech uses 5-20x more codebook entries than synthetic audio
- Progressive quality with real voice (hear the difference!)
- Semantic token patterns: common phonemes → common tokens
- Cross-sample token sharing: different utterances share semantic tokens

Output: `real_speech_original.wav` and `real_speech_1cb.wav` through `real_speech_8cb.wav`

### Step 7: Voice Identity Analysis

```bash
python tutorials/mimi_basics/step07_voice_analysis.py
```

What you'll learn:
- How voice identity is captured in Mimi's embeddings
- Speaker similarity via cosine similarity of mean embeddings
- Where voice identity lives: semantic vs acoustic codebooks
- How PersonaPlex creates voice prompts from Mimi tokens
- Implications for Vietnamese adaptation

## Key Takeaways

1. **Mimi compresses audio 1,920x** (24kHz → 12.5Hz tokens) at 1.1 kbps
2. **Split quantizer** separates WHAT is said (semantic) from HOW it sounds (acoustic)
3. **Codebook 0** (semantic) is trained via WavLM distillation — captures linguistic content
4. **Codebooks 1-7** (acoustic RVQ) capture voice timbre, pitch, prosody progressively
5. **Voice prompts** in PersonaPlex are sequences of Mimi tokens + LM KV cache state
6. **Vietnamese challenge**: semantic codebook may not distinguish tones (needs testing)

## File Structure

```
tutorials/mimi_basics/
├── README.md                       ← This file
├── step01_concepts.py              ← Conceptual guide (read-only)
├── step02_load_model.py            ← Load and inspect architecture
├── step03_encode.py                ← Encode audio → tokens
├── step04_decode.py                ← Decode tokens → audio + experiments
├── step05_embeddings.py            ← Internal representation analysis
├── step06_real_speech.py           ← Real speech encoding
├── step07_voice_analysis.py        ← Voice identity analysis
├── test_audio.wav                  ← Generated synthetic test audio
├── encoded_tokens.pt               ← Saved tokens (from step 3)
├── embeddings_analysis.pt          ← Saved embeddings (from step 5)
├── reconstructed_*cb.wav           ← Progressive reconstruction (step 4)
├── real_speech_*.wav               ← Real speech outputs (step 6)
└── experiment_*.wav                ← Token manipulation outputs (step 4)
```
