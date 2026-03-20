# Mimi Voice Embedding Deep Dive

## 1. Mimi Architecture Overview

Mimi is a **streaming neural audio codec** (96.2M parameters) that converts raw audio
waveforms into discrete tokens and back. It is the foundational component that makes
PersonaPlex's speech-to-speech modeling possible.

```
                         MIMI CODEC
┌──────────────────────────────────────────────────────┐
│                                                       │
│  24kHz Audio ──► [ENCODER] ──► [QUANTIZER] ──► Codes  │
│                                                       │
│  Codes ──► [DEQUANTIZER] ──► [DECODER] ──► 24kHz Audio│
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## 2. Encoder Architecture

The encoder is a **SEANet-style convolutional encoder** with a Transformer bottleneck.

### 2.1 Convolutional Layers

```
Input: Raw audio [B, 1, num_samples] at 24,000 Hz

Layer 0: Conv1d(1, num_filters, kernel_size=7)
    ↓
Downsampling Block 1: stride=8
    ├── ResnetBlock(num_filters, dilation=[1, 1])
    ├── ResnetBlock(num_filters, dilation=[3, 1])
    ├── ELU activation
    └── Conv1d(num_filters, num_filters*2, kernel=16, stride=8)
    ↓
Downsampling Block 2: stride=5
    ├── ResnetBlock(num_filters*2, dilation=[1, 1])
    ├── ResnetBlock(num_filters*2, dilation=[3, 1])
    ├── ELU activation
    └── Conv1d(num_filters*2, num_filters*4, kernel=10, stride=5)
    ↓
Downsampling Block 3: stride=4
    ├── ResnetBlock(num_filters*4, dilation=[1, 1])
    ├── ResnetBlock(num_filters*4, dilation=[3, 1])
    ├── ELU activation
    └── Conv1d(num_filters*4, num_filters*8, kernel=8, stride=4)
    ↓
Downsampling Block 4: stride=2
    ├── ResnetBlock(num_filters*8, dilation=[1, 1])
    ├── ResnetBlock(num_filters*8, dilation=[3, 1])
    ├── ELU activation
    └── Conv1d(num_filters*8, num_filters*16, kernel=4, stride=2)

Total stride: 8 × 5 × 4 × 2 = 320
24,000 Hz / 320 = 75 Hz intermediate rate
```

### 2.2 Transformer Encoder (Bottleneck)

After the convolutional encoder, a Transformer adds **temporal context**:

```
Transformer Encoder:
    - 8 layers
    - 8 attention heads
    - RoPE (Rotary Positional Embeddings)
    - Finite context window: 250 frames (20 seconds)
    - Causal masking (for streaming compatibility)
    - All operations are causal → enables real-time streaming
```

### 2.3 Downsampler

After the Transformer, a **learned downsampler** reduces from 75 Hz to 12.5 Hz:

```
75 Hz (post-conv) ──► Downsampler (stride=6) ──► 12.5 Hz (codebook rate)
```

### 2.4 Complete Encoder Flow

```
24,000 Hz PCM audio [B, 1, T_samples]
    ↓ Conv encoder (stride 320)
75 Hz continuous embeddings [B, hidden_dim, T_samples/320]
    ↓ Transformer encoder (8 layers, causal)
75 Hz contextualized embeddings [B, hidden_dim, T_samples/320]
    ↓ Learned downsampler (stride 6)
12.5 Hz embeddings [B, hidden_dim, T_frames]
    ↓ Ready for quantization
```

Where `T_frames = T_samples / (320 × 6) = T_samples / 1920`

---

## 3. Split Residual Vector Quantizer

This is Mimi's key innovation: it **splits** quantization into two parallel paths.

### 3.1 Architecture

```python
class MimiSplitResidualVectorQuantizer:
    """
    Two separate quantizer banks operating on the SAME embeddings.
    """
    num_semantic_quantizers = 1      # First codebook: SEMANTIC (VQ)
    num_acoustic_quantizers = 7      # Remaining 7: ACOUSTIC (RVQ)

    semantic_residual_vector_quantizer   # → 1 codebook
    acoustic_residual_vector_quantizer   # → 7 codebooks
```

### 3.2 Semantic Quantizer (Codebook 1)

The first codebook is trained with **WavLM distillation**:

```
Mimi Encoder output ──► VQ (codebook_size=2048) ──► Semantic token
                                    ↕
                           Distillation Loss
                                    ↕
                    WavLM embedding ──► Target representation
```

**How distillation works:**
1. Pass audio through WavLM (a pre-trained self-supervised speech model)
2. Pass same audio through Mimi encoder
3. Apply **cosine similarity loss** to make Mimi's first codebook vectors
   match WavLM's representations
4. Result: Codebook 1 tokens capture **what is being said** (phonemes, words)
   and are largely **invariant to voice identity**

```
Codebook 1 encodes: SEMANTIC information
    ✅ Phonemes, words, linguistic content
    ✅ Prosodic structure (intonation patterns)
    ❌ Speaker identity is partially removed
    ❌ Fine acoustic details stripped away
```

### 3.3 Acoustic Quantizer (Codebooks 2-8)

The remaining 7 codebooks use standard **Residual Vector Quantization**:

```
Input embedding ──► [Semantic VQ] ──► semantic_code, residual_1
    residual_1  ──► [Acoustic VQ layer 2] ──► acoustic_code_2, residual_2
    residual_2  ──► [Acoustic VQ layer 3] ──► acoustic_code_3, residual_3
    ...
    residual_6  ──► [Acoustic VQ layer 7] ──► acoustic_code_7, residual_7
    residual_7  ──► [Acoustic VQ layer 8] ──► acoustic_code_8
```

Each subsequent codebook captures finer details:
```
Codebook 2: Coarse acoustic structure (pitch contour, energy)
Codebook 3: Speaker timbre characteristics
Codebook 4: Formant details
Codebook 5: Fine spectral details
Codebook 6: Micro-prosodic features
Codebook 7: Residual acoustic texture
Codebook 8: Finest reconstruction details
```

### 3.4 Encode Method (Full)

```python
def encode(self, embeddings, num_quantizers=None):
    """
    embeddings: [B, hidden_dim, T_frames] from encoder

    Returns: codes [num_codebooks, B, T_frames]
        codes[0] = semantic token IDs in [0, 2047]
        codes[1] = acoustic RVQ level 1 token IDs in [0, 2047]
        ...
        codes[7] = acoustic RVQ level 7 token IDs in [0, 2047]
    """
    # Step 1: Semantic VQ (1 codebook)
    codes = self.semantic_residual_vector_quantizer.encode(embeddings)
    # codes shape: [1, B, T_frames]

    # Step 2: Acoustic RVQ (7 codebooks) on the SAME embeddings
    if num_quantizers > self.num_semantic_quantizers:
        acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
            embeddings,
            num_quantizers=(num_quantizers - self.num_semantic_quantizers)
        )
        # acoustic_codes shape: [7, B, T_frames]
        codes = torch.cat([codes, acoustic_codes], dim=0)

    return codes  # [8, B, T_frames]
```

**Critical insight**: Both the semantic and acoustic quantizers operate on the
**same input embeddings** — they are NOT sequential. The semantic VQ captures
linguistic content while the acoustic RVQ independently captures acoustic details.

### 3.5 Decode Method

```python
def decode(self, codes):
    """
    codes: [num_codebooks, B, T_frames]

    Returns: quantized embeddings [B, hidden_dim, T_frames]
    """
    # Decode semantic
    quantized_out = self.semantic_residual_vector_quantizer.decode(
        codes[:self.num_semantic_quantizers]  # codes[0:1]
    )

    # Add acoustic reconstruction
    if codes.shape[0] > self.num_semantic_quantizers:
        quantized_out += self.acoustic_residual_vector_quantizer.decode(
            codes[self.num_semantic_quantizers:]  # codes[1:8]
        )

    return quantized_out  # [B, hidden_dim, T_frames]
```

The decode is **additive**: semantic embedding + acoustic embedding = full reconstruction.

---

## 4. Decoder Architecture

Mirrors the encoder in reverse:

```
12.5 Hz quantized embeddings [B, hidden_dim, T_frames]
    ↓ Learned upsampler (stride 6)
75 Hz embeddings [B, hidden_dim, T_frames*6]
    ↓ Transformer decoder (8 layers, causal)
75 Hz contextualized embeddings
    ↓ Transposed Conv decoder (strides 2,4,5,8 — reverse of encoder)
24,000 Hz reconstructed audio [B, 1, T_samples]
```

---

## 5. Training Losses

Mimi is trained with three loss components (same as EnCodec):

### 5.1 Reconstruction Loss
```
L_recon = MSE(original_spectrogram, reconstructed_spectrogram)
    + L1(original_waveform, reconstructed_waveform)
```

### 5.2 Adversarial Loss (GAN)
```
Discriminator: classifies audio as real or reconstructed
Generator (Mimi): tries to fool the discriminator
L_adv = -log(D(Mimi_decode(Mimi_encode(x))))
```

### 5.3 WavLM Distillation Loss (Semantic)
```
L_distill = 1 - cosine_similarity(
    Mimi_semantic_codebook_vector,
    WavLM_embedding
)
```

Applied ONLY to the first codebook (semantic VQ).

### 5.4 Commitment Loss (VQ standard)
```
L_commit = ||embedding - stop_gradient(codebook_vector)||²
```

### 5.5 RVQ Dropout
During training, the number of active RVQ levels is randomly truncated.
This allows Mimi to produce decent audio even with fewer codebooks at inference:

```
# During training, randomly use 1 to 32 codebooks
num_active = random.randint(1, 32)
codes = quantizer.encode(embeddings, num_quantizers=num_active)
```

---

## 6. Voice Embedding in PersonaPlex

### 6.1 What IS a Voice Prompt?

A voice prompt is **NOT a single embedding vector**. It is a **sequence of
Mimi-encoded audio tokens** from a short reference speech sample.

```
Voice prompt .pt file contains:
{
    "embeddings": Tensor[N_frames, embed_dim],  # Pre-transformer embeddings
    "cache":      Tensor[...]                    # Transformer KV cache state
}
```

Where `N_frames` = reference_audio_duration_seconds × 12.5

### 6.2 How a Voice Prompt .pt File is Created

PersonaPlex creates voice prompts by **encoding reference audio through
the FULL model pipeline** (not just Mimi):

```python
# In LMGen._step_voice_prompt_core():

# CASE 1: From audio file (first time)
def create_voice_prompt_from_audio(audio_file):
    # 1. Load and normalize audio
    raw_audio = load_audio(audio_file, sample_rate=24000)
    raw_audio = normalize_audio(raw_audio, sample_rate=24000, target_lufs=-24.0)

    # 2. Chunk audio into frames (80ms = 1920 samples each)
    frame_size = 24000 / 12.5  # = 1920 samples per frame

    saved_embeddings = []

    for chunk in iterate_audio(raw_audio, frame_size):
        # 3. Encode audio chunk through Mimi
        #    chunk: [1, 1, 1920] → Mimi → [1, 8, 1] (8 codebook tokens)
        voice_prompt_frame_tokens = mimi.encode(chunk)
        # shape: [1, 8, 1] — one token per codebook for this 80ms frame

        # 4. Run through Moshi LM with special conditioning:
        #    - moshi_tokens = voice_prompt_frame_tokens (agent audio = voice sample)
        #    - text_token = 3 (zero/silence text token)
        #    - input_tokens = 440 Hz sine wave encoded (user channel placeholder)
        out, embeddings = lm_model.step(
            moshi_tokens=voice_prompt_frame_tokens,  # [1, 8, 1]
            text_token=3,                             # silence text
            input_tokens=encode_sine_frame(),          # [1, 8, 1] sine wave
            return_embeddings=True,
        )

        saved_embeddings.append(embeddings)

    # 5. Save the accumulated LM embeddings + KV cache
    torch.save({
        "embeddings": torch.stack(saved_embeddings, dim=0),
        "cache": streaming_state.cache,
    }, "voice_prompt.pt")


# CASE 2: From pre-saved .pt file (fast path)
def load_voice_prompt_from_pt(pt_file):
    state = torch.load(pt_file)

    # Replay embeddings through LM (skip Mimi encoding)
    for next_embed in state["embeddings"]:
        lm_model.step_embeddings(next_embed)

    # Restore KV cache directly
    streaming_state.cache.copy_(state["cache"])
```

### 6.3 The Complete Voice Prompt Pipeline (Visual)

```
Reference audio (e.g., 5 seconds of speech at 24kHz)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ STEP 1: Audio Loading & Normalization            │
│   load_audio("reference.wav", sr=24000)          │
│   normalize_audio(audio, target_lufs=-24.0)      │
│   Shape: [1, 120000] (5 sec × 24000 Hz)         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ STEP 2: Frame Chunking                           │
│   Split into 80ms frames: 120000 / 1920 = 62    │
│   Each frame: [1, 1, 1920] samples               │
└─────────────────────────────────────────────────┘
    │
    ▼ (for each of 62 frames)
┌─────────────────────────────────────────────────┐
│ STEP 3: Mimi Encoding                            │
│   Conv Encoder → Transformer → Downsampler → RVQ │
│   Input:  [1, 1, 1920] (80ms audio)             │
│   Output: [1, 8, 1] (8 codebook token IDs)      │
│                                                   │
│   tokens[0] = semantic token (word/phoneme info)  │
│   tokens[1-7] = acoustic tokens (voice details)  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ STEP 4: Moshi LM Forward Pass                    │
│                                                   │
│   Agent audio channel ← voice_prompt_tokens [1,8,1]
│   Agent text channel  ← zero_text_code (3)       │
│   User audio channel  ← sine wave tokens [1,8,1] │
│                                                   │
│   The LM "listens" to the reference voice and     │
│   builds an internal representation of how this   │
│   voice sounds, accumulating in the KV cache.     │
│                                                   │
│   Output: embeddings (pre-transformer repr.)      │
│   Also: KV cache updated with voice context       │
└─────────────────────────────────────────────────┘
    │
    ▼ (after all 62 frames)
┌─────────────────────────────────────────────────┐
│ STEP 5: Save Voice Prompt                        │
│                                                   │
│   torch.save({                                    │
│       "embeddings": [62, embed_dim],   ← LM repr │
│       "cache": transformer_kv_cache,   ← context  │
│   }, "voice.pt")                                  │
│                                                   │
│   The .pt file contains BOTH:                     │
│   1. The embedding sequence (for replay)          │
│   2. The KV cache state (for fast loading)        │
└─────────────────────────────────────────────────┘
```

### 6.4 Key Insight: Voice Prompt is NOT a Simple Embedding

Unlike speaker verification systems (e.g., x-vectors, d-vectors) that produce
a **single fixed-size vector** for a speaker, PersonaPlex's voice prompt is:

1. A **sequence** of embeddings (one per 80ms frame of reference audio)
2. Plus the **KV cache state** of the Transformer after processing all frames
3. The KV cache captures the **temporal context** of the voice — not just
   static timbre, but speaking rhythm, prosodic patterns, etc.

```
Traditional speaker embedding:  audio → model → [1, 256] fixed vector
PersonaPlex voice prompt:       audio → Mimi → LM → [N_frames, embed_dim] + KV cache
```

### 6.5 Why a Sine Wave on the User Channel?

During voice prompt processing, the user audio channel gets a **440 Hz sine wave**
(not silence):

```python
SINE_TOKENS = [precomputed_tokens]  # Fixed encoding of 440 Hz sine wave

def _encode_sine_frame(self) -> torch.Tensor:
    return torch.as_tensor(SINE_TOKENS, dtype=torch.long,
                           device=self.lm_model.device).view(1, 8, 1)
```

This is because:
- Moshi was trained with dual-stream audio (always two speakers)
- Complete silence on the user channel would be out-of-distribution
- A constant sine wave provides a stable, neutral "carrier" signal
- The model learns to ignore this and focus on the agent's voice

### 6.6 How Voice Conditioning Works During Conversation

After the voice prompt is loaded (Step 5), the system prompt phase completes:

```python
def step_system_prompts(self, mimi):
    self._step_voice_prompt(mimi)     # 1. Encode voice into KV cache
    self._step_audio_silence()         # 2. Brief silence (transition)
    self._step_text_prompt()           # 3. Encode text prompt tokens
    self._step_audio_silence()         # 4. Brief silence (ready)
    # Now the model's KV cache contains:
    #   - Voice characteristics from the reference audio
    #   - Role/persona from the text prompt
    #   - The model is "primed" to generate speech in this voice+role
```

During conversation, the Transformer **attends back** to the voice prompt
in the KV cache, ensuring all generated speech matches the reference voice.

---

## 7. What the .pt File Actually Contains

Let's examine the exact tensor shapes:

```python
import torch

# Load a voice prompt
state = torch.load("NATF0.pt")

print(state.keys())
# dict_keys(['embeddings', 'cache'])

# Embeddings: sequence of LM input embeddings
embeddings = state["embeddings"]
print(embeddings.shape)
# torch.Size([N, embed_dim])
# N = number of 80ms frames in the reference audio
# embed_dim = Moshi LM hidden size (4096)

# For a 5-second reference: N = 5 * 12.5 = 62 frames
# Shape: [62, 4096]

# Cache: Transformer KV cache after processing all voice frames
cache = state["cache"]
# Contains key/value tensors for all 32 Transformer layers
# This is the "memory" of how the voice sounds
```

### 7.1 Embedding Details

Each embedding in the sequence is the **pre-Transformer representation**
at one time step, computed by:

```python
# In LMModel.embed_codes():
embeddings = sum(
    emb_layer(code_stream)
    for emb_layer, code_stream
    in zip(self.emb, codes_per_stream)
)
# This sums embeddings from:
#   - text token embedding (stream 0)
#   - audio codebook 1 embedding (stream 1)
#   - audio codebook 2 embedding (stream 2)
#   - ...
#   - audio codebook 8 embedding (stream 8)
# Result: single combined embedding per timestep
```

### 7.2 Cache Details

The KV cache stores the Transformer's attention state:

```python
# For each of 32 Transformer layers:
#   key:   [batch, num_heads, seq_len, head_dim]
#   value: [batch, num_heads, seq_len, head_dim]
#
# Where:
#   num_heads = 32
#   head_dim = 4096 / 32 = 128
#   seq_len = number of voice prompt frames processed
```

---

## 8. Using HuggingFace Transformers API for Mimi

### 8.1 Basic Encode/Decode

```python
from transformers import MimiModel, AutoFeatureExtractor
import torch
import torchaudio

# Load model
model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

# Load audio (MUST be 24kHz)
audio, sr = torchaudio.load("speech.wav")
if sr != 24000:
    audio = torchaudio.functional.resample(audio, sr, 24000)

# Prepare input
inputs = feature_extractor(
    raw_audio=audio.squeeze().numpy(),
    sampling_rate=24000,
    return_tensors="pt"
)

# ENCODE: audio → tokens
encoder_output = model.encode(inputs["input_values"])
codes = encoder_output.audio_codes
# codes shape: [batch, num_codebooks=8, num_frames]
# codes values: integers in [0, 2047]

print(f"Audio duration: {audio.shape[1]/24000:.2f}s")
print(f"Number of frames: {codes.shape[2]}")
print(f"Frame rate: {codes.shape[2] / (audio.shape[1]/24000):.1f} Hz")
# Expected: 12.5 Hz

# Inspect individual codebooks
print(f"Codebook 1 (semantic): {codes[0, 0, :10]}")  # First 10 semantic tokens
print(f"Codebook 2 (acoustic): {codes[0, 1, :10]}")  # First 10 acoustic tokens

# DECODE: tokens → audio
reconstructed = model.decode(codes)
audio_out = reconstructed.audio_values
# audio_out shape: [batch, channels=1, num_samples]
```

### 8.2 Examining Semantic vs Acoustic Tokens

```python
# Encode with only semantic codebook (1 codebook)
semantic_only = model.encode(inputs["input_values"], num_quantizers=1)
audio_semantic = model.decode(semantic_only.audio_codes).audio_values

# Encode with all 8 codebooks
full_codes = model.encode(inputs["input_values"], num_quantizers=8)
audio_full = model.decode(full_codes.audio_codes).audio_values

# Compare:
# - semantic_only: recognizable speech but robotic, missing speaker details
# - audio_full: high-quality reconstruction with speaker identity preserved
```

### 8.3 Extracting Continuous Embeddings (Pre-Quantization)

```python
# Access the encoder's continuous embeddings BEFORE quantization
# This requires accessing model internals

with torch.no_grad():
    # Step 1: Run convolutional encoder
    encoder_hidden = model.encoder(inputs["input_values"])
    # shape: [B, hidden_dim, T_conv]

    # Step 2: Run Transformer encoder
    transformer_out = model.encoder_transformer(
        encoder_hidden.transpose(1, 2)
    )
    contextualized = transformer_out[0].transpose(1, 2)
    # shape: [B, hidden_dim, T_conv]

    # Step 3: Downsample to codebook rate
    embeddings = model.downsample(contextualized)
    # shape: [B, hidden_dim, T_frames] at 12.5 Hz

    # These are the continuous embeddings that get quantized
    # They contain ALL information about the audio
    # (semantic + acoustic + speaker identity)
    print(f"Continuous embedding shape: {embeddings.shape}")
    print(f"Hidden dim: {embeddings.shape[1]}")
```

---

## 9. Vietnamese Tone Representation Experiment

A critical experiment for your research: can Mimi's codebooks represent
Vietnamese tones correctly?

### 9.1 Experiment Design

```python
# Test: Encode Vietnamese minimal tone pairs
# If Mimi handles tones correctly, different tones should produce
# different token sequences (especially in codebook 1 / semantic)

tone_pairs = {
    "ma":  "ghost (level tone - ngang)",
    "mà":  "but (falling tone - huyền)",
    "má":  "mother (rising tone - sắc)",
    "mả":  "grave (dipping tone - hỏi)",
    "mã":  "horse/code (broken tone - ngã)",
    "mạ":  "rice seedling (heavy tone - nặng)",
}

results = {}
for word, meaning in tone_pairs.items():
    # Synthesize audio for each word using Vietnamese TTS
    audio = vietnamese_tts.synthesize(word, sr=24000)

    inputs = feature_extractor(raw_audio=audio, sampling_rate=24000,
                               return_tensors="pt")
    codes = model.encode(inputs["input_values"]).audio_codes

    results[word] = {
        "meaning": meaning,
        "semantic_tokens": codes[0, 0, :].tolist(),   # Codebook 1
        "acoustic_tokens_2": codes[0, 1, :].tolist(),  # Codebook 2
    }

# Compare: Do different tones produce different semantic tokens?
# If yes: Mimi can distinguish tones → fine-tuning may be straightforward
# If no: Mimi merges tones → need to retrain/fine-tune Mimi on Vietnamese
```

### 9.2 What to Expect

**Likely outcome**: Mimi's semantic codebook (trained via English WavLM distillation)
will likely **NOT** distinguish Vietnamese tones well, because:
- WavLM was trained primarily on English speech
- English does not use lexical tones
- Tonal information may be captured in acoustic codebooks (2-8) instead of semantic (1)

**Research contribution**: Analyzing and fixing this is a publishable finding.

### 9.3 Potential Fix: Fine-tune Mimi with Vietnamese WavLM

```
Option A: Use a Vietnamese/multilingual WavLM for distillation
    - Train Mimi's semantic codebook against Vietnamese WavLM embeddings
    - Vietnamese WavLM understands tones → semantic codebook will too

Option B: Fine-tune existing Mimi on Vietnamese audio
    - Keep the architecture, retrain on Vietnamese speech data
    - May need to adjust the distillation loss weight

Option C: Add a "tone codebook"
    - Add a 9th codebook specifically for tonal information
    - Trained with a tone-classification auxiliary loss
```

---

## 10. Summary: From Audio to Voice Embedding

```
┌─────────────────────────────────────────────────────────────┐
│              COMPLETE VOICE EMBEDDING PIPELINE                │
│                                                               │
│  Reference Audio (24kHz WAV, 3-10 seconds)                   │
│      │                                                        │
│      ▼                                                        │
│  Normalize to -24 LUFS                                        │
│      │                                                        │
│      ▼                                                        │
│  Chunk into 80ms frames (1920 samples each)                  │
│      │                                                        │
│      ▼ ────────── For each frame ──────────                   │
│      │                                                        │
│      ├──► Mimi Encoder                                        │
│      │    ├── Conv layers (stride 320: 1920→6 samples)        │
│      │    ├── Transformer (8 layers, causal)                  │
│      │    ├── Downsampler (stride 6: 6→1 frame)              │
│      │    └── Split RVQ                                       │
│      │        ├── Semantic VQ: 1 token (word content)         │
│      │        └── Acoustic RVQ: 7 tokens (voice details)     │
│      │    Output: [1, 8, 1] token IDs                        │
│      │                                                        │
│      ├──► Moshi LM Step                                       │
│      │    ├── Agent audio ← Mimi tokens from reference       │
│      │    ├── Agent text  ← silence token (3)                │
│      │    ├── User audio  ← 440Hz sine wave tokens           │
│      │    └── Updates KV cache with voice context             │
│      │    Output: embedding vector + updated KV cache        │
│      │                                                        │
│      ▼ ────────── End loop ──────────                        │
│                                                               │
│  Save to .pt file:                                            │
│      embeddings: [N_frames, 4096]  (LM input embeddings)     │
│      cache: KV cache state         (Transformer memory)       │
│                                                               │
│  This .pt file IS the "voice embedding"                       │
│  It captures: timbre, pitch, rhythm, prosody, speaking style  │
└─────────────────────────────────────────────────────────────┘
```
