# v5.7.1 — CRT Continuous Latent Refinement

## Motivation

Previous CRT versions (v5.3-v5.6) operate on discrete codebook tokens. The Mimi codec
uses a 32-codebook RVQ with 2048 entries each. CRT predicts token indices via cross-entropy
loss, but this introduces a hard quantization bottleneck: even perfect token prediction
still yields quantized audio. Fine acoustic details (breathiness, micro-prosody, vocal
texture) that fall between codebook entries are permanently lost.

v5.7.1 eliminates this bottleneck by operating directly in Mimi's continuous latent space.

## Architecture

### Target Space: quantizer.decode() output

The model operates in the **post-quantization embedding space**, not the pre-quantization
encoder output:

```python
codes = mimi.encode(audio).audio_codes   # [1, 32, T] discrete
embeddings = mimi.quantizer.decode(codes) # [1, 512, T] continuous, range +/-42
```

This is the space the Mimi decoder expects. Pre-quantization latents (range +/-0.09) are
a different space and would require the full encoder-quantizer pipeline to decode.

### Decoder Bypass (verified L1 = 0.0)

At inference, refined embeddings bypass the quantizer entirely:

```python
up = mimi.upsample(refined_emb)           # [1, 512, T_conv]
dec = mimi.decoder_transformer(up.T)[0].T  # [1, 512, T_conv]
audio = mimi.decoder(dec)                  # [1, 1, samples]
```

This produces audio identical to `mimi.decode(codes)` when given the same embeddings
(verified: L1 difference = 0.0). The bypass allows the model to output arbitrary
continuous values, not just points on the codebook grid.

### CRTLatent Model

Same Conformer backbone as CRT v5.6.1 (d_model=1280, n_layers=16, ~650M params),
but with continuous input/output:

- **Input projection**: Linear(512 -> 1280) + LayerNorm (replaces discrete embeddings)
- **Output projection**: Linear(1280 -> 512) + **residual connection**
- **Residual**: `output = proj(x) + corrupted_input` — predicts correction only
- **Loss**: L1 reconstruction + 0.1 * SECS speaker similarity (replaces cross-entropy)

The residual connection is key: in continuous space, the FAISS-retrieved embedding is a
reasonable base signal (correct speaker timbre, approximate phonetics). The model only
needs to predict smooth corrections, not reconstruct the full embedding from scratch.

### Comparison with CRT v5.6.1

| Aspect | v5.6.1 (discrete) | v5.7.1 (continuous) |
|--------|-------------------|---------------------|
| Input/Output | Token indices [B, N_CB, T] | Embeddings [B, T, 512] |
| Loss | Focal cross-entropy | L1 + cosine similarity |
| Output space | 2048 discrete points per CB | Continuous R^512 |
| Decode path | mimi.decode(codes) | upsample -> decoder_transformer -> decoder |
| Residual | N/A (classification) | Yes (predict correction) |
| Quantization | Yes (codebook lookup) | No (bypass) |

## Data Preprocessing

Requires a preprocessing step to extract continuous embeddings from existing discrete codes:

```bash
python scripts/preprocess_latents.py \
    --input_dir  data_storage/processed_merged \
    --output_dir data_storage/processed_merged_latents
```

This runs `mimi.quantizer.decode(codes)` for each speaker and stores embeddings as float16
(~5GB total for 3242 speakers). Much faster than re-encoding audio since we already have
the discrete codes.

## Training

FAISS corruption is unchanged: CB0-based retrieval from a different speaker provides
the corrupted signal. But instead of returning discrete tokens, we return the continuous
embeddings at the selected frame indices.

Random corruption replaces spans with Gaussian noise matching the embedding statistics.

Curriculum corruption phases: 30% -> 50% -> 70% over three equal thirds of training.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| latent_dim | 512 (fixed by Mimi) |
| d_model | 1280 |
| n_layers | 16 |
| steps | 120,000 |
| batch_size | 128 |
| infer_every | 5,000 |
| curriculum | 30% -> 50% -> 70% |
| secs_weight | 0.1 |
| dataset | processed_merged_latents (3242 speakers, ~110h) |

## Inference Outputs

Three audio files for comparison:

1. `faiss_baseline.wav` — Standard FAISS with discrete decode (same as v5.6.1)
2. `faiss_latent_bypass.wav` — FAISS-selected embeddings decoded via bypass (shows bypass quality matches discrete)
3. `crt_latent_refined.wav` — CRT-refined embeddings decoded via bypass (the new method)

## Expected Improvements

- Smoother transitions between frames (no codebook quantization jumps)
- Better preservation of fine acoustic detail (breathiness, room tone)
- Potential for higher SECS (speaker similarity) since the model can make fine-grained
  adjustments instead of picking from a fixed codebook vocabulary
