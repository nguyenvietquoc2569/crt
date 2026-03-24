# v6.0 — Codec Language Model for Voice Conversion

## Problem

CRT v5.x (retrieve-and-refine) hit a quality ceiling:

- **FAISS retrieval is frame-by-frame**, creating temporal discontinuities at token boundaries. Each frame is independently matched, ignoring the sequential nature of speech.
- **Non-autoregressive prediction** forces hard token decisions without considering left-to-right dependencies in CB0 (the semantic codebook).
- **Noisy FAISS input propagates errors**: when CB0 retrieval is wrong, CB1-31 predictions are doomed. v5.7 achieved ~57% CB1 accuracy at best because the NAR model was conditioned on noisy, frame-level-retrieved tokens.
- **CB0 is never converted** in v5.x. FAISS copies the nearest CB0 from the target speaker, but this is a lookup, not a learned transformation. Prosody and rhythm remain source-like.

## Solution

Two-stage codec language model that **generates** tokens from scratch instead of retrieving them:

- **Stage 1 (AR)**: Autoregressive transformer takes source CB0 + speaker embedding and generates target CB0 token-by-token. This handles prosody, rhythm, and phoneme realization adjustment to match the target speaker.
- **Stage 2 (NAR)**: Non-autoregressive Conformer takes the (clean) target CB0 + speaker embedding and generates all 31 acoustic codebooks (CB1-31) in parallel. This fills in timbre, breathiness, texture, and room tone.

```
Source audio -> Mimi encode -> source_CB0 [T]
                                  |
              +-------------------v-------------------+
              |  Stage 1: AR Semantic Model (~150M)    |
              |  Causal Transformer + FiLM per layer   |
              |  source_CB0 + spk_emb -> target_CB0   |
              +-------------------+-------------------+
                                  |
              +-------------------v-------------------+
              |  Stage 2: NAR Acoustic Model (~400M)   |
              |  Conformer + FiLM                      |
              |  target_CB0 + spk_emb -> CB1-31       |
              +-------------------+-------------------+
                                  |
              [target_CB0 + CB1-31] -> Mimi decode -> target audio
```

## Key Differences from v5.x

| Aspect | v5.x CRT | v6.0 Codec LM |
|---|---|---|
| CB0 handling | FAISS nearest-neighbor lookup (no learning) | AR model generates target CB0 autoregressively |
| CB1-31 input | Noisy FAISS-retrieved tokens | Clean CB0 (ground-truth or AR-generated) |
| CB1-31 method | NAR refine noisy -> clean | NAR generate from CB0 + speaker |
| Temporal coherence | Frame-independent retrieval | Causal attention (AR) + full attention (NAR) |
| Prosody transfer | None (copies source prosody) | AR model learns speaker-specific prosody |
| Max CB1 accuracy | ~57% (limited by FAISS noise) | Expected higher (clean input) |
| Architecture | Single Conformer | Two-stage: Causal Transformer + Conformer |
| Total params | ~400M | ~550M (150M AR + 400M NAR) |

## Why 31 CBs Should Work Now

v5.7 failed at predicting all 31 codebooks because the input was noisy FAISS-retrieved tokens. The NAR model had to simultaneously:
1. Correct errors in the noisy input
2. Predict 31 codebooks from that corrected signal

This is fundamentally harder than what v6.0 NAR does:
- CB0 input is **clean** (either ground-truth during training, or AR-generated at inference)
- The NAR model only needs to predict CB1-31 conditioned on reliable CB0
- CB0 perturbation during training (30% probability) makes NAR robust to AR errors

## Training

Three phases, run in order:

### Phase 1: Train NAR with ground-truth CB0

Validates the approach by training NAR on clean CB0 tokens. If NAR cannot learn CB1-31 from clean CB0, the whole approach fails.

```bash
sbatch run_nar.slurm
```

### Phase 2: Train AR for CB0 conversion

Train the autoregressive model to convert source CB0 to target CB0 with speaker conditioning.

```bash
sbatch run_ar.slurm
```

### Phase 3: Combined inference (AR -> NAR)

Chain AR output into NAR for end-to-end voice conversion.

```bash
sbatch run_inference.slurm
```

## Usage

### Training (individual stages)

```bash
# NAR (Stage 2) — train first to validate approach
sbatch run_nar.slurm

# AR (Stage 1) — train after NAR proves CB1-31 generation works
sbatch run_ar.slurm
```

### Inference (combined pipeline)

```bash
# Full pipeline: AR + NAR
sbatch run_inference.slurm

# Or run directly:
python inference.py \
    --source_audio /path/to/source.mp3 \
    --target_dir /path/to/target_speaker_audio/ \
    --ar_ckpt checkpoints_ar/ar_large_best.pt \
    --nar_ckpt checkpoints_nar/nar_xlarge_best.pt \
    --data_processed /path/to/processed_merged \
    --output_dir output/
```

### Inference outputs

The inference script generates 5 wav files for comparison:

| File | Description |
|---|---|
| `source_reencoded.wav` | Source audio round-tripped through Mimi (baseline) |
| `faiss_baseline.wav` | FAISS nearest-neighbor retrieval (v5.x approach) |
| `ar_only_cb0.wav` | AR-converted CB0 + source CB1-31 (tests AR quality) |
| `nar_only_cb1_31.wav` | Source CB0 + NAR-generated CB1-31 (tests NAR quality) |
| `full_ar_nar.wav` | AR CB0 + NAR CB1-31 (full pipeline) |

## File Structure

```
v6.0-codec-lm/
  train_ar.py          # Stage 1: AR semantic model (Causal Transformer + FiLM)
  train_nar.py         # Stage 2: NAR acoustic model (Conformer + FiLM)
  inference.py         # Combined 2-stage inference with baselines
  run_ar.slurm         # SLURM job for AR training
  run_nar.slurm        # SLURM job for NAR training
  run_inference.slurm  # SLURM job for combined inference
  README.md            # This file
```

## Model Details

### AR Semantic Model (Stage 1)

- Architecture: Causal Transformer Decoder with FiLM speaker conditioning per layer
- Size (large): d_model=768, n_heads=12, n_layers=12 (~150M params)
- Vocabulary: 2048 CB0 tokens + 1 separator token
- Input format: `[source_CB0, <SEP>, target_CB0]` with causal masking
- Training: teacher forcing with cross-entropy loss, 50% same-speaker pairs
- Inference: autoregressive generation with temperature + top-k sampling

### NAR Acoustic Model (Stage 2)

- Architecture: Conformer (bidirectional, full attention) with FiLM speaker conditioning
- Size (xlarge): d_model=1024, n_heads=16, n_layers=14 (~400M params)
- Input: CB0 tokens + speaker embedding
- Output: 31 independent heads, one per codebook (CB1-CB31)
- Training: focal cross-entropy + SECS speaker similarity loss
- CB0 perturbation (30%) during training for robustness to AR errors
