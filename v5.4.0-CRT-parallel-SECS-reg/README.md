# CRT v5.4.0 — Speaker Similarity Regularisation (SECS-reg)

## Problem this version solves

v5.3.1 results showed SECS **regression** vs the raw FAISS baseline:

| Method | SECS→target | MCD |
|---|---|---|
| FAISS baseline | 0.8792 | 106.53 |
| CRT v5.3.1 (1024) | 0.7652 | 129.43 |

The CE loss alone has no explicit pressure to preserve speaker identity.
The model minimises token prediction error but can do so by predicting
smooth "generic" tokens that drift away from the target speaker's timbre.

## What changed vs v5.3.1

### Speaker Identity Head + SECS Loss (Option 3)

```
Conformer stack output x [B, T, d_model]
          │
     mean over T
          │
     x_pool [B, d_model]
          │
     Linear(d_model → d_model//2) → GELU → Linear(d_model//2 → 256)
          │
     L2 normalise
          │
     spk_pred [B, 256]
          │
     cosine_loss = mean(1 - dot(spk_pred, spk_emb_normalised))
```

Total loss:
```
L_total = L_focal_CE + secs_weight * L_cosine_speaker
```

Default `--secs_weight 0.1`. Tune up (0.3) if SECS still regresses,
tune down (0.05) if CE loss stops improving.

### Why this works

The speaker head forces the model's internal representation (mean-pooled
over time) to predict the target speaker embedding. Backprop pushes the
Conformer layers to produce output that is collectively speaker-consistent
across the whole sequence — not just locally coherent frame-to-frame.

### What to watch in logs

Every 200 steps the log now shows:
```
Loss:X.XXXX CE:X.XXXX SECS:X.XXXX
```
- **CE**: should decrease from ~7.2 to ~4.5 (same as v5.3.1)
- **SECS loss**: should decrease from ~0.5 (random init) toward ~0.1-0.2
  (0.0 = perfect speaker prediction, practically unachievable)

If SECS loss stays high (> 0.4) after 5k steps, increase `--secs_weight`.
If CE loss stalls (> 5.5) after 10k steps, decrease `--secs_weight`.

### 60,000 Training Steps
Same as v5.3.2 — loss was still declining at 30k. SLURM time limit 48h.

---

# CRT v5.3 — Codebook Refinement Transformer (base documentation)

Voice conversion post-processor built on Kyutai's Mimi neural codec. Takes the
output of FAISS-based frame retrieval and refines it to produce temporally
coherent, speaker-consistent audio.

---

## Background and Motivation

### Mimi Codec Structure

Mimi is a neural audio codec from Kyutai that compresses audio into 32 discrete
codebook streams (Residual Vector Quantization, RVQ). Each stream operates at
12.5 Hz (80ms per frame) on 24kHz audio:

| Codebook | Role | Notes |
|----------|------|-------|
| CB0 | Semantic (WavLM-derived) | Captures linguistic content, speaker-independent |
| CB1–6 | Coarse acoustic | Fundamental timbre, formants, voice quality |
| CB7–12 | Fine acoustic | Breathiness, resonance, subtle vocal texture |
| CB13–18 | Finer acoustic | Micro-detail, fine spectral shape |
| CB19–31 | Residual | Room acoustics, recording conditions, noise floor |

Key property: all 32 CBs from the **same frame** must come from the **same
speaker** for valid Mimi decoding. Mixing CB0 from speaker A with CB1–31 from
speaker B produces incoherent, distorted audio (out-of-distribution RVQ chain).

### FAISS Frame Retrieval

The VC pipeline uses FAISS to find the best matching target-speaker frame for
each source frame:

1. Encode source audio → source CB0 tokens (semantic)
2. Encode target speaker audio → target CB0 tokens + all CBs
3. Build FAISS index on target CB0 embeddings (cosine similarity)
4. For each source frame: retrieve top-8 most similar target frames
5. Apply continuity constraint: penalise large jumps in frame index
   `score = 0.5 * cosine_sim + 0.5 / (1 + |frame_delta| / frame_rate)`
6. Assemble: replace all source codes with target codes at selected frames

This gives the correct target speaker identity (all CBs from real target frames)
but introduces **frame-boundary discontinuities** — consecutive output frames may
jump to very different positions in the target audio, creating audible artefacts.

### Why CRT

FAISS retrieval is strong for speaker identity but weak on temporal continuity.
CRT's job is to smooth these discontinuities while preserving the speaker
identity already present in the FAISS codes.

The key design insight: **CRT is trained with the same cross-speaker FAISS
corruption it will see at inference time**, so there is no train-test mismatch.
CRT only has to learn one thing: smooth jagged FAISS frame sequences.

---

## Architecture

### Overview

```
CB0 (FAISS)   ──► [Embedding]  ──┐
CB1  (FAISS)  ──► [Embedding]  ──┤
CB2  (FAISS)  ──► [Embedding]  ──┤
...                               ├──► sum ──► FiLM(spk_emb) ──► Conformer x14 ──► [Head x18] ──► refined CB1-18
CB18 (FAISS)  ──► [Embedding]  ──┘

spk_emb ──► [Linear → GELU → Linear] ──► FiLM gamma/beta
```

All 19 inputs (CB0 anchor + 18 noisy CBs) are embedded independently and summed
per frame. Speaker conditioning is applied via FiLM before the Conformer stack.

### Conformer Block

Each of the 14 layers follows the standard Conformer structure:

```
x → FFN(½ step) → MHSA+RoPE → DepthwiseConv(GLU) → FFN(½ step) → LayerNorm
```

- **FFN** (feed-forward, half-step residual): expands 4× with SiLU, projects back
- **MHSA + RoPE**: multi-head self-attention with rotary position embeddings;
  captures long-range frame dependencies (co-articulation, prosody)
- **DepthwiseConv (GLU)**: pointwise expand → GLU gate → depthwise conv (kernel=31,
  covers 2.5s at 12.5Hz) → LayerNorm → pointwise project. This is the core
  smoothing operator — each layer sweeps a 31-frame window and enforces local
  continuity exactly as a vocal tract physically evolves
- **Half-step residuals**: each FFN contributes 0.5× to the residual path,
  following the original Conformer paper for training stability

### FiLM Speaker Conditioning

```python
x = x * (1 + gamma) + beta
```

gamma and beta are predicted from the speaker embedding. At initialisation the
linear projection weights are near zero, so the layer starts as near-identity
and gradually learns speaker-specific scaling. This is more stable than
cross-attention (which had single K-V pair pathologies in earlier versions).

### Output Heads

One linear head per CB: `Linear(d_model, 2048)`. Each head independently
predicts the full 2048-token distribution for its codebook at every frame.
Loss is focal cross-entropy averaged across all 18 CB heads:

```
loss = mean over CBs [ -mean over frames [ (1 - p_correct)^gamma * log(p_correct) ] ]
```

Focal gamma=1.0 applies mild upweighting to hard (incorrect) frames without
destabilising training.

### Model Sizes

| Size | d_model | n_heads | n_layers | conv_kernel | Params |
|------|---------|---------|----------|-------------|--------|
| small | 128 | 4 | 4 | 15 | ~2M |
| medium | 256 | 4 | 6 | 31 | ~8M |
| large | 512 | 8 | 8 | 31 | ~28M |
| **xlarge** | **768** | **12** | **14** | **31** | **~90M** |

v5.3 trains xlarge only. The extra 4 layers vs v5 (10→14) give more smoothing
passes, which directly improves temporal continuity.

---

## Training

### Data

Reuses the processed cache built by V4 training (`data_storage/processed/`):
- `prepared_data.pt` — per-speaker code arrays `[32, T]` for all training speakers
- `speaker_embeddings.pkl` — 256-dim resemblyzer embeddings per speaker

No re-encoding is needed if V4 has already been run.

### Corruption Strategy

For each training sample the model receives a **corrupted** version of CB1–18
and must predict the original. Two corruption modes:

**FAISS corruption** (90% of samples — `faiss_frac=0.9`):
- Pick a random *different* speaker from the training set
- Run FAISS retrieval: use the current sample's CB0 to query that speaker's index
- Replace CB1–18 with the retrieved cross-speaker codes
- This directly simulates what happens at inference (source querying target)

**Random span masking** (10% of samples):
- CB1–6: corrupt 70% of frames with random tokens (heavy — CB1–6 are hardest)
- CB7–18: corrupt 30% of frames with random tokens (lighter — FAISS is already good)
- Spans of 3–10 consecutive frames are replaced together, not individual frames

The 90/10 split is deliberate: inference always sees FAISS-style noise, so the
model must be primarily optimised for that distribution. Random masking provides
regularisation and prevents overfitting to FAISS-specific patterns.

### Why faiss_frac=0.9 Fixes the Train-Test Mismatch

Earlier versions (v5.1) used 50% FAISS / 50% random. At inference the model
received clean FAISS CB1–6 (already 60–80% correct target tokens) but had been
trained half the time on completely random noise. Result: CRT tried to "fix"
already-correct tokens, degrading SECS from 0.88 → 0.74.

With faiss_frac=0.9 the training input distribution closely matches inference.
CRT learns to smooth FAISS-style cross-speaker noise, which is exactly what it
receives at inference time.

### Training Schedule

- **Steps**: 30,000 (was 6,000 in v5 — heavily undertrained for 90M params)
- **Batch size**: 32 × 128 frames = 4,096 frames per step
- **Optimiser**: AdamW, weight_decay=0.01
- **LR**: 3e-4 with cosine decay, 500-step linear warmup
- **Gradient clipping**: 1.0
- **Mixed precision**: fp16 AMP on CUDA

Checkpoints saved every 2,000 steps; best checkpoint (lowest loss) is kept
separately and loaded for inference.

### Expected Training Behaviour

| Step range | Expected loss | Notes |
|------------|--------------|-------|
| 0–500 | 7.6 → ~4.0 | Warmup; fast initial descent |
| 500–5,000 | ~4.0 → ~2.5 | Main learning phase |
| 5,000–15,000 | ~2.5 → ~1.8 | Diminishing returns, smoothing |
| 15,000–30,000 | ~1.8 → ~1.5 | Fine convergence |

CB accuracy at convergence (approximate, cross-speaker FAISS input):
- CB1: 55–65% (hardest — most speaker-specific)
- CB6: 70–80%
- CB12: 80–88%
- CB18: 88–93% (easiest — fine residuals, less speaker variation)

---

## Inference Pipeline

### Full Pipeline

```
source.mp3
    │
    ▼
[Mimi encode] → src_codes [32, T_src]
    │
    │  src_CB0 (semantic tokens)
    ▼
[FAISS index on target_CB0]
    │
    │  top-8 candidates per frame + continuity scoring
    ▼
selected[T_src] → target frame indices
    │
    ├──► faiss_codes [32, T_src]  (FAISS baseline, saved as faiss_baseline.wav)
    │
    │  CB0:    target CB0 at selected frames
    │  CB1-18: target CB1-18 at selected frames  (cross-speaker noisy input)
    ▼
[CRT model]  +  target_spk_emb
    │
    │  refined CB1-18
    ▼
refined_codes:
    CB0:    target CB0 (unchanged from FAISS)
    CB1-18: CRT output
    CB19-31: target CB19-31 (unchanged from FAISS)
    │
    ▼
[Mimi decode] → refined audio (saved as crt_xlarge_refined.wav)
```

### Continuity Scoring

The FAISS retrieval uses a combined score to prefer temporally smooth sequences:

```python
score[t] = 0.5 * cosine_similarity[t] + 0.5 / (1 + abs(candidate - prev_frame) / frame_rate)
```

Dividing by `frame_rate` (12.5) normalises the frame-distance penalty correctly.
Without this normalisation the penalty is ~0.00002 (negligible) and retrieval
degenerates to pure nearest-neighbour with no continuity.

### Output Files

| File | Description |
|------|-------------|
| `output/faiss_baseline.wav` | Raw FAISS retrieval, no CRT (reference) |
| `output/crt_xlarge_refined.wav` | CRT-refined output |

### Evaluation Metrics

- **SECS** (Speaker Embedding Cosine Similarity): cosine distance between
  resemblyzer embeddings of output and target reference. Higher = more
  target-speaker-like. Good VC: > 0.80. The FAISS baseline should be ~0.88.
- **MCD** (Mel Cepstral Distortion): spectral distance between source and output
  MFCCs. Lower = closer to source content. Less important than SECS for VC
  quality but useful to detect over-smoothing.

---

## Directory Structure

```
mimi_vc/
├── data_storage/              ← SHARED across versions
│   ├── processed/
│   │   ├── prepared_data.pt
│   │   └── speaker_embeddings.pkl
│   ├── hf_cache/              ← HuggingFace model cache
│   └── torch_cache/
├── target_audio/              ← SHARED: target speaker .wav/.flac files
├── source.mp3                 ← SHARED: source audio for inference
│
└── v5.3-CRT/                  ← This version
    ├── README.md
    ├── train_crt.py           ← Training + inference script
    ├── run_crt.slurm          ← SLURM job submission
    ├── checkpoints/           ← Saved during training
    │   ├── crt_xlarge.pt      ← Latest checkpoint (resumes from here)
    │   └── crt_xlarge_best.pt ← Best loss checkpoint (used for inference)
    ├── output/                ← Inference results
    │   ├── faiss_baseline.wav
    │   └── crt_xlarge_refined.wav
    └── logs/
        ├── crt_<jobid>.out
        └── crt_<jobid>.err
```

---

## How to Run

### Prerequisites

The V4 training pipeline must have been run first to build:
- `data_storage/processed/prepared_data.pt`
- `data_storage/processed/speaker_embeddings.pkl`
- Mimi model cached at `data_storage/hf_cache/`

### Submit Training Job

```bash
# Sync to cluster
rsync -av /path/to/v5.3-CRT/ eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/v5.3-CRT/

# Create directories and submit
ssh eaic@slurm.uit.edu.vn "
    mkdir -p /datastore/eaic/nvquoc/mimi_vc/v5.3-CRT/{checkpoints,output,logs}
    cd /datastore/eaic/nvquoc/mimi_vc
    sbatch v5.3-CRT/run_crt.slurm
"
```

### Monitor

```bash
# Live log
ssh eaic@slurm.uit.edu.vn "tail -f /datastore/eaic/nvquoc/mimi_vc/v5.3-CRT/logs/crt_<jobid>.out"

# Check queue
ssh eaic@slurm.uit.edu.vn "squeue -u eaic"
```

### Resume Interrupted Training

Training auto-resumes from `checkpoints/crt_xlarge.pt` if it exists.
Just resubmit the same slurm job — no flags needed.

### Run Inference Only (on existing checkpoint)

```bash
python train_crt.py \
    --phase refine \
    --model_size xlarge \
    --source_audio /datastore/eaic/nvquoc/mimi_vc/source.mp3 \
    --target_dir   /datastore/eaic/nvquoc/mimi_vc/target_audio
```

### Clear Checkpoints (start from scratch)

```bash
rm -f /datastore/eaic/nvquoc/mimi_vc/v5.3-CRT/checkpoints/crt_xlarge*.pt
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_size` | medium | One of: small / medium / large / xlarge |
| `--n_learn` | 18 | Number of CBs to predict (CB1 through CB{n_learn}) |
| `--steps` | 0 | Training steps (0 = use size default: 30000 for xlarge) |
| `--faiss_frac` | 0.5 | Fraction of training samples using FAISS corruption (use 0.9 for inference-matched training) |
| `--corrupt_ratio` | 0.7 | Fraction of CB1–6 tokens corrupted per sample |
| `--corrupt_ratio_hi` | 0.3 | Fraction of CB7+ tokens corrupted per sample |
| `--focal_gamma` | 1.0 | Focal loss exponent (0 = standard cross-entropy) |
| `--batch_size` | 32 | Samples per batch |
| `--seq_len` | 128 | Frames per sample (~10s at 12.5Hz) |
| `--lr` | 3e-4 | Peak learning rate |
| `--phase` | train | train / refine / all |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| v5.0 | Initial CRT with 4 sizes, n_learn=6, V4 first-pass integration |
| v5.1 | Git checkpoint before pipeline fix. FAISS CB1-6 in inference (mismatch with random-trained model → SECS drop 0.88→0.74) |
| v5.2 | n_learn=6→12, faiss_frac=0.5→0.9, V4 removed from pipeline |
| **v5.3** | n_learn=12→18, n_layers=10→14, steps=6000→30000, xlarge only |

### Why Extending to CB1–18 Helps

Each additional codebook group adds another layer of acoustic detail from the
target speaker:
- CB1–6 already in v5.0: coarse timbre, formant structure
- CB7–12 added in v5.2: breathiness, resonance, fine vocal texture
- CB13–18 added in v5.3: micro-spectral detail, fine residual correction

Beyond CB18 the codebooks (CB19–31) mostly encode room acoustics and recording
conditions rather than voice identity, so extending further yields diminishing
returns and risks over-fitting to the target's recording environment.

### Why n_layers 10→14

The Conformer's depthwise conv (kernel=31) sweeps a 2.5s window. With 14 layers
the effective receptive field computation spans significantly more context, and
each additional layer acts as another smoothing pass over the codebook sequence.
For a temporal denoising task, depth is more valuable than width (d_model).

---

## Troubleshooting

**"V4 processed cache not found"**
Run V4 training at least through the data preprocessing step first.

**Checkpoint shape mismatch on resume**
If you changed `n_learn` or `n_layers`, delete the old checkpoint:
`rm checkpoints/crt_xlarge*.pt`

**SECS degraded vs FAISS baseline**
The train-test distribution mismatch. Make sure `--faiss_frac 0.90` is set.
With 0.5 (the arg default), CRT over-corrects already-good FAISS tokens.

**Loss stuck at ~7.6 from step 0**
Bad random initialisation. Cancel and resubmit — PyTorch uses a different seed
each launch. The model should descend rapidly below 4.0 within 500 steps.

**OOM on A100**
Reduce `--batch_size` from 32 to 16. The xlarge model at seq_len=128 uses
~18GB VRAM at batch_size=32.
