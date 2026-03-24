# v5.6.1 — CRT CB1-18 + FAISS Passthrough CB19-31

## What changed from v5.6

Same training as v5.6 (curriculum corruption + SECS loss, d_model=1280, n_layers=16).

Two changes:
1. **Inference every 5000 steps** (was 10000) for finer progress tracking
2. **Explicit labeling** of the FAISS passthrough strategy in logs

## Inference Strategy

At inference time, all 32 codebooks are used:
- **CB0**: FAISS-retrieved target speaker CB0 (anchor, not predicted)
- **CB1-CB18**: CRT-refined predictions (the model's output)
- **CB19-CB31**: FAISS-retrieved target speaker tokens (passthrough, not predicted)

## Why not predict CB19-31

v5.7 (31 CB parallel) and v5.8 (31 CB sequential) both failed completely:
- Loss stuck at 7.15 (near random) for 10k+ steps
- CB1 accuracy: 2% (vs 34% for 18-CB v5.3.2 at same step)
- The higher codebooks encode micro-residual detail unpredictable from context

FAISS-retrieved CB19-31 tokens are noisy but carry real acoustic information.
The Mimi decoder uses them to reconstruct fine texture (breathiness, room tone).

## Key Parameters

| Parameter | Value |
|-----------|-------|
| n_learn | 18 (predict CB1-18 only) |
| d_model | 1280 |
| n_layers | 16 |
| steps | 120,000 |
| batch_size | 128 |
| infer_every | 5,000 |
| curriculum | 30% -> 50% -> 70% |
| secs_weight | 0.1 |
| dataset | processed_merged (3242 speakers, ~110h) |
