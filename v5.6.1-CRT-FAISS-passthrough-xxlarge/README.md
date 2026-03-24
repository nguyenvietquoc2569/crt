# v5.6.1-xxlarge — Larger Model (d=1536, L=20, ~1.1B params)

## What changed from v5.6.1

Same approach as v5.6.1 (curriculum + SECS + FAISS passthrough CB19-31), but with a larger model:

| | v5.6.1 (xlarge) | v5.6.1-xxlarge |
|---|---|---|
| d_model | 1280 | **1536** |
| n_heads | 16 | 16 (head_dim=96) |
| n_layers | 16 | **20** |
| Params | ~706M | **~1.1B** |
| batch_size | 128 | **64** (fits in 85GB) |
| VRAM est. | ~13GB | ~25GB |

## Why larger

v5.3.2 (706M) plateaued with CB1~57%, CB18~29%. More parameters give the Conformer backbone more capacity to model the complex mapping from noisy FAISS tokens to clean targets across 18 codebooks. The 20-layer depth also gives more refinement passes.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| model_size | xxlarge |
| d_model | 1536 |
| n_heads | 16 |
| n_layers | 20 |
| n_learn | 18 (predict CB1-18 only) |
| steps | 120,000 |
| batch_size | 64 |
| infer_every | 5,000 |
| curriculum | 30% -> 50% -> 70% |
| secs_weight | 0.1 |
| dataset | processed_merged (3242 speakers, ~110h) |
