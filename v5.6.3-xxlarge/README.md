# v5.6.3 — XXLarge CRT (d_model=1536, n_layers=18, ~1.1B params)

## What This Is

Same architecture as v5.6.2 (CRT + curriculum + SECS + confidence re-ranker)
but with a significantly larger model:

| Config | v5.6.2 xlarge | v5.6.3 xxlarge |
|--------|--------------|----------------|
| d_model | 1280 | **1536** |
| n_heads | 16 | **16** (head_dim=96) |
| n_layers | 16 | **18** |
| Params | ~706M | **~1.1B** |
| batch_size | 128 | **64** (larger model) |

## Why Bigger

v5.3.1 (419M) -> v5.3.2 (706M) gave a significant improvement:
- Loss: 4.76 -> 3.20 at 30k steps
- CB1: 40% -> 54%

More capacity helps the Conformer backbone encode richer representations
for the 18 output heads. The A100 85GB has plenty of headroom.

## Inference

Same as v5.6.2: confidence-based re-ranker at inference time.
- confidence_threshold=0.5: only override FAISS when CRT is confident
- rerank_topk=8: constrain to FAISS top-8 candidates

## Key Parameters

| Param | Value |
|-------|-------|
| d_model | 1536 |
| n_heads | 16 |
| n_layers | 18 |
| Params | ~1.1B |
| batch_size | 64 |
| steps | 120,000 |
| curriculum | 30% -> 50% -> 70% |
| secs_weight | 0.1 |
| data | processed_merged (3238 speakers) |
