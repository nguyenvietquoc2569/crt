# v8.0 — Progressive CB Training + RVQ Dropout + Zero-Out Inference

## Inspired by Mimi Codec's Training

Mimi was trained with **RVQ dropout** — randomly truncating to fewer codebook
levels during training. This means Mimi's decoder gracefully handles missing
codebooks. We exploit this in three ways:

## Three Innovations

### 1. Progressive CB Training
Focus model capacity on most important codebooks first, then expand:
```
Steps 0-40k:     CB1 only        → focused, fast learning
Steps 40k-80k:   CB1-3 → CB1-6   → expand to important CBs
Steps 80k-120k:  CB1-12 → CB1-18  → all CBs active
Steps 120k-240k: CB1-18 + hard corruption → final refinement
```

### 2. RVQ Dropout During Training (15%)
Randomly zero-out entire CB predictions, matching Mimi's training:
- Model learns that some CBs can be missing
- Regularization prevents overfitting to specific CB combinations
- Decoder already handles this (trained with RVQ dropout)

### 3. Zero-Out Low-Confidence at Inference
Instead of FAISS fallback for uncertain predictions, use token 0:
- Mimi decoder was trained to handle missing CBs
- Zero token = "no information" (decoder fills from context)
- May sound better than a wrong CRT prediction

## Key Parameters

| Param | Value |
|-------|-------|
| d_model | 1280 |
| n_layers | 16 |
| params | 707M |
| steps | 240,000 |
| rvq_dropout | 15% |
| progressive | 6 phases |
| data | processed_merged_v3 (4655 spk, 202h) |
