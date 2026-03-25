# v7.2 — FAISS-Bootstrapped Cross-Attention

## The Idea

Use FAISS as "training wheels" for cross-attention:
1. Phase 1: Full FAISS hints — model learns with explicit alignment guidance
2. Phase 2: Gradually drop FAISS — cross-attention must compensate
3. Phase 3: No FAISS — cross-attention works independently (bootstrapped)

## Why Previous Approaches Failed

- v5.6 CRT: FAISS + refinement works, but FAISS causes frame-boundary artifacts
- v7.0 Cross-attention: No FAISS, model can't learn alignment from scratch → mode collapse
- v7.1 AR: No FAISS, same plateau at loss 7.1

## v7.2 Solution

Combine FAISS input (from CRT) with cross-attention (from v7.0), then gradually
remove FAISS. The cross-attention learns alignment by watching FAISS, then operates
independently.

```
Phase 1 (0-40k):   faiss_drop=0%   → CRT-like, full FAISS hints
Phase 2 (40k-80k): faiss_drop=0→1  → gradual FAISS removal
Phase 3 (80k-120k): faiss_drop=100% → pure cross-attention
```

## Architecture

```
Source: CB0 + FAISS_CB1-18 (with dropout) → Embedding sum
Target: ref_codes [32, T_ref] → 32 CB embeddings → 4 ConformerBlocks → context

Source Encoder (x14 CrossAttentionConformerBlocks):
  Self-Attention → Cross-Attention(context) → DepthwiseConv → FFN

Output: 18 heads → CB1-18 logits
```

## Inference Variants

The model generates 5 outputs for comparison:
1. FAISS baseline (all 32 CB from FAISS)
2. Model + FAISS (faiss_drop=0, like CRT)
3. Model + FAISS sampled (T=0.8)
4. Model no-FAISS (faiss_drop=1, pure cross-attention)
5. Model no-FAISS sampled (T=0.8)

If #4/#5 approach #2/#3 quality, the bootstrapping worked.

## Key Parameters

| Param | Value |
|-------|-------|
| d_model | 1024 |
| n_heads | 16 |
| n_src_layers | 14 (cross-attention) |
| n_tgt_layers | 4 (self-attention) |
| T_ref | 128 (10s reference) |
| T_src | 128 (10s source) |
| batch_size | 32 |
| steps | 120,000 |
| faiss_frac | 0.90 |
| secs_weight | 0.1 |
