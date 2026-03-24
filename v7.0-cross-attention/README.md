# v7.0 — Cross-Attention Voice Conversion

## What This Is

Replaces FAISS nearest-neighbor retrieval with **learned cross-attention** over
target speaker tokens. Each source frame attends to ALL target frames via soft
attention — no hard frame selection, no frame-boundary discontinuities.

## Architecture

```
Source: src_cb0 [B, T_src] -> Embedding -> FiLM(spk) -> x_src

Target: ref_codes [B, 32, T_ref] -> sum(32 CB embeddings) / sqrt(32)
        -> FiLM(spk) -> 4 ConformerBlocks -> x_tgt (context)

Source Encoder (x14 CrossAttentionConformerBlocks):
  x = FFN1(x)
  x = SelfAttention(x)              <- source context (with RoPE)
  x = CrossAttention(Q=x, KV=x_tgt) <- replaces FAISS (no RoPE)
  x = DepthwiseConv(x)              <- local smoothing
  x = FFN2(x)

Output: 18 heads -> CB1-18 logits [B, 18, T_src, 2048]
```

## Why This Should Work Better Than FAISS

| Issue | FAISS (v5.x) | Cross-Attention (v7.0) |
|-------|--------------|----------------------|
| Frame selection | Hard argmax (1 frame) | Soft attention (all frames) |
| Similarity metric | Fixed cosine on CB0 | Learned multi-head attention |
| Temporal coherence | Simple continuity penalty | Full sequence context |
| Training | Not differentiable | End-to-end trainable |
| Smoothness | Frame boundaries visible | Soft blending, no boundaries |

## Training Strategy

No FAISS, no corruption. For each sample:
1. Pick target speaker, sample reference (20.5s) and target (10.2s) segments
2. Source CB0: 90% from different speaker (voice conversion), 10% same speaker (reconstruction)
3. Model learns: attend to reference for speaker characteristics, use CB0 for content

## Key Parameters

| Param | Value |
|-------|-------|
| d_model | 1024 |
| n_heads | 16 |
| n_src_layers | 14 (with cross-attention) |
| n_tgt_layers | 4 (self-attention only) |
| T_ref | 256 frames (20.5s reference) |
| T_src | 128 frames (10.2s source) |
| batch_size | 32 |
| steps | 120,000 |
| n_cb | 18 |
| cross_speaker_ratio | 0.9 |
| Params | ~450M |

## Deploy

```bash
rsync -av v7.0-cross-attention/ eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/v7.0-cross-attention/
ssh eaic@slurm.uit.edu.vn "mkdir -p /datastore/eaic/nvquoc/mimi_vc/v7.0-cross-attention/{checkpoints,output,logs} && cd /datastore/eaic/nvquoc/mimi_vc/v7.0-cross-attention && sbatch run_vc.slurm"
```
