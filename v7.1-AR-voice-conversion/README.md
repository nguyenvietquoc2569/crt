# v7.1 — Autoregressive Voice Conversion (VALL-E style)

## Why AR works where v7.0 (cross-attention NAR) failed

v7.0 used non-autoregressive cross-attention and suffered mode collapse — the model
predicted the same token for every frame because it couldn't learn frame-level alignment.

v7.1 uses **autoregressive generation with prefix conditioning**, which works because:
1. Each frame conditions on ALL previous frames (sequential coherence)
2. Voice prompt is simply prepended (no cross-attention alignment needed)
3. Teacher forcing provides strong gradient per position during training

## Architecture

```
Input: [target_prompt: T_ref frames] [SEP] [source CB0 + teacher-forced CB1-18: T_src]
         bidirectional attention          causal attention (+ attend to prefix)

Training: teacher forcing (all positions in parallel)
Inference: AR generation (one frame at a time, feed back predictions)
```

## CausalConformerBlock

Same as standard ConformerBlock but:
- Self-attention uses prefix-aware causal mask
- Depthwise conv uses left-only padding (no future leaking)

## Key Parameters

| Param | Value |
|-------|-------|
| d_model | 1024 |
| n_heads | 16 |
| n_layers | 14 |
| T_ref | 128 (10.2s prompt) |
| T_src | 64 (5.1s source) |
| batch_size | 32 |
| steps | 120,000 |
| temperature | 0.7 (inference sampling) |
| cross_speaker_ratio | 0.9 |
| Params | ~450M |

## Expected behavior

- Loss should descend faster than v7.0 (which plateaued at 6.2)
- By step 5k: loss < 6.0, CB1 > 15% (v7.0 was stuck at 7%)
- Inference should produce audible speech (not silence) from step 5k+
- AR generation produces one frame per forward pass (~640ms for 5s audio)
