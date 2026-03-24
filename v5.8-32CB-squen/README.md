# v5.8 — Sequential 3-Stage Prediction over 31 Codebooks

## Problem

v5.7 predicts all 31 codebooks in parallel (non-autoregressive). But RVQ is
inherently sequential — CB5 is the residual error after CB1-CB4. Predicting
them independently breaks the residual chain, creating inconsistent outputs.

## Solution: 3-Stage Sequential Prediction

Instead of one forward pass, run the shared Conformer backbone 3 times:

```
Stage 1: CB0 + FAISS[CB1-31] → Conformer → predict CB1-CB6
                                             ↓ (argmax, replace FAISS tokens)
Stage 2: CB0 + [pred CB1-6, FAISS CB7-31] → Conformer → predict CB7-CB18
                                             ↓ (argmax, replace FAISS tokens)
Stage 3: CB0 + [pred CB1-18, FAISS CB19-31] → Conformer → predict CB19-CB31
```

Each stage sees progressively cleaner input: stage 2 conditions on stage 1's
refined predictions instead of noisy FAISS tokens, and stage 3 conditions on
both stage 1 and 2 predictions.

## Why 3 stages, not 31?

- 31 sequential passes would be 31x slower
- Codebooks within each group have similar information density
- The 3-stage split matches the RVQ information hierarchy:
  - CB1-6: core speaker/phonetic identity
  - CB7-18: acoustic detail (formants, energy)
  - CB19-31: fine texture (noise, breathiness)

## Changes from v5.7

| Parameter | v5.7 | v5.8 |
|-----------|------|------|
| Forward passes/step | 1 | **3** (one per stage) |
| batch_size | 64 | **32** (3x more compute per step) |
| Architecture | Parallel heads | **Sequential stages** |
| Backbone | Shared | **Shared** (same weights, 3 passes) |

## Training

During training, all 3 stages run every step. The loss is computed over all
31 codebooks jointly. Gradients flow through all 3 stages (though the argmax
between stages is detached via `torch.no_grad()`).

The curriculum and SECS loss are identical to v5.7.

## Expected Impact

- Higher accuracy on CB7-CB31 (conditioned on refined CB1-6 instead of noisy FAISS)
- More consistent cross-codebook predictions (respects RVQ residual structure)
- Smoother, less robotic decoded audio
- ~3x slower per step (3 backbone passes), offset by smaller batch
