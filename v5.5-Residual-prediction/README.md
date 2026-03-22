# v5.5 Residual Prediction — Fixed Gate Schedule + SECS Loss

## Problem

CRT v5.3.1 predicts full CB tokens from scratch. While this produces smoother
transitions, it **loses speaker identity** — SECS drops from 0.88 (FAISS baseline)
to 0.77 (CRT). The model overwrites FAISS's correct speaker information along
with its errors.

v5.5 first attempt used learnable gates with regularization (gate_reg), but the
CE gradient overwhelmed the penalty — gates jumped to 0.60 within 200 steps,
degenerating to v5.3.1 behavior. Increasing gate_reg to 0.5 didn't help.

## Solution (v2): Fixed Gate Schedule

Instead of a learnable gate, use a **deterministic schedule** that forces the
model to learn residual corrections first, then gradually allows full override:

```
gate_schedule(step, total_steps):
    Phase 1 (steps 0 — total/3):     gate = 0.10  (trust FAISS 90%)
    Phase 2 (steps total/3 — 2*total/3): gate ramps 0.10 → 1.00
    Phase 3 (steps 2*total/3 — end):  gate = 1.00  (full model control)
```

For 60k steps:
- Steps 0-20k: gate=0.10 — model learns corrections, FAISS dominates output
- Steps 20k-40k: gate ramps 0.10→1.00 — gradual handoff
- Steps 40k-60k: gate=1.00 — model has full control (but learned residuals first)

```
final_logits = (1 - gate) * base_logits + gate * delta_logits
```

No gate parameters. No gate regularization loss. No adversarial tug-of-war.

## Architecture

- **Base**: Conformer backbone (same as v5.3.1)
- **Model size**: d_model=1280, n_heads=16, n_layers=16 (~707M params)
- **Removed**: gate_heads (no learnable gate parameters)
- **Added**: spk_head for SECS loss (speaker identity preservation)

## Three-Component Training

1. **Focal CE loss**: predicting correct CB tokens
2. **SECS loss** (weight=0.1): cosine similarity to target speaker embedding
3. **Curriculum corruption**: 30% → 50% → 70% over three phases

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| alpha | 10.0 | Temperature for FAISS one-hot base logits |
| secs_weight | 0.1 | Speaker similarity loss weight |
| steps | 60,000 | Fresh start, larger model |
| lr | 1e-4 | Same as v5.3.1 |
| corrupt_ratio | 0.50 | Base corruption level (curriculum scales it) |

## Curriculum Corruption

| Phase | Steps | corrupt_ratio | Gate |
|-------|-------|---------------|------|
| 1 (easy) | 0-20k | 30% | 0.10 (trust FAISS) |
| 2 (medium) | 20k-40k | 50% | 0.10 → 1.00 (ramp) |
| 3 (hard) | 40k-60k | 70% | 1.00 (full override) |

## Log Format

```
[xlarge] 200/60000 | Loss:X.XX CE:X.XX SECS:0.XX | CB1:X% ... | gate:0.10 | 0.6step/s | ...
```

## Expected Behavior

1. Steps 0-20k: gate=0.10, model learns small corrections, SECS stays near 0.88
2. Steps 20k-40k: gate ramps up, model takes more control, loss drops faster
3. Steps 40k-60k: gate=1.00, full model, but residual-aware from early training

## Timing Instrumentation

Logs include per-step timing breakdown:
```
data:1ms fwd:450ms bwd:1100ms
```
- data: PrefetchQueue wait time (CPU FAISS bottleneck if high)
- fwd: forward pass + loss (GPU)
- bwd: backward + optimizer (GPU)

## Changes from v5.5 v1 (learnable gate)

1. Removed gate_heads — no learnable gate parameters
2. Fixed gate schedule: 0.10 → ramp → 1.00 over 3 phases
3. Removed gate_reg from loss (no gate penalty needed)
4. Checkpoint loading uses strict=False (handles architecture changes)
5. Log simplified: single `gate:X.XX` instead of G1-G18 per-CB values
