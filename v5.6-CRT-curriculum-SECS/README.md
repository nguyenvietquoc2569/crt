# v5.6 CRT — Curriculum Corruption + SECS Speaker Similarity Loss

## What This Is

Combines the two best approaches:
- **v5.3.2**: d_model=1280, n_layers=16, 706M params, curriculum corruption (loss 2.76, CB1 57%)
- **v5.4.0**: SECS loss (speaker identity preservation via cosine similarity head)

## Why

v5.3.2 achieves excellent codebook accuracy (57% CB1, 29% CB18) but previous
versions showed SECS regression — the model produces smooth audio but drifts
from target speaker identity. Adding SECS loss directly penalizes speaker drift.

## Architecture

Same as v5.3.2 + speaker identity head:
- d_model=1280, n_heads=16, n_layers=16, conv_kernel=31 (~707M params)
- Conformer backbone + FiLM speaker conditioning
- New: `spk_head` — mean-pool CRT output -> Linear(1280, 640) -> GELU -> Linear(640, 256)

## Training Loss

```
L_total = L_focal_CE + 0.1 * L_SECS
```

- **L_focal_CE**: focal cross-entropy on CB1-18 predictions
- **L_SECS**: 1 - cosine_sim(spk_head(mean_pool(x)), target_spk_emb)

## Curriculum Corruption

| Phase | Steps | corrupt_ratio | corrupt_ratio_hi |
|-------|-------|---------------|------------------|
| 1 (easy) | 0-20k | 30% | 15% |
| 2 (medium) | 20k-40k | 50% | 20% |
| 3 (hard) | 40k-60k | 70% | 28% |

## Warm Start from v5.3.2

Loads checkpoint with `strict=False`:
- All conformer/head weights: loaded from v5.3.2
- `spk_head` weights: randomly initialized
- Optimizer: reset (architecture changed)

Copy v5.3.2 checkpoint before submitting:
```bash
cp /datastore/eaic/nvquoc/mimi_vc/v5.3.2-.../checkpoints/crt_xlarge*.pt \
   /datastore/eaic/nvquoc/mimi_vc/v5.6-CRT-curriculum-SECS/checkpoints/
```

## Log Format

```
[xlarge] 200/60000 | Loss:X.XX CE:X.XX SECS:0.XXX | CB1:X% ... | 0.7step/s | corrupt:30%
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 1280 |
| n_layers | 16 |
| secs_weight | 0.1 |
| lr | 1e-4 |
| steps | 60,000 |
| batch_size | 128 |
| focal_gamma | 1.0 |

## Deploy

```bash
rsync -av /Users/nguyenquoc/duplex/v5.6-CRT-curriculum-SECS/ \
  eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/v5.6-CRT-curriculum-SECS/

ssh eaic@slurm.uit.edu.vn bash << 'EOF'
mkdir -p /datastore/eaic/nvquoc/mimi_vc/v5.6-CRT-curriculum-SECS/{checkpoints,output,logs}
# Copy v5.3.2 checkpoint for warm start
cp /datastore/eaic/nvquoc/mimi_vc/v5.3.2-CRT-parallel-layers-16-d_model-1280-curriculum-corruption/checkpoints/crt_xlarge*.pt \
   /datastore/eaic/nvquoc/mimi_vc/v5.6-CRT-curriculum-SECS/checkpoints/
cd /datastore/eaic/nvquoc/mimi_vc/v5.6-CRT-curriculum-SECS
sbatch run_crt.slurm
EOF
```
