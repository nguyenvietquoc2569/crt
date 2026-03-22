# Project Conventions & Deployment Guide

## 1. Repository Layout

```
duplex/                               ← local dev machine (macOS)
├── codestyle.md                      ← this file
├── architecture.md                   ← high-level system architecture
├── docs/                             ← research docs, deep dives
├── report-3-21.docx                  ← experiment report (auto-generated)
│
├── v5/                               ← earliest CRT prototype
├── v5.3-CRT/                         ← baseline Conformer CRT
├── v5.3.1-CRT-parallel/              ← + PrefetchQueue + ThreadPoolExecutor
├── v5.3.1-CRT-parallel-layers-14-d_model-1024/  ← + scaled to d_model=1024
├── v5.3.2-CRT-parallel-layers-16-d_model-1280-curriculum-corruption/  ← + curriculum + larger model
├── v5.4.0-CRT-parallel-SECS-reg/    ← + speaker similarity loss
│
└── .venv/                            ← local Python venv (not deployed)
```

### Version naming convention

```
v{major}.{minor}.{patch}-{module}-{variant}
```

- **major**: architecture family (5 = CRT-based)
- **minor**: feature generation (3 = n_learn=18, 14-layer conformer; 4 = new loss function)
- **patch**: hyperparameter / infra variant
- **module**: `CRT`, `CRT-parallel`, etc.
- **variant**: key differentiator (e.g. `layers-16-d_model-1280-curriculum-corruption`)

Each version is a **self-contained directory** with its own `train_crt.py`, `run_crt.slurm`, and `README.md`. No cross-version imports. Copy the whole directory to create a new experiment.

---

## 2. Cluster Information

| Property | Value |
|---|---|
| Host | `eaic@slurm.uit.edu.vn` |
| Auth | SSH key (`ssh-copy-id eaic@slurm.uit.edu.vn`) |
| Scheduler | SLURM |
| GPU | NVIDIA A100-SXM4 80GB (85.1 GB reported) |
| CPUs per job | 8 (via `--cpus-per-task=8`) |
| Memory | 32 GB (via `--mem=32G`) |
| GPU allocation | `--gres=mps:a100:1` (MPS mode, shared GPU) |
| Time limit | 24h default, 48h for longer runs |
| Python | 3.12 (`module load shared python312`) |
| Virtualenv | `/datastore/eaic/nvquoc/mimi_vc_env` |

---

## 3. Cluster Directory Structure

```
/datastore/eaic/nvquoc/mimi_vc/          ← WORK_DIR on cluster
├── data_storage/                         ← SHARED across all versions
│   ├── processed/
│   │   ├── prepared_data.pt              ← 2338 speakers, [32, T] per speaker
│   │   └── speaker_embeddings.pkl        ← 256-dim resemblyzer embeddings
│   ├── hf_cache/                         ← HuggingFace model cache (Mimi)
│   └── torch_cache/
├── target_audio/                         ← SHARED: target speaker .wav/.flac
├── source.mp3                            ← SHARED: source audio for inference
│
├── v5.3-CRT/                             ← version directory
│   ├── train_crt.py
│   ├── run_crt.slurm
│   ├── README.md
│   ├── checkpoints/
│   │   ├── crt_xlarge.pt                 ← latest (resumes from here)
│   │   └── crt_xlarge_best.pt            ← best loss (used for inference)
│   ├── output/
│   │   ├── faiss_baseline.wav
│   │   └── crt_xlarge_refined.wav
│   └── logs/
│       ├── crt_<jobid>.out
│       └── crt_<jobid>.err
│
├── v5.3.1-CRT-parallel/
├── v5.3.1-CRT-parallel-layers-14-d_model-1024/
├── v5.3.2-CRT-parallel-layers-16-d_model-1280-curriculum-corruption/
└── v5.4.0-CRT-parallel-SECS-reg/
```

### Shared vs version-local

| Shared (never duplicate) | Version-local (per experiment) |
|---|---|
| `data_storage/processed/` | `train_crt.py` |
| `data_storage/hf_cache/` | `run_crt.slurm` |
| `target_audio/` | `README.md` |
| `source.mp3` | `checkpoints/`, `output/`, `logs/` |

---

## 4. Deployment Workflow

### 4.1 Push code to cluster

```bash
# Single file update
rsync -av \
  /Users/nguyenquoc/duplex/<VERSION>/train_crt.py \
  /Users/nguyenquoc/duplex/<VERSION>/run_crt.slurm \
  eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/<VERSION>/

# Full directory (new version)
rsync -av \
  /Users/nguyenquoc/duplex/<VERSION>/ \
  eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/<VERSION>/
```

### 4.2 Create directories and submit

```bash
ssh eaic@slurm.uit.edu.vn bash << 'EOF'
mkdir -p /datastore/eaic/nvquoc/mimi_vc/<VERSION>/{checkpoints,output,logs}
cd /datastore/eaic/nvquoc/mimi_vc/<VERSION>
sbatch run_crt.slurm
EOF
```

### 4.3 Monitor

```bash
# Check queue
ssh eaic@slurm.uit.edu.vn "squeue -u eaic"

# Live log (find jobid from squeue output)
ssh eaic@slurm.uit.edu.vn "tail -f /datastore/eaic/nvquoc/mimi_vc/<VERSION>/logs/crt_<JOBID>.out"
```

### 4.4 Cancel and restart

```bash
ssh eaic@slurm.uit.edu.vn bash << 'EOF'
# Cancel by job name pattern
squeue -u eaic --noheader -o "%i %j" | grep <PATTERN> | awk '{print $1}' | xargs -r scancel

# Delete checkpoint for fresh start
rm -f /datastore/eaic/nvquoc/mimi_vc/<VERSION>/checkpoints/*.pt

# Resubmit
cd /datastore/eaic/nvquoc/mimi_vc/<VERSION>
sbatch run_crt.slurm
EOF
```

### 4.5 Download results

```bash
# Output audio
scp eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/<VERSION>/output/*.wav ~/Downloads/

# Training logs
scp eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/<VERSION>/logs/crt_<JOBID>.out ~/Downloads/
```

### 4.6 Resume after SLURM timeout

Training auto-resumes from `checkpoints/crt_xlarge.pt`. Just resubmit:
```bash
ssh eaic@slurm.uit.edu.vn "cd /datastore/eaic/nvquoc/mimi_vc/<VERSION> && sbatch run_crt.slurm"
```

---

## 5. SLURM Script Convention

Every `run_crt.slurm` follows the same template:

```bash
#!/bin/bash
#SBATCH --job-name=crt_<SHORT_ID>       # e.g. crt_v531p, crt_v532, crt_v540
#SBATCH --output=<VERSION_DIR>/logs/crt_%j.out
#SBATCH --error=<VERSION_DIR>/logs/crt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00                 # 48:00:00 for 60k-step runs

REQUIRED_VRAM=20480

# --- Environment ---
module clear -f
module load shared python312
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate
unset CUDA_VISIBLE_DEVICES

# --- GPU Check ---
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi
BEST_GPU=$CHECK_OUT

# --- MPS Server ---
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# --- Paths ---
WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
VERSION_DIR=$WORK_DIR/<VERSION>
DATA_STORAGE=$WORK_DIR/data_storage
# ... (shared paths)

# --- Offline mode (no internet on compute nodes) ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Training ---
python -u $SCRIPT \
    --data_processed $DATA_STORAGE/processed \
    --ckpt_dir       $CKPT_DIR \
    --output_dir     $OUTPUT_DIR \
    --target_dir     $WORK_DIR/target_audio \
    --source_audio   $WORK_DIR/source.mp3 \
    --model_size     xlarge \
    ...
```

### Key rules

- Always `python -u` (unbuffered stdout for real-time log viewing)
- Always `unset CUDA_VISIBLE_DEVICES` before `gpu_check.sh`, then re-export
- Always set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`
- MPS pipe/log directories use `$SLURM_JOB_ID` to avoid collisions
- Log files: `crt_%j.out` and `crt_%j.err` where `%j` = SLURM job ID

---

## 6. Code Conventions

### File structure

Each version has exactly **one** main file: `train_crt.py`. It contains:

1. **Imports and constants** (top)
2. **MODEL_CONFIGS dict** (model hyperparameters)
3. **Model classes** (ConformerBlock, CRT)
4. **Data loading** (`load_data()`)
5. **Batch sampling** (`sample_batch()`, `PrefetchQueue`)
6. **Training loop** (`train()`)
7. **Inference** (`refine()`)
8. **Main** (`main()` with argparse)

### Style

- Follow existing style of the file, don't impose new patterns
- No unnecessary comments, docstrings, or error handling
- Delete unused code rather than commenting it out
- Keep functions focused on what they do
- Use `flush=True` on all `print()` calls (SLURM buffers stdout)

### GPU stats

```python
def _phys_gpu_idx():
    """Physical GPU index from CUDA_VISIBLE_DEVICES, not torch logical index."""
    vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if vis:
        return int(vis.split(',')[0])
    return torch.cuda.current_device()
```

Always use `_phys_gpu_idx()` for pynvml/nvidia-smi, never `torch.cuda.current_device()`.
Under CUDA MPS, the logical device (always 0) differs from the physical device.

### FAISS

- `faiss.omp_set_num_threads(1)` — must be called before building any index
- Each ThreadPoolExecutor worker gets its own `random.Random(seed)` for thread safety
- Do NOT use `faiss-gpu` with ThreadPoolExecutor — `StandardGpuResources` is not thread-safe (crashes with `StackDeviceMemory::Stack::returnAlloc` assertion)
- Keep indices on CPU, use 8 parallel workers with OMP threads=1

### torch.compile

**Do not use.** Removed in v5.3.1+. Reasons:
- `mode='reduce-overhead'` (CUDA graphs) conflicts with GradScaler dynamic loss scaling
- `mode='default'` (Triton kernels) has 15-30 min compilation warmup
- Both modes show misleading GPU utilisation in nvidia-smi
- PrefetchQueue + ThreadPoolExecutor already provides the main speedup
- The marginal 20-40% kernel fusion gain is not worth the complexity

### Checkpoint saving

```python
raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
state = {
    'step': step + 1,
    'model_state': raw_model.state_dict(),
    'opt_state': opt.state_dict(),
    'loss': avg_loss,
}
torch.save(state, ckpt_path)
```

- Always save `raw_model.state_dict()` (not `model.state_dict()` on compiled models)
- Save in background thread to avoid blocking GPU training
- When loading with architecture changes (e.g. new head), use `strict=False`

### Checkpoint loading with architecture changes

```python
missing, unexpected = raw_model.load_state_dict(ckpt['model_state'], strict=False)
if missing:
    print(f'  New keys (randomly init): {missing}', flush=True)
# Only restore optimizer if all keys matched (same architecture)
if not missing:
    opt.load_state_dict(ckpt['opt_state'])
```

---

## 7. Training Hyperparameters Reference

### What works

| Parameter | Proven value | Notes |
|---|---|---|
| lr | 1e-4 | 3e-4 causes plateau at loss ~7.0 |
| corrupt_ratio | 0.50 | 0.70 makes task too hard for initial learning |
| corrupt_ratio_hi | 0.20 | Scales with corrupt_ratio |
| faiss_frac | 0.90 | Matches inference distribution |
| focal_gamma | 1.0 | Mild focus on hard frames |
| batch_size | 128 | GPU memory allows it on A100 |
| seq_len | 128 | ~10.2s at 12.5Hz |
| dropout | 0.1 | Standard |
| weight_decay | 0.01 | AdamW default |
| warmup_steps | 500 | Linear warmup |
| grad_clip | 1.0 | Prevents gradient explosions |

### What fails

| Parameter | Bad value | Symptom |
|---|---|---|
| lr=3e-4 | Too high | Loss plateaus at ~7.01, oscillates |
| corrupt_ratio=0.70 | Too hard | Loss stuck, model can't learn from noisy context |
| torch.compile | Any mode | GradScaler conflict, long warmup, misleading GPU% |
| faiss-gpu + threads | StandardGpuResources | Assertion crash in StackDeviceMemory |

---

## 8. Experiment Tracking

### Log line format

```
[xlarge] STEP/TOTAL | Loss:X.XXXX | CB1:X% CB2:X% ... CB18:X% | X.Xstep/s | ETA Xs | GPU:X% X.X/85GB XW
```

v5.4.0 adds:
```
[xlarge] STEP/TOTAL | Loss:X.XXXX CE:X.XXXX SECS:X.XXXX | ...
```

v5.3.2 adds:
```
[xlarge] STEP/TOTAL | Loss:X.XXXX | ... | corrupt:X%
```

### Evaluation metrics (end of training)

```
>> SECS (Speaker Similarity)
     CRT-xlarge (ours) → target: 0.XXXX
     FAISS baseline    → target: 0.XXXX

>> MCD (Mel Cepstral Distortion, lower=closer to source)
     CRT-xlarge: XXX.XX
     FAISS baseline: XXX.XX
```

### Results so far

| Version | Params | Steps | Loss | CB1% | SECS | MCD |
|---|---|---|---|---|---|---|
| v5.3.1 (768) | 250.7M | 30k | 5.317 | 31.3% | 0.7565 | 132.79 |
| v5.3.1 (1024) | 419.5M | 30k | 4.761 | 40.5% | 0.7652 | 129.43 |
| FAISS baseline | - | - | - | - | 0.8792 | 106.53 |
| v5.3.2 (1280) | 706.4M | 60k | running | ~57% | - | - |
| v5.4.0 (1024+SECS) | 419.5M | 60k | running | - | - | - |
| v5.5 (residual) | 707.4M | 60k | **ABANDONED** | - | - | - |
| v5.6 (1280+SECS) | 707.4M | 60k | running | - | - | - |

### Training Data Statistics

```
Speakers:        2,338
Total frames:    3,616,157
Total duration:  289,292.6s (80.4h)

Per-speaker duration:
  Min:    47.8s
  Max:    150.4s
  Mean:   123.7s
  Median: 126.6s
  Std:    16.1s

Duration distribution:
    30-60s:     3 speakers (  0.1%)
   60-120s:   802 speakers ( 34.3%)
  120-300s: 1,533 speakers ( 65.6%)

Top 10 longest speakers:
  7939: 150.4s (1880 frames)
  7423: 149.8s (1873 frames)
  3054: 149.1s (1864 frames)
  489:  148.9s (1861 frames)
  4152: 148.9s (1861 frames)
  23:   148.9s (1861 frames)
  1870: 148.7s (1859 frames)
  3172: 148.5s (1856 frames)
  8718: 148.3s (1854 frames)
  46:   148.2s (1852 frames)

Bottom 10 shortest speakers:
  1708: 47.8s  (598 frames)
  844:  56.5s  (706 frames)
  1552: 58.3s  (729 frames)
  210:  61.5s  (769 frames)
  6753: 62.0s  (775 frames)
  6535: 62.1s  (776 frames)
  4757: 62.3s  (779 frames)
  149:  63.4s  (792 frames)
  2149: 64.9s  (811 frames)
  147:  65.9s  (824 frames)
```

Codec: Mimi (Kyutai), 32 codebooks, CB_SIZE=2048, 12.5 Hz frame rate, 24kHz audio.
CB0 = semantic (speaker-independent), CB1-31 = acoustic (speaker-specific).
FAISS index: per-speaker IndexFlatIP on CB0 embeddings (dim=256), cosine similarity.

---

## 9. Common Operations Cheat Sheet

```bash
# Check what's running
ssh eaic@slurm.uit.edu.vn "squeue -u eaic"

# Quick log peek (last 20 lines)
ssh eaic@slurm.uit.edu.vn "tail -20 /datastore/eaic/nvquoc/mimi_vc/<VERSION>/logs/crt_*.out"

# Cancel all my jobs
ssh eaic@slurm.uit.edu.vn "scancel -u eaic"

# Check GPU status on cluster
ssh eaic@slurm.uit.edu.vn "nvidia-smi"

# Check disk usage
ssh eaic@slurm.uit.edu.vn "du -sh /datastore/eaic/nvquoc/mimi_vc/*/checkpoints/"

# Copy checkpoint between versions (for warm-start)
ssh eaic@slurm.uit.edu.vn "cp /datastore/eaic/nvquoc/mimi_vc/<SRC>/checkpoints/crt_xlarge_best.pt \
                               /datastore/eaic/nvquoc/mimi_vc/<DST>/checkpoints/crt_xlarge.pt"
```

---

## 10. Creating a New Experiment Version

1. Copy the closest existing version:
   ```bash
   cp -r v5.3.1-CRT-parallel-layers-14-d_model-1024 v5.X.Y-CRT-<variant>
   ```

2. Update in the new directory:
   - `run_crt.slurm`: job name, VERSION_DIR path, hyperparameters
   - `train_crt.py`: model config, training logic changes
   - `README.md`: document what changed and why

3. Do NOT copy checkpoints, output, or logs (start fresh or warm-start explicitly).

4. Deploy:
   ```bash
   rsync -av /Users/nguyenquoc/duplex/v5.X.Y-CRT-<variant>/ \
     eaic@slurm.uit.edu.vn:/datastore/eaic/nvquoc/mimi_vc/v5.X.Y-CRT-<variant>/
   ```
