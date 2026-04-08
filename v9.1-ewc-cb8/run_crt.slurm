#!/bin/bash
#SBATCH --job-name=crt_v91
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/v9.1-ewc-cb8/logs/crt_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/v9.1-ewc-cb8/logs/crt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

REQUIRED_VRAM=75000

# =========================================================
# ENVIRONMENT
# =========================================================
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
echo "Job $SLURM_JOB_ID on GPU: $BEST_GPU"

# =========================================================
# MPS SERVER
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# PATHS
# =========================================================
WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
VERSION_DIR=$WORK_DIR/v9.1-ewc-cb8

# Shared across versions
DATA_STORAGE=$WORK_DIR/data_storage
TARGET_DIR=$WORK_DIR/target_audio
SOURCE_AUDIO=$WORK_DIR/source.mp3

# Version-local
SCRIPT=$VERSION_DIR/train_crt.py
CKPT_DIR=$VERSION_DIR/checkpoints
OUTPUT_DIR=$VERSION_DIR/output

export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HF_DATASETS_CACHE=$DATA_STORAGE/hf_cache/datasets
export TRANSFORMERS_CACHE=$DATA_STORAGE/hf_cache/hub
export HF_MODULES_CACHE=$DATA_STORAGE/hf_cache/modules
export TORCH_HOME=$DATA_STORAGE/torch_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p $CKPT_DIR $OUTPUT_DIR $VERSION_DIR/logs

# Warm-start from v9.0 checkpoint (has CB1-18, we keep CB1-8, reset step to 0)
V90_CKPT=$WORK_DIR/v9.0-ewc-progressive/checkpoints/crt_xlarge_best.pt
V90_CKPT_ALT=$WORK_DIR/v9.0-ewc-progressive/checkpoints/crt_xlarge.pt
if [ ! -f "$CKPT_DIR/crt_xlarge.pt" ]; then
    SRC_CKPT=""
    if [ -f "$V90_CKPT" ]; then SRC_CKPT=$V90_CKPT;
    elif [ -f "$V90_CKPT_ALT" ]; then SRC_CKPT=$V90_CKPT_ALT;
    fi
    if [ -n "$SRC_CKPT" ]; then
        echo "Warm-starting from v9.0 ($SRC_CKPT), reset step to 0..."
        python3 -c "
import torch
ckpt = torch.load('$SRC_CKPT', weights_only=False, map_location='cpu')
ckpt['step'] = 0
ckpt['loss'] = float('inf')
ckpt.pop('opt_state', None)
ckpt.pop('ewc_state', None)
ckpt.pop('phase_idx', None)
torch.save(ckpt, '$CKPT_DIR/crt_xlarge.pt')
print(f'  Saved warm-start from v9.0 (step=0, CB9-18 will be dropped by strict=False)')
"
    else
        echo "WARNING: No v9.0 checkpoint found, training from scratch"
    fi
fi

# =========================================================
# CRT v9.1 — Progressive CB + EWC, CB1-CB8 only
#
# Same as v9.0 but n_learn=8 instead of 18.
# Focuses model capacity on the most impactful codebooks (CB1-8).
# =========================================================

echo "================================================================"
echo " CRT v9.1 — Progressive CB + EWC (CB1-CB8)"
echo " $(date)"
echo "================================================================"

python -u $SCRIPT \
    --data_processed    $DATA_STORAGE/processed_merged_v3 \
    --ckpt_dir          $CKPT_DIR \
    --output_dir        $OUTPUT_DIR \
    --target_dir        $TARGET_DIR \
    --source_audio      $SOURCE_AUDIO \
    --model_size        xlarge \
    --n_learn           8 \
    --batch_size        128 \
    --seq_len           128 \
    --steps             240000 \
    --lr                1e-4 \
    --dropout           0.1 \
    --corrupt_ratio     0.30 \
    --corrupt_ratio_hi  0.15 \
    --faiss_frac        0.90 \
    --focal_gamma       1.0 \
    --secs_weight       0.1 \
    --rvq_dropout       0.15 \
    --progressive \
    --ewc_lambda        5000.0 \
    --ewc_samples       200 \
    --confidence_threshold 0.5 \
    --rerank_topk       8 \
    --phase             all

echo ""
echo "================================================================"
echo " Output files:"
ls -lh $OUTPUT_DIR/*.wav 2>/dev/null
echo "================================================================"
echo "Job completed: $(date)"
