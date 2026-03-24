#!/bin/bash
#SBATCH --job-name=crt_v58
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/v5.8-32CB-squen/logs/crt_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/v5.8-32CB-squen/logs/crt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

REQUIRED_VRAM=20480

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
VERSION_DIR=$WORK_DIR/v5.8-32CB-squen

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

# =========================================================
# CRT v5.8 — Sequential 3-stage prediction over 31 codebooks
#
# Stage 1: CB1-CB6   (core acoustic, from FAISS noisy input)
# Stage 2: CB7-CB18  (conditioned on stage 1 predictions)
# Stage 3: CB19-CB31 (conditioned on stage 1+2 predictions)
#
# Each stage re-runs the shared Conformer backbone with
# progressively cleaner input tokens.
# =========================================================

echo "================================================================"
echo " CRT v5.8 — 3-stage sequential + 31 CB + curriculum + SECS"
echo " $(date)"
echo "================================================================"

python -u $SCRIPT \
    --data_processed    $DATA_STORAGE/processed_merged \
    --ckpt_dir          $CKPT_DIR \
    --output_dir        $OUTPUT_DIR \
    --target_dir        $TARGET_DIR \
    --source_audio      $SOURCE_AUDIO \
    --model_size        xlarge \
    --n_learn           31 \
    --batch_size        32 \
    --seq_len           128 \
    --steps             120000 \
    --lr                1e-4 \
    --dropout           0.1 \
    --corrupt_ratio     0.30 \
    --corrupt_ratio_hi  0.15 \
    --faiss_frac        0.90 \
    --focal_gamma       1.0 \
    --secs_weight       0.1 \
    --curriculum \
    --phase             all

echo ""
echo "================================================================"
echo " Output files:"
ls -lh $OUTPUT_DIR/*.wav 2>/dev/null
echo "================================================================"
echo "Job completed: $(date)"
