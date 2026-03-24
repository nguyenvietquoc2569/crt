#!/bin/bash
#SBATCH --job-name=v60_nar
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/v6.0-codec-lm/logs/nar_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/v6.0-codec-lm/logs/nar_%j.err
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
VERSION_DIR=$WORK_DIR/v6.0-codec-lm

# Shared across versions
DATA_STORAGE=$WORK_DIR/data_storage
TARGET_DIR=$WORK_DIR/target_audio
SOURCE_AUDIO=$WORK_DIR/source.mp3

# Version-local
SCRIPT=$VERSION_DIR/train_nar.py
CKPT_DIR=$VERSION_DIR/checkpoints_nar
OUTPUT_DIR=$VERSION_DIR/output_nar

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
# v6.0 NAR — Stage 2: Non-Autoregressive Acoustic Model
#
# Given clean CB0 tokens + speaker embedding, generate all
# 18 acoustic codebooks (CB1-CB18) in parallel. (31 CBs failed — see v5.7/v5.8)
# =========================================================

echo "================================================================"
echo " v6.0 NAR — Stage 2 Acoustic Model (Conformer + FiLM)"
echo " $(date)"
echo "================================================================"

python -u $SCRIPT \
    --data_processed    $DATA_STORAGE/processed_merged \
    --ckpt_dir          $CKPT_DIR \
    --output_dir        $OUTPUT_DIR \
    --target_dir        $TARGET_DIR \
    --source_audio      $SOURCE_AUDIO \
    --model_size        xlarge \
    --n_cb              18 \
    --batch_size        128 \
    --seq_len           128 \
    --steps             120000 \
    --lr                1e-4 \
    --dropout           0.1 \
    --focal_gamma       1.0 \
    --secs_weight       0.1 \
    --cb0_perturb       0.3 \
    --phase             all

echo ""
echo "================================================================"
echo " Output files:"
ls -lh $OUTPUT_DIR/*.wav 2>/dev/null
echo "================================================================"
echo "Job completed: $(date)"
