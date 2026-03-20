#!/bin/bash
#SBATCH --job-name=mimi_prepare
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=mps:a100:1

set -e

# =========================================================
# PATHS
# =========================================================
WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
DATA_STORAGE=$WORK_DIR/data_storage
SCRIPT=$WORK_DIR/train_v3f.py

# Redirect ALL caches to datastore (avoid filling /home quota)
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HF_DATASETS_CACHE=$DATA_STORAGE/hf_cache/datasets
export TRANSFORMERS_CACHE=$DATA_STORAGE/hf_cache/hub
export HF_MODULES_CACHE=$DATA_STORAGE/hf_cache/modules
export TORCH_HOME=$DATA_STORAGE/torch_cache
export PIP_CACHE_DIR=$DATA_STORAGE/pip_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p $WORK_DIR/logs \
         $DATA_STORAGE/librispeech $DATA_STORAGE/processed \
         $DATA_STORAGE/hf_cache $DATA_STORAGE/pip_cache \
         $DATA_STORAGE/cache $DATA_STORAGE/torch_cache

# =========================================================
# MPS setup (required by UIT Slurm)
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$$
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

# =========================================================
# Activate venv
# =========================================================
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate

echo "=== Starting data preparation ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo ""

python -u $SCRIPT \
    --prepare_only \
    --data_raw       $DATA_STORAGE/librispeech \
    --data_processed $DATA_STORAGE/processed \
    --n_speakers     2338 \
    --utts_per_speaker 10 \
    --use_train360 \
    --use_train500

echo ""
echo "=== Data preparation done: $(date) ==="
echo "Files in data_processed:"
ls -lh $WORK_DIR/data_processed/

# Cleanup MPS
nvidia-cuda-mps-control quit 2>/dev/null || true
