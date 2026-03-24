#!/bin/bash
#SBATCH --job-name=vc_v70
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/v7.0-cross-attention/logs/vc_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/v7.0-cross-attention/logs/vc_%j.err
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
VERSION_DIR=$WORK_DIR/v7.0-cross-attention

DATA_STORAGE=$WORK_DIR/data_storage
TARGET_DIR=$WORK_DIR/target_audio
SOURCE_AUDIO=$WORK_DIR/source.mp3

SCRIPT=$VERSION_DIR/train_vc.py
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
# v7.0 — Cross-Attention Voice Conversion
#
# Replaces FAISS retrieval with learned cross-attention.
# Source CB0 attends to target speaker's full codebook
# tokens via soft attention — no hard frame selection.
# =========================================================

echo "================================================================"
echo " v7.0 Cross-Attention VC (d_model=1024, n_layers=14)"
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
    --batch_size        32 \
    --t_ref             256 \
    --t_src             128 \
    --steps             120000 \
    --lr                1e-4 \
    --dropout           0.1 \
    --n_tgt_layers      4 \
    --cross_speaker_ratio 0.9 \
    --focal_gamma       1.0 \
    --secs_weight       0.1 \
    --phase             all

echo ""
echo "================================================================"
echo " Output files:"
ls -lh $OUTPUT_DIR/*.wav 2>/dev/null
echo "================================================================"
echo "Job completed: $(date)"
