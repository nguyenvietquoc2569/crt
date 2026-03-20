#!/bin/bash
#SBATCH --job-name=mimi_infer_v3g
#SBATCH --output=logs/mimi_infer_v3g_%j.out
#SBATCH --error=logs/mimi_infer_v3g_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Inference only — no training, just loads checkpoint and runs VC
REQUIRED_VRAM=2048

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
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi
BEST_GPU=$CHECK_OUT
echo "Job $SLURM_JOB_ID on GPU: $BEST_GPU"

# =========================================================
# PRIVATE MPS SERVER
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# PATHS + HF OFFLINE MODE
# =========================================================
WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
DATA_STORAGE=$WORK_DIR/data_storage

export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HF_DATASETS_CACHE=$DATA_STORAGE/hf_cache/datasets
export TRANSFORMERS_CACHE=$DATA_STORAGE/hf_cache/hub
export HF_MODULES_CACHE=$DATA_STORAGE/hf_cache/modules
export TORCH_HOME=$DATA_STORAGE/torch_cache
export PIP_CACHE_DIR=$DATA_STORAGE/pip_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CKPT_DIR=$WORK_DIR/checkpoints_v3g
OUTPUT_DIR=$WORK_DIR/output_v3g
TARGET_DIR=$WORK_DIR/target_audio
SOURCE_AUDIO=$WORK_DIR/source.mp3
SCRIPT=$WORK_DIR/train_v3g.py

mkdir -p $OUTPUT_DIR logs

echo "=== Mimi-VC V3g Inference ==="
echo "Checkpoint: $CKPT_DIR"
echo "Source:     $SOURCE_AUDIO"
echo "Target dir: $TARGET_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Fix: fp16->fp32 cast before sf.write"

python -u $SCRIPT \
    --phase infer \
    --ckpt_dir $CKPT_DIR \
    --output_dir $OUTPUT_DIR \
    --target_dir $TARGET_DIR \
    --source_audio $SOURCE_AUDIO \
    --data_raw       $DATA_STORAGE/librispeech \
    --data_processed $DATA_STORAGE/processed \
    --d_model 768 \
    --n_heads 12 \
    --n_enc_layers 8 \
    --n_dec_layers 8

echo "=== Inference done: $(date) ==="
echo "Output files:"
ls -lh $OUTPUT_DIR/*.wav 2>/dev/null || echo "No wav files found"
