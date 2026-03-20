#!/bin/bash
#SBATCH --job-name=mimi_vc_v3f
#SBATCH --output=logs/mimi_vc_%j.out
#SBATCH --error=logs/mimi_vc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# vRAM needed: ~40GB for A100 (model + data + batch_size=48)
REQUIRED_VRAM=40960

# =========================================================
# ENVIRONMENT
# =========================================================
module clear -f
module load shared python312

# Activate venv
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate

# Unset Slurm CUDA_VISIBLE_DEVICES to let gpu_check find best GPU
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
CKPT_DIR=$WORK_DIR/checkpoints_v3f
OUTPUT_DIR=$WORK_DIR/output_v3f
TARGET_DIR=$WORK_DIR/target_audio    # put target speaker .mp3/.wav/.flac files here
SOURCE_AUDIO=$WORK_DIR/source.mp3    # source audio file for conversion
SCRIPT=$WORK_DIR/train_v3f.py

mkdir -p $CKPT_DIR $OUTPUT_DIR logs

# =========================================================
# RUN TRAINING
# =========================================================
echo "Starting Mimi-VC V3f training..."
echo "Checkpoint dir: $CKPT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Target audio: $TARGET_DIR"
echo "Source audio: $SOURCE_AUDIO"

python -u $SCRIPT \
    --ckpt_dir $CKPT_DIR \
    --output_dir $OUTPUT_DIR \
    --target_dir $TARGET_DIR \
    --source_audio $SOURCE_AUDIO \
    --data_raw       $DATA_STORAGE/librispeech \
    --data_processed $DATA_STORAGE/processed \
    --phase all \
    --batch_size 16 \
    --ft_batch 8 \
    --pretrain_steps 30000 \
    --ft_steps 3000 \
    --focal_gamma 2.0 \
    --ft_focal_gamma 1.5 \
    --dropout 0.05 \
    --spk_noise_std 0.1 \
    --cb0_noise_rate 0.30 \
    --n_speakers 2338 \
    --utts_per_speaker 10 \
    --use_train360 \
    --use_train500 \
    --d_model 768 \
    --n_heads 12 \
    --n_enc_layers 8 \
    --n_dec_layers 8

echo "Job completed: $(date)"
