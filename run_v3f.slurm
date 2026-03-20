#!/bin/bash
#SBATCH --job-name=mimi_vc_v3f
#SBATCH --output=logs/mimi_vc_%j.out
#SBATCH --error=logs/mimi_vc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:4
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
# PATHS
# =========================================================
WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
CKPT_DIR=$WORK_DIR/checkpoints_v3f
OUTPUT_DIR=$WORK_DIR/output_v3f
TARGET_DIR=$WORK_DIR/target_audio    # put target speaker .wav files here
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

python $SCRIPT \
    --ckpt_dir $CKPT_DIR \
    --output_dir $OUTPUT_DIR \
    --target_dir $TARGET_DIR \
    --source_audio $SOURCE_AUDIO \
    --phase all \
    --batch_size 48 \
    --ft_batch 32 \
    --pretrain_steps 30000 \
    --ft_steps 3000 \
    --focal_gamma 2.0 \
    --ft_focal_gamma 1.5

echo "Job completed: $(date)"
