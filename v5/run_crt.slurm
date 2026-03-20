#!/bin/bash
#SBATCH --job-name=mimi_crt_v5
#SBATCH --output=logs/crt_%j.out
#SBATCH --error=logs/crt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

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
DATA_STORAGE=$WORK_DIR/data_storage

export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HF_DATASETS_CACHE=$DATA_STORAGE/hf_cache/datasets
export TRANSFORMERS_CACHE=$DATA_STORAGE/hf_cache/hub
export HF_MODULES_CACHE=$DATA_STORAGE/hf_cache/modules
export TORCH_HOME=$DATA_STORAGE/torch_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CKPT_DIR=$WORK_DIR/checkpoints_v5
OUTPUT_DIR=$WORK_DIR/output_v5
TARGET_DIR=$WORK_DIR/target_audio
SOURCE_AUDIO=$WORK_DIR/source.mp3
SCRIPT=$WORK_DIR/v5/train_crt.py

mkdir -p $CKPT_DIR $OUTPUT_DIR logs

# =========================================================
# CRT v5 — train all 4 sizes, compare CB accuracy
# small (~2M) → medium (~8M) → large (~28M) → xlarge (~70M)
#
# Each size trains to its default step count, saves best ckpt,
# then runs refinement inference on source.mp3 → target_audio/.
# Results land in output_v5/crt_{size}_refined.wav
# =========================================================

COMMON="
    --data_processed $DATA_STORAGE/processed
    --ckpt_dir       $CKPT_DIR
    --output_dir     $OUTPUT_DIR
    --target_dir     $TARGET_DIR
    --source_audio   $SOURCE_AUDIO
    --batch_size     32
    --seq_len        128
    --lr             3e-4
    --dropout        0.1
    --corrupt_ratio  0.70
    --faiss_frac     0.50
    --focal_gamma    1.0
    --phase          all
"

echo "================================================================"
echo " CRT v5 — Codebook Refinement Transformer"
echo " Training 4 sizes: small / medium / large / xlarge"
echo "================================================================"

for SIZE in small medium large xlarge; do
    echo ""
    echo "-------- SIZE: $SIZE --------"
    python -u $SCRIPT $COMMON --model_size $SIZE
    echo "-------- DONE: $SIZE | $(date) --------"
done

echo ""
echo "================================================================"
echo " All sizes complete. Output files:"
ls -lh $OUTPUT_DIR/crt_*_refined.wav 2>/dev/null
echo "================================================================"
echo "Job completed: $(date)"
