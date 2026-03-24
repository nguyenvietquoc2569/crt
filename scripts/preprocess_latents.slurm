#!/bin/bash
#SBATCH --job-name=preproc_latents
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/scripts/logs/preproc_latents_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/scripts/logs/preproc_latents_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# =========================================================
# ENVIRONMENT
# =========================================================
module clear -f
module load shared python312
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate
unset CUDA_VISIBLE_DEVICES

# --- GPU Check ---
REQUIRED_VRAM=10240
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi
BEST_GPU=$CHECK_OUT
echo "Job $SLURM_JOB_ID on GPU: $BEST_GPU"

# --- MPS Setup ---
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

mkdir -p $WORK_DIR/scripts/logs

# =========================================================
# RUN
# =========================================================
echo "================================================================"
echo " Preprocess latents: extract quantizer.decode() embeddings"
echo " $(date)"
echo "================================================================"

python -u $WORK_DIR/scripts/preprocess_latents.py \
    --input_dir  $DATA_STORAGE/processed_merged \
    --output_dir $DATA_STORAGE/processed_merged_latents \
    --batch_size 64

echo ""
echo "================================================================"
echo "Job completed: $(date)"
