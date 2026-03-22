#!/bin/bash
#SBATCH --job-name=preproc_libritts
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/scripts/logs/preproc_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/scripts/logs/preproc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

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
# Dataset must be pre-downloaded on login node via:
#   python scripts/download_libritts_r.py --hf_cache $HF_DATASETS_CACHE

mkdir -p $WORK_DIR/scripts/logs $DATA_STORAGE/libritts_r_processed

echo "================================================================"
echo " LibriTTS-R Preprocessing (train-clean-360)"
echo " $(date)"
echo "================================================================"

# Step 1: Encode LibriTTS-R (reads raw wav files, no internet needed)
python -u $WORK_DIR/scripts/preprocess_libritts_r.py \
    --data_dir   $DATA_STORAGE/libritts_r_raw/LibriTTS_R/train-clean-360 \
    --output_dir $DATA_STORAGE/libritts_r_processed \
    --max_sec_per_speaker 150 \
    --max_utts_per_speaker 20

echo ""
echo "================================================================"
echo " Preprocessing done. Now merging..."
echo "================================================================"

# Step 2: Merge with existing data
python -u $WORK_DIR/scripts/merge_datasets.py \
    --original $DATA_STORAGE/processed \
    --new_data $DATA_STORAGE/libritts_r_processed/libritts_r_data.pt \
    --new_emb  $DATA_STORAGE/libritts_r_processed/libritts_r_embeddings.pkl \
    --output   $DATA_STORAGE/processed_merged

echo ""
echo "================================================================"
echo " All done: $(date)"
echo " Merged data at: $DATA_STORAGE/processed_merged/"
echo "================================================================"
