#!/bin/bash
#SBATCH --job-name=preproc_all
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/scripts/logs/preproc_all_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/scripts/logs/preproc_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00

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

WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
DATA_STORAGE=$WORK_DIR/data_storage
RAW_DIR=$DATA_STORAGE/dataset_raw

export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HF_DATASETS_CACHE=$DATA_STORAGE/hf_cache/datasets
export TRANSFORMERS_CACHE=$DATA_STORAGE/hf_cache/hub
export HF_MODULES_CACHE=$DATA_STORAGE/hf_cache/modules
export TORCH_HOME=$DATA_STORAGE/torch_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p $DATA_STORAGE/encoded_datasets $WORK_DIR/scripts/logs

echo "================================================================"
echo " Preprocessing all datasets"
echo " $(date)"
echo "================================================================"

# =========================================================
# Step 1: Encode LibriTTS-R train-clean-100
# =========================================================
echo ""
echo "--- LibriTTS-R train-clean-100 ---"
LIBRITTS100_DIR=$RAW_DIR/LibriTTS_R/train-clean-100
if [ -d "$LIBRITTS100_DIR" ]; then
    python -u $WORK_DIR/scripts/preprocess_dataset.py \
        --data_dir   $LIBRITTS100_DIR \
        --output_dir $DATA_STORAGE/encoded_datasets \
        --prefix     libritts100 \
        --format     libritts \
        --max_sec_per_speaker 150 \
        --max_utts_per_speaker 20
else
    echo "SKIP: $LIBRITTS100_DIR not found. Run download first."
fi

# =========================================================
# Step 2: Encode VCTK
# =========================================================
echo ""
echo "--- VCTK ---"
VCTK_DIR=$RAW_DIR/VCTK-Corpus-0.92/wav48_silence_trimmed
if [ -d "$VCTK_DIR" ]; then
    python -u $WORK_DIR/scripts/preprocess_dataset.py \
        --data_dir   $VCTK_DIR \
        --output_dir $DATA_STORAGE/encoded_datasets \
        --prefix     vctk \
        --format     vctk \
        --max_sec_per_speaker 150 \
        --max_utts_per_speaker 20
else
    echo "SKIP: $VCTK_DIR not found. Run download first."
fi

# =========================================================
# Step 3: Merge all into processed_merged_v2
# =========================================================
echo ""
echo "--- Merging all datasets ---"

# Build list of available encoded datasets
EXTRA_DATA=""
EXTRA_EMB=""

# LibriTTS-R 360 (from previous preprocessing)
if [ -f "$DATA_STORAGE/libritts_r_processed/libritts_r_data.pt" ]; then
    EXTRA_DATA="$EXTRA_DATA $DATA_STORAGE/libritts_r_processed/libritts_r_data.pt"
    EXTRA_EMB="$EXTRA_EMB $DATA_STORAGE/libritts_r_processed/libritts_r_embeddings.pkl"
fi

# LibriTTS-R 100
if [ -f "$DATA_STORAGE/encoded_datasets/libritts100_data.pt" ]; then
    EXTRA_DATA="$EXTRA_DATA $DATA_STORAGE/encoded_datasets/libritts100_data.pt"
    EXTRA_EMB="$EXTRA_EMB $DATA_STORAGE/encoded_datasets/libritts100_embeddings.pkl"
fi

# VCTK
if [ -f "$DATA_STORAGE/encoded_datasets/vctk_data.pt" ]; then
    EXTRA_DATA="$EXTRA_DATA $DATA_STORAGE/encoded_datasets/vctk_data.pt"
    EXTRA_EMB="$EXTRA_EMB $DATA_STORAGE/encoded_datasets/vctk_embeddings.pkl"
fi

if [ -n "$EXTRA_DATA" ]; then
    python -u $WORK_DIR/scripts/merge_all_datasets.py \
        --original  $DATA_STORAGE/processed \
        --extra     $EXTRA_DATA \
        --extra_emb $EXTRA_EMB \
        --output    $DATA_STORAGE/processed_merged_v2
else
    echo "ERROR: No encoded datasets found! Check preprocessing logs."
    exit 1
fi

echo ""
echo "================================================================"
echo " All done: $(date)"
echo "================================================================"
