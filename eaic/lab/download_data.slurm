#!/bin/bash
#SBATCH --job-name=mimi_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# No GPU needed for download

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
export PIP_CACHE_DIR=$DATA_STORAGE/pip_cache

mkdir -p $WORK_DIR/logs \
         $DATA_STORAGE/librispeech $DATA_STORAGE/processed \
         $DATA_STORAGE/hf_cache $DATA_STORAGE/pip_cache $DATA_STORAGE/cache

# =========================================================
# Activate venv
# =========================================================
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate

echo "=== Starting LibriSpeech download ==="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Destination: $WORK_DIR/data_raw"
echo ""

python $SCRIPT \
    --download_only \
    --data_raw       $DATA_STORAGE/librispeech \
    --data_processed $DATA_STORAGE/processed

echo ""
echo "=== Download complete: $(date) ==="
echo "data_raw size:"
du -sh $WORK_DIR/data_raw/
echo ""
echo "Next: sbatch $WORK_DIR/prepare_data.slurm"
