#!/bin/bash
# Download LibriSpeech dataset on the LOGIN NODE (has internet access).
# Run this BEFORE submitting prepare_data.slurm.
#
# Usage (on login node):
#   bash download_data.sh

set -e

WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
ENV_DIR=/datastore/eaic/nvquoc/mimi_vc_env
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

echo "=== Download: Mimi model + LibriSpeech (login node) ==="
echo "Data storage: $DATA_STORAGE"
echo ""

mkdir -p $WORK_DIR/logs \
         $DATA_STORAGE/librispeech $DATA_STORAGE/processed \
         $DATA_STORAGE/hf_cache $DATA_STORAGE/pip_cache \
         $DATA_STORAGE/cache $DATA_STORAGE/torch_cache

source $ENV_DIR/bin/activate

# 1. Pre-download Mimi model
echo "[1/2] Downloading Mimi model..."
python -c "
from transformers import MimiModel, AutoFeatureExtractor
print('Downloading kyutai/mimi...')
MimiModel.from_pretrained('kyutai/mimi')
AutoFeatureExtractor.from_pretrained('kyutai/mimi')
print('Mimi cached.')
"

# 2. Download LibriSpeech
echo ""
echo "[2/2] Downloading LibriSpeech train-clean-100 + train-clean-360..."
python $SCRIPT \
    --download_only \
    --data_raw       $DATA_STORAGE/librispeech \
    --data_processed $DATA_STORAGE/processed

echo ""
echo "=== Download complete ==="
echo "Now submit the encode job:"
echo "  sbatch $WORK_DIR/prepare_data.slurm"
