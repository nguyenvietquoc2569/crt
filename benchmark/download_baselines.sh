#!/bin/bash
# Download kNN-VC and FreeVC repos + pretrained models
# Run on login node (has internet): ssh eaic "bash /datastore/eaic/nvquoc/mimi_vc/benchmark/download_baselines.sh"

set -e

WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
BENCH_DIR=$WORK_DIR/benchmark

source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate

mkdir -p $BENCH_DIR

# === kNN-VC ===
echo "=== Downloading kNN-VC ==="
KNNVC_DIR=$BENCH_DIR/knn-vc
if [ ! -d "$KNNVC_DIR" ]; then
    cd $BENCH_DIR
    git clone https://github.com/bshall/knn-vc.git
fi
# Pre-download WavLM and HiFi-GAN models via huggingface hub
pip install --quiet huggingface_hub 2>/dev/null
python3 -c "
from huggingface_hub import hf_hub_download
import os
cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print('Downloading WavLM-Large...')
hf_hub_download('microsoft/wavlm-large', 'pytorch_model.bin', cache_dir=cache)
hf_hub_download('microsoft/wavlm-large', 'config.json', cache_dir=cache)
print('Downloading kNN-VC HiFi-GAN...')
hf_hub_download('bshall/knn-vc', 'prematch_g_02500000.pt', cache_dir=cache)
print('Done downloading kNN-VC models')
" 2>/dev/null || echo "WARNING: Could not pre-download kNN-VC models, will try at runtime"

# === FreeVC ===
echo "=== Downloading FreeVC ==="
FREEVC_DIR=$BENCH_DIR/FreeVC
if [ ! -d "$FREEVC_DIR" ]; then
    cd $BENCH_DIR
    git clone https://github.com/OlaWod/FreeVC.git
fi
cd $FREEVC_DIR
mkdir -p checkpoints

# FreeVC pretrained model
if [ ! -f "checkpoints/freevc.pth" ]; then
    echo "Downloading FreeVC pretrained model..."
    pip install --quiet gdown 2>/dev/null
    gdown "https://drive.google.com/uc?id=10vEhJBnEqGMhHqeFKjqKMOreAqIrm0AT" -O checkpoints/freevc.pth || \
    echo "ERROR: gdown failed. Manual download: https://drive.google.com/file/d/10vEhJBnEqGMhHqeFKjqKMOreAqIrm0AT"
fi

# FreeVC speaker encoder
mkdir -p speaker_encoder/ckpt
if [ ! -f "speaker_encoder/ckpt/pretrained_bak_5805000.pt" ]; then
    echo "Downloading FreeVC speaker encoder..."
    gdown "https://drive.google.com/uc?id=1YMp4F8YjE3qnVBPxuBIML1JhbKJFAdSe" -O speaker_encoder/ckpt/pretrained_bak_5805000.pt || \
    echo "ERROR: gdown failed for speaker encoder"
fi

# FreeVC WavLM
mkdir -p wavlm
if [ ! -f "wavlm/WavLM-Large.pt" ]; then
    echo "Downloading WavLM-Large for FreeVC..."
    gdown "https://drive.google.com/uc?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU" -O wavlm/WavLM-Large.pt || \
    echo "ERROR: gdown failed for WavLM. Manual: https://github.com/OlaWod/FreeVC#quick-start"
fi

# Install FreeVC deps
pip install --quiet praat-parselmouth torchvision 2>/dev/null

echo ""
echo "=== Download complete ==="
echo "kNN-VC: $KNNVC_DIR"
echo "FreeVC: $FREEVC_DIR"
echo "FreeVC model: $(ls -lh $FREEVC_DIR/checkpoints/freevc.pth 2>/dev/null || echo 'MISSING')"
echo "FreeVC spk_enc: $(ls -lh $FREEVC_DIR/speaker_encoder/ckpt/pretrained_bak_5805000.pt 2>/dev/null || echo 'MISSING')"
echo "FreeVC WavLM: $(ls -lh $FREEVC_DIR/wavlm/WavLM-Large.pt 2>/dev/null || echo 'MISSING')"
