#!/bin/bash
# Setup kNN-VC and FreeVC on the SLURM cluster for benchmarking
# Run once: ssh eaic "bash /datastore/eaic/nvquoc/mimi_vc/benchmark/setup_baselines.sh"

set -e

WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
BENCH_DIR=$WORK_DIR/benchmark
ENV_DIR=$WORK_DIR/benchmark_env

mkdir -p $BENCH_DIR

echo "=== Setting up benchmark environment ==="

# Create a separate venv to avoid dependency conflicts
if [ ! -d "$ENV_DIR" ]; then
    python3.12 -m venv $ENV_DIR
fi
source $ENV_DIR/bin/activate

# Install common deps
pip install --quiet torch torchaudio librosa soundfile resemblyzer numpy

# === kNN-VC ===
echo "=== Setting up kNN-VC ==="
KNNVC_DIR=$BENCH_DIR/knn-vc
if [ ! -d "$KNNVC_DIR" ]; then
    cd $BENCH_DIR
    git clone https://github.com/bshall/knn-vc.git
    cd knn-vc
    pip install --quiet -r requirements.txt 2>/dev/null || pip install --quiet wavlm hifigan
fi

# === FreeVC ===
echo "=== Setting up FreeVC ==="
FREEVC_DIR=$BENCH_DIR/FreeVC
if [ ! -d "$FREEVC_DIR" ]; then
    cd $BENCH_DIR
    git clone https://github.com/OlaWod/FreeVC.git
    cd FreeVC
    pip install --quiet -r requirements.txt 2>/dev/null || true
    # Download pretrained model
    mkdir -p checkpoints
    if [ ! -f "checkpoints/freevc.pth" ]; then
        echo "Downloading FreeVC pretrained model..."
        pip install --quiet gdown
        gdown https://drive.google.com/uc?id=1CgWYCR0ljtMm7k2K1-JULxvQNvPHOZAb -O checkpoints/freevc.pth 2>/dev/null || \
        echo "WARNING: Could not auto-download FreeVC model. Download manually from https://github.com/OlaWod/FreeVC#pretrained-models"
    fi
fi

echo "=== Setup complete ==="
echo "kNN-VC: $KNNVC_DIR"
echo "FreeVC: $FREEVC_DIR"
