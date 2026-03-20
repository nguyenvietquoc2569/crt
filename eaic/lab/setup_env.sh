#!/bin/bash
# Setup Python venv for Mimi-VC on UIT Slurm
# Run this ONCE on the login node before submitting jobs
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh

set -e

WORK_DIR=/datastore/eaic/nvquoc/mimi_vc
ENV_DIR=/datastore/eaic/nvquoc/mimi_vc_env
DATA_STORAGE=$WORK_DIR/data_storage

# Redirect ALL caches to datastore (avoid filling /home quota)
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export PIP_CACHE_DIR=$DATA_STORAGE/pip_cache

mkdir -p $DATA_STORAGE/hf_cache $DATA_STORAGE/pip_cache $DATA_STORAGE/cache

echo "=== Mimi-VC Environment Setup ==="
echo "Work dir: $WORK_DIR"
echo "Env dir: $ENV_DIR"

# Load Python
module clear -f
module load shared python312

# Create venv
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $ENV_DIR
else
    echo "Venv already exists at $ENV_DIR"
fi

source $ENV_DIR/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets
pip install faiss-cpu
pip install librosa soundfile matplotlib
pip install resemblyzer
pip install numpy scipy

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Create work directory:"
echo "     mkdir -p $WORK_DIR/target_audio"
echo ""
echo "  2. Copy your files:"
echo "     cp train_v3f.py $WORK_DIR/"
echo "     cp run_v3f.slurm $WORK_DIR/"
echo "     cp target_speaker.wav $WORK_DIR/target_audio/"
echo "     cp source.wav $WORK_DIR/"
echo ""
echo "  3. Submit job:"
echo "     cd $WORK_DIR && sbatch run_v3f.slurm"
echo ""
echo "  4. Monitor:"
echo "     squeue -u $USER"
echo "     tail -f logs/mimi_vc_*.out"
