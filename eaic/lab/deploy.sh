#!/bin/bash
# Deploy Mimi-VC V3f files to UIT Slurm cluster
# Run from your local Mac: bash eaic/lab/deploy.sh

set -e

REMOTE_USER=eaic
REMOTE_HOST=slurm.uit.edu.vn
REMOTE_DIR=/datastore/eaic/nvquoc/mimi_vc
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Mimi-VC Deploy to Slurm ==="
echo "Local:  $LAB_DIR"
echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo ""

# 1. Create remote directories
echo "[1/4] Creating remote directories..."
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR/target_audio $REMOTE_DIR/logs"

# 2. Copy training scripts
echo "[2/4] Copying scripts..."
scp "$LAB_DIR/train_v3f.py"        $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
scp "$LAB_DIR/run_v3f.slurm"       $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
scp "$LAB_DIR/prepare_data.slurm"  $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
scp "$LAB_DIR/download_data.sh"    $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
scp "$LAB_DIR/download_data.slurm" $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
scp "$LAB_DIR/setup_env.sh"        $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/

# 3. Copy source audio
echo "[3/4] Copying source audio..."
scp "$LAB_DIR/source.mp3" $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/source.mp3

# 4. Copy target audio folder
echo "[4/4] Copying target audio files..."
rsync -avz "$LAB_DIR/target_audio/" $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/target_audio/

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps on the cluster:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  bash setup_env.sh               # skip if already done"
echo "  sbatch download_data.slurm      # step 1: download LibriSpeech (~28GB, CPU job)"
echo "  tail -f logs/download_*.out     # monitor download"
echo "  sbatch prepare_data.slurm       # step 2: encode with GPU"
echo "  tail -f logs/prepare_*.out      # monitor encoding"
echo "  sbatch run_v3f.slurm            # step 3: train"
echo "  tail -f logs/mimi_vc_*.out"
