#!/bin/bash
#SBATCH --job-name=bench_full
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_full_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

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
BENCH_DIR=$WORK_DIR/benchmark

DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export TORCH_HOME=$DATA_STORAGE/torch_cache
# Allow downloads for Whisper/UTMOS if not cached
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

export PYTHONUNBUFFERED=1
mkdir -p $BENCH_DIR/logs $BENCH_DIR/results_full

# =========================================================
echo "================================================================"
echo " Full Benchmark: CRT v5.6.5 vs kNN-VC vs FreeVC"
echo " Metrics: SECS + MCD + WER (Whisper) + UTMOS (Neural MOS)"
echo " $(date)"
echo "================================================================"

SKIP_KNNVC=""
SKIP_FREEVC=""

if [ ! -d "$BENCH_DIR/knn-vc" ]; then
    echo "WARNING: kNN-VC not found. Skipping."
    SKIP_KNNVC="--skip_knnvc"
fi

if [ ! -f "$BENCH_DIR/FreeVC/checkpoints/freevc.pth" ]; then
    echo "WARNING: FreeVC model not found. Skipping."
    SKIP_FREEVC="--skip_freevc"
fi

# =========================================================
# Run full benchmark — v5.6.5 at step 210000
# =========================================================
python -u $BENCH_DIR/run_benchmark.py \
    --source_audio  $WORK_DIR/source.mp3 \
    --target_dir    $WORK_DIR/target_audio \
    --crt_output_dir $WORK_DIR/v5.6.5-secs-high/output \
    --output_dir    $BENCH_DIR/results_full \
    --knnvc_dir     $BENCH_DIR/knn-vc \
    --freevc_dir    $BENCH_DIR/FreeVC \
    --crt_step      210000 \
    $SKIP_KNNVC $SKIP_FREEVC

echo ""
echo "================================================================"
echo " Full benchmark completed: $(date)"
echo "================================================================"
