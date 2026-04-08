#!/bin/bash
#SBATCH --job-name=bench_v80f
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_v80f_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_v80f_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

REQUIRED_VRAM=20480

module clear -f
module load shared python312
source /datastore/eaic/nvquoc/mimi_vc_env/bin/activate
unset CUDA_VISIBLE_DEVICES

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
BENCH_DIR=$WORK_DIR/benchmark

DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export TORCH_HOME=$DATA_STORAGE/torch_cache
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export PYTHONUNBUFFERED=1

mkdir -p $BENCH_DIR/logs $BENCH_DIR/results_v8_full

echo "================================================================"
echo " Full Benchmark: CRT v8.0 — SECS + MCD + WER + UTMOS"
echo " $(date)"
echo "================================================================"

SKIP_KNNVC=""
SKIP_FREEVC=""
[ ! -d "$BENCH_DIR/knn-vc" ] && SKIP_KNNVC="--skip_knnvc"
[ ! -f "$BENCH_DIR/FreeVC/checkpoints/freevc.pth" ] && SKIP_FREEVC="--skip_freevc"

python -u $BENCH_DIR/run_benchmark.py \
    --source_audio  $WORK_DIR/source.mp3 \
    --target_dir    $WORK_DIR/target_audio \
    --crt_output_dir $WORK_DIR/v8.0-progressive-rvq-dropout/output \
    --output_dir    $BENCH_DIR/results_v8_full \
    --knnvc_dir     $BENCH_DIR/knn-vc \
    --freevc_dir    $BENCH_DIR/FreeVC \
    --crt_step      240000 \
    $SKIP_KNNVC $SKIP_FREEVC

echo ""
echo "================================================================"
echo " Benchmark v8.0 full completed: $(date)"
echo "================================================================"
