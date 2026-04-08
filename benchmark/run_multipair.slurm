#!/bin/bash
#SBATCH --job-name=bench_mp
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_mp_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_mp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

REQUIRED_VRAM=40000

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

mkdir -p $BENCH_DIR/logs $BENCH_DIR/results_multipair

echo "================================================================"
echo " Multi-Pair Benchmark (VCTK, 100 pairs)"
echo " CRT v5.6.5 vs kNN-VC vs FreeVC"
echo " $(date)"
echo "================================================================"

# Check VCTK is downloaded
if [ ! -f "$BENCH_DIR/vctk/eval_pairs.json" ]; then
    echo "ERROR: VCTK not downloaded. Run download_vctk.sh first."
    exit 1
fi

SKIP_KNNVC=""
SKIP_FREEVC=""
[ ! -d "$BENCH_DIR/knn-vc" ] && SKIP_KNNVC="--skip_knnvc"
[ ! -f "$BENCH_DIR/FreeVC/checkpoints/freevc.pth" ] && SKIP_FREEVC="--skip_freevc"

python -u $BENCH_DIR/run_multipair_benchmark.py \
    --pairs_json    $BENCH_DIR/vctk/eval_pairs.json \
    --max_pairs     100 \
    --crt_ckpt      $WORK_DIR/v5.6.5-secs-high/checkpoints/crt_xlarge_best.pt \
    --crt_script    $WORK_DIR/v5.6.5-secs-high/train_crt.py \
    --knnvc_dir     $BENCH_DIR/knn-vc \
    --freevc_dir    $BENCH_DIR/FreeVC \
    --output_dir    $BENCH_DIR/results_multipair \
    $SKIP_KNNVC $SKIP_FREEVC

echo ""
echo "================================================================"
echo " Multi-pair benchmark completed: $(date)"
echo "================================================================"
