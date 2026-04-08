#!/bin/bash
#SBATCH --job-name=bench_intg
#SBATCH --output=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_intg_%j.out
#SBATCH --error=/datastore/eaic/nvquoc/mimi_vc/benchmark/logs/bench_intg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00

REQUIRED_VRAM=10000

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
mkdir -p $WORK_DIR/benchmark/logs $WORK_DIR/benchmark/results_integrability

echo "================================================================"
echo " Integrability Benchmark for PersonaPlex"
echo " $(date)"
echo "================================================================"

python -u $WORK_DIR/benchmark/run_integrability_benchmark.py \
    --output_dir $WORK_DIR/benchmark/results_integrability

echo "================================================================"
echo " Completed: $(date)"
echo "================================================================"
