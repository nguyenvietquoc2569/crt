#!/bin/bash
#SBATCH --job-name=crt_lt
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_lt_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_lt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# The listening-test stimuli of ANALYSIS_PLAN_v1.md §5.4 (benchmark/listening_stimuli.py), on ceai after
# job_move_ceai.slurm (move.done): the plan from the record's 32 items, the pair lists and the T1, T2 and T4 outputs,
# then the build (the missing Mimi Q = 32 copies on this GPU in float16, cut, loudness normalisation, FLAC under ids
# that hide the condition, lt_data.js for the page; key.json for the organiser beside, not inside, the package).
# New out dir $J/v1/listening.
# The page itself (benchmark/listening/page/) is added on the Mac. Submit with submit_ceai.sh.

source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/ceai/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

WORK_DIR=/datastore/ceai/crt/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$J/v1/listening
[ -f $J/v1/move/move.done ] || { echo "ERROR: $J/v1/move/move.done missing: run job_move_ceai.slurm first"; exit 1; }
[ -e $OUT/key.json ] && { echo "ERROR: $OUT/key.json exists: move $OUT aside first"; exit 1; }
mkdir -p $OUT
cd $J
python -u listening_stimuli.py plan --items DEV_SELECTION_RECORD_v1.listening_items.json \
    --record DEV_SELECTION_RECORD_v1.json --vctk_pairs pairs/vctk2k.jsonl --ltc_pairs pairs/ltc2k.jsonl \
    --v1 $J/v1 --out $OUT/plan.json || exit 1
python -u listening_stimuli.py build --plan $OUT/plan.json --out $OUT/package --key $OUT/key.json --device cuda || exit 1
du -sh $OUT/package
echo "job $SLURM_JOB_ID $(date '+%F %T')" > $OUT/listening.done
