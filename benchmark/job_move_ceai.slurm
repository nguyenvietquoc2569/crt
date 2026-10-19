#!/bin/bash
#SBATCH --job-name=crt_move
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_move_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_move_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# The CRT tree's move from nvquoc to ceai (DEVIATIONS_v1.md item 4), run on ceai once the relay's last verify shows
# the copy identical and the job layer for ceai has been pushed. New out dir $J/v1/move:
#   1. move_paths.py rewrite: the old folder -> /datastore/ceai/crt in every text file and symlink, the job logs,
#      _transfer_logs and the two files that name both folders left as they are; log move_paths_<job>.tsv
#   2. move_paths.py check over every move_paths_*.tsv: logged hashes, the section-2 hashes of FREEZE_v1.sha256 mapped
#      back, no file or link still naming the old folder
#   3. the environment: pip freeze of envs/mimi_vc against FREEZE_v1_pip_freeze.txt, the imports metrics.py needs,
#      CUDA; compiled libraries under envs that still name the old folder are listed (they are not rewritten)
#   4. Dev scored again on ceai: metrics.py as job_d1_metrics.slurm ran it, on the first 32 rows of the D1 anchor-lab
#      manifest (the first two Whisper batches), then move_paths.py compare with the D1 rows: every number within
#      1e-4, Whisper hypotheses reported
# Writes move.done when all four pass. Submit with submit_ceai.sh.

source /etc/profile.d/modules.sh 2>/dev/null
module load python312
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

ROOT=/datastore/ceai/crt
OLD_ACCOUNT=nvquoc
OLD=/datastore/$OLD_ACCOUNT/crt          # no job script names the old home literally (tests/test_job_scripts.py)
WORK_DIR=$ROOT/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
OUT=$J/v1/move
mkdir -p $OUT
rm -f $OUT/move.done
FAILED=
stamp() { echo " $1: $(date)"; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }
KEEP="--keep _transfer_logs --keep mimi_vc/benchmark/logs --keep mimi_vc/benchmark/journal/logs
      --keep mimi_vc/benchmark/journal/move_paths.py --keep mimi_vc/benchmark/journal/job_move_ceai.slurm"

stamp "1. rewrite $OLD -> $ROOT"
python3 -u $J/move_paths.py rewrite --root $ROOT --old $OLD --new $ROOT --log $OUT/move_paths_$SLURM_JOB_ID.tsv $KEEP
st $? rewrite

stamp "2. check"
LOGS=$(ls $OUT/move_paths_*.tsv 2>/dev/null | sed 's/^/--log /')
python3 -u $J/move_paths.py check --root $ROOT --old $OLD --new $ROOT $LOGS --freeze $J/FREEZE_v1.sha256 $KEEP
st $? check

stamp "3. environment"
source /datastore/ceai/crt/envs/mimi_vc/bin/activate
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "  python $(which python), VIRTUAL_ENV $VIRTUAL_ENV"
pip freeze > $OUT/pip_freeze.txt
diff <(grep -v '^#' $J/FREEZE_v1_pip_freeze.txt | sort) <(sort $OUT/pip_freeze.txt)
st $? pip_freeze
python -c "import torch, whisper, transformers, resemblyzer, pyworld, pysptk, librosa, jiwer, soundfile
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
assert torch.cuda.is_available()"
st $? imports
grep -rl --include='*.so' "$OLD" $ROOT/envs 2>/dev/null | head -20 | sed 's/^/  .so naming the old folder: /'

stamp "4. Dev scored again on ceai"
python -u $J/metrics.py --in_jsonl $J/v1/d1/anc/dev/30/manifest.jsonl --max_rows 32 \
    --out_jsonl $OUT/dev_check_metrics.jsonl --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? dev_metrics
python3 -u $J/move_paths.py compare --new $OUT/dev_check_metrics.jsonl --ref $J/v1/d1/anc/dev/30/metrics.jsonl
st $? dev_compare

stamp "done"
du -sh $ROOT
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T')" > $OUT/move.done
[ -z "$FAILED" ]
