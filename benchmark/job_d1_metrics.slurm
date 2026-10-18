#!/bin/bash
#SBATCH --job-name=crt_d1_metrics
#SBATCH --output=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_d1_metrics_%j.out
#SBATCH --error=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_d1_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Job D1, part 2 of 2: scoring (ANALYSIS_PLAN_v1.md §8.2; why D1 has two jobs: job_d1_conv.slurm), after
# job_d1_conv.slurm wrote $J/v1/d1/conv.done:
#   metrics.py on the anchor-lab manifest (dB-MCD/F0, Whisper large-v3 only as on all of Dev, WavLM and Resemblyzer
#   SIM, UTMOS) and on manifest_src (SIM_src, SECS_src) -> $J/v1/d1/anc/dev/30/metrics.jsonl, metrics_src.jsonl
#   (select_crt_al.py --metrics). Estimate 12.1 h (§8.2); metrics.py resumes, so a job that hits its time limit is
#   resubmitted unchanged.
# metrics.py exits 0 even with error rows, so every output is counted: 1,000 rows per system and no error row, else
# the job fails. Writes metrics.done when both files are complete.

source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/nvquoc/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

WORK_DIR=/datastore/nvquoc/crt/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$J/v1/d1
ANC=$OUT/anc
TIMES=$OUT/timings_metrics.tsv
: > $TIMES
FAILED=
stamp() { echo " $1: $(date)"; echo "$(date +%s) $1" >> $TIMES; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }
rows_ok() {  # rows_ok <jsonl> [<rows per system>]: prints counts; fails on an error row that is not a reference
             # shortfall, or when a system other than mimi_copy_P / target_copy (pairs with P only) has another count
    python - "$@" <<'PY'
import collections, json, os, sys
if not os.path.exists(sys.argv[1]):
    sys.exit(f'  {sys.argv[1]}: missing')
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = collections.Counter(r['system'] for r in rows)
bad = [r for r in rows if 'error' in r and not r.get('shortfall')]
want = int(sys.argv[2]) if len(sys.argv) > 2 else None
p_only = ('mimi_copy_P', 'target_copy')
off = {s: k for s, k in n.items() if want is not None and k != want and s.split('@')[0] not in p_only}
print(f'  {sys.argv[1]}: {len(rows)} rows, {len(n)} systems, {min(n.values(), default=0)}-{max(n.values(), default=0)} '
      f'rows per system, {sum(bool(r.get("shortfall")) for r in rows)} shortfall rows, {len(bad)} other error rows'
      + (f' (e.g. {bad[0]["pair_id"]} {bad[0]["system"]}: {str(bad[0]["error"])[:300]})' if bad else '')
      + (f'; not {want} rows: {off}' if off else ''), flush=True)
sys.exit(1 if bad or off or not rows else 0)
PY
}

[ -f $OUT/conv.done ] || { echo "ERROR: $OUT/conv.done missing: job_d1_conv.slurm did not succeed"; exit 1; }
cat $OUT/conv.done
rm -f $OUT/metrics.done   # a re-run that fails must not leave the last run's marker

stamp "metrics: anchor lab"
python -u $J/metrics.py --in_jsonl $ANC/dev/30/manifest.jsonl --out_jsonl $ANC/dev/30/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? metrics
rows_ok $ANC/dev/30/metrics.jsonl 1000
st $? metrics_rows
stamp "metrics: anchor lab, source speaker"
python -u $J/metrics.py --in_jsonl $ANC/dev/30/manifest_src.jsonl --out_jsonl $ANC/dev/30/metrics_src.jsonl \
    --metrics sim,secs --device cuda
st $? metrics_src
rows_ok $ANC/dev/30/metrics_src.jsonl 1000
st $? metrics_src_rows
stamp "done"
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T')" > $OUT/metrics.done
[ -z "$FAILED" ]
