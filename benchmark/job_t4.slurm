#!/bin/bash
#SBATCH --job-name=crt_t4
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_t4_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_t4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# Job T4 (ANALYSIS_PLAN_v1.md §8.2, §2.4), after T3 ($J/v1/t3/t3.done, same selection.env), only for the baselines
# that the selection record lists (BASELINES in selection.env: knnvc = S5 kNN-VC, freevc = S6 FreeVC, seedvc = S7
# Seed-VC, facodec = S8 FACodec; none: nothing to run, job M follows T3): run_baselines.py at 30 s on VCTK-2k (the
# record's variant) and LTC-2k, kNN-VC also at 3 s and 180 s (E16), new out dir $J/v1/t4/bl. No metrics (job M).
# Every system runs with its adapter's fixed settings (baselines/*.py, no --cfg), as in the smoke test whose command
# the record holds, and in the same environment as job_baselines_smoke.slurm: seedvc with envs/seedvc activated
# (baselines/seedvc.py), the others under mimi_vc, MPLCONFIGDIR as there. The conv_cfg_sha of every row must equal
# the smoke test's (RUNBOOK_v1.md). Baseline speed was measured only in the
# smoke test (runs.jsonl, median RTF): run_baselines.py resumes, so a job that hits 72 h is resubmitted unchanged,
# and ONLY=<system>[:<system>...] (sbatch --export=ALL,ONLY=seedvc) limits one submission to some of the listed
# systems. Writes t4.done once every listed system has a complete manifest for every set and budget.

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
export MPLCONFIGDIR=$DATA_STORAGE/cache/matplotlib
OUT=$J/v1/t4
LP=$J/pairs/ltc2k.jsonl
BL=$OUT/bl
ENVS=/datastore/ceai/crt/envs
mkdir -p $OUT
TIMES=$OUT/timings.tsv
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

# ---- values from the selection record (as job_t1.slurm) ---------------------------------------------------------
SEL=$J/v1/selection.env
[ -f $SEL ] || { echo "ERROR: $SEL missing: write it from the committed selection record (RUNBOOK_v1.md)"; exit 1; }
source $SEL
SELSHA=$(sha256sum < $SEL | cut -d' ' -f1)
echo "selection $SEL (sha256 $SELSHA): record $RECORD_SHA256, τ_dev $TAU_DEV, CRT-AL $CRT_AL, S1u $S1U, E17 $E17," \
     "VCTK-2k $VCTK_TARGETS targets, baselines $BASELINES"
[[ $RECORD_SHA256 =~ ^[0-9a-f]{64}$ ]] && [[ " 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 " == *" $TAU_DEV "* ]] \
    && [[ " C1 C2 C3 C4 C5 none " == *" $CRT_AL "* ]] && [[ " us32_w0.2_0.2_0.1 us32_w0.5_0.3_0.3 " == *" $S1U "* ]] \
    && [[ " yes no " == *" $E17 "* ]] && [[ " 5 10 " == *" $VCTK_TARGETS "* ]] \
    && [[ $BASELINES =~ ^(none|(knnvc|freevc|seedvc|facodec)(,(knnvc|freevc|seedvc|facodec))*)$ ]] \
    || { echo "ERROR: $SEL holds a placeholder or a value outside the plan"; exit 1; }
if [ $VCTK_TARGETS = 10 ]; then VP=$J/pairs_vctk2k_t10/vctk2k.jsonl; else VP=$J/pairs/vctk2k.jsonl; fi
# ------------------------------------------------------------------------------------------------------------------

grep -q "selection sha256 $SELSHA" $J/v1/t3/t3.done 2>/dev/null \
    || { echo "ERROR: $J/v1/t3/t3.done missing or written under another selection.env"; exit 1; }
[ "$BASELINES" = none ] && { echo "  the record lists no baseline: nothing to run (job M follows T3)"; exit 0; }
RUN=${ONLY:-$BASELINES}
RUN=${RUN//:/,}
for s in ${RUN//,/ }; do
    [[ ",$BASELINES," == *",$s,"* ]] || { echo "ERROR: ONLY=$ONLY names $s, which the record does not list"; exit 1; }
done
for f in $VP $LP; do
    [ -f $f ] && ! grep -q -e /datastore/nvquoc -e /datastore/ceai/nvquoc -e /datastore/eaic $f \
        || { echo "ERROR: $f is missing or holds paths of an old home (nvquoc, ceai/nvquoc, eaic)"; exit 1; }
done
echo "  baselines listed: $BASELINES; this submission: $RUN"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/t4.done   # a re-run that fails must not leave the last run's marker

for SET in vctk2k ltc2k; do
    if [ $SET = vctk2k ]; then PL=$VP; else PL=$LP; fi
    for SYS in ${RUN//,/ }; do
        [ $SYS = seedvc ] && source $ENVS/seedvc/bin/activate
        for B in 30 $([ $SYS = knnvc ] && echo 3 180); do
            stamp "$SET $B s: $SYS"
            python -u $J/run_baselines.py --pairs $PL --set $SET --budget $B --systems $SYS --out_dir $BL --device cuda
            st $? ${SYS}_${SET}_$B
        done
        [ $SYS = seedvc ] && source $ENVS/mimi_vc/bin/activate
    done
done
stamp "done"
COMPLETE=1
for SET in vctk2k ltc2k; do
    if [ $SET = vctk2k ]; then N=$(grep -c . $VP); else N=$(grep -c . $LP); fi
    for SYS in ${BASELINES//,/ }; do
        for B in 30 $([ $SYS = knnvc ] && echo 3 180); do
            rows_ok $BL/$SET/$B/$SYS/manifest.jsonl $N || COMPLETE=0
        done
    done
done
du -sh $BL /datastore/ceai/crt
echo " finished, failed steps:${FAILED:- none}, every listed baseline complete: $COMPLETE: $(date)"
[ -z "$FAILED" ] && [ $COMPLETE = 1 ] && echo "job $SLURM_JOB_ID $(date '+%F %T') selection sha256 $SELSHA" \
    > $OUT/t4.done
[ -z "$FAILED" ]
