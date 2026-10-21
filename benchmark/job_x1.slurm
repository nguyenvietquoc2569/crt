#!/bin/bash
#SBATCH --job-name=crt_x1
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_x1_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_x1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# Job X1 (DEVIATIONS_v1.md item 5): an exploratory addition decided on 21 Oct 2026, after the test results
# (TEST_RESULTS_v1.md), so it is not part of the pre-registered analysis and cannot change any §7 decision. FreeVC,
# Seed-VC and FACodec at the 3 s reference budget on VCTK-2k (the record's variant) and LTC-2k, converted as job T4
# converts kNN-VC at 3 s (run_baselines.py with each adapter's fixed settings, no --cfg; seedvc under envs/seedvc) into
# the new out dir $J/v1/x1/bl, then scored as job M part t4 scores the baselines (lab_rows.py with the @src rows,
# metrics.py with Whisper large-v3 and base) into $J/v1/x1/m. Runs only after every pre-registered test job: t4.done
# and m/t4/m.done must carry the current selection.env's sha256. Nothing under T1-T4 or M is written. The comparisons
# (CRT-AL - each baseline at 3 s) run on the Mac: analyze_exploratory.py. run_baselines.py and metrics.py resume, so a
# job that hits its limit is resubmitted unchanged. Writes x1.done once every system has a complete manifest for both
# sets and both sets are scored.

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
OUT=$J/v1/x1
LP=$J/pairs/ltc2k.jsonl
BL=$OUT/bl
ENVS=/datastore/ceai/crt/envs
SYSTEMS="freevc seedvc facodec"
B=3
mkdir -p $OUT/m/rows $OUT/m/metrics
TIMES=$OUT/timings.tsv
: > $TIMES
FAILED=
stamp() { echo " $1: $(date)"; echo "$(date +%s) $1" >> $TIMES; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }
rows_ok() {  # rows_ok <jsonl> [<rows per system>]: prints counts; fails on an error row that is not a reference
             # shortfall, or when a system has another count
    python - "$@" <<'PY'
import collections, json, os, sys
if not os.path.exists(sys.argv[1]):
    sys.exit(f'  {sys.argv[1]}: missing')
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = collections.Counter(r['system'] for r in rows)
bad = [r for r in rows if 'error' in r and not r.get('shortfall')]
want = int(sys.argv[2]) if len(sys.argv) > 2 else None
off = {s: k for s, k in n.items() if want is not None and k != want}
print(f'  {sys.argv[1]}: {len(rows)} rows, {len(n)} systems, {min(n.values(), default=0)}-{max(n.values(), default=0)} '
      f'rows per system, {sum(bool(r.get("shortfall")) for r in rows)} shortfall rows, {len(bad)} other error rows'
      + (f' (e.g. {bad[0]["pair_id"]} {bad[0]["system"]}: {str(bad[0]["error"])[:300]})' if bad else '')
      + (f'; not {want} rows: {off}' if off else ''), flush=True)
sys.exit(1 if bad or off or not rows else 0)
PY
}

# ---- values from the selection record (as job_t4.slurm) ---------------------------------------------------------
SEL=$J/v1/selection.env
[ -f $SEL ] || { echo "ERROR: $SEL missing: write it from the committed selection record (RUNBOOK_v1.md)"; exit 1; }
source $SEL
SELSHA=$(sha256sum < $SEL | cut -d' ' -f1)
echo "selection $SEL (sha256 $SELSHA): record $RECORD_SHA256, CRT-AL $CRT_AL, VCTK-2k $VCTK_TARGETS targets," \
     "baselines $BASELINES"
[[ $RECORD_SHA256 =~ ^[0-9a-f]{64}$ ]] && [[ " 5 10 " == *" $VCTK_TARGETS "* ]] \
    && [[ $BASELINES =~ ^(none|(knnvc|freevc|seedvc|facodec)(,(knnvc|freevc|seedvc|facodec))*)$ ]] \
    || { echo "ERROR: $SEL holds a placeholder or a value outside the plan"; exit 1; }
if [ $VCTK_TARGETS = 10 ]; then VP=$J/pairs_vctk2k_t10/vctk2k.jsonl; else VP=$J/pairs/vctk2k.jsonl; fi
SE=$(dirname $VP)/vctk2k_source_E.json
# ------------------------------------------------------------------------------------------------------------------

for f in $J/v1/t4/t4.done $J/v1/m/t4/m.done; do
    grep -q "selection sha256 $SELSHA" $f 2>/dev/null \
        || { echo "ERROR: $f missing or written under another selection.env: X1 runs after every test job"; exit 1; }
done
for s in $SYSTEMS; do
    [[ ",$BASELINES," == *",$s,"* ]] || { echo "ERROR: the record does not list $s"; exit 1; }
done
for f in $VP $LP $SE; do
    [ -f $f ] && ! grep -q -e /datastore/nvquoc -e /datastore/ceai/nvquoc -e /datastore/eaic $f \
        || { echo "ERROR: $f is missing or holds paths of an old home (nvquoc, ceai/nvquoc, eaic)"; exit 1; }
done
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/x1.done   # a re-run that fails must not leave the last run's marker

for SET in vctk2k ltc2k; do
    if [ $SET = vctk2k ]; then PL=$VP; else PL=$LP; fi
    for SYS in $SYSTEMS; do
        [ $SYS = seedvc ] && source $ENVS/seedvc/bin/activate
        stamp "$SET $B s: $SYS"
        python -u $J/run_baselines.py --pairs $PL --set $SET --budget $B --systems $SYS --out_dir $BL --device cuda
        st $? ${SYS}_${SET}_$B
        [ $SYS = seedvc ] && source $ENVS/mimi_vc/bin/activate
    done
done
COMPLETE=1
for SET in vctk2k ltc2k; do
    if [ $SET = vctk2k ]; then PL=$VP; E=$SE; else PL=$LP; E=-; fi
    N=$(grep -c . $PL)
    INS=
    for SYS in $SYSTEMS; do
        rows_ok $BL/$SET/$B/$SYS/manifest.jsonl $N || COMPLETE=0
        INS="$INS --manifest $BL/$SET/$B/$SYS/manifest.jsonl"
    done
    [ $COMPLETE = 1 ] || { FAILED="$FAILED conversions_$SET"; break; }
    g=${SET}_$B
    stamp "$g: rows"
    python -u $J/lab_rows.py --pairs $PL $([ $E = - ] || echo --source_e $E) $INS --out $OUT/m/rows/$g.jsonl
    st $? rows_$g
    stamp "$g: metrics"
    python -u $J/metrics.py --in_jsonl $OUT/m/rows/$g.jsonl --out_jsonl $OUT/m/metrics/$g.jsonl \
        --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --device cuda
    st $? metrics_$g
    rows_ok $OUT/m/metrics/$g.jsonl $N
    st $? metrics_rows_$g
    stamp "$g: metrics, source speaker"
    python -u $J/metrics.py --in_jsonl $OUT/m/rows/${g}_src.jsonl --out_jsonl $OUT/m/metrics/${g}_src.jsonl \
        --metrics sim,secs --device cuda
    st $? metrics_src_$g
    rows_ok $OUT/m/metrics/${g}_src.jsonl
    st $? metrics_src_rows_$g
done
stamp "done"
du -sh $OUT /datastore/ceai/crt
echo " finished, failed steps:${FAILED:- none}, every system complete: $COMPLETE: $(date)"
[ -z "$FAILED" ] && [ $COMPLETE = 1 ] && echo "job $SLURM_JOB_ID $(date '+%F %T') selection sha256 $SELSHA" \
    > $OUT/x1.done
[ -z "$FAILED" ]
