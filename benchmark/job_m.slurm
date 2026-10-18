#!/bin/bash
#SBATCH --job-name=crt_m
#SBATCH --output=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_m_%j.out
#SBATCH --error=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_m_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# Job M (ANALYSIS_PLAN_v1.md §8.2): metrics.py on every test output, @src rows included, only after the last test
# conversion job (T4, or T3 if the record lists no baseline; §9: no test-set metric before every test conversion has
# finished), under the selection.env that job ran with. One part per submission, PART=t1|t2|t3|t4
# (bash submit_nvquoc.sh job_m.slurm --export=ALL,PART=t1), new out dir $J/v1/m/<part>:
#   t1  VCTK-2k 30 s: S1-S4, the Mimi copies, the natural controls (source_copy, target_copy from the FAISS rows, as
#       run_budget_check.slurm), A1-A5 and C1 (anc_*), S1u / E17 (front_*), CRT-AL if C2-C5 (cand_*)
#   t2  LTC-2k 30 s: the same, without P (no mimi_copy_P, no target_copy)
#   t3  the budget sweep, VCTK-2k and LTC-2k at 3 s and 180 s: S1-S3 and CRT-AL (anc_R3_s0_pm_rq_T1 or cand_*)
#   t4  the baselines (30 s; kNN-VC also 3 s and 180 s)
# Per set and budget: lab_rows.py merges the part's manifests (decode-lab rows renamed by tag) with sim_src_wavs
# (VCTK-2k: the variant's source-E file) into rows/<set>_<budget>.jsonl and its @src rows, then metrics.py with
# Whisper large-v3 and base (§3.2) -> metrics/<set>_<budget>.jsonl, and --metrics sim,secs on the @src rows ->
# metrics/<set>_<budget>_src.jsonl. Why parts: 147,000-238,000 rows (§8.2) at >= 1.1 s per target-side row are
# 33-52 h plus Whisper-base, which can pass 72 h in one job with the 10-target list; a part is about 8-20 h.
# metrics.py resumes, so a part that hits its time limit is resubmitted unchanged. Shortfall error rows (180 s,
# LTC-2k) stay error rows and are allowed; any other error row fails the part. Writes m.done in the part's dir.

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
case ${PART:-} in
    t1|t2|t3|t4) ;;
    *) echo "ERROR: PART=${PART:-} is not t1, t2, t3 or t4 (bash submit_nvquoc.sh job_m.slurm --export=ALL,PART=t1)"
       exit 1 ;;
esac
OUT=$J/v1/m/$PART
LP=$J/pairs/ltc2k.jsonl
mkdir -p $OUT/rows $OUT/metrics
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
natural() {  # natural <run_conversions manifest> <out>: source_copy and, with P, target_copy rows from the FAISS rows
    python - "$1" "$2" <<'PY'
import json, sys
rows = []
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r['system'] == 'faiss' and not r.get('error'):
        rows.append(dict(r, system='source_copy', wav=r['source_wav']))
        if r.get('parallel_wav'):
            rows.append(dict(r, system='target_copy', wav=r['parallel_wav'], copy_control=True))
open(sys.argv[2], 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(len(rows), 'natural-control rows ->', sys.argv[2])
PY
}
score() {  # score <set>_<budget> <pairs> <source-E file or -> <rows per system or -> <lab_rows.py inputs...>
    local g=$1 pl=$2 se=$3 n=$4 s
    shift 4
    stamp "$g: rows"
    python -u $J/lab_rows.py --pairs $pl $([ $se = - ] || echo --source_e $se) "$@" --out $OUT/rows/$g.jsonl
    s=$?
    st $s rows_$g
    [ $s -eq 0 ] || return
    stamp "$g: metrics"
    python -u $J/metrics.py --in_jsonl $OUT/rows/$g.jsonl --out_jsonl $OUT/metrics/$g.jsonl \
        --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --device cuda
    st $? metrics_$g
    rows_ok $OUT/metrics/$g.jsonl $([ $n = - ] || echo $n)
    st $? metrics_rows_$g
    stamp "$g: metrics, source speaker"
    python -u $J/metrics.py --in_jsonl $OUT/rows/${g}_src.jsonl --out_jsonl $OUT/metrics/${g}_src.jsonl \
        --metrics sim,secs --device cuda
    st $? metrics_src_$g
    rows_ok $OUT/metrics/${g}_src.jsonl
    st $? metrics_src_rows_$g
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
SE=$(dirname $VP)/vctk2k_source_E.json
case $CRT_AL in C2|C3|C4|C5) CAND=yes ;; *) CAND= ;; esac
# ------------------------------------------------------------------------------------------------------------------

LAST=$J/v1/t4/t4.done
[ "$BASELINES" = none ] && LAST=$J/v1/t3/t3.done
grep -q "selection sha256 $SELSHA" $LAST 2>/dev/null \
    || { echo "ERROR: $LAST missing or written under another selection.env: test conversions not all done"; exit 1; }
NV=$(grep -c . $VP)
NL=$(grep -c . $LP)
rm -f $OUT/m.done   # a re-run that fails must not leave the last run's marker

case $PART in
t1)
    T=$J/v1/t1
    natural $T/rc/vctk2k/30/manifest.jsonl $OUT/rows/natural_vctk2k_30.jsonl
    st $? natural
    score vctk2k_30 $VP $SE $NV --manifest $T/rc/vctk2k/30/manifest.jsonl --manifest $T/mc/vctk2k/30/manifest.jsonl \
        --manifest $T/anc/vctk2k/30/manifest.jsonl --manifest $OUT/rows/natural_vctk2k_30.jsonl \
        --lab $T/lab/front front $([ -n "$CAND" ] && echo --lab $T/lab/cand cand)
    ;;
t2)
    T=$J/v1/t2
    natural $T/rc/ltc2k/30/manifest.jsonl $OUT/rows/natural_ltc2k_30.jsonl
    st $? natural
    score ltc2k_30 $LP - $NL --manifest $T/rc/ltc2k/30/manifest.jsonl --manifest $T/mc/ltc2k/30/manifest.jsonl \
        --manifest $T/anc/ltc2k/30/manifest.jsonl --manifest $OUT/rows/natural_ltc2k_30.jsonl \
        --lab $T/lab/front front $([ -n "$CAND" ] && echo --lab $T/lab/cand cand)
    ;;
t3)
    T=$J/v1/t3
    for SET in vctk2k ltc2k; do
        if [ $SET = vctk2k ]; then PL=$VP; E=$SE; else PL=$LP; E=-; fi
        for B in 3 180; do
            score ${SET}_$B $PL $E - --manifest $T/rc/$SET/$B/manifest.jsonl \
                $([ $CRT_AL = C1 ] && echo --manifest $T/anc/$SET/$B/manifest.jsonl) \
                $([ -n "$CAND" ] && echo --lab $T/lab/cand_${SET}_$B cand)
        done
    done
    ;;
t4)
    T=$J/v1/t4/bl
    [ "$BASELINES" = none ] && { echo "  the record lists no baseline: nothing to score"; exit 0; }
    for SET in vctk2k ltc2k; do
        if [ $SET = vctk2k ]; then PL=$VP; E=$SE; N=$NV; else PL=$LP; E=-; N=$NL; fi
        for B in 30 3 180; do
            INS=
            for SYS in ${BASELINES//,/ }; do
                [ $B = 30 ] || [ $SYS = knnvc ] && INS="$INS --manifest $T/$SET/$B/$SYS/manifest.jsonl"
            done
            [ -z "$INS" ] || score ${SET}_$B $PL $E $N $INS
        done
    done
    ;;
esac
stamp "done"
du -sh $OUT /datastore/nvquoc/crt
echo " finished part $PART, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T') selection sha256 $SELSHA" > $OUT/m.done
[ -z "$FAILED" ]
