#!/bin/bash
#SBATCH --job-name=mvc_xmvc1
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_xmvc1_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_xmvc1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# X-mvc-1 (mvc/ANALYSIS_PLAN_mvc_v1.md §4.10; DEVIATIONS_mvc_v1.md operational note 4), exploratory and descriptive:
# R1x's 300k EMA snapshot $MVC/runs/r1x/ema_0300000.pt as system mvc_r1x300, on one set per submission (ONLY=vctk2k
# or ONLY=ltc2k, two submissions on two slots) at the 30 s budget only, with mvc_v1's frozen inference settings: the
# record's mvc_v1.json with only run_dir and ckpt changed. It writes only under X=$MVC/xmvc1, never under the T1 tree
# $MVC/test or the journal's $J/v1, and runs the T1 commands of t1_lib.sh (whose $T is pointed at $X here).
# It refuses to start unless T1 is done under the record (the m.done of $MVC/test/m/t1, t2, t3, x2/x2.done and
# guard/guard.done, §4.10 "after T1 of mvc_v1 is complete"), when the selection or frozen-file check fails (the record,
# its cfg JSONs, the describe() commit of $C/*.py, the frozen ceai files), when the snapshot is missing, when
# $X/<set>.done exists, when job M's parts t1, t2, t4 or X2 are not done under the journal selection (the drift checks
# read their rows) or when X2's speechbrain site folder does not import. Then, in this order:
#   - $X/cfg/mvc_r1x300.json from the record's mvc_v1.json (xmvc1_check.py cfg: every other key unchanged) and
#     $X/snapshot.sha256, the sha256 of the two files run_mvc.py loads from run_dir: the snapshot and stats.npz, which
#     must be R1's frozen stats.npz (FREEZE_mvc_v1.sha256); on a resubmission neither may have changed;
#   - run_mvc.py at 30 s into $X/conv/<set>/30/mvc_r1x300/, then the row check: one row per pair under the record's
#     describe() commit, the cfg JSON and the snapshot. If it fails, nothing is scored, and a resubmission runs
#     run_mvc.py again, which keeps every successful conversion and retries error rows. Once it passes (every pair has
#     been attempted), $X/conv/<set>.conv.done ("job <id> <date> snapshot sha256 <sha256> selection sha256
#     <RECORD_SHA256>") is written; from then on run_mvc.py never runs again on the set, the check runs again, and the
#     error rows that remain are failed conversions (§5.4), counted as in T1. With $X/m/<part> present and no
#     conv.done the job refuses (no conversion once a metric may exist, §5.4);
#   - the drift checks of §8.3 into $X/drift/: the ECAPA smoke test and the ECAPA check (X2's first 32 VCTK-2k rows
#     re-scored, |Δ sim_ecapa| <= 0.001), then the WavLM-SV, UTMOS and Whisper check (job M's command on 32 rows of
#     job M); if one fails, nothing is scored;
#   - lab_rows.py (VCTK-2k with the source-E file) and metrics.py with the T1 commands (t1_score, --wavlm_model on
#     every call) into $X/m/<part>/{rows,metrics}/<set>_30{,_src}.jsonl, part t1 = VCTK-2k and t2 = LTC-2k as in T1;
#   - score_ecapa.py on the target and @src rows (X2's site folder and model, its own savedir) into
#     $X/x2/ecapa/<set>_30{,_src}.jsonl;
#   - acoustic_guard.py on the 30 s rows into $X/guard/<set>_30.jsonl; its printed means go to /dev/null and the
#     summary file of means it writes beside its output (in $X/tmp/guard_<set>/) is deleted unread (§8.3).
# When every step and row check passed, $X/<set>.sha256 (xmvc1_check.py files: the sha256 of the cfg, snapshot.sha256,
# conv.done and the set's rows, metrics, ECAPA and guard files, paths relative to $X, for the copy to the Mac) and then
# $X/<set>.done ("job <id> <date> snapshot sha256 <sha256> selection sha256 <RECORD_SHA256>") are written.
# run_mvc.py (before conv.done), metrics.py and score_ecapa.py resume and acoustic_guard.py reruns whole, so a job that
# stopped is resubmitted unchanged. score_ecapa.py never re-scores a row already in its output: when the ECAPA
# check reports error rows on converted rows, resubmit with ECAPA_RESCORE=1, which deletes the set's two X-mvc-1 ECAPA
# files unread, and log that in DEVIATIONS_mvc_v1.md as an operational change. Prints counts, never a metric value;
# summarize.py is never run on these rows.
#   submit_xmvc1.sh submits it (slot_wait.sh, then submit_ceai.sh with --export=ALL,ONLY=vctk2k and ONLY=ltc2k)
case ${ONLY:-} in
  vctk2k) PART=t1 ;;
  ltc2k) PART=t2 ;;
  *) echo "ERROR: ONLY=${ONLY:-} is not vctk2k or ltc2k (--export=ALL,ONLY=vctk2k)"; exit 1 ;;
esac
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/t1_lib.sh
SET=$ONLY
NAME=mvc_r1x300
SNAP_DIR=$MVC/runs/r1x
SNAP=$SNAP_DIR/ema_0300000.pt
T1_TREE=$T                        # the T1 tree of §8.2: only its done files are read
X=$MVC/xmvc1
T=$X                              # t1_lib.sh's helpers (t1_score, the drift checks) write under $T: here $X
ECAPA_SAVEDIR=$X/x2/ecapa_model_$SET   # the two sets may run at once
OUT=$X/m/$PART
X_CHECK=$C/jobs/xmvc1_check.py
CONV_DONE=$X/conv/$SET.conv.done
t1_selection $RECORD_CFG || { echo "ERROR: selection or frozen-file check failed: nothing run"; exit 1; }
for f in m/t1/m.done m/t2/m.done m/t3/m.done x2/x2.done guard/guard.done; do
  t1_done_ok $T1_TREE/$f || { echo "ERROR: $T1_TREE/$f missing or written under another record: T1 of mvc_v1 is" \
                                   "not done (§4.10): nothing run"; exit 1; }
done
[ -f $SNAP ] || { echo "ERROR: the snapshot $SNAP is missing: nothing run"; exit 1; }
[ ! -e $X/$SET.done ] || { echo "ERROR: $X/$SET.done exists: X-mvc-1 on $SET is done; nothing run"; exit 1; }
t1_journal_done || { echo "ERROR: nothing run"; exit 1; }
t1_ecapa_env || { echo "ERROR: X2's speechbrain site folder does not import: nothing run"; exit 1; }
case ${ECAPA_RESCORE:-} in
  '') ;;
  1) for f in $X/x2/ecapa/${SET}_30{,_src}.jsonl; do
       [ ! -e $f ] || { rm -f $f && echo " ECAPA_RESCORE=1: deleted $f unread (log it as an operational change)"; }
     done ;;
  *) echo "ERROR: ECAPA_RESCORE=$ECAPA_RESCORE is not 1"; exit 1 ;;
esac
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
mkdir -p $X/cfg
TIMES=$X/timings_${SET}_$SLURM_JOB_ID.tsv

stamp "cfg and snapshot"
python -u $X_CHECK cfg --src $RECORD_CFG/$CFG_MVC_V1 --run_dir $SNAP_DIR --ckpt $(basename $SNAP) \
    --out $X/cfg/$NAME.json || { echo "ERROR: the X-mvc-1 cfg is not mvc_v1.json with run_dir and ckpt changed"; exit 1; }
python -u $X_CHECK snapshot --snapshot $SNAP --stats $SNAP_DIR/stats.npz --ref_stats $(dirname $SNAPSHOT)/stats.npz \
    --freeze $C/FREEZE_mvc_v1.sha256 --out $X/snapshot.sha256 || { echo "ERROR: nothing converted"; exit 1; }
SNAP_SHA=$(head -n 1 $X/snapshot.sha256 | cut -d' ' -f1)
x_done_line() { echo "job $SLURM_JOB_ID $(date '+%F %T') snapshot sha256 $SNAP_SHA selection sha256 $RECORD_SHA256"; }
if [ -e $CONV_DONE ]; then
  grep -q "snapshot sha256 $SNAP_SHA selection sha256 $RECORD_SHA256\$" $CONV_DONE \
    || { echo "ERROR: $CONV_DONE was written under another snapshot or record: nothing run"; exit 1; }
  echo " $CONV_DONE exists: every pair has been attempted, no more conversion (§5.4); the manifest is checked again"
else
  [ ! -e $OUT ] || { echo "ERROR: $OUT exists without $CONV_DONE: no conversion once a metric may exist (§5.4)"; exit 1; }
  stamp "$SET 30 s: $NAME"
  python -u $C/run_mvc.py --journal $J --pairs $J/pairs/$SET.jsonl --set $SET --budget 30 --name $NAME \
      --cfg "$(cat $X/cfg/$NAME.json)" --out_dir $X/conv --device cuda
  echo "  run_mvc.py exit $?"
fi
stamp "conversion check"
python -u $X_CHECK conv --manifest $X/conv/$SET/30/$NAME/manifest.jsonl --pairs $J/pairs/$SET.jsonl --set $SET \
    --name $NAME --cfg $X/cfg/$NAME.json --snapshot_sha $X/snapshot.sha256 --commit $DESCRIBE_COMMIT \
  || { echo "ERROR: the manifest does not pass its check: nothing scored; resubmit unchanged while $CONV_DONE does" \
            "not exist (run_mvc.py retries error rows until every pair has been attempted, §5.4)"; exit 1; }
[ -e $CONV_DONE ] || x_done_line > $CONV_DONE
stamp "ECAPA smoke"
t1_score_ecapa --smoke || { echo "ERROR: the ECAPA smoke test failed: nothing scored"; exit 1; }
stamp "ECAPA drift check"
t1_drift_ecapa x_$SET || { echo "ERROR: the ECAPA drift check failed: nothing scored; find the cause and log it as" \
                                "a deviation (§8.3)"; exit 1; }
stamp "scorer drift check"
t1_drift_m x_$SET || { echo "ERROR: the scorer drift check failed: nothing scored; find the cause and log it as a" \
                            "deviation (§8.3)"; exit 1; }
mkdir -p $OUT/rows $OUT/metrics   # metrics.py appends to its output and creates no folder
t1_score $OUT $SET 30 $NAME

mkdir -p $X/x2/ecapa
for k in '' _src; do
  stamp "ECAPA ${SET}_30$k"
  t1_score_ecapa --systems $NAME --rows $OUT/rows/${SET}_30$k.jsonl --out $X/x2/ecapa/${SET}_30$k.jsonl
  echo "  score_ecapa.py exit $?"
done
python -u $T1_CHECK ecapa --rows_dir $OUT/rows --ecapa_dir $X/x2/ecapa --g ${SET}_30 --pairs $J/pairs/$SET.jsonl \
    --systems $NAME
st $? ecapa_$SET

stamp "guards $SET"
GIN=$X/guard/input/${SET}_30.jsonl
GTMP=$X/tmp/guard_$SET            # outside $X/guard, so the other set's guard check never sees its summary file
python -u $T1_CHECK guard_rows --out $GIN --src $OUT/rows/${SET}_30.jsonl=$NAME
S=$?
st $S guard_input_$SET
if [ $S -eq 0 ]; then
  rm -rf $GTMP $X/guard/${SET}_30.jsonl           # a stopped or earlier run's output, unread
  python -u $C/acoustic_guard.py --rows $GIN --out $GTMP/${SET}_30.jsonl --workers ${SLURM_CPUS_PER_TASK:-8} > /dev/null
  S=$?
  rm -f $GTMP/${SET}_30.summary.json              # acoustic_guard.py's means: deleted unread (§8.3)
  [ $S -ne 0 ] || mv $GTMP/${SET}_30.jsonl $X/guard/${SET}_30.jsonl || S=1
  rm -rf $GTMP
  st $S guard_$SET
  python -u $T1_CHECK guard --guard $X/guard/${SET}_30.jsonl --pairs $J/pairs/$SET.jsonl --systems $NAME
  st $? guard_rows_$SET
fi
stamp "done"
du -sh $X
if [ -z "$FAILED" ]; then
  python -u $X_CHECK files --root $X --out $X/$SET.sha256 cfg/$NAME.json snapshot.sha256 conv/$SET.conv.done \
      m/$PART/{rows,metrics}/${SET}_30{,_src}.jsonl x2/ecapa/${SET}_30{,_src}.jsonl guard/${SET}_30.jsonl
  st $? sha256_$SET
fi
echo " finished X-mvc-1 on $SET, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && x_done_line > $X/$SET.done
[ -z "$FAILED" ]
