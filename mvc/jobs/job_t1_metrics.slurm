#!/bin/bash
#SBATCH --job-name=mvc_t1_m
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_t1_m_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_t1_m_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# T1 metrics of mvc/ANALYSIS_PLAN_mvc_v1.md (§3.1, §8.2, §8.3), one part per submission, PART=t1|t2|t3, into $T/m/<part>:
#   t1  VCTK-2k 30 s: mvc_v1, mvc_v1_q32, mvc_copy_latent
#   t2  LTC-2k 30 s: the same three
#   t3  mvc_v1 at 3 s and 180 s on VCTK-2k and LTC-2k
# It refuses to start without $T/conv/conv.done written under the same record (no test metric before every T1
# conversion exists, §8.3), when the selection or frozen-file check fails, when a conversion manifest no longer passes
# its check, when job M's parts t1, t2, t4 or X2 are not done under the journal selection, or when X2's speechbrain
# site folder does not import. Then the scorer drift checks of §8.3, before any MimiFlow-VC test row is scored (the
# order of §8.1), into $T/drift/ under names that a requeue does not reuse: the ECAPA smoke test and the ECAPA check
# (X2's first 32 VCTK-2k rows re-scored, |Δ sim_ecapa| <= 0.001), then the WavLM-SV, UTMOS and Whisper check (job M's
# command on 32 rows of job M); if one fails, nothing is scored. Per set and budget: lab_rows.py ->
# rows/<set>_<budget>{,_src}.jsonl, metrics.py with job M's command (--metrics mcd,wer,sim,secs,utmos --whisper_models
# large-v3,base) on the target rows and --metrics sim,secs on the @src rows, --wavlm_model on both -> metrics/, then the
# row checks (shortfall rows only on the 101 LTC-2k pairs at 180 s). metrics.py resumes, so a part that hits its limit
# is resubmitted unchanged. m.done ("job <id> <date> selection sha256 <RECORD_SHA256>") is written only when every step
# and check passed. Prints row counts, never a metric value; summarize.py is never run on these rows (§8.3).
#   submit_t1.sh submits it (slot_wait.sh, then submit_ceai.sh with --export=ALL,PART=t1, t2, t3)
case ${PART:-} in
  t1|t2|t3) ;;
  *) echo "ERROR: PART=${PART:-} is not t1, t2 or t3 (--export=ALL,PART=t1)"; exit 1 ;;
esac
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/t1_lib.sh
t1_selection $T/cfg || { echo "ERROR: selection or frozen-file check failed: nothing scored"; exit 1; }
t1_done_ok $T/conv/conv.done \
  || { echo "ERROR: $T/conv/conv.done missing or written under another record: T1 conversions not all done"; exit 1; }
t1_conv_check || { echo "ERROR: a T1 manifest no longer passes its check: nothing scored"; exit 1; }
t1_journal_done || { echo "ERROR: nothing scored"; exit 1; }
t1_ecapa_env || { echo "ERROR: X2's speechbrain site folder does not import: nothing scored"; exit 1; }
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
OUT=$T/m/$PART
mkdir -p $OUT/rows $OUT/metrics   # metrics.py appends to its output and creates no folder
TIMES=$OUT/timings.tsv
: > $TIMES
rm -f $OUT/m.done   # a re-run that fails must not leave the last run's marker
ECAPA_SAVEDIR=$T/drift/ecapa_model_$PART   # the three parts may run at once

stamp "ECAPA smoke"
t1_score_ecapa --smoke || { echo "ERROR: the ECAPA smoke test failed: nothing scored"; exit 1; }
stamp "ECAPA drift check"
t1_drift_ecapa $PART || { echo "ERROR: the ECAPA drift check failed: nothing scored; find the cause and log it as a" \
                               "deviation (§8.3)"; exit 1; }
stamp "scorer drift check"
t1_drift_m $PART || { echo "ERROR: the scorer drift check failed: nothing scored; find the cause and log it as a" \
                           "deviation (§8.3)"; exit 1; }
case $PART in
  t1) t1_score $OUT vctk2k 30 $T1_SYSTEMS ;;
  t2) t1_score $OUT ltc2k 30 $T1_SYSTEMS ;;
  t3) for SET in $T1_SETS; do
        for B in 3 180; do
          t1_score $OUT $SET $B mvc_v1
        done
      done ;;
esac
stamp "done"
du -sh $OUT
echo " finished part $PART, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && t1_done_line > $OUT/m.done
[ -z "$FAILED" ]
