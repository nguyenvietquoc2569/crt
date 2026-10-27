#!/bin/bash
#SBATCH --job-name=mvc_t1_x2
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_t1_x2_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_t1_x2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# T1 ECAPA (mvc/ANALYSIS_PLAN_mvc_v1.md §3.5, §6.1) and acoustic guards (§3.8, §6.3), after the three metrics parts
# ($T/m/t1, t2, t3 m.done under the same record; the order of §8.1), each of which ran both drift checks of §8.3 before
# scoring. It refuses to start when the selection or frozen-file check fails, when job M's parts t1, t2, t4 or X2 are
# not done under the journal selection, or when X2's speechbrain site folder does not import.
# ECAPA: X2's site folder, model and script; the smoke test, then the ECAPA drift check of §8.3 again on this node (X2's
# first 32 VCTK-2k rows re-scored into $T/drift/, |Δ sim_ecapa| <= 0.001; if it fails, nothing is scored), then
# score_ecapa.py on the 30 s target and @src rows of mvc_v1, mvc_v1_q32 and mvc_copy_latent (rows of parts t1 and t2)
# -> $T/x2/ecapa/<set>_30{,_src}.jsonl, and $T/x2/x2.done when their checks pass.
# Guards: acoustic_guard.py on the 30 s rows of mvc_v1, mvc_v1_q32, cand_pm18_rq and the four baselines (job M's rows,
# read only) -> $T/guard/<set>_30.jsonl, and $T/guard/guard.done when every system has a row per pair; its printed
# means go to /dev/null and its summary file of means is deleted unread (§8.3). score_ecapa.py resumes and
# acoustic_guard.py reruns whole, so a job that stopped is resubmitted unchanged. score_ecapa.py never re-scores a row
# already in its output, error rows included: when the check reports error rows on converted rows (a transient CUDA
# error, say), resubmit with ECAPA_RESCORE=1, which deletes the four T1 ECAPA files unread and scores them again
# (refused once x2.done exists), and log that in DEVIATIONS_mvc_v1.md as an operational change. Prints counts only.
#   submit_t1.sh submits it (slot_wait.sh, then submit_ceai.sh with --export=ALL[,ECAPA_RESCORE=1])
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/t1_lib.sh
t1_selection $T/cfg || { echo "ERROR: selection or frozen-file check failed: nothing scored"; exit 1; }
for p in t1 t2 t3; do
  t1_done_ok $T/m/$p/m.done || { echo "ERROR: $T/m/$p/m.done missing or written under another record"; exit 1; }
done
t1_journal_done || { echo "ERROR: nothing scored"; exit 1; }
t1_ecapa_env || { echo "ERROR: X2's speechbrain site folder does not import: nothing scored"; exit 1; }
case ${ECAPA_RESCORE:-} in
  '') ;;
  1) [ ! -e $T/x2/x2.done ] || { echo "ERROR: ECAPA_RESCORE=1 but $T/x2/x2.done exists: nothing deleted"; exit 1; }
     for f in $T/x2/ecapa/{vctk2k,ltc2k}_30{,_src}.jsonl; do
       [ ! -e $f ] || { rm -f $f && echo " ECAPA_RESCORE=1: deleted $f unread (log it as an operational change)"; }
     done ;;
  *) echo "ERROR: ECAPA_RESCORE=$ECAPA_RESCORE is not 1"; exit 1 ;;
esac
mkdir -p $T/x2/ecapa $T/guard/input
TIMES=$T/x2/timings.tsv
: > $TIMES
rm -f $T/x2/x2.done $T/guard/guard.done   # a re-run that fails must not leave the last run's markers

stamp "ECAPA smoke"
t1_score_ecapa --smoke || { echo "ERROR: the ECAPA smoke test failed: nothing scored"; exit 1; }
stamp "ECAPA drift check"
t1_drift_ecapa x2 || { echo "ERROR: the ECAPA drift check failed: nothing scored; find the cause and log it as a" \
                            "deviation (§8.3)"; exit 1; }
t1_ecapa vctk2k t1
t1_ecapa ltc2k t2
X2_FAILED=$FAILED
[ -z "$X2_FAILED" ] && t1_done_line > $T/x2/x2.done
FAILED=
t1_guard vctk2k t1
t1_guard ltc2k t2
[ -z "$FAILED" ] && t1_done_line > $T/guard/guard.done
stamp "done"
du -sh $T/x2 $T/guard
echo " finished, ECAPA failed steps:${X2_FAILED:- none}; guard failed steps:${FAILED:- none}: $(date)"
[ -z "$X2_FAILED$FAILED" ]
