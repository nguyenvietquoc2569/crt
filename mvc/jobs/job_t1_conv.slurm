#!/bin/bash
#SBATCH --job-name=mvc_t1_conv
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_t1_conv_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_t1_conv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# T1 conversions of mvc/ANALYSIS_PLAN_mvc_v1.md (§2, §5.4, §8.1-8.3), only after the selection record is committed and
# pushed (§8.4 step 6) and copied to $C with mvc_selection.env, MVC_SELECTION_RECORD_v1_cfg/ (t1_lib.sh) and
# DEVIATIONS_mvc_v1.md, whose line on the push of the record (tag mvc-selection-record-v1) must be filled in.
# It refuses to start when that line is missing or still "(to be filled in)" (§8.3 first bullet), when the env lacks a
# value or a value does not match its file (the record, the three cfg JSONs, the snapshot, the describe() commit of
# $C/*.py), when a frozen ceai file (FREEZE_mvc_v1.sha256 Section 2) differs, when $T/conv/conv.done exists (every pair
# has been attempted: no more conversion, §5.4) or when $T/m exists (no conversion once a metric may exist, §5.4). It
# installs the three cfg JSONs into $T/cfg and runs run_mvc.py into $T/conv/<set>/<budget>/<name>/: mvc_v1 at 30, 3
# and 180 s, mvc_v1_q32 and mvc_copy_latent at 30 s, on VCTK-2k and LTC-2k, or on one set with ONLY=vctk2k / ONLY=ltc2k
# (two submissions on two slots). run_mvc.py keeps every successful conversion and retries error rows, so a job that
# stopped before every pair was attempted is resubmitted unchanged (§5.4).
# conv.done ("job <id> <date> selection sha256 <RECORD_SHA256>") is written as soon as every manifest of both sets has
# one row per pair under the record's commit, settings and snapshot, with shortfall rows exactly on the 101 LTC-2k
# pairs at 180 s and nowhere else (§8.2). Other error rows are failed conversions (§5.4): counted, never retried after
# conv.done. With the ONLY split, the job that ends second writes it. Prints counts only.
#   submit_t1.sh submits it (slot_wait.sh, then submit_ceai.sh with --export=ALL,ONLY=vctk2k and ONLY=ltc2k)
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/t1_lib.sh
case ${ONLY:-} in
  ''|vctk2k|ltc2k) ;;
  *) echo "ERROR: ONLY=$ONLY is neither vctk2k nor ltc2k"; exit 1 ;;
esac
[ ! -e $T/conv/conv.done ] || { echo "ERROR: $T/conv/conv.done exists: every pair has been attempted (§5.4)"; exit 1; }
[ ! -e $T/m ] || { echo "ERROR: $T/m exists: no T1 conversion once a metric may exist (§5.4)"; exit 1; }
t1_pushed || { echo "ERROR: the push of the selection record is not on record: nothing converted (§8.3)"; exit 1; }
t1_selection $RECORD_CFG $T/cfg || { echo "ERROR: selection or frozen-file check failed: nothing converted"; exit 1; }
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
mkdir -p $T/conv
TIMES=$T/conv/timings_$SLURM_JOB_ID.tsv

for SET in ${ONLY:-$T1_SETS}; do
  t1_convert $SET
done
stamp "checks, both sets"
t1_conv_check
S=$?
stamp "done"
du -sh $T/conv
COMPLETE=no
[ $S = 0 ] && COMPLETE=yes
echo " finished, run_mvc.py exits non-zero:${FAILED:- none}; every manifest of both sets has a row per pair: $COMPLETE: $(date)"
[ $S = 0 ] && t1_done_line > $T/conv/conv.done
[ $S = 0 ]
