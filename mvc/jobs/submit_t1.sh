#!/bin/bash
# submit_t1.sh [conv [SET ...] | metrics [PART ...] | ecapa]: submits the T1 jobs of mvc/ANALYSIS_PLAN_mvc_v1.md in the
# order of §8.1, each after slot_wait.sh (the CRT/P1 split) through submit_ceai.sh (the account's 3 jobs, the 1 h gap,
# the 03:00-04:10 blackout). It starts at the stage named (default conv) and runs every later one:
#   conv     job_t1_conv.slurm with ONLY=<set> for each SET (default vctk2k ltc2k), then waits for $T/conv/conv.done;
#   metrics  job_t1_metrics.slurm with PART=<part> for each PART (default t1 t2 t3), then waits for the m.done of t1,
#            t2 and t3;
#   ecapa    job_t1_ecapa.slurm (ECAPA and guards); ECAPA_RESCORE=1 bash submit_t1.sh ecapa passes ECAPA_RESCORE=1.
# A wait ends when the done files exist, or stops the script (exit 1, nothing more submitted) once every job it waits
# for has left the queue without them; then resubmit what stopped by naming the stage, e.g. "submit_t1.sh conv ltc2k"
# or "submit_t1.sh metrics t2". The jobs check what they need again (conv.done and m.done under the record) and refuse
# otherwise. The script only tests that done files exist; it reads no metric file. It refuses the conv stage once
# conv.done or $T/m exists, and a part whose m.done exists (the jobs would refuse or redo them, after a 1 h gap).
# Tell the interpreter session before running it (§8.1). Run it in tmux on login01:
#   bash -l -c "bash /datastore/ceai/crt/mimi_vc/mvc/jobs/submit_t1.sh 2>&1 | tee -a /datastore/ceai/crt/mimi_vc/mvc/logs/submit_t1.log"
M=/datastore/ceai/crt/mimi_vc/mvc
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
T=/datastore/ceai/crt/mimi_vc/data_storage/mvc/test
POLL=${POLL:-300}
source /etc/profile.d/modules.sh 2>/dev/null
module load slurm >/dev/null 2>&1

say() { echo "$(date '+%F %T') submit_t1: $*"; }

submit() {  # submit <script> [sbatch args]: slot_wait.sh, then submit_ceai.sh; sets JID to the new job's id
  local out
  JID=
  bash $M/jobs/slot_wait.sh || return 1
  out=$(cd $J && bash submit_ceai.sh $M/jobs/$1 "${@:2}" | tee /dev/stderr)
  JID=$(awk -F'\t' '/^submitted: /{print $2}' <<< "$out")
  [ -n "$JID" ] || { say "ERROR: no job id from submit_ceai.sh for $*"; return 1; }
  say "$1 $* is job $JID"
}

wait_done() {  # wait_done "<done files>" <job id>...: 0 once every done file exists; 1 once every job has left the
               # queue while one is missing. A failed squeue is retried, never read as an empty queue
  local files=$1 f miss q j gone
  shift
  while :; do
    miss=
    for f in $files; do [ -e $f ] || miss="$miss $f"; done
    [ -z "$miss" ] && { say "done:$(for f in $files; do printf ' %s' $f; done)"; return 0; }
    if q=$(squeue -h -u ceai -o %i 2>/dev/null); then
      gone=1
      for j in "$@"; do grep -qx "$j" <<< "$q" && gone=0; done
      if [ $gone = 1 ]; then
        miss=
        for f in $files; do [ -e $f ] || miss="$miss $f"; done
        [ -z "$miss" ] && continue
        say "ERROR: jobs $* have left the queue without$miss; read their logs, then resubmit what stopped"
        return 1
      fi
    fi
    sleep $POLL
  done
}

STAGE=${1:-conv}
[ $# -gt 0 ] && shift
case $STAGE in
  conv|metrics|ecapa) ;;
  *) say "ERROR: the stage $STAGE is not conv, metrics or ecapa"; exit 1 ;;
esac
for x in "$@"; do
  case $STAGE:$x in
    conv:vctk2k|conv:ltc2k|metrics:t1|metrics:t2|metrics:t3) ;;
    *) say "ERROR: $x is not a set (conv: vctk2k, ltc2k) or a part (metrics: t1, t2, t3) of the stage $STAGE"; exit 1 ;;
  esac
done

if [ $STAGE = conv ]; then
  [ ! -e $T/conv/conv.done ] && [ ! -e $T/m ] || { say "ERROR: $T/conv/conv.done or $T/m exists: no conversion"; exit 1; }
  IDS=
  for SET in ${@:-vctk2k ltc2k}; do
    submit job_t1_conv.slurm --export=ALL,ONLY=$SET || exit 1
    IDS="$IDS $JID"
  done
  wait_done $T/conv/conv.done $IDS || exit 1
  STAGE=metrics
  set --
fi
if [ $STAGE = metrics ]; then
  for PART in ${@:-t1 t2 t3}; do
    [ ! -e $T/m/$PART/m.done ] || { say "ERROR: $T/m/$PART/m.done exists: the part is done; nothing submitted"; exit 1; }
  done
  IDS=
  for PART in ${@:-t1 t2 t3}; do
    submit job_t1_metrics.slurm --export=ALL,PART=$PART || exit 1
    IDS="$IDS $JID"
  done
  wait_done "$T/m/t1/m.done $T/m/t2/m.done $T/m/t3/m.done" $IDS || exit 1
fi
submit job_t1_ecapa.slurm --export=ALL${ECAPA_RESCORE:+,ECAPA_RESCORE=$ECAPA_RESCORE} || exit 1
say "all T1 jobs submitted; DNSMOS runs on the Mac after the copy (§3.7)"
