#!/bin/bash
# chain_submit.sh <after-job-name> <newer-than-job-id> <script.slurm> [--export=...]: waits until the FIRST log
# $LOGS/<after-job-name>_<id>.out with id > <newer-than-job-id> ends with "finished rc=0", then submits <script.slurm>
# through submit_ceai.sh (which enforces the account's slot, gap and blackout rules), after slot_wait.sh (the agreed
# CRT/P1 split). It stops, submitting nothing,
# when that log ends with a non-zero rc. Run in tmux on the login node: bash -l -c "bash chain_submit.sh ...".
LOGS=/datastore/ceai/crt/mimi_vc/mvc/logs
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
AFTER=$1 MIN=$2 SCRIPT=$3
shift 3
log() { echo "$(date '+%F %T') chain $SCRIPT: $*"; }
while true; do
  f=$(ls $LOGS/${AFTER}_*.out 2>/dev/null | awk -F_ -v m=$MIN '{id=$NF; sub(/\.out$/,"",id); if (id+0 > m) print id+0, $0}' \
      | sort -n | head -1 | cut -d' ' -f2)
  if [ -n "$f" ] && grep -q "^finished rc=" $f; then
    if grep -q "^finished rc=0" $f; then log "$f finished OK, submitting"; break; fi
    log "$f failed: $(grep '^finished rc=' $f); nothing submitted"; exit 1
  fi
  sleep 120
done
bash /datastore/ceai/crt/mimi_vc/mvc/jobs/slot_wait.sh
cd $J && bash submit_ceai.sh $SCRIPT "$@"
