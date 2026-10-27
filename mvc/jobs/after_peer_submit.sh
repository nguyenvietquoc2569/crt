#!/bin/bash
# after_peer_submit.sh <not-before HH:MM> <give-up HH:MM>: waits until another ceai job has been submitted after the
# not-before time (the peer session that queued first goes first), or until the give-up time, so two submit loops
# never race at the end of a blackout. Then returns; submit_ceai.sh enforces the 1 h gap itself.
source /etc/profile.d/modules.sh 2>/dev/null
module load slurm >/dev/null 2>&1
NB=$(date -d "$1" +%s); GU=$(date -d "$2" +%s)
while true; do
  last=$(sacct -u ceai -S "$(date -d @$NB +%FT%T)" -X -n -o Submit 2>/dev/null | sort | tail -1 | tr -d " ")
  [ -n "$last" ] && { echo "$(date "+%F %T") after_peer_submit: a job was submitted at $last"; exit 0; }
  [ $(date +%s) -ge $GU ] && { echo "$(date "+%F %T") after_peer_submit: nothing submitted since $1, going ahead"; exit 0; }
  sleep 120
done
