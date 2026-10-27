#!/bin/bash
# slot_wait.sh: returns once CRT may submit one more job on ceai under the agreed split. Since the Qwen server moved
# to nvquoc (25 Oct 2026) the split is CRT <= 2 running or pending and P1 <= 1 (interpreter session, same day). On
# 27 Oct the user let CRT use all 3 slots while P1 is idle: CRT_MAX (default 3) applies only while no P1 job is
# running or pending; with a P1 job in the queue CRT keeps <= 2;
# submit_ceai.sh still enforces the account's 3 jobs and the 1 h gap. CRT jobs are named mvc_* or crt_*; anything
# else except qwen-serve is counted as P1. Polls every 300 s.
source /etc/profile.d/modules.sh 2>/dev/null
module load slurm >/dev/null 2>&1
while true; do
  names=$(squeue -h -u ceai -t RUNNING,PENDING -o %j 2>/dev/null) || { sleep 300; continue; }
  crt=$(echo "$names" | grep -cE '^(mvc_|crt_)')
  other=$(echo "$names" | grep -vE '^(mvc_|crt_|qwen-serve)' | grep -c .)
  max=${CRT_MAX:-3}; [ "$other" -gt 0 ] && max=2
  if [ "$crt" -lt "$max" ]; then
    echo "$(date '+%F %T') slot_wait: CRT $crt, P1 $other -> may submit"; exit 0
  fi
  echo "$(date '+%F %T') slot_wait: CRT $crt, P1 $other -> waiting"
  sleep 300
done
