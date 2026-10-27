#!/bin/bash
# submit_after.sh <job id> <script.slurm> [sbatch args]: waits until submit_ceai.log records a submission with a job id
# above <job id> (someone else's turn first), then slot_wait.sh (the agreed CRT/P1 split), then submits <script.slurm>
# through submit_ceai.sh (slots, the 1 h gap, the blackout). Run in tmux on login01: bash -l -c "bash submit_after.sh ...".
LOG=/datastore/ceai/crt/_transfer_logs/submit_ceai.log
MIN=$1 SCRIPT=$2
shift 2
until awk -F'\t' -v m="$MIN" '$2 + 0 > m' "$LOG" | grep -q .; do sleep 120; done
echo "$(date '+%F %T') submit_after: a job above $MIN was submitted"
bash /datastore/ceai/crt/mimi_vc/mvc/jobs/slot_wait.sh
cd /datastore/ceai/crt/mimi_vc/benchmark/journal && bash submit_ceai.sh "$SCRIPT" "$@"
