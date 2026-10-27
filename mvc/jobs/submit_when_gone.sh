#!/bin/bash
# submit_when_gone.sh <job id> <script.slurm> [sbatch args]: waits until <job id> has left the queue (any end state),
# then slot_wait.sh and submit_ceai.sh. Run in tmux on login01: bash -l -c "bash submit_when_gone.sh ...".
JOB=$1 SCRIPT=$2
shift 2
source /etc/profile.d/modules.sh 2>/dev/null
module load slurm >/dev/null 2>&1
while squeue -h -j "$JOB" -o %T 2>/dev/null | grep -q .; do sleep 300; done
echo "$(date '+%F %T') submit_when_gone: job $JOB has left the queue"
bash /datastore/ceai/crt/mimi_vc/mvc/jobs/slot_wait.sh
cd /datastore/ceai/crt/mimi_vc/benchmark/journal && bash submit_ceai.sh "$SCRIPT" "$@"
