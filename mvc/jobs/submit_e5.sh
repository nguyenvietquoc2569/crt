#!/bin/bash
# submit_e5.sh (27 Oct): wait for R1 150k EMA snapshot, then slot_wait + submit_ceai (1 h gap, blackout) for E5.
M=/datastore/ceai/crt/mimi_vc/mvc
until [ -f /datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1/ema_0150000.pt ]; do sleep 300; done
echo "$(date '+%F %T') ema_0150000.pt present"
sleep 120                                         # let the atomic save settle
bash $M/jobs/slot_wait.sh
cd /datastore/ceai/crt/mimi_vc/benchmark/journal && bash submit_ceai.sh $M/jobs/job_eval.slurm \
  --export=ALL,TAG=e5,BUDGETS=30,EVALS_FILE=$M/jobs/e5_evals.txt
