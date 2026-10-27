#!/bin/bash
# submit_r2_0.sh (ANALYSIS_PLAN_mvc_v1 §4.2): wait for R1's 200k EMA snapshot, then slot_wait + submit_ceai for R2.0.
M=/datastore/ceai/crt/mimi_vc/mvc
until [ -f /datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1/ema_0200000.pt ] && \
      ! squeue -h -u ceai -n mvc_r1 -o %T 2>/dev/null | grep -q .; do sleep 300; done
echo "$(date '+%F %T') R1 finished, ema_0200000.pt present"
bash $M/jobs/slot_wait.sh
cd /datastore/ceai/crt/mimi_vc/benchmark/journal && bash submit_ceai.sh $M/jobs/job_eval.slurm \
  --export=ALL,TAG=r2_0,BUDGETS=30,EVALS_FILE=$M/jobs/r2_0_evals.txt
