#!/bin/bash
# submit_r2_1_r1x.sh (28 Oct, after R2.0 chose ema_0200000.pt and triggered R1x): R2.1's screen on the 200k snapshot,
# then R1x (ANALYSIS_PLAN_mvc_v1 §4.10), one after the other through slot_wait + submit_ceai (1 h gap, blackout).
M=/datastore/ceai/crt/mimi_vc/mvc
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
bash $M/jobs/slot_wait.sh
(cd $J && bash submit_ceai.sh $M/jobs/job_eval.slurm --export=ALL,TAG=r2_1,BUDGETS=30,EVALS_FILE=$M/jobs/r2_1_k200_evals.txt)
bash $M/jobs/slot_wait.sh
(cd $J && bash submit_ceai.sh $M/jobs/job_r1x.slurm --export=ALL)
echo "$(date '+%F %T') submit_r2_1_r1x done"
