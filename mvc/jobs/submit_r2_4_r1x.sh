#!/bin/bash
# submit_r2_4_r1x.sh (28 Oct): R2.4 (the chosen configuration on full Dev at 3 s and 180 s),
# then R1x, one after the other through slot_wait + submit_ceai (1 h gap, 03:00-04:10 blackout).
M=/datastore/ceai/crt/mimi_vc/mvc
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
bash $M/jobs/slot_wait.sh
(cd $J && BUDGETS=3,180 bash submit_ceai.sh $M/jobs/job_eval.slurm --export=ALL,TAG=r2_4,SEL=0/1,EVALS_FILE=$M/jobs/r2_4_evals.txt)   # BUDGETS via the environment: --export splits at commas
bash $M/jobs/slot_wait.sh
(cd $J && bash submit_ceai.sh $M/jobs/job_r1x.slurm --export=ALL)
echo "$(date '+%F %T') submit_r2_4_r1x done"
