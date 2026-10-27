#!/bin/bash
# submit_r2_3_r1x.sh (28 Oct): R2.3's full-Dev runs (the finalists and mvc_r1_nosng, SEL=0/1 = all 1,000 Dev pairs),
# then R1x, one after the other through slot_wait + submit_ceai (1 h gap, 03:00-04:10 blackout).
M=/datastore/ceai/crt/mimi_vc/mvc
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
bash $M/jobs/slot_wait.sh
(cd $J && bash submit_ceai.sh $M/jobs/job_eval.slurm --export=ALL,TAG=r2_3,SEL=0/1,BUDGETS=30,EVALS_FILE=$M/jobs/r2_3_evals.txt)
bash $M/jobs/slot_wait.sh
(cd $J && bash submit_ceai.sh $M/jobs/job_r1x.slurm --export=ALL)
echo "$(date '+%F %T') submit_r2_3_r1x done"
