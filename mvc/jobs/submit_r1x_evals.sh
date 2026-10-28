#!/bin/bash
# submit_r1x_evals.sh [K ...] (default 250 300): the exploratory R1x evaluation of ANALYSIS_PLAN_mvc_v1 §4.10. For each
# K it waits until R1x has written $MVC/runs/r1x/ema_0<K>000.pt (unchanged in size for 2 minutes), then submits
# job_eval.slurm on Dev-S100 at 30 s with TAG=r1x<K> and jobs/r1x<K>_evals.txt (the reference setting and mvc_v1's
# setting on that snapshot; after_r1x<K>.sh adds ECAPA and the guards), through slot_wait.sh and submit_ceai.sh.
# Run it in tmux on login01: bash -l -c "bash .../submit_r1x_evals.sh 2>&1 | tee -a .../logs/submit_r1x_evals.log"
M=/datastore/ceai/crt/mimi_vc/mvc
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
R=/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x
for K in ${@:-250 300}; do
  f=$R/ema_0${K}000.pt
  until [ -f $f ] && s=$(stat -c %s $f) && sleep 120 && [ "$s" = "$(stat -c %s $f)" ]; do sleep 300; done
  echo "$(date '+%F %T') submit_r1x_evals: $f is there ($(stat -c %s $f) B)"
  bash $M/jobs/slot_wait.sh
  (cd $J && bash submit_ceai.sh $M/jobs/job_eval.slurm --export=ALL,TAG=r1x$K,BUDGETS=30,EVALS_FILE=$M/jobs/r1x${K}_evals.txt)
done
echo "$(date '+%F %T') submit_r1x_evals done"
