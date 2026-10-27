#!/bin/bash
#SBATCH --job-name=mvc_eval
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_eval_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Mimi-VC (paper 3): Dev quick-look (eval_dev.sh) of the items in EVALS ("NAME=CFGJSON;...", or the file EVALS_FILE),
# Dev selection SEL
# ("i/n" shard or "@file" pair ids; default @pairs/dev_s100.txt) at each budget in BUDGETS (comma list, default 30), results under
# $MVC/eval/$TAG. If jobs/after_$TAG.sh exists when the job starts, it is sourced after the evals (more probes in the
# same slot).
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/eval_dev.sh
FAILED=
[ -n "${EVALS:-}" ] || EVALS=$(cat ${EVALS_FILE:?set EVALS or EVALS_FILE})   # a file avoids quoting JSON through --export
[ -n "$EVALS" ] || { echo "empty EVALS; nothing to do"; exit 1; }
for b in $(echo ${BUDGETS:-30} | tr , " "); do
  eval_dev ${TAG:?set TAG} ${SEL:-@$WORK_DIR/mvc/pairs/dev_s100.txt} "$EVALS" $b
done
if [ -f $WORK_DIR/mvc/jobs/after_$TAG.sh ]; then echo "=== after_$TAG.sh: $(date)"; source $WORK_DIR/mvc/jobs/after_$TAG.sh; fi
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
