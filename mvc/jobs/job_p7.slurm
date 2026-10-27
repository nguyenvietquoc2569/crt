#!/bin/bash
#SBATCH --job-name=mvc_p7
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_p7_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_p7_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
# Mimi-VC P7, the naturalness bottleneck ladder (26 Oct), all same-speaker reconstructions of utterances of at most
# MAX_S seconds (default 10, the training crop), N each (default 200), scored with WER / SIM / UTMOS under $MVC/eval/p7:
#   tr_copy, tr_V   seen speakers and seen utterances (train-clean-100): the codec ceiling and RUN's fit of its own data;
#   dv_copy, dv_V   unseen speakers (dev-clean): the matched ceiling and generalisation;
#   dv_V_t<t>       SDEdit ladder: the ODE starts from the utterance's own latent noised to t (INIT_TS), which locates the
#                   part of the trajectory where naturalness is lost;
#   extra items     EXTRA="NAME=CFGJSON;..." more dev-clean reconstructions (e.g. other checkpoints).
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/eval_dev.sh
FAILED=
source $WORK_DIR/mvc/jobs/p7_ladder.sh
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
