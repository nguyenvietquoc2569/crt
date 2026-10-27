#!/bin/bash
# S2 (PLAN.md 4), run by job_pilot.slurm in the background of its training: Mimi encoding of
# train-other-500 (train-clean-360 was encoded by job 5640), then the perturbed codes (K=2, 4 workers, so training
# keeps CPU for its data) and CAM++ embeddings of train-clean-360 and train-other-500. Every step resumes.
set -o pipefail
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
export OMP_NUM_THREADS=2
S=train-clean-360,train-other-500
echo " S2 encode: $(date)"
python encode_mimi.py --root $RAW --subsets train-other-500 --out $MVC/mimi || exit 1
echo " S2 perturbed: $(date)"
python encode_perturbed.py --shards $(dirs $S) --raw $RAW --K 2 --workers 4 || exit 2
echo " S2 speaker embeddings: $(date)"
python spk_embed.py --shards $(dirs $S) --raw $RAW || exit 3
du -sh $MVC/mimi/* 2>/dev/null
echo "S2 finished: $(date)"
