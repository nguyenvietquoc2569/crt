#!/bin/bash
#SBATCH --job-name=mvc_s3
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_s3_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_s3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
# Mimi-VC S3 (after the P5 gate record, PLAN.md): the perturbed Whisper-small stream (draw 0) for train-clean-360 and
# train-other-500, which the R1 content route now needs (whisper_feats.py, 8 workers; every shard resumes), then the
# full data audit (audit_data.py pass 1 on the shards not yet audited, with UTMOS, and pass 2 over train-clean-100,
# dev-clean, train-clean-360 and train-other-500, replacing results/audit_v1).
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
FAILED=
S=train-clean-360,train-other-500
echo " whisper $S: $(date)"
python whisper_feats.py --shards $(dirs $S) --raw $RAW --K 1 --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED whisper"
echo " audit: $(date)"
python -u audit_data.py --shards $(dirs train-clean-100,dev-clean,$S) --raw $RAW --utmos --journal $J \
    --workers $SLURM_CPUS_PER_TASK --force --out $WORK_DIR/mvc/results/audit_v1 || FAILED="$FAILED audit"
du -sh $MVC/mimi/* 2>/dev/null
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
