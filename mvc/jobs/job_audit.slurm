#!/bin/bash
#SBATCH --job-name=mvc_audit
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_audit_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_audit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Mimi-VC (paper 3): per-utterance data audit (audit_data.py). Pass 1 (audio statistics, latent RMS, UTMOS) on every
# encoded shard of SUBSETS ('+' or ',' separated; default train-clean-100+dev-clean+train-clean-360+train-other-500; a
# subset whose shards are not written yet is skipped), then pass 2 over whatever is audited: speaker cosine from
# .spk.npy, CB0 agreement from .pcodes<k>.npy where they exist. Results: $WORK_DIR/mvc/results/audit_v1. Rerun to
# extend: finished shards are skipped, and pass 2 is redone over everything (FORCE=1 lets it replace exclude.txt).
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
SUBSETS=${SUBSETS:-train-clean-100,dev-clean,train-clean-360,train-other-500}
SUBSETS=${SUBSETS//+/,}
D=$(for s in $(echo $SUBSETS | tr , ' '); do [ -d $MVC/mimi/$s ] && echo $MVC/mimi/$s; done | paste -sd,)
echo "audit $D: $(date)"
python -u audit_data.py --shards $D --raw $RAW --utmos --journal $J --workers $SLURM_CPUS_PER_TASK \
    --out $WORK_DIR/mvc/results/audit_v1 $([ "${FORCE:-0}" = 1 ] && echo --force)
rc=$?
echo "finished rc=$rc: $(date)"
exit $rc
