#!/bin/bash
#SBATCH --job-name=crt_d1_conv
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_d1_conv_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_d1_conv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Job D1, part 1 of 2: the conversions (ANALYSIS_PLAN_v1.md §8.2), after the freeze and its public timestamp (§8.4):
# the 1,000 Dev pairs at 30 s, new out dir $J/v1/d1. Submit with submit_ceai.sh (RUNBOOK_v1.md).
#   1. run_conversions.py --systems faiss --dump_logits (rc/): S1, and the R1 logits that S1u and E17 (D2) and
#      codeword_clustering.py (D2) read
#   2. crt_anchor_lab.py in the full Dev mode, 36 systems (the R1 and R3 τ grids, A1-A5, C1), R3 logits dumped (anc/)
#   3. crt_anchor_lab.py --summarize_only over all dumps (T9; D1 is not sharded, so it rewrites step 2's summary)
#   4. check_d0.py: D1 vs D0 on the 50 screen pairs (S1/S3 codes, A1-A5 and C1 out_codes) -> repro_d1_vs_d0.json
#      (select_crt_al.py --repro). A mismatch is reported and does not stop job_d1_metrics.slurm: §4.4 reports the
#      selection on both the first and the fixed code.
# Scoring is job_d1_metrics.slurm. Why two jobs: the file times of job 2634 (run_id 06:31:26, anchor_summary written
# 06:32) suggest that 50 pairs x 28 systems took about a minute, so 1,000 pairs x 36 systems should take 1-2 h, but
# this was never measured (D0 measures it: check_d0.json info.timings, x 20 for 1,000 pairs); scoring is 36,000 rows
# x 1.1 s + 36,000 @src rows x 0.074 s = 12.1 h (§8.2). One job would likely fit 72 h; split, each part has its own
# time limit (48 h) however slow the conversions turn out, and the conversion errors and the screen-pair
# reproduction are seen before 12 h of scoring. Every step resumes, so a job that hits its time limit is
# resubmitted unchanged.
# Starts only if D0 passed ($J/v1/d0/check_d0.json). Writes conv.done when steps 1-3 succeeded.

source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/ceai/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

WORK_DIR=/datastore/ceai/crt/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$J/v1/d1
D0=$J/v1/d0
P=$J/pairs/dev.jsonl
RC=$OUT/rc
ANC=$OUT/anc
mkdir -p $OUT
TIMES=$OUT/timings_conv.tsv
: > $TIMES
FAILED=
stamp() { echo " $1: $(date)"; echo "$(date +%s) $1" >> $TIMES; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }

[ -f $P ] && ! grep -q -e /datastore/nvquoc -e /datastore/ceai/nvquoc -e /datastore/eaic $P \
    || { echo "ERROR: $P is missing or holds paths of an old home (nvquoc, ceai/nvquoc, eaic)"; exit 1; }
[ -f $D0/check_d0.json ] || { echo "ERROR: $D0/check_d0.json missing: run job_d0.slurm first"; exit 1; }
python - <<PY || exit 1
import json, sys
d = json.load(open('$D0/check_d0.json'))
print('D0:', d['summary'])
sys.exit(0 if d.get('passed') is True else 'ERROR: D0 did not pass ($D0/check_d0.json)')
PY
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/conv.done   # a re-run that fails must not leave the last run's marker

stamp "1. conversions: FAISS, logits dumped"
python -u $J/run_conversions.py --pairs $P --set dev --budget 30 --systems faiss \
    --dump_logits --out_dir $RC --device cuda
st $? conversions
stamp "2. anchor lab, full Dev mode"
python -u $J/crt_anchor_lab.py --pairs $P --set dev --budget 30 --conv_dir $RC \
    --out_dir $ANC --dump_logits --dump_recipes R3 --device cuda
st $? anchor
stamp "3. anchor lab summary over all dumps"
python -u $J/crt_anchor_lab.py --pairs $P --set dev --budget 30 --out_dir $ANC --summarize_only
st $? summarize
stamp "4. D1 vs D0 on the screen pairs"
du -sb $RC/dumps $ANC/dumps
du -sh $RC $ANC /datastore/ceai/crt
python -u $J/check_d0.py --pairs $P --ref_d0 $D0 --rc $RC --anc $ANC --sizes $RC/dumps/dev/30 \
    --sizes $ANC/dumps/dev/30 --label 'D1 vs D0' --out $OUT/repro_d1_vs_d0.json
echo "  D1 vs D0 reproduction: exit $? (repro_d1_vs_d0.json)"
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T')" > $OUT/conv.done
[ -z "$FAILED" ]
