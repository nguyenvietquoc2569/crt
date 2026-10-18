#!/bin/bash
#SBATCH --job-name=round4
#SBATCH --output=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/round4_%j.out
#SBATCH --error=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/round4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Round 4 on the 50 Dev pairs of shard 0/20 (paper/journal_extension/crt_math_ideas_verified.md; results so far in
# docs/training_tips.md C15-C16):
#   1. crt_anchor_lab.py with CRT hints from us32 unit selection (--path_mode us32; round 2's best FAISS variant) and
#      R1/R3 logits dumped; its own metrics + source-speaker SIM
#   2. crt_decode_lab.py on those dumps (R3 posterior, output CB0 = source): MBR, variance restoration, shrinkage,
#      deep-level drop, speaker prior
#   3. crt_decode_lab.py combos on the published-path anchor dumps (job 2634): shrinkage + variance restoration
#   4. metrics.py on steps 2-3
#   5. crt_truth_fit.py v2 (T2.2 a-f: full-input error, calibration, Shapley levels, tau_chow, gate monotonicity,
#      oracle headroom) on the 307 self-conversion dumps of job 2629
# Steps 2-5 run even if an earlier analysis step fails; step 1 must succeed.

source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/nvquoc/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

WORK_DIR=/datastore/nvquoc/crt/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
C=$J/clustering_check
U=$J/anchor_us32

echo " 1. anchor lab, us32 hints: $(date)"
python -u $J/crt_anchor_lab.py --pairs $J/pairs/dev.jsonl --set dev --budget 30 --shard 0/20 --conv_dir $C \
    --out_dir $U --device cuda --path_mode us32 --dump_logits --dump_recipes R1,R3 || exit 1
python -u $J/metrics.py --in_jsonl $U/dev/30/manifest.shard0of20.jsonl --out_jsonl $U/dev/30/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
S1=$?
python -u $J/metrics.py --in_jsonl $U/dev/30/manifest_src.shard0of20.jsonl --out_jsonl $U/dev/30/metrics_src.jsonl \
    --metrics sim,secs --device cuda

echo " 2. offline decoders on the us32-hint dumps: $(date)"
python -u $J/crt_decode_lab.py --conv_dir $C/dev/30 --dumps $U/dumps/dev/30 --out_dir $C/lab_r4u \
    --systems faiss,pm_rq_T1,pm_rq_T0.75,gv1_pm_cont_T1,gv0.5_pm_cont_T1,shrink0.75_rq,gv1_shrink0.75_cont,pm18_rq,spk1_pm_rq \
    --logits_from $U/dumps/dev/30 --logits_key R3 --cb0 src --device cuda
S2=$?
echo " 3. offline combos on the published-path anchor dumps: $(date)"
python -u $J/crt_decode_lab.py --conv_dir $C/dev/30 --dumps $C/dumps/dev/30 --out_dir $C/lab_r4p \
    --systems gv1_shrink0.75_cont,gv0.5_shrink0.75_cont,shrink0.85_rq,gv1_shrink0.85_cont \
    --logits_from $J/anchor_lab/dumps/dev/30 --logits_key R3 --cb0 src --device cuda
S3=$?
python - <<PY
import json, os
rows = []
for d, tag in (('$C/lab_r4u', 'r4u'), ('$C/lab_r4p', 'r4p')):
    f = f'{d}/manifest.jsonl'
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l); r['system'] = r['system'].replace('lab_', f'{tag}_', 1); rows.append(r)
open('$C/lab_r4_manifest.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(len(rows), 'rows')
PY
echo " 4. metrics: $(date)"
python -u $J/metrics.py --in_jsonl $C/lab_r4_manifest.jsonl --out_jsonl $C/lab_r4_metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
S4=$?
echo " 5. truth fit v2: $(date)"
python -u $J/crt_truth_fit.py --dumps $J/truth_set/dumps/self/30 --out $J/truth_set/truth_fit_v2 --device cuda
S5=$?
echo " finished: anchor-metrics $S1, lab_r4u $S2, lab_r4p $S3, metrics $S4, truth v2 $S5: $(date)"
du -sh $U $C/lab_r4u $C/lab_r4p /datastore/nvquoc/crt
exit $(( S1 || S2 || S3 || S4 || S5 ))
