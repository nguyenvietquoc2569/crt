#!/bin/bash
#SBATCH --job-name=crt_d0
#SBATCH --output=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_d0_%j.out
#SBATCH --error=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_d0_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Job D0 (ANALYSIS_PLAN_v1.md §8.1 items 3 and 9), before the freeze, on the 50 screen pairs: rows k % 20 == 0 of
# pairs/dev.jsonl, i.e. --shard 0/20 of the whole list (crt_anchor_lab needs the whole list for the source and third
# speakers), 30 s references, new out dir $J/v1/d0. Submit with submit_nvquoc.sh (RUNBOOK_v1.md).
#   1. pair lists (CPU): the source-E file of pairs/vctk2k.jsonl and the 10-target VCTK-2k variant in
#      pairs_vctk2k_t10/, each made only if absent (a re-run never rewrites a list that may already be hashed);
#      make_pairs --check on both directories (every Dev / LTC-2k source is a target, every VCTK-2k source has E)
#   2. run_conversions.py: FAISS, CRT on the τ grid 0.1-0.9 (so T1's crt_tau:<τ_dev> is exercised whatever τ_dev
#      is) and CRT argmax, logits dumped (rc/)
#   3. run_conversions.py: the Mimi Q = 32 copies of the source and of P, in their own out dir (mc/), so that a P
#      read cannot fail S1-S4 rows
#   4. crt_anchor_lab.py in the full Dev mode, R3 logits dumped (anc/; the D1 command)
#   5. crt_anchor_lab.py in the test-set mode R1+R3 with T1's systems A1-A5 and C1 (anc_r1r3/): its out_codes are
#      checked like those of step 4, so the test-set mode is tested before any test-set run (codes only; its WAVs
#      decode the same codes and are not scored)
#   6. crt_anchor_lab.py --summarize_only over the dumps of anc/ (T9)
#   7. crt_decode_lab.py: C1-C5 on the R3 logits with the source CB0 (lab/cand); S1u settings and E17 (lab/front)
#   8. rows: lab_rows.py merges lab/cand and lab/front (cand_*, front_*) into lab/manifest.jsonl, and the step 2
#      rows, the Mimi copies and the natural controls (source_copy, target_copy, built as in run_budget_check.slurm)
#      into rows/conv.jsonl; both with their @src rows
#   9. metrics.py on every manifest, Whisper large-v3 only (Dev); @src rows with --metrics sim,secs
#  10. check_d0.py: the §8.1 item 9 checks against the screen under $J (clustering_check: job of
#      run_clustering_check.slurm; anchor_lab: job 2634; clustering_check/lab_r3a, lab_r4p: run_lab_r3.slurm,
#      run_round4.slurm), the pair-list check, the step times and dump sizes -> check_d0.json (D1 starts only if it
#      passed; select_crt_al.py --repro)
# Every step runs even if an earlier one failed, so one submission gives every check; the job exits 1 if any failed.
# --cpus-per-task=8 as in the screen jobs: the torch thread count is part of the numerics (run_conversions.py).

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
OUT=$J/v1/d0
P=$J/pairs/dev.jsonl
T10=$J/pairs_vctk2k_t10
RC=$OUT/rc
MC=$OUT/mc
ANC=$OUT/anc
AR=$OUT/anc_r1r3
LAB=$OUT/lab
ROWS=$OUT/rows
TEST_SYS=R1_s0_pm_rq_T1,R3_r0_pm_rq_T1,R3_s0_argmax,R3_s0_tau0.4,src0_faiss,R3_s0_pm_rq_T1   # A1-A5, C1
mkdir -p $OUT $ROWS
TIMES=$OUT/timings.tsv
: > $TIMES
FAILED=
stamp() { echo " $1: $(date)"; echo "$(date +%s) $1" >> $TIMES; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }

for f in $P $J/pairs/vctk2k.jsonl $J/pairs/ltc2k.jsonl; do
    [ -f $f ] && ! grep -q -e /datastore/ceai -e /datastore/eaic $f \
        || { echo "ERROR: $f is missing or holds /datastore/ceai or /datastore/eaic paths (RUNBOOK_v1.md, 0)"; exit 1; }
done
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/check_d0.json   # a re-run that fails must not leave the last run's marker

stamp "1. pair lists"
if [ -f $J/pairs/vctk2k_source_E.json ]; then
    echo "  kept $J/pairs/vctk2k_source_E.json"
else
    python -u $J/make_pairs.py --source_e_only --output_dir $J/pairs 2>&1 | tee -a $OUT/make_pairs.log
    st ${PIPESTATUS[0]} source_E
fi
if [ -f $T10/summary.json ]; then
    echo "  kept $T10 (its summary.json exists)"
else
    python -u $J/make_pairs.py --sets vctk2k --vctk_targets 10 --output_dir $T10 2>&1 | tee -a $OUT/make_pairs.log
    st ${PIPESTATUS[0]} vctk2k_t10
fi
python -u $J/make_pairs.py --check --sets dev,vctk2k,ltc2k --output_dir $J/pairs 2>&1 | tee -a $OUT/make_pairs.log
st ${PIPESTATUS[0]} check_pairs
python -u $J/make_pairs.py --check --sets vctk2k --output_dir $T10 2>&1 | tee -a $OUT/make_pairs.log
st ${PIPESTATUS[0]} check_pairs_t10
sha256sum $P $J/pairs/vctk2k.jsonl $J/pairs/ltc2k.jsonl $J/pairs/vctk2k_source_E.json $T10/vctk2k.jsonl \
    $T10/vctk2k_source_E.json $T10/summary.json 2>&1 | tee -a $OUT/make_pairs.log

stamp "2. conversions: FAISS, CRT τ 0.1-0.9, CRT argmax"
python -u $J/run_conversions.py --pairs $P --set dev --budget 30 --shard 0/20 \
    --systems faiss,crt_tau:0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,crt_argmax --dump_logits --out_dir $RC --device cuda
st $? conversions
stamp "3. Mimi copies"
python -u $J/run_conversions.py --pairs $P --set dev --budget 30 --shard 0/20 --systems mimi_copy_src,mimi_copy_P \
    --out_dir $MC --device cuda
st $? mimi_copies

stamp "4. anchor lab, full Dev mode"
python -u $J/crt_anchor_lab.py --pairs $P --set dev --budget 30 --shard 0/20 --conv_dir $RC --out_dir $ANC \
    --dump_logits --dump_recipes R3 --device cuda
st $? anchor_full
stamp "5. anchor lab, test-set mode R1+R3"
python -u $J/crt_anchor_lab.py --pairs $P --set dev --budget 30 --shard 0/20 --conv_dir $RC --out_dir $AR \
    --mode R1+R3 --systems $TEST_SYS --dump_logits --dump_recipes R3 --device cuda
st $? anchor_r1r3
stamp "6. anchor lab summary over all dumps"
python -u $J/crt_anchor_lab.py --pairs $P --set dev --budget 30 --out_dir $ANC --summarize_only
st $? summarize

stamp "7a. decode lab: C1-C5"
python -u $J/crt_decode_lab.py --conv_dir $RC/dev/30 --dumps $RC/dumps/dev/30 --out_dir $LAB/cand \
    --systems pm_rq_T1,pm18_rq,gv1_pm_cont_T1,gv1_shrink0.85_cont,gv0.5_shrink0.75_cont \
    --logits_from $ANC/dumps/dev/30 --logits_key R3 --cb0 src --device cuda
st $? decode_cand
stamp "7b. decode lab: S1u settings, E17"
python -u $J/crt_decode_lab.py --conv_dir $RC/dev/30 --dumps $RC/dumps/dev/30 --out_dir $LAB/front \
    --systems us32_w0.2_0.2_0.1,us32_w0.5_0.3_0.3,lmbr_k2_g1_T1 --device cuda
st $? decode_front

stamp "8. rows"
python - <<PY
import json
rows = []
for l in open('$RC/dev/30/manifest.shard0of20.jsonl'):
    r = json.loads(l)
    if r['system'] == 'faiss' and not r.get('error'):
        rows.append(dict(r, system='source_copy', wav=r['source_wav']))
        if r.get('parallel_wav'):
            rows.append(dict(r, system='target_copy', wav=r['parallel_wav'], copy_control=True))
open('$ROWS/natural.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(len(rows), 'natural-control rows')
PY
st $? natural
python -u $J/lab_rows.py --pairs $P --lab $LAB/cand cand --lab $LAB/front front --out $LAB/manifest.jsonl
st $? lab_rows
python -u $J/lab_rows.py --pairs $P --manifest $RC/dev/30/manifest.shard0of20.jsonl \
    --manifest $MC/dev/30/manifest.shard0of20.jsonl --manifest $ROWS/natural.jsonl --out $ROWS/conv.jsonl
st $? conv_rows

stamp "9a. metrics: anchor lab"
python -u $J/metrics.py --in_jsonl $ANC/dev/30/manifest.shard0of20.jsonl --out_jsonl $ANC/dev/30/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? metrics_anchor
python -u $J/metrics.py --in_jsonl $ANC/dev/30/manifest_src.shard0of20.jsonl --out_jsonl $ANC/dev/30/metrics_src.jsonl \
    --metrics sim,secs --device cuda
st $? metrics_anchor_src
stamp "9b. metrics: decode lab"
python -u $J/metrics.py --in_jsonl $LAB/manifest.jsonl --out_jsonl $LAB/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? metrics_lab
python -u $J/metrics.py --in_jsonl $LAB/manifest_src.jsonl --out_jsonl $LAB/metrics_src.jsonl \
    --metrics sim,secs --device cuda
st $? metrics_lab_src
stamp "9c. metrics: conversions, Mimi copies, natural controls"
python -u $J/metrics.py --in_jsonl $ROWS/conv.jsonl --out_jsonl $ROWS/conv_metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? metrics_conv
python -u $J/metrics.py --in_jsonl $ROWS/conv_src.jsonl --out_jsonl $ROWS/conv_metrics_src.jsonl \
    --metrics sim,secs --device cuda
st $? metrics_conv_src

stamp "10. checks"
du -sb $RC/dumps $MC/dumps $ANC/dumps $AR/dumps
du -sh $RC $MC $ANC $AR $LAB /datastore/nvquoc/crt
python -u $J/check_d0.py --pairs $P --ref_screen $J --rc $RC --anc $ANC --anc $AR --cand $LAB/cand \
    --cand_metrics $LAB/metrics.jsonl --lists $J/pairs:dev,vctk2k,ltc2k --lists $T10:vctk2k --timings $TIMES \
    --runs $RC/dev/30/runs.jsonl --runs $MC/dev/30/runs.jsonl --sizes $RC/dumps/dev/30 --sizes $MC/dumps/dev/30 \
    --sizes $ANC/dumps/dev/30 --sizes $AR/dumps/dev/30 --label 'D0 vs the screen' --out $OUT/check_d0.json
st $? check_d0
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
