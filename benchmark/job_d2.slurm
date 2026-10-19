#!/bin/bash
#SBATCH --job-name=crt_d2
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_d2_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_d2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# Job D2 (ANALYSIS_PLAN_v1.md §8.2), after D1 ($J/v1/d1/conv.done): the 1,000 Dev pairs at 30 s, new out dir
# $J/v1/d2. Submit with submit_ceai.sh (RUNBOOK_v1.md).
#   1. crt_decode_lab.py on the D1 dumps, the §8.2 D2 commands: C1-C5 on the R3 logits of the D1 anchor dumps with the
#      source CB0 (lab/cand), the S1u settings and E17 on the R1 logits with the retrieved CB0 (lab/front). Each must
#      exit 0 with 1,000 pairs per system before it is scored: it writes its manifest afresh and has no resume, so a
#      killed run leaves a partial manifest
#   2. run_conversions.py: the Mimi Q = 32 copies of the source and of P (mc/: a new out dir, so the D1 dumps that
#      step 1 reads are not rewritten)
#   3. rows: lab_rows.py merges lab/cand and lab/front (cand_*, front_*) -> lab/manifest.jsonl and manifest_src.jsonl;
#      the natural controls (source_copy, target_copy from the D1 FAISS rows, as in run_budget_check.slurm) and the
#      copies -> controls/manifest.jsonl and manifest_src.jsonl
#   4. codeword_clustering.py on the D1 R1 dumps (T2, T3 of §6) -> codeword_clustering.{json,md}
#   5. metrics.py, Whisper large-v3 only (Dev), on lab/ and controls/; their @src rows with --metrics sim,secs
#   6. check_d0.py: D2 vs D0 on the 50 screen pairs (C1-C5 WAVs, else the metrics tolerance) -> repro_d2_vs_d0.json
# For select_crt_al.py: lab/metrics.jsonl, lab/metrics_src.jsonl, controls/metrics.jsonl, controls/metrics_src.jsonl
# and repro_d2_vs_d0.json. Scoring is ~11,000 target-side rows x 1.1 s + as many @src rows x 0.074 s = 3.6 h, plus
# the decodes (D0's timings x 20). --time is the 3-day maximum: crt_decode_lab.py cannot resume, so a resubmission
# after a time-out repeats both decode calls, and §8.2 sizes them against the 3-day limit. Writes d2.done when
# every step succeeded.

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
OUT=$J/v1/d2
D0=$J/v1/d0
D1=$J/v1/d1
P=$J/pairs/dev.jsonl
LAB=$OUT/lab
MC=$OUT/mc
CTL=$OUT/controls
mkdir -p $OUT $CTL
TIMES=$OUT/timings.tsv
: > $TIMES
FAILED=
stamp() { echo " $1: $(date)"; echo "$(date +%s) $1" >> $TIMES; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }
rows_ok() {  # rows_ok <jsonl> [<rows per system>]: prints counts; fails on an error row that is not a reference
             # shortfall, or when a system other than mimi_copy_P / target_copy (pairs with P only) has another count
    python - "$@" <<'PY'
import collections, json, os, sys
if not os.path.exists(sys.argv[1]):
    sys.exit(f'  {sys.argv[1]}: missing')
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = collections.Counter(r['system'] for r in rows)
bad = [r for r in rows if 'error' in r and not r.get('shortfall')]
want = int(sys.argv[2]) if len(sys.argv) > 2 else None
p_only = ('mimi_copy_P', 'target_copy')
off = {s: k for s, k in n.items() if want is not None and k != want and s.split('@')[0] not in p_only}
print(f'  {sys.argv[1]}: {len(rows)} rows, {len(n)} systems, {min(n.values(), default=0)}-{max(n.values(), default=0)} '
      f'rows per system, {sum(bool(r.get("shortfall")) for r in rows)} shortfall rows, {len(bad)} other error rows'
      + (f' (e.g. {bad[0]["pair_id"]} {bad[0]["system"]}: {str(bad[0]["error"])[:300]})' if bad else '')
      + (f'; not {want} rows: {off}' if off else ''), flush=True)
sys.exit(1 if bad or off or not rows else 0)
PY
}

[ -f $D1/conv.done ] || { echo "ERROR: $D1/conv.done missing: job_d1_conv.slurm did not finish successfully"; exit 1; }
[ -f $D1/metrics.done ] || echo "  WARN: $D1/metrics.done missing: the D1 scoring has not finished successfully"
[ -f $P ] && ! grep -q -e /datastore/nvquoc -e /datastore/ceai/nvquoc -e /datastore/eaic $P \
    || { echo "ERROR: $P is missing or holds paths of an old home (nvquoc, ceai/nvquoc, eaic)"; exit 1; }
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/d2.done   # a re-run that fails must not leave the last run's marker

stamp "1a. decode lab: C1-C5"
python -u $J/crt_decode_lab.py --conv_dir $D1/rc/dev/30 --dumps $D1/rc/dumps/dev/30 --out_dir $LAB/cand \
    --systems pm_rq_T1,pm18_rq,gv1_pm_cont_T1,gv1_shrink0.85_cont,gv0.5_shrink0.75_cont \
    --logits_from $D1/anc/dumps/dev/30 --logits_key R3 --cb0 src --device cuda
S_CAND=$?
rows_ok $LAB/cand/manifest.jsonl 1000 || S_CAND=1
st $S_CAND decode_cand
stamp "1b. decode lab: S1u settings, E17"
python -u $J/crt_decode_lab.py --conv_dir $D1/rc/dev/30 --dumps $D1/rc/dumps/dev/30 --out_dir $LAB/front \
    --systems us32_w0.2_0.2_0.1,us32_w0.5_0.3_0.3,lmbr_k2_g1_T1 --device cuda
S_FRONT=$?
rows_ok $LAB/front/manifest.jsonl 1000 || S_FRONT=1
st $S_FRONT decode_front

stamp "2. Mimi copies"
python -u $J/run_conversions.py --pairs $P --set dev --budget 30 --systems mimi_copy_src,mimi_copy_P \
    --out_dir $MC --device cuda
st $? mimi_copies

stamp "3. rows"
S_LAB=1
if [ $S_CAND -eq 0 ] && [ $S_FRONT -eq 0 ]; then
    python -u $J/lab_rows.py --pairs $P --lab $LAB/cand cand --lab $LAB/front front --out $LAB/manifest.jsonl
    S_LAB=$?
fi
st $S_LAB lab_rows
python - <<PY
import json
rows = []
for l in open('$D1/rc/dev/30/manifest.jsonl'):
    r = json.loads(l)
    if r['system'] == 'faiss' and not r.get('error'):
        rows.append(dict(r, system='source_copy', wav=r['source_wav']))
        if r.get('parallel_wav'):
            rows.append(dict(r, system='target_copy', wav=r['parallel_wav'], copy_control=True))
open('$CTL/natural.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(len(rows), 'natural-control rows')
PY
st $? natural
python -u $J/lab_rows.py --pairs $P --manifest $MC/dev/30/manifest.jsonl --manifest $CTL/natural.jsonl \
    --out $CTL/manifest.jsonl
S_CTL=$?
st $S_CTL controls_rows

stamp "4. codeword clustering on the D1 R1 dumps"
python -u $J/codeword_clustering.py --dumps $D1/rc/dumps/dev/30 --out $OUT/codeword_clustering
st $? codeword_clustering

stamp "5a. metrics: decode lab"
if [ $S_LAB -eq 0 ]; then
    python -u $J/metrics.py --in_jsonl $LAB/manifest.jsonl --out_jsonl $LAB/metrics.jsonl \
        --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
    st $? metrics_lab
    rows_ok $LAB/metrics.jsonl 1000
    st $? metrics_lab_rows
    python -u $J/metrics.py --in_jsonl $LAB/manifest_src.jsonl --out_jsonl $LAB/metrics_src.jsonl \
        --metrics sim,secs --device cuda
    st $? metrics_lab_src
    rows_ok $LAB/metrics_src.jsonl 1000
    st $? metrics_lab_src_rows
else
    echo "  decode-lab outputs incomplete: not scored"
fi
stamp "5b. metrics: controls"
if [ $S_CTL -eq 0 ]; then
    python -u $J/metrics.py --in_jsonl $CTL/manifest.jsonl --out_jsonl $CTL/metrics.jsonl \
        --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
    st $? metrics_controls
    rows_ok $CTL/metrics.jsonl 1000
    st $? metrics_controls_rows
    python -u $J/metrics.py --in_jsonl $CTL/manifest_src.jsonl --out_jsonl $CTL/metrics_src.jsonl \
        --metrics sim,secs --device cuda
    st $? metrics_controls_src
    rows_ok $CTL/metrics_src.jsonl 1000
    st $? metrics_controls_src_rows
fi

stamp "6. D2 vs D0 on the screen pairs"
python -u $J/check_d0.py --pairs $P --ref_d0 $D0 --cand $LAB/cand --cand_metrics $LAB/metrics.jsonl \
    --label 'D2 vs D0' --out $OUT/repro_d2_vs_d0.json
echo "  D2 vs D0 reproduction: exit $? (repro_d2_vs_d0.json)"
du -sh $LAB $MC /datastore/ceai/crt
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T')" > $OUT/d2.done
[ -z "$FAILED" ]
