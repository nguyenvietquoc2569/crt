#!/bin/bash
#SBATCH --job-name=crt_t2
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_t2_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_t2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# Job T2 (ANALYSIS_PLAN_v1.md §8.2), after T1 ($J/v1/t1/t1.done, written under the same selection.env): LTC-2k at
# 30 s, as T1 without parallel recordings, new out dir $J/v1/t2. No metrics (job M). Submit with submit_ceai.sh.
#   1. run_conversions.py: S1 FAISS, S3 CRT τ = 0.4, S2 CRT τ_dev (left out when τ_dev = 0.4), S4 CRT argmax, logits
#      dumped for S1u and E17 (rc/)
#   2. run_conversions.py: the Mimi copy of the source (mc/; LTC-2k has no P, so no mimi_copy_P)
#   3. crt_anchor_lab.py --mode R1+R3: A1-A5, plus C1 if CRT-AL is C1; no --source_e (every LTC-2k source is a target,
#      so sim_src_wavs is its own target E); R3 logits dumped only if CRT-AL is one of C2-C5 (anc/)
#   4. crt_decode_lab.py: S1u and, if it entered, E17 (lab/front); CRT-AL if it is one of C2-C5 (lab/cand)
# Selection values from $J/v1/selection.env as in job_t1.slurm; 3-day --time as there (crt_decode_lab.py reruns
# whole). Writes t2.done when every step succeeded.

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
OUT=$J/v1/t2
LP=$J/pairs/ltc2k.jsonl
RC=$OUT/rc
MC=$OUT/mc
ANC=$OUT/anc
LAB=$OUT/lab
mkdir -p $OUT
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

# ---- values from the selection record (as job_t1.slurm) ---------------------------------------------------------
SEL=$J/v1/selection.env
[ -f $SEL ] || { echo "ERROR: $SEL missing: write it from the committed selection record (RUNBOOK_v1.md)"; exit 1; }
source $SEL
SELSHA=$(sha256sum < $SEL | cut -d' ' -f1)
echo "selection $SEL (sha256 $SELSHA): record $RECORD_SHA256, τ_dev $TAU_DEV, CRT-AL $CRT_AL, S1u $S1U, E17 $E17," \
     "VCTK-2k $VCTK_TARGETS targets, baselines $BASELINES"
[[ $RECORD_SHA256 =~ ^[0-9a-f]{64}$ ]] && [[ " 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 " == *" $TAU_DEV "* ]] \
    && [[ " C1 C2 C3 C4 C5 none " == *" $CRT_AL "* ]] && [[ " us32_w0.2_0.2_0.1 us32_w0.5_0.3_0.3 " == *" $S1U "* ]] \
    && [[ " yes no " == *" $E17 "* ]] && [[ " 5 10 " == *" $VCTK_TARGETS "* ]] \
    && [[ $BASELINES =~ ^(none|(knnvc|freevc|seedvc|facodec)(,(knnvc|freevc|seedvc|facodec))*)$ ]] \
    || { echo "ERROR: $SEL holds a placeholder or a value outside the plan"; exit 1; }
RC_SYS=faiss,crt_tau:0.4$([ $TAU_DEV = 0.4 ] || echo ,$TAU_DEV),crt_argmax
ANC_SYS=R1_s0_pm_rq_T1,R3_r0_pm_rq_T1,R3_s0_argmax,R3_s0_tau0.4,src0_faiss$([ $CRT_AL = C1 ] && echo ,R3_s0_pm_rq_T1)
case $CRT_AL in
    C2) CAND=pm18_rq ;;
    C3) CAND=gv1_pm_cont_T1 ;;
    C4) CAND=gv1_shrink0.85_cont ;;
    C5) CAND=gv0.5_shrink0.75_cont ;;
    *) CAND= ;;
esac
ANC_DUMP=$([ -n "$CAND" ] && echo --dump_logits --dump_recipes R3)
FRONT=$S1U$([ $E17 = yes ] && echo ,lmbr_k2_g1_T1)
# ------------------------------------------------------------------------------------------------------------------

grep -q "selection sha256 $SELSHA" $J/v1/t1/t1.done 2>/dev/null \
    || { echo "ERROR: $J/v1/t1/t1.done missing or written under another selection.env"; exit 1; }
[ -f $LP ] && ! grep -q -e /datastore/nvquoc -e /datastore/ceai/nvquoc -e /datastore/eaic $LP \
    || { echo "ERROR: $LP is missing or holds paths of an old home (nvquoc, ceai/nvquoc, eaic)"; exit 1; }
N=$(grep -c . $LP)
echo "  $LP: $N pairs; run_conversions $RC_SYS; anchor lab $ANC_SYS $ANC_DUMP; front $FRONT; cand ${CAND:-none}"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/t2.done   # a re-run that fails must not leave the last run's marker

stamp "1. conversions: S1-S4, logits dumped"
python -u $J/run_conversions.py --pairs $LP --set ltc2k --budget 30 --systems $RC_SYS --dump_logits \
    --out_dir $RC --device cuda
st $? conversions
stamp "2. Mimi copy of the source"
python -u $J/run_conversions.py --pairs $LP --set ltc2k --budget 30 --systems mimi_copy_src \
    --out_dir $MC --device cuda
st $? mimi_copy
stamp "3. anchor lab, R1+R3 mode"
python -u $J/crt_anchor_lab.py --pairs $LP --set ltc2k --budget 30 --conv_dir $RC --out_dir $ANC --mode R1+R3 \
    --systems $ANC_SYS $ANC_DUMP --device cuda
st $? anchor
stamp "4a. decode lab: S1u, E17"
python -u $J/crt_decode_lab.py --conv_dir $RC/ltc2k/30 --dumps $RC/dumps/ltc2k/30 --out_dir $LAB/front \
    --systems $FRONT --device cuda
S=$?
rows_ok $LAB/front/manifest.jsonl $N || S=1
st $S decode_front
if [ -n "$CAND" ]; then
    stamp "4b. decode lab: CRT-AL $CRT_AL ($CAND)"
    python -u $J/crt_decode_lab.py --conv_dir $RC/ltc2k/30 --dumps $RC/dumps/ltc2k/30 --out_dir $LAB/cand \
        --systems $CAND --logits_from $ANC/dumps/ltc2k/30 --logits_key R3 --cb0 src --device cuda
    S=$?
    rows_ok $LAB/cand/manifest.jsonl $N || S=1
    st $S decode_cand
fi
stamp "done"
for f in $RC/ltc2k/30/manifest.jsonl $MC/ltc2k/30/manifest.jsonl $ANC/ltc2k/30/manifest.jsonl; do
    rows_ok $f $N
    st $? "rows:$f"
done
du -sb $RC/dumps $MC/dumps $ANC/dumps
du -sh $RC $MC $ANC $LAB /datastore/ceai/crt
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T') selection sha256 $SELSHA" > $OUT/t2.done
[ -z "$FAILED" ]
