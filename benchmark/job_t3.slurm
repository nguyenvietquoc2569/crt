#!/bin/bash
#SBATCH --job-name=crt_t3
#SBATCH --output=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_t3_%j.out
#SBATCH --error=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_t3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# Job T3 (ANALYSIS_PLAN_v1.md §8.2, E16 of §1.3), after T2 ($J/v1/t2/t2.done, same selection.env): the budget sweep
# at 3 s and 180 s on VCTK-2k (the record's variant) and LTC-2k, new out dir $J/v1/t3. No metrics (job M). The
# kNN-VC part of the sweep is in T4. Per set and budget:
#   1. run_conversions.py: S1 FAISS, S3 CRT τ = 0.4 and S2 CRT τ_dev (left out when τ_dev = 0.4); no logits (rc/)
#   2. CRT-AL: C1 is decoded by crt_anchor_lab.py --mode R1+R3 --systems R3_s0_pm_rq_T1 (anc/); C2-C5 need the R3
#      logits: crt_anchor_lab.py --mode R1+R3 --dump_logits --dump_recipes R3 decoding src0_faiss only (--systems
#      cannot be empty; that WAV is not scored), then crt_decode_lab.py on them (lab/cand_<set>_<budget>); under
#      fallback F3 there is no CRT-AL and only step 1 runs
# Some LTC-2k targets have less than 180 s of audio (pairs/summary.json): their pairs get shortfall error rows in
# steps 1-2 and no decode-lab row, as §1.3 expects; every other error row fails the job. VCTK-2k rows get
# sim_src_wavs from the variant's source-E file. Selection values from $J/v1/selection.env as in job_t1.slurm;
# 3-day --time as there (crt_decode_lab.py reruns whole). Writes t3.done when every step succeeded.

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
OUT=$J/v1/t3
LP=$J/pairs/ltc2k.jsonl
RC=$OUT/rc
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
if [ $VCTK_TARGETS = 10 ]; then VP=$J/pairs_vctk2k_t10/vctk2k.jsonl; else VP=$J/pairs/vctk2k.jsonl; fi
SE=$(dirname $VP)/vctk2k_source_E.json
RC_SYS=faiss,crt_tau:0.4$([ $TAU_DEV = 0.4 ] || echo ,$TAU_DEV)
case $CRT_AL in
    C1) ANC_SYS=R3_s0_pm_rq_T1; CAND= ;;
    C2) ANC_SYS=src0_faiss; CAND=pm18_rq ;;
    C3) ANC_SYS=src0_faiss; CAND=gv1_pm_cont_T1 ;;
    C4) ANC_SYS=src0_faiss; CAND=gv1_shrink0.85_cont ;;
    C5) ANC_SYS=src0_faiss; CAND=gv0.5_shrink0.75_cont ;;
    none) ANC_SYS=; CAND= ;;
esac
ANC_DUMP=$([ -n "$CAND" ] && echo --dump_logits --dump_recipes R3)
# ------------------------------------------------------------------------------------------------------------------

grep -q "selection sha256 $SELSHA" $J/v1/t2/t2.done 2>/dev/null \
    || { echo "ERROR: $J/v1/t2/t2.done missing or written under another selection.env"; exit 1; }
for f in $VP $LP; do
    [ -f $f ] && ! grep -q -e /datastore/ceai -e /datastore/eaic $f \
        || { echo "ERROR: $f is missing or holds /datastore/ceai or /datastore/eaic paths"; exit 1; }
done
[ -f $SE ] || { echo "ERROR: $SE missing"; exit 1; }
echo "  run_conversions $RC_SYS; anchor lab ${ANC_SYS:-none} $ANC_DUMP; cand ${CAND:-none}"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1
rm -f $OUT/t3.done   # a re-run that fails must not leave the last run's marker

for SET in vctk2k ltc2k; do
    if [ $SET = vctk2k ]; then PL=$VP; SRC_E="--source_e $SE"; else PL=$LP; SRC_E=; fi
    N=$(grep -c . $PL)
    for B in 3 180; do
        stamp "$SET $B s: conversions"
        python -u $J/run_conversions.py --pairs $PL --set $SET --budget $B --systems $RC_SYS \
            --out_dir $RC --device cuda
        st $? conversions_${SET}_$B
        rows_ok $RC/$SET/$B/manifest.jsonl $N
        st $? rows_conversions_${SET}_$B
        if [ -n "$ANC_SYS" ]; then
            stamp "$SET $B s: anchor lab (CRT-AL $CRT_AL)"
            python -u $J/crt_anchor_lab.py --pairs $PL --set $SET --budget $B --conv_dir $RC --out_dir $ANC \
                --mode R1+R3 --systems $ANC_SYS $SRC_E $ANC_DUMP --device cuda
            st $? anchor_${SET}_$B
            rows_ok $ANC/$SET/$B/manifest.jsonl $N
            st $? rows_anchor_${SET}_$B
        fi
        if [ -n "$CAND" ]; then
            stamp "$SET $B s: decode lab (CRT-AL $CRT_AL, $CAND)"
            python -u $J/crt_decode_lab.py --conv_dir $RC/$SET/$B --dumps $RC/dumps/$SET/$B \
                --out_dir $LAB/cand_${SET}_$B --systems $CAND --logits_from $ANC/dumps/$SET/$B --logits_key R3 \
                --cb0 src --device cuda
            st $? decode_${SET}_$B
            rows_ok $LAB/cand_${SET}_$B/manifest.jsonl
            st $? rows_decode_${SET}_$B
        fi
    done
done
stamp "done"
du -sb $RC/dumps $ANC/dumps
du -sh $RC $ANC $LAB /datastore/nvquoc/crt
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ] && echo "job $SLURM_JOB_ID $(date '+%F %T') selection sha256 $SELSHA" > $OUT/t3.done
[ -z "$FAILED" ]
