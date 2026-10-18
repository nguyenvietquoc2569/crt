#!/bin/bash
#SBATCH --job-name=crt_ot_screen
#SBATCH --output=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_ot_screen_%j.out
#SBATCH --error=/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_ot_screen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# Round 7 screen (pre-freeze, exploratory; 18 Oct 2026): Gaussian optimal transport in decoder space
# (crt_decode_lab.py gaussian_ot) on the 50 screen pairs, read from job D0's dumps (no new conversion or CRT forward):
#   - CRT-free conversion: the source's own latent and CB0 mapped to the target reference's statistics
#     (src_cont = the unmapped control, ot0 = mean shift only, ot16 / ot64 = full map in the top-16 / 64 PCs,
#     ot16nm = the map without the mean shift);
#   - CRT-AL + transport: ot16 / ot64 of the R3 posterior-mean latent with the source CB0 (C3's base).
# Rows are tagged r7_<system>; metrics with Whisper-large-v3 only (Dev) and the @src rows. Submit with
# submit_nvquoc.sh. Output: $J/v1/ot_screen.

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
OUT=$J/v1/ot_screen
D0=$J/v1/d0
P=$J/pairs/dev.jsonl
SYS=src_cont,ot0_src_cont,ot16_src_cont,ot64_src_cont,ot16nm_src_cont,ot16_pm_cont_T1,ot64_pm_cont_T1
[ -f $D0/check_d0.json ] && grep -q '"passed": true' $D0/check_d0.json \
    || { echo "ERROR: no passed D0 ($D0/check_d0.json)"; exit 1; }
mkdir -p $OUT
FAILED=
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }

echo " 1. decode lab, round 7: $(date)"
python -u $J/crt_decode_lab.py --conv_dir $D0/rc/dev/30 --dumps $D0/rc/dumps/dev/30 --out_dir $OUT/lab \
    --systems $SYS --logits_from $D0/anc/dumps/dev/30 --logits_key R3 --cb0 src --device cuda
st $? decode
echo " 2. rows: $(date)"
python -u $J/lab_rows.py --pairs $P --lab $OUT/lab r7 --out $OUT/manifest.jsonl
st $? lab_rows
echo " 3. metrics: $(date)"
python -u $J/metrics.py --in_jsonl $OUT/manifest.jsonl --out_jsonl $OUT/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? metrics
python -u $J/metrics.py --in_jsonl $OUT/manifest_src.jsonl --out_jsonl $OUT/metrics_src.jsonl \
    --metrics sim,secs --device cuda
st $? metrics_src
du -sh $OUT /datastore/nvquoc/crt
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
