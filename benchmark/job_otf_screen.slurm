#!/bin/bash
#SBATCH --job-name=crt_otf_screen
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_otf_screen_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/logs/crt_otf_screen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# Round 7b screen (pre-freeze, exploratory; 18 Oct 2026): the Gaussian optimal transport of round 7 applied to the WHOLE
# decoder input (semantic + acoustic projections; crt_decode_lab.py otf<k>), because round 7 (job 4335) showed that
# mapping the acoustic projection alone leaves the source speaker (SIM to target 0.74-0.75 vs 0.735 for the copy).
# Systems: otf0nm_src_cont (no map: checks the new decode path against the Mimi copy), otf0 (mean shift), otf16 / 64 /
# 128 (full map in the top-k PCs of the reference's decoder inputs), otf64nm (no mean shift), and CRT-AL + map
# (otf16 / otf64 of the R3 posterior-mean latent with the source CB0). Read from job D0's dumps; rows r7f_<system>;
# Whisper-large-v3 only (Dev) and the @src rows. Submit with submit_ceai.sh. Output: $J/v1/otf_screen.

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
OUT=$J/v1/otf_screen
D0=$J/v1/d0
P=$J/pairs/dev.jsonl
SYS=otf0nm_src_cont,otf0_src_cont,otf16_src_cont,otf64_src_cont,otf128_src_cont,otf64nm_src_cont,otf16_pm_cont_T1,otf64_pm_cont_T1
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
python -u $J/lab_rows.py --pairs $P --lab $OUT/lab r7f --out $OUT/manifest.jsonl
st $? lab_rows
echo " 3. metrics: $(date)"
python -u $J/metrics.py --in_jsonl $OUT/manifest.jsonl --out_jsonl $OUT/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
st $? metrics
python -u $J/metrics.py --in_jsonl $OUT/manifest_src.jsonl --out_jsonl $OUT/metrics_src.jsonl \
    --metrics sim,secs --device cuda
st $? metrics_src
du -sh $OUT /datastore/ceai/crt
echo " finished, failed steps:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
