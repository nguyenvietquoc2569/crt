#!/bin/bash
#SBATCH --job-name=baselines_smoke
#SBATCH --output=/datastore/ceai/crt/mimi_vc/benchmark/journal/logs/baselines_smoke_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/benchmark/journal/logs/baselines_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# Baseline smoke test (ANALYSIS_PLAN_v1 §2.4, §8.1 item 10): S5 kNN-VC, S6 FreeVC, S7 Seed-VC and S8 FACodec each
# convert the first 20 Dev screen pairs (rows k of pairs/dev.jsonl with k mod 20 = 0 in file order, k = 0, 20, ...,
# 380) at the 30 s budget with run_baselines.py into the new out dir $J/v1/baselines_smoke, one call per env; --record
# writes each system's entry (commit, checkpoint sha256, sample rate, command line) into baselines_record.json for the
# selection record (§4.6):
#   1. knnvc,freevc,facodec in the mimi_vc env (facodec.py adds envs/facodec's einops to sys.path itself)
#   2. seedvc in envs/seedvc (a venv layered over mimi_vc)
#   3. metrics.py (WER large-v3, SIM, UTMOS) on their manifests, only to check that the outputs are speech
# run_baselines.py runs each system in its own interpreter and goes on after a failed one, and every step runs whatever
# an earlier one returned. A baseline counts as running when it converts these 20 pairs and is in the record; the
# smoke numbers are not used to change any setting (§2.4). metrics.py exits 0 even with error rows, so the summary
# fails the job unless every baseline has all pairs converted, recorded and scored.
# Slurm does not create the log directory: mkdir -p $J/logs first. Submit with submit_ceai.sh.

source /etc/profile.d/modules.sh 2>/dev/null
module load python312
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo "$(git --version) (FreeVC, Seed-VC and FACodec read their commit with git)"

WORK_DIR=/datastore/ceai/crt/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
ENVS=/datastore/ceai/crt/envs
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export MPLCONFIGDIR=$DATA_STORAGE/cache/matplotlib
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=$J/v1/baselines_smoke
P=$OUT/pairs_smoke.jsonl

source $ENVS/mimi_vc/bin/activate
mkdir -p $OUT
python -c "import sys; open(sys.argv[2], 'w').writelines([l for l in open(sys.argv[1]) if l.strip()][::20][:20])" \
    $J/pairs/dev.jsonl $P || exit 1
! grep -q -e /datastore/nvquoc -e /datastore/ceai/nvquoc -e /datastore/eaic $P \
    || { echo "ERROR: $P holds paths of an old home (nvquoc, ceai/nvquoc, eaic)"; exit 1; }
echo " smoke pairs: $(wc -l < $P) rows of $J/pairs/dev.jsonl, sha256 $(sha256sum $P | cut -c1-64)"

echo " 1. kNN-VC, FreeVC, FACodec (mimi_vc env): $(date)"
python -u $J/run_baselines.py --pairs $P --set dev --budget 30 --systems knnvc,freevc,facodec --out_dir $OUT \
    --device cuda --record
S1=$?
echo " 2. Seed-VC (seedvc env): $(date)"
source $ENVS/seedvc/bin/activate
python -u $J/run_baselines.py --pairs $P --set dev --budget 30 --systems seedvc --out_dir $OUT --device cuda --record
S2=$?
source $ENVS/mimi_vc/bin/activate
echo " 3. metrics: $(date)"
cat $OUT/dev/30/*/manifest.jsonl > $OUT/manifest_smoke.jsonl
python -u $J/metrics.py --in_jsonl $OUT/manifest_smoke.jsonl --out_jsonl $OUT/metrics_smoke.jsonl \
    --metrics wer,sim,utmos --whisper_models large-v3 --device cuda
S3=$?
python - <<PY
import json, os, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, '$J')
import metrics
n = sum(1 for l in open('$P') if l.strip())
code = {'knnvc': $S1, 'freevc': $S1, 'seedvc': $S2, 'facodec': $S1}
rec = json.load(open('$OUT/baselines_record.json')) if os.path.exists('$OUT/baselines_record.json') else {}
scored = metrics.load_results('$OUT/metrics_smoke.jsonl') if os.path.exists('$OUT/metrics_smoke.jsonl') else []


def mean(rows, k):
    return np.mean([r[k] for r in rows]) if rows else float('nan')


short = []
for s in code:   # clip: PCM_16 samples at full scale (kNN-VC's -16 LUFS gain can saturate)
    f = f'$OUT/dev/30/{s}/manifest.jsonl'
    rows = metrics.load_results(f) if os.path.exists(f) else []
    ok = [r for r in rows if 'error' not in r]
    clip = [int((np.abs(sf.read(r['wav'], dtype='int16')[0].astype(np.int32)) >= 32767).sum()) for r in ok]
    m = [r for r in scored if r['system'] == s and 'error' not in r]
    wer = metrics.corpus_wer(m)['wer'] if m else float('nan')
    if len(ok) < n or s not in rec or len(m) < n:
        short.append(s)
    print(f"{s}: call exit {code[s]}, {len(ok)}/{len(rows)} pairs converted, record "
          f"{rec.get(s, {}).get('conv_cfg_sha', 'missing')}; {len(m)} scored: WER {wer:.1f} %, "
          f"SIM {mean(m, 'sim_wavlm'):.3f}, UTMOS {mean(m, 'utmos'):.2f}; {sum(c > 0 for c in clip)} wavs with "
          f"{sum(clip)} samples at full scale")
print(f"baselines without all {n} pairs converted, recorded and scored: {','.join(short) or 'none'}")
sys.exit(1 if short else 0)
PY
S4=$?
echo " finished: mimi_vc env $S1, seedvc env $S2, metrics $S3, summary $S4: $(date)"
echo " files under ~/.cache and ~/.config newer than the pair file (this job's caches are all on /datastore):"
find ~/.cache ~/.config -type f -newer $P 2>/dev/null | head -20
du -sh $OUT
exit $(( S1 || S2 || S3 || S4 ))
