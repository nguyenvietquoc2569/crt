#!/bin/bash
#SBATCH --job-name=mvc_encode
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_encode_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_encode_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=14:00:00
# Mimi-VC (paper 3): encode LibriTTS-R subsets with Mimi once (encode_mimi.py): continuous 512-d latents and the 32
# codes per 12.5 Hz frame, into $DATA_STORAGE/mvc/mimi/<subset>/shard_*. Resumes at the first missing shard, so a job
# that hits its limit is resubmitted unchanged. SUBSETS comes from --export (default: train-clean-100,dev-clean).
# Then encode_perturbed.py writes the codes of K=2 information-perturbed copies of every utterance (shard.pcodes<k>),
# CAM++ speaker embeddings (spk_embed.py, shard.spk.npy), and probe P0 (probe_speaker_info.py) measures the
# speaker information in CB0 / perturbed CB0 on dev-clean. In --export, separate subsets with '+', not ','.
source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/ceai/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"; nvidia-smi -L
WORK_DIR=/datastore/ceai/crt/mimi_vc
DATA_STORAGE=$WORK_DIR/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SUBSETS=${SUBSETS:-train-clean-100,dev-clean}
SUBSETS=${SUBSETS//+/,}      # sbatch --export splits its value at commas: pass SUBSETS=a+b there (job 5640 lost a subset)
cd $WORK_DIR/mvc
M=$DATA_STORAGE/mvc/mimi
python encode_mimi.py --root $DATA_STORAGE/dataset_raw/LibriTTS_R --subsets $SUBSETS --out $M
rc=$?
if [ $rc -eq 0 ]; then
  echo " perturbed: $(date)"
  DIRS=$(echo $SUBSETS | tr , '\n' | sed "s|^|$M/|" | paste -sd,)
  python encode_perturbed.py --shards $DIRS --raw $DATA_STORAGE/dataset_raw/LibriTTS_R --K 2 --workers $SLURM_CPUS_PER_TASK
  rc=$?
fi
if [ $rc -eq 0 ]; then
  echo " speaker embeddings: $(date)"
  python spk_embed.py --shards $DIRS --raw $DATA_STORAGE/dataset_raw/LibriTTS_R
  rc=$?
fi
if [ $rc -eq 0 ] && [ -d $M/dev-clean ]; then
  echo " probe P0: $(date)"
  mkdir -p $WORK_DIR/mvc/results
  python probe_speaker_info.py --shards $M/dev-clean --K 2 --out $WORK_DIR/mvc/results/p0_speaker_info.json
  rc=$?
fi
du -sh $DATA_STORAGE/mvc/mimi/* 2>/dev/null
echo "finished rc=$rc: $(date)"
exit $rc
