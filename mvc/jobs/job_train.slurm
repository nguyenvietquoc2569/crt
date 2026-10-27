#!/bin/bash
#SBATCH --job-name=mvc_train
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_train_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# Mimi-VC (paper 3): train MimiFlow-VC (train.py) into $MVC/runs/$RUN. --export: RUN (run name), TRAIN (comma list of
# subsets under $MVC/mimi, default train-clean-100), VAL (default dev-clean), ARGS (extra train.py options), HOURS
# (time budget in h, default 70: the job checkpoints and stops before the 72 h limit; resubmitting resumes). In --export
# separate list items with '+'; a value with commas (ARGS) must come from the submitting shell's environment instead.
source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/ceai/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"; nvidia-smi -L
WORK_DIR=/datastore/ceai/crt/mimi_vc
DATA_STORAGE=$WORK_DIR/data_storage
MVC=$DATA_STORAGE/mvc
export XDG_CACHE_HOME=$DATA_STORAGE/cache HF_HOME=$DATA_STORAGE/hf_cache HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN=${RUN:?set RUN}
TRAIN=${TRAIN:-train-clean-100}
VAL=${VAL:-dev-clean}
TRAIN=${TRAIN//+/,} VAL=${VAL//+/,}     # sbatch --export splits its value at commas: pass lists as a+b
HOURS=${HOURS:-70}
TD=$(echo $TRAIN | tr , '\n' | sed "s|^|$MVC/mimi/|" | paste -sd,)
VD=$(echo $VAL | tr , '\n' | sed "s|^|$MVC/mimi/|" | paste -sd,)
cd $WORK_DIR/mvc
echo "train $TD | val $VD | args $ARGS | $(date)"
python train.py --train_dirs $TD --val_dirs $VD --out $MVC/runs/$RUN --max_hours $HOURS $ARGS
rc=$?
echo "finished rc=$rc: $(date)"
exit $rc
