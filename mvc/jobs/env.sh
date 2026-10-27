# Sourced by the mvc job scripts: modules, the frozen mimi_vc env, offline caches and the shared paths.
source /etc/profile.d/modules.sh 2>/dev/null
module load python312
source /datastore/ceai/crt/envs/mimi_vc/bin/activate
echo "Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES"; nvidia-smi -L
WORK_DIR=/datastore/ceai/crt/mimi_vc
J=$WORK_DIR/benchmark/journal
DATA_STORAGE=$WORK_DIR/data_storage
MVC=$DATA_STORAGE/mvc
RAW=$DATA_STORAGE/dataset_raw/LibriTTS_R
export XDG_CACHE_HOME=$DATA_STORAGE/cache HF_HOME=$DATA_STORAGE/hf_cache HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
dirs() { echo $1 | tr , '\n' | sed "s|^|$MVC/mimi/|" | paste -sd,; }    # subset list -> shard dirs
cd $WORK_DIR/mvc
