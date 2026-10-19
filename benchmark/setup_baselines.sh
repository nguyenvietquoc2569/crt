#!/bin/bash
# Envs of the baselines S5-S8 (ANALYSIS_PLAN_v1 §2.4) as built on the HPC on 18 Oct 2026; download_baselines.sh fetches
# their code and weights (either script may run first). Login node only (venv and pip downloads):
#   ssh <user>@<hpc-login-host> "bash /datastore/ceai/crt/mimi_vc/benchmark/journal/setup_baselines.sh"
#   S5 kNN-VC, S6 FreeVC  the mimi_vc env itself (python 3.12.13, torch 2.5.1+cu121; envs/build_mimi_vc_env.sh with
#                         envs/requirements_mimi_vc.txt), no new env. FreeVC's utils.py imports torchvision, which
#                         mimi_vc lacks, only for training; baselines/freevc.py puts an empty module in its place.
#   S7 Seed-VC            envs/seedvc, a venv layered over mimi_vc (baselines/seedvc_requirements.txt). 280 MB.
#   S8 FACodec            envs/facodec, einops 0.8.2 only (baselines/facodec_requirements.txt); facodec.py runs under
#                         the mimi_vc python and appends this env's site-packages to sys.path. 17 MB.
# An existing env is kept. Both are then compared with their requirements file (pip freeze of the env's own packages).

set -e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python312
E=/datastore/ceai/crt/envs
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
SP=lib/python3.12/site-packages
[ -x $E/mimi_vc/bin/python ] || { echo "ERROR: no $E/mimi_vc; build it with $E/build_mimi_vc_env.sh"; exit 1; }

echo "=== S7 Seed-VC ==="
# mimi_vc_base.pth appends mimi_vc's site-packages after the layer's own, so the layer's packages win. The 18 Oct build
# ran pip install --no-cache-dir transformers==4.46.3 tokenizers==0.20.3 huggingface-hub==0.36.2 munch==4.0.0
# einops==0.8.0 descript-audio-codec==1.0.0; seedvc_requirements.txt is the layer it resolved, installed as it is.
if [ ! -d $E/seedvc ]; then
    python3.12 -m venv $E/seedvc
    echo $E/mimi_vc/$SP > $E/seedvc/$SP/mimi_vc_base.pth
    $E/seedvc/bin/pip install --no-cache-dir --no-deps -r $J/baselines/seedvc_requirements.txt
fi

echo "=== S8 FACodec ==="
if [ ! -d $E/facodec ]; then
    python3.12 -m venv $E/facodec
    $E/facodec/bin/pip install --no-cache-dir -r $J/baselines/facodec_requirements.txt
fi

for e in seedvc facodec; do
    diff <(grep -v '^#' $J/baselines/${e}_requirements.txt) <($E/$e/bin/pip freeze --path $E/$e/$SP) \
        || { echo "ERROR: $E/$e differs from baselines/${e}_requirements.txt"; exit 1; }
    echo "ok $E/$e"
done
du -sh $E/seedvc $E/facodec
