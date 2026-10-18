#!/bin/bash
# Code and weights of the baselines S5-S8 (ANALYSIS_PLAN_v1 §2.4): the commits and files that baselines/*.py were
# checked against on the HPC on 18 Oct 2026, as the selection record lists them; setup_baselines.sh builds their envs.
# Login node only (git clone and downloads; the HF files through the mimi_vc env's huggingface_hub):
#   ssh <user>@<hpc-login-host> "bash /datastore/nvquoc/crt/mimi_vc/benchmark/journal/download_baselines.sh"
# A missing clone is made at its commit; an existing one must be a clean checkout of it. A weight file that already
# has its sha256 is kept, any other one is fetched and checked, so a rerun fetches only what is missing.
#   S5 kNN-VC   clone (29 Mar 2026) loaded by torch.hub; hubconf.py's v0.1 release files in
#               $TORCH_HOME/hub/checkpoints: WavLM-Large.pt (29 Mar) and prematch_g_02500000.pt (18 Oct).
#               g_02500000.pt, also there, is not loaded (prematched=True). 3.7 MB code, 1.33 GB weights.
#   S6 FreeVC   clone (29 Mar 2026; it tracks the speaker encoder) with checkpoints/freevc.pth, §2.4's S6 checkpoint
#               (the old setup_baselines.sh fetched it with gdown; the HPC copy dates from 29 Mar; the authors' HF
#               Space holds the same bytes and is the source here) and wavlm/WavLM-Large.pt, the same file as kNN-VC's.
#               1.7 GB.
#   S7 Seed-VC  clone of the archived repository and the 8 HF files of inference.py's load_models(), downloaded at
#               main on 18 Oct 2026: DiT v1 small and its config, CAM++, BigVGAN, whisper-small. 122 MB code, 1.8 GB.
#   S8 FACodec  shallow, blob-less sparse clone of Amphion's models/codec/ns3_codec; the V2 encoder and decoder at
#               the model repo's revision 314afc3. 1 MB code, 430 MB.

set -e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python312

B=/datastore/nvquoc/crt/mimi_vc/benchmark
DATA_STORAGE=/datastore/nvquoc/crt/mimi_vc/data_storage
export XDG_CACHE_HOME=$DATA_STORAGE/cache
export HF_HOME=$DATA_STORAGE/hf_cache
export HUGGINGFACE_HUB_CACHE=$DATA_STORAGE/hf_cache/hub
export TORCH_HOME=$DATA_STORAGE/torch_cache
PY=/datastore/nvquoc/crt/envs/mimi_vc/bin/python
CK=$TORCH_HOME/hub/checkpoints

sha() { sha256sum $1 | cut -d' ' -f1; }
check() {   # file sha256
    [ "$(sha $1)" = $2 ] || { echo "ERROR: $1 does not have sha256 $2"; exit 1; }
    echo "ok $1"
}
fetch() {   # url file sha256
    if [ -f $2 ] && [ "$(sha $2)" = $3 ]; then echo "ok $2"; return; fi
    mkdir -p $(dirname $2)
    curl -fL --retry 3 -o $2.part $1
    [ "$(sha $2.part)" = $3 ] || { echo "ERROR: $1 does not have sha256 $3"; exit 1; }
    mv $2.part $2
    echo "fetched $2"
}
hf() {   # repo revision file sha256: hf_hub_download into $HF_HOME/hub
    local p
    p=$($PY -c 'import sys, huggingface_hub as h; print(h.hf_hub_download(*sys.argv[1:3], revision=sys.argv[3]))' \
        $1 $3 $2)
    check $p $4
}
pin_main() {   # repo revision: refs/main as the download at main left it; seedvc.py resolves main offline
    mkdir -p $HUGGINGFACE_HUB_CACHE/models--${1/\//--}/refs
    printf %s $2 > $HUGGINGFACE_HUB_CACHE/models--${1/\//--}/refs/main
}
at() {   # dir commit: a clean checkout of commit (tracked files)
    local g="git -C $1 --no-optional-locks"
    [ "$($g rev-parse HEAD)" = $2 ] && [ -z "$($g status --porcelain --untracked-files=no)" ] \
        || { echo "ERROR: $1 is not a clean checkout of $2"; exit 1; }
    echo "ok $1 at $2"
}
clone() {   # url dir commit
    [ -d $2 ] || { git clone $1 $2 && git -C $2 checkout -q $3; }
    at $2 $3
}

echo "=== S5 kNN-VC ==="
clone https://github.com/bshall/knn-vc.git $B/knn-vc c616845c4e309e24d5927f15adbdf277a3d65358
U=https://github.com/bshall/knn-vc/releases/download/v0.1
fetch $U/WavLM-Large.pt $CK/WavLM-Large.pt 6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f
fetch $U/prematch_g_02500000.pt $CK/prematch_g_02500000.pt \
    f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5

echo "=== S6 FreeVC ==="
F=$B/FreeVC
clone https://github.com/OlaWod/FreeVC.git $F 60032a7b1b030831ae17a4b6959656d88df6cc9c
check $F/speaker_encoder/ckpt/pretrained_bak_5805000.pt bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca
U=https://huggingface.co/spaces/OlaWod/FreeVC/resolve/f4edb1744332092e8fe6c840cc67d81d3d72a8f9
fetch $U/checkpoints/freevc.pth $F/checkpoints/freevc.pth \
    e2cc2d047f63b80d1d6780e37611cec11a01d597560393b1fe6118158b3bd47f
[ -f $F/wavlm/WavLM-Large.pt ] || cp $CK/WavLM-Large.pt $F/wavlm/
check $F/wavlm/WavLM-Large.pt 6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f

echo "=== S7 Seed-VC ==="
S=/datastore/nvquoc/crt/baselines/seed-vc
clone https://github.com/Plachtaa/seed-vc $S 51383efd921027683c89e5348211d93ff12ac2a8
R=257283f9f41585055e8f858fba4fd044e5caed6e
hf Plachta/Seed-VC $R DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth \
    8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720
hf Plachta/Seed-VC $R config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
    27a91ac41bdb41d0b6951c63642d791b4a70d60b9fc5550eb645f9ddcb3c94da
pin_main Plachta/Seed-VC $R
R=e4b6ede7ce16997aff4ae69fbca1f0175e2afede
hf funasr/campplus $R campplus_cn_common.bin 3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8
pin_main funasr/campplus $R
R=633ff708ed5b74903e86ff1298cf4a98e921c513
hf nvidia/bigvgan_v2_22khz_80band_256x $R bigvgan_generator.pt \
    e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced
hf nvidia/bigvgan_v2_22khz_80band_256x $R config.json 88a1f47acf747db0b21e97a389d838566147f7a5464583ff5c8d819d870f03ee
pin_main nvidia/bigvgan_v2_22khz_80band_256x $R
R=973afd24965f72e36ca33b3055d56a652f456b4d
hf openai/whisper-small $R model.safetensors 1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b
hf openai/whisper-small $R config.json e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4
hf openai/whisper-small $R preprocessor_config.json 9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666
pin_main openai/whisper-small $R

echo "=== S8 FACodec ==="
A=$B/facodec/Amphion
C=26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
if [ ! -d $A ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/open-mmlab/Amphion.git $A
    git -C $A sparse-checkout set models/codec/ns3_codec
    git -C $A fetch --depth 1 origin $C   # needed once main has moved past C
    git -C $A checkout -q $C
fi
at $A $C
R=314afc3ea1455ba881a0e484ef9408b6cb996736
hf amphion/naturalspeech3_facodec $R ns3_facodec_encoder_v2.bin \
    26636b05867f02f8da3690efb8c36f82909f0a8801ccc4bfdc73cdecf5f9c470
hf amphion/naturalspeech3_facodec $R ns3_facodec_decoder_v2.bin \
    e6a38d81916affae40a72f5517f39ebadeec4fefea67b074f21d4ec3a0156e3a

echo "=== Storage ==="
H=$HUGGINGFACE_HUB_CACHE
du -sh $B/knn-vc $CK/WavLM-Large.pt $CK/prematch_g_02500000.pt $F $S \
    $H/models--{Plachta--Seed-VC,funasr--campplus,nvidia--bigvgan_v2_22khz_80band_256x,openai--whisper-small} \
    $B/facodec $H/models--amphion--naturalspeech3_facodec
