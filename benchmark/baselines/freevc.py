"""
S6 FreeVC (ANALYSIS_PLAN_v1 §2.4) for run_baselines.py, contract in baselines/__init__.py. Flagged VCTK-seen: FreeVC
is trained on VCTK 0.92 (downsample.py: the mic2 channel; filelists/train.txt, 107 speakers), so every VCTK pair
converts between speakers it has seen (baselines.md §1.1), and most of our mic1 recordings are takes it was trained on
through the other microphone: all 20 Dev and 40 VCTK-2k speakers, 191 / 200 Dev and 385 / 400 VCTK-2k source
utterances and 190 / 200 and 203 / 209 of their 30 s reference utterances are in train.txt. §5.4 keeps it out of the
listening test.
  Code       OlaWod/FreeVC at the commit of the clone in `root` (load() exits if a tracked file differs from it), with
             its own modules imported: configs/freevc.json (16 kHz, use_spk), checkpoints/freevc.pth (strict load),
             speaker_encoder/ckpt/pretrained_bak_5805000.pt and wavlm/WavLM-Large.pt.
  Inference  convert.py's loop body. g = SpeakerEncoder.embed_utterance(librosa.effects.trim(ref, top_db=20)[0]): the
             whole cut reference R[budget], concatenated by the harness, takes the place of convert.py's one target
             file; trim drops only its leading and trailing silence; the 1.6 s partials (rate 1.3, min_coverage 0.75)
             are averaged. Cached per reference. c = utils.get_content: the last WavLM-Large layer of the raw source
             waveform. net_g.infer(c, g=g): the prior encoder samples z = m + randn * exp(logs) (noise scale 1; the
             harness seeds torch per pair), the flow runs in reverse, the HiFi-GAN decoder (tanh output) gives 16 kHz.
             float32 under torch.no_grad, net_g and WavLM in eval mode, the speaker encoder as built (no dropout).
Deviations from convert.py (the same steps; only the environment changes numbers):
  - Loading: absolute paths in place of paths relative to the working directory. utils.get_cmodel is repeated with
    map_location='cpu' and .to(device), since it hard-codes 'wavlm/WavLM-Large.pt' and .cuda(rank); the speaker
    encoder is built on `device` (convert.py leaves its default, CUDA when available).
  - Inputs: the harness loads the source with librosa.load(sr=16000, mono=True), the call convert.py makes, and the
    reference with make_pairs.load_reference at 16 kHz.
  - Output: returned as float32 and written as PCM_16 by the harness (convert.py writes a float32 WAV with scipy). The
    decoder's tanh keeps every sample within [-1, 1], so nothing clips.
  - Environment: the mimi_vc env (python 3.12, torch 2.5.1, librosa 0.11, numpy 2.4, scipy 1.17) in place of
    requirements.txt's pins (torch 1.10, librosa 0.8.1, numpy 1.21, scipy 1.7, none of them built for python 3.12).
    Numerically, librosa >= 0.10 resamples with soxr_hq (0.8.1: resampy kaiser_best) and zero-pads the edge frames of
    trim's RMS and of the speaker encoder's mel spectrogram (0.8.1: reflection); torch 2.x runs WavLM's attention
    through scaled_dot_product_attention, and torch >= 1.12 runs float32 matmuls without the TF32 that 1.10 allowed
    on Ampere and later GPUs (cuDNN convolutions allow TF32 in both). utils.py imports torchvision, which mimi_vc
    lacks, only for its training augmentations (transform, stretch); an empty module stands in for it, so any use
    would fail, not change a number.
"""
import importlib.util
import json
import logging
import os
import subprocess
import sys
import types

import librosa

REPO = 'https://github.com/OlaWod/FreeVC'
ROOT = '/datastore/nvquoc/crt/mimi_vc/benchmark/FreeVC'
SPK_CKPT = 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'   # convert.py
WAVLM_CKPT = 'wavlm/WavLM-Large.pt'                          # utils.get_cmodel
TOP_DB = 20                                                  # convert.py: librosa.effects.trim of the target
SR = 16000                                                   # configs/freevc.json data.sampling_rate
M = {}


def load(device, root=ROOT, hpfile='configs/freevc.json', ptfile='checkpoints/freevc.pth'):
    """convert.py's model loading; hpfile and ptfile are its --hpfile and --ptfile, relative to root."""
    git = ['git', '-C', root, '--no-optional-locks']
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=no'], text=True)
    if changed:
        raise SystemExit(f'ERROR: {root} differs from its commit {commit}:\n{changed}')
    if importlib.util.find_spec('torchvision') is None:
        sys.modules['torchvision'] = types.ModuleType('torchvision')
    sys.path.insert(0, root)
    import torch
    import utils
    from models import SynthesizerTrn
    from speaker_encoder.voice_encoder import SpeakerEncoder
    from wavlm import WavLM, WavLMConfig
    logging.getLogger('numba').setLevel(logging.WARNING)   # as convert.py: importing utils sets root logging to DEBUG

    hpfile, ptfile = os.path.join(root, hpfile), os.path.join(root, ptfile)
    hps = utils.get_hparams_from_file(hpfile)
    if hps.data.sampling_rate != SR or not hps.model.use_spk:
        raise SystemExit(f'ERROR: {hpfile}: the adapter runs convert.py\'s use_spk path at {SR} Hz')
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length,
                           **hps.model).to(device)
    net_g.eval()
    utils.load_checkpoint(ptfile, net_g, None, True)
    ckpt = torch.load(os.path.join(root, WAVLM_CKPT), map_location='cpu')
    cmodel = WavLM(WavLMConfig(ckpt['cfg'])).to(device)
    cmodel.load_state_dict(ckpt['model'])
    cmodel.eval()
    smodel = SpeakerEncoder(os.path.join(root, SPK_CKPT), device=device)
    M.update(torch=torch, utils=utils, device=device, net_g=net_g, cmodel=cmodel, smodel=smodel, commit=commit,
             root=root, hpfile=hpfile, ptfile=ptfile, config=json.load(open(hpfile)), ref=None)


def convert(src, ref, sr):
    torch = M['torch']
    with torch.no_grad():
        if ref is not M['ref']:
            wav_tgt, _ = librosa.effects.trim(ref, top_db=TOP_DB)
            g_tgt = M['smodel'].embed_utterance(wav_tgt)
            M.update(ref=ref, g_tgt=torch.from_numpy(g_tgt).unsqueeze(0).to(M['device']))
        wav_src = torch.from_numpy(src).unsqueeze(0).to(M['device'])
        c = M['utils'].get_content(M['cmodel'], wav_src)
        audio = M['net_g'].infer(c, g=M['g_tgt'])
    return audio[0][0].data.cpu().float().numpy(), SR


def describe():
    return {'repo': REPO, 'commit': M['commit'], 'flags': ['VCTK-seen'],
            'checkpoints': [M['ptfile'], os.path.join(M['root'], SPK_CKPT), os.path.join(M['root'], WAVLM_CKPT)],
            'sr_out': SR,
            'settings': {'hpfile': M['hpfile'], 'config': M['config'], 'strict_load': True,
                         'content': 'utils.get_content: last WavLM-Large layer of the raw waveform',
                         'ref_trim_top_db': TOP_DB, 'embed_rate': 1.3, 'embed_min_coverage': 0.75,
                         'ref_used': 'all of R[budget] but the ends that trim strips; embed_utterance drops a last '
                                     'partial under 75 % coverage, at most 0.37 s of the tail',
                         'prior_noise_scale': 1.0, 'dtype': 'float32'}}
