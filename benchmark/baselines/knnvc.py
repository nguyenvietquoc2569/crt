"""
S5 kNN-VC (ANALYSIS_PLAN_v1 §2.4), adapter for run_baselines.py (contract in baselines/__init__.py).
  Code       bshall/knn-vc at COMMIT (upstream master since 16 Jan 2026), the clean git checkout REPO_DIR; its 18
             files are byte-identical to the torch.hub cache copy torch_cache/hub/bshall_knn-vc_master. Loaded by
             torch.hub from REPO_DIR (source='local': no GitHub lookup, no download of the code).
  Weights    hubconf.py reads <torch.hub.get_dir()>/checkpoints/WavLM-Large.pt and prematch_g_02500000.pt, the v0.1
             release files of its URLs, so TORCH_HOME must be /datastore/nvquoc/crt/mimi_vc/data_storage/torch_cache
             as in the job scripts. load() stops when one is missing instead of letting hubconf download it to
             another cache.
  Inference  the README quickstart with §2.4's prematched=True and k = 4 (also upstream's defaults), every other
             argument at its default:
               knn_vc = torch.hub.load(REPO_DIR, 'knn_vc', source='local', prematched=True, pretrained=True)
               query_seq = knn_vc.get_features(src)            # WavLM-Large layer 6, no VAD (vad_trigger_level 0)
               matching_set = knn_vc.get_matching_set([ref])   # torchaudio Vad at trigger level 7, layer 6
               out_wav = knn_vc.match(query_seq, matching_set, topk=4)   # synth_set = matching set, -16 LUFS out
             float32, 16 kHz in and out (hifigan/config_v1_wavlm.json). The matching set is computed once per
             reference object (the harness passes one array per target reference).
Deviations from upstream:
  - The harness hands over R[budget] as one waveform (each utterance for its keep_s seconds, concatenated, cut to
    budget_s), so the matching set comes from one WavLM pass over the whole reference and the VAD trims only its two
    ends; upstream's get_matching_set(paths) encodes and trims each reference utterance separately. The SPIE
    comparison also passed one concatenated reference (run_multipair_benchmark.py, run_knnvc_pair).
  - Source and reference are resampled to 16 kHz by the harness (librosa.load, soxr_hq), not by
    torchaudio.functional.resample in get_features.
match()'s gain to -16 LUFS can push peaks past full scale; soundfile 0.13.1 opens every file with libsndfile clipping
on, so run_conversions.write_wav saturates them in the PCM_16 file, as sf.write of upstream's output would.
describe()['code_sha256'] hashes every repository file knn_vc imports or reads, so an edited checkout changes
conv_cfg_sha instead of passing as COMMIT.
"""
import hashlib
import os
import sys

import torch

REPO = 'https://github.com/bshall/knn-vc'
COMMIT = 'c616845c4e309e24d5927f15adbdf277a3d65358'
REPO_DIR = '/datastore/nvquoc/crt/mimi_vc/benchmark/knn-vc'
CODE = ('hubconf.py', 'matcher.py', 'knnvc_utils.py', 'wavlm/WavLM.py', 'wavlm/modules.py', 'hifigan/models.py',
        'hifigan/utils.py', 'hifigan/config_v1_wavlm.json')
SR = 16000
DEFAULTS = {'prematched': True, 'topk': 4, 'ref_vad_trigger_level': 7, 'tgt_loudness_db': -16}

_knn_vc, _cfg, _ckpts = None, None, None
_matching_sets = {}   # id(ref) -> (ref, matching set); holding ref keeps its id from being reused


def load(device, **cfg):
    global _knn_vc, _cfg, _ckpts
    if set(cfg) - set(DEFAULTS):
        raise SystemExit(f'ERROR: knnvc cfg keys {sorted(set(cfg) - set(DEFAULTS))} not in {sorted(DEFAULTS)}')
    _cfg = dict(DEFAULTS, **cfg)
    d = os.path.join(torch.hub.get_dir(), 'checkpoints')
    _ckpts = [os.path.join(d, 'WavLM-Large.pt'),
              os.path.join(d, ('prematch_' if _cfg['prematched'] else '') + 'g_02500000.pt')]
    missing = [p for p in _ckpts if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'ERROR: kNN-VC weights missing: {missing} (TORCH_HOME={os.environ.get("TORCH_HOME")})')
    _knn_vc = torch.hub.load(REPO_DIR, 'knn_vc', source='local', prematched=_cfg['prematched'], pretrained=True,
                             progress=False, device=device)


def convert(src, ref, sr):
    if id(ref) not in _matching_sets:
        _matching_sets[id(ref)] = (ref, _knn_vc.get_matching_set([torch.from_numpy(ref.copy())],
                                                                 vad_trigger_level=_cfg['ref_vad_trigger_level']))
    query_seq = _knn_vc.get_features(torch.from_numpy(src))
    wav = _knn_vc.match(query_seq, _matching_sets[id(ref)][1], topk=_cfg['topk'],
                        tgt_loudness_db=_cfg['tgt_loudness_db'])
    return wav.numpy(), _knn_vc.sr


def describe():
    return {'repo': REPO, 'commit': COMMIT, 'repo_dir': REPO_DIR, 'checkpoints': _ckpts, 'sr_out': _knn_vc.sr,
            'settings': {**_cfg, 'wavlm_layer': sys.modules['matcher'].SPEAKER_INFORMATION_LAYER,
                         'matching_set': 'get_matching_set([ref]): all of R[budget] as one waveform, VAD-trimmed at '
                                         'its two ends', 'src_vad_trigger_level': 0, 'synth_set': 'the matching set',
                         'target_duration': None, 'dtype': 'float32',
                         'resample': '16 kHz by the harness (librosa.load, soxr_hq)'},
            'code_sha256': {f: hashlib.sha256(open(os.path.join(REPO_DIR, f), 'rb').read()).hexdigest() for f in CODE}}
