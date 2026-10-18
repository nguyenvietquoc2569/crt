"""
S7 Seed-VC v1 small (ANALYSIS_PLAN_v1 §2.4) for run_baselines.py. Code: Plachtaa/seed-vc at its last commit
51383efd (2025-04-20; the repository is archived), cloned to REPO. Model: the offline VC model
seed-uvit-whisper-small-wavenet (v1.0, 98M), HF Plachta/Seed-VC
DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth with config_dit_mel_seed_uvit_whisper_small_wavenet.yml
(its model and mel parameters equal the repository preset);
content from the openai/whisper-small encoder in float16, style from CAM++ (funasr/campplus campplus_cn_common.bin,
byte-identical to the copy in the repository), vocoder BigVGAN nvidia/bigvgan_v2_22khz_80band_256x. 22.05 kHz in and
out.
  load()     inference.py load_models() on its default branch (no --checkpoint, --f0-condition False: DiT, whisper,
             bigvgan), with the repository's own modules.
  convert()  inference.py main() from the loaded source and reference on (l. 272-400), line for line: Whisper
             features of the 16 kHz source (30 s windows with 5 s overlap beyond 30 s) and reference, 80-band mels,
             the CAM++ embedding of the reference's mean-normalised kaldi fbank, the length regulator, then the
             source condition in windows of 30 s of mel frames minus the reference's, each with the whole reference
             as prompt: CFM Euler solver with CFG under float16 autocast, BigVGAN, 16-frame cos^2 crossfades. With a
             25 s reference the window is 427 frames (4.96 s), so longer sources are converted in several windows.
  Settings   the pinned commit's defaults: inference_cfg_rate 0.7, length_adjust 1.0, fp16 True; diffusion_steps 25
             (§2.4: the README's documented default and recommended command; inference.py's argparse default is 30).
             The diffusion noise is torch.randn on the device, so the harness seed fixes it.
Deviations from inference.py:
  - weights come from the shared HF cache (HF_HOME; run with HF_HUB_OFFLINE=1) instead of ./checkpoints and
    HF_HUB_CACHE=./checkpoints/hf_cache; same repositories and files;
  - the reference is R[budget] (make_pairs.load_reference: several utterances concatenated) instead of one file.
    Upstream keeps its first 25 s (ref_audio[:sr * 25]), so the 30 s and 180 s budgets both use 25 s (ref_max_s);
  - the device is the harness's (cuda or cpu, never mps); warnings are not silenced; no RTF print; the output is
    returned as float32 (the harness writes PCM_16) instead of torchaudio.save as a float WAV.
Environment: /datastore/nvquoc/crt/envs/seedvc, a venv layered over mimi_vc (seedvc_requirements.txt): Seed-VC's
transformers 4.46.3, tokenizers 0.20.3 and a 0.x huggingface_hub (BigVGAN's from_pretrained needs it) over mimi_vc's
torch 2.5.1+cu121, numpy 2.4.3 and librosa 0.11.0, where the repository pins torch 2.4.0, numpy 1.26.4 and
librosa 0.10.2. describe() lists the four weight files (Whisper's is model.safetensors, the only weights file in the
cache) and their configs; the HF snapshot commits are in the paths.
"""
import subprocess
import sys

import numpy as np

REPO = '/datastore/nvquoc/crt/baselines/seed-vc'
REPO_URL = 'https://github.com/Plachtaa/seed-vc'
DIT = ('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth',
       'config_dit_mel_seed_uvit_whisper_small_wavenet.yml')
SR = 22050
_st = {}


def load(device, diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7, fp16=True):
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, WhisperModel
    sys.path.insert(0, REPO)
    from modules.audio import mel_spectrogram
    from modules.bigvgan import bigvgan
    from modules.campplus.DTDNN import CAMPPlus
    from modules.commons import build_model, load_checkpoint, recursive_munch

    device = torch.device(device)
    dit_checkpoint_path, dit_config_path = (hf_hub_download(DIT[0], f) for f in DIT[1:])
    config = yaml.safe_load(open(dit_config_path, 'r'))
    model_params = recursive_munch(config['model_params'])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage='DiT')
    model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path, load_only_params=True, ignore_modules=[],
                                     is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    campplus_ckpt_path = hf_hub_download('funasr/campplus', 'campplus_cn_common.bin')
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location='cpu'))
    campplus_model.eval()
    campplus_model.to(device)

    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                               return_tensors='pt',
                                               return_attention_mask=True)
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    sp = config['preprocess_params']['spect_params']
    mel_fn_args = {'n_fft': sp['n_fft'], 'win_size': sp['win_length'], 'hop_size': sp['hop_length'],
                   'num_mels': sp['n_mels'], 'sampling_rate': config['preprocess_params']['sr'],
                   'fmin': sp.get('fmin', 0), 'fmax': None if sp.get('fmax', 'None') == 'None' else 8000,
                   'center': False}

    git = ['git', '-C', REPO]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=no'], text=True).strip():
        commit += '-dirty'
    bigvgan_files = [hf_hub_download(bigvgan_name, f) for f in ('bigvgan_generator.pt', 'config.json')]
    whisper_files = [hf_hub_download(whisper_name, f)
                     for f in ('model.safetensors', 'config.json', 'preprocessor_config.json')]
    _st.update(device=device, model=model, campplus=campplus_model, vocoder=bigvgan_model, semantic_fn=semantic_fn,
               mel_fn=lambda x: mel_spectrogram(x, **mel_fn_args), commit=commit,
               checkpoints=[dit_checkpoint_path, campplus_ckpt_path, bigvgan_files[0], whisper_files[0]],
               configs=[dit_config_path, bigvgan_files[1], *whisper_files[1:]],
               settings={'model': 'seed-uvit-whisper-small-wavenet', 'diffusion_steps': diffusion_steps,
                         'length_adjust': length_adjust, 'inference_cfg_rate': inference_cfg_rate, 'fp16': fp16,
                         'f0_condition': False, 'auto_f0_adjust': False, 'semi_tone_shift': 0, 'ref_max_s': 25,
                         'context_window_s': 30, 'overlap_frames': 16, 'n_quantizers': 3, 'temperature': 1.0,
                         'whisper_dtype': 'float16', 'whisper_window_s': 30, 'whisper_overlap_s': 5,
                         'style': 'campplus on kaldi fbank (80 bins, dither 0), mean-normalised',
                         'vocoder': 'bigvgan, use_cuda_kernel False, weight norm removed',
                         'setup_caches': {'max_batch_size': 1, 'max_seq_length': 8192}, 'mel': mel_fn_args})


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def convert(src, ref, sr):
    import torch
    import torchaudio
    model, semantic_fn, vocoder_fn, campplus_model, mel_fn, device = (
        _st[k] for k in ('model', 'semantic_fn', 'vocoder', 'campplus', 'mel_fn', 'device'))
    s = _st['settings']
    diffusion_steps, length_adjust, inference_cfg_rate, fp16 = (
        s[k] for k in ('diffusion_steps', 'length_adjust', 'inference_cfg_rate', 'fp16'))
    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    with torch.no_grad():
        source_audio = torch.tensor(src).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref[:sr * 25]).unsqueeze(0).float().to(device)

        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([buffer, converted_waves_16k[
                        :, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
                S_alt = semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k)

        mel = mel_fn(source_audio.to(device).float())
        mel2 = mel_fn(ref_audio.to(device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        prompt_condition = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)[0]

        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
                vc_target = model.cfm.inference(cat_condition,
                                                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                mel2, style2, None, diffusion_steps,
                                                inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = vocoder_fn(vc_target.float()).squeeze()
            vc_wave = vc_wave[None, :]
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
            else:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                        overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
    return np.concatenate(generated_wave_chunks), sr


def describe():
    return {'repo': REPO_URL, 'commit': _st['commit'], 'repo_dir': REPO, 'checkpoints': _st['checkpoints'],
            'configs': _st['configs'], 'sr_out': SR, 'settings': _st['settings']}
