"""
S8 FACodec (ANALYSIS_PLAN_v1 §2.4) for run_baselines.py, contract in baselines/__init__.py: the zero-shot voice
conversion of NaturalSpeech 3's factorized codec, FACodecEncoderV2 / FACodecDecoderV2, one pass, no sampling.
  Code       open-mmlab/Amphion at the commit of the sparse checkout <root>/Amphion, pinned to 26f6883 (main on 18 Oct
             2026; models/codec/ns3_codec is unchanged since 58dc870 "Refine facodec models", 13 Mar 2024). load()
             exits if a tracked file differs from the commit. Imported as Amphion.models.codec.ns3_codec, the README's
             import. Weights amphion/naturalspeech3_facodec (Apache-2.0) at revision 314afc3:
             ns3_facodec_encoder_v2.bin and ns3_facodec_decoder_v2.bin, strict load into the README's constructors
             (ENCODER, DECODER).
  Inference  The README's V2 voice-conversion example as the authors' demo Space runs it (amphion/
             naturalspeech3_facodec app.py, codec_voice_conversion_v2, with the same code and weights). Each waveform
             is zero-padded at the end by 200 - len % 200 samples; enc_out = encoder(wav); prosody =
             encoder.get_prosody_feature(wav), the first 20 of 80 log-mel bins, hop 200; decoder(enc_out, prosody,
             eval_vq=False, vq=True) gives the codes (1 prosody, 2 content and 3 residual codebooks) and spk_embs,
             the timbre encoder's mean over all frames. Output = decoder.inference(decoder.vq2emb(codes_src,
             use_residual=False), spk_embs_ref): the source's prosody and content codes with the reference's timbre,
             the residual (acoustic detail) codes dropped. R thus supplies only the timbre vector, from all of R;
             prosody comes from the source. spk_embs_ref is cached per reference. float32 under torch.no_grad, both
             modules in eval mode, 16 kHz in and out.
Deviations from the README example (the first two are the Space's code; loading and skipped outputs change no
number):
  - eval(): the V2 snippet converts without it, which would leave the dropout (p = 0.1) of the prosody and timbre
    transformer encoders active; the V1 snippet and the Space call eval() on every module.
  - Padding: the README pads nothing. Without padding an input with len % 200 >= 152 has one encoder frame more
    than mel frames, and quantize() fails adding the prosody and content streams. The output keeps the padded
    length, as in the Space: 1 to 200 samples (<= 12.5 ms) longer than the source.
  - Loading: absolute paths of the HF snapshot in place of hf_hub_download; torch.load(map_location='cpu',
    weights_only=True), then .to(device). The files are plain CPU state dicts (OrderedDict, FloatStorage), buffers
    included, so the saved mel filterbank replaces the one librosa builds in the constructor.
  - Only what the conversion uses is computed (the Space also decodes both reconstructions); in eval mode nothing
    else draws random numbers or changes state.
  - Environment: the mimi_vc env (python 3.12, torch 2.5.1, torchaudio 2.5.1, librosa 0.11, pyworld) plus einops
    0.8.2, the one ns3_codec import mimi_vc lacks, from envs/facodec (EINOPS_SITE, appended to sys.path;
    facodec_requirements.txt). The Space pins no versions.
"""
import os
import subprocess
import sys

import numpy as np

REPO = 'https://github.com/open-mmlab/Amphion'
ROOT = '/datastore/nvquoc/crt/mimi_vc/benchmark/facodec'   # holds the sparse checkout Amphion/
EINOPS_SITE = '/datastore/nvquoc/crt/envs/facodec/lib/python3.12/site-packages'
HF_REPO, HF_REVISION = 'amphion/naturalspeech3_facodec', '314afc3ea1455ba881a0e484ef9408b6cb996736'
CKPT_DIR = ('/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--amphion--naturalspeech3_facodec/'
            f'snapshots/{HF_REVISION}')
CKPTS = ('ns3_facodec_encoder_v2.bin', 'ns3_facodec_decoder_v2.bin')
ENCODER = dict(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)   # README: V2 takes the V1 parameters
DECODER = dict(in_channels=256, upsample_initial_channel=1024, ngf=32, up_ratios=[5, 5, 4, 2], vq_num_q_c=2,
               vq_num_q_p=1, vq_num_q_r=3, vq_dim=256, codebook_dim=8, codebook_size_prosody=10,
               codebook_size_content=10, codebook_size_residual=10, use_gr_x_timbre=True, use_gr_residual_f0=True,
               use_gr_residual_phone=True)
HOP = 200                                                    # README Q1: 16 kHz, hop 200
SR = 16000
M = {}


def load(device, root=ROOT, ckpt_dir=CKPT_DIR):
    repo = os.path.join(root, 'Amphion')
    git = ['git', '-C', repo, '--no-optional-locks']
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=no'], text=True)
    if changed:
        raise SystemExit(f'ERROR: {repo} differs from its commit {commit}:\n{changed}')
    sys.path.insert(0, root)
    sys.path.append(EINOPS_SITE)
    import einops
    import torch
    from Amphion.models.codec.ns3_codec import FACodecDecoderV2, FACodecEncoderV2

    enc, dec = FACodecEncoderV2(**ENCODER), FACodecDecoderV2(**DECODER)
    paths = [os.path.join(ckpt_dir, f) for f in CKPTS]
    for model, path in zip((enc, dec), paths):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.to(device).eval()
    M.update(torch=torch, device=device, enc=enc, dec=dec, commit=commit, paths=paths,
             einops=f'{einops.__version__} {os.path.dirname(einops.__file__)}', ref=None)


def convert(src, ref, sr):
    torch, enc, dec = M['torch'], M['enc'], M['dec']

    def quantize(x):   # the Space's padding, then the README's encoder and decoder(vq=True) calls
        wav = torch.from_numpy(np.pad(x, (0, HOP - len(x) % HOP))).to(M['device'])[None, None]
        return dec(enc(wav), enc.get_prosody_feature(wav), eval_vq=False, vq=True)

    with torch.no_grad():
        if ref is not M['ref']:
            M.update(ref=ref, spk_embs=quantize(ref)[4])
        vq_id = quantize(src)[1]
        audio = dec.inference(dec.vq2emb(vq_id, use_residual=False), M['spk_embs'])
    return audio[0, 0].cpu().numpy(), SR


def describe():
    return {'repo': REPO, 'commit': M['commit'], 'weights': f'{HF_REPO}@{HF_REVISION}', 'einops': M['einops'],
            'checkpoints': M['paths'], 'sr_out': SR,
            'settings': {'path': 'FACodecEncoderV2/FACodecDecoderV2 voice conversion (HF README; Space '
                                 'codec_voice_conversion_v2)',
                         'encoder': ENCODER, 'decoder': DECODER, 'strict_load': True, 'eval': True, 'eval_vq': False,
                         'pad': f'np.pad(x, (0, {HOP} - len(x) % {HOP})) on source and reference; output not trimmed',
                         'codes': 'vq2emb(source codes, use_residual=False): prosody (1) + content (2), residual (3) '
                                  'dropped',
                         'timbre': "spk_embs of the padded reference: the timbre encoder's mean over all its frames",
                         'ref_used': 'all of R[budget], for the timbre only', 'dtype': 'float32'}}
