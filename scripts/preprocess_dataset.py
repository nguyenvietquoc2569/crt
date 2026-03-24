#!/usr/bin/env python3
"""
Encode a speech dataset with Mimi + Resemblyzer (GPU job, offline).

Supports two directory formats:
  libritts: {speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{utt_id}.wav
  vctk:     {speaker_id}/{speaker_id}_{utt_id}_mic1.flac  (flat per-speaker)

For each speaker:
  1. Load wav files from disk (no internet needed)
  2. Batch Mimi encode -> [32, T] codes (GPU)
  3. Resemblyzer embed -> [256] speaker embedding (GPU)
  4. Save incrementally every 100 speakers (resume-capable)

Output:
  <output_dir>/<prefix>_data.pt
  <output_dir>/<prefix>_embeddings.pkl

Usage:
  python preprocess_dataset.py \
    --data_dir /path/to/LibriTTS_R/train-clean-100 \
    --output_dir /path/to/output \
    --prefix libritts100 \
    --format libritts

  python preprocess_dataset.py \
    --data_dir /path/to/VCTK-Corpus-0.92/wav48_silence_trimmed \
    --output_dir /path/to/output \
    --prefix vctk \
    --format vctk
"""

import argparse
import gc
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio


def scan_libritts(data_dir):
    """Scan LibriTTS-R directory: speaker/chapter/speaker_chapter_utt.wav"""
    speaker_wavs = defaultdict(list)
    for spk_id in sorted(os.listdir(data_dir)):
        spk_dir = os.path.join(data_dir, spk_id)
        if not os.path.isdir(spk_dir):
            continue
        for chapter in sorted(os.listdir(spk_dir)):
            ch_dir = os.path.join(spk_dir, chapter)
            if not os.path.isdir(ch_dir):
                continue
            for fname in sorted(os.listdir(ch_dir)):
                if fname.endswith('.wav'):
                    speaker_wavs[spk_id].append(os.path.join(ch_dir, fname))
    return speaker_wavs


def scan_vctk(data_dir):
    """Scan VCTK directory: speaker/speaker_utt_micX.flac (or .wav)"""
    speaker_wavs = defaultdict(list)
    for spk_id in sorted(os.listdir(data_dir)):
        spk_dir = os.path.join(data_dir, spk_id)
        if not os.path.isdir(spk_dir):
            continue
        for fname in sorted(os.listdir(spk_dir)):
            if fname.endswith('.flac') or fname.endswith('.wav'):
                # VCTK has mic1 and mic2; prefer mic1
                if '_mic2' in fname:
                    continue
                speaker_wavs[spk_id].append(os.path.join(spk_dir, fname))
    return speaker_wavs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Path to extracted dataset')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--prefix', required=True, help='Output file prefix (e.g. libritts100, vctk)')
    p.add_argument('--format', required=True, choices=['libritts', 'vctk'])
    p.add_argument('--max_sec_per_speaker', type=float, default=150.0)
    p.add_argument('--max_utts_per_speaker', type=int, default=20)
    p.add_argument('--max_utt_sec', type=float, default=15.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')

    # --- Load Mimi ---
    from transformers import MimiModel, AutoFeatureExtractor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print('Loading Mimi...', flush=True)
    mimi = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR = feat_ext.sampling_rate  # 24000
    mimi_dtype = next(mimi.parameters()).dtype
    print(f'  Mimi ready: SR={SR}, {mimi.config.num_quantizers} CBs', flush=True)

    def encode_batch(wav_list):
        if not wav_list:
            return []
        max_len = max(len(w) for w in wav_list)
        padded = np.zeros((len(wav_list), max_len), dtype=np.float32)
        masks = np.zeros((len(wav_list), max_len), dtype=np.float32)
        for i, w in enumerate(wav_list):
            padded[i, :len(w)] = w
            masks[i, :len(w)] = 1.0
        input_values = torch.from_numpy(padded).unsqueeze(1).to(device=device, dtype=mimi_dtype)
        padding_mask = torch.from_numpy(masks).to(device=device, dtype=mimi_dtype)
        with torch.no_grad():
            out = mimi.encode(input_values, padding_mask)
        codes = out.audio_codes.cpu()
        results = []
        for i, w in enumerate(wav_list):
            n_frames = int(np.ceil(len(w) / (SR / 12.5)))
            if n_frames > 0:
                results.append(codes[i, :, :n_frames])
        return results

    # --- Load Resemblyzer ---
    from resemblyzer import VoiceEncoder
    print('Loading Resemblyzer...', flush=True)
    spk_enc = VoiceEncoder(device=device)

    # --- Scan wav files ---
    print(f'Scanning {args.data_dir} (format={args.format})...', flush=True)
    if args.format == 'libritts':
        speaker_wavs = scan_libritts(args.data_dir)
    else:
        speaker_wavs = scan_vctk(args.data_dir)
    speakers = sorted(speaker_wavs.keys())
    total_wavs = sum(len(v) for v in speaker_wavs.values())
    print(f'  {len(speakers)} speakers, {total_wavs} audio files', flush=True)

    # --- Resume support ---
    partial_path = os.path.join(args.output_dir, f'{args.prefix}_partial.pt')
    speaker_data = {}
    speaker_embeddings = {}
    if os.path.exists(partial_path):
        print(f'Resuming from {partial_path}...', flush=True)
        partial = torch.load(partial_path, weights_only=False, map_location='cpu')
        speaker_data = partial['speaker_data']
        speaker_embeddings = partial.get('speaker_embeddings', {})
        print(f'  {len(speaker_data)} speakers already done', flush=True)
    done_set = set(speaker_data.keys())

    # --- Encode ---
    max_frames = int(args.max_sec_per_speaker * 12.5)
    t0 = time.time()
    for si, spk_id in enumerate(speakers):
        key = f'{args.prefix}_{spk_id}'
        if key in done_set:
            continue

        wavs = []
        total_samples = 0
        max_samples = int(args.max_sec_per_speaker * SR)
        wav_files = speaker_wavs[spk_id][:args.max_utts_per_speaker]

        for wav_path in wav_files:
            try:
                wav, orig_sr = sf.read(wav_path, dtype='float32')
                # Handle stereo -> mono
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if orig_sr != SR:
                    wav = torchaudio.functional.resample(
                        torch.from_numpy(wav).unsqueeze(0), orig_sr, SR
                    ).squeeze(0).numpy()
                wav = wav[:int(args.max_utt_sec * SR)]
                if len(wav) >= SR and total_samples + len(wav) <= max_samples:
                    wavs.append(wav)
                    total_samples += len(wav)
            except Exception as e:
                print(f'  [warn] skip {wav_path}: {e}', flush=True)

        if not wavs:
            continue

        # Mimi encode (batched)
        code_tensors = encode_batch(wavs)
        if not code_tensors:
            continue

        cat_codes = torch.cat(code_tensors, dim=1)
        if cat_codes.shape[1] > max_frames:
            cat_codes = cat_codes[:, :max_frames]

        speaker_data[key] = {'codes': cat_codes}

        # Resemblyzer embedding
        raw_audio = np.concatenate(wavs)
        wav_t = torch.from_numpy(raw_audio).unsqueeze(0).float()
        wav16k = torchaudio.functional.resample(wav_t, SR, 16000).squeeze(0).numpy()
        speaker_embeddings[key] = spk_enc.embed_utterance(wav16k)
        del raw_audio, wav16k, wav_t, wavs, code_tensors

        if (si + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'  {len(speaker_data)}/{len(speakers)} speakers | '
                  f'{elapsed:.0f}s | {cat_codes.shape[1]} frames for {key}', flush=True)

        if len(speaker_data) % 100 == 0:
            print(f'  [checkpoint] saving {len(speaker_data)} speakers...', flush=True)
            torch.save({'speaker_data': speaker_data,
                        'speaker_embeddings': speaker_embeddings}, partial_path)
            gc.collect()

    # --- Save final ---
    elapsed = time.time() - t0
    total_frames = sum(v['codes'].shape[1] for v in speaker_data.values())
    print(f'\nDone: {len(speaker_data)} speakers, {total_frames} frames '
          f'({total_frames/12.5:.0f}s = {total_frames/12.5/3600:.1f}h) in {elapsed:.0f}s', flush=True)

    data_path = os.path.join(args.output_dir, f'{args.prefix}_data.pt')
    emb_path = os.path.join(args.output_dir, f'{args.prefix}_embeddings.pkl')

    torch.save({'speaker_data': speaker_data}, data_path)
    with open(emb_path, 'wb') as f:
        pickle.dump({'speaker_embeddings': speaker_embeddings}, f)

    print(f'Saved: {data_path} ({os.path.getsize(data_path)/1e9:.2f} GB)')
    print(f'Saved: {emb_path}')

    if os.path.exists(partial_path):
        os.remove(partial_path)


if __name__ == '__main__':
    main()
