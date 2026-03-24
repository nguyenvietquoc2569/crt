#!/usr/bin/env python3
"""
Preprocess latents: extract continuous embeddings from discrete codes.

Reads processed_merged/prepared_data.pt (speaker_data with 'codes' per speaker),
runs mimi.quantizer.decode(codes) to get [512, T] continuous embeddings,
and saves updated data with embeddings as float16 to a new directory.

Usage:
    python preprocess_latents.py \
        --input_dir  /path/to/processed_merged \
        --output_dir /path/to/processed_merged_latents \
        --batch_size 64
"""

import argparse
import os
import shutil
import time

import torch
import gc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True,
                   help='Directory with prepared_data.pt and speaker_embeddings.pkl')
    p.add_argument('--output_dir', required=True,
                   help='Output directory (new, will not overwrite input)')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Number of speakers to batch through quantizer.decode')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    input_path = os.path.join(args.input_dir, 'prepared_data.pt')
    emb_path = os.path.join(args.input_dir, 'speaker_embeddings.pkl')
    output_path = os.path.join(args.output_dir, 'prepared_data.pt')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input not found: {input_path}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy speaker_embeddings.pkl to output dir
    out_emb_path = os.path.join(args.output_dir, 'speaker_embeddings.pkl')
    if not os.path.exists(out_emb_path):
        print(f'Copying speaker_embeddings.pkl to {args.output_dir}...')
        shutil.copy2(emb_path, out_emb_path)

    # Resume support: load partial output if it exists
    if os.path.exists(output_path):
        print(f'Loading partial output from {output_path}...')
        cached = torch.load(output_path, weights_only=False, map_location='cpu')
        speaker_data = cached['speaker_data']
        done_keys = {k for k, v in speaker_data.items() if 'embeddings' in v}
        print(f'  {len(done_keys)}/{len(speaker_data)} speakers already processed')
    else:
        print(f'Loading input from {input_path}...')
        cached = torch.load(input_path, weights_only=False, map_location='cpu')
        speaker_data = cached['speaker_data']
        done_keys = set()
        print(f'  {len(speaker_data)} speakers to process')

    # Speakers still needing embeddings
    todo_keys = [k for k in speaker_data if k not in done_keys]
    if not todo_keys:
        print('All speakers already processed. Done.')
        return

    # Load Mimi quantizer
    print('Loading Mimi model for quantizer.decode()...')
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained('kyutai/mimi').to(device).eval()
    quantizer = mimi.quantizer

    t0 = time.time()
    total = len(todo_keys)
    save_every = 200  # save partial progress every N speakers
    total_frames_done = 0

    print(f'\n{"="*60}')
    print(f'Extracting continuous embeddings via quantizer.decode()')
    print(f'  Speakers to process: {total}')
    print(f'  Already done: {len(done_keys)}')
    print(f'  Device: {device}')
    print(f'{"="*60}', flush=True)

    for i, key in enumerate(todo_keys):
        codes = speaker_data[key]['codes']  # [32, T]
        T = codes.shape[1]
        with torch.no_grad():
            emb = quantizer.decode(codes.unsqueeze(0).to(device))  # [1, 512, T]
        speaker_data[key]['embeddings'] = emb[0].cpu().half()  # [512, T] fp16
        total_frames_done += T

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            sps = (i + 1) / elapsed
            eta = (total - i - 1) / sps if sps > 0 else 0
            mem_gb = total_frames_done * 512 * 2 / 1e9
            pct = (i + 1) / total * 100
            print(f'  [{i+1:4d}/{total}] {pct:5.1f}% | {sps:.1f} spk/s | '
                  f'{total_frames_done} frames ({mem_gb:.2f}GB) | '
                  f'ETA {eta:.0f}s ({eta/60:.1f}min) | '
                  f'last: {key} ({T} frames)', flush=True)

        if (i + 1) % save_every == 0:
            print(f'  >> Checkpoint: saving {len(done_keys) + i + 1} speakers...', flush=True)
            torch.save({'speaker_data': speaker_data}, output_path)
            gc.collect()

    # Final save
    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'Saving final output to {output_path}...')
    torch.save({'speaker_data': speaker_data}, output_path)

    # Stats
    total_frames = sum(v['embeddings'].shape[1] for v in speaker_data.values()
                       if 'embeddings' in v)
    size_gb = total_frames * 512 * 2 / 1e9  # fp16 = 2 bytes
    file_size = os.path.getsize(output_path) / 1e9
    print(f'Done in {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'  Speakers: {len(speaker_data)}')
    print(f'  Total frames: {total_frames}')
    print(f'  Total duration: {total_frames/12.5:.0f}s ({total_frames/12.5/3600:.1f}h)')
    print(f'  Embedding size: {size_gb:.2f} GB (fp16)')
    print(f'  File size: {file_size:.2f} GB')
    print(f'{"="*60}')

    del mimi, quantizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
