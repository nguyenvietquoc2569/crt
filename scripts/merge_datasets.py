#!/usr/bin/env python3
"""
Merge existing prepared_data.pt with new LibriTTS-R data.

Usage:
  python merge_datasets.py \
    --original /path/to/data_storage/processed \
    --new_data /path/to/libritts_r_data.pt \
    --new_emb  /path/to/libritts_r_embeddings.pkl \
    --output   /path/to/processed_merged
"""

import argparse
import os
import pickle

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', required=True, help='Path to original processed/ dir')
    p.add_argument('--new_data', required=True, help='Path to libritts_r_data.pt')
    p.add_argument('--new_emb', required=True, help='Path to libritts_r_embeddings.pkl')
    p.add_argument('--output', required=True, help='Output dir for merged data')
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load original
    print('Loading original data...', flush=True)
    orig = torch.load(os.path.join(args.original, 'prepared_data.pt'),
                      weights_only=False, map_location='cpu')
    with open(os.path.join(args.original, 'speaker_embeddings.pkl'), 'rb') as f:
        orig_emb = pickle.load(f)

    orig_sd = orig['speaker_data']
    orig_se = orig_emb['speaker_embeddings']
    print(f'  Original: {len(orig_sd)} speakers', flush=True)

    # Load new
    print('Loading LibriTTS-R data...', flush=True)
    new = torch.load(args.new_data, weights_only=False, map_location='cpu')
    with open(args.new_emb, 'rb') as f:
        new_emb = pickle.load(f)

    new_sd = new['speaker_data']
    new_se = new_emb['speaker_embeddings']
    print(f'  LibriTTS-R: {len(new_sd)} speakers', flush=True)

    # Check for collisions
    overlap = set(orig_sd.keys()) & set(new_sd.keys())
    if overlap:
        print(f'  WARNING: {len(overlap)} overlapping keys: {list(overlap)[:5]}...')
        print(f'  Skipping overlapping speakers from new data.')
        for k in overlap:
            del new_sd[k]
            if k in new_se:
                del new_se[k]

    # Merge
    merged_sd = {**orig_sd, **new_sd}
    merged_se = {**orig_se, **new_se}
    print(f'\nMerged: {len(merged_sd)} speakers ({len(orig_sd)} + {len(new_sd)})', flush=True)

    # Also carry over cb_class_weights if present (recompute would be better but this works)
    merged_data = {'speaker_data': merged_sd}
    if 'cb_class_weights' in orig:
        merged_data['cb_class_weights'] = orig['cb_class_weights']

    # Save
    out_data = os.path.join(args.output, 'prepared_data.pt')
    out_emb = os.path.join(args.output, 'speaker_embeddings.pkl')

    print(f'Saving merged data...', flush=True)
    torch.save(merged_data, out_data)
    with open(out_emb, 'wb') as f:
        pickle.dump({'speaker_embeddings': merged_se}, f)

    print(f'  {out_data} ({os.path.getsize(out_data)/1e9:.2f} GB)')
    print(f'  {out_emb}')

    # Stats
    frame_rate = 12.5
    total_frames = sum(v['codes'].shape[1] for v in merged_sd.values())
    durations = [v['codes'].shape[1] / frame_rate for v in merged_sd.values()]
    durations = np.array(durations)

    print(f'\n=== Merged Dataset Statistics ===')
    print(f'Speakers:        {len(merged_sd)}')
    print(f'Total frames:    {total_frames:,}')
    print(f'Total duration:  {total_frames/frame_rate:.1f}s ({total_frames/frame_rate/3600:.1f}h)')
    print(f'Per-speaker:     min={durations.min():.1f}s, max={durations.max():.1f}s, '
          f'mean={durations.mean():.1f}s, median={np.median(durations):.1f}s')

    # Source breakdown
    orig_frames = sum(v['codes'].shape[1] for v in orig_sd.values())
    new_frames = sum(v['codes'].shape[1] for v in new_sd.values())
    print(f'\nOriginal:   {len(orig_sd):>5} speakers, {orig_frames/frame_rate/3600:.1f}h')
    print(f'LibriTTS-R: {len(new_sd):>5} speakers, {new_frames/frame_rate/3600:.1f}h')
    print(f'Total:      {len(merged_sd):>5} speakers, {total_frames/frame_rate/3600:.1f}h')


if __name__ == '__main__':
    main()
