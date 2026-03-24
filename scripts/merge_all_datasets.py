#!/usr/bin/env python3
"""
Merge all preprocessed datasets into one unified prepared_data.pt.

Merges the original data with any number of additional dataset files.

Usage:
  python merge_all_datasets.py \
    --original /path/to/processed \
    --extra /path/to/libritts100_data.pt /path/to/vctk_data.pt \
    --extra_emb /path/to/libritts100_embeddings.pkl /path/to/vctk_embeddings.pkl \
    --output /path/to/processed_merged_v2
"""

import argparse
import os
import pickle

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', required=True, help='Path to original processed/ dir')
    p.add_argument('--extra', nargs='+', required=True, help='Paths to extra *_data.pt files')
    p.add_argument('--extra_emb', nargs='+', required=True, help='Paths to extra *_embeddings.pkl files')
    p.add_argument('--output', required=True, help='Output dir for merged data')
    args = p.parse_args()

    assert len(args.extra) == len(args.extra_emb), \
        f'Mismatch: {len(args.extra)} data files vs {len(args.extra_emb)} embedding files'

    os.makedirs(args.output, exist_ok=True)

    # Load original
    print('Loading original data...', flush=True)
    orig = torch.load(os.path.join(args.original, 'prepared_data.pt'),
                      weights_only=False, map_location='cpu')
    with open(os.path.join(args.original, 'speaker_embeddings.pkl'), 'rb') as f:
        orig_emb = pickle.load(f)

    merged_sd = dict(orig['speaker_data'])
    merged_se = dict(orig_emb['speaker_embeddings'])
    print(f'  Original: {len(merged_sd)} speakers', flush=True)

    # Merge each extra dataset
    for data_path, emb_path in zip(args.extra, args.extra_emb):
        name = os.path.basename(data_path).replace('_data.pt', '')
        print(f'\nLoading {name}...', flush=True)

        new = torch.load(data_path, weights_only=False, map_location='cpu')
        with open(emb_path, 'rb') as f:
            new_emb = pickle.load(f)

        new_sd = new['speaker_data']
        new_se = new_emb['speaker_embeddings']
        print(f'  {name}: {len(new_sd)} speakers', flush=True)

        # Check for collisions
        overlap = set(merged_sd.keys()) & set(new_sd.keys())
        if overlap:
            print(f'  WARNING: {len(overlap)} overlapping keys, skipping duplicates')
            for k in overlap:
                del new_sd[k]
                if k in new_se:
                    del new_se[k]

        new_frames = sum(v['codes'].shape[1] for v in new_sd.values())
        print(f'  Adding {len(new_sd)} speakers ({new_frames/12.5/3600:.1f}h)', flush=True)
        merged_sd.update(new_sd)
        merged_se.update(new_se)

    # Save
    print(f'\n=== Merged Dataset ===')
    print(f'Total speakers: {len(merged_sd)}')

    merged_data = {'speaker_data': merged_sd}
    if 'cb_class_weights' in orig:
        merged_data['cb_class_weights'] = orig['cb_class_weights']

    out_data = os.path.join(args.output, 'prepared_data.pt')
    out_emb = os.path.join(args.output, 'speaker_embeddings.pkl')

    print(f'Saving...', flush=True)
    torch.save(merged_data, out_data)
    with open(out_emb, 'wb') as f:
        pickle.dump({'speaker_embeddings': merged_se}, f)

    print(f'  {out_data} ({os.path.getsize(out_data)/1e9:.2f} GB)')
    print(f'  {out_emb}')

    # Stats
    total_frames = sum(v['codes'].shape[1] for v in merged_sd.values())
    durations = np.array([v['codes'].shape[1] / 12.5 for v in merged_sd.values()])

    print(f'\n=== Final Statistics ===')
    print(f'Speakers:        {len(merged_sd)}')
    print(f'Total frames:    {total_frames:,}')
    print(f'Total duration:  {total_frames/12.5:.1f}s ({total_frames/12.5/3600:.1f}h)')
    print(f'Per-speaker:     min={durations.min():.1f}s, max={durations.max():.1f}s, '
          f'mean={durations.mean():.1f}s, median={np.median(durations):.1f}s')

    # Per-source breakdown
    sources = {}
    for k in merged_sd:
        src = k.split('_')[0] if '_' in k else 'original'
        if src not in sources:
            sources[src] = {'count': 0, 'frames': 0}
        sources[src]['count'] += 1
        sources[src]['frames'] += merged_sd[k]['codes'].shape[1]

    print(f'\nPer-source breakdown:')
    for src, info in sorted(sources.items()):
        print(f'  {src:<15} {info["count"]:>5} speakers, {info["frames"]/12.5/3600:.1f}h')


if __name__ == '__main__':
    main()
