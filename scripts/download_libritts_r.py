#!/usr/bin/env python3
"""
Download LibriTTS-R train-clean-360 from OpenSLR and extract.

Run this on the login node (has internet). The SLURM encode job
reads the extracted wav files offline.

Usage:
  python download_libritts_r.py --output_dir /datastore/eaic/nvquoc/mimi_vc/data_storage/libritts_r_raw
"""

import argparse
import os
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True, help='Where to download and extract')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    url = 'https://www.openslr.org/resources/141/train_clean_360.tar.gz'
    tar_path = os.path.join(args.output_dir, 'train_clean_360.tar.gz')
    extract_dir = os.path.join(args.output_dir, 'LibriTTS_R')

    # Check if already extracted
    if os.path.isdir(extract_dir) and os.path.isdir(os.path.join(extract_dir, 'train-clean-360')):
        print(f'Already extracted at {extract_dir}/train-clean-360/', flush=True)
        _print_stats(os.path.join(extract_dir, 'train-clean-360'))
        return

    # Download
    if not os.path.exists(tar_path):
        print(f'Downloading {url}...', flush=True)
        print(f'  To: {tar_path}', flush=True)
        t0 = time.time()
        subprocess.run(['wget', '-c', '-O', tar_path, url], check=True)
        elapsed = time.time() - t0
        size_gb = os.path.getsize(tar_path) / 1e9
        print(f'  Downloaded {size_gb:.1f} GB in {elapsed:.0f}s', flush=True)
    else:
        size_gb = os.path.getsize(tar_path) / 1e9
        print(f'Already downloaded: {tar_path} ({size_gb:.1f} GB)', flush=True)

    # Extract
    print(f'Extracting to {args.output_dir}...', flush=True)
    t0 = time.time()
    subprocess.run(['tar', 'xzf', tar_path, '-C', args.output_dir], check=True)
    elapsed = time.time() - t0
    print(f'  Extracted in {elapsed:.0f}s', flush=True)

    _print_stats(os.path.join(extract_dir, 'train-clean-360'))

    print(f'\nDone. You can now submit the encode SLURM job.')
    print(f'Data at: {extract_dir}/train-clean-360/')


def _print_stats(data_dir):
    """Quick stats on the extracted dataset."""
    speakers = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]
    n_wavs = 0
    for spk in speakers:
        spk_dir = os.path.join(data_dir, spk)
        for chapter in os.listdir(spk_dir):
            ch_dir = os.path.join(spk_dir, chapter)
            if os.path.isdir(ch_dir):
                n_wavs += len([f for f in os.listdir(ch_dir) if f.endswith('.wav')])
    print(f'  Speakers: {len(speakers)}', flush=True)
    print(f'  WAV files: {n_wavs}', flush=True)


if __name__ == '__main__':
    main()
