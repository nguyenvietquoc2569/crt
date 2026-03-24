#!/usr/bin/env python3
"""
Download speech datasets for CRT training. Run on login node (has internet).

Supported datasets:
  libritts-clean-100  LibriTTS-R train-clean-100 (247 speakers, ~33h)
  libritts-clean-360  LibriTTS-R train-clean-360 (904 speakers, ~30h)  [already done]
  vctk                VCTK Corpus (110 speakers, ~44h)

Usage:
  python download_datasets.py --dataset libritts-clean-100 --output_dir /path/to/raw
  python download_datasets.py --dataset vctk --output_dir /path/to/raw
  python download_datasets.py --dataset all --output_dir /path/to/raw
"""

import argparse
import os
import subprocess
import time


DATASETS = {
    'libritts-clean-100': {
        'url': 'https://www.openslr.org/resources/141/train_clean_100.tar.gz',
        'tar_name': 'train_clean_100.tar.gz',
        'extract_check': 'LibriTTS_R/train-clean-100',
        'format': 'libritts',
    },
    'libritts-clean-360': {
        'url': 'https://www.openslr.org/resources/141/train_clean_360.tar.gz',
        'tar_name': 'train_clean_360.tar.gz',
        'extract_check': 'LibriTTS_R/train-clean-360',
        'format': 'libritts',
    },
    'vctk': {
        'url': 'https://huggingface.co/datasets/edinburghcstr/vctk/resolve/main/VCTK-Corpus-0.92.zip',
        'tar_name': 'VCTK-Corpus-0.92.zip',
        'extract_check': 'VCTK-Corpus-0.92/wav48_silence_trimmed',
        'format': 'vctk',
    },
}


def download_and_extract(name, info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, info['tar_name'])
    check_path = os.path.join(output_dir, info['extract_check'])

    # Check if already extracted
    if os.path.isdir(check_path):
        print(f'[{name}] Already extracted at {check_path}', flush=True)
        _print_stats(name, check_path, info['format'])
        return

    # Download
    if not os.path.exists(archive_path):
        print(f'[{name}] Downloading {info["url"]}...', flush=True)
        print(f'  To: {archive_path}', flush=True)
        t0 = time.time()
        subprocess.run(['wget', '-c', '-O', archive_path, info['url']], check=True)
        elapsed = time.time() - t0
        size_gb = os.path.getsize(archive_path) / 1e9
        print(f'  Downloaded {size_gb:.1f} GB in {elapsed:.0f}s', flush=True)
    else:
        size_gb = os.path.getsize(archive_path) / 1e9
        print(f'[{name}] Already downloaded: {archive_path} ({size_gb:.1f} GB)', flush=True)

    # Extract
    print(f'[{name}] Extracting...', flush=True)
    t0 = time.time()
    if archive_path.endswith('.zip'):
        subprocess.run(['unzip', '-q', '-o', archive_path, '-d', output_dir], check=True)
    else:
        subprocess.run(['tar', 'xzf', archive_path, '-C', output_dir], check=True)
    elapsed = time.time() - t0
    print(f'  Extracted in {elapsed:.0f}s', flush=True)

    _print_stats(name, check_path, info['format'])
    print(f'[{name}] Done. Data at: {check_path}')


def _print_stats(name, data_dir, fmt):
    if fmt == 'libritts':
        speakers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        n_wavs = 0
        for spk in speakers:
            spk_dir = os.path.join(data_dir, spk)
            for chapter in os.listdir(spk_dir):
                ch_dir = os.path.join(spk_dir, chapter)
                if os.path.isdir(ch_dir):
                    n_wavs += len([f for f in os.listdir(ch_dir) if f.endswith('.wav')])
        print(f'  [{name}] {len(speakers)} speakers, {n_wavs} wav files', flush=True)
    elif fmt == 'vctk':
        speakers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        n_wavs = 0
        for spk in speakers:
            spk_dir = os.path.join(data_dir, spk)
            n_wavs += len([f for f in os.listdir(spk_dir)
                          if f.endswith('.wav') or f.endswith('.flac')])
        print(f'  [{name}] {len(speakers)} speakers, {n_wavs} audio files', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=list(DATASETS.keys()) + ['all'],
                   help='Dataset to download')
    p.add_argument('--output_dir', required=True, help='Where to download and extract')
    args = p.parse_args()

    if args.dataset == 'all':
        for name, info in DATASETS.items():
            download_and_extract(name, info, args.output_dir)
    else:
        download_and_extract(args.dataset, DATASETS[args.dataset], args.output_dir)

    print('\n========================================')
    print('All downloads complete.')
    print('Next: submit the encode SLURM job:')
    print('  sbatch scripts/preprocess_all.slurm')


if __name__ == '__main__':
    main()
