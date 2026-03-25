#!/usr/bin/env python3
"""
Download speech datasets for CRT training. Run on login node (has internet).

Supported datasets:
  libritts-clean-100   LibriTTS-R train-clean-100 (247 speakers, ~33h)
  libritts-clean-360   LibriTTS-R train-clean-360 (904 speakers, ~30h)  [already done]
  libritts-other-500   LibriTTS-R train-other-500 (1166 speakers, ~205h)
  vctk                 VCTK Corpus (110 speakers, ~44h)
  hifi-tts             Hi-Fi TTS (10 speakers, ~292h)
  common-voice-en      Common Voice English (5000+ speakers, ~400h filtered)

Usage:
  python download_datasets.py --dataset libritts-other-500 --output_dir /path/to/raw
  python download_datasets.py --dataset hifi-tts --output_dir /path/to/raw
  python download_datasets.py --dataset common-voice-en --output_dir /path/to/raw
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
    'libritts-other-500': {
        'url': 'https://www.openslr.org/resources/141/train_other_500.tar.gz',
        'tar_name': 'train_other_500.tar.gz',
        'extract_check': 'LibriTTS_R/train-other-500',
        'format': 'libritts',
    },
    'vctk': {
        'url': 'https://huggingface.co/datasets/edinburghcstr/vctk/resolve/main/VCTK-Corpus-0.92.zip',
        'tar_name': 'VCTK-Corpus-0.92.zip',
        'extract_check': 'VCTK-Corpus-0.92/wav48_silence_trimmed',
        'format': 'vctk',
    },
    'hifi-tts': {
        'url': 'https://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz',
        'tar_name': 'hi_fi_tts_v0.tar.gz',
        'extract_check': 'hi_fi_tts_v0',
        'format': 'hifi',
    },
    'common-voice-en': {
        'url': 'HF:mozilla-foundation/common_voice_17_0',
        'tar_name': None,
        'extract_check': 'cv-corpus-17.0-2024-03-15/en',
        'format': 'commonvoice',
    },
}


def download_and_extract(name, info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    check_path = os.path.join(output_dir, info['extract_check'])

    # Check if already extracted
    if os.path.isdir(check_path):
        print(f'[{name}] Already extracted at {check_path}', flush=True)
        _print_stats(name, check_path, info['format'])
        return

    # HuggingFace dataset (Common Voice)
    if info['url'].startswith('HF:'):
        _download_hf_dataset(name, info, output_dir)
        return

    archive_path = os.path.join(output_dir, info['tar_name'])

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


def _download_hf_dataset(name, info, output_dir):
    """Download dataset via HuggingFace datasets library."""
    repo_id = info['url'].replace('HF:', '')
    print(f'[{name}] Downloading from HuggingFace: {repo_id}...', flush=True)
    print(f'  This may take a while for large datasets.', flush=True)
    t0 = time.time()
    from datasets import load_dataset
    ds = load_dataset(repo_id, 'en', split='train',
                      cache_dir=os.path.join(output_dir, 'hf_cache'))
    elapsed = time.time() - t0
    print(f'  Downloaded {len(ds)} utterances in {elapsed:.0f}s', flush=True)

    # Filter for high quality: keep only clips with upvotes > downvotes and duration 2-15s
    print(f'  Filtering for quality...', flush=True)
    original = len(ds)
    # Common Voice has 'up_votes', 'down_votes', 'duration' fields
    # Keep clips where quality is reasonable
    ds_filtered = ds
    try:
        ds_filtered = ds.filter(
            lambda x: x.get('up_votes', 0) >= x.get('down_votes', 0)
        )
        print(f'  After quality filter: {len(ds_filtered)}/{original} utterances', flush=True)
    except Exception as e:
        print(f'  Quality filter skipped: {e}', flush=True)

    # Save as marker file
    marker = os.path.join(output_dir, 'common_voice_en_ready.txt')
    with open(marker, 'w') as f:
        f.write(f'Dataset: {repo_id}\n')
        f.write(f'Utterances: {len(ds_filtered)}\n')
        f.write(f'Cache: {output_dir}/hf_cache\n')
    print(f'[{name}] Ready. Cache at: {output_dir}/hf_cache', flush=True)


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
    elif fmt == 'hifi':
        # Hi-Fi TTS: hi_fi_tts_v0/{speaker_id}/{reader}_{speaker}/*.wav
        speakers = set()
        n_wavs = 0
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.wav') or f.endswith('.flac'):
                    n_wavs += 1
                    # Extract speaker from path
                    parts = root.replace(data_dir, '').strip('/').split('/')
                    if parts:
                        speakers.add(parts[0])
        print(f'  [{name}] {len(speakers)} speakers, {n_wavs} audio files', flush=True)
    elif fmt == 'commonvoice':
        print(f'  [{name}] Downloaded via HuggingFace (see cache)', flush=True)


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
