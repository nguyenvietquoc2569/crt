#!/usr/bin/env python3
"""
Informal A/B Listening Test for CRT Voice Conversion Paper.
Presents randomized audio pairs and collects preference ratings.

Usage:
  1. Download audio samples from cluster:
     scp -r eaic:/datastore/eaic/nvquoc/mimi_vc/benchmark/results_full/ benchmark/results_full/
     scp -r eaic:/datastore/eaic/nvquoc/mimi_vc/benchmark/results_multipair/ benchmark/results_multipair/
     scp eaic:/datastore/eaic/nvquoc/mimi_vc/source.mp3 benchmark/

  2. Run the test:
     python3 benchmark/listening_test.py --samples_dir benchmark/results_full/

  3. Results saved to benchmark/listening_test_results.json
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description='Informal listening test for VC methods')
    p.add_argument('--samples_dir', default='benchmark/results_full/',
                   help='Directory with audio samples')
    p.add_argument('--multipair_dir', default='benchmark/results_multipair/',
                   help='Directory with multi-pair samples')
    p.add_argument('--source', default='benchmark/source.mp3',
                   help='Source audio file')
    p.add_argument('--output', default='benchmark/listening_test_results.json',
                   help='Output results file')
    p.add_argument('--n_comparisons', type=int, default=10,
                   help='Number of A/B comparisons per rater')
    p.add_argument('--rater_name', default=None,
                   help='Name of the rater (prompted if not given)')
    return p.parse_args()


def play_audio(filepath):
    """Play audio file using system player."""
    if sys.platform == 'darwin':
        subprocess.run(['afplay', filepath], capture_output=True)
    elif sys.platform == 'linux':
        subprocess.run(['aplay', filepath], capture_output=True)
    else:
        print(f'  [Cannot auto-play on {sys.platform}. Open: {filepath}]')


def collect_single_pair_ratings(samples_dir, source_path):
    """Collect ratings from single-pair samples (results_full directory)."""
    methods = {}

    # Map method names to files
    file_map = {
        'CRT conf>0.4': 'crt_conf_0.4-210000-step.wav',
        'FAISS baseline': 'faiss_baseline-210000-step.wav',
        'CRT argmax': 'crt_argmax-210000-step.wav',
        'CRT conf>0.6': 'crt_conf_0.6-210000-step.wav',
    }

    # Also check for knnvc and freevc
    knnvc = os.path.join(samples_dir, 'knnvc_output.wav')
    freevc = os.path.join(samples_dir, 'freevc_output.wav')
    if os.path.exists(knnvc):
        file_map['kNN-VC'] = 'knnvc_output.wav'
    if os.path.exists(freevc):
        file_map['FreeVC'] = 'freevc_output.wav'

    available = {}
    for name, fname in file_map.items():
        path = os.path.join(samples_dir, fname)
        if os.path.exists(path):
            available[name] = path

    return available


def collect_multipair_ratings(multipair_dir):
    """Collect ratings from multi-pair samples."""
    available = {}
    if not os.path.exists(multipair_dir):
        return available

    pairs = sorted([d for d in os.listdir(multipair_dir)
                    if os.path.isdir(os.path.join(multipair_dir, d))])

    for pair_name in pairs[:5]:  # Use first 5 pairs
        pair_dir = os.path.join(multipair_dir, pair_name)
        for f in os.listdir(pair_dir):
            if f.endswith('.wav'):
                method = 'unknown'
                if '_crt04' in f:
                    method = f'CRT-{pair_name}'
                elif '_faiss' in f:
                    method = f'FAISS-{pair_name}'
                elif '_knnvc' in f:
                    method = f'kNN-VC-{pair_name}'
                elif '_freevc' in f:
                    method = f'FreeVC-{pair_name}'
                if method != 'unknown':
                    available[method] = os.path.join(pair_dir, f)

    return available


def run_mos_test(available_samples, source_path, rater_name):
    """Run Mean Opinion Score test: rate each sample 1-5."""
    print('\n' + '='*60)
    print(' PART 1: Mean Opinion Score (MOS)')
    print(' Rate each sample on naturalness and quality (1-5)')
    print(' 1=Bad  2=Poor  3=Fair  4=Good  5=Excellent')
    print('='*60)

    if source_path and os.path.exists(source_path):
        print('\n  First, listen to the SOURCE audio (reference):')
        input('  Press Enter to play source...')
        play_audio(source_path)

    # Randomize order
    items = list(available_samples.items())
    random.shuffle(items)

    ratings = {}
    for i, (name, path) in enumerate(items):
        label = f'Sample {chr(65+i)}'  # A, B, C, ...
        print(f'\n  [{i+1}/{len(items)}] {label}')
        input(f'  Press Enter to play {label}...')
        play_audio(path)

        while True:
            try:
                score = int(input(f'  Rate {label} (1-5): '))
                if 1 <= score <= 5:
                    break
                print('  Please enter 1-5')
            except ValueError:
                print('  Please enter a number 1-5')

        # Optional replay
        replay = input(f'  Replay? (y/n): ').strip().lower()
        if replay == 'y':
            play_audio(path)
            rescore = input(f'  Change rating? (Enter to keep {score}, or new 1-5): ').strip()
            if rescore and rescore.isdigit() and 1 <= int(rescore) <= 5:
                score = int(rescore)

        ratings[name] = {
            'score': score,
            'label': label,
            'file': path,
        }
        print(f'  Recorded: {label} = {score}/5')

    return ratings


def run_ab_preference(available_samples, source_path, n_comparisons):
    """Run A/B preference test between method pairs."""
    print('\n' + '='*60)
    print(' PART 2: A/B Preference Test')
    print(' For each pair, choose which sounds better')
    print(' or "tie" if equal')
    print('='*60)

    methods = list(available_samples.keys())
    if len(methods) < 2:
        print('  Not enough samples for A/B test')
        return []

    # Generate comparison pairs
    pairs = []
    key_pairs = [
        ('CRT conf>0.4', 'FAISS baseline'),
        ('CRT conf>0.4', 'kNN-VC'),
        ('CRT conf>0.4', 'FreeVC'),
        ('FAISS baseline', 'kNN-VC'),
        ('FAISS baseline', 'FreeVC'),
        ('CRT conf>0.4', 'CRT argmax'),
    ]
    for a, b in key_pairs:
        if a in available_samples and b in available_samples:
            pairs.append((a, b))

    # Limit to n_comparisons
    pairs = pairs[:n_comparisons]

    results = []
    for i, (method_a, method_b) in enumerate(pairs):
        # Randomize which is played first
        if random.random() > 0.5:
            first, second = method_a, method_b
            first_label, second_label = 'A', 'B'
        else:
            first, second = method_b, method_a
            first_label, second_label = 'A', 'B'

        print(f'\n  [{i+1}/{len(pairs)}] Comparison')

        if source_path and os.path.exists(source_path) and i == 0:
            print('  Reference (source):')
            input('  Press Enter to play source...')
            play_audio(source_path)

        input(f'  Press Enter to play Sample A...')
        play_audio(available_samples[first])

        input(f'  Press Enter to play Sample B...')
        play_audio(available_samples[second])

        while True:
            choice = input(f'  Which is better? (A/B/tie/replay): ').strip().lower()
            if choice == 'replay':
                input('  Press Enter for A...')
                play_audio(available_samples[first])
                input('  Press Enter for B...')
                play_audio(available_samples[second])
                continue
            if choice in ['a', 'b', 'tie']:
                break
            print('  Enter A, B, tie, or replay')

        winner = first if choice == 'a' else (second if choice == 'b' else 'tie')
        results.append({
            'sample_a': first,
            'sample_b': second,
            'choice': choice.upper(),
            'winner': winner,
        })
        print(f'  Recorded: {choice.upper()}')

    return results


def main():
    args = parse_args()

    print('='*60)
    print(' CRT Voice Conversion — Informal Listening Test')
    print('='*60)

    # Get rater name
    rater_name = args.rater_name
    if not rater_name:
        rater_name = input('\n  Your name: ').strip()
        if not rater_name:
            rater_name = 'anonymous'

    # Find available samples
    available = collect_single_pair_ratings(args.samples_dir, args.source)

    if not available:
        # Try multipair
        available = collect_multipair_ratings(args.multipair_dir)

    if not available:
        print(f'\n  ERROR: No audio samples found in {args.samples_dir}')
        print(f'  Download from cluster first:')
        print(f'    scp -r eaic:/datastore/eaic/nvquoc/mimi_vc/benchmark/results_full/ benchmark/results_full/')
        print(f'    scp -r eaic:/datastore/eaic/nvquoc/mimi_vc/v5.6.5-secs-high/output/ benchmark/crt_output/')
        print(f'    scp eaic:/datastore/eaic/nvquoc/mimi_vc/source.mp3 benchmark/')
        sys.exit(1)

    print(f'\n  Found {len(available)} samples:')
    for name, path in sorted(available.items()):
        size = os.path.getsize(path) // 1024
        print(f'    {name:<25} ({size}KB)')

    source = args.source if os.path.exists(args.source) else None
    if source:
        print(f'  Source audio: {source}')
    else:
        print(f'  WARNING: Source audio not found at {args.source}')

    # Run MOS test
    mos_ratings = run_mos_test(available, source, rater_name)

    # Run A/B preference test
    ab_results = run_ab_preference(available, source, args.n_comparisons)

    # Summary
    print(f'\n{"="*60}')
    print(f' RESULTS SUMMARY — Rater: {rater_name}')
    print(f'{"="*60}')

    print(f'\n  MOS Ratings:')
    print(f'  {"Method":<25} {"Score":>5}')
    print(f'  {"-"*32}')
    for name, r in sorted(mos_ratings.items(), key=lambda x: -x[1]['score']):
        print(f'  {name:<25} {r["score"]:>5}/5')

    if ab_results:
        print(f'\n  A/B Preferences:')
        for r in ab_results:
            print(f'  {r["sample_a"]} vs {r["sample_b"]} → {r["winner"]}')

    # Save
    output_data = {
        'rater': rater_name,
        'timestamp': datetime.now().isoformat(),
        'mos_ratings': mos_ratings,
        'ab_preferences': ab_results,
        'n_samples': len(available),
        'samples': {k: v for k, v in available.items()},
    }

    # Append to existing results if file exists
    all_results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
    all_results.append(output_data)

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Results saved to {args.output}')
    print(f'  Run again with different --rater_name for more raters.')
    print(f'\n  Aggregate MOS after collecting all raters:')
    print(f'    python3 -c "')
    print(f'    import json; d=json.load(open(\'{args.output}\'))')
    print(f'    methods={{}}')
    print(f'    for r in d:')
    print(f'      for m,v in r[\"mos_ratings\"].items():')
    print(f'        methods.setdefault(m,[]).append(v[\"score\"])')
    print(f'    import numpy as np')
    print(f'    for m,s in sorted(methods.items(), key=lambda x: -np.mean(x[1])):')
    print(f'      print(f\"  {{m:<25}} {{np.mean(s):.2f}} +/- {{np.std(s):.2f}} (n={{len(s)}})\")  "')


if __name__ == '__main__':
    main()
