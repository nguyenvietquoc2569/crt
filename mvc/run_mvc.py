#!/usr/bin/env python3
"""Convert a frozen pair list with Mimi-VC through the journal harness, without touching its frozen files.

It reuses run_baselines.convert_pair (source load, make_pairs.load_reference for R[budget], per-pair seeding, output
checks, atomic PCM_16 write and the manifest row schema) and run_conversions.load_pairs / load_manifest, so the rows
flow unchanged into lab_rows.py, metrics.py, score_ecapa.py and stats.py. Output: <out_dir>/<set>/<budget>/<name>/
<pair_id>.wav and manifest[.shard<i>of<n>].jsonl, resumable under the same conv_cfg_sha.

    python run_mvc.py --journal $J --pairs $J/pairs/dev.jsonl --set dev --budget 30 --name mvc_v1 \
        --cfg '{"run_dir": ".../runs/v1", "steps": 16}' --out_dir .../mvc/conv --shard 0/5
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--journal', required=True, help='the journal tooling directory ($J)')
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--set', required=True)
    ap.add_argument('--budget', default='30')
    ap.add_argument('--name', required=True)
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--shard', default='')
    ap.add_argument('--max_pairs', type=int, default=0)
    ap.add_argument('--pair_ids', default='')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    a.compat_spie = False
    a.out_dir = os.path.abspath(a.out_dir)
    return a


def run(a, adapter, rb, rc, log=print):
    rows = rc.load_pairs(a)
    run_dir = os.path.join(a.out_dir, a.set, a.budget, a.name)
    os.makedirs(run_dir, exist_ok=True)
    manifest = os.path.join(run_dir, 'manifest.jsonl' if not a.shard else 'manifest.shard{}of{}.jsonl'.format(*a.shard))
    prev = rc.load_manifest(manifest)
    todo = [r for r in rows if not (prev.get((r['pair_id'], a.name)) and 'error' not in prev[(r['pair_id'], a.name)]
                                    and os.path.exists(prev[(r['pair_id'], a.name)]['wav']))]
    if not todo:
        log(f'{a.name}: all {len(rows)} pairs complete -> {manifest}')
        return 0
    desc = rb.describe(adapter, a.name)
    cfg = {'system': a.name, 'describe': desc, 'sr_in': adapter.SR, 'device_type': a.device,
           'reference': 'make_pairs.load_reference', 'seed': 'int(sha256(pair_id)[:8], 16)', 'wav_subtype': 'PCM_16'}
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    other = {r.get('conv_cfg_sha') for r in prev.values()} - {sha}
    if other:
        raise SystemExit(f'{manifest} holds rows of conv_cfg {sorted(map(str, other))}, now {sha}; use another --out_dir')
    with open(os.path.join(run_dir, 'runs.jsonl'), 'a') as f:
        f.write(json.dumps({'event': 'start', 'time': time.strftime('%F %T'), 'conv_cfg_sha': sha, 'conv_cfg': cfg},
                           default=str) + '\n')
    log(f'{a.name}: {len(rows)} pairs, {len(todo)} to convert, conv_cfg {sha} -> {manifest}')
    cache, n_err, t0 = {}, 0, time.time()
    with open(manifest, 'a') as out:
        for i, row in enumerate(todo):
            try:
                r = rb.convert_pair(adapter, desc['sr_out'], row, a, a.name, cache)
            except Exception as e:
                msg = f'{type(e).__name__}: {e}'
                n_err += not msg.startswith('ValueError: reference shortfall')
                log(f'  ERROR {row["pair_id"]}: {msg}\n{traceback.format_exc()}')
                r = rb.base_row(a, row, a.name)
                r.update(error=msg, shortfall=msg.startswith('ValueError: reference shortfall'))
            r.update(conv_cfg_sha=sha)
            out.write(json.dumps(r) + '\n')
            out.flush()
            if (i + 1) % 50 == 0:
                log(f'  [{i + 1}/{len(todo)}] {time.time() - t0:.0f}s')
    kept = rc.load_manifest(manifest)
    with open(manifest + '.tmp', 'w') as f:
        for r in kept.values():
            f.write(json.dumps(r) + '\n')
    os.replace(manifest + '.tmp', manifest)
    log(f'{a.name}: done, {len(todo) - n_err} converted, {n_err} errors in {time.time() - t0:.0f}s -> {manifest}')
    return n_err


def main(argv=None):
    a = parse(argv)
    sys.path.insert(0, a.journal)
    import run_baselines as rb
    import run_conversions as rc
    if a.shard:
        a.shard = rc.parse_shard(a.shard)
    import vc_adapter
    ad = vc_adapter.MimiVC()
    ad.load(a.device, **json.loads(a.cfg))
    return 1 if run(a, ad, rb, rc, log=lambda m: print(m, flush=True)) else 0


if __name__ == '__main__':
    sys.exit(main())
