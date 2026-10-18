#!/usr/bin/env python3
"""
T4 — conversion harness for the baselines of the CRT journal extension (ANALYSIS_PLAN_v1 §2.4: S5 kNN-VC, S6 FreeVC,
S7 Seed-VC, S8 FACodec; the smoke test of §8.1 item 10). Every baseline runs through its adapter module (contract in
baselines/__init__.py: SR, load, convert, describe) and gets the outputs, manifest rows, resume and error rows of
run_conversions.py, so metrics.py scores both alike.
  Systems    --systems: comma list of baselines.SYSTEMS (knnvc, freevc, seedvc, facodec; the manifest's system names).
             A system's module is imported and loaded only when the system has rows to convert (or with --record).
             With several systems the harness runs itself once per system, so each adapter gets a fresh interpreter:
             the baseline repositories share top-level module names (wavlm, models, utils). A system that needs its
             own env is run by that env's python with --systems <name>.
  Inputs     The source as run_conversions.convert_pair loads it, librosa.load(source_path, sr=SR, mono=True). The
             reference by make_pairs.load_reference(R[budget], SR), the loader run_conversions.get_reference calls,
             cached per target reference under the same key; get_reference itself needs the Mimi pipeline, so its
             cache logic is copied below. Every system thus gets the same R (the same utterances, each for its keep_s
             seconds, concatenated and cut to exactly budget_s) at its own input rate SR (Mimi's is 24 kHz). A
             shortfall entry raises in load_reference and gives shortfall error rows, as there. Nothing of CRT or
             Mimi is imported.
  Convert    adapter.convert(src, ref, SR) -> (wav, sr_out), timed per pair. random, numpy and torch are seeded with
             int(sha256(pair_id)[:8], 16) before the call, so a sampling baseline (FreeVC's prior noise, Seed-VC's
             diffusion noise) gives the same output in any shard or resumed run. wav must be finite, 1-D and at
             describe()['sr_out']; it is written as PCM_16 by run_conversions.write_wav.
Outputs under --out_dir (made absolute):
  <set>/<budget>/<system>/<pair_id>.wav
  <set>/<budget>/<system>/manifest[.shard<i>of<n>].jsonl  one row per pair, directly consumable by metrics.py: the
      run_conversions.base_row fields (pair_id, set, budget, system, tau null, wav, ref_text, parallel_wav,
      sim_ref_wavs, source_wav, R, wer_primary, source_is_shared_text, genders, source_speaker, target_speaker; dump
      null), ref_s, src_s, out_s, conv_time_s (load_src, ref_load (0 when the reference came from the cache),
      ref_cached, convert, write_wav), run_id, conv_cfg_sha. The CRT-only fields (n_frames, n_ref_frames,
      same_codes_as, coverage) are absent. A failed pair gets a row with `error` and `shortfall` (wav path kept, stale
      file removed); it is retried on the next run and fails its metrics.py row instead of vanishing. Resume skips
      finished rows written under the same conv_cfg; a different conv_cfg (describe() with the checkpoint sha256s,
      --cfg, SR, device type, library versions) is an error. At the end the file is compacted to the last row per
      pair. One manifest per system, so separate processes never share a file.
  <set>/<budget>/<system>/runs.jsonl  a start record (conv_cfg, args, host, gpu, load seconds) and an end record
      (counts, per-stage seconds, median RTF = convert / src_s) per invocation.
  baselines_record.json  with --record: per system, conv_cfg (describe() with checkpoints {path: sha256}, sr_in,
      --cfg, versions, ...), conv_cfg_sha, env (python, prefix, cache and thread variables), host, gpu, the exact
      command line and cwd, set, budget, pairs, manifest and its row counts, for the selection record (§2.4, §4.6).
      Merged into the file; an entry is replaced only under the same conv_cfg_sha. Concurrent --record runs can drop
      each other's entries, so write it from one job per system (the smoke test).
"""
import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
import traceback

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines  # noqa: E402
import make_pairs  # noqa: E402
import run_conversions as rc  # noqa: E402

DESCRIBE_KEYS = ('repo', 'commit', 'checkpoints', 'sr_out', 'settings')
VERSIONS = ('numpy', 'librosa', 'soxr', 'soundfile', 'torch', 'torchaudio', 'transformers')
ENV_VARS = ('HF_HOME', 'HF_HUB_OFFLINE', 'TORCH_HOME', 'XDG_CACHE_HOME', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES',
            'SLURM_JOB_ID')


def parse_args():
    p = argparse.ArgumentParser(allow_abbrev=False)   # child_argv rewrites --systems by its full spelling
    p.add_argument('--pairs', required=True, help='make_pairs.py JSONL')
    p.add_argument('--set', required=True, help='must match the set field of the pair rows')
    p.add_argument('--budget', default='30', help='reference budget in s: 3, 30 or 180')
    p.add_argument('--systems', required=True, help=f'comma list of {", ".join(baselines.SYSTEMS)}')
    p.add_argument('--cfg', default='{}', help="JSON {system: {keyword: value}}: overrides passed to that system's "
                   'load(); entries of systems not in this call are ignored')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--shard', default='', help='i/n with 0 <= i < n: convert pair rows k with k %% n == i')
    p.add_argument('--max_pairs', type=int, default=0, help='first N pair rows (before sharding)')
    p.add_argument('--pair_ids', default='', help='comma list: only these pair_ids')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help='cuda fails at once when CUDA is unavailable; auto falls back to cpu')
    p.add_argument('--record', action='store_true',
                   help="merge each system's entry into <out_dir>/baselines_record.json")
    args = p.parse_args()
    args.compat_spie = False   # read by run_conversions.load_pairs and base_row
    return args


def parse_systems(spec):
    names = [s.strip() for s in spec.split(',')]
    if not set(names) <= set(baselines.SYSTEMS) or len(set(names)) != len(names):
        raise SystemExit(f'ERROR: --systems {spec!r}: give distinct names of {",".join(baselines.SYSTEMS)}')
    return names


def parse_cfg(spec):
    try:
        cfg = json.loads(spec)
    except ValueError:
        cfg = None
    if not (isinstance(cfg, dict) and set(cfg) <= set(baselines.SYSTEMS)
            and all(isinstance(v, dict) for v in cfg.values())):
        raise SystemExit(f'ERROR: --cfg {spec!r} is not a JSON object {{system: {{keyword: value}}}}')
    return cfg


def child_argv(name):
    """This invocation's command line with --systems <name>."""
    argv, it = [], iter(sys.argv[1:])
    for a in it:
        if a == '--systems':
            next(it)
        elif not a.startswith('--systems='):
            argv.append(a)
    return [sys.executable, os.path.abspath(sys.argv[0]), *argv, '--systems', name]


def get_device(spec):
    if spec == 'cpu':
        return 'cpu'
    import torch
    if spec == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('ERROR: --device cuda, but CUDA is not available')
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_all(pair_id):
    """random, numpy and (once imported) torch seeded from the pair_id, so outputs do not depend on the pair order."""
    s = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16)
    random.seed(s)
    np.random.seed(s)
    if 'torch' in sys.modules:
        sys.modules['torch'].manual_seed(s)


def describe(adapter, name):
    """adapter.describe(), checked for the contract keys, with the sha256 of every checkpoint."""
    d = dict(adapter.describe())
    miss = [k for k in DESCRIBE_KEYS if k not in d]
    if miss:
        raise SystemExit(f'ERROR: baselines.{name}.describe() lacks {miss}')
    d['checkpoints'] = {os.path.abspath(p): rc.sha256_file(p) for p in d['checkpoints']}
    return d


def update_record(path, name, cfg_sha, entry=None):
    """Exits if `path` records `name` under another conv_cfg; with `entry`, merges it in."""
    rec = json.load(open(path)) if os.path.exists(path) else {}
    old = rec.get(name, {}).get('conv_cfg_sha', cfg_sha)
    if old != cfg_sha:
        raise SystemExit(f'ERROR: {path} records {name} under conv_cfg {old}, now {cfg_sha}; use another --out_dir')
    if entry is not None:
        rec[name] = dict(entry, conv_cfg_sha=cfg_sha)
        tmp = f'{path}.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, path)


# ── References, one pair ─────────────────────────────────────

def get_reference(row, args, sr, cache):
    """R[budget] by make_pairs.load_reference at sr, cached under run_conversions.get_reference's key; read-only, one
    array object per target reference. Returns (wav, load seconds, cached)."""
    entry = row['R'][args.budget]
    key = (row['target_speaker'], args.budget, tuple(entry['paths']), tuple(entry['keep_s']))
    if key in cache:
        return cache[key], 0.0, True
    t = time.time()
    wav = make_pairs.load_reference(entry, sr)
    wav.flags.writeable = False
    cache[key] = wav
    return wav, time.time() - t, False


def base_row(args, row, name):
    return dict(rc.base_row(args, row, name, None), dump=None)   # a baseline has no codes to dump


def convert_pair(adapter, sr_out, row, args, name, cache):
    tm = {}
    t = time.time()
    src = librosa.load(row['source_path'], sr=adapter.SR, mono=True)[0].astype(np.float32)
    tm['load_src'] = time.time() - t
    ref, tm['ref_load'], tm['ref_cached'] = get_reference(row, args, adapter.SR, cache)
    seed_all(row['pair_id'])
    t = time.time()
    wav, sr = adapter.convert(src, ref, adapter.SR)
    tm['convert'] = time.time() - t
    wav = np.asarray(wav)
    if wav.ndim != 1 or not len(wav) or wav.dtype.kind != 'f' or not np.isfinite(wav).all() or sr != sr_out:
        raise ValueError(f'{name}.convert returned a {wav.dtype} array of shape {wav.shape} at {sr} Hz; expected a '
                         f'finite 1-D float array at {sr_out} Hz')
    r = base_row(args, row, name)
    t = time.time()
    rc.write_wav(r['wav'], wav, sr_out)
    tm['write_wav'] = time.time() - t
    r.update(ref_s=round(len(ref) / adapter.SR, 3), src_s=round(len(src) / adapter.SR, 3),
             out_s=round(len(wav) / sr_out, 3),
             conv_time_s={k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm.items()})
    return r


# ── One system ───────────────────────────────────────────────

def run_system(args, name, rows, device):
    """Converts `rows` with the adapter of `name`. Returns the number of errors (shortfalls not counted)."""
    t_start = time.time()
    run_dir = os.path.join(args.out_dir, args.set, args.budget, name)
    os.makedirs(run_dir, exist_ok=True)
    manifest = os.path.join(run_dir, 'manifest.jsonl')
    if args.shard:
        manifest = os.path.join(run_dir, 'manifest.shard{}of{}.jsonl'.format(*args.shard))
    prev = rc.load_manifest(manifest)

    def finished(r):
        return r is not None and 'error' not in r and os.path.exists(r['wav'])
    todo = [row for row in rows if not finished(prev.get((row['pair_id'], name)))]
    if not todo and not args.record:
        print(f'run_baselines {name}: {args.set} budget {args.budget}, all {len(rows)} pairs complete -> {manifest}',
              flush=True)
        return 0

    tl = {}
    t = time.time()
    adapter = importlib.import_module(f'baselines.{name}')
    adapter.load(device, **args.cfg.get(name, {}))
    tl['load'] = time.time() - t
    t = time.time()
    desc = describe(adapter, name)
    tl['describe'] = time.time() - t
    cfg = {'system': name, 'describe': desc, 'sr_in': adapter.SR, 'cfg': args.cfg.get(name, {}), 'device_type': device,
           'reference': 'make_pairs.load_reference', 'seed': 'int(sha256(pair_id)[:8], 16)', 'wav_subtype': 'PCM_16',
           'versions': {m: getattr(sys.modules[m], '__version__', None) for m in VERSIONS if m in sys.modules}}
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    other = {r.get('conv_cfg_sha') for r in prev.values()} - {cfg_sha}
    if other:
        raise SystemExit(f'ERROR: {manifest} holds rows of conv_cfg {sorted(map(str, other))}, now {cfg_sha}; '
                         f'use another --out_dir')
    record = os.path.join(args.out_dir, 'baselines_record.json')
    if args.record:
        update_record(record, name, cfg_sha)
    run_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{socket.gethostname()}-{os.getpid()}'
    gpu = sys.modules['torch'].cuda.get_device_name() if device == 'cuda' else None
    with open(os.path.join(run_dir, 'runs.jsonl'), 'a') as f:
        f.write(json.dumps({'run_id': run_id, 'event': 'start', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'conv_cfg_sha': cfg_sha, 'conv_cfg': cfg, 'args': vars(args), 'host': socket.gethostname(),
                            'gpu': gpu, 'load_s': {k: round(v, 2) for k, v in tl.items()}}) + '\n')
    print(f'run_baselines {run_id}: {args.set} budget {args.budget}, {name} ({desc["repo"]} @ '
          f'{str(desc["commit"])[:12]}), {len(rows)} pairs, {len(todo)} to convert ({len(rows) - len(todo)} already '
          f'complete), device {device}{f" ({gpu})" if gpu else ""}, {adapter.SR} -> {desc["sr_out"]} Hz, conv_cfg '
          f'{cfg_sha}, load {json.dumps({k: round(v, 2) for k, v in tl.items()})} -> {manifest}', flush=True)

    cache, n_err, n_short, stage, rtf = {}, 0, 0, {}, []
    with open(manifest, 'a') as out:
        for i, row in enumerate(todo):
            t0 = time.time()
            try:
                r = convert_pair(adapter, desc['sr_out'], row, args, name, cache)
            except Exception as e:
                msg = f'{type(e).__name__}: {e}'
                short = msg.startswith('ValueError: reference shortfall')
                n_err += not short
                n_short += short
                print(f'  ERROR {row["pair_id"]}: {msg}\n{traceback.format_exc()}', flush=True)
                r = base_row(args, row, name)
                if os.path.exists(r['wav']):
                    os.remove(r['wav'])
                r.update(error=msg, shortfall=short)
            r.update(run_id=run_id, conv_cfg_sha=cfg_sha)
            out.write(json.dumps(r) + '\n')
            out.flush()
            if 'error' not in r:
                ts = r['conv_time_s']
                for k, v in ts.items():
                    if k != 'ref_cached' and not (k == 'ref_load' and ts['ref_cached']):
                        stage.setdefault(k, []).append(v)
                rtf.append(ts['convert'] / r['src_s'])
                print(f'  [{i + 1}/{len(todo)}] {row["pair_id"]} src={r["src_s"]:.2f}s ref={r["ref_s"]:.1f}s'
                      f'{" (cached)" if ts["ref_cached"] else ""} out={r["out_s"]:.2f}s | '
                      + ' '.join(f'{k}={v:.2f}' for k, v in ts.items() if isinstance(v, float) and v > 0)
                      + f' | {time.time() - t0:.1f}s', flush=True)

    kept = rc.load_manifest(manifest)
    with open(manifest + '.tmp', 'w') as f:
        for r in kept.values():
            f.write(json.dumps(r) + '\n')
    os.replace(manifest + '.tmp', manifest)
    summary = {k: {'n': len(v), 'mean_s': round(float(np.mean(v)), 4), 'median_s': round(float(np.median(v)), 4),
                   'max_s': round(float(np.max(v)), 4)} for k, v in stage.items()}
    rtf_median = round(float(np.median(rtf)), 4) if rtf else None
    with open(os.path.join(run_dir, 'runs.jsonl'), 'a') as f:
        f.write(json.dumps({'run_id': run_id, 'event': 'end', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'n_pairs_done': len(todo) - n_err - n_short, 'n_errors': n_err, 'n_shortfall': n_short,
                            'stage_s': summary, 'rtf_median': rtf_median,
                            'total_s': round(time.time() - t_start, 1)}) + '\n')
    if args.record:
        err = [r for r in kept.values() if 'error' in r]
        env = {'executable': sys.executable, 'prefix': sys.prefix, 'python': platform.python_version(),
               'vars': {k: os.environ.get(k) for k in ENV_VARS}}
        update_record(record, name, cfg_sha, {
            **cfg, 'env': env, 'host': socket.gethostname(), 'gpu': gpu,
            'command': shlex.join(getattr(sys, 'orig_argv', [sys.executable] + sys.argv)), 'cwd': os.getcwd(),
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'run_id': run_id, 'set': args.set, 'budget': int(args.budget),
            'pairs': os.path.abspath(args.pairs), 'manifest': manifest, 'n_rows': len(kept),
            'n_ok': len(kept) - len(err), 'n_error': sum(not r['shortfall'] for r in err),
            'n_shortfall': sum(r['shortfall'] for r in err), 'rtf_median': rtf_median})
    print('  per-stage seconds (ref_load per target reference, the rest per pair):')
    for k, v in summary.items():
        print(f'    {k:<10} n={v["n"]:<4} mean={v["mean_s"]:.3f} median={v["median_s"]:.3f} max={v["max_s"]:.3f}')
    print(f'done {name} in {time.time() - t_start:.0f}s: {len(todo) - n_err - n_short} pairs converted, {n_err} '
          f'errors, {n_short} reference shortfalls, median RTF {rtf_median} -> {manifest} ({len(kept)} rows)'
          + (f'; recorded in {record}' if args.record else ''), flush=True)
    return n_err


def main():
    args = parse_args()
    names = parse_systems(args.systems)
    args.cfg = parse_cfg(args.cfg)
    args.out_dir = os.path.abspath(args.out_dir)
    if args.shard:
        args.shard = rc.parse_shard(args.shard)
    if args.budget not in ('3', '30', '180'):
        raise SystemExit(f'ERROR: --budget {args.budget} not in 3, 30, 180')
    if len(names) > 1:   # one adapter per interpreter (docstring)
        print(f'run_baselines: {len(names)} systems, one process each: {",".join(names)}', flush=True)
        failed = [n for n in names if subprocess.call(child_argv(n)) != 0]
        if failed:
            raise SystemExit(f'ERROR: run_baselines failed for {",".join(failed)}')
        return
    rows = rc.load_pairs(args)
    device = get_device(args.device)
    if run_system(args, names[0], rows, device):
        sys.exit(1)


if __name__ == '__main__':
    main()
