#!/usr/bin/env python3
"""
Checks of the X-mvc-1 job (mvc/ANALYSIS_PLAN_mvc_v1.md §4.10, DEVIATIONS_mvc_v1.md operational note 4), run by
job_xmvc1.slurm: R1x's 300k snapshot on VCTK-2k and LTC-2k at 30 s with mvc_v1's frozen inference settings. Standard
library only; the helpers are t1_check.py's. Every subcommand prints counts, paths and identities (sha256, commits),
never a metric value, and exits 1 after its ERROR lines. The metric, ECAPA and guard rows are checked by t1_check.py.

  cfg       the X-mvc-1 cfg JSON: the record's mvc_v1.json with only run_dir and ckpt replaced, every other key
            unchanged (and both replaced keys different from mvc_v1's); written to --out, or, when --out exists, the
            copy there must be identical.
  snapshot  the two files run_mvc.py loads from run_dir (vc_adapter.py): the snapshot and stats.npz, whose sha256 must be
            that of R1's stats.npz (--ref_stats) in FREEZE_mvc_v1.sha256 (mvc_v1's normalisation statistics); both
            sha256 in sha256sum format at --out, snapshot first; when --out exists, it must be identical (neither file
            changed between submissions).
  conv      the manifest of one set at 30 s: one row per pair of the list, of this system, set and budget; one conv_cfg
            in runs.jsonl whose describe() names --commit (the record's DESCRIBE_COMMIT: the frozen code), the cfg
            JSON's settings and the snapshot with its sha256; no shortfall row (§1: none at 30 s); other error rows are
            failed conversions (§5.4), counted, not refused; the WAV of every other row present.
  files     the sha256 of the result files named (paths relative to --root) in sha256sum format at --out, for the copy
            to the Mac (analyze_xmvc1.py and dnsmos_xmvc1.py check the copy against it); a missing file is refused.
"""
import argparse
import json
import os
import re
import sys

import t1_check as tc

CHANGED = ('run_dir', 'ckpt')
BUDGET = 30


def build_cfg(src, run_dir, ckpt, out, P):
    try:
        v1 = json.load(open(src))
    except (OSError, ValueError) as e:
        P(f'{src}: {e}')
        return
    if not isinstance(v1, dict) or v1.get('mode') != 'convert' or any(k not in v1 for k in CHANGED):
        P(f'{src} is not a mode convert cfg with run_dir and ckpt (mvc_v1.json, §2.1)')
        return
    cfg = dict(v1, run_dir=run_dir, ckpt=ckpt)
    diff = sorted(k for k in set(v1) | set(cfg) if v1.get(k, '?') != cfg.get(k, '?'))
    if diff != sorted(CHANGED):
        P(f'the X-mvc-1 cfg differs from {src} in {diff}, not exactly in {sorted(CHANGED)}')
        return
    text = json.dumps(cfg) + '\n'
    if os.path.exists(out):
        if open(out).read() != text:
            P(f'{out} exists and differs from mvc_v1.json with run_dir {run_dir} and ckpt {ckpt}')
            return
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        tmp = f'{out}.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            f.write(text)
        os.replace(tmp, out)
    print(f'cfg {out}: {src} (sha256 {tc.sha256_file(src)}) with run_dir {run_dir} and ckpt {ckpt}; the other '
          f'{len(v1) - len(CHANGED)} keys unchanged', flush=True)


def frozen_sha(freeze, path, P):
    """The sha256 FREEZE_mvc_v1.sha256 gives the absolute path, or None."""
    got = [m[1] for m in (re.fullmatch(r'([0-9a-f]{64})  (/\S+)', x.rstrip('\n')) for x in open(freeze))
           if m and m[2] == path] if os.path.isfile(freeze) else []
    if len(got) != 1:
        P(f'{freeze} gives {len(got)} sha256 of {path}, not one')
        return None
    return got[0]


def snapshot(path, stats, ref_stats, freeze, out, P):
    for p in (path, stats):
        if not os.path.isfile(p):
            P(f'{p} missing')
    want = frozen_sha(freeze, ref_stats, P)
    if P:
        return
    lines = [f'{tc.sha256_file(p)}  {p}\n' for p in (path, stats)]
    if lines[1].split()[0] != want:
        P(f'{stats} has sha256 {lines[1].split()[0]}, not that of R1\'s {ref_stats} ({want}, FREEZE_mvc_v1.sha256): '
          f'mvc_v1\'s settings convert with R1\'s normalisation statistics')
        return
    if os.path.exists(out):
        if open(out).read() != ''.join(lines):
            P(f'{out} names other files or sha256 than {path} and {stats} have now: a file changed')
            return
    else:
        tmp = f'{out}.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            f.write(''.join(lines))
        os.replace(tmp, out)
    print(f'snapshot {path} sha256 {lines[0].split()[0]}, {stats} sha256 {want} (R1\'s, as frozen) -> {out}', flush=True)


def read_snapshot(path, P):
    """(snapshot, sha256) from the snapshot subcommand's output."""
    m = re.fullmatch(r'([0-9a-f]{64})  (/\S+)\n[0-9a-f]{64}  /\S+/stats\.npz\n', open(path).read()) \
        if os.path.isfile(path) else None
    if not m:
        P(f'{path} missing or not the two sha256sum lines of the snapshot and stats.npz')
        return None, None
    return m[2], m[1]


def files(root, paths, out, P):
    miss = [p for p in paths if os.path.isabs(p) or not os.path.isfile(os.path.join(root, p))]
    if miss:
        P(f'{len(miss)} files missing under {root} or not relative paths: {miss}')
        return
    tmp = f'{out}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(''.join(f'{tc.sha256_file(os.path.join(root, p))}  {p}\n' for p in paths))
    os.replace(tmp, out)
    print(f'files {out}: sha256 of {len(paths)} files under {root}', flush=True)


def check_conv(manifest, pairs, set_, name, cfg_path, snap_file, commit, P):
    rows, ids = tc.pair_list(pairs)
    tc.shortfall_pairs(rows, set_, BUDGET, P)
    snap, snap_sha = read_snapshot(snap_file, P)
    cfg = json.load(open(cfg_path))
    tag = f'{set_} {BUDGET} s {name}'
    if not os.path.isfile(manifest):
        P(f'{tag}: {manifest} missing')
        return
    d = os.path.dirname(manifest)
    other = sorted(x for x in os.listdir(d) if x.startswith('manifest') and x.endswith('.jsonl') and
                   x != 'manifest.jsonl')
    if other:
        P(f'{tag}: {d} also holds {other}')
    by = {}
    for r in tc.read_jsonl(manifest):
        if (r.get('system'), r.get('set'), r.get('budget')) != (name, set_, BUDGET):
            P(f'{tag}: a row of {r.get("system")}, {r.get("set")}, {r.get("budget")} s in {manifest}')
        elif r['pair_id'] in by:
            P(f'{tag}: {r["pair_id"]} twice in {manifest}')
        by[r['pair_id']] = r
    if set(by) != set(ids):
        P(f'{tag}: rows on {len(set(by) & set(ids))} of the {len(ids)} pairs, {len(set(by) - set(ids))} outside the '
          f'list')
    runs = os.path.join(d, 'runs.jsonl')
    starts = [e for e in tc.read_jsonl(runs) if e.get('event') == 'start'] if os.path.isfile(runs) else []
    shas = {e.get('conv_cfg_sha') for e in starts}
    if len(shas) != 1:
        P(f'{tag}: {runs} holds {len(starts)} start events of {len(shas)} conv_cfg, not one conv_cfg')
    for e in starts:
        dsc = (e.get('conv_cfg') or {}).get('describe') or {}
        if dsc.get('commit') != commit:
            P(f'{tag}: converted with describe() commit {dsc.get("commit")}, not the record\'s DESCRIBE_COMMIT {commit}')
        s = {k: v for k, v in (dsc.get('settings') or {}).items() if k not in ('train_step', 'train_args')}
        if s != cfg:
            diff = sorted(k for k in set(s) | set(cfg) if s.get(k, '?') != cfg.get(k, '?'))
            P(f'{tag}: converted with settings that differ from {cfg_path} in {diff}')
        if snap and dsc.get('checkpoints') != {snap: snap_sha}:
            P(f'{tag}: converted with the checkpoints {dsc.get("checkpoints")}, not {{{snap}: {snap_sha}}}')
    sha = next(iter(shas)) if len(shas) == 1 else None
    off = sum(r.get('conv_cfg_sha') != sha for r in by.values())
    if off:
        P(f'{tag}: {off} rows not of the conv_cfg {sha} of {runs}')
    sf = sum('error' in r and r.get('shortfall') is True for r in by.values())
    if sf:
        P(f'{tag}: {sf} shortfall rows; §1 has no reference shortfall at {BUDGET} s')
    failed = sum('error' in r and r.get('shortfall') is not True for r in by.values())
    nowav = sum(not os.path.isfile(r.get('wav', '')) for r in by.values() if 'error' not in r)
    if nowav:
        P(f'{tag}: {nowav} rows without error whose WAV is missing')
    print(f'  conv {tag}: {len(by)} rows for {len(ids)} pairs, {failed} failed (error rows; once every pair has a row '
          f'they are failed conversions, §5.4), conv_cfg {sha}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('cfg')
    s.add_argument('--src', required=True, help="the record's mvc_v1.json")
    s.add_argument('--run_dir', required=True)
    s.add_argument('--ckpt', required=True)
    s.add_argument('--out', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--snapshot', required=True)
    s.add_argument('--stats', required=True, help='run_dir/stats.npz')
    s.add_argument('--ref_stats', required=True, help="R1's stats.npz, as FREEZE_mvc_v1.sha256 names it")
    s.add_argument('--freeze', required=True, help='FREEZE_mvc_v1.sha256')
    s.add_argument('--out', required=True)
    s = sub.add_parser('conv')
    s.add_argument('--manifest', required=True)
    s.add_argument('--pairs', required=True)
    s.add_argument('--set', required=True, choices=tc.SETS)
    s.add_argument('--name', required=True)
    s.add_argument('--cfg', required=True)
    s.add_argument('--snapshot_sha', required=True, help='the snapshot subcommand\'s output')
    s.add_argument('--commit', required=True, help="the record's DESCRIBE_COMMIT")
    s = sub.add_parser('files')
    s.add_argument('--root', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('paths', nargs='+')
    a = ap.parse_args(argv)
    P = tc.Problems()
    if a.cmd == 'cfg':
        build_cfg(a.src, a.run_dir, a.ckpt, a.out, P)
    elif a.cmd == 'snapshot':
        snapshot(a.snapshot, a.stats, a.ref_stats, a.freeze, a.out, P)
    elif a.cmd == 'conv':
        check_conv(a.manifest, a.pairs, a.set, a.name, a.cfg, a.snapshot_sha, a.commit, P)
    elif a.cmd == 'files':
        files(a.root, a.paths, a.out, P)
    for p in P:
        print(f'ERROR: {p}', flush=True)
    return 1 if P else 0


if __name__ == '__main__':
    sys.exit(main())
