#!/usr/bin/env python3
"""
crt_decode_lab.py outputs as metrics.py input, SIM_src included (ANALYSIS_PLAN_v1.md §3.1 M8, §8.1 item 5).

crt_decode_lab.py names its rows lab_<system> in every --out_dir, copies them from the run_conversions.py faiss row,
which has no sim_src_wavs, and writes no @src manifest. This helper
  1. merges: each --lab <out_dir> <tag> reads <out_dir>/manifest.jsonl and renames its rows lab_<system> to
     <tag>_<system>, as run_round4.slurm does (a missing manifest is skipped and reported; a row not named lab_* is an
     error). Each --manifest <file> (e.g. run_conversions.py rows such as mimi_copy_src / mimi_copy_P) keeps its
     system names. A (pair_id, system) may occur only once in the merged rows, and no input row may be an @src row.
  2. sets sim_src_wavs, the source speaker's E set, on every row. Dev and LTC-2k (every source speaker is a target):
     that speaker's target E set in --pairs, from the first pair row whose target_speaker it is, as crt_anchor_lab.py
     main builds spk_rows, so SIM_src and SIM_tgt use one centroid per speaker. VCTK-2k (no source is a target):
     --source_e, make_pairs.py's source-E JSON {source speaker: E paths}; a value {"paths": [...], ...} (an E entry as
     in the pair rows) is read as its paths. A source speaker without an E set (not a target, absent from --source_e,
     or an empty list) gets sim_src_wavs = null. Every row's pair_id must be a --pairs row with the same
     source_speaker, and a row that already has sim_src_wavs must have exactly these paths.
  3. writes the @src rows with the transformation of crt_anchor_lab.py main (manifest_src) for every row without
     error that has a source E set: system := <system>@src, sim_ref_wavs := sim_src_wavs, target_speaker :=
     source_speaker, true_target_speaker := target_speaker. metrics.py --metrics sim,secs on them gives SIM_src and
     SECS_src. The rows left out (no source E set; error rows) are counted in the printed summary; in --out they are
     the rows with sim_src_wavs null.
Outputs: --out (the merged rows with sim_src_wavs, metrics.py input) and --out_src (default: --out with _src before
the extension, e.g. manifest.jsonl -> manifest_src.jsonl); they must differ from each other and from every input.
"""
import argparse
import collections
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help='make_pairs.py JSONL of the rows (e.g. pairs/dev.jsonl)')
    p.add_argument('--source_e', default='', help='make_pairs.py source-E JSON (VCTK-2k); default: the source '
                   "speaker's target E set in --pairs (Dev, LTC-2k)")
    p.add_argument('--lab', nargs=2, action='append', default=[], metavar=('OUT_DIR', 'TAG'),
                   help='crt_decode_lab.py --out_dir and its tag: rows lab_<system> -> <tag>_<system> (repeatable)')
    p.add_argument('--manifest', action='append', default=[], help='another manifest JSONL, system names kept '
                   '(repeatable)')
    p.add_argument('--out', required=True, help='merged manifest JSONL')
    p.add_argument('--out_src', default='', help='@src manifest JSONL (default: --out with _src before the extension)')
    return p.parse_args()


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(path + '.tmp', path)


def describe(rows):
    n = collections.Counter(r['system'] for r in rows)
    uneven = f' (pairs per system {min(n.values())}-{max(n.values())})' if len(set(n.values())) > 1 else ''
    return f'{len(rows)} rows, {len(n)} systems, {len({r["pair_id"] for r in rows})} pairs{uneven}'


def merge(labs, manifests):
    """Rows of the decode-lab out_dirs [(out_dir, tag)], lab_<system> renamed <tag>_<system> (run_round4.slurm), then
    the rows of the other manifests unchanged. Returns (rows, one report line per input)."""
    rows, report = [], []
    for d, tag in labs:
        f = os.path.join(d, 'manifest.jsonl')
        if not os.path.exists(f):
            report.append(f'{tag}: {f} missing, skipped')
            continue
        rs = read_jsonl(f)
        bad = sorted({r['system'] for r in rs if not r['system'].startswith('lab_')})
        if bad:
            raise SystemExit(f'ERROR: {f} has rows not named lab_*: {bad[:3]}')
        for r in rs:
            r['system'] = r['system'].replace('lab_', f'{tag}_', 1)
        rows += rs
        report.append(f'{tag}: {describe(rs)} <- {f}')
    for f in manifests:
        rs = read_jsonl(f)
        rows += rs
        report.append(f'{describe(rs)} <- {f}')
    dup = [k for k, n in collections.Counter((r['pair_id'], r['system']) for r in rows).items() if n > 1]
    if dup:
        raise SystemExit(f'ERROR: {len(dup)} (pair_id, system) occur more than once in the inputs, e.g. {dup[0]}')
    src = sorted({r['system'] for r in rows if r['system'].endswith('@src')})
    if src:     # the transformation of an @src row would lose the target (true_target_speaker := source)
        raise SystemExit(f'ERROR: the inputs hold @src rows {src[:3]}: give the manifests they were made from')
    return rows, report


def source_e_sets(pair_rows, source_e=''):
    """{speaker: E paths}: from the --source_e file, else each target's E set from its first pair row (spk_rows of
    crt_anchor_lab.py main)."""
    if source_e:
        raw = json.load(open(source_e))
        if not isinstance(raw, dict):
            raise SystemExit(f'ERROR: {source_e} is not a JSON object {{source speaker: E paths}}')
        out = {}
        for s, e in raw.items():
            out[s] = e.get('paths') if isinstance(e, dict) else e
            if not isinstance(out[s], list) or not all(isinstance(x, str) for x in out[s]):
                raise SystemExit(f'ERROR: {source_e}: {s!r} is neither a list of E paths nor an E entry with "paths"')
        return out
    spk_rows = {}
    for r in pair_rows:
        spk_rows.setdefault(r['target_speaker'], r)
    return {s: r['E']['paths'] for s, r in spk_rows.items()}


def add_sim_src(rows, pair_rows, e_sets):
    """Sets sim_src_wavs on every row (None without a source E set). Returns the source speakers without one."""
    src_of = {r['pair_id']: r['source_speaker'] for r in pair_rows}
    for r in rows:
        if src_of.get(r['pair_id']) != r['source_speaker']:
            raise SystemExit(f'ERROR: {r["pair_id"]} {r["system"]}: not a --pairs row with source speaker '
                             f'{r["source_speaker"]}')
        e = e_sets.get(r['source_speaker']) or None
        if r.get('sim_src_wavs') is not None and r['sim_src_wavs'] != e:
            raise SystemExit(f'ERROR: {r["pair_id"]} {r["system"]}: sim_src_wavs differs from the source E set')
        r['sim_src_wavs'] = e
    return sorted({r['source_speaker'] for r in rows if r['sim_src_wavs'] is None})


def src_rows(rows):
    """The manifest_src rows of crt_anchor_lab.py main, for the rows without error that have a source E set."""
    return [dict(r, system=r['system'] + '@src', sim_ref_wavs=r['sim_src_wavs'], target_speaker=r['source_speaker'],
                 true_target_speaker=r['target_speaker'])
            for r in rows if 'error' not in r and r['sim_src_wavs'] is not None]


def main():
    a = parse_args()
    if not a.lab and not a.manifest:
        raise SystemExit('ERROR: give at least one --lab or --manifest')
    out_src = a.out_src or '{}_src{}'.format(*os.path.splitext(a.out))
    ins = [a.pairs, a.source_e, *(os.path.join(d, 'manifest.jsonl') for d, _ in a.lab), *a.manifest]
    outs = {os.path.realpath(a.out), os.path.realpath(out_src)}
    if len(outs) < 2 or outs & {os.path.realpath(f) for f in ins if f}:
        raise SystemExit(f'ERROR: --out {a.out} and --out_src {out_src} must differ from each other and from every '
                         f'input')
    rows, report = merge(a.lab, a.manifest)
    pair_rows = read_jsonl(a.pairs)
    no_e = add_sim_src(rows, pair_rows, source_e_sets(pair_rows, a.source_e))
    src = src_rows(rows)
    write_jsonl(a.out, rows)
    write_jsonl(out_src, src)
    left = [r for r in rows if 'error' not in r and r['sim_src_wavs'] is None]
    print('lab_rows: ' + '; '.join(report), flush=True)
    print(f'  {describe(rows)} -> {a.out}')
    print(f'  {len(src)} @src rows -> {out_src}; left out: {len(left)} rows ({len({r["pair_id"] for r in left})} '
          f'pairs) without a source E set (source speakers {no_e}), {sum("error" in r for r in rows)} error rows',
          flush=True)


if __name__ == '__main__':
    main()
