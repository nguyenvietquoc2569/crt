#!/usr/bin/env python3
"""DNSMOS of mvc_r1x300, the X-mvc-1 system (ANALYSIS_PLAN_mvc_v1.md §4.10, §3.7; DEVIATIONS_mvc_v1.md operational
note 4), at 30 s on both sets on the Mac, after the X-mvc-1 tree ($MVC/xmvc1 on ceai) is copied to --xr with sha256 and
before analyze_xmvc1.py runs. mvc_v1's DNSMOS rows are T1's (dnsmos_t1.py) and are not scored again.

It is dnsmos_t1.py's machinery, imported unchanged (its sha256 is checked): the frozen dnsmos.py and the DNSMOS models
are checked by sha256; each set needs <xr>/<set>.done under the MimiFlow-VC selection record and its 30 s metric rows
<xr>/m/<t1|t2>/metrics/<set>_30.jsonl with the sha256 that <xr>/<set>.sha256 (written by the job on ceai) lists for
them. The rows give pair_id, system and wav (no metric value is read or printed; any system other than mvc_r1x300 stops
the run; a row with an error, a failed conversion whose WAV does not exist, goes without its wav, so dnsmos.py writes
an error row for it). dnsmos_t1.run_batch copies the WAVs from ceai read-only into <xr>/wav/ below their absolute path,
runs dnsmos.py --norm_lufs -26 into <xr>/dnsmos/<set>_30_mvc_r1x300.jsonl, deletes the summary file of means unread
(§8.3) and then the WAVs (unless --keep_wav). A batch whose output already has one row per input row is skipped, so a
stopped run is resumed by running it again.

The copy from ceai, from the repository root, once $X/vctk2k.done and $X/ltc2k.done exist (no WAV is copied here):

    D=/datastore/ceai/crt/mimi_vc/data_storage/mvc/xmvc1
    ssh ceai@hpc.uit.edu.vn "cd $D && cut -c67- vctk2k.sha256 ltc2k.sha256 | sort -u && ls vctk2k.sha256 ltc2k.sha256 \
        vctk2k.done ltc2k.done" > xmvc1_files.txt
    rsync -a --files-from=xmvc1_files.txt ceai@hpc.uit.edu.vn:$D/ paper/mimi_vc/results/xmvc1/
    .venv-stats/bin/python mvc/analysis/dnsmos_xmvc1.py --xr paper/mimi_vc/results/xmvc1 \
        --models paper/mimi_vc/models/dnsmos

Then copy <xr>/dnsmos/ back to $D/dnsmos/ with its sha256, as T1 did (DEVIATIONS_mvc_v1.md, after the T1 analysis,
note 1).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dnsmos_t1 as dt  # noqa: E402

SYSTEM = 'mvc_r1x300'
RECORD_SHA256 = '9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42'        # MVC_SELECTION_RECORD_v1.md
DNSMOS_T1_SHA256 = 'bfd93a3d1c6c823c2a4afa534e4f28df8cb398cbd98ed9efc7ecb6b5069d746d'     # T1's driver, run unchanged


def listed(xr, s):
    """{path relative to xr: sha256} of <xr>/<set>.sha256."""
    lst = os.path.join(xr, f'{s}.sha256')
    if not os.path.exists(lst):
        raise SystemExit(f'ERROR: {lst} missing: copy it from ceai with the files it lists')
    return {p: h for h, p in (line.rstrip('\n').split('  ', 1) for line in open(lst) if line.strip())}


def batches(xr):
    """[(set, SYSTEM, rows)] with rows [{pair_id, system[, wav]}] holding the ceai path; one row per pair."""
    out = []
    for s in dt.SETS:
        done = os.path.join(xr, f'{s}.done')
        if not os.path.exists(done) or not open(done).read().strip().endswith(f'selection sha256 {RECORD_SHA256}'):
            raise SystemExit(f'ERROR: {done} missing or not written under the selection record {RECORD_SHA256}: '
                             f'X-mvc-1 on {s} is not done')
        rel = f'm/{dt.PART[s]}/metrics/{s}_30.jsonl'
        path = os.path.join(xr, rel)
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {path} missing')
        if listed(xr, s).get(rel) != dt.sha256(path):
            raise SystemExit(f'ERROR: {path} is not the file {xr}/{s}.sha256 lists: copy it again from ceai')
        rows = []
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('system') != SYSTEM:
                raise SystemExit(f'ERROR: {path} holds a row of {r.get("system")}; the X-mvc-1 rows are {SYSTEM}\'s only')
            row = {'pair_id': r['pair_id'], 'system': SYSTEM}
            if r.get('wav') and 'error' not in r:     # a failed conversion's WAV does not exist: dnsmos.py's error row
                row['wav'] = dt.remote_path(r['wav'])
            rows.append(row)
        ids = [r['pair_id'] for r in rows]
        if not ids or len(ids) != len(set(ids)):
            raise SystemExit(f'ERROR: {path}: {len(ids)} rows of {SYSTEM}, {len(set(ids))} pairs')
        out.append((s, SYSTEM, rows))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xr', required=True, help='the X-mvc-1 results on the Mac (layout as $MVC/xmvc1 on ceai)')
    ap.add_argument('--models', required=True, help='the durable DNSMOS folder (paper/mimi_vc/models/dnsmos)')
    ap.add_argument('--remote', default=dt.REMOTE)
    ap.add_argument('--python', default=dt.PYTHON, help='command that runs mvc/dnsmos.py with its dependencies')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--keep_wav', action='store_true')
    a = ap.parse_args(argv)
    if dt.sha256(dt.__file__) != DNSMOS_T1_SHA256:
        raise SystemExit(f'ERROR: {dt.__file__} differs from the sha256 {DNSMOS_T1_SHA256} T1 ran with')
    if dt.sha256(dt.DNSMOS_PY) != dt.DNSMOS_SHA256:
        raise SystemExit(f'ERROR: {dt.DNSMOS_PY} differs from the frozen sha256 {dt.DNSMOS_SHA256} (§3.7)')
    for k, v in dt.MODEL_SHA256.items():
        if dt.sha256(os.path.join(a.models, k)) != v:
            raise SystemExit(f'ERROR: {a.models}/{k} differs from the frozen sha256 {v} (§3.7)')
    work = batches(a.xr)
    a.mr = a.xr                                   # run_batch writes <mr>/dnsmos/<set>_30_<system>.jsonl, <mr>/wav/
    os.makedirs(os.path.join(a.xr, 'dnsmos', 'input'), exist_ok=True)
    os.makedirs(os.path.join(a.xr, 'wav'), exist_ok=True)
    print(f'{len(work)} batches, {sum(len(r) for _, _, r in work)} rows', flush=True)
    for b in work:
        print(dt.run_batch(*b, a), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
