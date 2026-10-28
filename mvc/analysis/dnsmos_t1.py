#!/usr/bin/env python3
"""DNSMOS of the T1 systems and their comparators at 30 s on the Mac (ANALYSIS_PLAN_mvc_v1.md §3.7, §6.2, §8.1), after
the T1 metric rows are copied to MR with sha256 and before analyze_mvc.py runs.

For each set and each system of analyze_mvc.DNSMOS_SYSTEMS it takes the system's 30 s target rows from the metric rows
(MimiFlow-VC: MR/m/t1|t2/metrics/<set>_30.jsonl; journal job M: V1/m/t1|t2 for cand_pm18_rq, mimi_copy_src and
source_copy, V1/m/t4 for the baselines) and keeps only pair_id, system and wav: no metric value is read or printed.
The WAVs are copied read-only from ceai by rsync into MR/wav/ with their absolute ceai path kept below that prefix. The
journal rows on the Mac still name /datastore/nvquoc/crt, which moved to /datastore/ceai/crt on 21 Oct 2026
(benchmark/DEVIATIONS_v1.md item 4), so that prefix is mapped. The rows are rewritten to point to the copies and
mvc/dnsmos.py (sha256 checked) scores them with --norm_lufs -26 into MR/dnsmos/<set>_30_<system>.jsonl. Its printed
means go to /dev/null and the summary file of means it writes beside its output is deleted unread (§8.3). The batch's
WAVs are then deleted, because the Mac has no room for all of them (ceai keeps the originals), unless --keep_wav.
A batch whose output already has one row per input row is skipped, so a stopped run is resumed by running it again.

    .venv-stats/bin/python mvc/analysis/dnsmos_t1.py --mr paper/mimi_vc/results \
        --v1 paper/journal_extension/results/v1 --models paper/mimi_vc/models/dnsmos
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DNSMOS_PY = os.path.join(ROOT, 'mvc', 'dnsmos.py')
DNSMOS_SHA256 = '330b1d35ba6441f3aa3369d8dacd7c5e99eef9852134c37e10e5244a35b66a16'   # §3.7
MODEL_SHA256 = {'sig_bak_ovr.onnx': '269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd',
                'model_v8.onnx': '9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91'}
NORM_LUFS = '-26'
MOVED = ('/datastore/nvquoc/crt/', '/datastore/ceai/crt/')
REMOTE = 'ceai@hpc.uit.edu.vn'
PYTHON = 'uvx --with numpy --with soundfile --with soxr --with librosa --with onnxruntime --with pyloudnorm python'
SETS = ('vctk2k', 'ltc2k')
PART = {'vctk2k': 't1', 'ltc2k': 't2'}
MVC_SYSTEMS = ('mvc_v1', 'mvc_v1_q32', 'mvc_copy_latent')
JOURNAL_SYSTEMS = ('cand_pm18_rq', 'mimi_copy_src', 'source_copy')
BASELINES = ('knnvc', 'freevc', 'seedvc', 'facodec')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sources(mr, v1, s):
    """(metric rows file, systems taken from it) of set s."""
    return [(os.path.join(mr, 'm', PART[s], 'metrics', f'{s}_30.jsonl'), MVC_SYSTEMS),
            (os.path.join(v1, 'm', PART[s], 'metrics', f'{s}_30.jsonl'), JOURNAL_SYSTEMS),
            (os.path.join(v1, 'm', 't4', 'metrics', f'{s}_30.jsonl'), BASELINES)]


def remote_path(p):
    return MOVED[1] + p[len(MOVED[0]):] if p.startswith(MOVED[0]) else p


def batches(mr, v1):
    """[(set, system, rows)] with rows [{pair_id, system[, wav]}] holding the ceai path; one row per pair and system."""
    out = []
    for s in SETS:
        for path, systems in sources(mr, v1, s):
            if not os.path.exists(path):
                raise SystemExit(f'ERROR: {path} missing')
            got = {n: [] for n in systems}
            for line in open(path):
                r = json.loads(line)
                if r.get('system') in got:
                    row = {'pair_id': r['pair_id'], 'system': r['system']}
                    if r.get('wav'):
                        row['wav'] = remote_path(r['wav'])
                    got[r['system']].append(row)
            for n in systems:
                ids = [r['pair_id'] for r in got[n]]
                if not ids or len(ids) != len(set(ids)):
                    raise SystemExit(f'ERROR: {path}: {len(ids)} rows of {n}, {len(set(ids))} pairs')
                out.append((s, n, got[n]))
    return out


def n_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def fetch(paths, wav_root, remote):
    """rsync the absolute ceai paths into wav_root, keeping each path below it."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(''.join(p.lstrip('/') + '\n' for p in paths))
        lst = f.name
    try:
        r = subprocess.run(['rsync', '-a', f'--files-from={lst}', f'{remote}:/', wav_root + '/'],
                           stdout=subprocess.DEVNULL)
    finally:
        os.unlink(lst)
    return r.returncode


def run_batch(s, n, rows, a):
    out_dir = os.path.join(a.mr, 'dnsmos')
    out = os.path.join(out_dir, f'{s}_30_{n}.jsonl')
    if os.path.exists(out) and n_lines(out) == len(rows):
        return f'{s} {n}: {len(rows)} rows already scored, skipped'
    wav_root = os.path.join(a.mr, 'wav')
    local = [dict(r, wav=wav_root + r['wav']) if 'wav' in r else dict(r) for r in rows]
    paths = sorted({r['wav'] for r in rows if 'wav' in r})
    code = fetch(paths, wav_root, a.remote)
    if code != 0:
        raise SystemExit(f'ERROR: {s} {n}: rsync exit {code}: nothing scored')
    missing = [p for p in paths if not os.path.exists(wav_root + p)]
    if missing:
        raise SystemExit(f'ERROR: {s} {n}: {len(missing)} WAVs not copied: nothing scored')
    inp = os.path.join(out_dir, 'input', f'{s}_30_{n}.jsonl')
    with open(inp, 'w') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in local))
    cmd = shlex.split(a.python) + [DNSMOS_PY, '--models', a.models, '--rows', inp, '--out', out,
                                   '--norm_lufs', NORM_LUFS, '--threads', str(a.threads)]
    code = subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode
    summary = out[:-len('.jsonl')] + '.summary.json'
    if os.path.exists(summary):
        os.unlink(summary)                                    # dnsmos.py's means: deleted unread (§8.3)
    if not a.keep_wav:
        for p in paths:
            os.unlink(wav_root + p)
    if code != 0 or not os.path.exists(out) or n_lines(out) != len(rows):
        raise SystemExit(f'ERROR: {s} {n}: dnsmos.py exit {code}; the output lacks rows')
    n_err = sum('error' in json.loads(line) for line in open(out) if line.strip())
    return f'{s} {n}: {len(rows)} rows, {len(paths)} WAVs, {n_err} error rows'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mr', required=True, help='the MimiFlow-VC results on the Mac (paper/mimi_vc/results)')
    ap.add_argument('--v1', required=True, help='the journal results on the Mac (paper/journal_extension/results/v1)')
    ap.add_argument('--models', required=True, help='the durable DNSMOS folder (paper/mimi_vc/models/dnsmos)')
    ap.add_argument('--remote', default=REMOTE)
    ap.add_argument('--python', default=PYTHON, help='command that runs mvc/dnsmos.py with its dependencies')
    ap.add_argument('--workers', type=int, default=3, help='batches at once (each holds its WAVs on disk)')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--keep_wav', action='store_true')
    a = ap.parse_args(argv)
    if sha256(DNSMOS_PY) != DNSMOS_SHA256:
        raise SystemExit(f'ERROR: {DNSMOS_PY} differs from the frozen sha256 {DNSMOS_SHA256} (§3.7)')
    for k, v in MODEL_SHA256.items():
        if sha256(os.path.join(a.models, k)) != v:
            raise SystemExit(f'ERROR: {a.models}/{k} differs from the frozen sha256 {v} (§3.7)')
    work = batches(a.mr, a.v1)
    os.makedirs(os.path.join(a.mr, 'dnsmos', 'input'), exist_ok=True)
    os.makedirs(os.path.join(a.mr, 'wav'), exist_ok=True)
    print(f'{len(work)} batches, {sum(len(r) for _, _, r in work)} rows', flush=True)
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda b: run_batch(*b, a), work):
            print(msg, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
