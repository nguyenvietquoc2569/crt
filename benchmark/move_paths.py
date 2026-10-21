#!/usr/bin/env python3
"""
Moves the absolute paths of the CRT tree after it was copied to another HPC account (19 Oct 2026: nvquoc -> ceai,
DEVIATIONS_v1.md item 4). The relay copies /datastore/nvquoc/crt byte for byte to /datastore/ceai/crt; this script
then rewrites the old prefix to the new one in every text file and symlink target under the new root, so that the
job scripts, pair lists, manifests, metrics rows and the environments point at the new folder. Standard library only.

  rewrite  --root R --old O --new N --log L [--keep DIR ...]
           A file is text when its first 8 KiB hold no NUL byte; files with a known binary suffix are not read, and
           job logs (*.log, *.out, *.err) and the --keep folders (relative to R) stay as they are, as records of
           what ran where (--keep also takes single files). The whole tree is scanned first: if any text file to change already holds N, nothing is
           changed (the reverse mapping N -> O would not be exact). Each file is written to a temporary file in its
           folder and renamed over the original (mode kept, mtime = now; a hard link elsewhere keeps the old text).
           L (must not exist) gets one line per change, written as it happens: path relative to R, file|link, the
           number of replacements, then sha256 before and after (a file) or the target before and after (a link).
  check    --root R --old O --new N --log L [--log L2 ...] [--freeze FREEZE_v1.sha256] [--keep DIR ...]
           1. every logged file has its logged sha256 now, and mapped back (N -> O) its sha256 from before; every
              logged link points to its logged target;
           2. every section-2 entry of the freeze list (J/... under R/mimi_vc/benchmark/journal, W/... under
              R/mimi_vc) has its frozen sha256, mapped back first if it was rewritten;
           3. no text file or symlink under R still holds O (job logs and --keep folders aside).
  compare  --new A.jsonl --ref B.jsonl [--tol 1e-4]
           metrics.py rows of the same (pair_id, system) scored on both accounts: the largest |difference| of every
           number in A that B also has (top level and inside dicts such as wer_large_v3), and how many Whisper
           hypotheses are identical. Whisper decodes in batches, so its hypotheses and counts are reported only.
Every command exits 1 when a check fails.
"""
import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile

# '.pth' is not in the list: Python's site-packages path files (e.g. an env's mimi_vc_base.pth) are text and must be
# rewritten; torch checkpoints named .pth hold NUL bytes in their first 8 KiB, so read_text still skips them.
BINARY_SUFFIXES = {'.wav', '.flac', '.mp3', '.ogg', '.npz', '.npy', '.pt', '.ckpt', '.safetensors', '.bin',
                   '.so', '.pyc', '.whl', '.gz', '.tgz', '.tar', '.zip', '.faiss', '.index', '.pkl', '.onnx', '.h5',
                   '.png', '.jpg', '.pdf', '.docx'}
LOG_SUFFIXES = {'.log', '.out', '.err'}
FREEZE_DIRS = {'J': 'mimi_vc/benchmark/journal', 'W': 'mimi_vc'}
REPORT_ONLY = ('wer_', 'cer_', 'time_s')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def walk(root, keep=()):
    """Yields (relative path, 'file'|'link') for every regular file and symlink under root, sorted, without following
    links; skips the keep folders and files and the job logs."""
    keep = {k.strip('/') for k in keep}
    for d, dirs, files in os.walk(root):
        rel_d = os.path.relpath(d, root)
        rel_d = '' if rel_d == '.' else rel_d
        dirs[:] = sorted(x for x in dirs if os.path.join(rel_d, x) not in keep)
        for name in sorted(files) + [x for x in dirs if os.path.islink(os.path.join(d, x))]:
            rel = os.path.join(rel_d, name)
            p = os.path.join(root, rel)
            if rel in keep:
                continue
            if os.path.islink(p):
                yield rel, 'link'
            elif os.path.splitext(name)[1].lower() not in LOG_SUFFIXES and stat.S_ISREG(os.lstat(p).st_mode):
                yield rel, 'file'


def read_text(path):
    """The file's bytes, or None when it is binary (suffix, or a NUL byte in its first 8 KiB)."""
    if os.path.splitext(path)[1].lower() in BINARY_SUFFIXES:
        return None
    with open(path, 'rb') as f:
        head = f.read(8192)
        if b'\0' in head:
            return None
        return head + f.read()


def replace_file(path, data):
    mode = stat.S_IMODE(os.stat(path).st_mode)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.move_paths.')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def replace_link(path, target):
    tmp = os.path.join(os.path.dirname(path), f'.move_paths.{os.getpid()}.{os.path.basename(path)}')
    os.symlink(target, tmp)
    os.replace(tmp, path)


def rewrite(root, old, new, log, keep=()):
    o, n = old.encode(), new.encode()
    if os.path.exists(log):
        sys.exit(f'{log} exists: give every run its own log')
    plan, refused = [], []
    for rel, kind in walk(root, keep):
        p = os.path.join(root, rel)
        if kind == 'link':
            if old in os.readlink(p):
                plan.append((rel, kind))
            continue
        data = read_text(p)
        if data is not None and o in data:
            (refused if n in data else plan).append((rel, kind))
    if refused:
        print(f'REFUSED, nothing changed: {len(refused)} text files already hold {new}, e.g. {refused[:5]}')
        return 1
    with open(log, 'w') as lf:
        for rel, kind in plan:
            p = os.path.join(root, rel)
            if kind == 'link':
                t = os.readlink(p)
                replace_link(p, t.replace(old, new))
                lf.write(f'{rel}\tlink\t{t.count(old)}\t{t}\t{t.replace(old, new)}\n')
            else:
                data = read_text(p)
                out = data.replace(o, n)
                replace_file(p, out)
                lf.write(f'{rel}\tfile\t{data.count(o)}\t{sha256(data)}\t{sha256(out)}\n')
            lf.flush()
    files = sum(k == 'file' for _, k in plan)
    print(f'rewrote {old} -> {new} under {root}: {files} files, {len(plan) - files} links -> {log}')
    return 0


def read_log(paths):
    rows = {}
    for path in paths:
        for line in open(path):
            rel, kind, count, before, after = line.rstrip('\n').split('\t')
            rows[rel] = (kind, int(count), before, after)
    return rows


def check(root, old, new, logs, freeze=None, keep=()):
    o, n = old.encode(), new.encode()
    bad = []
    done = read_log(logs)
    for rel, (kind, _, before, after) in sorted(done.items()):
        p = os.path.join(root, rel)
        if kind == 'link':
            if not os.path.islink(p) or os.readlink(p) != after:
                bad.append(f'link {rel} does not point to {after}')
            continue
        data = open(p, 'rb').read() if os.path.isfile(p) else None
        if data is None or sha256(data) != after:
            bad.append(f'{rel}: missing or changed since the rewrite')
        elif sha256(data.replace(n, o)) != before:
            bad.append(f'{rel}: mapped back, it does not have its sha256 from before the rewrite')
    n_freeze = 0
    if freeze:
        for line in open(freeze):
            parts = line.split()
            if len(parts) != 2 or line.startswith('#') or parts[1].split('/', 1)[0] not in FREEZE_DIRS:
                continue
            h, ref = parts
            top, rest = ref.split('/', 1)
            rel = os.path.join(FREEZE_DIRS[top], rest)
            p = os.path.join(root, rel)
            n_freeze += 1
            if not os.path.isfile(p):
                bad.append(f'frozen {ref}: missing ({p})')
                continue
            data = open(p, 'rb').read()
            if rel in done:
                data = data.replace(n, o)
            if sha256(data) != h:
                bad.append(f'frozen {ref}: sha256 {sha256(data)} is not the frozen {h}')
    left = 0
    for rel, kind in walk(root, keep):
        p = os.path.join(root, rel)
        if kind == 'link':
            hit = old in os.readlink(p)
        else:
            data = read_text(p)
            hit = data is not None and o in data
        if hit:
            left += 1
            if left <= 5:
                bad.append(f'{rel} still holds {old}')
    if left > 5:
        bad.append(f'... {left} files or links still hold {old} in all')
    print(f'check {root}: {len(done)} logged changes, {n_freeze} frozen files, {len(bad)} problems')
    for b in bad:
        print('  ' + b)
    return 1 if bad else 0


def numbers(row, prefix=''):
    """Flattens the numbers of a metrics row: {'mcd_db': 8.7, 'wer_large_v3.S': 13, ...} (bools left out)."""
    out = {}
    for k, v in row.items():
        key = prefix + k
        if isinstance(v, dict):
            out.update(numbers(v, key + '.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def compare(new_path, ref_path, tol=1e-4):
    ref = {}
    for line in open(ref_path):
        if line.strip():
            r = json.loads(line)
            ref[(r['pair_id'], r['system'])] = r
    worst, missing, errors, hyp_same, hyp_n, rows = {}, [], [], 0, 0, 0
    for line in open(new_path):
        if not line.strip():
            continue
        a = json.loads(line)
        key = (a['pair_id'], a['system'])
        if 'error' in a:
            errors.append(key)
            continue
        if key not in ref:
            missing.append(key)
            continue
        rows += 1
        b = ref[key]
        na, nb = numbers(a), numbers(b)
        for k in na.keys() & nb.keys():
            x, y = na[k], nb[k]
            d = 0.0 if math.isnan(x) and math.isnan(y) else math.inf if math.isnan(x) or math.isnan(y) else abs(x - y)
            worst[k] = max(worst.get(k, 0.0), d)
        for k, v in a.items():
            if k.startswith('wer_') and isinstance(v, dict) and isinstance(b.get(k), dict) and 'hyp' in v:
                hyp_n += 1
                hyp_same += v['hyp'] == b[k].get('hyp')
    over = {k: d for k, d in worst.items() if d > tol and not k.startswith(REPORT_ONLY)}
    print(f'compare {new_path} with {ref_path}: {rows} rows, {len(missing)} without a reference row, '
          f'{len(errors)} error rows; identical Whisper hypotheses {hyp_same}/{hyp_n}')
    for k in sorted(worst):
        flag = ' OVER' if k in over else (' (reported only)' if k.startswith(REPORT_ONLY) else '')
        print(f'  max |diff| {k}: {worst[k]:.3g}{flag}')
    return 1 if over or missing or errors or not rows else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('rewrite', 'check'):
        s = sub.add_parser(name)
        s.add_argument('--root', required=True)
        s.add_argument('--old', required=True)
        s.add_argument('--new', required=True)
        s.add_argument('--log', required=True, action='append' if name == 'check' else 'store')
        s.add_argument('--keep', action='append', default=[], help='folder or file under root left as it is (repeatable)')
        if name == 'check':
            s.add_argument('--freeze', help='FREEZE_v1.sha256: check its section-2 (J/..., W/...) entries')
    s = sub.add_parser('compare')
    s.add_argument('--new', required=True)
    s.add_argument('--ref', required=True)
    s.add_argument('--tol', type=float, default=1e-4)
    a = ap.parse_args(argv)
    if a.cmd in ('rewrite', 'check'):
        for x in (a.old, a.new):
            if not x.startswith('/') or x.endswith('/'):
                ap.error(f'{x}: --old and --new are absolute folder paths without a trailing /')
        if a.old in a.new or a.new in a.old:
            ap.error('--old and --new must not contain each other')
    if a.cmd == 'rewrite':
        return rewrite(a.root, a.old, a.new, a.log, a.keep)
    if a.cmd == 'check':
        return check(a.root, a.old, a.new, a.log, a.freeze, a.keep)
    return compare(a.new, a.ref, a.tol)


if __name__ == '__main__':
    sys.exit(main())
