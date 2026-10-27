#!/usr/bin/env python3
"""
Checks of the T1 jobs of mvc/ANALYSIS_PLAN_mvc_v1.md (§2, §4.8, §5.4, §8.2, §8.3), run by job_t1_conv.slurm,
job_t1_metrics.slurm and job_t1_ecapa.slurm through t1_lib.sh. Standard library only. Every subcommand prints counts,
paths and identities (sha256, commits), never a metric value of a T1 system, and exits 1 after its ERROR lines.

  selection   mvc_selection.env (select_mvc.py --stage record --final) against the files it names: the record (sha256 =
              RECORD_SHA256, status complete, and its snapshot, describe() commit and cfg lines), the three cfg JSONs
              (sha256; mvc_v1 mode convert, decode latent, run_dir/ckpt = SNAPSHOT; mvc_v1_q32 = mvc_v1 with decode codes;
              mvc_copy_latent mode copy_latent with the same keys), the snapshot (sha256) and the describe() commit of the
              code dir (vc_adapter.describe: sha256 over its *.py in sorted order, first 16 hex). --install DIR then
              copies the cfg JSONs into DIR; a copy already there must be identical.
  frozen      every absolute-path line (Section 2, ceai) of FREEZE_mvc_v1.sha256: the file exists with that sha256.
  pushed      DEVIATIONS_mvc_v1.md records the push of the selection record (tag mvc-selection-record-v1) and no longer
              says "(to be filled in)" there (§8.3 first bullet, §8.4 steps 6-7).
  conv        every T1 manifest (§8.2: mvc_v1 at 30, 3 and 180 s, mvc_v1_q32 and mvc_copy_latent at 30 s, both sets):
              one row per pair of the list; one conv_cfg whose describe() names DESCRIBE_COMMIT, the cfg JSON's settings
              and the snapshot with its sha256 (none for copy_latent); shortfall rows exactly on the list's pairs with a
              reference shortfall, which §1 puts at 101 pairs of LTC-2k at 180 s and nowhere else; other error rows are
              failed conversions (§5.4), counted, not refused; the WAV of every other row present.
  metrics     a part's rows/<g>.jsonl, rows/<g>_src.jsonl, metrics/<g>.jsonl and metrics/<g>_src.jsonl (g =
              <set>_<budget>): the part's systems on every pair, one metric row per row, error rows only where the
              conversion failed, shortfall rows as in conv, @src rows exactly lab_rows.py's (rows without error with a
              source E set), with no error.
  drift_m     §8.3: a metrics.py --max_rows re-score of job M's rows against job M's metrics: sim_wavlm and utmos equal
              on every row, large-v3 and base hypotheses identical on at least --min_same rows.
  ecapa_drift_rows  §8.3: the input rows (job M's rows files) of X2's first --n ECAPA rows without error.
  drift_ecapa §8.3: their re-score against X2's rows, |Δ sim_ecapa| <= --tol on every row.
  ecapa       the ECAPA rows of the T1 systems against the part's rows: target rows on every row, error rows only where
              the conversion failed, @src rows on every @src row, with no error, and no other system.
  guard_rows  the acoustic-guard input (§3.8): the rows of the named systems from each FILE=sys,sys source.
  guard       acoustic_guard.py's output: the named systems on every pair and nothing else; rows without a measurement
              are counted (the guards are descriptive); no *.summary.json (acoustic_guard.py's means, §8.3) in its folder.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ENV_KEYS = ('RECORD_SHA256', 'SNAPSHOT', 'SNAPSHOT_SHA256', 'CFG_MVC_V1', 'CFG_MVC_V1_SHA256', 'CFG_MVC_V1_Q32',
            'CFG_MVC_V1_Q32_SHA256', 'CFG_MVC_COPY_LATENT', 'CFG_MVC_COPY_LATENT_SHA256', 'DESCRIBE_COMMIT')
CFG_VARS = {'mvc_v1': 'CFG_MVC_V1', 'mvc_v1_q32': 'CFG_MVC_V1_Q32', 'mvc_copy_latent': 'CFG_MVC_COPY_LATENT'}
SETS = ('vctk2k', 'ltc2k')
COMBOS = (('mvc_v1', 30), ('mvc_v1_q32', 30), ('mvc_copy_latent', 30), ('mvc_v1', 3), ('mvc_v1', 180))  # §2, §8.2
SHORTFALL = {('ltc2k', 180): 101}   # §1: the LTC-2k pairs of targets 3729 and 6930 at 180 s; no other set or budget
HEX64, HEX16 = re.compile(r'[0-9a-f]{64}'), re.compile(r'[0-9a-f]{16}')


class Problems(list):
    def __call__(self, msg):
        self.append(msg)


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def describe_commit(code_dir):
    """vc_adapter.MimiVC.describe()['commit']: sha256 over the bytes of every *.py of the folder, in sorted order."""
    h = hashlib.sha256()
    for f in sorted(Path(code_dir).resolve().glob('*.py')):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def keyed(rows, label, P):
    """{(pair_id, system): row}; a key twice is a problem."""
    out = {}
    for r in rows:
        k = (r.get('pair_id'), r.get('system'))
        if k in out:
            P(f'{label}: {k} twice')
        out[k] = r
    return out


def pair_list(path):
    rows = read_jsonl(path)
    return rows, [r['pair_id'] for r in rows]


def shortfall_pairs(rows, set_, budget, P):
    """The list's pairs whose budget reference has a shortfall (make_pairs.py R[b]['shortfall']), against §1's count."""
    s = {r['pair_id'] for r in rows if ((r.get('R') or {}).get(str(budget)) or {}).get('shortfall')}
    want = SHORTFALL.get((set_, budget), 0)
    if len(s) != want:
        P(f'the {set_} list has {len(s)} pairs with a {budget} s reference shortfall; §1 says {want}')
    return s


# ── selection ────────────────────────────────────────────────

def read_env(path, P):
    env = {}
    if not os.path.isfile(path):
        P(f'{path} missing: write it with select_mvc.py --stage record --final and copy it with the record')
        return env
    for i, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r'([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./+-]+)', line)
        if not m:
            P(f'{path}:{i}: not a KEY=VALUE line without quotes or spaces')
            continue
        if m[1] in env:
            P(f'{path}: {m[1]} twice')
        env[m[1]] = m[2]
    miss, extra = [k for k in ENV_KEYS if k not in env], sorted(set(env) - set(ENV_KEYS))
    if miss:
        P(f'{path} lacks {miss}')
    if extra:
        P(f'{path} holds {extra}, which select_mvc.py does not write')
    for k, v in env.items():
        if k.endswith('SHA256') and not HEX64.fullmatch(v):
            P(f'{k}={v} is not a sha256')
    if 'DESCRIBE_COMMIT' in env and not HEX16.fullmatch(env['DESCRIBE_COMMIT']):
        P(f'DESCRIBE_COMMIT={env["DESCRIBE_COMMIT"]} is not 16 hex digits (vc_adapter.describe)')
    if 'SNAPSHOT' in env and not os.path.isabs(env['SNAPSHOT']):
        P(f'SNAPSHOT={env["SNAPSHOT"]} is not an absolute path')
    for name, var in CFG_VARS.items():
        if var in env and env[var] != f'{name}.json':
            P(f'{var}={env[var]}, not {name}.json (select_mvc.py writes <system>.json)')
    return env


def check_selection(env_path, record, cfg_dir, code_dir, install, P):
    env = read_env(env_path, P)
    if P:
        return env
    if not os.path.isfile(record):
        P(f'the record {record} missing')
    else:
        h = sha256_file(record)
        if h != env['RECORD_SHA256']:
            P(f'{record} has sha256 {h}, not RECORD_SHA256 {env["RECORD_SHA256"]}')
        text = open(record, encoding='utf-8').read()
        need = ['Status: **complete**.', f'- Snapshot: `{env["SNAPSHOT"]}`, sha256 {env["SNAPSHOT_SHA256"]}.',
                f'- describe() commit: {env["DESCRIBE_COMMIT"]} ']
        need += [f'| `{env[v]}` | {env[v + "_SHA256"]} |' for v in CFG_VARS.values()]
        miss = [x for x in need if x not in text]
        if miss:
            P(f'{record} lacks {miss}')
    cfgs = {}
    for name, var in CFG_VARS.items():
        p = os.path.join(cfg_dir, env[var])
        if not os.path.isfile(p):
            P(f'{p} missing')
            continue
        h = sha256_file(p)
        if h != env[var + '_SHA256']:
            P(f'{p} has sha256 {h}, not {var}_SHA256 {env[var + "_SHA256"]}')
            continue
        try:
            cfgs[name] = json.load(open(p))
        except ValueError as e:
            P(f'{p}: {e}')
    if len(cfgs) == len(CFG_VARS):
        v1, q, c = cfgs['mvc_v1'], cfgs['mvc_v1_q32'], cfgs['mvc_copy_latent']
        if not all(isinstance(x, dict) for x in (v1, q, c)):
            P('a cfg JSON is not an object')
        else:
            if v1.get('mode') != 'convert' or v1.get('decode') != 'latent':
                P('mvc_v1.json is not mode convert with decode latent (§2.1)')
            if os.path.abspath(os.path.join(str(v1.get('run_dir')), str(v1.get('ckpt')))) != env['SNAPSHOT']:
                P(f'mvc_v1.json names {v1.get("run_dir")}/{v1.get("ckpt")}, not SNAPSHOT {env["SNAPSHOT"]}')
            if q != dict(v1, decode='codes'):
                P('mvc_v1_q32.json is not mvc_v1.json with "decode": "codes" (§2.2)')
            if c.get('mode') != 'copy_latent' or set(c) != set(v1):
                P('mvc_copy_latent.json is not mode copy_latent with every key of mvc_v1.json (§2.2)')
    snap = env['SNAPSHOT']
    if not os.path.isfile(snap):
        P(f'the snapshot {snap} missing')
    else:
        h = sha256_file(snap)
        if h != env['SNAPSHOT_SHA256']:
            P(f'{snap} has sha256 {h}, not SNAPSHOT_SHA256 {env["SNAPSHOT_SHA256"]}')
    c = describe_commit(code_dir)
    if c != env['DESCRIBE_COMMIT']:
        P(f'the *.py of {code_dir} give the describe() commit {c}, not DESCRIBE_COMMIT {env["DESCRIBE_COMMIT"]}: '
          f'mvc/*.py differ from the code of the record (§2.1)')
    if install and not P:
        os.makedirs(install, exist_ok=True)
        for var in CFG_VARS.values():
            src, dst = os.path.join(cfg_dir, env[var]), os.path.join(install, env[var])
            if os.path.exists(dst):
                if sha256_file(dst) != env[var + '_SHA256']:
                    P(f'{dst} exists and differs from the record\'s {env[var]}')
                continue
            tmp = f'{dst}.tmp{os.getpid()}'
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
    if not P:
        print(f'selection {env_path}: record {record} (sha256 {env["RECORD_SHA256"]}), snapshot {snap} (sha256 '
              f'{env["SNAPSHOT_SHA256"]}), describe() commit {env["DESCRIBE_COMMIT"]}, cfg '
              f'{", ".join(env[v] for v in CFG_VARS.values())} in {cfg_dir}' + (f', installed in {install}' if install
                                                                               else ''), flush=True)
    return env


def check_frozen(path, P):
    n = bad = 0
    if not os.path.isfile(path):
        P(f'{path} missing')
        return
    for line in open(path):
        m = re.fullmatch(r'([0-9a-f]{64})  (/\S+)', line.rstrip('\n'))
        if not m:
            continue
        n += 1
        if not os.path.isfile(m[2]):
            P(f'{m[2]} missing')
        elif sha256_file(m[2]) != m[1]:
            P(f'{m[2]} differs from its frozen sha256 {m[1]}')
        else:
            continue
        bad += 1
    if not n:
        P(f'{path} holds no absolute-path (ceai) line')
    print(f'frozen: {n - bad} of the {n} ceai files of {path} as frozen', flush=True)


def check_pushed(path, P):
    """§8.3 first bullet, §8.4 steps 6-7: the push of the selection record is on record in DEVIATIONS_mvc_v1.md."""
    if not os.path.isfile(path):
        P(f'{path} missing: copy mvc/DEVIATIONS_mvc_v1.md, with the push of the selection record filled in, to ceai')
        return
    lines = [x.strip() for x in open(path, encoding='utf-8') if 'mvc-selection-record-v1' in x]
    if not lines:
        P(f'{path} has no line on the push of the selection record (tag mvc-selection-record-v1)')
    elif any('to be filled in' in x for x in lines):
        P(f'{path}: the push of the selection record is still "(to be filled in)": push the record (§8.4 step 6), '
          f'fill in the line and copy the file to ceai')
    else:
        print(f'pushed: {lines[0][:300]}', flush=True)


# ── conversions ──────────────────────────────────────────────

def check_conv(conv_dir, pairs, env_path, cfg_dir, P):
    env = read_env(env_path, P)
    if P:
        return
    cfgs = {n: json.load(open(os.path.join(cfg_dir, env[v]))) for n, v in CFG_VARS.items()}
    n_failed = 0
    for set_ in SETS:
        rows, ids = pair_list(pairs[set_])
        shorts = {b: shortfall_pairs(rows, set_, b, P) for b in sorted({b for _, b in COMBOS})}
        for name, b in COMBOS:
            tag = f'{set_} {b} s {name}'
            short = shorts[b]
            d = os.path.join(conv_dir, set_, str(b), name)
            man = os.path.join(d, 'manifest.jsonl')
            if not os.path.isfile(man):
                P(f'{tag}: {man} missing')
                continue
            other = sorted(x for x in os.listdir(d) if x.startswith('manifest') and x.endswith('.jsonl')
                           and x != 'manifest.jsonl')
            if other:
                P(f'{tag}: {d} also holds {other}')
            by = {}
            for r in read_jsonl(man):
                if (r.get('system'), r.get('set'), r.get('budget')) != (name, set_, b):
                    P(f'{tag}: a row of {r.get("system")}, {r.get("set")}, {r.get("budget")} s in {man}')
                elif r['pair_id'] in by:
                    P(f'{tag}: {r["pair_id"]} twice in {man}')
                by[r['pair_id']] = r
            if set(by) != set(ids):
                P(f'{tag}: rows on {len(set(by) & set(ids))} of the {len(ids)} pairs, {len(set(by) - set(ids))} outside '
                  f'the list')
            runs = os.path.join(d, 'runs.jsonl')
            starts = [e for e in read_jsonl(runs) if e.get('event') == 'start'] if os.path.isfile(runs) else []
            shas = {e.get('conv_cfg_sha') for e in starts}
            if len(shas) != 1:
                P(f'{tag}: {runs} holds {len(starts)} start events of {len(shas)} conv_cfg, not one conv_cfg')
            want_ck = {env['SNAPSHOT']: env['SNAPSHOT_SHA256']} if cfgs[name].get('mode') == 'convert' else {}
            for e in starts:
                dsc = (e.get('conv_cfg') or {}).get('describe') or {}
                if dsc.get('commit') != env['DESCRIBE_COMMIT']:
                    P(f'{tag}: converted with describe() commit {dsc.get("commit")}, not DESCRIBE_COMMIT '
                      f'{env["DESCRIBE_COMMIT"]}')
                s = {k: v for k, v in (dsc.get('settings') or {}).items() if k not in ('train_step', 'train_args')}
                if s != cfgs[name]:
                    diff = sorted(k for k in set(s) | set(cfgs[name]) if s.get(k, '?') != cfgs[name].get(k, '?'))
                    P(f'{tag}: converted with settings that differ from {env[CFG_VARS[name]]} in {diff}')
                if dsc.get('checkpoints') != want_ck:
                    P(f'{tag}: converted with the checkpoints {dsc.get("checkpoints")}, not {want_ck}')
            sha = next(iter(shas)) if len(shas) == 1 else None
            off = sum(r.get('conv_cfg_sha') != sha for r in by.values())
            if off:
                P(f'{tag}: {off} rows not of the conv_cfg {sha} of {runs}')
            sf = {k for k, r in by.items() if 'error' in r and r.get('shortfall') is True}
            failed = sorted(k for k, r in by.items() if 'error' in r and k not in sf)
            if sf != short:
                P(f'{tag}: shortfall rows on {len(sf)} pairs, {len(sf & short)} of them among the list\'s {len(short)} '
                  f'pairs with a shortfall')
            nowav = sum(not os.path.isfile(r.get('wav', '')) for r in by.values() if 'error' not in r)
            if nowav:
                P(f'{tag}: {nowav} rows without error whose WAV is missing')
            n_failed += len(failed)
            print(f'  conv {tag}: {len(by)} rows for {len(ids)} pairs, {len(sf)} shortfall, {len(failed)} failed, '
                  f'conv_cfg {sha}', flush=True)
    print(f'conv: {n_failed} failed conversions in all (error rows other than shortfall rows; once every pair has a '
          f'row they are failed conversions, §5.4)', flush=True)


# ── metrics ──────────────────────────────────────────────────

def check_metrics(d, g, pairs, systems, P):
    set_, b = g.rsplit('_', 1)
    b = int(b)
    rows, ids = pair_list(pairs)
    short = shortfall_pairs(rows, set_, b, P)
    f = {'rows': f'{d}/rows/{g}.jsonl', 'src_rows': f'{d}/rows/{g}_src.jsonl', 'met': f'{d}/metrics/{g}.jsonl',
         'src_met': f'{d}/metrics/{g}_src.jsonl'}
    miss = [p for p in f.values() if not os.path.isfile(p)]
    if miss:
        P(f'{g}: missing {miss}')
        return
    R, SR = keyed(read_jsonl(f['rows']), f['rows'], P), keyed(read_jsonl(f['src_rows']), f['src_rows'], P)
    M, SM = keyed(read_jsonl(f['met']), f['met'], P), keyed(read_jsonl(f['src_met']), f['src_met'], P)
    got = sorted({s for _, s in R})
    if got != sorted(systems):
        P(f'{g}: rows of {got}, not of {sorted(systems)}')
    for s in systems:
        have = {p for p, x in R if x == s}
        if have != set(ids):
            P(f'{g} {s}: rows on {len(have & set(ids))} of the {len(ids)} pairs, {len(have - set(ids))} outside the list')
    if set(M) != set(R):
        P(f'{g}: {len(set(R) - set(M))} rows without a metric row, {len(set(M) - set(R))} metric rows of no row')
    off = sum((r.get('set'), r.get('budget')) != (set_, b) for r in M.values())
    if off:
        P(f'{g}: {off} metric rows of another set or budget')
    conv_err = {k for k, r in R.items() if 'error' in r}
    score_err = sorted(k for k, r in M.items() if 'error' in r and k not in conv_err)
    if score_err:
        P(f'{g}: {len(score_err)} scoring errors (e.g. {score_err[0]}: {str(M[score_err[0]]["error"])[:200]}); '
          f'resubmit the part (metrics.py retries error rows)')
    for s in systems:
        flagged = {p for (p, x), r in M.items() if x == s and r.get('shortfall')}
        if flagged != short:
            P(f'{g} {s}: shortfall rows on {len(flagged)} pairs; the list has {len(short)} pairs with a shortfall')
    n_short = sum(bool(r.get('shortfall')) for r in R.values() if 'error' in r)
    want_src = {(p, s + '@src') for (p, s), r in R.items() if 'error' not in r and r.get('sim_src_wavs')}
    if set(SR) != want_src:
        P(f'{g}: {len(SR)} @src rows, not the {len(want_src)} of the rows without error with a source E set')
    if set(SM) != set(SR):
        P(f'{g}: {len(set(SR) - set(SM))} @src rows without a metric row, {len(set(SM) - set(SR))} @src metric rows '
          f'of no row')
    src_err = sorted(k for k, r in SM.items() if 'error' in r)
    if src_err:
        P(f'{g}: {len(src_err)} @src scoring errors (e.g. {src_err[0]}: {str(SM[src_err[0]]["error"])[:200]})')
    print(f'  {g}: {len(M)} metric rows ({len(systems)} systems x {len(ids)} pairs), {n_short} shortfall rows, '
          f'{len(conv_err) - n_short} failed conversions, {len(score_err)} scoring errors; @src {len(SM)} metric rows, '
          f'{len(src_err)} errors', flush=True)


# ── drift checks (§8.3) ──────────────────────────────────────

def drift_m(new, ref, n, min_same, P):
    N = read_jsonl(new) if os.path.isfile(new) else []
    Rf = {(r['pair_id'], r['system']): r for r in read_jsonl(ref)}
    if len(N) != n:
        P(f'{new}: {len(N)} rows, not {n}')
    bad = d_sim = d_ut = same = same_l = same_b = 0
    for r in N:
        q = Rf.get((r.get('pair_id'), r.get('system')))
        try:
            if 'error' in r or q is None or 'error' in q:
                raise KeyError
            ds, du = r['sim_wavlm'] != q['sim_wavlm'], r['utmos'] != q['utmos']
            hl = r['wer_large_v3']['hyp'] == q['wer_large_v3']['hyp']
            hb = r['wer_base']['hyp'] == q['wer_base']['hyp']
        except (KeyError, TypeError):
            bad += 1
            continue
        d_sim, d_ut = d_sim + ds, d_ut + du
        same_l, same_b, same = same_l + hl, same_b + hb, same + (hl and hb)
    if bad:
        P(f'{bad} re-scored rows have an error, lack a value or have no row without error in {ref}')
    if d_sim or d_ut:
        P(f'sim_wavlm differs on {d_sim} rows and utmos on {d_ut}; §8.3 requires max |Δ| = 0')
    if same < min_same:
        P(f'identical large-v3 and base hypotheses on {same} rows; §8.3 requires at least {min_same} of {n}')
    print(f'drift (WavLM-SV, UTMOS, Whisper): {len(N)} rows of {ref} re-scored: sim_wavlm differs on {d_sim}, utmos on '
          f'{d_ut}; hypotheses identical on {same} (large-v3 {same_l}, base {same_b}) -> {"FAIL" if P else "pass"}',
          flush=True)


def ecapa_drift_rows(x2, rows_files, n, out, P):
    X = [r for r in read_jsonl(x2) if 'error' not in r][:n]
    keys = [(r['pair_id'], r['system']) for r in X]
    idx = {}
    for f in rows_files:
        for r in read_jsonl(f):
            idx.setdefault((r['pair_id'], r['system']), r)
    miss = [k for k in keys if k not in idx]
    if len(keys) < n:
        P(f'{x2}: {len(keys)} rows without error, not {n}')
    if miss:
        P(f'{len(miss)} of X2\'s first rows have no input row in {rows_files} (e.g. {miss[0]})')
    if not P:
        write_jsonl(out, [idx[k] for k in keys])
    print(f'ECAPA drift input: {len(keys)} rows of {x2} (systems {sorted({s for _, s in keys})}) -> {out}', flush=True)


def drift_ecapa(new, ref, n, tol, P):
    N = read_jsonl(new) if os.path.isfile(new) else []
    Rf = {(r['pair_id'], r['system']): r for r in read_jsonl(ref) if 'error' not in r}
    if len(N) != n:
        P(f'{new}: {len(N)} rows, not {n}')
    bad = over = 0
    for r in N:
        q = Rf.get((r.get('pair_id'), r.get('system')))
        if 'error' in r or q is None or not isinstance(r.get('sim_ecapa'), float) or \
                not isinstance(q.get('sim_ecapa'), float):
            bad += 1
        elif not abs(r['sim_ecapa'] - q['sim_ecapa']) <= tol:
            over += 1
    if bad:
        P(f'{bad} re-scored rows have an error, lack sim_ecapa or have no row in {ref}')
    if over:
        P(f'|Δ sim_ecapa| > {tol} on {over} rows (§8.3)')
    print(f'drift (ECAPA): {len(N)} rows of {ref} re-scored: |Δ sim_ecapa| > {tol} on {over}, unusable {bad} -> '
          f'{"FAIL" if P else "pass"}', flush=True)


# ── ECAPA and guards ─────────────────────────────────────────

def check_ecapa(rows_dir, ecapa_dir, g, pairs, systems, P):
    _, ids = pair_list(pairs)
    f = {'rows': f'{rows_dir}/{g}.jsonl', 'src_rows': f'{rows_dir}/{g}_src.jsonl', 'e': f'{ecapa_dir}/{g}.jsonl',
         'es': f'{ecapa_dir}/{g}_src.jsonl'}
    miss = [p for p in f.values() if not os.path.isfile(p)]
    if miss:
        P(f'ECAPA {g}: missing {miss}')
        return
    R, SR = keyed(read_jsonl(f['rows']), f['rows'], P), keyed(read_jsonl(f['src_rows']), f['src_rows'], P)
    E, ES = keyed(read_jsonl(f['e']), f['e'], P), keyed(read_jsonl(f['es']), f['es'], P)
    want = {k for k in R if k[1] in systems}
    for s in systems:
        if {p for p, x in want if x == s} != set(ids):
            P(f'ECAPA {g} {s}: the input rows do not cover the {len(ids)} pairs')
    if set(E) != want:
        P(f'ECAPA {g}: {len(want - set(E))} rows not scored, {len(set(E) - want)} rows of another system or pair')
    conv_err = {k for k in want if 'error' in R[k]}
    err = {k for k, r in E.items() if 'error' in r}
    if err - conv_err:
        P(f'ECAPA {g}: {len(err - conv_err)} error rows on converted rows (score_ecapa.py does not retry them): '
          f'resubmit with ECAPA_RESCORE=1, which deletes the T1 ECAPA files unread, and log it in DEVIATIONS_mvc_v1.md '
          f'as an operational change')
    want_s = {k for k in SR if k[1].split('@')[0] in systems}
    if set(ES) != want_s:
        P(f'ECAPA {g}_src: {len(want_s - set(ES))} @src rows not scored, {len(set(ES) - want_s)} rows of another '
          f'system or pair')
    s_err = sum('error' in r for r in ES.values())
    if s_err:
        P(f'ECAPA {g}_src: {s_err} error rows: resubmit with ECAPA_RESCORE=1 (as above)')
    print(f'  ECAPA {g}: {len(E)} rows ({len(systems)} systems x {len(ids)} pairs), {len(err)} errors ({len(conv_err)} '
          f'failed conversions); @src {len(ES)} rows, {s_err} errors', flush=True)


def guard_rows(out, sources, P):
    rows, seen, counts = [], set(), {}
    for spec in sources:
        path, _, sy = spec.rpartition('=')
        want = sy.split(',')
        if not os.path.isfile(path):
            P(f'{path} missing')
            continue
        here = set()
        for r in read_jsonl(path):
            if r.get('system') not in want:
                continue
            k = (r['pair_id'], r['system'])
            if k in seen:
                P(f'{k} twice in the guard sources')
            seen.add(k)
            here.add(r['system'])
            rows.append(r)
            counts[r['system']] = counts.get(r['system'], 0) + 1
        absent = [s for s in want if s not in here]
        if absent:
            P(f'{path} holds no row of {absent}')
    if not P:
        write_jsonl(out, rows)
    print(f'guard input {out}: {len(rows)} rows ({counts})', flush=True)


def check_guard(guard, pairs, systems, P):
    _, ids = pair_list(pairs)
    if not os.path.isfile(guard):
        P(f'{guard} missing')
        return
    G = keyed(read_jsonl(guard), guard, P)
    for s in systems:
        have = {p for p, x in G if x == s}
        if have != set(ids):
            P(f'{guard} {s}: rows on {len(have & set(ids))} of the {len(ids)} pairs, {len(have - set(ids))} outside')
    other = sorted({x for _, x in G} - set(systems))
    if other:
        P(f'{guard}: rows of {other}')
    summ = sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.dirname(os.path.abspath(guard))) for f in fs
                  if f.endswith('.summary.json'))
    if summ:
        P(f'{summ}: acoustic_guard.py\'s summaries of means, which no T1 file may hold (§8.3); delete them unread')
    err = sum('error' in r for r in G.values())
    print(f'  guard {guard}: {len(G)} rows ({len(systems)} systems x {len(ids)} pairs), {err} rows without a measurement '
          f'(counted; the guards are descriptive)', flush=True)


# ── main ─────────────────────────────────────────────────────

def pairs_arg(items):
    out = dict(x.split('=', 1) for x in items)
    if sorted(out) != sorted(SETS):
        raise SystemExit(f'ERROR: --pairs needs {SETS} (SET=PATH), got {sorted(out)}')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('selection')
    s.add_argument('--env', required=True)
    s.add_argument('--record', required=True)
    s.add_argument('--cfg_dir', required=True)
    s.add_argument('--code_dir', required=True)
    s.add_argument('--install', default='')
    s = sub.add_parser('frozen')
    s.add_argument('--freeze', required=True)
    s = sub.add_parser('pushed')
    s.add_argument('--deviations', required=True)
    s = sub.add_parser('conv')
    s.add_argument('--conv_dir', required=True)
    s.add_argument('--cfg_dir', required=True)
    s.add_argument('--env', required=True)
    s.add_argument('--pairs', action='append', default=[], help='SET=PATH, for vctk2k and ltc2k')
    s = sub.add_parser('metrics')
    s.add_argument('--dir', required=True)
    s.add_argument('--g', required=True)
    s.add_argument('--pairs', required=True)
    s.add_argument('--systems', required=True)
    s = sub.add_parser('drift_m')
    s.add_argument('--new', required=True)
    s.add_argument('--ref', required=True)
    s.add_argument('--n', type=int, required=True)
    s.add_argument('--min_same', type=int, required=True)
    s = sub.add_parser('ecapa_drift_rows')
    s.add_argument('--x2', required=True)
    s.add_argument('--rows', action='append', required=True)
    s.add_argument('--n', type=int, required=True)
    s.add_argument('--out', required=True)
    s = sub.add_parser('drift_ecapa')
    s.add_argument('--new', required=True)
    s.add_argument('--ref', required=True)
    s.add_argument('--n', type=int, required=True)
    s.add_argument('--tol', type=float, required=True)
    s = sub.add_parser('ecapa')
    s.add_argument('--rows_dir', required=True)
    s.add_argument('--ecapa_dir', required=True)
    s.add_argument('--g', required=True)
    s.add_argument('--pairs', required=True)
    s.add_argument('--systems', required=True)
    s = sub.add_parser('guard_rows')
    s.add_argument('--out', required=True)
    s.add_argument('--src', action='append', required=True, help='FILE=system,system')
    s = sub.add_parser('guard')
    s.add_argument('--guard', required=True)
    s.add_argument('--pairs', required=True)
    s.add_argument('--systems', required=True)
    a = ap.parse_args(argv)
    P = Problems()
    if a.cmd == 'selection':
        check_selection(a.env, a.record, a.cfg_dir, a.code_dir, a.install, P)
    elif a.cmd == 'frozen':
        check_frozen(a.freeze, P)
    elif a.cmd == 'pushed':
        check_pushed(a.deviations, P)
    elif a.cmd == 'conv':
        check_conv(a.conv_dir, pairs_arg(a.pairs), a.env, a.cfg_dir, P)
    elif a.cmd == 'metrics':
        check_metrics(a.dir, a.g, a.pairs, a.systems.split(','), P)
    elif a.cmd == 'drift_m':
        drift_m(a.new, a.ref, a.n, a.min_same, P)
    elif a.cmd == 'ecapa_drift_rows':
        ecapa_drift_rows(a.x2, a.rows, a.n, a.out, P)
    elif a.cmd == 'drift_ecapa':
        drift_ecapa(a.new, a.ref, a.n, a.tol, P)
    elif a.cmd == 'ecapa':
        check_ecapa(a.rows_dir, a.ecapa_dir, a.g, a.pairs, a.systems.split(','), P)
    elif a.cmd == 'guard_rows':
        guard_rows(a.out, a.src, P)
    elif a.cmd == 'guard':
        check_guard(a.guard, a.pairs, a.systems.split(','), P)
    for p in P:
        print(f'ERROR: {p}', flush=True)
    return 1 if P else 0


if __name__ == '__main__':
    sys.exit(main())
