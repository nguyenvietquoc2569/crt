#!/usr/bin/env python3
"""
The D0 checks of ANALYSIS_PLAN_v1.md §8.1 item 9, and the reproduction check of §4.4 (D1 and D2 must reproduce D0 on
the 50 screen pairs, with the same comparisons and tolerances). Pure numpy / json / soundfile, no model. Run at the
end of job_d0.slurm (D0 vs the screen), job_d1_conv.slurm (D1 vs D0) and job_d2.slurm (D2 vs D0).

Screen pairs: the rows k of --pairs (pairs/dev.jsonl) with k % 20 == 0, the 50 pairs of --shard 0/20.
Reference, --ref_screen J (journal/ on the HPC) or --ref_d0 DIR (job D0's output root, layout of job_d0.slurm):
  run_conversions dumps  J/clustering_check/dumps/dev/30 (run_clustering_check.slurm)    DIR/rc/dumps/dev/30
  crt_anchor_lab dumps   J/anchor_lab/dumps/dev/30 (job 2634, run_anchor_lab.slurm)       DIR/anc/dumps/dev/30
  C1-C3 WAVs, rows       J/clustering_check/lab_r3a/<system>/<pair>.wav, rows r3a_<system>  DIR/lab/cand/<system>/,
                         of J/clustering_check/lab_r3_metrics.jsonl (run_lab_r3.slurm)      rows cand_<system> of
  C4, C5 WAVs, rows      J/clustering_check/lab_r4p/<system>/, rows r4p_<system> of         DIR/lab/metrics.jsonl
                         J/clustering_check/lab_r4_metrics.jsonl (run_round4.slurm step 3)
Checks, each run when its inputs are given; a pair whose file is missing on either side fails the check:
  codes      --rc OUT_DIR (run_conversions --out_dir): S1, the retrieved codes, and S3, the τ = 0.4 gate (CB1-18 of
             the retrieved codes replaced by crt_argmax where mask_tau0.4 holds, or crt_conf > 0.4 in a dump without
             that mask, as run_conversions.gate), equal the reference's on every pair.
  out_codes  --anc OUT_DIR (crt_anchor_lab --out_dir, repeatable): the out_codes of A1-A5 and C1, matched by name
             through the dumps' systems, equal the reference's on every pair.
  wavs       --cand DIR (crt_decode_lab --out_dir of C1-C5) and --cand_metrics (its metrics.py rows cand_<system>):
             per candidate, the PCM_16 WAVs equal the reference's sample for sample on every pair; if they do not,
             the candidate passes only if on every pair the Whisper large-v3 hypothesis is identical and SIM
             (sim_wavlm) and UTMOS differ by at most 0.001 and 0.01 from the reference rows. The largest sample
             difference is reported either way.
  lists      --lists DIR:SETS (repeatable): make_pairs.check_lists (every Dev / LTC-2k source speaker is a target,
             every VCTK-2k source speaker has a source-E set, no E set holds a source or foreign recording).
Reported, not checked: --timings (the job's "<epoch s> <step>" stamps: seconds per step, per pair and per 1,000
pairs), --runs (run_conversions runs.jsonl: the last end record's counts and per-stage seconds), --sizes DUMPS_DIR
(files, bytes, and the compressed bytes per pair of each array group: every *_logits array on its own, the other
arrays of each anchor recipe, the rest).
--out: JSON {"passed": bool, "summary": str, "checks": {...}, "info": {...}}; select_crt_al.py --repro reads "passed"
and "summary". Exit 1 unless every check that ran passed.
"""
import argparse
import collections
import contextlib
import glob
import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_pairs  # noqa: E402

SCREEN_MOD = 20
TAU = 0.4
ANC = {'A1': 'R1_s0_pm_rq_T1', 'A2': 'R3_r0_pm_rq_T1', 'A3': 'R3_s0_argmax', 'A4': 'R3_s0_tau0.4',
       'A5': 'src0_faiss', 'C1': 'R3_s0_pm_rq_T1'}             # crt_anchor_lab names (SPEC)
CANDS = {'C1': 'pm_rq_T1', 'C2': 'pm18_rq', 'C3': 'gv1_pm_cont_T1', 'C4': 'gv1_shrink0.85_cont',
         'C5': 'gv0.5_shrink0.75_cont'}                         # crt_decode_lab names
SIM_TOL, UTMOS_TOL = 0.001, 0.01
RECIPES = ('R1', 'R3', 'R3s', 'Psim', 'G0.5')
SHOW = 10    # pair ids listed per kind of failure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help='pairs/dev.jsonl; the screen is its rows k %% 20 == 0')
    p.add_argument('--set', default='dev')
    p.add_argument('--budget', default='30')
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument('--ref_screen', help='the journal/ directory that holds the screen outputs')
    ref.add_argument('--ref_d0', help="job D0's output root (OUT of job_d0.slurm)")
    p.add_argument('--rc', default='', help='run_conversions --out_dir to check (S1, S3 codes)')
    p.add_argument('--anc', action='append', default=[], help='crt_anchor_lab --out_dir to check (repeatable)')
    p.add_argument('--cand', default='', help='crt_decode_lab --out_dir of C1-C5 to check')
    p.add_argument('--cand_metrics', default='', help='metrics.py rows of the --cand WAVs (cand_<system>)')
    p.add_argument('--lists', action='append', default=[], help='DIR:SETS for make_pairs.check_lists (repeatable)')
    p.add_argument('--timings', default='', help='the job\'s "<epoch s> <step>" stamps')
    p.add_argument('--runs', action='append', default=[], help='run_conversions runs.jsonl (repeatable)')
    p.add_argument('--sizes', action='append', default=[], help='a dumps directory to measure (repeatable)')
    p.add_argument('--label', default='', help='what is checked, for the summary (default: "vs <reference>")')
    p.add_argument('--out', required=True, help='JSON verdict')
    return p.parse_args()


def screen_ids(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r['pair_id'] for r in rows[::SCREEN_MOD]]


def reference(a):
    """{'name', 'rc' and 'anc' (--out_dir roots), 'cand': {candidate: (WAV dir, metrics file, row prefix)}}."""
    if a.ref_screen:
        c = os.path.join(a.ref_screen, 'clustering_check')
        r3 = (os.path.join(c, 'lab_r3a'), os.path.join(c, 'lab_r3_metrics.jsonl'), 'r3a_')
        r4 = (os.path.join(c, 'lab_r4p'), os.path.join(c, 'lab_r4_metrics.jsonl'), 'r4p_')
        return {'name': 'the screen', 'rc': c, 'anc': os.path.join(a.ref_screen, 'anchor_lab'),
                'cand': {'C1': r3, 'C2': r3, 'C3': r3, 'C4': r4, 'C5': r4}}
    lab = (os.path.join(a.ref_d0, 'lab', 'cand'), os.path.join(a.ref_d0, 'lab', 'metrics.jsonl'), 'cand_')
    return {'name': 'D0', 'rc': os.path.join(a.ref_d0, 'rc'), 'anc': os.path.join(a.ref_d0, 'anc'),
            'cand': {c: lab for c in CANDS}}


def dump_path(out_dir, a, pid):
    return os.path.join(out_dir, 'dumps', a.set, a.budget, pid + '.npz')


def both(*paths):
    return all(os.path.exists(p) for p in paths)


def gated(z):
    """S1 and S3 codes of a run_conversions dump (retrieved_codes; run_conversions.gate at τ = 0.4) and the mask."""
    s1 = z['retrieved_codes'].astype(np.int64)
    s3 = s1.copy()
    lo = int(z['crt_first_cb']) if 'crt_first_cb' in z.files else 1
    tok = z['crt_argmax'].astype(np.int64)
    key = f'mask_tau{TAU:g}'
    mask = z[key].astype(bool) if key in z.files else z['crt_conf'] > np.float32(TAU)
    sel = s3[lo:lo + len(tok)]
    sel[mask] = tok[mask]
    return s1, s3, mask


def read_pair(out, pid, fn, *paths):
    """fn(*files) for one pair; a missing or unreadable file is recorded in out and gives None."""
    if not both(*paths):
        out['missing'].append(pid)
        return None
    try:
        return fn(*paths)
    except Exception as e:     # an unreadable file fails its pair
        out['unreadable'][pid] = f'{type(e).__name__}: {e}'
        return None


def load_gated(*paths):
    res = []
    for p in paths:
        with np.load(p) as z:
            res.append(gated(z))
    return res


def check_codes(test, ref, ids, a):
    out = {'test': test, 'ref': ref, 'n_pairs': len(ids), 'missing': [], 'unreadable': {}, 's1_differ': [],
           's3_differ': [], 'mask_differ_tokens': 0, 'n_equal': 0}
    for pid in ids:
        got = read_pair(out, pid, load_gated, dump_path(test, a, pid), dump_path(ref, a, pid))
        if got is None:
            continue
        (t1, t3, tm), (r1, r3, rm) = got
        e1, e3 = np.array_equal(t1, r1), np.array_equal(t3, r3)
        if not e1:
            out['s1_differ'].append(pid)
        if not e3:
            out['s3_differ'].append(pid)
        out['n_equal'] += e1 and e3
        out['mask_differ_tokens'] += int((tm != rm).sum()) if tm.shape == rm.shape else int(tm.size)
    out['passed'] = out['n_equal'] == len(ids)
    return out


def load_out_codes(*paths):
    res = []
    for p in paths:
        with np.load(p) as z:
            oc = z['out_codes']
            res.append({str(s): oc[i] for i, s in enumerate(z['systems'])})
    return res


def check_out_codes(test, ref, ids, a):
    out = {'test': test, 'ref': ref, 'n_pairs': len(ids), 'missing': [], 'unreadable': {},
           'systems': {k: {'system': s, 'n_equal': 0, 'differ': [], 'absent': []} for k, s in ANC.items()}}
    for pid in ids:
        got = read_pair(out, pid, load_out_codes, dump_path(test, a, pid), dump_path(ref, a, pid))
        if got is None:
            continue
        ct, cr = got
        for v in out['systems'].values():
            s = v['system']
            if s not in ct or s not in cr:
                v['absent'].append(pid)
            elif np.array_equal(ct[s], cr[s]):
                v['n_equal'] += 1
            else:
                v['differ'].append(pid)
    out['passed'] = all(v['n_equal'] == len(ids) for v in out['systems'].values())
    return out


def load_rows(path):
    """metrics.py rows {(pair_id, system): row}, the last row per key (metrics.load_results)."""
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r['pair_id'], r['system'])] = r
    return out


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def metric_diff(t, r):
    """'' when two metrics.py rows agree within the §8.1 item 9 tolerance, else the reason."""
    if t is None or r is None:
        return 'no metrics row'
    if 'error' in t or 'error' in r:
        return 'error row'
    ht, hr = ((x.get('wer_large_v3') or {}).get('hyp') for x in (t, r))
    if ht is None or hr is None:
        return 'no Whisper large-v3 hypothesis'
    if ht != hr:
        return 'Whisper hypothesis differs'
    for k, tol in (('sim_wavlm', SIM_TOL), ('utmos', UTMOS_TOL)):
        if not (finite(t.get(k)) and finite(r.get(k))):
            return f'no {k}'
        if abs(t[k] - r[k]) > tol:
            return f'|d{k}| = {abs(t[k] - r[k]):.4g} > {tol}'
    return ''


def read_pcm(*paths):
    return [sf.read(p, dtype='int16') for p in paths]


def check_wavs(cand, cand_metrics, ref, ids):
    rows_t, rows_r, out = load_rows(cand_metrics), {}, {}
    for c, s in CANDS.items():
        rdir, rmet, pre = ref['cand'][c]
        if rmet not in rows_r:
            rows_r[rmet] = load_rows(rmet)
        v = {'system': s, 'test': os.path.join(cand, s), 'ref': os.path.join(rdir, s), 'n_pairs': len(ids),
             'missing': [], 'unreadable': {}, 'n_identical': 0, 'shape_differs': [], 'max_abs_diff': 0}
        for pid in ids:
            got = read_pair(v, pid, read_pcm, os.path.join(cand, s, pid + '.wav'), os.path.join(rdir, s, pid + '.wav'))
            if got is None:
                continue
            (x, sx), (y, sy) = got
            n = min(len(x), len(y))
            d = int(np.abs(x[:n].astype(np.int32) - y[:n].astype(np.int32)).max()) if n else 0
            v['max_abs_diff'] = max(v['max_abs_diff'], d)
            if sx != sy or x.shape != y.shape:
                v['shape_differs'].append(pid)
            elif d == 0:
                v['n_identical'] += 1
        v['max_abs_diff_float'] = v['max_abs_diff'] / 32768
        v['wavs_identical'] = v['n_identical'] == len(ids)
        v['metrics_check'] = None
        if not v['wavs_identical']:
            pr = [(pid, rows_t.get((pid, 'cand_' + s)), rows_r[rmet].get((pid, pre + s))) for pid in ids]
            bad = {pid: w for pid, w in ((pid, metric_diff(t, r)) for pid, t, r in pr) if w}
            ok = [(t, r) for _, t, r in pr if t and r and 'error' not in t and 'error' not in r]
            v['metrics_check'] = {
                'test_rows': cand_metrics, 'ref_rows': rmet, 'test_system': 'cand_' + s, 'ref_system': pre + s,
                'n_within': len(ids) - len(bad), 'n_failed': len(bad), 'failed': dict(list(bad.items())[:SHOW]),
                **{f'max_d{k}': max((abs(t[k] - r[k]) for t, r in ok if finite(t.get(k)) and finite(r.get(k))),
                                    default=None) for k in ('sim_wavlm', 'utmos')}}
        v['passed'] = (not v['missing'] and not v['unreadable']
                       and (v['wavs_identical'] or v['metrics_check']['n_failed'] == 0))
        out[c] = v
    return out


def check_lists(specs):
    out = []
    for spec in specs:
        if ':' not in spec:
            raise SystemExit(f'ERROR: --lists {spec!r} is not DIR:SETS')
        d, sets = spec.rsplit(':', 1)
        buf, err = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(buf):
                ok = bool(make_pairs.check_lists(d, sets.split(',')))
        except Exception as e:    # a list that cannot be read fails the check
            ok, err = False, f'{type(e).__name__}: {e}'
        lines = buf.getvalue().splitlines() + ([f'[check] ERROR {err}'] if err else [])
        print('\n'.join(lines), flush=True)
        out.append({'dir': d, 'sets': sets.split(','), 'passed': ok, 'lines': lines})
    return out


def timings(path, n):
    st = [(int(t), step) for t, _, step in (line.strip().partition(' ') for line in open(path)) if t]
    return [{'step': s0, 's': t1 - t0, 's_per_pair': round((t1 - t0) / n, 2),
             'h_per_1000_pairs': round((t1 - t0) / n * 1000 / 3600, 2)} for (t0, s0), (t1, _) in zip(st, st[1:])]


def runs_info(path):
    ends = [r for r in (json.loads(line) for line in open(path) if line.strip()) if r.get('event') == 'end']
    keys = ('time', 'n_pairs_done', 'n_errors', 'n_shortfall', 'total_s', 'stage_s')
    return {'path': path, **({k: ends[-1].get(k) for k in keys} if ends else {'end': None})}


def group(name):
    if name.endswith('_logits'):
        return name
    head = name.split('_', 1)[0]
    return head if head in RECIPES else 'other'


def sizes(d):
    files = sorted(glob.glob(os.path.join(d, '*.npz')))
    g, bad = collections.Counter(), []
    for f in files:
        try:
            with zipfile.ZipFile(f) as z:
                for i in z.infolist():
                    g[group(i.filename[:-4] if i.filename.endswith('.npy') else i.filename)] += i.compress_size
        except zipfile.BadZipFile:
            bad.append(os.path.basename(f))
    n, tot = len(files), sum(os.path.getsize(f) for f in files)
    return {'dir': d, 'n_files': n, 'bytes': tot, 'mb_per_pair': round(tot / n / 1e6, 3) if n else None,
            'mb_per_pair_by_group': {k: round(v / n / 1e6, 3) for k, v in sorted(g.items())} if n else {},
            'unreadable': bad}


def summarize(checks):
    parts = []
    if 'codes' in checks:
        c = checks['codes']
        parts.append(f"S1/S3 codes {'OK' if c['passed'] else 'FAIL'} ({c['n_equal']}/{c['n_pairs']} pairs equal"
                     + (f", missing {len(c['missing'])}" if c['missing'] else '')
                     + (f", unreadable {len(c['unreadable'])}" if c['unreadable'] else '') + ')')
    for c in checks.get('out_codes', []):
        bad = {k: c['n_pairs'] - v['n_equal'] for k, v in c['systems'].items() if v['n_equal'] < c['n_pairs']}
        parts.append(f"A1-A5,C1 out_codes of {os.path.basename(os.path.normpath(c['test']))} "
                     + ('OK' if c['passed'] else f"FAIL (missing {len(c['missing'])}, unreadable "
                                                 f"{len(c['unreadable'])}, pairs not equal {bad})"))
    if 'wavs' in checks:
        w = []
        for k, v in checks['wavs'].items():
            m = v['metrics_check']
            w.append(f"{k} " + ('identical' if v['wavs_identical'] else
                                f"within tolerance (max sample diff {v['max_abs_diff']})" if v['passed'] else
                                f"FAIL (missing {len(v['missing'])}, unreadable {len(v['unreadable'])}, metrics "
                                f"failed {m['n_failed'] if m else '-'}, max sample diff {v['max_abs_diff']})"))
        parts.append('C1-C5 WAVs: ' + ', '.join(w))
    if 'lists' in checks:
        parts.append('pair lists ' + ('OK' if all(x['passed'] for x in checks['lists']) else
                                      'FAIL ' + str([f"{x['dir']}:{','.join(x['sets'])}" for x in checks['lists']
                                                     if not x['passed']])))
    return '; '.join(parts)


def main():
    a = parse_args()
    ids = screen_ids(a.pairs)
    ref = reference(a)
    checks = {}
    if a.rc:
        checks['codes'] = check_codes(a.rc, ref['rc'], ids, a)
    if a.anc:
        checks['out_codes'] = [check_out_codes(d, ref['anc'], ids, a) for d in a.anc]
    if a.cand:
        checks['wavs'] = check_wavs(a.cand, a.cand_metrics, ref, ids)
    if a.lists:
        checks['lists'] = check_lists(a.lists)
    if not checks:
        raise SystemExit('ERROR: no check requested (--rc, --anc, --cand, --lists)')
    flags = {'codes': lambda c: c['passed'], 'out_codes': lambda c: all(x['passed'] for x in c),
             'wavs': lambda c: all(x['passed'] for x in c.values()), 'lists': lambda c: all(x['passed'] for x in c)}
    passed = all(flags[k](v) for k, v in checks.items())
    info = {'timings': timings(a.timings, len(ids)) if a.timings and os.path.exists(a.timings) else None,
            'runs': [runs_info(f) for f in a.runs if os.path.exists(f)], 'sizes': [sizes(d) for d in a.sizes]}
    label = a.label or f"vs {ref['name']}"
    res = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'label': label, 'reference': ref,
           'pairs': os.path.abspath(a.pairs), 'n_screen_pairs': len(ids),
           'tolerance': {'sim_wavlm': SIM_TOL, 'utmos': UTMOS_TOL}, 'passed': passed,
           'summary': f'{label} ({len(ids)} screen pairs): {summarize(checks)}', 'checks': checks, 'info': info}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.tmp', 'w') as f:
        json.dump(res, f, indent=1)
    os.replace(a.out + '.tmp', a.out)
    for k, v in checks.items():
        for x in (v if isinstance(v, list) else v.values() if k == 'wavs' else [v]):
            if not x['passed']:
                print(f'  FAIL {k}: ' + json.dumps({kk: vv for kk, vv in x.items() if kk != 'lines'})[:2000],
                      flush=True)
    for t in info['timings'] or []:
        print(f"  step {t['step']}: {t['s']} s, {t['s_per_pair']} s per pair, "
              f"{t['h_per_1000_pairs']} h per 1,000 pairs")
    for s in info['sizes']:
        print(f"  dumps {s['dir']}: {s['n_files']} files, {s['mb_per_pair']} MB per pair {s['mb_per_pair_by_group']}")
    print(f"check_d0: passed = {passed}: {res['summary']} -> {a.out}", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
