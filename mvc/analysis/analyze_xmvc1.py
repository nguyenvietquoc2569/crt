#!/usr/bin/env python3
"""
Exploratory addition X-mvc-1 of mvc/ANALYSIS_PLAN_mvc_v1.md §4.10 (DEVIATIONS_mvc_v1.md operational note 4), run on the
Mac after the X-mvc-1 tree is copied from ceai and DNSMOS has scored it (dnsmos_xmvc1.py). CPU only. Descriptive:
estimates with 95 % CIs only. It has no hypothesis test, no multiplicity correction and no decision, and it never
replaces or relabels a mvc_v1 result.

mvc_r1x300 is R1x's 300k EMA snapshot converted with mvc_v1's frozen inference settings, at 30 s on VCTK-2k and LTC-2k.
For each set: the levels with 95 % CIs of mvc_r1x300 and mvc_v1 (capped WER, SIM, UTMOS, SIM_src, ECAPA SIM, DNSMOS
OVRL and P808, the acoustic guards) and the paired differences mvc_r1x300 − mvc_v1 over the pairs where neither failed
(capped WER, SIM, UTMOS, ECAPA SIM, DNSMOS OVRL). On VCTK-2k: the paired natural-level leakage NL per stratum of both
systems, computed by analyze_mvc.nl_family (§3.3), and their difference, the paired SIM_src difference over the same
pairs with P (target_copy's term cancels pair by pair, so it equals NL(mvc_r1x300) − NL(mvc_v1) when neither system
failed on the stratum's pairs). Every number comes from analyze_tests.py, stats.py and
analyze_mvc.py, imported unchanged and checked by sha256 (crossed bootstrap B = 10,000, seed 20260928; VCTK-2k
independent, LTC-2k linked, as analyze_mvc.py). stats.compare's p value and Wilcoxon test and NL's CI-rule decision are
dropped from every result (ci_only).

Inputs (all required; a missing one is a refusal before any output):
  --xr          the X-mvc-1 results, laid out as $MVC/xmvc1 on ceai: cfg/mvc_r1x300.json, snapshot.sha256 (the
                snapshot and run_dir/stats.npz, which must be R1's frozen stats.npz), <set>.done (naming the snapshot
                sha256 and the MimiFlow-VC selection record), <set>.sha256 (the job's list of its result files: every
                file listed must be here with that sha256, and every X-mvc-1 file read here other than DNSMOS's must be
                listed), m/t1 (VCTK-2k) and m/t2 (LTC-2k) metrics, x2/ecapa/<set>_30{,_src}.jsonl,
                dnsmos/<set>_30_mvc_r1x300.jsonl, guard/<set>_30.jsonl, with rows of mvc_r1x300 only.
  --mr          the T1 results (MR): mvc_v1's rows in m/t1, m/t2 (m.done naming the record), x2/ecapa, dnsmos, guard.
  --v1          the journal results (V1): the pair lists, the VCTK-2k source-E file and target_copy@src (m/t1, for NL).
  --t1          TEST_RESULTS_mvc_v1.json, written by the analyze_mvc.py imported here: every --mr and --v1 file read
                here must be one of its inputs (same sha256), so every mvc_v1 number is from the T1 rows as analysed.
  --record_cfg  the selection record's mvc_v1.json (sha256 as in mvc_selection.env): the X-mvc-1 cfg differs from it
                only in run_dir and ckpt, and snapshot.sha256 names that run_dir/ckpt.
Writes --out (.md) and <out stem>.json with the sha256 of every input; refuses to overwrite either.

    /Users/harry/duplex/.venv-stats/bin/python mvc/analysis/analyze_xmvc1.py
"""
import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_mvc as am  # noqa: E402  (puts benchmark/ on sys.path)
import analyze_tests as at  # noqa: E402
import stats  # noqa: E402

ANALYZE_MVC_SHA256 = '9897f7e0e99fc6dbaf09d1ac12cf9b5dbb0b63afc3685741ee74501788a5958c'   # the record's, T1's
RECORD_SHA256 = '9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42'        # MVC_SELECTION_RECORD_v1.md
MVC_V1_CFG_SHA256 = '00184073f7c3caae6551e76f38f2e4d508bbdc844488fd9cb63968b04db0fbbd'    # CFG_MVC_V1_SHA256
R1_STATS_SHA256 = 'f6aae232b3e493e848d2f030776c7551a175184e6f73717b26ed9049a02935ef'      # FREEZE: runs/r1/stats.npz
X, MVC, SRC, SETS = 'mvc_r1x300', am.MVC, am.SRC, am.SETS
SYSTEMS = (X, MVC)
PART = {'vctk2k': 't1', 'ltc2k': 't2'}
RUN, CKPT = 'r1x', 'ema_0300000.pt'
LEVELS = (('wer', 'capped WER %'), ('sim', 'SIM'), ('utmos', 'UTMOS'), ('sim_src', 'SIM_src'),
          ('ecapa_sim', 'ECAPA SIM'), ('dnsmos_ovrl', 'DNSMOS OVRL'), ('dnsmos_p808', 'DNSMOS P808'))
DIFFS = ('wer', 'sim', 'utmos', 'ecapa_sim', 'dnsmos_ovrl')
# the keys of a stats.level / stats.compare result kept here: estimates, CIs and counts, never p or Wilcoxon
KEEP = ('metric', 'n_pairs', 'fail', 'fail_a', 'fail_b', 'undefined', 'n', 'est', 'lo', 'hi', 'se', 'redraws', 'B',
        'mode', 'n_src', 'n_tgt', 'a', 'b', 'note', 'imputed_a', 'imputed_b', 'unimputable', 'worst_a', 'worst_b')


def notes(snap, snap_sha, stats_npz):
    return ['**Exploratory (§4.10).** X-mvc-1 was decided on 29 Oct 2026 from R1x\'s 300k Dev-S100 numbers, after T1 '
            'of mvc_v1 was complete and recorded (DEVIATIONS_mvc_v1.md, operational note 4). It is in no hypothesis '
            'family of §5.',
            '**Descriptive only.** Estimates with 95 % CIs; nothing here is a test or a decision.',
            '**It never replaces or relabels a mvc_v1 result.** The mvc_v1 numbers below are computed from the T1 rows '
            'that TEST_RESULTS_mvc_v1.md analysed (same sha256), with the same functions and seed; the T1 results '
            'stand as reported there.',
            f'**mvc_r1x300** is R1x\'s EMA snapshot at step 300k, `{snap}` (sha256 {snap_sha}), converted with mvc_v1\'s '
            'frozen inference settings: the selection record\'s mvc_v1.json with only run_dir and ckpt changed, and '
            f'R1\'s normalisation statistics (`{stats_npz}`, sha256 {R1_STATS_SHA256}, as frozen for R1). 30 s '
            'reference budget only, VCTK-2k and LTC-2k.',
            'R1x continued R1 from step 200k with a restarted learning-rate schedule in its own run directory (§4.10); '
            'it never becomes mvc_v1 (§0.3 row 9).',
            '**R1x\'s 250k snapshot was unusable**: a transient loss spike at step ~239.4k corrupted the EMA '
            '(DEVIATIONS_mvc_v1.md, operational note 3); the EMA had recovered by 300k (operational note 4). No other '
            'R1x snapshot runs on the test sets.',
            'Δ = mvc_r1x300 − mvc_v1 per pair, over the endpoint\'s pairs where neither conversion failed; failed '
            'conversions are counted and, where the counts differ, shown at their worst case (§5.4).']


# ── code, cfg, done files, inputs ────────────────────────────

def check_code():
    """analyze_mvc.check_frozen (stats.py, analyze_tests.py, analyze_exploratory.py) and analyze_mvc.py itself."""
    frozen = am.check_frozen()
    got = at.sha256(am.__file__)
    if got != ANALYZE_MVC_SHA256:
        raise SystemExit(f'ERROR: {am.__file__} has sha256 {got}, not the {ANALYZE_MVC_SHA256} of the selection record '
                         f'and TEST_RESULTS_mvc_v1')
    return frozen, got


def check_cfg(xr, record_cfg):
    """The X-mvc-1 cfg is the record's mvc_v1.json with only run_dir and ckpt changed (to R1x's 300k snapshot), and
    snapshot.sha256 names that snapshot and run_dir/stats.npz, with R1's frozen stats.npz sha256. Returns (info,
    inputs)."""
    got = at.sha256(record_cfg)
    if got != MVC_V1_CFG_SHA256:
        raise SystemExit(f'ERROR: {record_cfg} has sha256 {got}, not the record\'s mvc_v1.json {MVC_V1_CFG_SHA256}')
    paths = {'cfg': os.path.join(xr, 'cfg', f'{X}.json'), 'snapshot': os.path.join(xr, 'snapshot.sha256')}
    for p in paths.values():
        if not os.path.exists(p):
            raise SystemExit(f'ERROR: {p} missing')
    v1c, xc = json.load(open(record_cfg)), json.load(open(paths['cfg']))
    diff = sorted(k for k in set(v1c) | set(xc) if v1c.get(k, '?') != xc.get(k, '?'))
    if diff != ['ckpt', 'run_dir'] or xc.get('ckpt') != CKPT or \
            os.path.basename(str(xc.get('run_dir')).rstrip('/')) != RUN:
        raise SystemExit(f'ERROR: {paths["cfg"]} differs from mvc_v1.json in {diff}; X-mvc-1 changes only run_dir (to '
                         f'.../{RUN}) and ckpt (to {CKPT})')
    m = re.fullmatch(r'([0-9a-f]{64})  (/\S+)\n([0-9a-f]{64})  (/\S+)\n', open(paths['snapshot']).read())
    snap, st = os.path.join(xc['run_dir'], xc['ckpt']), os.path.join(xc['run_dir'], 'stats.npz')
    if not m or m[2] != snap or m[4] != st:
        raise SystemExit(f'ERROR: {paths["snapshot"]} is not the sha256sum lines of {snap} and {st}')
    if m[3] != R1_STATS_SHA256:
        raise SystemExit(f'ERROR: {paths["snapshot"]}: {st} has sha256 {m[3]}, not R1\'s frozen stats.npz '
                         f'{R1_STATS_SHA256}')
    info = {'cfg': xc, 'changed': {k: [v1c[k], xc[k]] for k in diff}, 'snapshot': snap, 'snapshot_sha256': m[1],
            'stats_npz': st, 'stats_npz_sha256': m[3]}
    return info, [{'path': os.path.abspath(p), 'sha256': at.sha256(p)} for p in (*paths.values(), record_cfg)]


def check_done(xr, mr, v1, snap_sha):
    """X-mvc-1's <set>.done under the record and the snapshot; T1 parts t1, t2 under the record; job M part t1 (NL's
    target_copy@src) under the journal selection."""
    done = {}
    for s in SETS:
        line = am.done_line(os.path.join(xr, f'{s}.done'), RECORD_SHA256, f'X-mvc-1 on {s}')
        if f'snapshot sha256 {snap_sha} ' not in line:
            raise SystemExit(f'ERROR: {xr}/{s}.done ({line!r}) does not name the snapshot sha256 {snap_sha}')
        done[f'X-mvc-1 {s}'] = line
    done.update({f'MimiFlow-VC T1 part {k}': v for k, v in
                 am.check_parts(os.path.join(mr, 'm'), ['t1', 't2'], RECORD_SHA256, 'MimiFlow-VC T1').items()})
    done['journal job M part t1'] = am.done_line(os.path.join(v1, 'm', 't1', 'm.done'), am.JOURNAL_SELECTION_SHA256,
                                                 'journal job M part t1')
    return done


def check_lists(xr):
    """<xr>/<set>.sha256, written by the job before <set>.done: every file listed is here with that sha256. Returns
    ({absolute path: sha256}, the lists as inputs)."""
    listed, inputs = {}, []
    for s in SETS:
        lst = os.path.join(xr, f'{s}.sha256')
        if not os.path.exists(lst):
            raise SystemExit(f'ERROR: {lst} missing: copy it from ceai with the files it lists')
        for line in open(lst):
            m = re.fullmatch(r'([0-9a-f]{64})  ([^/\s]\S*)\n', line)
            if not m:
                raise SystemExit(f'ERROR: {lst}: {line!r} is not a sha256sum line with a relative path')
            path = os.path.abspath(os.path.join(xr, m[2]))
            if not os.path.exists(path) or at.sha256(path) != m[1]:
                raise SystemExit(f'ERROR: {path} is not the file {lst} lists (sha256 {m[1]}): copy it again from ceai')
            listed[path] = m[1]
        inputs.append({'path': os.path.abspath(lst), 'sha256': at.sha256(lst)})
    return listed, inputs


def check_t1_inputs(t1_path, inputs):
    """Every T1 and journal file read here is an input of TEST_RESULTS_mvc_v1 (same sha256)."""
    t1 = json.load(open(t1_path))
    if t1.get('script_sha256') != ANALYZE_MVC_SHA256:
        raise SystemExit(f'ERROR: {t1_path} was written by analyze_mvc.py {t1.get("script_sha256")}, not '
                         f'{ANALYZE_MVC_SHA256}')
    known = {x.get('sha256') for x in t1.get('inputs') or []}
    bad = [x['path'] for x in inputs if x['sha256'] not in known]
    if bad:
        raise SystemExit(f'ERROR: {bad} are not inputs of {t1_path} (sha256): the mvc_v1 and journal rows must be '
                         f'those the T1 analysis read')


def keep_systems(R, keep):
    return {b: {s: v for s, v in sy.items() if s.split('@')[0] in keep} for b, sy in R.items()}


def load(xr, mr, v1, pairs, src_e):
    """The four TestSets of each set (metrics, ECAPA, DNSMOS, guards) with the rows of mvc_r1x300, mvc_v1 and, on
    VCTK-2k, target_copy; X-mvc-1 files hold mvc_r1x300 only and the T1 files T1's systems only. Returns the sets and
    the inputs of X-mvc-1 and of T1 / the journal."""
    sets, esets, dsets, gsets, xin, tin = {}, {}, {}, {}, [], []
    for s in SETS:
        Rx, ins = at.load_set(os.path.join(xr, 'm'), s, [(PART[s], 30)], pairs[s])
        xin += ins
        Rm, ins = at.load_set(os.path.join(mr, 'm'), s, [(PART[s], 30)], pairs[s])
        tin += ins
        parts = [('X-mvc-1', am.check_systems(Rx, (X,), f'{s} X-mvc-1')),
                 ('MimiFlow-VC T1', keep_systems(am.check_systems(Rm, am.MVC_SYSTEMS, f'{s} MimiFlow-VC T1'), (MVC,)))]
        if s == 'vctk2k':
            Rj, ins = at.load_set(os.path.join(v1, 'm'), s, [('t1', 30)], pairs[s])
            tin += ins
            parts.append(('journal job M', keep_systems(Rj, ('target_copy',))))
        sets[s] = am.make_set(s, pairs[s], am.merge(parts, whole=True), src_e, keep=SYSTEMS + ('target_copy',))
        Ex, ins = am.load_ecapa([('X-mvc-1', os.path.join(xr, 'x2', 'ecapa'), (X,))], s, pairs[s])
        xin += ins
        Em, ins = am.load_ecapa([('MimiFlow-VC T1', os.path.join(mr, 'x2', 'ecapa'), am.MVC_SYSTEMS)], s, pairs[s])
        tin += ins
        E = am.merge([('X-mvc-1 ECAPA', Ex), ('MimiFlow-VC T1 ECAPA', keep_systems(Em, (MVC,)))], whole=True)
        esets[s] = am.make_set(s, pairs[s], E, src_e, keep=SYSTEMS)
        Dx, ins = am.load_glob(os.path.join(xr, 'dnsmos'), s, pairs[s], 'DNSMOS')
        xin += ins
        Dm, inp = am.load_rows(os.path.join(mr, 'dnsmos', f'{s}_30_{MVC}.jsonl'), s, pairs[s])
        tin.append(inp)
        D = am.merge([('X-mvc-1 DNSMOS', am.check_systems(Dx, (X,), f'{s} X-mvc-1 DNSMOS')),
                      ('MimiFlow-VC T1 DNSMOS', am.check_systems({30: Dm}, (MVC,), f'{s} MimiFlow-VC T1 DNSMOS'))],
                     whole=True)
        dsets[s] = am.make_set(s, pairs[s], D, src_e, keep=SYSTEMS)
        Gx, ins = am.load_glob(os.path.join(xr, 'guard'), s, pairs[s], 'guard')
        xin += ins
        Gm, inp = am.load_rows(os.path.join(mr, 'guard', f'{s}_30.jsonl'), s, pairs[s])
        tin.append(inp)
        G = am.merge([('X-mvc-1 guard', am.check_systems(Gx, (X,), f'{s} X-mvc-1 guard')),
                      ('MimiFlow-VC T1 guard', keep_systems({30: Gm}, (MVC,)))], whole=True)
        gsets[s] = am.make_set(s, pairs[s], G, src_e, keep=SYSTEMS)
    return sets, esets, dsets, gsets, xin, tin


def required(sets, esets, dsets):
    """One metric, ECAPA and DNSMOS row per pair at 30 s for both systems, @src rows as lab_rows.py and score_ecapa.py
    write them (analyze_mvc.src_problems), and target_copy@src without error on every VCTK-2k pair with P (NL).
    Returns the problems."""
    out = []
    for s in SETS:
        ts, es, ds = sets[s], esets[s], dsets[s]
        n, with_e = len(ts.pairs), set(am.src_pids(ts))
        for sy in SYSTEMS:
            for what, k in (('', len(ts.rows(sy))), (' ECAPA', len(es.rows(sy))), (' DNSMOS', len(ds.rows(sy)))):
                if k != n:
                    out.append(f'{s}{what} {sy}: {k} rows for {n} pairs')
            out += am.src_problems(f'{s} {sy}{SRC}', ts.rows(sy), ts.rows(sy + SRC), with_e)
            out += am.src_problems(f'{s} ECAPA {sy}{SRC}', es.rows(sy), es.rows(sy + SRC), with_e)
    v = sets['vctk2k']
    par, rows = set(v.subset('par')), v.rows('target_copy' + SRC)
    bad = sorted(k for k in par if stats.failed(rows.get(k)) or stats._value(rows[k], 'sim_wavlm') is None)
    if set(rows) != par or bad:
        out.append(f'vctk2k target_copy@src: rows on {len(set(rows) & par)} of the {len(par)} pairs with P, '
                   f'{len(set(rows) - par)} elsewhere, {len(bad)} failed or undefined')
    return out


# ── analysis ─────────────────────────────────────────────────

def ci_only(res):
    """A stats.level / stats.compare result without its p value, Wilcoxon test or decision."""
    if not res:
        return res
    out = {k: res[k] for k in KEEP if k in res}
    if res.get('worst_case'):
        out['worst_case'] = ci_only(res['worst_case'])
    return out


def levels(ts, es, ds, systems=SYSTEMS):
    """Each system's levels with 95 % CIs: capped WER (wer_primary pairs), SIM, UTMOS, SIM_src (sources with an E set),
    ECAPA SIM, DNSMOS OVRL and P808 (all pairs), as analyze_mvc.py's levels, ECAPA and DNSMOS levels."""
    keep = set(am.src_pids(ts))
    out = {}
    for n in systems:
        rs = ts.rows(n)
        row = {'n_rows': len(rs), 'fail': sum(stats.failed(r) for r in rs.values())}
        for q in ('wer', 'sim', 'utmos', 'sim_src'):
            row[q] = ci_only(ts.level(n, q, pids=keep if q == 'sim_src' else None))
        row['ecapa_sim'] = ci_only(stats.level(es.rows(n), es.subset('all'), es.spk, am.ECAPA_KEY, mode=es.mode))
        for k in ('OVRL', 'P808'):
            row[f'dnsmos_{k.lower()}'] = ci_only(stats.level(ds.rows(n), ds.subset('all'), ds.spk, k, mode=ds.mode))
        out[n] = row
    return out


def guards(gs, systems=SYSTEMS):
    """analyze_mvc.guard_levels of one set for the two systems."""
    out = {}
    for n in systems:
        rs = gs.rows(n)
        row = {'n_rows': len(rs), 'fail': sum(stats.failed(r) for r in rs.values())}
        if rs:
            row.update({k: ci_only(stats.level(rs, gs.subset('all'), gs.spk, k, mode=gs.mode)) for k in am.GUARD_FIELDS})
        out[n] = row
    return out


def differences(s, sets, esets, dsets):
    """mvc_r1x300 − mvc_v1, paired: capped WER, SIM and UTMOS through analyze_tests.hypothesis (the endpoint's pairs,
    metric, worst case and bootstrap mode); ECAPA SIM and DNSMOS OVRL as analyze_mvc.ecapa_checks and dnsmos_checks."""
    es, ds = esets[s], dsets[s]
    res = {q: at.hypothesis(sets, f'X-{s}-{q}', s, q, X, MVC, 'wer_capped') for q in ('wer', 'sim', 'utmos')}
    res['ecapa_sim'] = stats.compare(es.rows(X), es.rows(MVC), es.subset('all'), es.spk, am.ECAPA_KEY, mode=es.mode,
                                     worst='min')
    res['dnsmos_ovrl'] = stats.compare(ds.rows(X), ds.rows(MVC), ds.subset('all'), ds.spk, 'OVRL', mode=ds.mode,
                                       worst='min')
    label = dict(LEVELS)
    return [dict(ci_only(res[q]), id=f'X-{s}-{q}', set=s, quantity=q, label=label[q], sys_a=X, sys_b=MVC)
            for q in DIFFS]


def nl(v):
    """§3.3 on VCTK-2k: NL per stratum of each system by analyze_mvc.nl_family, and Δ NL = the paired SIM_src
    difference mvc_r1x300 − mvc_v1 over the stratum's pairs with P (target_copy's term cancels pair by pair, so it is
    NL(mvc_r1x300) − NL(mvc_v1) when neither system failed there)."""
    fam = {x: am.nl_family(v, x=x, ids=(f'NL-{x}-same', f'NL-{x}-cross'), beside=False) for x in SYSTEMS}
    out = {}
    for g in stats.STRATA:
        pids, no_e = am.nl_pairs(v, g)
        d = stats.compare(v.rows(X + SRC), v.rows(MVC + SRC), pids, v.spk, 'sim_wavlm', mode=v.mode, worst='max')
        out[g] = {'n_pairs_P': len(pids), 'n_no_source_E': no_e, 'diff': ci_only(d),
                  **{x: dict(ci_only(fam[x]['strata'][g]), target_copy_failures=fam[x]['strata'][g]['fail_b'])
                     for x in SYSTEMS}}
    return out


def analyse(sets, esets, dsets, gsets):
    return {'sets': {s: {'levels': levels(sets[s], esets[s], dsets[s]), 'differences': differences(s, sets, esets, dsets),
                         'guards': guards(gsets[s])} for s in SETS},
            'nl': nl(sets['vctk2k'])}


# ── report ───────────────────────────────────────────────────

f, ci, digits, civ = at.f, at.ci, at.digits, am.civ


def render(res):
    fr = res['frozen']
    L = ['# Exploratory addition X-mvc-1: R1x at 300k on the test sets (mvc/ANALYSIS_PLAN_mvc_v1.md §4.10)', '',
         f'Written {res["written"]} by `mvc/analysis/analyze_xmvc1.py` (sha256 {res["script_sha256"]}; analyze_mvc.py '
         f'{res["analyze_mvc_sha256"]}; stats.py {fr["stats"]}; analyze_tests.py {fr["analyze_tests"]}; '
         f'analyze_exploratory.py {fr["analyze_exploratory"]}). Crossed bootstrap B = {stats.B}, seed {stats.SEED}: '
         f'VCTK-2k independent, LTC-2k linked.', ''] + [f'- {x}' for x in res['notes']] + ['']
    for s in SETS:
        r = res['sets'][s]
        L += [f'## {s}, 30 s', '', '### Levels (95 % CIs)', '',
              '| System | rows (failed) | ' + ' | '.join(lab for _, lab in LEVELS) + ' |', '|---|---|' + '---|' * len(LEVELS)]
        for n, row in r['levels'].items():
            L.append(f'| {n} | {row["n_rows"]} ({row["fail"]}) | '
                     + ' | '.join(ci(row.get(q), digits(q)) for q, _ in LEVELS) + ' |')
        L += ['', f'### {X} − {MVC}, paired (95 % CIs)', '',
              f'| Quantity | {X} | {MVC} | Δ [95 % CI] | SE | n pairs (failed {X}/{MVC}) | failed conversions at '
              f'their worst case: Δ [95 % CI] |', '|---|---|---|---|---|---|---|']
        for h in r['differences']:
            d, wc = digits(h['quantity']), h.get('worst_case')
            L.append(f'| {h["label"]} | {f(h.get("a"), d)} | {f(h.get("b"), d)} | {civ(h, d)} | {f(h.get("se"), 4)} | '
                     f'{h.get("n", 0)} ({h.get("fail_a", 0)}/{h.get("fail_b", 0)}) | {civ(wc, d) if wc else "–"} |')
        L += ['', '### Acoustic guards (95 % CIs)', '',
              '| System | rows (failed) | ' + ' | '.join(am.GUARD_FIELDS) + ' |', '|---|---|' + '---|' * len(am.GUARD_FIELDS)]
        for n, row in r['guards'].items():
            L.append(f'| {n} | {row["n_rows"]} ({row["fail"]}) | ' + ' | '.join(ci(row.get(k), 3) for k in am.GUARD_FIELDS)
                     + ' |')
        L.append('')
    L += ['## Natural-level leakage NL on VCTK-2k, 30 s (§3.3, computed by analyze_mvc.nl_family)', '',
          'NL_g = mean over the stratum\'s pairs with a parallel recording P of SIM_src(system) − SIM_src(target_copy), '
          f'paired per pair. Δ NL is the paired SIM_src difference {X} − {MVC} over the same pairs; target_copy\'s '
          f'term cancels pair by pair, so it equals NL({X}) − NL({MVC}) when neither system failed on the stratum\'s '
          'pairs.', '',
          f'| Stratum | pairs with P (no source E) | NL {X} [95 % CI] | NL {MVC} [95 % CI] | Δ NL [95 % CI] | n (failed '
          f'{X}/{MVC}) |', '|---|---|---|---|---|---|']
    for g, r in res['nl'].items():
        d = r['diff']
        L.append(f'| {g} | {r["n_pairs_P"]} ({r["n_no_source_E"]}) | {civ(r[X])} | {civ(r[MVC])} | {civ(d)} | '
                 f'{d.get("n", 0)} ({d.get("fail_a", 0)}/{d.get("fail_b", 0)}) |')
    L += ['', '## Inputs', '', f'- snapshot: `{res["snapshot"]}` sha256 {res["snapshot_sha256"]}',
          f'- normalisation statistics: `{res["stats_npz"]}` sha256 {res["stats_npz_sha256"]} (R1\'s, as frozen)',
          f'- changed against mvc_v1.json: {res["changed"]}']
    L += [f'- {k}: `{v}`' for k, v in res['done'].items()]
    for group, label in (('x_mvc_1', 'X-mvc-1'), ('t1', 'T1 and journal (inputs of TEST_RESULTS_mvc_v1)'),
                         ('other', 'selection record cfg and T1 results')):
        L += [f'- {label}: `{x["path"]}` sha256 {x["sha256"]}' + (f' ({x["rows"]} rows)' if 'rows' in x else '')
              for x in res['inputs'][group]]
    return '\n'.join(L) + '\n'


# ── main ─────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--xr', default=os.path.join(ROOT, 'paper', 'mimi_vc', 'results', 'xmvc1'))
    ap.add_argument('--mr', default=os.path.join(ROOT, 'paper', 'mimi_vc', 'results'))
    ap.add_argument('--v1', default=os.path.join(ROOT, 'paper', 'journal_extension', 'results', 'v1'))
    ap.add_argument('--t1', default=os.path.join(ROOT, 'mvc', 'TEST_RESULTS_mvc_v1.json'))
    ap.add_argument('--record_cfg', default=os.path.join(ROOT, 'mvc', 'MVC_SELECTION_RECORD_v1_cfg', 'mvc_v1.json'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'mvc', 'EXPLORATORY_XMVC1_v1.md'))
    a = ap.parse_args(argv)
    frozen, am_sha = check_code()
    for path in (a.out, os.path.splitext(a.out)[0] + '.json'):
        if os.path.exists(path):
            raise SystemExit(f'ERROR: {path} exists: an earlier output is never overwritten; give a new --out')
    for path in (a.t1, a.record_cfg):
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {path} missing')
    info, cin = check_cfg(a.xr, a.record_cfg)                  # cfg and snapshot.sha256 (X-mvc-1), the record's cfg
    other = cin[2:] + [{'path': os.path.abspath(a.t1), 'sha256': at.sha256(a.t1)}]
    done = check_done(a.xr, a.mr, a.v1, info['snapshot_sha256'])
    listed, lin = check_lists(a.xr)
    pp = {'vctk2k': os.path.join(a.v1, 'pairs', 'vctk2k.jsonl'), 'ltc2k': os.path.join(a.v1, 'pairs', 'ltc2k.jsonl'),
          'source_e': os.path.join(a.v1, 'pairs', 'vctk2k_source_E.json')}
    for path in pp.values():
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {path} missing')
    pairs = {s: at.read_pairs(pp[s]) for s in SETS}
    src_e = {k for k, v in json.load(open(pp['source_e'])).items() if (v or {}).get('paths')}
    sets, esets, dsets, gsets, xin, tin = load(a.xr, a.mr, a.v1, pairs, src_e)
    xin = cin[:2] + xin
    dn = os.path.join(os.path.abspath(a.xr), 'dnsmos') + os.sep    # DNSMOS rows are written on the Mac
    bad = [x['path'] for x in xin if not x['path'].startswith(dn) and listed.get(x['path']) != x['sha256']]
    if bad:
        raise SystemExit(f'ERROR: {bad} are not listed with their sha256 in {a.xr}/<set>.sha256: not the files the job '
                         f'wrote on ceai')
    xin += lin
    tin = [{'path': os.path.abspath(p), 'sha256': at.sha256(p)} for p in pp.values()] + tin
    check_t1_inputs(a.t1, tin)
    problems = required(sets, esets, dsets)
    if problems:
        raise SystemExit('ERROR: ' + '; '.join(problems))
    res = analyse(sets, esets, dsets, gsets)
    res.update(written=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
               script_sha256=at.sha256(os.path.abspath(__file__)), analyze_mvc_sha256=am_sha, frozen=frozen,
               notes=notes(info['snapshot'], info['snapshot_sha256'], info['stats_npz']), done=done,
               inputs={'x_mvc_1': xin, 't1': tin, 'other': other}, **info)
    md = render(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fo:
        fo.write(md)
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as fo:
        json.dump(res, fo, indent=1, default=float)
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
