#!/usr/bin/env python3
"""
The T1 statistics of mvc/ANALYSIS_PLAN_mvc_v1.md (§5, §6, §7), run once on the Mac after T1 (§8.1, §8.3). CPU only.
Built on benchmark/analyze_exploratory.py; benchmark/analyze_tests.py and benchmark/stats.py are imported unchanged
and their sha256 checked at start (crossed bootstrap B = 10,000, seed 20260928; VCTK-2k independent, LTC-2k linked).

Inputs (all required; a missing one is a refusal before any output):
  --v1     the journal results (V1): m/t1..t4 (job M), x1/ (job X1), x2/ (job X2's ECAPA rows), pairs/ (the two test
           lists and the VCTK-2k source-E file, sha256 as in TEST_RESULTS_v1.md). Every m.done, x1.done and x2.done
           names the journal selection sha256 9c7275f5...bc105.
  --mr     the MimiFlow-VC results (MR): m/t1 (VCTK-2k 30 s), m/t2 (LTC-2k 30 s), m/t3 (both sets, 3 and 180 s), each
           with m.done naming the sha256 of --record; x2/ecapa/<set>_30{,_src}.jsonl; dnsmos/<set>_30*.jsonl;
           guard/<set>_30*.jsonl. The m parts and the ECAPA rows hold only the T1 systems of §8.2 (t1, t2 and ECAPA:
           mvc_v1, mvc_v1_q32, mvc_copy_latent; t3: mvc_v1; each with its @src twin); any other system stops the run.
  --record the committed MimiFlow-VC selection record, MVC_SELECTION_RECORD_v1.json (select_mvc.py), status
           'complete', with its .md beside it, written by the same select_mvc.py run (the .md carries the .json's
           status, 'written' stamp and analyze_mvc sha256 values). The T1 m.done files name the sha256 of the record:
           of its .md (select_mvc.py's RECORD_SHA256) or of the .json. It must give the sha256 of this script and of
           mvc/tests/test_analyze_mvc.py (§4.8), and both must match the files run. mvc_v1's Dev-900 numbers
           ("selected on Dev", §4.9, §9) are read from stages/<final_round>/r2_3/d900/numbers/<mvc_v1 config>; a record
           without them is refused.
Families (§5.1): H1 mvc_v1 − cand_pm18_rq (Holm over 6); NL the paired natural-level leakage on VCTK-2k (CI rule, +0.03,
both strata needed); L mvc_v1 − knnvc on SIM_src through stats.gate_l_strata (+0.01, all four needed); S mvc_v1 − each
baseline (Holm over 24); NI against seedvc and knnvc (one-sided p of §5.2, Holm over 12 at 0.025); RB 3 s and 180 s
against 30 s (Holm over 12) and the RBN checks; Q mvc_v1_q32 − mvc_v1 (Holm over 6). LTC-2k at 180 s: the 101 pairs
with a reference shortfall leave every 180 s pair list before any comparison. Then the power of §5.5, the
sensitivities of §5.6, the descriptive analyses of §5.7, the ECAPA and DNSMOS checks and flags of §6, the outcome of
§7.1 and the wording of §7.2-7.3. Writes --out (.md) and <out stem>.json with every number and the sha256 of every
input.

    /Users/harry/duplex/.venv-stats/bin/python mvc/analysis/analyze_mvc.py
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BENCH = os.path.join(ROOT, 'benchmark')
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import analyze_exploratory as ax  # noqa: E402
import analyze_tests as at  # noqa: E402
import stats  # noqa: E402

FROZEN = {'stats': 'a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6',
          'analyze_tests': '03f2bbf76f79338a46c0ad191b59979c6358dd1f6c7aaee908508a927cf2a933',
          'analyze_exploratory': 'ac53e40bac8ca55bf6327f3dbea3bf09268994c0d46465931b585de49a10ced3'}
JOURNAL_SELECTION_SHA256 = '9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105'
PAIR_SHA256 = {'vctk2k': 'bdbf36a32b61eff625e2acaf40163835582ca9e391079c11ca72e6bf51601c1d',
               'ltc2k': '32fc920516f0b6e4d239ddc85994cca8f746538763e5ad71ab2e208203c56e45',
               'source_e': 'ccbd53dfc507ad86b814a5e2cac2f8538f2bc7ca58a5d41f0f3c86e9627cb8ea'}
EXPECTED = {'shortfall': 101, 'par': 980}           # §1: LTC-2k 180 s shortfall pairs; VCTK-2k pairs with P
TEST_FILE = os.path.join(ROOT, 'mvc', 'tests', 'test_analyze_mvc.py')

SRC, SETS = at.SRC, at.SETS
MVC, Q32, COPY = 'mvc_v1', 'mvc_v1_q32', 'mvc_copy_latent'
MVC_SYSTEMS = (MVC, Q32, COPY)
T3_SYSTEMS = (MVC,)                                 # §8.2: part t3 is mvc_v1 at 3 and 180 s
CRT = 'cand_pm18_rq'
BASELINES = ('knnvc', 'freevc', 'seedvc', 'facodec')
NI_BASELINES = ('seedvc', 'knnvc')
FAMILY_SYSTEMS = (MVC, Q32, CRT) + BASELINES
PLAN_SYSTEMS = MVC_SYSTEMS + (CRT,) + BASELINES + ('faiss', 'source_copy', 'target_copy', 'mimi_copy_src',
                                                   'mimi_copy_P')
# analyze_tests.levels() orders systems with system_order(), which reads the 's2' key (§5.8)
SYSTEMS = {'crt_al': CRT, 'choice': 'C2', 'faiss': 'faiss', 's3': 'crt_tau0.4', 's2': 'crt_tau0.9', 'tau_dev': 0.9,
           'baselines': list(BASELINES)}
ECAPA_KEY = 'sim_ecapa'
ECAPA_SYSTEMS = MVC_SYSTEMS + (CRT,) + BASELINES + ('faiss', 'source_copy', 'target_copy')
DNSMOS_SYSTEMS = MVC_SYSTEMS + (CRT,) + BASELINES + ('mimi_copy_src', 'source_copy')
DNSMOS_FIELDS = ('SIG', 'BAK', 'OVRL', 'P808')
GUARD_SYSTEMS = (MVC, Q32, CRT) + BASELINES
GUARD_FIELDS = ('hnr', 'hnr_src', 'voiced_kept', 'f0_pcc', 'st_sd_ratio', 'rms_db_diff')
ENDPOINTS = at.S_ENDPOINTS
NL_MARGIN = 0.03
NI_MARGIN = {'wer': 1.0, 'sim': 0.010, 'utmos': 0.10}
NI_ALPHA = 0.025
# id, budget, quantity, CI limit, bound: RBN-1/2 need the lower limit above the bound, RBN-3 the upper limit below it
RBN = (('RBN-1', 3, 'sim', 'lo', -0.06), ('RBN-2', 180, 'sim', 'lo', -0.005), ('RBN-3', 3, 'wer', 'hi', 1.5))
# family: (alpha, m, two-sided) of the §5.5 multiplier z_{1-alpha/(2m)} + z_0.80 (two-sided) or z_{1-alpha/m} + z_0.80
POWER = {'H1': (0.05, 6, True), 'Q': (0.05, 6, True), 'S': (0.05, 24, True), 'NI': (0.025, 12, False),
         'RB': (0.05, 12, True), 'NL': (0.025, 1, False), 'L': (0.025, 1, False)}
OUTCOMES = {'A': 'mvc_v1 improves on CRT-AL in intelligibility, speaker similarity and naturalness on both test sets',
            'A−': 'as A, with each non-rejected endpoint reported as "not significant" with its CI; no sentence claims '
                  'improvement on every endpoint',
            'B': 'trade-off: the abstract names the endpoint and set on which CRT-AL is better',
            'C': 'improvement is claimed only on the rejected endpoints; the others read "no improvement shown"'}
SHOULD = {'vctk2k': {'wer': 5.0, 'sim': 0.89, 'utmos': 3.6}, 'ltc2k': {'wer': 3.5, 'sim': 0.93, 'utmos': 3.9}}
STRETCH = {'vctk2k': {'wer': 3.0, 'sim': 0.91}, 'ltc2k': {'wer': 2.5, 'sim': 0.945}}
NOT_CONFIRMED_SPK = 'not confirmed by a second speaker model'
NOT_CONFIRMED_NAT = 'not confirmed by a second naturalness predictor'
HEX64 = re.compile(r'[0-9a-f]{64}')
NOTES = ['FreeVC was trained on VCTK ("VCTK-seen"): its VCTK-2k rows stay in family S, but no claim of beating or '
         'matching FreeVC on VCTK-2k is made from them (§5.1, §7.3).',
         'Seed-VC uses only the first 25 s of the reference at 30 s and 180 s.',
         'MimiFlow-VC decodes in fp32; the journal\'s Mimi copies decode 32 codes in fp16. Hardware: H200 MIG 2g.35gb '
         'on ceai.',
         'LTC-2k at 180 s: the 101 pairs with a reference shortfall (targets 3729, 6930) are removed from every 180 s '
         'pair list before any comparison, for every system (§1, §5.4).',
         'The decisions rest on one speaker model (WavLM-SV), with ECAPA as the second; the NL margin of +0.03 was set '
         'on Dev; LTC-2k has no natural-level check.',
         'Wilcoxon signed-rank p and the rank-biserial r are possibly anti-conservative, since pairs share speakers '
         '(§5.2).']


# ── frozen code, record, inputs ──────────────────────────────

def check_frozen():
    """§5.8: analyze_tests.py and stats.py (and analyze_exploratory.py, FREEZE_mvc_v1.sha256) unchanged."""
    got = {'stats': at.sha256(stats.__file__), 'analyze_tests': at.sha256(at.__file__),
           'analyze_exploratory': at.sha256(ax.__file__)}
    bad = {k: v for k, v in got.items() if v != FROZEN[k]}
    if bad:
        raise SystemExit(f'ERROR: frozen journal code changed (sha256 {bad}; expected {FROZEN})')
    for m in (stats, at, ax):
        if os.path.dirname(os.path.abspath(m.__file__)) != BENCH:
            raise SystemExit(f'ERROR: {m.__name__} was imported from {m.__file__}, not from {BENCH}')
    return got


def check_sha(path, want, what):
    got = at.sha256(path)
    if got != want:
        raise SystemExit(f'ERROR: {path} ({what}) has sha256 {got}, expected {want}')
    return got


def record_shas(obj, filename):
    """Every sha256 the record gives for a file whose base name is filename: a key naming the file with a 64-hex value,
    or an entry {'path' or 'file': ..., 'sha256': ...}."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and os.path.basename(k) == filename and isinstance(v, str) and HEX64.fullmatch(v):
                out.append(v)
            if isinstance(v, (dict, list)):
                out += record_shas(v, filename)
        p = obj.get('path') or obj.get('file')
        if (isinstance(p, str) and os.path.basename(p) == filename and isinstance(obj.get('sha256'), str)
                and HEX64.fullmatch(obj['sha256'])):
            out.append(obj['sha256'])
    elif isinstance(obj, list):
        for v in obj:
            out += record_shas(v, filename)
    return out


def check_record(rec, script, test_file):
    """§4.8: the record holds the sha256 of analyze_mvc.py and of its tests; the files run must be those."""
    got = {}
    for path in (script, test_file):
        name = os.path.basename(path)
        want = record_shas(rec, name)
        if not want:
            raise SystemExit(f'ERROR: the selection record gives no sha256 for {name} (§4.8)')
        got[name] = at.sha256(path)
        if any(w != got[name] for w in want):
            raise SystemExit(f'ERROR: {path} has sha256 {got[name]}; the selection record names {sorted(set(want))}')
    return got


def check_record_md(rec, md):
    """§4.8: the record's .md and .json come from the same select_mvc.py run: the .md carries the .json's status line
    with its 'written' stamp (select_mvc.render) and every analyze_mvc sha256 the .json gives."""
    head = f'Status: **{rec.get("status")}**. Written {rec.get("written")} '
    if not rec.get('written') or head not in md:
        raise SystemExit(f'ERROR: the selection record .md does not carry the .json\'s line {head.strip()!r}: the two '
                         f'files are not from the same select_mvc.py run')
    for name in (os.path.basename(__file__), os.path.basename(TEST_FILE)):
        missing = sorted(w for w in set(record_shas(rec, name)) if w not in md)
        if missing:
            raise SystemExit(f'ERROR: the selection record .md does not name the sha256 {missing} of {name} that the '
                             f'.json gives')


def dev900(rec):
    """mvc_v1's Dev-900 numbers from the record (select_mvc.py's R2.3 Dev-900 table of the chosen configuration):
    capped WER, SIM, UTMOS and NL per stratum. A record without them is refused (§4.9, §9)."""
    try:
        x = rec['stages'][f'{rec["final_round"]}/r2_3']['d900']['numbers'][rec['mvc_v1']['config']]
        out = {'wer': x['wer'], 'sim': x['sim'], 'utmos': x['utmos'],
               'nl': {g: x['nl'][g]['est'] for g in stats.STRATA}}
    except (KeyError, TypeError) as e:
        raise SystemExit(f'ERROR: the selection record gives no Dev-900 numbers of mvc_v1 (stages/<final_round>/r2_3/'
                         f'd900/numbers/<mvc_v1 config>: {e!r}); they are shown next to the test numbers (§4.9, §9)')
    bad = [k for k, v in list(out.items())[:3] + [(f'nl {g}', v) for g, v in out['nl'].items()]
           if not isinstance(v, (int, float)) or not np.isfinite(v)]
    if bad:
        raise SystemExit(f'ERROR: the selection record\'s Dev-900 numbers of mvc_v1 lack {bad}')
    return out


def done_line(path, sha, what):
    """A done file of the form 'job <id> <date> selection sha256 <sha>' naming the expected sha256."""
    if not os.path.exists(path):
        raise SystemExit(f'ERROR: {path} missing: {what} has not finished')
    line = open(path).read().strip()
    m = re.search(r'selection sha256 ([0-9a-f]{64})', line)
    if not m or m.group(1) != sha:
        raise SystemExit(f'ERROR: {path} ({line!r}) does not name the selection sha256 {sha}')
    return line


def check_parts(m_dir, parts, sha, what):
    """analyze_tests.check_done on the parts, all under one selection sha256, which must be `sha` (or one of them,
    when a tuple)."""
    lines = at.check_done(m_dir, parts)
    got = {re.search(r'selection sha256 ([0-9a-f]{64})', v).group(1) for v in lines.values()}
    ok = sha if isinstance(sha, tuple) else (sha,)
    if len(got) != 1 or not got <= set(ok):
        raise SystemExit(f'ERROR: {what} parts {parts} name the selection sha256 {sorted(got)}, expected '
                         f'{" or ".join(ok)}')
    return lines


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def shortfall_pairs(pairs):
    """The pairs whose 180 s reference has a shortfall (make_pairs.py: R['180']['shortfall'])."""
    return {p['pair_id'] for p in pairs if ((p.get('R') or {}).get('180') or {}).get('shortfall')}


def make_set(name, pairs, R, src_e=None, keep=PLAN_SYSTEMS):
    """analyze_tests.TestSet with the rows of the systems in keep (and their @src rows); LTC-2k carries its 180 s
    shortfall pairs."""
    if keep is not None:
        R = {b: {s: v for s, v in systems.items() if s.split('@')[0] in keep} for b, systems in R.items()}
    ts = at.TestSet(name, pairs, R, src_e if name == 'vctk2k' else None)
    ts.shortfall = shortfall_pairs(pairs) if name == 'ltc2k' else set()
    return ts


def merge(parts, whole=False):
    """{budget: {system: {pair_id: row}}} from several sources [(label, R)]; stops on any (pair_id, system, budget)
    found twice (§5.8), and with whole=True on any (system, budget) that comes from two sources, as
    analyze_exploratory.load does."""
    out, came = {}, {}
    for label, R in parts:
        for b, systems in R.items():
            for s, rows in systems.items():
                if whole and (s, b) in came:
                    raise SystemExit(f'ERROR: {s} at {b} s comes from both {came[(s, b)]} and {label}')
                came[(s, b)] = label
                bucket = out.setdefault(b, {}).setdefault(s, {})
                for pid, r in rows.items():
                    if pid in bucket:
                        raise SystemExit(f'ERROR: ({pid}, {s}, {b} s) found twice ({label})')
                    bucket[pid] = r
    return out


def check_systems(R, allowed, what):
    """§8.2, §9: MimiFlow-VC test rows hold only the T1 systems of their part and their @src twins; any other system
    (a configuration R2 did not choose, a journal system) stops the run. Returns R."""
    ok = {x for s in allowed for x in (s, s + SRC)}
    for b, systems in R.items():
        bad = sorted(set(systems) - ok)
        if bad:
            raise SystemExit(f'ERROR: {what} at {b} s holds rows of {bad}; its T1 systems are {sorted(ok)} (§8.2, §9)')
    return R


def load_rows(path, name, pairs, budget=None):
    """{system: {pair_id: row}} of one file; a pair outside the list, another set or budget, or a duplicate stops."""
    ids, R = {p['pair_id'] for p in pairs}, {}
    rows = read_jsonl(path)
    for r in rows:
        if r.get('pair_id') not in ids:
            raise SystemExit(f'ERROR: {path}: pair {r.get("pair_id")} is not in the {name} list')
        if r.get('set') not in (None, name) or (budget is not None and int(r.get('budget', budget)) != budget):
            raise SystemExit(f'ERROR: {path}: row {r["pair_id"]} {r.get("system")} has set {r.get("set")}, budget '
                             f'{r.get("budget")}')
        bucket = R.setdefault(r.get('system'), {})
        if r['pair_id'] in bucket:
            raise SystemExit(f'ERROR: {path}: ({r["pair_id"]}, {r.get("system")}) twice')
        bucket[r['pair_id']] = r
    return R, {'path': os.path.abspath(path), 'sha256': at.sha256(path), 'rows': len(rows)}


def load_ecapa(dirs, name, pairs):
    """ECAPA rows at 30 s of <dir>/<name>_30{,_src}.jsonl for each (label, dir, T1 systems or None), merged with the
    duplicate checks; a directory with T1 systems holds no other system."""
    parts, inputs = [], []
    for label, d, allowed in dirs:
        for suffix in ('', '_src'):
            path = os.path.join(d, f'{name}_30{suffix}.jsonl')
            if not os.path.exists(path):
                raise SystemExit(f'ERROR: {path} missing')
            R, inp = load_rows(path, name, pairs, 30)
            if allowed is not None:
                check_systems({30: R}, allowed, f'{label} ECAPA {path}')
            parts.append((f'{label} {os.path.basename(path)}', {30: R}))
            inputs.append(inp)
    return merge(parts, whole=True), inputs


def load_glob(d, name, pairs, what):
    """Rows of every <d>/<name>_30*.jsonl (DNSMOS or guard output), merged with the duplicate check."""
    files = sorted(glob.glob(os.path.join(d, f'{name}_30*.jsonl')))
    if not files:
        raise SystemExit(f'ERROR: no {what} rows {os.path.join(d, name)}_30*.jsonl')
    parts, inputs = [], []
    for path in files:
        R, inp = load_rows(path, name, pairs)
        parts.append((f'{what} {os.path.basename(path)}', {30: R}))
        inputs.append(inp)
    return merge(parts), inputs


# ── row checks ───────────────────────────────────────────────

def src_problems(label, tgt, src, with_e):
    """The @src rows as lab_rows.py and score_ecapa.py write them: one on every pair whose target row has no error and
    whose source has an E set (with_e), none on a pair whose source has none. A failed conversion has an error target
    row and no @src row, and counts as a failure (§5.4); an @src row beside a target row whose scoring failed is
    kept."""
    need = {k for k in with_e if not stats.failed(tgt.get(k))}
    missing, extra = need - set(src), set(src) - with_e
    out = []
    if missing:
        out.append(f'{label}: no row on {len(missing)} of the {len(need)} pairs whose target row has no error')
    if extra:
        out.append(f'{label}: rows on {len(extra)} pairs whose source has no E set')
    return out


def required(sets, esets, dsets):
    """§5.8 required(): one row per pair at 30 s for every family system and mvc_copy_latent, and @src rows on every
    pair whose target row has no error (src_problems); mvc_v1 at 3 s and 180 s (LTC-2k at 180 s: shortfall rows
    exactly on the list's shortfall pairs); target_copy@src without error on every VCTK-2k pair with P and on no other;
    the ECAPA rows (target rows of mvc_v1, mvc_v1_q32, mvc_copy_latent, CRT-AL and the baselines on every pair, their
    @src rows as src_problems; VCTK-2k source_copy on every pair and target_copy, target_copy@src on the pairs with P);
    the DNSMOS rows of every §3.7 system. Returns the problems."""
    out = []
    for s in SETS:
        ts, n = sets[s], len(sets[s].pairs)
        with_e = set(src_pids(ts))
        for sy in FAMILY_SYSTEMS + (COPY,):
            k = len(ts.rows(sy, 30))
            if k != n:
                out.append(f'{s} {sy} 30 s: {k} rows for {n} pairs')
            out += src_problems(f'{s} {sy}{SRC} 30 s', ts.rows(sy, 30), ts.rows(sy + SRC, 30), with_e)
        for b in (3, 180):
            rows = ts.rows(MVC, b)
            if len(rows) != n:
                out.append(f'{s} {MVC} {b} s: {len(rows)} rows for {n} pairs')
            short = ts.shortfall if b == 180 else set()
            flagged = {k for k, r in rows.items() if r.get('shortfall')}
            if flagged != short:
                out.append(f'{s} {MVC} {b} s: shortfall rows on {len(flagged)} pairs; the list has {len(short)} '
                           f'shortfall pairs')
        es, ds = esets[s], dsets[s]
        for sy in MVC_SYSTEMS + (CRT,) + BASELINES:
            k = len(es.rows(sy))
            if k != n:
                out.append(f'{s} ECAPA {sy}: {k} rows for {n} pairs')
            out += src_problems(f'{s} ECAPA {sy}{SRC}', es.rows(sy), es.rows(sy + SRC), with_e)
        for sy in DNSMOS_SYSTEMS:
            k = len(ds.rows(sy))
            if k != n:
                out.append(f'{s} DNSMOS {sy}: {k} rows for {n} pairs')
    v, ev = sets['vctk2k'], esets['vctk2k']
    par = set(v.subset('par'))
    for label, rows, key in (('target_copy@src', v.rows('target_copy' + SRC), 'sim_wavlm'),
                             ('ECAPA target_copy', ev.rows('target_copy'), ECAPA_KEY),
                             ('ECAPA target_copy@src', ev.rows('target_copy' + SRC), ECAPA_KEY)):
        bad = sorted(k for k in par if stats.failed(rows.get(k)) or stats._value(rows[k], key) is None)
        if set(rows) != par or bad:
            out.append(f'vctk2k {label}: rows on {len(set(rows) & par)} of the {len(par)} pairs with P, '
                       f'{len(set(rows) - par)} elsewhere, {len(bad)} failed or undefined')
    if len(ev.rows('source_copy')) != len(v.pairs):
        out.append(f'vctk2k ECAPA source_copy: {len(ev.rows("source_copy"))} rows for {len(v.pairs)} pairs')
    return out


# ── comparisons ──────────────────────────────────────────────

def endpoint_pids(ts, q, budgets):
    """The endpoint's pairs; at 180 s on LTC-2k without the shortfall pairs, for both budgets of a contrast."""
    pids = ts.subset(at.Q[q][1])
    if 180 in budgets and ts.shortfall:
        pids = [k for k in pids if k not in ts.shortfall]
    return pids


def hyp(sets, hid, s, q, a, b, wer='wer_capped', budgets=(30, 30), mode=None):
    """Δ = A(budgets[0]) − B(budgets[1]) on one endpoint: analyze_tests.hypothesis at 30 s, otherwise the same
    stats.compare call with the budgets (as analyze_exploratory.comparison)."""
    if tuple(budgets) == (30, 30):
        h = at.hypothesis(sets, hid, s, q, a, b, wer, mode=mode)
    else:
        ts = sets[s]
        metric = wer if q == 'wer' else at.Q[q][0]
        res = stats.compare(ts.quantity_rows(a, q, budgets[0]), ts.quantity_rows(b, q, budgets[1]),
                            endpoint_pids(ts, q, budgets), ts.spk, metric, mode=mode or ts.mode, worst=at.Q[q][3],
                            n_ref=ts.n_ref)
        label = at.LABEL[{'wer_capped': 'wer', 'wer': 'wer_u'}.get(metric, q)]
        h = {'id': hid, 'set': s, 'quantity': q, 'label': label, 'sys_a': a, 'sys_b': b, **res}
    h.update(budget_a=budgets[0], budget_b=budgets[1])
    return h


def holm_family(name, hyps, alpha=0.05):
    flags = stats.holm([h.get('p') for h in hyps], alpha)
    at.set_flags(hyps, flags)
    return {'family': name, 'hypotheses': hyps, 'm': len(hyps), 'alpha': alpha, 'holm': f'over {len(hyps)}'}


def _reps(ok, ca, cb, spk, metric, mode):
    """The replicates stats.compare forms (stats._diff): same design, statistic, seed and redraws."""
    if not ok:
        return None, None
    design = stats.Design([spk[p][0] for p in ok], [spk[p][1] for p in ok], mode)
    if metric in stats.COUNTS:
        est, reps, _ = stats.bootstrap(design, np.column_stack([ca, cb]), stats._ratio_diff, den=(1, 3))
    else:
        est, reps, _ = stats.bootstrap(design, ca[:, 0] - cb[:, 0], stats._mean)
    return est, reps


def replicates(rows_a, rows_b, pids, spk, metric, mode, wer_key=stats.WER_KEY):
    """(estimate, B replicates of Δ*) of stats.compare(rows_a, rows_b, pids, ...)."""
    ok, (ca, cb), _, _ = stats.collect([rows_a, rows_b], pids, metric, wer_key)
    return _reps(ok, ca, cb, spk, metric, mode)


def worst_case_columns(rows_a, rows_b, pids, metric, worst, n_ref, wer_key=stats.WER_KEY):
    """The pairs and columns of stats._worst_case (every failed conversion at its worst case), or None."""
    if metric not in stats.COUNTS and worst not in ('min', 'max'):
        return None

    def worst_of(rows):
        v = [x[0] for x in (stats._value(r, metric, wer_key) for r in rows.values() if not stats.failed(r))
             if x is not None]
        return (min(v) if worst == 'min' else max(v)) if v else None
    wv = None if metric in stats.COUNTS else (worst_of(rows_a), worst_of(rows_b))
    ok, ca, cb = [], [], []
    for p in pids:
        rs = (rows_a.get(p), rows_b.get(p))
        vs = [None if stats.failed(r) else stats._value(r, metric, wer_key) for r in rs]
        if any(not stats.failed(r) and v is None for r, v in zip(rs, vs)):
            continue
        if metric in stats.COUNTS:
            n = next((v[1] for v in vs if v is not None), n_ref.get(p))
            fills = [None if n is None else (n, n)] * 2
        else:
            fills = [None if x is None else (x,) for x in wv]
        fail = [stats.failed(r) for r in rs]
        if any(f and x is None for f, x in zip(fail, fills)):
            continue
        vs = [fills[i] if fail[i] else vs[i] for i in range(2)]
        ok.append(p)
        ca.append(vs[0])
        cb.append(vs[1])
    if not ok:
        return ok, None, None
    return ok, np.array(ca, float).reshape(len(ok), -1), np.array(cb, float).reshape(len(ok), -1)


def ni_p(reps, sign, delta):
    """§5.2: p_NI = (1 + #{s_q Δ* <= −δ_q}) / (B + 1)."""
    if reps is None:
        return None
    return (1 + int(np.sum(sign * np.asarray(reps) <= -delta))) / (len(reps) + 1)


def ci_rule(res, sign, delta):
    """The single-test CI rule: the 2.5th percentile above −δ (SIM, UTMOS), the 97.5th below +δ (WER)."""
    lim = res.get('lo') if sign > 0 else res.get('hi')
    if lim is None:
        return None
    return bool(lim > -delta) if sign > 0 else bool(lim < delta)


def ni_hyp(sets, hid, s, q, a, b, wer='wer_capped', mode=None):
    h = hyp(sets, hid, s, q, a, b, wer, mode=mode)
    ts = sets[s]
    metric = wer if q == 'wer' else at.Q[q][0]
    sign, delta = at.Q[q][2], NI_MARGIN[q]
    ra, rb, pids = ts.quantity_rows(a, q), ts.quantity_rows(b, q), ts.subset(at.Q[q][1])
    _, reps = replicates(ra, rb, pids, ts.spk, metric, mode or ts.mode)
    h.update(margin=delta, sign=sign, bound=('Δ < +' if sign < 0 else 'Δ > −') + f'{delta:g}', p_ni=ni_p(reps, sign, delta),
             ci_rule=ci_rule(h, sign, delta))
    if h.get('worst_case'):
        w = worst_case_columns(ra, rb, pids, metric, at.Q[q][3], ts.n_ref)
        if w is not None and w[0]:
            _, rw = _reps(*w, ts.spk, metric, mode or ts.mode)
            h['worst_case'].update(p_ni=ni_p(rw, sign, delta), ci_rule=ci_rule(h['worst_case'], sign, delta))
    return h


def ni_family(hyps, key='p_ni'):
    flags = stats.holm([h.get(key) for h in hyps], NI_ALPHA)
    for h, f in zip(hyps, flags):
        h['reject'], h['favours'], h['non_inferior'] = f, None, f
    return {'family': 'NI', 'hypotheses': hyps, 'm': len(hyps), 'alpha': NI_ALPHA,
            'holm': f'over {len(hyps)} at α {NI_ALPHA}'}


def build_families(sets, wer='wer_capped', mode_ltc=None):
    """H1, S, NI, RB and Q (§5.1) with one WER definition and one LTC-2k bootstrap mode."""
    def md(s):
        return mode_ltc if s == 'ltc2k' else None
    H1 = holm_family('H1', [hyp(sets, f'H1-{i}', s, q, MVC, CRT, wer, mode=md(s))
                            for i, (s, q) in enumerate(ENDPOINTS, 1)])
    S = []
    for bl in BASELINES:
        for i, (s, q) in enumerate(ENDPOINTS, 1):
            h = hyp(sets, f'S-{bl}-{i}', s, q, MVC, bl, wer, mode=md(s))
            if bl == 'freevc' and s == 'vctk2k':
                h['flag'] = 'VCTK-seen'
            S.append(h)
    S = holm_family('S', S)
    NI = ni_family([ni_hyp(sets, f'NI-{bl}-{i}', s, q, MVC, bl, wer, mode=md(s))
                    for bl in NI_BASELINES for i, (s, q) in enumerate(ENDPOINTS, 1)])
    RB = []
    for b in (3, 180):
        for i, (s, q) in enumerate(ENDPOINTS, 1):
            h = hyp(sets, f'RB-{b}-{i}', s, q, MVC, MVC, wer, budgets=(b, 30), mode=md(s))
            h.update(sys_a=f'{MVC} {b} s', sys_b=f'{MVC} 30 s')
            RB.append(h)
    RB = holm_family('RB', RB)
    Q = holm_family('Q', [hyp(sets, f'Q-{i}', s, q, Q32, MVC, wer, mode=md(s))
                          for i, (s, q) in enumerate(ENDPOINTS, 1)])
    return {'H1': H1, 'S': S, 'NI': NI, 'RB': RB, 'Q': Q}


def rbn(RB):
    """RBN-1..3 per set from the RB contrasts (CI rule, no Holm)."""
    by = {h['id']: h for h in RB['hypotheses']}
    out = []
    for s in SETS:
        for rid, b, q, lim, bound in RBN:
            i = ENDPOINTS.index((s, q)) + 1
            h = by[f'RB-{b}-{i}']
            v = h.get(lim)
            out.append({'id': rid, 'set': s, 'from': h['id'], 'quantity': q, 'limit': lim, 'bound': bound,
                        'est': h.get('est'), 'lo': h.get('lo'), 'hi': h.get('hi'), 'n': h.get('n'),
                        'passes': None if v is None else bool(v > bound if lim == 'lo' else v < bound)})
    return out


# ── leakage: NL and L ────────────────────────────────────────

def nl_pairs(ts, g):
    """The stratum's VCTK-2k pairs with P whose source speaker has a source-E set, and the number left out."""
    sub = [p for p in ts.pairs if stats.stratum(p['genders']) == g and p.get('P')]
    keep = [p['pair_id'] for p in sub if ts.src_e is None or p['source_speaker'] in ts.src_e]
    return keep, len(sub) - len(keep)


def unpaired_nl(ts, x, g, key='sim_wavlm'):
    """summarize.py's value: X's mean SIM_src over all the stratum's pairs minus target_copy's over its pairs with P
    (both over the pairs where X's row has no error, and target_copy's term where its target row has no error, as
    summarize.py), with a crossed-bootstrap CI from the same weighted sums (descriptive; it decides nothing)."""
    sub, no_e = stats.endpoint_pairs(ts.pairs, 'gate_' + g, ts.src_e)
    xt, xr, tt, tr = ts.rows(x), ts.rows(x + SRC), ts.rows('target_copy'), ts.rows('target_copy' + SRC)
    use, cols = [], []
    for p in sub:
        k = p['pair_id']
        if stats.failed(xt.get(k)):
            continue            # summarize.py: the pairs where X's row has no error, for both terms
        vx = None if stats.failed(xr.get(k)) else stats._value(xr[k], key)
        vt = (None if (not p.get('P') or stats.failed(tt.get(k)) or stats.failed(tr.get(k)))
              else stats._value(tr[k], key))
        if vx is None and vt is None:
            continue
        use.append(k)
        cols.append([vx[0] if vx else 0.0, 1.0 if vx else 0.0, vt[0] if vt else 0.0, 1.0 if vt else 0.0])
    out = {'n_x': int(sum(c[1] for c in cols)), 'n_target_copy': int(sum(c[3] for c in cols)), 'n_no_source_E': no_e,
           'est': None}
    if not out['n_x'] or not out['n_target_copy']:
        return out
    design = stats.Design([ts.spk[k][0] for k in use], [ts.spk[k][1] for k in use], ts.mode)
    est, reps, red = stats.bootstrap(design, np.array(cols), lambda S: S[:, 0] / S[:, 1] - S[:, 2] / S[:, 3],
                                     den=(1, 3))
    c = np.array(cols)
    return dict(out, **stats.summary(est, reps, red, design, diff=False), x_mean=float(c[:, 0].sum() / c[:, 1].sum()),
                target_copy_mean=float(c[:, 2].sum() / c[:, 3].sum()))


def nl_family(ts, x=MVC, key='sim_wavlm', ids=('NL1s', 'NL1c'), beside=True):
    """§3.3: NL_g(X) = stats.compare(X@src, target_copy@src, the stratum's pairs with P, 'sim_wavlm', worst 'max'),
    decided by stats.gate_l(d, margin=0.03). Never through stats.gate_l_strata."""
    out = {}
    for hid, g in zip(ids, stats.STRATA):
        pids, no_e = nl_pairs(ts, g)
        d = stats.compare(ts.rows(x + SRC), ts.rows('target_copy' + SRC), pids, ts.spk, key, mode=ts.mode,
                          worst='max')
        r = dict(d, id=hid, stratum=g, system=x, rejects=stats.gate_l(d, margin=NL_MARGIN), n_no_source_E=no_e,
                 target_copy_failures=d['fail_b'])
        if r.get('worst_case'):
            r['worst_case']['rejects'] = stats.gate_l(r['worst_case'], margin=NL_MARGIN)
        if beside:
            r['summarize_value'] = unpaired_nl(ts, x, g, key)
            r['sim_x_same_pairs'] = stats.level(ts.rows(x), pids, ts.spk, 'sim_wavlm', mode=ts.mode)
            r['sim_target_copy_same_pairs'] = stats.level(ts.rows('target_copy'), pids, ts.spk, 'sim_wavlm',
                                                          mode=ts.mode)
        out[g] = r
    return {'strata': out, 'all_reject': all(r['rejects'] is True for r in out.values()),
            'worst_case_all_reject': all((r['worst_case'] if r.get('worst_case') else r)['rejects'] is True
                                         for r in out.values())}


def family_l(sets, mode_ltc=None):
    """§3.4, §5.1: Δ_src,g = SIM_src(mvc_v1) − SIM_src(knnvc) through stats.gate_l_strata (margin stats.MARGIN_L)."""
    out = {}
    ids = {('vctk2k', 'same'): 'L1s', ('vctk2k', 'cross'): 'L1c', ('ltc2k', 'same'): 'L2s', ('ltc2k', 'cross'): 'L2c'}
    for s in SETS:
        ts = sets[s]
        mode = mode_ltc if (s == 'ltc2k' and mode_ltc) else ts.mode
        res = stats.gate_l_strata(ts.rows(MVC + SRC), ts.rows('knnvc' + SRC), ts.pairs, mode=mode,
                                  src_e=ts.src_e if s == 'vctk2k' else None)
        for g, r in res.items():
            if r.get('worst_case'):
                r['worst_case']['rejects'] = stats.gate_l(r['worst_case'])
            out[ids[(s, g)]] = dict(r, id=ids[(s, g)], set=s, stratum=g)
    wc = {k: (r['worst_case']['rejects'] if r.get('worst_case') else r['rejects']) for k, r in out.items()}
    return {'strata': out, 'all_reject': all(r['rejects'] is True for r in out.values()),
            'worst_case': {'used': [k for k, r in out.items() if r.get('worst_case')], 'rejects': wc,
                           'all_reject': all(v is True for v in wc.values())}}


# ── second models ────────────────────────────────────────────

def ecapa_m9(tgt, src):
    out = {}
    for p in set(tgt) | set(src):
        t, s = tgt.get(p), src.get(p)
        vt = None if stats.failed(t) else stats._value(t, ECAPA_KEY)
        vs = None if stats.failed(s) else stats._value(s, ECAPA_KEY)
        out[p] = {'m9': vs[0] - vt[0]} if vt and vs else {'error': 'no SIM or SIM_src'}
    return out


def coverage_stat(S):
    return (S[:, 0] - S[:, 1]) / (S[:, 2] - S[:, 1])


def coverage(ets, x, key=ECAPA_KEY):
    """§3.6: (E_out − E_src) / (E_tgt − E_src) over the VCTK-2k pairs with P, from the weighted sums of each replicate
    (independent crossed bootstrap)."""
    par = ets.subset('par')
    ok, cols, fails, undefined = stats.collect([ets.rows(x), ets.rows('source_copy'), ets.rows('target_copy')], par,
                                               key)
    out = {'system': x, 'n_pairs': len(par), 'n': len(ok), 'fail': fails, 'undefined': undefined, 'est': None}
    if not ok:
        return out
    c = np.column_stack(cols)
    design = stats.Design([ets.spk[p][0] for p in ok], [ets.spk[p][1] for p in ok], ets.mode)
    est, reps, red = stats.bootstrap(design, c, coverage_stat)
    e = c.mean(0)
    return dict(out, **stats.summary(est, reps, red, design, diff=False), e_out=float(e[0]), e_src=float(e[1]),
                e_tgt=float(e[2]))


def agreement(ets, ts):
    """Spearman ρ over all (pair, system) values with both WavLM-SV and ECAPA SIM, and Kendall τ of the system means
    (as analyze_x2.py)."""
    from scipy.stats import kendalltau, spearmanr
    xs, ys, means = [], [], {}
    for n in ECAPA_SYSTEMS:
        e, w = ets.rows(n), ts.rows(n)
        both = [k for k in e if k in w and not stats.failed(e[k]) and not stats.failed(w[k])
                and stats._value(e[k], ECAPA_KEY) and stats._value(w[k], 'sim_wavlm')]
        if not both:
            continue
        a, b = [w[k]['sim_wavlm'] for k in both], [e[k][ECAPA_KEY] for k in both]
        xs += a
        ys += b
        means[n] = (float(np.mean(a)), float(np.mean(b)))
    out = {'n_values': len(xs), 'means': means, 'spearman_rho': None, 'kendall_tau_means': None}
    if len(xs) > 2:
        rho = spearmanr(xs, ys)
        out.update(spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue))
    names = sorted(means)
    if len(names) > 2:
        tau = kendalltau([means[n][0] for n in names], [means[n][1] for n in names])
        out.update(kendall_tau_means=float(tau.statistic), kendall_p=float(tau.pvalue))
    return out


def ecapa_checks(sets, esets):
    """§6.1: ECAPA SIM and SIM_src per stratum, mvc_v1 minus each baseline and CRT-AL; ECAPA NL on VCTK-2k; coverage;
    agreement; ECAPA levels."""
    sim, src = [], []
    for s in SETS:
        e = esets[s]
        for b in BASELINES + (CRT,):
            sim.append(dict(stats.compare(e.rows(MVC), e.rows(b), e.subset('all'), e.spk, ECAPA_KEY, mode=e.mode,
                                          worst='min'), id=f'E-{b}-{s}', set=s, sys_a=MVC, sys_b=b))
            for g in stats.STRATA:
                sub, no_e = stats.endpoint_pairs(e.pairs, 'gate_' + g, e.src_e)
                src.append(dict(stats.compare(e.rows(MVC + SRC), e.rows(b + SRC), [p['pair_id'] for p in sub], e.spk,
                                              ECAPA_KEY, mode=e.mode, worst='max'),
                                id=f'Esrc-{b}-{s}-{g}', set=s, stratum=g, sys_a=MVC, sys_b=b, n_no_source_E=no_e))
    cov = {x: coverage(esets['vctk2k'], x) for x in (MVC, Q32, COPY, CRT) + BASELINES + ('faiss',)
           if esets['vctk2k'].has(x)}
    levels = {}
    for s in SETS:
        e = esets[s]
        for n in ECAPA_SYSTEMS:
            if not e.has(n):
                continue
            row = {'sim': stats.level(e.rows(n), e.subset('all'), e.spk, ECAPA_KEY, mode=e.mode)}
            if e.has(n + SRC):
                pids = [p['pair_id'] for p in e.pairs if e.src_e is None or p['source_speaker'] in e.src_e]
                mm = ecapa_m9(e.rows(n), e.rows(n + SRC))
                row.update(sim_src=stats.level(e.rows(n + SRC), pids, e.spk, ECAPA_KEY, mode=e.mode),
                           m9=stats.level(mm, pids, e.spk, 'm9', mode=e.mode), share_d_pos=at.share_pos(mm, pids))
            levels.setdefault(s, {})[n] = row
    return {'sim': sim, 'sim_src': src, 'nl': nl_family(esets['vctk2k'], key=ECAPA_KEY, beside=False),
            'coverage': cov, 'agreement': {s: agreement(esets[s], sets[s]) for s in SETS}, 'levels': levels}


def dnsmos_checks(dsets):
    """§6.2: OVRL and P808, mvc_v1 minus each baseline and CRT-AL on both sets; the DNSMOS levels."""
    comp = []
    for field in ('OVRL', 'P808'):
        for s in SETS:
            d = dsets[s]
            for b in BASELINES + (CRT,):
                comp.append(dict(stats.compare(d.rows(MVC), d.rows(b), d.subset('all'), d.spk, field, mode=d.mode,
                                               worst='min'), id=f'D-{field}-{b}-{s}', field=field, set=s, sys_a=MVC,
                                 sys_b=b))
    levels = {s: {n: {k: stats.level(dsets[s].rows(n), dsets[s].subset('all'), dsets[s].spk, k, mode=dsets[s].mode)
                      for k in DNSMOS_FIELDS} for n in DNSMOS_SYSTEMS if dsets[s].has(n)} for s in SETS}
    return {'comparisons': comp, 'levels': levels}


def guard_levels(gsets):
    """§3.8, §6.3: the acoustic guards of every guard system, descriptive."""
    out = {}
    for s in SETS:
        g = gsets[s]
        for n in GUARD_SYSTEMS:
            rs = g.rows(n)
            row = {'n_rows': len(rs), 'fail': sum(stats.failed(r) for r in rs.values())}
            if rs:
                row.update({k: stats.level(rs, g.subset('all'), g.spk, k, mode=g.mode) for k in GUARD_FIELDS})
            out.setdefault(s, {})[n] = row
    return out


def apply_flags(fams, L, NL, ec, dn):
    """§6.1 and §6.2 flags, printed beside the WavLM-SV and UTMOS results they qualify."""
    esim = {(h['sys_b'], h['set']): h for h in ec['sim']}
    ovrl = {(h['sys_b'], h['set']): h for h in dn['comparisons'] if h['field'] == 'OVRL'}

    def against(x):
        return x and x.get('est') is not None and x['est'] < 0 and x.get('hi') is not None and x['hi'] < 0
    for fam in ('S', 'H1'):
        for h in fams[fam]['hypotheses']:
            if not (h.get('reject') and h.get('favours') == MVC):
                continue
            if fam == 'S' and h['quantity'] == 'sim' and against(esim.get((h['sys_b'], h['set']))):
                h['second_model_flag'] = NOT_CONFIRMED_SPK
            if h['quantity'] == 'utmos' and against(ovrl.get((h['sys_b'], h['set']))):
                h['second_model_flag'] = NOT_CONFIRMED_NAT
    esrc = {(h['set'], h['stratum']): h for h in ec['sim_src'] if h['sys_b'] == 'knnvc'}
    for r in L['strata'].values():
        e = esrc.get((r['set'], r['stratum']))
        if r['rejects'] and e and e.get('est') is not None and e['est'] > 0 and e.get('lo') is not None and e['lo'] > 0:
            r['second_model_flag'] = NOT_CONFIRMED_SPK
    for g, r in NL['strata'].items():
        e = ec['nl']['strata'][g]
        if r['rejects'] and e.get('lo') is not None and e['lo'] > NL_MARGIN:
            r['second_model_flag'] = 'not confirmed by ECAPA'


# ── descriptive ──────────────────────────────────────────────

def src_pids(ts, g=None):
    """Pairs that enter a SIM_src quantity (VCTK-2k: the source has an E set), optionally one stratum."""
    return [p['pair_id'] for p in ts.pairs if (g is None or stats.stratum(p['genders']) == g)
            and (ts.src_e is None or p['source_speaker'] in ts.src_e)]


def levels(ts):
    """analyze_tests.levels of the plan's systems at 30 s; SIM_src quantities without the sources lacking an E set."""
    lv = at.levels(ts, SYSTEMS)
    keep = set(src_pids(ts))
    if len(keep) < len(ts.pairs):
        for n, row in lv.items():
            for q in ('sim_src', 'secs_src', 'm9'):
                if q in row:
                    row[q] = ts.level(n, q, pids=keep)
            if 'share_d_pos' in row:
                row['share_d_pos'] = at.share_pos(ts.m9_rows(n), list(keep))
    return lv


def src_m9_strata(sets):
    """§5.7: SIM_src and M9 per stratum, mvc_v1 minus each baseline and CRT-AL, with the share D > 0."""
    out = []
    for s in SETS:
        ts = sets[s]
        m9 = {n: ts.m9_rows(n) for n in (MVC, CRT) + BASELINES}
        for g in stats.STRATA:
            pids = src_pids(ts, g)
            for b in BASELINES + (CRT,):
                out.append({'set': s, 'stratum': g, 'sys_b': b,
                            'sim_src': stats.compare(ts.rows(MVC + SRC), ts.rows(b + SRC), pids, ts.spk, 'sim_wavlm',
                                                     mode=ts.mode, worst='max'),
                            'm9': stats.compare(m9[MVC], m9[b], pids, ts.spk, 'm9', mode=ts.mode),
                            'share_d_pos': {MVC: at.share_pos(m9[MVC], pids), b: at.share_pos(m9[b], pids)}})
    return out


def gender_table(ts):
    out = {}
    keep = set(src_pids(ts))
    for n in (MVC, CRT) + BASELINES:
        if not ts.has(n):
            continue
        for g in ('F2F', 'F2M', 'M2F', 'M2M'):
            pids = set(ts.subset(g))
            out.setdefault(n, {})[g] = {q: ts.level(n, q, pids=pids & keep if q == 'sim_src' else pids)
                                        for q in ('wer', 'sim', 'utmos', 'sim_src') if q != 'sim_src' or ts.has(n + SRC)}
    return out


def budget_table(sets):
    """The levels of mvc_v1, CRT-AL and kNN-VC at 3, 30 and 180 s (E16); LTC-2k at 180 s on the 1,699 pairs."""
    out = {}
    for s in SETS:
        ts = sets[s]
        for n in (MVC, CRT, 'knnvc'):
            for b in (3, 30, 180):
                if not ts.has(n, b):
                    continue
                for q in ('wer', 'wer_u', 'sim', 'utmos') + (('mcd',) if s == 'vctk2k' else ()):
                    pids = set(endpoint_pids(ts, q, (b,)))
                    out.setdefault(s, {}).setdefault(n, {}).setdefault(q, {})[b] = ts.level(n, q, b, pids=pids)
    return out


def descriptive_comparisons(sets):
    """§5.7: at 3 s mvc_v1 − each baseline (24, Holm as a guide); at 180 s mvc_v1 − kNN-VC (6); at 3 s and 180 s
    mvc_v1 − CRT-AL (12); mvc_v1 − mvc_copy_latent and mvc_copy_latent − mimi_copy_src at 30 s."""
    b3 = [hyp(sets, f'D3-{bl}-{i}', s, q, MVC, bl, budgets=(3, 3)) for bl in BASELINES
          for i, (s, q) in enumerate(ENDPOINTS, 1)]
    for h in b3:
        if h['sys_b'] == 'freevc' and h['set'] == 'vctk2k':
            h['flag'] = 'VCTK-seen'
    guide = stats.holm([h.get('p') for h in b3])
    at.set_flags(b3, guide)
    return {'baselines_3s': {'hypotheses': b3, 'holm': f'over {len(b3)} (guide only, descriptive)'},
            'knnvc_180s': [hyp(sets, f'D180-knnvc-{i}', s, q, MVC, 'knnvc', budgets=(180, 180))
                           for i, (s, q) in enumerate(ENDPOINTS, 1)],
            'crt_3s_180s': [hyp(sets, f'D{b}-crt-{i}', s, q, MVC, CRT, budgets=(b, b)) for b in (3, 180)
                            for i, (s, q) in enumerate(ENDPOINTS, 1)],
            'mvc_minus_copy_latent': [hyp(sets, f'DC-{i}', s, q, MVC, COPY) for i, (s, q) in enumerate(ENDPOINTS, 1)],
            'copy_latent_minus_mimi_copy': [hyp(sets, f'DM-{i}', s, q, COPY, 'mimi_copy_src')
                                            for i, (s, q) in enumerate(ENDPOINTS, 1)]}


# ── sensitivity, power ───────────────────────────────────────

def ni_worst_case(NI):
    hyps = []
    for h in NI['hypotheses']:
        wc = h.get('worst_case') or {}
        hyps.append(dict(h, p_ni=wc['p_ni']) if wc.get('p_ni') is not None else dict(h))
    other = ni_family(hyps)
    return {'worst_case_used': [h['id'] for h in NI['hypotheses'] if (h.get('worst_case') or {}).get('p_ni')
                                is not None], 'changed': at.changed(NI, other)}


def union_h1_s(H1, S):
    hyps = [dict(h) for h in H1['hypotheses'] + S['hypotheses']]
    flags = stats.holm([h.get('p') for h in hyps])
    at.set_flags(hyps, flags)
    dec = {h['id']: (h['reject'], h['favours']) for h in hyps}
    base = {**at.decisions(H1), **at.decisions(S)}
    return {'m': len(hyps), 'decisions': dec, 'changed': {'holm': sorted(k for k, v in dec.items() if v != base[k])}}


def sensitivity(sets, fams, NL, L):
    unc = build_families(sets, wer='wer')
    ind = build_families(sets, mode_ltc=stats.INDEPENDENT)
    l_ind = family_l(sets, mode_ltc=stats.INDEPENDENT)
    worst = {k: at.worst_case_family(fams[k]) for k in ('H1', 'S', 'RB', 'Q')}
    worst['NI'] = ni_worst_case(fams['NI'])
    return {'wer_uncapped': {k: {'changed': at.changed(fams[k], v), 'family': v} for k, v in unc.items()},
            'ltc_independent': {k: {'changed': at.changed(fams[k], v), 'family': v} for k, v in ind.items()},
            'ltc_independent_l': {k: l_ind['strata'][k]['rejects'] for k in ('L2s', 'L2c')},
            'worst_case': worst,
            'worst_case_nl': {r['id']: (r['worst_case'] if r.get('worst_case') else r)['rejects']
                              for r in NL['strata'].values()},
            'worst_case_l': L['worst_case'],
            'union_h1_s': union_h1_s(fams['H1'], fams['S']),
            'outcome': {'wer_uncapped': outcome(unc['H1']), 'ltc_independent': outcome(ind['H1'])}}


def multiplier(fam):
    from scipy.stats import norm
    alpha, m, two = POWER[fam]
    return float(norm.ppf(1 - alpha / (2 * m if two else m)) + norm.ppf(0.80))


def power(fams, NL, L):
    """§5.5 with mvc_v1's realised SEs: MDE = multiplier × SE for the two-sided families; for NI the true Δ that gives
    80 % power against its margin; for NL and L the true value that gives 80 % power below the margin."""
    out = {'multipliers': {k: multiplier(k) for k in POWER}, 'rows': []}
    for k in ('H1', 'S', 'RB', 'Q', 'NI'):
        z = out['multipliers'][k]
        for h in fams[k]['hypotheses']:
            se = h.get('se')
            row = {'family': k, 'id': h['id'], 'se': se, 'mde': None if se is None else z * se, 'needs': None}
            if se is not None:
                row['needs'] = (f'abs(Δ) ≥ {z * se:.4f}' if k != 'NI' else
                                f'Δ ≤ {h["margin"] - z * se:+.4f}' if h['sign'] < 0 else
                                f'Δ ≥ {-h["margin"] + z * se:+.4f}')
            out['rows'].append(row)
    z = out['multipliers']['NL']
    for fam, margin, res in (('NL', NL_MARGIN, NL['strata'].values()), ('L', stats.MARGIN_L, L['strata'].values())):
        for r in res:
            se = r.get('se')
            out['rows'].append({'family': fam, 'id': r['id'], 'se': se, 'mde': None if se is None else z * se,
                                'needs': None if se is None else f'true value ≤ {margin - z * se:+.4f}'})
    return out


# ── §7 ───────────────────────────────────────────────────────

def outcome(H1):
    """§7.1 from H1's decisions and point estimates. Returns the outcome (None when no row applies: a non-rejected
    estimate that is missing or exactly 0, with no rejection favouring CRT-AL and none against mvc_v1)."""
    hs = H1['hypotheses']
    fav = {h['id']: at.favours(h, h['quantity'], (h['sys_a'], h['sys_b'])) for h in hs}
    rej_mvc = [h['id'] for h in hs if h['reject'] and h['favours'] == MVC]
    rej_crt = [h['id'] for h in hs if h['reject'] and h['favours'] == CRT]
    non = [h['id'] for h in hs if not h['reject']]
    non_mvc = [k for k in non if fav[k] == MVC]
    non_crt = [k for k in non if fav[k] == CRT]
    if rej_crt:
        o = 'B'
    elif len(rej_mvc) == len(hs):
        o = 'A'
    elif non and len(non_mvc) == len(non):
        o = 'A−'
    elif non_crt:
        o = 'C'
    else:
        o = None
    return {'outcome': o, 'framing': OUTCOMES.get(o, 'no row of §7.1 applies'), 'reject_favour_mvc': rej_mvc,
            'reject_favour_crt_al': rej_crt, 'not_rejected': non, 'not_rejected_favour_mvc': non_mvc,
            'not_rejected_favour_crt_al': non_crt,
            'not_rejected_no_direction': [k for k in non if fav[k] is None]}


def leakage_text(NL, L):
    """§7.2."""
    fail_nl = [r['id'] for r in NL['strata'].values() if r['rejects'] is not True]
    flags = [f'{r["id"]}: {r["second_model_flag"]}' for r in NL['strata'].values() if r.get('second_model_flag')]
    if not fail_nl:
        nl = ('no source-speaker leakage beyond the natural other-speaker level (+0.03, WavLM-SV) on VCTK-2k'
              + (f' ({"; ".join(flags)})' if flags else ''))
    else:
        nl = (f'leakage above the natural level not excluded ({", ".join(fail_nl)}); no claim of a converter without '
              f'source leakage')
    fail_l = [k for k, r in L['strata'].items() if r['rejects'] is not True]
    lflags = [f'{k}: {r["second_model_flag"]}' for k, r in L['strata'].items() if r.get('second_model_flag')]
    lt = ('not leakier than kNN-VC (margin +0.01)' + (f' ({"; ".join(lflags)})' if lflags else '') if not fail_l else
          f'leakage relative to kNN-VC not shown to be within +0.01 ({", ".join(fail_l)})')
    return {'nl': nl, 'nl_failed': fail_nl, 'l': lt, 'l_failed': fail_l, 'nl_in_abstract': bool(fail_nl)}


def wording(fams):
    """§7.3 per baseline, set and endpoint: 'better' / 'worse' only from S, 'comparable' only from NI; every statement
    the two families allow is given."""
    ni = {(h['sys_b'], h['set'], h['quantity']): h for h in fams['NI']['hypotheses']}
    out = []
    for h in fams['S']['hypotheses']:
        b, s, q = h['sys_b'], h['set'], h['quantity']
        seen = b == 'freevc' and s == 'vctk2k'
        n = ni.get((b, s, q))
        parts = []
        if h['reject'] and h['favours'] == MVC:
            parts.append('no claim (VCTK-seen)' if seen else f'better than {b}')
        elif h['reject'] and h['favours'] == b:
            parts.append(f'worse than {b}')
        if n is not None and n['reject'] and not seen:
            parts.append(f'comparable to {b} (non-inferiority, {n["bound"]})')
        w = '; '.join(parts) or 'no claim; the CI is given'
        out.append({'id': h['id'], 'baseline': b, 'set': s, 'quantity': q, 'wording': w,
                    'flag': h.get('second_model_flag'), 'ci': [h.get('lo'), h.get('hi')],
                    'ni': None if n is None else n['reject']})
    return out


def tiers(oc, cov, lv, rb_rows, RB):
    """§7.3: Must, Should and Stretch as point-estimate benchmarks (never tests)."""
    def est(s, q):
        return ((lv[s].get(MVC) or {}).get(q) or {}).get('est')
    rb3 = {h['set']: h for h in RB['hypotheses'] if h['id'] in ('RB-3-1', 'RB-3-2')}
    should, stretch = {}, {}
    for s in SETS:
        w = rb3[s].get('est')
        should[s] = {'wer': est(s, 'wer'), 'sim': est(s, 'sim'), 'utmos': est(s, 'utmos'), 'wer_3s_minus_30s': w,
                     'rbn3_ci_rule': next(r['passes'] for r in rb_rows if r['id'] == 'RBN-3' and r['set'] == s)}
        t = SHOULD[s]
        should[s]['meets'] = all(x is not None for x in (should[s]['wer'], should[s]['sim'], should[s]['utmos'], w)) \
            and should[s]['wer'] <= t['wer'] and should[s]['sim'] >= t['sim'] and should[s]['utmos'] >= t['utmos'] \
            and w < 1.5
        u = STRETCH[s]
        stretch[s] = {'meets': est(s, 'wer') is not None and est(s, 'sim') is not None and est(s, 'wer') <= u['wer']
                      and est(s, 'sim') >= u['sim']}
    c = (cov.get(MVC) or {}).get('est')
    return {'must': {'outcome_a': oc['outcome'] == 'A', 'coverage': c, 'coverage_ge_0.50': c is not None and c >= 0.5,
                     'leakage_reported_per_stratum': True,
                     'meets': oc['outcome'] == 'A' and c is not None and c >= 0.5},
            'should': dict(should, meets=all(should[s]['meets'] for s in SETS)),
            'stretch': dict(stretch, meets=all(stretch[s]['meets'] for s in SETS))}


# ── analysis ─────────────────────────────────────────────────

def analyse(sets, esets, dsets, gsets):
    fams = build_families(sets)
    NL = nl_family(sets['vctk2k'])
    L = family_l(sets)
    rb_rows = rbn(fams['RB'])
    ec = ecapa_checks(sets, esets)
    dn = dnsmos_checks(dsets)
    apply_flags(fams, L, NL, ec, dn)
    oc = outcome(fams['H1'])
    lv = {s: levels(sets[s]) for s in SETS}
    return {'families': fams, 'nl': NL, 'l': L, 'rbn': rb_rows,
            'nl_reference': {x: nl_family(sets['vctk2k'], x=x, ids=(f'{x}/same', f'{x}/cross'), beside=False)
                             for x in (CRT,) + BASELINES},
            'outcome': oc, 'leakage': leakage_text(NL, L), 'wording': wording(fams),
            'tiers': tiers(oc, ec['coverage'], lv, rb_rows, fams['RB']), 'power': power(fams, NL, L),
            'sensitivity': sensitivity(sets, fams, NL, L), 'ecapa': ec, 'dnsmos': dn, 'guards': guard_levels(gsets),
            'levels': lv, 'eer': {s: at.eer_table(sets[s]) for s in SETS},
            'src_m9_strata': src_m9_strata(sets), 'gender': {s: gender_table(sets[s]) for s in SETS},
            'budget_table': budget_table(sets), 'descriptive': descriptive_comparisons(sets),
            'shortfall': {s: sorted(sets[s].shortfall) for s in SETS},
            'no_source_E': {s: len(sets[s].pairs) - len(src_pids(sets[s])) for s in SETS}}


# ── report ───────────────────────────────────────────────────

f, ci, digits = at.f, at.ci, at.digits


def civ(x, d=3):
    return f'{f(x.get("est"), d)} [{f(x.get("lo"), d)}, {f(x.get("hi"), d)}]' if x and x.get('est') is not None \
        else 'n/a'


def fmt_h(h):
    """analyze_tests.fmt_h, with a §6 flag shown next to the hypothesis id."""
    flag = '; '.join(x for x in (h.get('flag'), h.get('second_model_flag')) if x)
    return at.fmt_h(dict(h, flag=flag) if flag else h)


HEAD = at.HEAD
DEV_COL = 'mvc_v1 on Dev-900, selected on Dev'


def dev_line(dev):
    return (f'mvc_v1 on Dev-900, selected on Dev: capped WER {f(dev["wer"], 2)}, SIM {f(dev["sim"])}, UTMOS '
            f'{f(dev["utmos"])}, NL same / cross {f(dev["nl"]["same"])} / {f(dev["nl"]["cross"])}.')


def render_family(title, fam, caption='', dev=None):
    """A family table; with dev, mvc_v1's Dev-900 value of each endpoint in a last column (§4.9, §9)."""
    head = HEAD if dev is None else '\n'.join(x + y for x, y in zip(HEAD.split('\n'), (f' {DEV_COL} |', '---|')))
    L = [f'## {title}', '', f'Holm {fam["holm"]}, α {fam["alpha"]}.' + (f' {caption}' if caption else ''), '', head]
    return L + [fmt_h(h) + ('' if dev is None else f' {f(dev[h["quantity"]], digits(h["quantity"]))} |')
                for h in fam['hypotheses']] + ['']


def render(res):
    fams, NL, Lg, ec, dn = res['families'], res['nl'], res['l'], res['ecapa'], res['dnsmos']
    L = ['# MimiFlow-VC test results v1 (mvc/ANALYSIS_PLAN_mvc_v1.md §5–§7)', '',
         f'Written {res["written"]} by `mvc/analysis/analyze_mvc.py` (sha256 {res["script_sha256"]}; tests '
         f'{res["record_checked"].get("test_analyze_mvc.py")}; stats.py {res["frozen"]["stats"]}; analyze_tests.py '
         f'{res["frozen"]["analyze_tests"]}; analyze_exploratory.py {res["frozen"]["analyze_exploratory"]}). '
         f'Selection record sha256 {res["record_sha256"]} (.json), {res["record_md_sha256"]} (.md). Crossed '
         f'bootstrap B = {stats.B}, seed {stats.SEED}: '
         f'VCTK-2k independent, LTC-2k linked.', ''] + [f'- {x}' for x in NOTES] + ['']
    o, lk = res['outcome'], res['leakage']
    L += ['## Outcome (§7.1) and leakage (§7.2)', '',
          f'- **Outcome {o["outcome"] or "none"}**: {o["framing"]}. Rejections favouring mvc_v1: '
          f'{", ".join(o["reject_favour_mvc"]) or "none"}; favouring CRT-AL: {", ".join(o["reject_favour_crt_al"]) or "none"}; '
          f'not rejected: {", ".join(o["not_rejected"]) or "none"} (point estimate favours CRT-AL: '
          f'{", ".join(o["not_rejected_favour_crt_al"]) or "none"}).',
          f'- **Natural level (NL)**: {lk["nl"]}.' + (' This goes in the abstract.' if lk['nl_in_abstract'] else ''),
          f'- **Relative to kNN-VC (L)**: {lk["l"]}.', '']
    dev = res['dev900']
    L += [dev_line(dev), '']
    t = res['tiers']
    L += ['### Tiers of PLAN §6.1 (point-estimate benchmarks, not tests)', '',
          f'- Must: outcome A {t["must"]["outcome_a"]}, ECAPA coverage {f(t["must"]["coverage"])} (≥ 0.50: '
          f'{t["must"]["coverage_ge_0.50"]}), leakage reported per stratum → **{t["must"]["meets"]}**.']
    for s in SETS:
        x = t['should'][s]
        L.append(f'- Should, {s}: capped WER {f(x["wer"], 2)}, SIM {f(x["sim"])}, UTMOS {f(x["utmos"])}, WER(3 s) − '
                 f'WER(30 s) {f(x["wer_3s_minus_30s"], 2)} (RBN-3 CI rule {x["rbn3_ci_rule"]}) → {x["meets"]}; '
                 f'Stretch → {t["stretch"][s]["meets"]}.')
    L += [f'- Should overall {t["should"]["meets"]}; Stretch overall {t["stretch"]["meets"]}.', '']
    L += render_family('Family H1 (primary): mvc_v1 − CRT-AL (cand_pm18_rq)', fams['H1'], dev=dev)
    L += ['## Family NL: natural-level leakage on VCTK-2k, H0 NL_g ≥ +0.03', '',
          '| Test | Stratum | SIM_src mvc_v1 | SIM_src target_copy | NL [95 % CI] | SE | Rejects | n (no source E; failed '
          'mvc_v1/target_copy) | Worst case | summarize.py value [95 % CI] | SIM mvc_v1 / target_copy, same pairs | '
          f'ECAPA NL [95 % CI] | Flag | NL, {DEV_COL} |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g, r in NL['strata'].items():
        wc, e = r.get('worst_case'), ec['nl']['strata'][g]
        L.append(f'| {r["id"]} | {g} | {f(r.get("a"))} | {f(r.get("b"))} | {civ(r)} | {f(r.get("se"))} | '
                 f'{r["rejects"]} | {r.get("n", 0)} ({r["n_no_source_E"]}; {r["fail_a"]}/{r["fail_b"]}) | '
                 f'{"hi " + f(wc.get("hi")) + ", rejects " + str(wc.get("rejects")) if wc else "–"} | '
                 f'{civ(r["summarize_value"])} | {civ(r["sim_x_same_pairs"])} / {civ(r["sim_target_copy_same_pairs"])} | '
                 f'{civ(e)} | {r.get("second_model_flag", "")} | {f(dev["nl"][g])} |')
    L += ['', f'Both reject (the claim): {NL["all_reject"]}; at the worst case: {NL["worst_case_all_reject"]}.', '',
          'Reference, paired NL of the comparators (descriptive): ' + '; '.join(
              f'{x} {f(v["strata"]["same"].get("est"))} / {f(v["strata"]["cross"].get("est"))}'
              for x, v in res['nl_reference'].items()) + '.', '']
    L += ['## Family L: SIM_src(mvc_v1) − SIM_src(kNN-VC), H0 Δ_src ≥ +0.01', '',
          '| Test | Set / stratum | SIM_src mvc_v1 | SIM_src kNN-VC | Δ_src [95 % CI] | Rejects | n (no source E; failed) '
          '| Worst case | Flag |', '|---|---|---|---|---|---|---|---|---|']
    for k, r in Lg['strata'].items():
        wc = r.get('worst_case')
        L.append(f'| {k} | {r["set"]} / {r["stratum"]} | {f(r.get("a"))} | {f(r.get("b"))} | {civ(r)} | {r["rejects"]} '
                 f'| {r.get("n", 0)} ({r["n_no_source_E"]}; {r.get("fail_a", 0)}/{r.get("fail_b", 0)}) | '
                 f'{"hi " + f(wc.get("hi")) + ", rejects " + str(wc.get("rejects")) if wc else "–"} | '
                 f'{r.get("second_model_flag", "")} |')
    L += ['', f'All four reject (the claim): {Lg["all_reject"]}; at the worst case: {Lg["worst_case"]["all_reject"]}.',
          '']
    nl_cap = (f'NL (§7.2): NL1s {NL["strata"]["same"]["rejects"]}, NL1c {NL["strata"]["cross"]["rejects"]}; '
              f'{lk["nl"]}.')
    L += render_family('Family S: mvc_v1 − each baseline', fams['S'], nl_cap, dev=dev)
    L += ['## Family NI: non-inferiority against Seed-VC and kNN-VC', '',
          f'Holm {fams["NI"]["holm"]} on p_NI = (1 + #{{s_q Δ* ≤ −δ_q}}) / (B + 1). {nl_cap}', '',
          f'| Hypothesis | Set | Endpoint | mvc_v1 | B | Δ [95 % CI] | Margin | p_NI | CI rule | Holm | n pairs | {DEV_COL} |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h in fams['NI']['hypotheses']:
        n = digits(h['quantity'])
        L.append(f'| {h["id"]} | {h["set"]} | {h["label"]} | {f(h.get("a"), n)} | {f(h.get("b"), n)} | '
                 f'{civ(h, n)} | {h["bound"]} | {f(h.get("p_ni"), 4)} | {h["ci_rule"]} | '
                 f'{"non-inferior" if h["reject"] else "not rejected"} | {h.get("n", 0)} ({h.get("fail_a", 0)}/'
                 f'{h.get("fail_b", 0)} failed) | {f(dev[h["quantity"]], n)} |')
    L.append('')
    L += render_family('Family RB: mvc_v1 at 3 s and 180 s − mvc_v1 at 30 s', fams['RB'],
                       'LTC-2k at 180 s and its 30 s comparator on the pairs without a reference shortfall.')
    L += ['### RBN (CI rule, no Holm)', '', '| Check | Set | From | Δ [95 % CI] | Rule | Passes |',
          '|---|---|---|---|---|---|']
    for r in res['rbn']:
        L.append(f'| {r["id"]} | {r["set"]} | {r["from"]} | {civ(r, digits(r["quantity"]))} | '
                 f'{"lower" if r["limit"] == "lo" else "upper"} limit {"above" if r["limit"] == "lo" else "below"} '
                 f'{r["bound"]:+g} | {r["passes"]} |')
    L.append('')
    L += render_family('Family Q: mvc_v1_q32 − mvc_v1', fams['Q'], dev=dev)
    L += ['## Wording against the baselines (§7.3)', '', '| Hypothesis | Set | Endpoint | Wording | Flag |',
          '|---|---|---|---|---|']
    L += [f'| {w["id"]} | {w["set"]} | {w["quantity"]} | {w["wording"]} | {w["flag"] or ""} |' for w in res['wording']]
    p = res['power']
    L += ['', '## Power with the realised SEs (§5.5)', '',
          'Multipliers: ' + ', '.join(f'{k} {v:.2f}' for k, v in p['multipliers'].items()) + '.', '',
          '| Family | Hypothesis | SE | MDE | 80 % power needs |', '|---|---|---|---|---|']
    for r in p['rows']:
        L.append(f'| {r["family"]} | {r["id"]} | {f(r["se"], 4)} | {f(r["mde"], 4)} | {r["needs"] or "–"} |')
    sens = res['sensitivity']
    L += ['', '## Sensitivity analyses (§5.6)', '']
    for k, v in sens['wer_uncapped'].items():
        L.append(f'- Uncapped corpus WER, family {k}: decisions that change: {at.fmt_changed(v["changed"])}.')
    for k, v in sens['ltc_independent'].items():
        L.append(f'- LTC-2k with the independent bootstrap, family {k}: decisions that change: '
                 f'{at.fmt_changed(v["changed"])}.')
    L.append(f'- LTC-2k with the independent bootstrap, L: L2s {sens["ltc_independent_l"]["L2s"]}, L2c '
             f'{sens["ltc_independent_l"]["L2c"]}.')
    for k, v in sens['worst_case'].items():
        L.append(f'- Failed conversions at their worst case, family {k}: computed for '
                 f'{", ".join(v["worst_case_used"]) or "no hypothesis (equal failure counts)"}; decisions that change: '
                 f'{at.fmt_changed(v["changed"])}.')
    L.append(f'- Worst case, NL: {sens["worst_case_nl"]}; L: {sens["worst_case_l"]["rejects"]} (all four '
             f'{sens["worst_case_l"]["all_reject"]}).')
    u = sens['union_h1_s']
    L.append(f'- H1 ∪ S, Holm over {u["m"]}: decisions that change: {", ".join(u["changed"]["holm"]) or "none"}.')
    L.append(f'- Outcome with uncapped WER: {sens["outcome"]["wer_uncapped"]["outcome"]}; with LTC-2k independent: '
             f'{sens["outcome"]["ltc_independent"]["outcome"]}.')
    L += ['- NL beside the summarize.py value: in the NL table (descriptive; it decides nothing).', '']
    L += ['## Second speaker model: ECAPA (§6.1)', '', '| Comparison | Set | mvc_v1 | B | Δ SIM [95 % CI] | p | n |',
          '|---|---|---|---|---|---|---|']
    L += [f'| {h["id"]} | {h["set"]} | {f(h.get("a"))} | {f(h.get("b"))} | {civ(h)} | {f(h.get("p"), 4)} | '
          f'{h.get("n", 0)} |' for h in ec['sim']]
    L += ['', '| Comparison | Set / stratum | Δ SIM_src [95 % CI] | p | n |', '|---|---|---|---|---|']
    L += [f'| {h["id"]} | {h["set"]} / {h["stratum"]} | {civ(h)} | {f(h.get("p"), 4)} | {h.get("n", 0)} |'
          for h in ec['sim_src']]
    L += ['', '| ECAPA coverage, VCTK-2k pairs with P, 30 s | Coverage [95 % CI] | E_out | E_src | E_tgt | n |',
          '|---|---|---|---|---|---|']
    L += [f'| {x} | {civ(c)} | {f(c.get("e_out"))} | {f(c.get("e_src"))} | {f(c.get("e_tgt"))} | {c["n"]} |'
          for x, c in ec['coverage'].items()]
    L += ['', '| Set | values | Spearman ρ | Kendall τ of system means |', '|---|---|---|---|']
    L += [f'| {s} | {a["n_values"]} | {f(a["spearman_rho"])} | {f(a["kendall_tau_means"])} |'
          for s, a in ec['agreement'].items()]
    L += ['', '## Second naturalness predictor: DNSMOS (§6.2, descriptive)', '',
          '| Comparison | Set | mvc_v1 | B | Δ [95 % CI] | n |', '|---|---|---|---|---|---|']
    L += [f'| {h["id"]} | {h["set"]} | {f(h.get("a"))} | {f(h.get("b"))} | {civ(h)} | {h.get("n", 0)} |'
          for h in dn['comparisons']]
    L += ['', '## Descriptive (CIs only)', '']
    for s in SETS:
        L += [f'### Levels, {s}, 30 s', '',
              '| System | rows (failed) | capped WER % | WER % | capped WER % base | WER % base | capped CER % | SIM | SECS '
              '| UTMOS | SIM_src | SECS_src | M9 D | share D > 0 | EER | dB-MCD | log-F0 RMSE | V/UV error | F0 PCC |',
              '|---|' + '---|' * 18]
        for n, row in res['levels'][s].items():
            flag = f' [{row["flag"]}]' if row.get('flag') else ''
            L.append(f'| {n}{flag} | {row["n_rows"]} ({row["fail"]}) | ' + ' | '.join(
                ci(row.get(q), digits(q)) for q in ('wer', 'wer_u', 'wer_b', 'wer_bu', 'cer', 'sim', 'secs', 'utmos',
                                                    'sim_src', 'secs_src', 'm9'))
                + f' | {f((row.get("share_d_pos") or {}).get("share"))} | {f(res["eer"][s].get(n))} | '
                + ' | '.join(ci(row.get(q), digits(q)) for q in ('mcd', 'lf0', 'vuv', 'f0pcc')) + ' |')
        L += ['', dev_line(dev), '']
    L += ['### SIM_src and M9 per stratum: mvc_v1 − comparator', '',
          '| Set / stratum | B | Δ SIM_src [95 % CI] | Δ M9 [95 % CI] | share D > 0 mvc_v1 / B |', '|---|---|---|---|---|']
    for r in res['src_m9_strata']:
        sh = r['share_d_pos']
        L.append(f'| {r["set"]} / {r["stratum"]} | {r["sys_b"]} | {civ(r["sim_src"])} | {civ(r["m9"])} | '
                 f'{f(sh[MVC]["share"])} / {f(sh[r["sys_b"]]["share"])} |')
    d = res['descriptive']
    L += ['', f'### mvc_v1 − each baseline at 3 s (descriptive; Holm {d["baselines_3s"]["holm"]})', '', HEAD]
    L += [fmt_h(h) for h in d['baselines_3s']['hypotheses']]
    for key, title in (('knnvc_180s', 'mvc_v1 − kNN-VC at 180 s'), ('crt_3s_180s', 'mvc_v1 − CRT-AL at 3 s and 180 s'),
                       ('mvc_minus_copy_latent', 'mvc_v1 − mvc_copy_latent, 30 s'),
                       ('copy_latent_minus_mimi_copy', 'mvc_copy_latent − mimi_copy_src, 30 s')):
        L += ['', f'### {title} (CIs only)', '', '| Id | Set | Endpoint | A | B | Δ [95 % CI] | n (failed A/B) |',
              '|---|---|---|---|---|---|---|']
        for h in d[key]:
            n = digits(h['quantity'])
            L.append(f'| {h["id"]} | {h["set"]} | {h["label"]} | {f(h.get("a"), n)} | {f(h.get("b"), n)} | '
                     f'{civ(h, n)} | {h.get("n", 0)} ({h.get("fail_a", 0)}/{h.get("fail_b", 0)}) |')
    L += ['', '### Budget table (E16)', '', '| Set | System | Endpoint | 3 s | 30 s | 180 s |', '|---|---|---|---|---|---|']
    for s, dd in res['budget_table'].items():
        for n, qs in dd.items():
            for q, bs in qs.items():
                L.append(f'| {s} | {n} | {at.LABEL[q]} | ' + ' | '.join(
                    ci(bs.get(b), digits(q)) + (f' (n {bs[b]["n"]})' if b in bs else '') for b in (3, 30, 180)) + ' |')
    L += ['', '### Gender pairs', '']
    for s, dd in res['gender'].items():
        L += [f'{s}:', '', '| System | Pair | capped WER % | SIM | UTMOS | SIM_src |', '|---|---|---|---|---|---|']
        for n, gs in dd.items():
            for gp, v in gs.items():
                L.append(f'| {n} | {gp} | ' + ' | '.join(ci(v.get(q), digits(q)) for q in ('wer', 'sim', 'utmos',
                                                                                              'sim_src')) + ' |')
        L.append('')
    for s in SETS:
        L += [f'### ECAPA levels, {s}', '', '| System | SIM | SIM_src | M9 D | share D > 0 |', '|---|---|---|---|---|']
        for n, row in ec['levels'][s].items():
            L.append(f'| {n} | ' + ' | '.join(ci(row.get(q), 3) for q in ('sim', 'sim_src', 'm9'))
                     + f' | {f((row.get("share_d_pos") or {}).get("share"))} |')
        L += ['', f'### DNSMOS levels, {s}', '', '| System | ' + ' | '.join(DNSMOS_FIELDS) + ' |',
              '|---|' + '---|' * len(DNSMOS_FIELDS)]
        for n, row in dn['levels'][s].items():
            L.append(f'| {n} | ' + ' | '.join(ci(row.get(q), 3) for q in DNSMOS_FIELDS) + ' |')
        L += ['', f'### Acoustic guards, {s}', '', '| System | rows (failed) | ' + ' | '.join(GUARD_FIELDS) + ' |',
              '|---|---|' + '---|' * len(GUARD_FIELDS)]
        for n, row in res['guards'][s].items():
            L.append(f'| {n} | {row["n_rows"]} ({row["fail"]}) | ' + ' | '.join(ci(row.get(q), 3) for q in GUARD_FIELDS)
                     + ' |')
        L.append('')
    L += ['## Inputs', ''] + [f'- {k}: `{v}`' for k, v in res['done'].items()] + \
         [f'- `{x["path"]}` sha256 {x["sha256"]}' + (f' ({x["rows"]} rows)' if 'rows' in x else '')
          for x in res['inputs']]
    return '\n'.join(L) + '\n'


# ── main ─────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--v1', default=os.path.join(ROOT, 'paper', 'journal_extension', 'results', 'v1'))
    ap.add_argument('--mr', default=os.path.join(ROOT, 'paper', 'mimi_vc', 'results'))
    ap.add_argument('--record', default=os.path.join(ROOT, 'mvc', 'MVC_SELECTION_RECORD_v1.json'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'mvc', 'TEST_RESULTS_mvc_v1.md'))
    a = ap.parse_args(argv)
    frozen = check_frozen()
    for path in (a.out, os.path.splitext(a.out)[0] + '.json'):
        if os.path.exists(path):
            raise SystemExit(f'ERROR: {path} exists: analyze_mvc.py runs once and both outputs of a re-run are kept '
                             f'(§8.3); give a new --out')
    if not os.path.exists(a.record):
        raise SystemExit(f'ERROR: {a.record} missing: the committed selection record is required')
    rec_md = os.path.splitext(a.record)[0] + '.md'
    if not os.path.exists(rec_md):
        raise SystemExit(f'ERROR: {rec_md} missing: the selection record is the .md and .json select_mvc.py writes '
                         f'together')
    rec_shas = (at.sha256(a.record), at.sha256(rec_md))
    rec = json.load(open(a.record))
    if not isinstance(rec, dict) or rec.get('status') != 'complete':
        raise SystemExit(f'ERROR: {a.record} has status {rec.get("status") if isinstance(rec, dict) else None!r}, '
                         f'not complete: not the selection record')
    checked = check_record(rec, os.path.abspath(__file__), TEST_FILE)
    check_record_md(rec, open(rec_md).read())
    dev = dev900(rec)
    pp = {'vctk2k': os.path.join(a.v1, 'pairs', 'vctk2k.jsonl'), 'ltc2k': os.path.join(a.v1, 'pairs', 'ltc2k.jsonl'),
          'source_e': os.path.join(a.v1, 'pairs', 'vctk2k_source_E.json')}
    for k, path in pp.items():
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {path} missing')
        check_sha(path, PAIR_SHA256[k], k)
    inputs = [{'path': os.path.abspath(p), 'sha256': at.sha256(p)} for p in [a.record, rec_md] + list(pp.values())]
    pairs = {s: at.read_pairs(pp[s]) for s in SETS}
    src_e = {k for k, v in json.load(open(pp['source_e'])).items() if (v or {}).get('paths')}
    n_short, n_par = len(shortfall_pairs(pairs['ltc2k'])), sum(bool(p.get('P')) for p in pairs['vctk2k'])
    if (n_short, n_par) != (EXPECTED['shortfall'], EXPECTED['par']):
        raise SystemExit(f'ERROR: the lists have {n_short} LTC-2k shortfall pairs and {n_par} VCTK-2k pairs with P; '
                         f'§1 says {EXPECTED["shortfall"]} and {EXPECTED["par"]}')
    m_j, m_r = os.path.join(a.v1, 'm'), os.path.join(a.mr, 'm')
    done = {f'journal job M part {k}': v for k, v in
            check_parts(m_j, ['t1', 't2', 't3', 't4'], JOURNAL_SELECTION_SHA256, 'journal job M').items()}
    done['journal job X1'] = ax.x1_done(os.path.join(a.v1, 'x1'), {k: v for k, v in done.items()})
    done['journal job X2'] = done_line(os.path.join(a.v1, 'x2', 'x2.done'), JOURNAL_SELECTION_SHA256, 'job X2')
    done.update({f'MimiFlow-VC T1 part {k}': v for k, v in
                 check_parts(m_r, ['t1', 't2', 't3'], rec_shas, 'MimiFlow-VC T1').items()})
    sets, esets, dsets, gsets = {}, {}, {}, {}
    for s in SETS:
        main_part = 't1' if s == 'vctk2k' else 't2'
        J, ins = at.load_set(m_j, s, [(main_part, 30), ('t3', 3), ('t3', 180), ('t4', 30), ('t4', 3), ('t4', 180)],
                             pairs[s])
        inputs += ins
        X, ins = at.load_set(os.path.join(a.v1, 'x1'), s, [('m', 3)], pairs[s])
        inputs += ins
        M = []
        for part, files, allowed in ((main_part, [(main_part, 30)], MVC_SYSTEMS), ('t3', [('t3', 3), ('t3', 180)],
                                                                                 T3_SYSTEMS)):
            Rp, ins = at.load_set(m_r, s, files, pairs[s])
            inputs += ins
            label = f'MimiFlow-VC T1 part {part}'
            M.append((label, check_systems(Rp, allowed, f'{s} {label}')))
        R = merge([('journal job M', J), ('journal job X1', X)] + M, whole=True)
        sets[s] = make_set(s, pairs[s], R, src_e)
        E, ins = load_ecapa([('journal X2', os.path.join(a.v1, 'x2', 'ecapa'), None),
                             ('MimiFlow-VC', os.path.join(a.mr, 'x2', 'ecapa'), MVC_SYSTEMS)], s, pairs[s])
        inputs += ins
        esets[s] = make_set(s, pairs[s], E, src_e, keep=ECAPA_SYSTEMS)
        D, ins = load_glob(os.path.join(a.mr, 'dnsmos'), s, pairs[s], 'DNSMOS')
        inputs += ins
        dsets[s] = make_set(s, pairs[s], D, src_e, keep=DNSMOS_SYSTEMS)
        G, ins = load_glob(os.path.join(a.mr, 'guard'), s, pairs[s], 'guard')
        inputs += ins
        gsets[s] = make_set(s, pairs[s], G, src_e, keep=GUARD_SYSTEMS)
    problems = required(sets, esets, dsets)
    if problems:
        raise SystemExit('ERROR: ' + '; '.join(problems))
    res = analyse(sets, esets, dsets, gsets)
    res.update(written=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
               script_sha256=at.sha256(os.path.abspath(__file__)), frozen=frozen, record_sha256=rec_shas[0],
               record_md_sha256=rec_shas[1], record_checked=checked, dev900=dev, inputs=inputs, done=done)
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
