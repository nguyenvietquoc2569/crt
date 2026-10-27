#!/usr/bin/env python3
"""
Dev selection R2 of mvc/ANALYSIS_PLAN_mvc_v1.md §4: R2.0 snapshot (§4.2), R2.1 Dev-S100 screen (§4.3, the
re-application of D-mvc-2), R2.2 finalists (§4.4), R2.3 the choice on full Dev with the Dev-900 repeat (§4.5), R2.4
reference budgets (§4.6), B1 and the other triggers (§4.7), and the selection record (§4.8):
mvc/MVC_SELECTION_RECORD_v1.{md,json}, the three frozen inference JSONs and mvc/mvc_selection.env. CPU only, on the
Mac (.venv-stats). Statistics
from benchmark/stats.py, imported unchanged and checked by sha256: linked crossed bootstrap of the Dev speakers,
B = 10,000, seed 20260928.

Stages, run in this order, each re-deriving every earlier stage from the files and stopping on any difference from that
stage's saved output in --state_dir (§4.1):
  --stage r2_0              R2.0 from --ref (Dev-S100 at the reference setting, 150k and 200k) and --recon.
  --stage r2_1 [--round r]  R2.1 from the eleven --screen runs of the round's snapshot.
  --stage r2_2 [--round r]  R2.2: the finalists; prints the full-Dev runs to make (finalists and mvc_r1_nosng).
  --stage r2_3 [--round r]  R2.3 from the --full runs (1,000 pairs; Dev-900 = the 900 outside Dev-S100).
  --stage r2_4 [--round r]  R2.4 from the --budget runs (3 s, 180 s); 'pending' until the runs the rule calls for exist.
  --stage b1                B1's keep rule from --b1_run (only when B1 is triggered).
  --stage record [--final]  everything again, then the record. Without --final: a draft, placeholders listed, no env
                            file. With --final: refuses unless every operator item is given and nothing is missing.
--round b1 is the second R2 on B1's final EMA snapshot (only when B1 is kept); R2.0 is not repeated.

Inputs (copied from ceai to the Mac; nothing is computed on ceai):
  --eval_root   the Mac copy of $MVC/eval (= /datastore/ceai/crt/mimi_vc/data_storage/mvc/eval), same layout. A run is
                named by its folder relative to it, TAG/NAME[_bBUDGET] as eval_dev.sh writes it; the script reads
                metrics/dev_<B>.jsonl, metrics/dev_<B>_src.jsonl, ecapa/dev_<B>{,_src}.jsonl (screen and full runs) and
                conv/dev/<B>/<NAME>/runs.jsonl (the describe() commit, the settings, the snapshot and its sha256).
                A reconstruction run (--recon) is TAG/NAME with metrics.jsonl (eval_dev.sh recon_eval).
  --sha256      sha256sum lists made on ceai (paths absolute or relative to $MVC/eval). Every file read under
                --eval_root must be listed with the same sha256, or the script refuses.
  --pairs       $J/pairs/dev.jsonl (sha256 as in mvc/FREEZE_mvc_v1.sha256, or the journal's Mac copy V1/pairs/dev.jsonl).
  --target_copy_src  the journal's Dev target_copy@src rows, $J/v1/d2/controls/metrics_src.jsonl (or V1/d2/
                controls_metrics_src.jsonl); both known sha256 values are pinned below.
  mvc/pairs/dev_s100.txt is read from the repository (sha256 as in FREEZE_mvc_v1.sha256).
Every run is checked before its numbers are used: one row per pair in each file (100 = Dev-S100 or 1,000 = full Dev,
the same pairs in every file of the run), no pair twice, rows of the run's own system, set dev and budget; every metrics
row carries the run's conv_cfg_sha; the settings equal the §2.1 JSON with only the step's values changed (only R2.0's
150k run may carry a stats_path, as job e5's did); every run of a round uses the round's snapshot with one sha256 and one
describe() commit. Error rows are counted and left out. Run identity comes from the settings (CFG, SNG w, neg_src_until;
run_dir and ckpt give the round), never from names; the reconstruction runs, which carry no settings, are bound to their
snapshot by their system name r2_0_k150_rec / r2_0_k200_rec (mvc/jobs/after_r2_0.sh).
Code identity (§2.1): the R1 round (R2.0, R2.1-R2.4 on R1, B1's parent) must carry FROZEN_COMMIT, the describe() hash of
the 24 mvc/*.py of FREEZE_mvc_v1.sha256 (checked at start when the Mac files are as frozen). B1 and its second R2 carry one
commit of their own, which may differ when B1 needed a change to mvc/*.py. The record's DESCRIBE_COMMIT is the final
round's commit; --final refuses unless the Mac mvc/*.py give that commit, and lists every mvc/*.py changed since the freeze,
whose new sha256 must appear in mvc/DEVIATIONS_mvc_v1.md (§10).
TAGs (§4.1): every run and reconstruction used must sit under a TAG that no job before the freeze used (PRE_R2_TAGS, the
TAGs under $MVC/eval at the freeze); the one exception is job e5's e5/r1k150_ns1.0 as R2.0's 150k run (§4.2).

Quantities (§4.1): WER = capped corpus WER (stats 'wer_capped', Whisper-large-v3) over the wer_primary pairs; SIM, UTMOS
= means over all pairs; NL_g = stats.compare(X@src, target_copy@src, the stratum's pairs with P, 'sim_wavlm', linked,
worst='max') (§3.3; never stats.gate_l_strata), est and se, with target_copy failing on none of them; the summarize.py
value (X's mean SIM_src over the stratum minus target_copy's over its pairs with P, summarize.py:97-103) is reported and
decides nothing; condition 5 as job e3's rule5.py computed it: the cross-gender mean per-pair WavLM ΔSIM_src (c - c0) over
the pairs where both WavLM @src rows are usable and the mean per-pair ECAPA ΔSIM_src over the pairs where both ECAPA @src
rows are usable, each on its own pairs; met when the ECAPA mean has the WavLM mean's sign and |ECAPA| >= 0.5 |WavLM|.
Choices where the plan is silent (each also in the record):
  - 'Ties' (§4.1): orderings compare WER at two decimals, SIM and UTMOS at three (Python round); thresholds use the
    unrounded point estimates with a tolerance of 1e-9.
  - R2.2 'ties as in §4.1' and 'the rest by lowest WER' use R2.3's order (WER, higher SIM, higher UTMOS, smaller w,
    smaller CFG, plain before interval). Without a D-mvc-2 pick, at most three other eligible configurations.
  - R2.2 under trigger i (four smallest max_g NL_g) and fallback F break exact NL ties by smaller w, smaller CFG, plain
    before interval (F: after the higher SIM).
  - The D-mvc-2 pick includes w 0 (the smallest w with an eligible CFG 0 entry).
  - In the B1 round, mvc_r1_nosng is B1's snapshot at CFG 0, SNG 0.
  - Conditions 2-4 and the R2.3 WER check compare the configurations' own levels (§4.1); B1's keep rule is paired
    (stats.compare, pairs where neither failed).
  - The Dev-900 repeat orders the same finalists (E3, or F); its first is compared with the 1,000-pair first; R2.4 is
    not repeated on it.
  - A number that cannot be computed counts as not met and stops the stage.
  - The contamination test's result is recorded whatever it is; a failure does not stop the record (§4.8 asks for the
    result only).

Operator items for the record (placeholders in a draft; required with --final): --jobs TAG=JOBID for every R2 and B1 eval
TAG (every TAG of a run used, of a reconstruction, and every TAG after the freeze in the census, also of runs refused or
repeated; one job per TAG; b1_train=JOBID when B1 ran) with --sacct (`sacct -P -X -o JobID,JobName,Submit,End,State`
output); --other_tag TAG=WHAT for a TAG after the freeze that is no R2 or B1 job (such as R1x's Dev-S100 evals, §4.10);
--contamination (JSON {"passed": bool, "manifest": path, "manifest_sha256": hex, "command": str}: mvc/tests/
test_train.py on the training ID manifest of mvc_v1's model); --r1_hashes (sha256sum of $MVC/runs/r1/{last.pt,args.json,
stats.npz} and of every ema_*.pt: ema_0010000.pt ... ema_0200000.pt, every 10k as R1's args.json, and ema_last.pt,
§4.10); --census (Dev configurations ever scored), made on ceai with
  cd $MVC/eval && shopt -s nullglob && for s in */*/summary.json; do d=${s%/summary.json}; for m in $d/metrics/dev_*.jsonl;
  do case $m in *_src.jsonl) continue;; esac; b=${m##*/dev_}; python3 -c 'import sys,json,hashlib; i=sorted({json.loads(l)
  ["pair_id"] for l in open(sys.argv[1]) if l.strip()}); print(sys.argv[2], sys.argv[3], len(i), hashlib.sha256("\\n".join(i)
  .encode()).hexdigest(), sep="\\t")' $m $s ${b%.jsonl}; done; done > census.tsv
and mvc/analysis/analyze_mvc.py with mvc/tests/test_analyze_mvc.py present (their sha256 go into the record).

Example (MR = paper/mimi_vc/results; the arguments can live in a file read with @file):
  .venv-stats/bin/python mvc/analysis/select_mvc.py --stage r2_1 --eval_root MR/r2/eval --sha256 MR/r2/eval.sha256 \\
      --pairs MR/r2/journal/dev.jsonl --target_copy_src MR/r2/journal/controls_metrics_src.jsonl --state_dir MR/r2/state \\
      --ref r2_0/r2_0_k150_ref --ref r2_0/r2_0_k200_ref --recon k150=r2_0rec/r2_0_k150_rec \\
      --recon k200=r2_0rec/r2_0_k200_rec --screen r2_1/<name> (x 11)
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MVC = HERE.parent
REPO = MVC.parent
sys.path.insert(0, str(REPO / 'benchmark'))
sys.path.insert(0, str(MVC))
import stats  # noqa: E402
import vc_adapter  # noqa: E402

STATS_SHA256 = 'a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6'   # plan header
FREEZE_PATH = MVC / 'FREEZE_mvc_v1.sha256'
S100_PATH = MVC / 'pairs' / 'dev_s100.txt'
S100_KEY = 'mvc/pairs/dev_s100.txt'
ANALYZE_MVC = MVC / 'analysis' / 'analyze_mvc.py'
ANALYZE_MVC_TEST = MVC / 'tests' / 'test_analyze_mvc.py'
SELECT_TEST = MVC / 'tests' / 'test_select_mvc.py'
PLAN_KEY = 'mvc/ANALYSIS_PLAN_mvc_v1.md'
CEAI_J = '/datastore/ceai/crt/mimi_vc/benchmark/journal'
CEAI_DEV_PAIRS = CEAI_J + '/pairs/dev.jsonl'
CEAI_LISTS = (CEAI_J + '/pairs/dev.jsonl', CEAI_J + '/pairs/vctk2k.jsonl', CEAI_J + '/pairs/ltc2k.jsonl',
              CEAI_J + '/pairs/vctk2k_source_E.json', '/datastore/ceai/crt/mimi_vc/mvc/pairs/dev_s100.txt')
MAC_DEV_PAIRS_SHA256 = 'ff5cd1147e1236652cfee022c4fdfd7a9bd63ac11bc64989cbdb82a7097ec194'
TARGET_COPY_SRC_SHA256 = {
    '4f11b82e480630697b683ee38c5e76bf30e9a9a231587a9e2d08c78f95262289': 'ceai $J/v1/d2/controls/metrics_src.jsonl',
    '77c17cc5679fa5171f4a93299fdbc7b0fdb0a6727fb30e1ed0f85c1e33638b23':
        'Mac V1/d2/controls_metrics_src.jsonl (the same rows, paths before the 21 Oct move)'}
MVC_DS = '/datastore/ceai/crt/mimi_vc/data_storage/mvc'
R1_DIR = MVC_DS + '/runs/r1'
EVAL_PREFIX = MVC_DS + '/eval/'
SNAP = {'k150': 'ema_0150000.pt', 'k200': 'ema_0200000.pt'}
RECON_NAME = 'r2_0_{}_rec'                                   # mvc/jobs/after_r2_0.sh: recon_eval r2_0_${k}_rec
R1_FILES = ('last.pt', 'args.json', 'stats.npz') + tuple(f'ema_{s:07d}.pt' for s in range(10000, 200001, 10000)) + \
    ('ema_last.pt',)                                         # R1's args.json: steps 200000, snap_every 10000
CODE_DIR = MVC
FROZEN_COMMIT = 'ba09a8efc7678e38'      # vc_adapter.describe() of the 24 mvc/*.py of FREEZE_mvc_v1.sha256
DEVIATIONS = MVC / 'DEVIATIONS_mvc_v1.md'
# The TAGs under $MVC/eval at the freeze (ls on ceai, 27 Oct 2026; the folder last changed at 15:54:07, before 16:28).
PRE_R2_TAGS = frozenset(('e1', 'e1rec', 'e2', 'e3', 'e4', 'e4rec', 'e5', 'floors', 'n1', 'n1pre', 'p6', 'p6sweep', 'p7',
                         'p8', 'p8rec', 'pilot', 'r5', 'r5pre', 'recon'))
E5_REF = 'e5/r1k150_ns1.0'                                   # §4.2: job e5's 150k run, usable for R2.0
CFG_GRID, W_GRID, INTERVAL_W = (0.0, 1.0), (0.0, 0.5, 0.75, 1.0), (0.5, 0.75, 1.0)
PLAIN, INTERVAL = 1.0, 0.5
GRID = tuple([(c, w, PLAIN) for c in CFG_GRID for w in W_GRID] + [(0.0, w, INTERVAL) for w in INTERVAL_W])
REF_CONF, NOSNG_CONF = (0.0, 1.0, PLAIN), (0.0, 0.0, PLAIN)
N_S100, N_FULL, N_RECON = 100, 1000, 300
NL_MARGIN, EPS, R5_LEVEL = 0.03, 1e-9, 3.82 - 0.3
STRATA = tuple(stats.STRATA)                                   # ('same', 'cross')
GENDER_PAIRS = ('F2F', 'F2M', 'M2F', 'M2M')
SECTION_2_1 = dict(steps=16, sway=-1.0, prompt_max=None, prompt_chunk=750, perturb_src=False, decode='latent',
                   seed_from_pair=True, mode='convert', whisper_src='clean', solver='euler', temp=0.0, cfg_rescale=0.0,
                   sharpen=0.0, gv=0.0, stats_path=None, decoder_ckpt=None, init_t=0.0, match_rms=False, relax=0.15,
                   relax_k=2, cfg_branch='std')
VARIED = ('run_dir', 'ckpt', 'cfg', 'neg_src', 'neg_src_until')
if set(SECTION_2_1) | set(VARIED) != set(vc_adapter.DEFAULTS):
    raise ImportError('the §2.1 JSON does not name every key of vc_adapter.DEFAULTS')
STAGES = ('r2_0', 'r2_1', 'r2_2', 'r2_3', 'r2_4')
STAGE_IDS = ('r1/r2_0', 'r1/r2_1', 'r1/r2_2', 'r1/r2_3', 'r1/r2_4', 'b1', 'b1/r2_1', 'b1/r2_2', 'b1/r2_3', 'b1/r2_4')
DNSMOS_NOTE = ('DNSMOS runs on the Mac (§3.7): the 30 s WAVs named by the metric rows are copied read-only from ceai to '
               'MR/wav/ with their absolute path kept below that prefix, and the rows are rewritten to point there; no '
               'audio is converted or edited. Models: $DS/models/dnsmos/ and paper/mimi_vc/models/dnsmos/.')


def die(msg):
    raise SystemExit(f'ERROR: {msg}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def describe_commit():
    """vc_adapter.describe()'s code hash: sha256 over the bytes of every mvc/*.py, sorted, first 16 hex digits."""
    h = hashlib.sha256()
    for f in sorted(CODE_DIR.glob('*.py')):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def code_identity():
    """The Mac mvc/*.py against FREEZE_mvc_v1.sha256 (§2.1, §8.4): the describe() commit they give and the files changed
    since the freeze (new sha256, None when removed). When none changed, the commit must be FROZEN_COMMIT."""
    frozen = {k: v for k, v in read_freeze(FREEZE_PATH).items() if re.fullmatch(r'mvc/[^/]+\.py', k)}
    if not frozen:
        die(f'{FREEZE_PATH} lists no mvc/*.py')
    cur = {f'mvc/{p.name}': sha256(p) for p in CODE_DIR.glob('*.py')}
    changed = {k: cur.get(k) for k in sorted(set(frozen) | set(cur)) if frozen.get(k) != cur.get(k)}
    commit = describe_commit()
    if not changed and commit != FROZEN_COMMIT:
        die(f'the mvc/*.py of FREEZE_mvc_v1.sha256 give describe() commit {commit}, not FROZEN_COMMIT {FROZEN_COMMIT}')
    return {'current': commit, 'frozen': FROZEN_COMMIT, 'changed_since_freeze': changed}


def tag_of(rel):
    return norm_key(rel).split('/', 1)[0]


# ── comparisons, configurations ──────────────────────────────

def _ok(*v):
    return all(x is not None for x in v)


def le(a, b):
    return _ok(a, b) and a <= b + EPS


def ge(a, b):
    return _ok(a, b) and a >= b - EPS


def lt(a, b):
    return _ok(a, b) and a < b - EPS


def gt(a, b):
    return _ok(a, b) and a > b + EPS


def add(a, d):
    return None if a is None else a + d


def diff(a, b):
    return a - b if _ok(a, b) else None


def ckey(c):
    cfg, w, u = c
    return f'cfg{cfg:g}_w{w:g}' + ('' if u == PLAIN else f'_u{u:g}')


def from_key(k):
    m = re.fullmatch(r'cfg([\d.]+)_w([\d.]+)(?:_u([\d.]+))?', k)
    return float(m[1]), float(m[2]), float(m[3]) if m[3] else PLAIN


def interval(c):
    return c[2] != PLAIN


def tie_tail(k):
    c = from_key(k)
    return c[1], c[0], int(interval(c))          # smaller w, smaller CFG, plain before interval


def _r(x, n):
    return math.inf if x is None else round(x, n)


def order_key(nums, k):
    """R2.3's order: the lowest WER (two decimals); ties -> higher SIM, higher UTMOS (three decimals) -> smaller w ->
    smaller CFG -> plain before interval."""
    x = nums[k]
    return (_r(x['wer'], 2), -_r(x['sim'], 3) if x['sim'] is not None else math.inf,
            -_r(x['utmos'], 3) if x['utmos'] is not None else math.inf) + tie_tail(k)


def max_nl(x):
    v = [x['nl'][g]['est'] for g in STRATA]
    return math.inf if any(e is None for e in v) else max(v)


def mvc_json(run_dir, ckpt, conf):
    """The §2.1 JSON with every key of vc_adapter.DEFAULTS, in its order."""
    v = dict(SECTION_2_1, run_dir=run_dir, ckpt=ckpt, cfg=conf[0], neg_src=conf[1], neg_src_until=conf[2])
    return {k: v[k] for k in vc_adapter.DEFAULTS}


def eval_item(rnd, run_dir, ckpt, conf):
    return f'{rnd}_{ckey(conf)}={json.dumps(mvc_json(run_dir, ckpt, conf))}'


class Trace(list):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def __call__(self, msg):
        print(f'[{self.tag}] {msg}', flush=True)
        self.append(msg)


def f(x, n=3, sign=False):
    return '–' if x is None else (f'{x:+.{n}f}' if sign else f'{x:.{n}f}')


# ── inputs ───────────────────────────────────────────────────

def norm_key(p):
    p = p.strip()
    if p.startswith(EVAL_PREFIX):
        p = p[len(EVAL_PREFIX):]
    while p.startswith('./'):
        p = p[2:]
    return p


def read_sha_lists(paths):
    idx = {}
    for path in paths:
        for line in open(path):
            if not line.strip():
                continue
            m = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line.rstrip('\n'))
            if not m:
                die(f'{path}: not a sha256sum line: {line.rstrip()!r}')
            k = norm_key(m[2])
            if idx.get(k, m[1]) != m[1]:
                die(f'{path}: {k} is listed with two different sha256 values')
            idx[k] = m[1]
    return idx


def read_freeze(path):
    out = {}
    for line in open(path):
        m = re.fullmatch(r'([0-9a-f]{64})  (\S.*)', line.rstrip('\n'))
        if m:
            out[m[2]] = m[1]
    return out


class Inputs:
    """The Mac copy of $MVC/eval and the ceai sha256 lists; every file read is checked and remembered."""

    def __init__(self, root, sha_lists):
        self.root = root
        self.index = read_sha_lists(sha_lists)
        self.read = {}

    def path(self, rel):
        return os.path.join(self.root, rel)

    def checked(self, rel):
        key = norm_key(rel)
        p = self.path(key)
        if not os.path.isfile(p):
            die(f'{p} missing')
        h = sha256(p)
        want = self.index.get(key)
        if want is None:
            die(f'{key}: not in any --sha256 list made on ceai')
        if h != want:
            die(f'{key}: sha256 {h} differs from the ceai value {want} (copy it again)')
        self.read[key] = h
        return p


class Dev:
    def __init__(self, pairs, s100_ids, tc):
        self.pairs = pairs
        self.order = [p['pair_id'] for p in pairs]
        self.info = {p['pair_id']: p for p in pairs}
        self.spk = {k: (p['source_speaker'], p['target_speaker']) for k, p in self.info.items()}
        s = set(s100_ids)
        self.sets = {'s100': [k for k in self.order if k in s], 'full': list(self.order),
                     'd900': [k for k in self.order if k not in s]}
        self.ids = {k: set(v) for k, v in self.sets.items()}
        self.tc = tc

    def g(self, k):
        return stats.stratum(self.info[k]['genders'])

    def wer(self, ids):
        return [k for k in ids if self.info[k]['wer_primary']]

    def stratum(self, ids, g):
        return [k for k in ids if self.g(k) == g]

    def par(self, ids, g):
        return [k for k in ids if self.info[k].get('P') and self.g(k) == g]

    def gp(self, ids, gp):
        return [k for k in ids if self.info[k]['genders'] == gp]


def load_dev(pairs_path, tc_path):
    fr = read_freeze(FREEZE_PATH)
    ins = {}
    h = sha256(pairs_path)
    if h not in (fr.get(CEAI_DEV_PAIRS), MAC_DEV_PAIRS_SHA256):
        die(f'{pairs_path}: sha256 {h} is neither the ceai Dev list of FREEZE_mvc_v1.sha256 nor the journal Mac copy')
    ins['pairs'] = {'sha256': h}
    paths = {'pairs': os.path.abspath(pairs_path)}
    pairs = [json.loads(x) for x in open(pairs_path) if x.strip()]
    ids = [p['pair_id'] for p in pairs]
    if len(pairs) != N_FULL or len(set(ids)) != N_FULL or {p['set'] for p in pairs} != {'dev'}:
        die(f'{pairs_path}: not the 1,000-pair Dev list with unique pair_ids')
    h = sha256(S100_PATH)
    if h != fr.get(S100_KEY):
        die(f'{S100_PATH}: sha256 {h} differs from FREEZE_mvc_v1.sha256')
    ins['dev_s100'] = {'sha256': h}
    paths['dev_s100'] = str(S100_PATH)
    s100 = [x.strip() for x in open(S100_PATH) if x.strip()]
    if len(s100) != N_S100 or len(set(s100)) != N_S100 or not set(s100) <= set(ids):
        die(f'{S100_PATH}: not 100 distinct Dev pair_ids')
    h = sha256(tc_path)
    if h not in TARGET_COPY_SRC_SHA256:
        die(f'{tc_path}: sha256 {h} is not the journal\'s Dev controls metrics_src.jsonl')
    ins['target_copy_src'] = {'sha256': h}
    paths['target_copy_src'] = os.path.abspath(tc_path)
    paths['target_copy_src_what'] = TARGET_COPY_SRC_SHA256[h]
    tc = {}
    for x in open(tc_path):
        if not x.strip():
            continue
        r = json.loads(x)
        if r.get('system') != 'target_copy@src':
            continue
        if r['pair_id'] in tc:
            die(f'{tc_path}: target_copy@src row {r["pair_id"]} twice')
        tc[r['pair_id']] = r
    info = {p['pair_id']: p for p in pairs}
    par = [k for k in ids if info[k].get('P')]
    bad = [k for k in par if stats.failed(tc.get(k)) or stats._value(tc[k], 'sim_wavlm') is None]
    if bad:
        die(f'target_copy@src has no usable row on {len(bad)} Dev pairs with P (e.g. {bad[:3]}); NL needs every one')
    ins['target_copy_src']['n_p_pairs'] = len(par)
    return Dev(pairs, s100, tc), ins, paths


class Run:
    """One eval folder: its rows, describe() and settings."""


def read_rows(inp, run, rel, system, dev):
    p = inp.checked(rel)
    rows, n_err = {}, 0
    for i, line in enumerate(open(p), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('system') != system:
            die(f'{rel}: line {i} has system {r.get("system")!r}, not {system!r}')
        if r.get('set', 'dev') != 'dev' or int(r.get('budget', run.budget)) != run.budget:
            die(f'{rel}: line {i} has set {r.get("set")!r}, budget {r.get("budget")!r}')
        k = r.get('pair_id')
        if k not in dev.info:
            die(f'{rel}: pair {k!r} is not in the Dev list')
        if k in rows:
            die(f'{rel}: pair {k} occurs twice (duplicate row); one row per pair (§4.1)')
        rows[k] = r
        n_err += 'error' in r
    run.files[norm_key(rel)] = inp.read[norm_key(rel)]
    run.counts[rel[len(run.rel) + 1:]] = {'rows': len(rows), 'errors': n_err}
    return rows


def load_run(inp, dev, rel, ecapa):
    rel = norm_key(rel).rstrip('/')
    mdir = inp.path(rel + '/metrics')
    if not os.path.isdir(mdir):
        die(f'{rel}: no metrics/ folder under --eval_root')
    bs = sorted({int(m[1]) for x in os.listdir(mdir) for m in [re.fullmatch(r'dev_(\d+)\.jsonl', x)] if m})
    if len(bs) != 1:
        die(f'{rel}: metrics/ holds the budgets {bs}; a run has one')
    r = Run()
    r.rel, r.budget, r.files, r.counts = rel, bs[0], {}, {}
    base = rel.rsplit('/', 1)[-1]
    r.name = base[:-len(f'_b{r.budget}')] if r.budget != 30 and base.endswith(f'_b{r.budget}') else base
    r.tgt = read_rows(inp, r, f'{rel}/metrics/dev_{r.budget}.jsonl', r.name, dev)
    r.src = read_rows(inp, r, f'{rel}/metrics/dev_{r.budget}_src.jsonl', r.name + '@src', dev)
    r.etgt = r.esrc = None
    if ecapa:
        r.etgt = read_rows(inp, r, f'{rel}/ecapa/dev_{r.budget}.jsonl', r.name, dev)
        r.esrc = read_rows(inp, r, f'{rel}/ecapa/dev_{r.budget}_src.jsonl', r.name + '@src', dev)
    ids = set(r.tgt)
    r.pairset = 's100' if ids == dev.ids['s100'] else 'full' if ids == dev.ids['full'] else None
    if r.pairset is None:
        die(f'{rel}: {len(ids)} rows, not one per pair of Dev-S100 ({N_S100}) or full Dev ({N_FULL}): the run is '
            f'incomplete; complete it or repeat it under a new TAG (§4.1)')
    for name, rows in (('@src', r.src), ('ECAPA', r.etgt), ('ECAPA @src', r.esrc)):
        if rows is not None and set(rows) != ids:
            die(f'{rel}: the {name} rows cover {len(rows)} pairs, not the {len(ids)} pairs of its target rows: one row '
                f'per pair in every file (§4.1). lab_rows.py writes no @src row for a failed conversion, so the run is '
                f'incomplete: complete it (run_mvc.py retries error rows) or repeat it under a new TAG')
    rj = f'{rel}/conv/dev/{r.budget}/{r.name}/runs.jsonl'
    p = inp.checked(rj)
    r.files[norm_key(rj)] = inp.read[norm_key(rj)]
    st = [e for e in (json.loads(x) for x in open(p) if x.strip()) if e.get('event') == 'start']
    if not st or len({e['conv_cfg_sha'] for e in st}) != 1:
        die(f'{rj}: not one conv_cfg (start events {len(st)})')
    r.conv_cfg_sha = st[-1]['conv_cfg_sha']
    d = st[-1]['conv_cfg']['describe']
    r.commit = d.get('commit')
    ck = d.get('checkpoints')
    if not isinstance(ck, dict) or len(ck) != 1:
        die(f'{rj}: describe() names {ck!r}, not one checkpoint with its sha256')
    (r.ckpt_path, r.ckpt_sha), = ck.items()
    s = d.get('settings') or {}
    r.settings = {k: v for k, v in s.items() if k not in ('train_step', 'train_args')}
    r.train_step, r.train_args = s.get('train_step'), s.get('train_args')
    try:
        r.conf = (float(s['cfg']), float(s['neg_src']), float(s['neg_src_until']))
    except (KeyError, TypeError, ValueError):
        die(f'{rj}: settings lack cfg, neg_src or neg_src_until')
    r.run_dir, r.ckpt = s.get('run_dir'), s.get('ckpt')
    if r.run_dir == R1_DIR and r.commit != FROZEN_COMMIT:
        die(f'{rel}: describe() commit {r.commit}, but the frozen mvc/*.py (FREEZE_mvc_v1.sha256) give '
            f'{FROZEN_COMMIT}: run it again with the frozen code (§2.1, §4.2)')
    for rows, what in ((r.tgt, 'metrics'), (r.src, 'metrics @src')):
        bad = [k for k, x in rows.items() if x.get('conv_cfg_sha') != r.conv_cfg_sha]
        if bad:
            die(f'{rel}: {len(bad)} {what} rows do not carry the conv_cfg_sha {r.conv_cfg_sha} of runs.jsonl')
    return r


def check_settings(r, run_dir, ckpt, allow_stats=False):
    want = mvc_json(run_dir, ckpt, r.conf)
    bad = [(k, r.settings.get(k), want[k]) for k in want
           if r.settings.get(k) != want[k] and not (allow_stats and k == 'stats_path')]
    extra = sorted(set(r.settings) - set(want))
    if bad or extra:
        die(f'{r.rel}: settings differ from the §2.1 JSON at {ckey(r.conf)} (key, run, plan): {bad}; unknown keys '
            f'{extra} (§4.1)')


def load_recon(inp, rel):
    rel = norm_key(rel).rstrip('/')
    p = inp.checked(rel + '/metrics.jsonl')
    rows, systems = {}, set()
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if r['pair_id'] in rows:
            die(f'{rel}/metrics.jsonl: {r["pair_id"]} twice (duplicate row)')
        rows[r['pair_id']] = r
        systems.add(r.get('system'))
    if len(systems) != 1 or len(rows) != N_RECON:
        die(f'{rel}/metrics.jsonl: {len(rows)} rows of {sorted(map(str, systems))}, not the 300 dev-clean '
            f'reconstructions of one system')
    return {'rel': rel, 'rows': rows, 'system': systems.pop(), 'errors': sum('error' in r for r in rows.values()),
            'files': {norm_key(rel + '/metrics.jsonl'): inp.read[norm_key(rel + '/metrics.jsonl')]}}


class Ctx:
    def __init__(self, inp, dev, dev_inputs, code, refs=(), recon=None, screen=(), full=(), budget=(), b1run=None,
                 dev_paths=None):
        self.inp, self.dev, self.dev_inputs, self.code, self.dev_paths = inp, dev, dev_inputs, code, dev_paths or {}
        self.refs, self.recon, self.screen, self.full, self.budget = list(refs), recon or {}, list(screen), \
            list(full), list(budget)
        self.b1run = b1run
        self.used = set()
        for group, key in ((self.refs, lambda r: r.ckpt), (self.screen, lambda r: (r.run_dir, r.ckpt, r.conf)),
                           (self.full, lambda r: (r.run_dir, r.ckpt, r.conf)),
                           (self.budget, lambda r: (r.run_dir, r.ckpt, r.conf, r.budget))):
            seen = {}
            for r in group:
                if key(r) in seen:
                    die(f'{seen[key(r)]} and {r.rel} are runs of the same configuration; one must be left out')
                seen[key(r)] = r.rel


def check_new_tags(refs, others, recon):
    """§4.1: every R2 job uses a new TAG that no earlier job used; job e5's 150k run is the one pre-R2 run R2.0 may use
    (§4.2)."""
    bad = [r.rel for r in refs if tag_of(r.rel) in PRE_R2_TAGS and r.rel != E5_REF]
    bad += [r.rel for r in others if tag_of(r.rel) in PRE_R2_TAGS]
    bad += [rc['rel'] for rc in recon.values() if tag_of(rc['rel']) in PRE_R2_TAGS]
    if bad:
        die(f'{bad}: under a TAG that a job before the freeze used ({sorted(PRE_R2_TAGS)}); every R2 job uses a new TAG '
            f'under $MVC/eval that no earlier job used (§4.1)')


def load_context(a):
    inp = Inputs(a.eval_root, a.sha256)
    dev, dev_inputs, dev_paths = load_dev(a.pairs, a.target_copy_src)
    code = code_identity()
    recon = {}
    for x in a.recon:
        k, _, rel = x.partition('=')
        if k not in SNAP or not rel or k in recon:
            die(f'--recon {x!r}: give k150=RUN and k200=RUN once each')
        recon[k] = load_recon(inp, rel)
        if recon[k]['system'] != RECON_NAME.format(k):
            die(f'--recon {x}: its rows are of system {recon[k]["system"]!r}, not {RECON_NAME.format(k)!r}, the name '
                f'mvc/jobs/after_r2_0.sh gives the reconstruction with {SNAP[k]}; a reconstruction carries no settings, '
                f'so the name is its only link to the snapshot')
    ctx = Ctx(inp, dev, dev_inputs, code,
              refs=[load_run(inp, dev, x, False) for x in a.ref], recon=recon,
              screen=[load_run(inp, dev, x, True) for x in a.screen],
              full=[load_run(inp, dev, x, True) for x in a.full],
              budget=[load_run(inp, dev, x, False) for x in a.budget],
              b1run=load_run(inp, dev, a.b1_run, False) if a.b1_run else None, dev_paths=dev_paths)
    check_new_tags(ctx.refs, ctx.screen + ctx.full + ctx.budget + ([ctx.b1run] if ctx.b1run else []), recon)
    return ctx


# ── quantities ───────────────────────────────────────────────

def valid(row, key):
    return row is not None and not stats.failed(row) and stats._value(row, key) is not None


def num(dev, run, ids):
    """§4.1 point estimates over ids: capped WER over the wer_primary pairs, mean SIM and UTMOS; failures counted."""
    return {'wer': stats.point(run.tgt, dev.wer(ids), 'wer_capped')[0],
            'sim': stats.point(run.tgt, ids, 'sim_wavlm')[0], 'utmos': stats.point(run.tgt, ids, 'utmos')[0],
            'n': len(ids), 'fail': sum(stats.failed(run.tgt.get(k)) for k in ids)}


def summarize_value(dev, run, ids, g):
    """summarize.py's sim_src_minus_target_copy (l. 97-103): the run's mean SIM_src over the stratum's pairs without
    error minus target_copy's mean over those of them with P. Reported; it decides nothing."""
    ok = [k for k in dev.stratum(ids, g) if k in run.tgt and 'error' not in run.tgt[k]]
    xs = [run.src[k]['sim_wavlm'] for k in ok if run.src.get(k, {}).get('sim_wavlm') is not None]
    ts = [dev.tc[k]['sim_wavlm'] for k in ok if dev.tc.get(k, {}).get('sim_wavlm') is not None]
    return float(np.mean(xs) - np.mean(ts)) if xs and ts else None


def nl(dev, run, ids, g):
    """NL_g (§3.3): paired on the stratum's pairs with P, against the journal's Dev target_copy@src rows, linked
    crossed bootstrap; target_copy must fail on none of those pairs."""
    d = stats.compare(run.src, dev.tc, dev.par(ids, g), dev.spk, 'sim_wavlm', mode=stats.LINKED, worst='max')
    if d['fail_b']:
        die(f'target_copy@src fails on {d["fail_b"]} pairs with P: a worst case must come only from {run.rel}')
    out = {k: d.get(k) for k in ('est', 'se', 'lo', 'hi', 'n', 'n_pairs', 'fail_a', 'fail_b', 'undefined',
                                 'redraws')}
    wc = d.get('worst_case')
    if wc:
        out['worst_case'] = {k: wc.get(k) for k in ('est', 'se', 'lo', 'hi', 'n', 'imputed_a')}
    out['summarize'] = summarize_value(dev, run, ids, g)
    return out


def mean_delta(x, x0, ids, key):
    """Mean per-pair x - x0 of `key` over the ids where both rows are usable (no error, a finite value), and their count."""
    ks = [k for k in ids if valid(x.get(k), key) and valid(x0.get(k), key)]
    return (float(np.mean([x[k][key] - x0[k][key] for k in ks])) if ks else None), len(ks)


def delta5(dev, run, run0, ids):
    """Condition 5 as job e3 computed it (its rule5.py): per stratum, the mean per-pair ΔSIM_src (run - run0) of WavLM
    over the pairs where both WavLM @src rows are usable, and that of ECAPA over the pairs where both ECAPA @src rows are
    usable, each on its own pairs; met when the cross-gender ECAPA mean has the WavLM mean's sign and at least half its
    size. Same-gender and target-side (ΔSIM) means are recorded beside it."""
    out = {}
    for g in STRATA + ('target',):
        if g == 'target':
            a, a0, e, e0, sub = run.tgt, run0.tgt, run.etgt, run0.etgt, list(ids)
        else:
            a, a0, e, e0, sub = run.src, run0.src, run.esrc, run0.esrc, dev.stratum(ids, g)
        dw, nw = mean_delta(a, a0, sub, 'sim_wavlm')
        de, ne = mean_delta(e, e0, sub, 'sim_ecapa')
        out[g] = {'wavlm': dw, 'ecapa': de, 'ratio': de / dw if _ok(dw, de) and dw else None, 'n_wavlm': nw,
                  'n_ecapa': ne}
    c = out['cross']
    out['met'] = bool(_ok(c['wavlm'], c['ecapa']) and np.sign(c['ecapa']) == np.sign(c['wavlm'])
                      and abs(c['ecapa']) >= 0.5 * abs(c['wavlm']) - EPS)
    return out


# ── the rules (pure: numbers in, decisions out) ──────────────

def r2_0_rule(a, b):
    """§4.2: a = 150k, b = 200k numbers (wer, sim, utmos, recon_wer). Returns (conditions, snapshot key, R1x)."""
    cond = {'recon_wer': gt(b['recon_wer'], add(a['recon_wer'], 1.0)), 'sim': lt(b['sim'], add(a['sim'], -0.010)),
            'utmos': lt(b['utmos'], add(a['utmos'], -0.05)), 'wer': gt(b['wer'], add(a['wer'], 2.0))}
    r1x = {'sim_rose': gt(b['sim'], a['sim']), 'utmos_rose': gt(b['utmos'], a['utmos'])}
    r1x['runs'] = r1x['sim_rose'] or r1x['utmos_rose']
    return cond, ('k150' if any(cond.values()) else 'k200'), r1x


def screen_rule(nums, d5):
    """§4.3. nums: {key: {wer, sim, utmos, nl: {same, cross: {est}}}} for the eleven configurations; d5: {key: {met}}
    for the w > 0 configurations. Returns flags, eligibility (with the reason), the D-mvc-2 pick, trigger i."""
    flags, elig = {}, {}
    for c in GRID:
        k, x = ckey(c), nums[ckey(c)]
        c1 = all(le(x['nl'][g]['est'], NL_MARGIN) for g in STRATA)
        if c[1] == 0:
            flags[k] = {'1': c1, '2': True, '3': True, '4': True, '5': True}
            continue
        x0 = nums[ckey((c[0], 0.0, PLAIN))]
        flags[k] = {'1': c1, '2': ge(x['sim'], add(x0['sim'], -0.005)), '3': ge(x['utmos'], add(x0['utmos'], -0.08)),
                    '4': le(diff(x['wer'], x0['wer']), 2.0), '5': bool(d5[k]['met'])}
    for c in GRID:
        k, fl = ckey(c), flags[ckey(c)]
        if c[1] == 0:
            elig[k] = (fl['1'], 'w 0: condition 1' + ('' if fl['1'] else ' fails'))
        elif not interval(c):
            bad = [i for i, v in fl.items() if not v]
            elig[k] = (not bad, 'conditions 1-5 met' if not bad else f'fails {", ".join(bad)}')
        else:
            fp = flags[ckey((0.0, c[1], PLAIN))]
            only3 = not fp['3'] and all(fp[i] for i in '1245')
            bad = [i for i, v in fl.items() if not v]
            elig[k] = (only3 and not bad,
                       ('replaces the plain cell (it fails condition 3 only); ' if only3 else
                        'not eligible: the plain cell does not fail condition 3 alone; ') +
                       ('conditions 1-5 met' if not bad else f'fails {", ".join(bad)}'))
    eligible = [ckey(c) for c in GRID if elig[ckey(c)][0]]
    zero = [from_key(k) for k in eligible if from_key(k)[0] == 0.0]
    pick = ckey(min(zero, key=lambda c: (c[1], int(interval(c))))) if zero else None
    return {'flags': flags, 'eligibility': {k: {'eligible': v[0], 'why': v[1]} for k, v in elig.items()},
            'eligible': eligible, 'pick': pick, 'trigger_i': not eligible}


def finalists_rule(nums, eligible, pick):
    """§4.4 (and §4.3's last bullet when nothing is eligible)."""
    if not eligible:
        fin = sorted(nums, key=lambda k: (max_nl(nums[k]),) + tie_tail(k))[:4]
        return {'finalists': fin, 'how': 'no eligible configuration (trigger i): the four smallest Dev-S100 max_g NL_g'}
    best_sim = max((nums[k]['sim'] for k in eligible if nums[k]['sim'] is not None), default=None)
    best_ut = max((nums[k]['utmos'] for k in eligible if nums[k]['utmos'] is not None), default=None)
    others = [k for k in eligible if k != pick]
    bounded = [k for k in others if ge(nums[k]['sim'], add(best_sim, -0.010))
               and ge(nums[k]['utmos'], add(best_ut, -0.05))]
    take = sorted(bounded, key=lambda k: order_key(nums, k))[:3]
    filled = []
    if len(take) < 3:
        filled = sorted([k for k in others if k not in bounded], key=lambda k: order_key(nums, k))[:3 - len(take)]
    return {'finalists': ([pick] if pick else []) + take + filled, 'best_sim': best_sim, 'best_utmos': best_ut,
            'within_bounds': bounded, 'taken_by_bounds': take, 'taken_by_wer': filled,
            'how': ('the D-mvc-2 pick ' + (pick or '(none)') + ', then the eligible configurations within SIM >= best '
                    '- 0.010 and UTMOS >= best - 0.05 by R2.3\'s order' +
                    (', the rest by the lowest WER' if filled else ''))}


def choice_rule(nums, finalists, nosng_wer):
    """§4.5 on one pair set: L-pass, the best sets, the WER check, E3 and its order, fallback F."""
    per = {}
    for k in finalists:
        x = nums[k]
        per[k] = {'l_pass': all(le(x['nl'][g]['est'], diff(NL_MARGIN, x['nl'][g]['se'])) for g in STRATA),
                  'wer_minus_nosng': diff(x['wer'], nosng_wer)}
        per[k]['wer_check'] = le(per[k]['wer_minus_nosng'], 1.0)
    lp = [k for k in finalists if per[k]['l_pass']]
    best_sim = max((nums[k]['sim'] for k in lp if nums[k]['sim'] is not None), default=None)
    best_ut = max((nums[k]['utmos'] for k in lp if nums[k]['utmos'] is not None), default=None)
    for k in finalists:
        p = per[k]
        p['sim_ok'] = ge(nums[k]['sim'], add(best_sim, -0.010))
        p['utmos_ok'] = ge(nums[k]['utmos'], add(best_ut, -0.05))
        p['e3'] = p['l_pass'] and p['sim_ok'] and p['utmos_ok'] and p['wer_check']
    order = sorted([k for k in finalists if per[k]['e3']], key=lambda k: order_key(nums, k))
    fb = None
    if not order:
        top = max((nums[k]['sim'] for k in finalists if nums[k]['sim'] is not None), default=None)
        cand = [k for k in finalists if ge(nums[k]['sim'], add(top, -0.010))]
        if cand:
            fb = min(cand, key=lambda k: (max_nl(nums[k]), -_r(nums[k]['sim'], 3)) + tie_tail(k))
    return {'per': per, 'best_sim': best_sim, 'best_utmos': best_ut, 'order': order, 'fallback_f': fb,
            'first': order[0] if order else fb}


def budget_rule(full, sims):
    """§4.6. full: R2.3's full-Dev result (order, fallback_f, numbers); sims: {key: {3: SIM, 180: SIM}} of the
    configurations with both budget runs. Returns the checks and the choice, or status pending with what is needed."""
    def check(k):
        s30 = full['numbers'][k]['sim']
        s3, s180 = sims[k][3], sims[k][180]
        return {'config': k, 'sim30': s30, 'sim3': s3, 'sim180': s180, 'ok3': ge(s3, add(s30, -0.06)),
                'ok180': ge(s180, add(s30, -0.005)), 'pass': ge(s3, add(s30, -0.06)) and ge(s180, add(s30, -0.005))}
    order, fb, checked = full['order'], full['fallback_f'], []
    if order:
        choice = None
        for k in order[:2]:
            if k not in sims:
                return {'status': 'pending', 'needs': [k], 'checked': checked}
            checked.append(check(k))
            if checked[-1]['pass']:
                choice = k
                break
        failed = choice is None
        return {'status': 'complete', 'source': 'E3', 'checked': checked, 'choice': choice or order[0],
                'r2_4_failed': failed}
    if fb is None:
        return {'status': 'complete', 'source': None, 'checked': [], 'choice': None, 'r2_4_failed': None}
    if fb not in sims:
        return {'status': 'pending', 'needs': [fb], 'checked': []}
    c = check(fb)
    return {'status': 'complete', 'source': 'F', 'checked': [c], 'choice': fb, 'r2_4_failed': not c['pass']}


def triggers_rule(trigger_i, x):
    """§4.7 on the R2 choice or fallback F (x: its full-Dev numbers): trigger ii, NL_g > +0.03 (point estimate) in either
    stratum; B1 if trigger i or ii; the R5 trigger, full-Dev UTMOS below 3.82 - 0.3 = 3.52 (reported only)."""
    trig_ii = any(gt(x['nl'][g]['est'], NL_MARGIN) for g in STRATA)
    return {'trigger_i': trigger_i, 'trigger_ii': trig_ii, 'b1_triggered': trigger_i or trig_ii,
            'r5': lt(x['utmos'], R5_LEVEL)}


def b1_rule(d):
    """§4.7 keep rule on B1 - parent (paired point estimates)."""
    fl = {'sim_src_same': le(d['sim_src_same'], -0.02), 'sim_src_cross': le(d['sim_src_cross'], -0.02),
          'sim': ge(d['sim'], -0.005), 'utmos': ge(d['utmos'], -0.05), 'wer': le(d['wer'], 0.5)}
    return fl, all(fl.values())


# ── stages ───────────────────────────────────────────────────

def _need_numbers(x, what):
    miss = [k for k in ('wer', 'sim', 'utmos') if x.get(k) is None]
    miss += [f'NL {g}' for g in STRATA if 'nl' in x and x['nl'][g]['est'] is None]
    if miss:
        die(f'{what}: {", ".join(miss)} cannot be computed (every pair failed?)')


def stage_r2_0(ctx):
    T = Trace('R2.0')
    T('§4.2: ema_0150000.pt against ema_0200000.pt at the reference setting (temp 0, relax 0.15 x 2, CFG 0, SNG 1.0) '
      'on Dev-S100, and dev-clean reconstruction')
    inv = {v: k for k, v in SNAP.items()}
    refs = {}
    for r in ctx.refs:
        k = inv.get(r.ckpt)
        if r.run_dir != R1_DIR or k is None:
            die(f'{r.rel}: --ref is R1\'s {list(SNAP.values())}, not {r.run_dir}/{r.ckpt}')
        if r.pairset != 's100' or r.budget != 30 or r.conf != REF_CONF:
            die(f'{r.rel}: R2.0 runs Dev-S100 at 30 s at CFG 0, SNG 1.0 (got {r.pairset}, {r.budget} s, '
                f'{ckey(r.conf)})')
        check_settings(r, R1_DIR, SNAP[k], allow_stats=(k == 'k150'))
        refs[k] = r
        ctx.used.add(r.rel)
    miss = [SNAP[k] for k in SNAP if k not in refs] + [f'reconstruction {k}' for k in SNAP if k not in ctx.recon]
    if miss:
        die(f'R2.0 needs {miss} (--ref, --recon); the reference-setting JSONs: ' +
            '; '.join(eval_item('r2_0', R1_DIR, SNAP[k], REF_CONF) for k in SNAP if k not in refs))
    if set(ctx.recon['k150']['rows']) != set(ctx.recon['k200']['rows']):
        die('the two reconstruction runs cover different utterances')
    n, files = {}, {}
    for k in SNAP:
        r, rc = refs[k], ctx.recon[k]
        x = num(ctx.dev, r, ctx.dev.sets['s100'])
        x['recon_wer'] = stats.point(rc['rows'], sorted(rc['rows']), 'wer_capped')[0]
        _need_numbers(x, r.rel)
        if x['recon_wer'] is None:
            die(f'{rc["rel"]}: no reconstruction WER')
        x.update(run=r.rel, snapshot=r.ckpt_path, snapshot_sha256=r.ckpt_sha, counts=r.counts, recon=rc['rel'],
                 recon_errors=rc['errors'], stats_path=r.settings.get('stats_path'))
        files.update(r.files)
        files.update(rc['files'])
        n[k] = x
        T(f'{SNAP[k]} ({r.rel}; sha256 {r.ckpt_sha}): WER {f(x["wer"], 2)} %, SIM {f(x["sim"])}, UTMOS '
          f'{f(x["utmos"])}, failed {x["fail"]}; reconstruction WER {f(x["recon_wer"], 2)} % ({rc["rel"]}, '
          f'{rc["errors"]} errors)')
    cond, snap, r1x = r2_0_rule(n['k150'], n['k200'])
    T(f'conditions for 150k: recon WER(200k) > recon WER(150k) + 1.0: {cond["recon_wer"]}; SIM(200k) < SIM(150k) - '
      f'0.010: {cond["sim"]}; UTMOS(200k) < UTMOS(150k) - 0.05: {cond["utmos"]}; WER(200k) > WER(150k) + 2.0: '
      f'{cond["wer"]}')
    T(f'snapshot: {SNAP[snap]} (sha256 {n[snap]["snapshot_sha256"]})')
    T(f'R1x (§4.10): SIM rose {r1x["sim_rose"]}, UTMOS rose {r1x["utmos_rose"]} -> R1x '
      f'{"runs" if r1x["runs"] else "does not run"}')
    return {'stage': 'r2_0', 'commit': FROZEN_COMMIT, 'dev_inputs': ctx.dev_inputs, 'numbers': n, 'conditions': cond,
            'snapshot': {'key': snap, 'run_dir': R1_DIR, 'ckpt': SNAP[snap], 'path': n[snap]['snapshot'],
                         'sha256': n[snap]['snapshot_sha256'], 'commit': FROZEN_COMMIT}, 'r1x': r1x, 'inputs': files,
            'trace': T}


def _round_runs(group, rnd, snap):
    return [r for r in group if (r.run_dir == R1_DIR) == (rnd == 'r1')]


def _check_round_run(r, snap, what):
    if r.run_dir != snap['run_dir'] or r.ckpt != snap['ckpt'] or r.ckpt_sha != snap['sha256']:
        die(f'{r.rel}: {what} uses {r.run_dir}/{r.ckpt} (sha256 {r.ckpt_sha}), not the round\'s snapshot '
            f'{snap["run_dir"]}/{snap["ckpt"]} ({snap["sha256"]})')
    if r.commit != snap['commit']:
        die(f'{r.rel}: {what} carries describe() commit {r.commit}, not the round\'s {snap["commit"]}: one code for '
            f'the whole round (§2.1)')
    check_settings(r, snap['run_dir'], snap['ckpt'])


def stage_r2_1(ctx, rnd, snap):
    T = Trace(f'{rnd} R2.1')
    T(f'§4.3: Dev-S100 screen on {snap["run_dir"]}/{snap["ckpt"]}: eight plain cells and three interval variants')
    by = {}
    for r in _round_runs(ctx.screen, rnd, snap):
        if r.conf not in GRID:
            die(f'{r.rel}: {ckey(r.conf)} is not one of the eleven R2.1 configurations')
        if r.pairset != 's100' or r.budget != 30:
            die(f'{r.rel}: the screen runs Dev-S100 at 30 s (got {r.pairset}, {r.budget} s)')
        _check_round_run(r, snap, 'the screen run')
        by[ckey(r.conf)] = r
    miss = [c for c in GRID if ckey(c) not in by]
    if miss:
        die(f'R2.1 needs Dev-S100 runs (with ECAPA) of {[ckey(c) for c in miss]}: ' +
            '; '.join(eval_item(f'r2_1{"b1" if rnd == "b1" else ""}', snap['run_dir'], snap['ckpt'], c)
                      for c in miss))
    ids = ctx.dev.sets['s100']
    nums, d5, files = {}, {}, {}
    for c in GRID:
        k, r = ckey(c), by[ckey(c)]
        ctx.used.add(r.rel)
        files.update(r.files)
        nums[k] = dict(num(ctx.dev, r, ids), nl={g: nl(ctx.dev, r, ids, g) for g in STRATA}, run=r.rel,
                       counts=r.counts)
        _need_numbers(nums[k], r.rel)
    for c in GRID:
        if c[1] > 0:
            d5[ckey(c)] = delta5(ctx.dev, by[ckey(c)], by[ckey((c[0], 0.0, PLAIN))], ids)
    rule = screen_rule(nums, d5)
    for c in GRID:
        k, x = ckey(c), nums[ckey(c)]
        fl, el = rule['flags'][k], rule['eligibility'][k]
        d = d5.get(k)
        T(f'{k} ({x["run"]}): WER {f(x["wer"], 2)}, SIM {f(x["sim"])}, UTMOS {f(x["utmos"])}; NL same '
          f'{f(x["nl"]["same"]["est"], 3, True)} (SE {f(x["nl"]["same"]["se"])}), cross '
          f'{f(x["nl"]["cross"]["est"], 3, True)} (SE {f(x["nl"]["cross"]["se"])}); summarize.py value '
          f'{f(x["nl"]["same"]["summarize"], 3, True)} / {f(x["nl"]["cross"]["summarize"], 3, True)}'
          + (f'; cross ΔSIM_src WavLM {f(d["cross"]["wavlm"], 3, True)}, ECAPA {f(d["cross"]["ecapa"], 3, True)}'
             if d else '') + f'; conditions {" ".join(f"{i}:{int(v)}" for i, v in fl.items())}; eligible '
          f'{el["eligible"]} ({el["why"]})')
    T(f'eligible: {rule["eligible"] or "none"}; D-mvc-2 pick: {rule["pick"] or "none"}; B1 trigger i: '
      f'{rule["trigger_i"]}')
    return {'stage': 'r2_1', 'round': rnd, 'snapshot': snap, 'numbers': nums, 'd5': d5, **rule, 'inputs': files,
            'trace': T}


def stage_r2_2(r21):
    T = Trace(f'{r21["round"]} R2.2')
    nums = r21['numbers']
    res = finalists_rule(nums, r21['eligible'], r21['pick'])
    T(f'§4.4: {res["how"]}')
    if r21['eligible']:
        T(f'bests over the eligible configurations: SIM {f(res["best_sim"])}, UTMOS {f(res["best_utmos"])}; within '
          f'the bounds: {res["within_bounds"] or "none"}')
    for k in res['finalists']:
        x = nums[k]
        T(f'finalist {k}: WER {f(x["wer"], 2)}, SIM {f(x["sim"])}, UTMOS {f(x["utmos"])}, max_g NL '
          f'{f(max_nl(x), 3, True)}')
    snap = r21['snapshot']
    need = list(dict.fromkeys(res['finalists'] + [ckey(NOSNG_CONF)]))
    T('full-Dev runs to make (1,000 pairs, 30 s, ECAPA), finalists and mvc_r1_nosng: ' +
      '; '.join(eval_item('r2_full' + ('b1' if r21['round'] == 'b1' else ''), snap['run_dir'], snap['ckpt'],
                          from_key(k)) for k in need))
    return {'stage': 'r2_2', 'round': r21['round'], **res, 'full_dev_runs': need, 'trace': T}


def stage_r2_3(ctx, rnd, snap, r22):
    T = Trace(f'{rnd} R2.3')
    fin, nos = r22['finalists'], ckey(NOSNG_CONF)
    need = list(dict.fromkeys(fin + [nos]))
    by = {}
    for r in _round_runs(ctx.full, rnd, snap):
        k = ckey(r.conf)
        if k not in need:
            die(f'{r.rel}: a full-Dev run of {k}, which is neither a finalist nor mvc_r1_nosng ({need})')
        if r.pairset != 'full' or r.budget != 30:
            die(f'{r.rel}: the R2.3 runs are full Dev at 30 s (got {r.pairset}, {r.budget} s)')
        _check_round_run(r, snap, 'the full-Dev run')
        by[k] = r
    miss = [k for k in need if k not in by]
    if miss:
        die(f'R2.3 needs full-Dev runs (with ECAPA) of {miss}: ' +
            '; '.join(eval_item('r2_full', snap['run_dir'], snap['ckpt'], from_key(k)) for k in miss))
    files = {}
    for r in by.values():
        ctx.used.add(r.rel)
        files.update(r.files)
    out = {'stage': 'r2_3', 'round': rnd, 'finalists': fin, 'nosng': nos, 'runs': {k: r.rel for k, r in by.items()},
           'counts': {k: r.counts for k, r in by.items()}}
    for setname, label in (('full', 'full Dev (1,000 pairs)'), ('d900', 'Dev-900 (outside Dev-S100)')):
        ids = ctx.dev.sets[setname]
        nums = {}
        for k in need:
            nums[k] = dict(num(ctx.dev, by[k], ids), nl={g: nl(ctx.dev, by[k], ids, g) for g in STRATA})
            _need_numbers(nums[k], f'{by[k].rel} on {label}')
        res = choice_rule(nums, fin, nums[nos]['wer'])
        T(f'§4.5 on {label}; mvc_r1_nosng ({by[nos].rel}) WER {f(nums[nos]["wer"], 2)}')
        for k in fin:
            x, p = nums[k], res['per'][k]
            T(f'{k}: WER {f(x["wer"], 2)} ({f(p["wer_minus_nosng"], 2, True)} vs nosng, check {p["wer_check"]}), SIM '
              f'{f(x["sim"])}, UTMOS {f(x["utmos"])}; NL same {f(x["nl"]["same"]["est"], 3, True)} (SE '
              f'{f(x["nl"]["same"]["se"])}), cross {f(x["nl"]["cross"]["est"], 3, True)} (SE '
              f'{f(x["nl"]["cross"]["se"])}); L-pass {p["l_pass"]}, SIM ok {p["sim_ok"]}, UTMOS ok {p["utmos_ok"]}, '
              f'E3 {p["e3"]}')
        T(f'best SIM {f(res["best_sim"])}, best UTMOS {f(res["best_utmos"])} (over the L-passing finalists); E3 order: '
          f'{res["order"] or "empty"}' + (f'; fallback F: {res["fallback_f"]}' if not res['order'] else ''))
        out[setname] = dict(res, numbers=nums)
    a, b = out['full']['first'], out['d900']['first']
    out['d900_agrees'] = a == b
    T(f'Dev-900 check: first on 1,000 pairs {a}, on Dev-900 {b}' +
      ('' if a == b else ' -> both are reported; the choice made on 1,000 pairs stands'))
    first = out['full']['order'][:1] or ([out['full']['fallback_f']] if out['full']['fallback_f'] else [])
    T('R2.4 runs to make (full Dev at 3 s and 180 s): ' +
      '; '.join(eval_item('r2_4', snap['run_dir'], snap['ckpt'], from_key(k)) + f' (BUDGETS=3,180)' for k in first))
    out.update(inputs=files, trace=T)
    return out


def stage_r2_4(ctx, rnd, snap, r21, r23):
    T = Trace(f'{rnd} R2.4')
    full, fin = r23['full'], r23['finalists']
    runs = {}
    for r in _round_runs(ctx.budget, rnd, snap):
        k = ckey(r.conf)
        if k not in fin:
            die(f'{r.rel}: a budget run of {k}, which is not a finalist')
        if r.pairset != 'full' or r.budget not in (3, 180):
            die(f'{r.rel}: R2.4 runs full Dev at 3 s and 180 s (got {r.pairset}, {r.budget} s)')
        _check_round_run(r, snap, 'the budget run')
        runs.setdefault(k, {})[r.budget] = r
    ids = ctx.dev.sets['full']
    levels = {k: {b: dict(num(ctx.dev, r, ids), run=r.rel, counts=r.counts) for b, r in v.items()}
              for k, v in runs.items()}
    sims = {k: {b: v[b]['sim'] for b in (3, 180)} for k, v in levels.items() if set(v) >= {3, 180}}
    res = budget_rule(full, sims)
    out = {'stage': 'r2_4', 'round': rnd, 'levels': levels, **res}
    if res['status'] == 'pending':
        T(f'§4.6 pending: the full-Dev runs at 3 s and 180 s of {res["needs"]} are needed: ' +
          '; '.join(eval_item('r2_4', snap['run_dir'], snap['ckpt'], from_key(k)) for k in res['needs']))
        out['trace'] = T
        return out
    files = {}
    for k, v in runs.items():
        for r in v.values():
            ctx.used.add(r.rel)
            files.update(r.files)
    for c in res['checked']:
        T(f'{c["config"]}: SIM 3 s {f(c["sim3"])} >= {f(c["sim30"])} - 0.06: {c["ok3"]}; SIM 180 s {f(c["sim180"])} '
          f'>= {f(c["sim30"])} - 0.005: {c["ok180"]} -> {"passes" if c["pass"] else "fails"}')
    used = {c['config'] for c in res['checked']}
    out['not_used'] = sorted(k for k in runs if k not in used)
    if out['not_used']:
        T(f'budget runs not called for by the rule (recorded, not used): {out["not_used"]}')
    ch = res['choice']
    if ch is None:
        die('R2.3 left no configuration: E3 is empty and no finalist is within 0.010 of the best SIM')
    x = full['numbers'][ch]
    trig = triggers_rule(r21['trigger_i'], x)
    trig_ii, r5 = trig['trigger_ii'], trig['r5']
    out.update(trig, inputs=files)
    if res['source'] == 'E3':
        T(f'choice: {ch} ' + ('(the first of E3\'s order that passes)' if not res['r2_4_failed'] else
                              '(no checked configuration passed: the first of the order; the failure is recorded)'))
    else:
        T(f'E3 is empty: fallback F {ch}; R2.4 for the record: {"passes" if not res["r2_4_failed"] else "fails"}')
    T(f'B1 trigger i (no eligible configuration in R2.1): {r21["trigger_i"]}; trigger ii (NL_g > +0.03 on full Dev: '
      f'same {f(x["nl"]["same"]["est"], 3, True)}, cross {f(x["nl"]["cross"]["est"], 3, True)}): {trig_ii} -> B1 '
      f'{"triggered" if out["b1_triggered"] else "not triggered"}')
    T(f'R5 trigger (reported only; R5 is not run): full-Dev UTMOS {f(x["utmos"])} < {R5_LEVEL:.2f}: {r5}')
    out['trace'] = T
    return out


def stage_b1(ctx, d):
    T = Trace('B1')
    r21, r24 = d['r1/r2_1'], d['r1/r2_4']
    if not r24['b1_triggered']:
        die('B1 is not triggered (§4.7): there is no B1 stage')
    parent_key = ckey(NOSNG_CONF) if r21['trigger_i'] else r24['choice']
    parents = [r for r in ctx.full if r.run_dir == R1_DIR and ckey(r.conf) == parent_key]
    if len(parents) != 1:
        die(f'B1 needs R1\'s full-Dev run of {parent_key} as its parent')
    par, b = parents[0], ctx.b1run
    if b is None:
        die(f'B1 needs --b1_run: B1\'s final EMA snapshot on full Dev at 30 s at the parent setting {parent_key}')
    if b.run_dir in (None, R1_DIR) or b.pairset != 'full' or b.budget != 30 or ckey(b.conf) != parent_key:
        die(f'{b.rel}: B1\'s run must be its own run_dir on full Dev at 30 s at {parent_key} (got {b.run_dir}, '
            f'{b.pairset}, {b.budget} s, {ckey(b.conf)})')
    steps = b.train_args.get('steps') if isinstance(b.train_args, dict) else None
    if not isinstance(b.train_step, int) or b.train_step != steps or b.ckpt != f'ema_{b.train_step:07d}.pt':
        die(f'{b.rel}: B1\'s final EMA snapshot is ema_<its --steps, 7 digits>.pt (§4.7; train.py writes ema_last.pt '
            f'too, which is never named, §4.2); the run uses {b.ckpt} at train step {b.train_step}, and B1\'s --steps '
            f'is {steps}')
    check_settings(b, b.run_dir, b.ckpt)
    ctx.used.add(b.rel)
    dev, ids = ctx.dev, ctx.dev.sets['full']
    cmp = {'sim_src_same': stats.compare(b.src, par.src, dev.stratum(ids, 'same'), dev.spk, 'sim_wavlm', worst='max'),
           'sim_src_cross': stats.compare(b.src, par.src, dev.stratum(ids, 'cross'), dev.spk, 'sim_wavlm',
                                          worst='max'),
           'sim': stats.compare(b.tgt, par.tgt, ids, dev.spk, 'sim_wavlm', worst='min'),
           'utmos': stats.compare(b.tgt, par.tgt, ids, dev.spk, 'utmos', worst='min'),
           'wer': stats.compare(b.tgt, par.tgt, dev.wer(ids), dev.spk, 'wer_capped', n_ref=None)}
    delta = {k: v.get('est') for k, v in cmp.items()}
    if any(v is None for v in delta.values()):
        die(f'B1 keep rule: {[k for k, v in delta.items() if v is None]} cannot be computed')
    flags, kept = b1_rule(delta)
    keep = ('est', 'se', 'lo', 'hi', 'n', 'fail_a', 'fail_b', 'a', 'b', 'p')
    T(f'B1\'s final EMA snapshot {b.run_dir}/{b.ckpt} (train step {b.train_step} of {steps}), describe() commit '
      f'{b.commit}' + ('' if b.commit == FROZEN_COMMIT else f' (the frozen code gives {FROZEN_COMMIT}: B1 changed '
                                                            f'mvc/*.py)'))
    T(f'§4.7 keep rule, B1 ({b.rel}, {b.run_dir}/{b.ckpt}, sha256 {b.ckpt_sha}) - R1 parent ({par.rel}, {parent_key}):'
      f' ΔSIM_src same {f(delta["sim_src_same"], 3, True)}, cross {f(delta["sim_src_cross"], 3, True)} (<= -0.02); '
      f'ΔSIM {f(delta["sim"], 3, True)} (>= -0.005); ΔUTMOS {f(delta["utmos"], 3, True)} (>= -0.05); capped WER '
      f'{f(delta["wer"], 2, True)} (<= +0.5) -> {flags} -> B1 {"kept" if kept else "not kept"}')
    return {'stage': 'b1', 'parent': {'config': parent_key, 'run': par.rel}, 'run': b.rel,
            'snapshot': {'run_dir': b.run_dir, 'ckpt': b.ckpt, 'path': b.ckpt_path, 'sha256': b.ckpt_sha,
                         'train_step': b.train_step, 'commit': b.commit},
            'deltas': {k: {x: v.get(x) for x in keep if x in v} for k, v in cmp.items()}, 'flags': flags,
            'kept': kept, 'inputs': dict(b.files, **par.files), 'trace': T}


def derive(ctx, upto):
    """Every stage up to `upto` (a STAGE_IDS entry, or 'record': every stage that applies), from the files."""
    last = len(STAGE_IDS) if upto == 'record' else STAGE_IDS.index(upto) + 1
    want = set(STAGE_IDS[:last])
    d = {'r1/r2_0': stage_r2_0(ctx)}
    for rnd in ('r1', 'b1'):
        if rnd == 'b1':
            if 'b1' not in want:
                return d
            if not d['r1/r2_4']['b1_triggered']:
                if upto != 'record':
                    die('B1 is not triggered (§4.7): there is no B1 stage or second R2')
                return d
            d['b1'] = stage_b1(ctx, d)
            if 'b1/r2_1' not in want:
                return d
            if not d['b1']['kept']:
                if upto != 'record':
                    die('B1 is not kept (§4.7): there is no second R2')
                return d
            snap = d['b1']['snapshot']
        else:
            snap = {k: d['r1/r2_0']['snapshot'][k] for k in ('run_dir', 'ckpt', 'path', 'sha256', 'commit')}
        if f'{rnd}/r2_1' not in want:
            return d
        d[f'{rnd}/r2_1'] = stage_r2_1(ctx, rnd, snap)
        if f'{rnd}/r2_2' not in want:
            return d
        d[f'{rnd}/r2_2'] = stage_r2_2(d[f'{rnd}/r2_1'])
        if f'{rnd}/r2_3' not in want:
            return d
        d[f'{rnd}/r2_3'] = stage_r2_3(ctx, rnd, snap, d[f'{rnd}/r2_2'])
        if f'{rnd}/r2_4' not in want:
            return d
        d[f'{rnd}/r2_4'] = stage_r2_4(ctx, rnd, snap, d[f'{rnd}/r2_1'], d[f'{rnd}/r2_3'])
        if d[f'{rnd}/r2_4']['status'] != 'complete' and (upto == 'record' or STAGE_IDS.index(upto) >
                                                         STAGE_IDS.index(f'{rnd}/r2_4')):
            die(f'{rnd} R2.4 is pending: it needs the budget runs of {d[f"{rnd}/r2_4"]["needs"]}')
        if rnd == 'b1' and d['b1/r2_4'].get('b1_triggered'):
            d['b1/r2_4']['trace'].append('the second R2 meets a B1 trigger again: nothing more starts; its choice or '
                                         'fallback F is mvc_v1 and §7.2 reports the leakage (§4.7)')
    return d


# ── stage state ──────────────────────────────────────────────

def canon(x):
    return json.dumps(x, sort_keys=True, default=lambda o: o.item() if isinstance(o, np.generic) else list(o))


def normal(x):
    return json.loads(canon(x))


def _strip(x):
    return {k: v for k, v in x.items() if k != 'written'}


def first_difference(a, b, path=''):
    if type(a) is not type(b):
        return f'{path or "/"}: {a!r} vs {b!r}'
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f'{path}/{k}: only in {"the saved" if k in a else "the re-derived"} output'
            r = first_difference(a[k], b[k], f'{path}/{k}')
            if r:
                return r
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{path}: {len(a)} vs {len(b)} items'
        for i, (x, y) in enumerate(zip(a, b)):
            r = first_difference(x, y, f'{path}[{i}]')
            if r:
                return r
        return None
    return None if a == b else f'{path}: saved {a!r}, re-derived {b!r}'


def state_path(state_dir, sid):
    return os.path.join(state_dir, sid.replace('/', '_') + '.json')


def check_saved(state_dir, derived, skip=None):
    for sid, out in derived.items():
        if sid == skip:
            continue
        p = state_path(state_dir, sid)
        if not os.path.exists(p):
            die(f'stage {sid} has no saved output ({p}): run it first')
        saved, new = _strip(json.load(open(p))), _strip(normal(out))
        if saved.get('status') == 'pending':
            die(f'stage {sid} was saved while pending ({p}): run --stage {sid.split("/")[-1]} again first')
        if canon(saved) != canon(new):
            die(f'STOP (§4.1): stage {sid} re-derived from the files differs from its saved output {p}: '
                f'{first_difference(saved, new)}')
        print(f'[check] stage {sid}: re-derived, identical to {p}', flush=True)


def save_stage(state_dir, sid, out):
    p = state_path(state_dir, sid)
    new = normal(out)
    if os.path.exists(p):
        old = json.load(open(p))
        if old.get('status') != 'pending' and canon(_strip(old)) != canon(_strip(new)):
            die(f'STOP (§4.1): stage {sid} differs from its earlier output {p}: '
                f'{first_difference(_strip(old), _strip(new))}')
    os.makedirs(state_dir, exist_ok=True)
    new['written'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(p + '.tmp', 'w') as fo:
        json.dump(new, fo, indent=1)
    os.replace(p + '.tmp', p)
    print(f'-> {p}', flush=True)


# ── record ───────────────────────────────────────────────────

def level(dev, rows, ids, metric):
    x = stats.level(rows, ids, dev.spk, metric, mode=stats.LINKED)
    return {k: x.get(k) for k in ('est', 'lo', 'hi', 'se', 'n', 'fail', 'undefined')}


def share_pos(rows, ids):
    v = [rows[k]['m9'] for k in ids if k in rows and not stats.failed(rows[k])]
    return {'share': float(np.mean(np.array(v) > 0)) if v else None, 'n': len(v)}


def tables(dev, run, ids, nl_stage):
    """The §4.8 tables of one run on one pair set: WER (capped, uncapped), SIM, UTMOS with CIs; NL from the stage;
    ECAPA SIM and SIM_src per stratum; M9 per stratum with the share D > 0; the gender-pair table."""
    m9 = stats.margin_rows(run.tgt, run.src)
    out = {'wer': level(dev, run.tgt, dev.wer(ids), 'wer_capped'), 'wer_uncapped': level(dev, run.tgt, dev.wer(ids),
                                                                                          'wer'),
           'sim': level(dev, run.tgt, ids, 'sim_wavlm'), 'utmos': level(dev, run.tgt, ids, 'utmos'), 'nl': nl_stage,
           'm9': {g: dict(level(dev, m9, dev.stratum(ids, g), 'm9'), share_pos=share_pos(m9, dev.stratum(ids, g)))
                  for g in STRATA},
           'gender': {gp: {'wer': level(dev, run.tgt, dev.wer(dev.gp(ids, gp)), 'wer_capped'),
                           'sim': level(dev, run.tgt, dev.gp(ids, gp), 'sim_wavlm'),
                           'utmos': level(dev, run.tgt, dev.gp(ids, gp), 'utmos'),
                           'sim_src': level(dev, run.src, dev.gp(ids, gp), 'sim_wavlm')} for gp in GENDER_PAIRS}}
    if run.etgt is not None:
        out['ecapa'] = {'sim': level(dev, run.etgt, ids, 'sim_ecapa'),
                        'sim_src': {g: level(dev, run.esrc, dev.stratum(ids, g), 'sim_ecapa') for g in STRATA}}
    return out


def read_sacct(paths, jobs):
    rows = []
    for path in paths:
        lines = [x.rstrip('\n') for x in open(path) if x.strip()]
        head = lines[0].split('|') if lines else []
        rows += [dict(zip(head, x.split('|'))) for x in lines[1:]]
    out = {}
    for tag, jid in jobs.items():
        m = [r for r in rows if r.get('JobID') == jid]
        out[tag] = {'job_id': jid, 'name': m[0].get('JobName') if m else None,
                    'submit': m[0].get('Submit') if m else None, 'end': m[0].get('End') if m else None,
                    'state': m[0].get('State') if m else None}
    return out


def read_census(path, dev, r2_tags, other_tags):
    """§4.8: the Dev configurations ever scored. A summary.json under a TAG of the jobs before the freeze counts by its
    pair set and budget; one under an R2 or B1 TAG (r2_tags) is an R2 run, used or not (refused, repeated); one under a
    TAG declared with --other_tag (after the freeze, no R2 or B1 job) counts by that TAG; any other TAG is undeclared."""
    def h(ids):
        return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    cls = {h(dev.sets['s100']): 'Dev-S100', h(dev.order[0::20]): 'screen 0/20', h(dev.order): 'full Dev'}
    seen, before, other, r2, undeclared = set(), {}, {}, [], set()
    for line in open(path):
        if not line.strip():
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) != 4 or not re.fullmatch(r'[0-9a-f]{64}', parts[3]):
            die(f'{path}: not a census line (summary.json, budget, pairs, sha256 of the sorted pair_ids): {line!r}')
        s, b, _, ih = parts
        rel = norm_key(s).rsplit('/summary.json', 1)[0]
        if (s, b) in seen:
            continue
        seen.add((s, b))
        t, c = tag_of(rel), cls.get(ih, 'other')
        if t in r2_tags:
            r2.append({'run': rel, 'budget': b, 'pairs': c})
        elif t in PRE_R2_TAGS:
            before.setdefault(c, {}).setdefault(b, 0)
            before[c][b] += 1
        elif t in other_tags:
            other.setdefault(t, {}).setdefault(c, {}).setdefault(b, 0)
            other[t][c][b] += 1
        else:
            undeclared.add(t)
    return {'before_r2': before, 'r2_runs': r2, 'after_freeze_not_r2': other, 'undeclared_tags': sorted(undeclared),
            'sha256': sha256(path), 'path': os.path.abspath(path)}


def parse_tagged(entries, flag):
    out, pr = {}, []
    for x in entries:
        t, _, v = x.partition('=')
        if not t or not v:
            die(f'{flag} {x!r}: give TAG=VALUE')
        if out.get(t, v) != v:
            pr.append(f'{flag}: TAG {t} is given twice ({out[t]!r}, {v!r}); every R2 job uses a new TAG that no other '
                      f'job used (§4.1)')
        out.setdefault(t, v)
    return out, pr


def operator_items(a, ctx, d):
    """The §4.8 items only the operator can supply; returns (items, placeholders, problems)."""
    items, ph, pr = {}, [], []
    jobs, p1 = parse_tagged(a.jobs, '--jobs')
    others, p2 = parse_tagged(a.other_tag, '--other_tag')
    pr += p1 + p2
    used_tags = {tag_of(rel) for rel in ctx.used if rel != E5_REF} | {tag_of(rc['rel']) for rc in ctx.recon.values()}
    r2_tags = used_tags | {t for t in jobs if t != 'b1_train'}
    bad = sorted(t for t in r2_tags if t in PRE_R2_TAGS)
    if bad:
        pr.append(f'--jobs names TAGs of the jobs before the freeze as R2 or B1 jobs: {bad} (§4.1)')
    bad = sorted(t for t in others if t in r2_tags or t in PRE_R2_TAGS)
    if bad:
        pr.append(f'--other_tag names R2, B1 or pre-freeze TAGs: {bad}')
    items['other_tags'] = others
    census = None
    if a.census:
        census = items['census'] = read_census(a.census, ctx.dev, r2_tags, others)
        if census['undeclared_tags']:
            ph.append(f'TAGs after the freeze in the census that no run used here and no option declares: '
                      f'{census["undeclared_tags"]}: --jobs TAG=JOBID when an R2 or B1 job wrote it (also when its runs '
                      f'were refused or repeated), else --other_tag TAG=WHAT')
        listed = {x['run'] for x in census['r2_runs']}
        miss = sorted(rel for rel in ctx.used if rel != E5_REF and rel not in listed)
        if miss:
            pr.append(f'the census lacks the runs used {miss}: make it again after the last R2 job')
    else:
        ph.append('the Dev configurations ever scored under $MVC/eval (--census)')
    tags = sorted(r2_tags) + (['b1_train'] if 'b1' in d else [])
    items['jobs'] = read_sacct(a.sacct, jobs) if a.sacct else {}
    miss = [t for t in tags if t not in jobs]
    if miss or not a.sacct:
        ph.append(f'job IDs and sacct Submit/End of every R2 and B1 job: --jobs TAG=JOBID for {miss or tags} and '
                  f'--sacct')
    else:
        bad = [t for t in jobs if not items['jobs'][t]['submit'] or items['jobs'][t]['end'] in (None, '', 'Unknown')]
        if bad:
            pr.append(f'jobs without Submit or End in the --sacct files: {[(t, jobs[t]) for t in bad]}')
    if a.contamination:
        c = json.load(open(a.contamination))
        items['contamination'] = dict(c, file=os.path.abspath(a.contamination), file_sha256=sha256(a.contamination))
        if not all(k in c for k in ('passed', 'manifest', 'manifest_sha256', 'command')) or \
                not isinstance(c['passed'], bool):
            pr.append('--contamination lacks one of passed (true or false), manifest, manifest_sha256, command')
        elif not c['passed']:
            print('WARNING: the contamination test (PLAN §3.1, mvc/tests/test_train.py) failed; the record states it',
                  flush=True)
    else:
        ph.append('the contamination test on the training ID manifest of mvc_v1\'s model (--contamination)')
    if a.r1_hashes:
        idx = read_sha_lists([a.r1_hashes])
        names = {k.rsplit('/', 1)[-1]: v for k, v in idx.items() if k.startswith(R1_DIR + '/')}
        items['r1_files'] = {'file': os.path.abspath(a.r1_hashes), 'sha256': names}
        fr = read_freeze(FREEZE_PATH)
        miss = [n for n in R1_FILES if n not in names]
        if miss:
            pr.append(f'--r1_hashes lacks $MVC/runs/r1/{miss} (last.pt, args.json, stats.npz and every ema_*.pt, '
                      f'§4.10)')
        for n in ('args.json', 'stats.npz'):
            want = fr.get(f'{R1_DIR}/{n}')
            if n in names and want and names[n] != want:
                pr.append(f'$MVC/runs/r1/{n}: {names[n]} differs from FREEZE_mvc_v1.sha256 ({want})')
        for k, x in d['r1/r2_0']['numbers'].items():
            if SNAP[k] in names and names[SNAP[k]] != x['snapshot_sha256']:
                pr.append(f'{SNAP[k]}: --r1_hashes {names[SNAP[k]]}, describe() of {x["run"]} {x["snapshot_sha256"]}')
    else:
        ph.append('sha256 of $MVC/runs/r1/{last.pt,args.json,stats.npz,ema_*.pt} (--r1_hashes, §4.10)')
    am, at = ANALYZE_MVC, ANALYZE_MVC_TEST
    if am.exists() and at.exists():
        items['analyze_mvc'] = {'analyze_mvc.py': sha256(am), 'test_analyze_mvc.py': sha256(at)}
    else:
        ph.append('mvc/analysis/analyze_mvc.py and mvc/tests/test_analyze_mvc.py, written and reviewed (§5.8)')
    return items, ph, pr


def code_items(ctx, commit):
    """§2.1 and §10: the record's describe() commit against the Mac mvc/*.py, and the files changed since the freeze."""
    code, pr = dict(ctx.code, record=commit), []
    if code['current'] != commit:
        pr.append(f'the Mac mvc/*.py give describe() commit {code["current"]}, but mvc_v1\'s runs carry {commit}: '
                  f'mvc/*.py does not change between the selection record and the end of T1 (§2.1), so the record is '
                  f'written with the code it names')
    changed = code['changed_since_freeze']
    if changed:
        text = DEVIATIONS.read_text() if DEVIATIONS.exists() else ''
        miss = [k for k, v in changed.items() if (v or k) not in text]
        if miss:
            pr.append(f'mvc/*.py changed since the freeze without a deviation entry naming the new sha256 (or, for a '
                      f'removed file, its path) in {DEVIATIONS.name} (§10): {miss}')
    return code, pr


def build_record(ctx, d, a):
    dev = ctx.dev
    final_round = 'b1' if d.get('b1', {}).get('kept') else 'r1'
    R = {s: d.get(f'{final_round}/{s}') for s in STAGES[1:]}
    snap = d['b1']['snapshot'] if final_round == 'b1' else d['r1/r2_0']['snapshot']
    choice = R['r2_4']['choice']
    conf = from_key(choice)
    cfgs = {'mvc_v1': mvc_json(snap['run_dir'], snap['ckpt'], conf)}
    cfgs['mvc_v1_q32'] = dict(cfgs['mvc_v1'], decode='codes')
    cfgs['mvc_copy_latent'] = dict(vc_adapter.DEFAULTS, mode='copy_latent')
    texts = {k: json.dumps(v) + '\n' for k, v in cfgs.items()}
    cfg_files = {k: {'file': f'{k}.json', 'sha256': hashlib.sha256(t.encode()).hexdigest(), 'json': cfgs[k]}
                 for k, t in texts.items()}
    rounds = {}
    for rnd in ('r1', 'b1'):
        if f'{rnd}/r2_3' not in d:
            continue
        r23 = d[f'{rnd}/r2_3']
        runs = {k: next(r for r in ctx.full if r.rel == rel) for k, rel in r23['runs'].items()}
        rounds[rnd] = {'tables': {s: {k: tables(dev, runs[k], dev.sets[s], r23[s]['numbers'][k]['nl']) for k in runs}
                                  for s in ('full', 'd900')}}
        r21 = d[f'{rnd}/r2_1']
        scr = {k: next(r for r in ctx.screen if r.rel == x['run']) for k, x in r21['numbers'].items()}
        rounds[rnd]['screen_ci'] = {k: {q: level(dev, r.tgt, dev.wer(dev.sets['s100']) if q == 'wer_capped' else
                                                 dev.sets['s100'], q) for q in ('wer_capped', 'sim_wavlm', 'utmos')}
                                    for k, r in scr.items()}
    ch_runs = {30: next(r for r in ctx.full if r.rel == d[f'{final_round}/r2_3']['runs'][choice])}
    for r in ctx.budget:
        if r.rel in [x['run'] for x in R['r2_4']['levels'].get(choice, {}).values()]:
            ch_runs[r.budget] = r
    rb = {b: {q: level(dev, r.tgt, dev.wer(dev.sets['full']) if q == 'wer_capped' else dev.sets['full'], q)
              for q in ('wer_capped', 'sim_wavlm', 'utmos')} for b, r in sorted(ch_runs.items())}
    items, ph, pr = operator_items(a, ctx, d)
    code, pc = code_items(ctx, snap['commit'])
    pr += pc
    unused = sorted({r.rel for r in ctx.refs + ctx.screen + ctx.full + ctx.budget + ([ctx.b1run] if ctx.b1run else [])}
                    - ctx.used)
    if unused:
        pr.append(f'runs given but used by no stage: {unused}')
    fr = read_freeze(FREEZE_PATH)
    trig = {'r1': {k: d['r1/r2_4'][k] for k in ('trigger_i', 'trigger_ii', 'b1_triggered', 'r5')}}
    if 'b1/r2_4' in d:
        trig['b1'] = {k: d['b1/r2_4'][k] for k in ('trigger_i', 'trigger_ii', 'b1_triggered', 'r5')}
    here = Path(__file__).resolve()
    rec = {'status': None, 'placeholders': ph, 'problems': pr, 'final_round': final_round,
           'mvc_v1': {'config': choice, 'source': R['r2_4']['source'], 'snapshot': snap,
                      'describe_commit': snap['commit'], 'lineage': lineage(d, final_round), 'cfg_files': cfg_files},
           'code': code,
           'stages': {k: normal(v) for k, v in d.items()}, 'rounds': normal(rounds), 'rb_dev_counterparts': normal(rb),
           'triggers': trig, 'operator': normal(items),
           'scripts': {'select_mvc.py': sha256(here), 'test_select_mvc.py': sha256(SELECT_TEST)
                       if SELECT_TEST.exists() else None,
                       'stats.py': sha256(stats.__file__), 'FREEZE_mvc_v1.sha256': sha256(FREEZE_PATH),
                       'plan': fr.get(PLAN_KEY)},
           'ceai_lists': {p: fr.get(p) for p in CEAI_LISTS}, 'dnsmos': DNSMOS_NOTE,
           'inputs': {'dev': {k: dict(v, path=ctx.dev_paths.get(k)) for k, v in ctx.dev_inputs.items()},
                      'target_copy_src_what': ctx.dev_paths.get('target_copy_src_what'),
                      'files': dict(sorted(ctx.inp.read.items()))},
           'dev_runs_used': sorted(ctx.used), 'recon_runs_used': sorted(rc['rel'] for rc in ctx.recon.values())}
    return rec, texts


def lineage(d, final_round):
    s0 = d['r1/r2_0']['snapshot']
    r1 = f'R1 ({R1_DIR}: the 121M DiT of its args.json, 200k steps on 501 h of LibriTTS-R), EMA snapshot {s0["ckpt"]} ' \
         f'(sha256 {s0["sha256"]}) chosen by R2.0'
    if final_round == 'r1':
        return r1 + '; ' + ('B1 ran and was not kept' if 'b1' in d else 'B1 not triggered')
    b = d['b1']['snapshot']
    return (f'B1 ({b["run_dir"]}): shifted-content fine-tune of {r1}; final EMA snapshot {b["ckpt"]} (sha256 '
            f'{b["sha256"]}, train step {b.get("train_step")}); kept by the §4.7 rule')


def fci(x, n=3, sign=False):
    if not x or x.get('est') is None:
        return '–'
    return f'{f(x["est"], n, sign)} [{f(x.get("lo"), n, sign)}, {f(x.get("hi"), n, sign)}]'


def render(rec, d):
    m = rec['mvc_v1']
    L = ['# MimiFlow-VC selection record v1 (ANALYSIS_PLAN_mvc_v1.md §4)', '',
         f'Status: **{rec["status"]}**. Written {rec["written"]} by `mvc/analysis/select_mvc.py` (sha256 '
         f'{rec["scripts"]["select_mvc.py"]}; tests {rec["scripts"]["test_select_mvc.py"]}) with `benchmark/stats.py` '
         f'(sha256 {rec["scripts"]["stats.py"]}); plan sha256 {rec["scripts"]["plan"]}. Linked crossed bootstrap of the '
         f'Dev speakers, B = {stats.B}, seed {stats.SEED}.', '']
    if rec['placeholders'] or rec['problems']:
        L += ['## Missing before the final record', ''] + [f'- PLACEHOLDER: {x}' for x in rec['placeholders']] + \
             [f'- PROBLEM: {x}' for x in rec['problems']] + ['']
    L += ['## mvc_v1', '', f'- Configuration: **{m["config"]}** ({m["source"]}), round {rec["final_round"]}.',
          f'- Lineage: {m["lineage"]}.', f'- Snapshot: `{m["snapshot"]["path"]}`, sha256 {m["snapshot"]["sha256"]}.',
          f'- describe() commit: {m["describe_commit"]} (frozen code: {rec["code"]["frozen"]}; the Mac mvc/*.py at '
          f'writing: {rec["code"]["current"]}).',
          '- mvc/*.py changed since the freeze (new sha256; deviations, §10): ' +
          ('; '.join(f'`{k}` {v}' for k, v in rec['code']['changed_since_freeze'].items()) or 'none') + '.', '',
          '| File | sha256 | JSON |', '|---|---|---|']
    L += [f'| `{x["file"]}` | {x["sha256"]} | `{json.dumps(x["json"])}` |' for x in m['cfg_files'].values()]
    r0 = d['r1/r2_0']
    L += ['', '## R2.0: snapshot (§4.2)', '', '| Snapshot | run | WER % | SIM | UTMOS | recon WER % | sha256 |',
          '|---|---|---|---|---|---|---|']
    for k, x in r0['numbers'].items():
        L.append(f'| {SNAP[k]} | `{x["run"]}` | {f(x["wer"], 2)} | {f(x["sim"])} | {f(x["utmos"])} | '
                 f'{f(x["recon_wer"], 2)} | {x["snapshot_sha256"]} |')
    L += ['', f'Conditions for 150k: {r0["conditions"]}. Snapshot: **{r0["snapshot"]["ckpt"]}**. R1x (§4.10): '
              f'{"runs" if r0["r1x"]["runs"] else "does not run (neither SIM nor UTMOS rose)"} ({r0["r1x"]}).', '']
    for rnd in ('r1', 'b1'):
        if f'{rnd}/r2_1' not in d:
            continue
        r21, r22, r23, r24 = (d[f'{rnd}/{s}'] for s in STAGES[1:])
        L += [f'## Round {rnd}: R2.1 Dev-S100 screen (§4.3)', '',
              '| Config | WER % [CI] | SIM [CI] | UTMOS [CI] | NL same (SE) | NL cross (SE) | summarize.py same / cross | '
              'cross ΔSIM_src WavLM / ECAPA | 1 | 2 | 3 | 4 | 5 | eligible |', '|---|---|---|---|---|---|---|---|---|'
              '---|---|---|---|---|']
        ci = rec['rounds'][rnd]['screen_ci']
        for k, x in r21['numbers'].items():
            fl, dd = r21['flags'][k], r21['d5'].get(k)
            L.append(f'| {k} | {fci(ci[k]["wer_capped"], 2)} | {fci(ci[k]["sim_wavlm"])} | {fci(ci[k]["utmos"])} | '
                     f'{f(x["nl"]["same"]["est"], 3, True)} ({f(x["nl"]["same"]["se"])}) | '
                     f'{f(x["nl"]["cross"]["est"], 3, True)} ({f(x["nl"]["cross"]["se"])}) | '
                     f'{f(x["nl"]["same"]["summarize"], 3, True)} / {f(x["nl"]["cross"]["summarize"], 3, True)} | '
                     + (f'{f(dd["cross"]["wavlm"], 3, True)} / {f(dd["cross"]["ecapa"], 3, True)}' if dd else '–')
                     + ' | ' + ' | '.join('yes' if fl[i] else 'no' for i in '12345')
                     + f' | {r21["eligibility"][k]["why"]} |')
        L += ['', f'D-mvc-2 pick on this snapshot: **{r21["pick"] or "none"}**; B1 trigger i: {r21["trigger_i"]}.', '',
              f'## Round {rnd}: R2.2 finalists (§4.4)', '', f'{r22["how"]}: {", ".join(r22["finalists"])}.', '']
        for s, title in (('full', 'full Dev'), ('d900', 'Dev-900 (selected on Dev)')):
            t, res = rec['rounds'][rnd]['tables'][s], r23[s]
            L += [f'## Round {rnd}: R2.3 on {title} (§4.5)', '',
                  '| Config | WER % [CI] | uncapped WER % | SIM [CI] | UTMOS [CI] | NL same ± SE (summarize.py) | NL cross ± '
                  'SE (summarize.py) | ECAPA SIM | ECAPA SIM_src same / cross | M9 same / cross (share D > 0) | '
                  'L-pass | WER check | E3 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for k, x in t.items():
                p = res['per'].get(k, {})
                nlx = x['nl']
                e = x.get('ecapa') or {}
                L.append(f'| {k}{" (nosng)" if k == r23["nosng"] else ""} | {fci(x["wer"], 2)} | '
                         f'{f(x["wer_uncapped"]["est"], 2)} | {fci(x["sim"])} | {fci(x["utmos"])} | '
                         + ' | '.join(f'{f(nlx[g]["est"], 3, True)} ± {f(nlx[g]["se"])} '
                                      f'({f(nlx[g]["summarize"], 3, True)})' for g in STRATA)
                         + f' | {fci(e.get("sim"))} | '
                         + ' / '.join(fci((e.get('sim_src') or {}).get(g)) for g in STRATA) + ' | '
                         + ' / '.join(f'{f(x["m9"][g]["est"], 3, True)} ({f(x["m9"][g]["share_pos"]["share"], 2)})'
                                      for g in STRATA)
                         + f' | {p.get("l_pass", "–")} | {p.get("wer_check", "–")} | {p.get("e3", "–")} |')
            L += ['', f'Best SIM {f(res["best_sim"])}, best UTMOS {f(res["best_utmos"])}; E3 order: '
                      f'{", ".join(res["order"]) or "empty"}' + (f'; fallback F: {res["fallback_f"]}'
                                                                   if not res['order'] else '') + '.', '',
                  '| Config | Pair | WER % | SIM | UTMOS | SIM_src |', '|---|---|---|---|---|---|']
            for k, x in t.items():
                for gp, v in x['gender'].items():
                    L.append(f'| {k} | {gp} | {fci(v["wer"], 2)} | {fci(v["sim"])} | {fci(v["utmos"])} | '
                             f'{fci(v["sim_src"])} |')
            L.append('')
        L += [f'Dev-900 check: first on 1,000 pairs {r23["full"]["first"]}, on Dev-900 {r23["d900"]["first"]}'
              + (' (agree).' if r23['d900_agrees'] else ' (differ: both reported; the 1,000-pair choice stands).'), '',
              f'## Round {rnd}: R2.4 reference budgets (§4.6)', '']
        L += [f'- {c["config"]}: SIM 3 s {f(c["sim3"])}, 30 s {f(c["sim30"])}, 180 s {f(c["sim180"])} -> '
              f'{"passes" if c["pass"] else "fails"}' for c in r24['checked']]
        L += [f'- Choice: **{r24["choice"]}** ({r24["source"]}); R2.4 failure recorded: {r24["r2_4_failed"]}.', '']
        L += [f'Triggers: B1 (i) {r24["trigger_i"]}, (ii) {r24["trigger_ii"]} -> B1 '
              f'{"triggered" if r24["b1_triggered"] else "not triggered"}; R5 (reported only) {r24["r5"]}.', '']
        if rnd == 'r1' and 'b1' in d:
            b = d['b1']
            L += ['## B1 (§4.7)', '', f'Parent {b["parent"]["config"]} (`{b["parent"]["run"]}`); B1 run `{b["run"]}`, '
                  f'snapshot `{b["snapshot"]["path"]}` sha256 {b["snapshot"]["sha256"]}.', '',
                  '| Δ (B1 − parent) | est [CI] | rule | met |', '|---|---|---|---|']
            rules = {'sim_src_same': '≤ −0.02', 'sim_src_cross': '≤ −0.02', 'sim': '≥ −0.005', 'utmos': '≥ −0.05',
                     'wer': '≤ +0.5'}
            for k, v in b['deltas'].items():
                L.append(f'| {k} | {fci(v, 2 if k == "wer" else 3, True)} | {rules[k]} | {b["flags"][k]} |')
            L += ['', f'B1 {"kept" if b["kept"] else "not kept"}.', '']
        if rnd == 'b1' and r24['b1_triggered']:
            L += ['The second R2 meets a B1 trigger again: nothing more starts (§4.7); §7.2 reports the leakage.', '']
    L += ['## Dev counterparts of family RB (mvc_v1 on full Dev)', '', '| Budget | WER % | SIM | UTMOS |',
          '|---|---|---|---|']
    L += [f'| {b} s | {fci(v["wer_capped"], 2)} | {fci(v["sim_wavlm"])} | {fci(v["utmos"])} |'
          for b, v in rec['rb_dev_counterparts'].items()]
    op = rec['operator']
    L += ['', '## Dev configurations ever scored (§4.8)', '']
    if op.get('census'):
        c = op['census']

        def by_budget(by):
            return ', '.join(f'{n} at {b} s' for b, n in sorted(by.items(), key=lambda x: int(x[0])))
        L.append('Before R2 (summary.json files under the TAGs of the jobs before the freeze):')
        L += [f'- {cls}: {by_budget(by)}' for cls, by in sorted(c['before_r2'].items())]
        used = set(rec['dev_runs_used'])
        n_used = sum(x['run'] in used for x in c['r2_runs'])
        L.append(f'- plus the R2 runs (summary.json files under the R2 and B1 TAGs, used or not): {len(c["r2_runs"])}, '
                 f'of which {n_used} used by the selection and {len(c["r2_runs"]) - n_used} not used (refused or '
                 f'repeated); by pair set: ' + '; '.join(
                     f'{p} {sum(x["pairs"] == p for x in c["r2_runs"])}' for p in sorted({x['pairs']
                                                                                         for x in c['r2_runs']})) +
                 '. The dev-clean reconstructions of R2.0 are no Dev configuration and are not counted.')
        for t, by in sorted(c['after_freeze_not_r2'].items()):
            L.append(f'- after the freeze, no R2 or B1 job, TAG {t} ({op["other_tags"].get(t)}): ' +
                     '; '.join(f'{cls} {by_budget(b)}' for cls, b in sorted(by.items())))
        L.append(f'- census file sha256 {c["sha256"]}.')
    else:
        L.append('- PLACEHOLDER (--census)')
    am, co, r1f = op.get('analyze_mvc'), op.get('contamination'), op.get('r1_files')
    L += ['', '## Other items (§4.8)', '',
          '- analyze_mvc.py and its tests: ' + ('; '.join(f'`{k}` {v}' for k, v in am.items()) if am else
                                                'PLACEHOLDER'),
          '- Contamination test (PLAN §3.1): ' + (f'**{"passed" if co.get("passed") is True else "FAILED"}**; '
                                                  f'manifest `{co.get("manifest")}` sha256 '
                                                  f'{co.get("manifest_sha256")}; `{co.get("command")}`; evidence file '
                                                  f'sha256 {co.get("file_sha256")}' if co else 'PLACEHOLDER'),
          '- R1 run files (§4.10): ' + ('; '.join(f'`{k}` {v}' for k, v in sorted(r1f['sha256'].items())) if r1f
                                        else 'PLACEHOLDER'), f'- DNSMOS: {rec["dnsmos"]}',
          '- ceai sha256 of the pair lists and the source-E file: ' +
          '; '.join(f'`{p}` {h}' for p, h in rec['ceai_lists'].items()), '', '| Job (TAG) | ID | Submit | End | State |',
          '|---|---|---|---|---|']
    L += [f'| {t} | {j["job_id"]} | {j["submit"]} | {j["end"]} | {j["state"]} |' for t, j in op.get('jobs', {}).items()]
    L += ['', '## Inputs', '', f'- Dev pairs: `{rec["inputs"]["dev"]["pairs"]["path"]}` sha256 '
                                f'{rec["inputs"]["dev"]["pairs"]["sha256"]}',
          f'- Dev-S100: sha256 {rec["inputs"]["dev"]["dev_s100"]["sha256"]}',
          f'- target_copy@src: {rec["inputs"]["target_copy_src_what"]}, sha256 '
          f'{rec["inputs"]["dev"]["target_copy_src"]["sha256"]}',
          f'- Runs used: {len(rec["dev_runs_used"])} Dev runs; reconstructions {rec["recon_runs_used"]}.', '',
          '| File (under $MVC/eval) | sha256 |', '|---|---|']
    L += [f'| `{k}` | {v} |' for k, v in rec['inputs']['files'].items()]
    L += ['', '## Choices where the plan is silent', ''] + [f'- {x}' for x in CHOICES]
    L += ['', '## Traces (every step, as printed)', '']
    for sid, x in d.items():
        L += [f'### {sid}', ''] + [f'- {t}' for t in x['trace']] + ['']
    return '\n'.join(L) + '\n'


CHOICES = ['Ties (§4.1): orderings compare WER at two decimals, SIM and UTMOS at three; thresholds use unrounded point '
           'estimates (tolerance 1e-9).',
           'R2.2 orders by R2.3\'s full order; without a D-mvc-2 pick at most three other eligible configurations.',
           'Exact NL ties (R2.2 under trigger i, fallback F) are broken by smaller w, smaller CFG, plain before interval.',
           'The D-mvc-2 pick includes w 0.',
           'In the B1 round, mvc_r1_nosng is B1\'s snapshot at CFG 0, SNG 0.',
           'Conditions 2-4 and the R2.3 WER check compare levels; the B1 keep rule is paired.',
           'Condition 5 as job e3\'s rule5.py: the WavLM mean over the cross-gender pairs where both WavLM @src rows '
           'are usable and the ECAPA mean over those where both ECAPA @src rows are usable, separately; a row is usable '
           'without error and with a finite value (e3 took every row carrying the value).',
           'The contamination test\'s result is recorded; a failure does not stop the record.',
           'The Dev-900 repeat compares the first of the order (or F); R2.4 is not repeated on Dev-900.']


def write_record(a, rec, texts, d):
    rec['written'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if a.final:
        if rec['placeholders'] or rec['problems']:
            die('the final record needs: ' + '; '.join(rec['placeholders'] + rec['problems']))
        rec['status'] = 'complete'
    else:
        rec['status'] = 'DRAFT: not the selection record' + (' (placeholders and problems listed below)'
                                                             if rec['placeholders'] or rec['problems'] else '')
    md = render(rec, d)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.final:
        os.makedirs(a.cfg_dir, exist_ok=True)
        for k, t in texts.items():
            with open(os.path.join(a.cfg_dir, f'{k}.json'), 'w') as fo:
                fo.write(t)
    with open(a.out, 'w') as fo:
        fo.write(md)
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as fo:
        json.dump(rec, fo, indent=1, default=float)
    print(md)
    if a.final:
        m, c = rec['mvc_v1'], rec['mvc_v1']['cfg_files']
        env = [f'RECORD_SHA256={sha256(a.out)}', f'SNAPSHOT={m["snapshot"]["path"]}',
               f'SNAPSHOT_SHA256={m["snapshot"]["sha256"]}']
        for k, name in (('mvc_v1', 'CFG_MVC_V1'), ('mvc_v1_q32', 'CFG_MVC_V1_Q32'),
                        ('mvc_copy_latent', 'CFG_MVC_COPY_LATENT')):
            env += [f'{name}={c[k]["file"]}', f'{name}_SHA256={c[k]["sha256"]}']
        env.append(f'DESCRIBE_COMMIT={m["describe_commit"]}')
        with open(a.env, 'w') as fo:
            fo.write('\n'.join(env) + '\n')
        print(f'-> {a.out}, {os.path.splitext(a.out)[0]}.json, {a.env}, {a.cfg_dir}/', flush=True)
    else:
        print(f'-> {a.out} (draft; no env file)', flush=True)


# ── main ─────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], fromfile_prefix_chars='@')
    ap.add_argument('--stage', required=True, choices=STAGES + ('b1', 'record'))
    ap.add_argument('--round', default='r1', choices=('r1', 'b1'))
    ap.add_argument('--eval_root', required=True, help='the Mac copy of $MVC/eval')
    ap.add_argument('--sha256', action='append', default=[], help='sha256sum list made on ceai (repeatable)')
    ap.add_argument('--pairs', required=True, help='$J/pairs/dev.jsonl')
    ap.add_argument('--target_copy_src', required=True, help='$J/v1/d2/controls/metrics_src.jsonl')
    ap.add_argument('--state_dir', required=True, help='saved stage outputs')
    ap.add_argument('--ref', action='append', default=[], help='R2.0 run (150k, 200k), relative to --eval_root')
    ap.add_argument('--recon', action='append', default=[], help='k150=RUN, k200=RUN (reconstruction)')
    ap.add_argument('--screen', action='append', default=[], help='R2.1 run (repeatable)')
    ap.add_argument('--full', action='append', default=[], help='full-Dev 30 s run (finalists, mvc_r1_nosng)')
    ap.add_argument('--budget', action='append', default=[], help='full-Dev 3 s or 180 s run (R2.4)')
    ap.add_argument('--b1_run', default=None, help='B1 on full Dev at the parent setting')
    ap.add_argument('--out', default=str(MVC / 'MVC_SELECTION_RECORD_v1.md'))
    ap.add_argument('--env', default=str(MVC / 'mvc_selection.env'))
    ap.add_argument('--cfg_dir', default=str(MVC / 'MVC_SELECTION_RECORD_v1_cfg'))
    ap.add_argument('--sacct', action='append', default=[])
    ap.add_argument('--jobs', action='append', default=[],
                    help='TAG=JOBID: every R2 and B1 eval TAG, also of runs refused or repeated; b1_train')
    ap.add_argument('--other_tag', action='append', default=[],
                    help='TAG=WHAT: a TAG after the freeze that is no R2 or B1 job (such as R1x\'s evals)')
    ap.add_argument('--contamination', default=None)
    ap.add_argument('--r1_hashes', default=None)
    ap.add_argument('--census', default=None)
    ap.add_argument('--final', action='store_true')
    a = ap.parse_args(argv)
    if a.stage == 'r2_0' and a.round != 'r1':
        ap.error('R2.0 is not repeated for B1 (§4.7)')
    if a.final and a.stage != 'record':
        ap.error('--final belongs to --stage record')
    return a


def main(argv=None):
    a = parse_args(argv)
    h = sha256(stats.__file__)
    if h != STATS_SHA256:
        die(f'benchmark/stats.py has sha256 {h}, not the frozen {STATS_SHA256}')
    ctx = load_context(a)
    sid = a.stage if a.stage in ('b1', 'record') else f'{a.round}/{a.stage}'
    d = derive(ctx, sid)
    if sid == 'record':
        check_saved(a.state_dir, d)
        rec, texts = build_record(ctx, d, a)
        write_record(a, rec, texts, d)
        return 0
    check_saved(a.state_dir, d, skip=sid)
    save_stage(a.state_dir, sid, d[sid])
    return 0


if __name__ == '__main__':
    sys.exit(main())
