#!/usr/bin/env python3
"""
Stimuli of the MOS/SMOS listening test of MimiFlow-VC (mvc_v1), under mvc/LISTENING_ADDENDUM_mvc_v1.md (the addendum
that ANALYSIS_PLAN_mvc_v1.md §6.4 asks for). The order is: freeze the addendum, the design file and this script
(commit, tag listening-addendum-mvc-v1, push the tag), then draw the items, then build the stimuli. The design follows
the journal's listening test (benchmark/ANALYSIS_PLAN_v1.md §5.4, benchmark/listening_stimuli.py), without CMOS.

  draw   (CPU, stdlib) --design design.json --out items.json [--addendum <file> --addendum_sha256 <hex>]
         [--vctk_pairs --ltc_pairs --mvc_rows --journal_rows] [--throwaway_seed <text>]
         Per set and gender combination (F2F, F2M, M2F, M2M): the pair_ids sorted, then shuffled with
         random.Random('<seed>:listening:<set>:<g>') (the journal's convention, select_crt_al.listening_order); where
         prefer_parallel is set for the set, the pairs with a parallel recording P come first, each group in that order.
         The cells are filled in the order vctk2k F2F, F2M, M2F, M2M, then ltc2k likewise; the items of a cell are the
         first n_per_stratum pairs of its order that are eligible and pass the speaker and sentence rules given the items
         already taken in the set (ranks 0, 1, ... in the order taken). Every pair passed over is reported with its
         reason. A pair is eligible when:
           - it is not in exclude_pairs, and its source_dur is within source_range;
           - nat, the natural target clip: P where the pair has one (not eligible unless P_dur is within natural_range),
             else the target's E utterance (the rows' sim_ref_wavs) within natural_range closest in length to the
             source (ties: path); never cut, never in the conversion reference R (any budget);
           - ref, the SMOS reference: another E utterance, not in R, within ref_range and closest to ref_target_s
             (ties: path); if none and ref_fallback, the shortest E utterance longer than ref_range[1], cut to it;
           - src, the natural source speech (source_path), where a block uses it: never nat, ref or in R;
           - every converted condition of the design has a 30 s row without an error that agrees with the pair list:
               mvc_v1  the MimiFlow-VC T1 rows (<mvc_rows>/t1|t2/metrics/<set>_30.jsonl)
               crtal   CRT-AL, the journal rows of cand_pm18_rq (<journal_rows>/t1|t2/metrics/<set>_30.jsonl)
               seedvc, knnvc, facodec   the journal T4 rows (<journal_rows>/t4/metrics/<set>_30.jsonl)
         speaker_rule: within a set, a source speaker in at most one item, and a target speaker in at most one item.
         sentence_rule: within a set, a source sentence (source_text, case and punctuation ignored) in at most one item.
         Practice items ([[set, genders, condition]]), after all items: the first eligible pair of the cell's order after
         its last item whose target speaker no item and no earlier practice item of the set has.
         Only pair_id, system, wav, error and the fields checked against the pair list are read from a row: no metric.
         Writes items.json (orders, items, practice, every pair passed over, the design and the sha256 of every input)
         and items.json.sha256. On the real pair lists the draw needs the freeze (as build, with the design file);
         --throwaway_seed draws with another seed string for a feasibility check, and build refuses such items.
  build  (CPU) --items items.json --addendum <file> --addendum_sha256 <hex> [--out --key_dir --cache --remote
         --manifest]
         The freeze: the addendum has that sha256; this repository's tag listening-addendum-mvc-v1 holds the addendum,
         the design file named in items.json and this script, each identical to the file on disk; and the tag on
         origin (git ls-remote) points at the same commit. Then build draws again from the design, pair lists and rows
         named in items.json and refuses unless the result equals items.json. It fetches the audio read-only from ceai
         (rsync --files-from; the journal's /datastore/nvquoc/crt paths moved to /datastore/ceai/crt) into --cache,
         loads every clip as 24 kHz mono, cuts the fallback references (50 ms fade-out), normalises to an active-speech
         RMS of -26 dBFS (20 ms frames within 40 dB of the loudest; the gain is lowered when the peak would pass
         -1 dBFS; the journal's normalisation), appends pad_ms of silence (uniform in the design's range, drawn with
         secrets, so that durations do not tell the systems apart) and writes, through a temporary folder renamed on
         success: <out>/stimuli/<id>.flac (16-bit) under random ids; <out>/lt_data.json (the journal page's LT_DATA,
         version 1: practice, MOS with the natural target clips as gold items, SMOS with the reference and the
         cross-gender natural source clips as gold_src items, attention items, no CMOS, durations to 0.1 s);
         <key_dir>/key.json (what every stimulus and trial is, with the native rate and sha256 of every source file);
         and the manifest (sha256 of the clips, the key, lt_data.json, items.json, the design and the frozen files; no
         condition). --out, --key_dir and --cache lie outside the repository, and neither the key nor the cache is
         inside the package or contains it.
         --dry_run skips the freeze and the fetch, for synthetic inputs only: it refuses the real pair lists and any
         /datastore path, and marks lt_data.json, key.json and the manifest as a dry run.
Ids are random (secrets), not derived from the public seed, and the trial and stimulus lists are written in a random
order, so lt_data.json says nothing about what a clip is.

    .venv-stats/bin/python mvc/mos/build_stimuli.py draw --design mvc/mos/design_mvc_v1.json --out <items.json> \\
        --addendum mvc/LISTENING_ADDENDUM_mvc_v1.md --addendum_sha256 <sha256>
    .venv-stats/bin/python mvc/mos/build_stimuli.py build --items <items.json> --addendum <addendum.md> \\
        --addendum_sha256 <sha256>        (or: uvx --with numpy --with soundfile --with soxr python ...)
Defaults: --out $MOS_DATA_DIR/package, --key_dir $MOS_DATA_DIR/private, --cache $MOS_DATA_DIR/private/cache,
--manifest mvc/LISTENING_BUILD_mvc_v1.json (dry run: <key_dir>/build_manifest.json); MOS_DATA_DIR: ~/mos_data.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SELF = os.path.abspath(__file__)
TAG = 'listening-addendum-mvc-v1'
REMOTE = 'ceai@hpc.uit.edu.vn'
MOVED = ('/datastore/nvquoc/crt/', '/datastore/ceai/crt/')
REMOTE_PREFIX = '/datastore/'
REAL_PAIR_LISTS = {   # VCTK-2k and LTC-2k: the Mac copies and the ceai copies (MVC_SELECTION_RECORD_v1.json)
    'bdbf36a32b61eff625e2acaf40163835582ca9e391079c11ca72e6bf51601c1d',
    '32fc920516f0b6e4d239ddc85994cca8f746538763e5ad71ab2e208203c56e45',
    '242727e0570365f8a030d27cccacdf1b4a35aa38f21cfe0e99c523d3bb0ea309',
    '882824e969221ae06966eb57e6de5f98e391940da14e66e16ee6296512c825bf'}
SETS = ('vctk2k', 'ltc2k')
PART = {'vctk2k': 't1', 'ltc2k': 't2'}
CELLS = ('F2F', 'F2M', 'M2F', 'M2M')
CROSS = ('F2M', 'M2F')
BUDGET = 30
SYSTEMS = {'mvc_v1': ('mvc', 'mvc_v1'), 'crtal': ('journal', 'cand_pm18_rq'), 'seedvc': ('t4', 'seedvc'),
           'knnvc': ('t4', 'knnvc'), 'facodec': ('t4', 'facodec')}
NATURAL = ('nat', 'src')
CONDITIONS = NATURAL + tuple(SYSTEMS)
SR = 24000
TARGET_DBFS, PEAK_DBFS, ACTIVE_DB, FRAME_S, FADE_S = -26.0, -1.0, 40.0, 0.02, 0.05
DUR_STEP = 0.1        # the durations raters' browsers receive
REQUIRED = ('seed', 'n_per_stratum', 'mos_conditions', 'smos_conditions', 'halves', 'source_range', 'natural_range',
            'ref_range', 'ref_target_s', 'ref_fallback', 'speaker_rule', 'sentence_rule', 'pad_ms')
OPTIONAL = ('sets', 'prefer_parallel', 'exclude_pairs', 'practice', 'attention', 'title')
RATING = range(1, 6)
DEFAULT_CONFIG = {'title': 'Listening test: voice conversion', 'contact': 'the organiser',
                  'consent_html': '<p>[Consent text approved for this study goes here.]</p>', 'halves': False,
                  'require_full_play': True, 'headphone_check': {'trials': 6, 'pass': 5, 'attempts': 2},
                  'submit_url': None}
SYSTEM_RE = re.compile(r'"system":\s*"([^"]*)"')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def remote_path(p):
    return MOVED[1] + p[len(MOVED[0]):] if p and p.startswith(MOVED[0]) else p


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def shown(path):
    """A path as recorded: relative to the repo when inside it."""
    return os.path.relpath(os.path.realpath(path), os.path.realpath(ROOT)) if inside(path, ROOT) \
        else os.path.abspath(path)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── design ───────────────────────────────────────────────────

def all_conditions(d):
    return list(dict.fromkeys(d['mos_conditions'] + d['smos_conditions']))


def check_range(d, k, lo_min=0.0):
    r = d[k]
    if not (isinstance(r, list) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r)
            and lo_min <= r[0] <= r[1]):
        raise SystemExit(f'ERROR: {k} must be [low, high] with {lo_min} <= low <= high')


def check_design(d):
    """The design file: REQUIRED and OPTIONAL keys (see the module text); attention {mos|smos: [[set, genders, r]]}."""
    missing, unknown = [k for k in REQUIRED if k not in d], [k for k in d if k not in REQUIRED + OPTIONAL]
    if missing or unknown:
        raise SystemExit(f'ERROR: the design lacks {missing} or has unknown keys {unknown}')
    n, sets = d['n_per_stratum'], d.get('sets', list(SETS))
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise SystemExit('ERROR: n_per_stratum must be a positive integer')
    for k in ('mos_conditions', 'smos_conditions'):
        c = d[k]
        if not c or any(x not in CONDITIONS for x in c) or len(set(c)) != len(c):
            raise SystemExit(f'ERROR: {k} {c} empty, unknown or repeated (known: {list(CONDITIONS)})')
    if 'nat' not in d['mos_conditions']:
        raise SystemExit('ERROR: mos_conditions must include nat (the gold items of the screening)')
    for k in ('halves', 'ref_fallback', 'speaker_rule', 'sentence_rule'):
        if not isinstance(d[k], bool):
            raise SystemExit(f'ERROR: {k} must be true or false')
    for k in ('source_range', 'natural_range', 'ref_range'):
        check_range(d, k)
    check_range(d, 'pad_ms')
    if not all(isinstance(x, int) for x in d['pad_ms']) or d['pad_ms'][1] > 1000:
        raise SystemExit('ERROR: pad_ms must be whole milliseconds, at most 1000')
    if not d['ref_range'][0] <= d['ref_target_s'] <= d['ref_range'][1]:
        raise SystemExit('ERROR: ref_target_s must lie within ref_range')
    if not sets or any(s not in SETS for s in sets) or len(set(sets)) != len(sets):
        raise SystemExit(f'ERROR: sets must be a non-empty subset of {list(SETS)}')
    if d['halves'] and n % 2:
        raise SystemExit('ERROR: halves need an even n_per_stratum')
    if any(s not in sets or not isinstance(v, bool) for s, v in d.get('prefer_parallel', {}).items()):
        raise SystemExit('ERROR: prefer_parallel maps a set of the design to true or false')
    ex = d.get('exclude_pairs', [])
    if not all(isinstance(k, str) for k in ex) or len(set(ex)) != len(ex):
        raise SystemExit('ERROR: exclude_pairs must be distinct pair_ids')
    for s, c, cond in d.get('practice', []):
        if s not in sets or c not in CELLS or cond not in d['mos_conditions']:
            raise SystemExit(f'ERROR: practice cell {[s, c, cond]} is not in the design (MOS conditions only)')
    for block, specs in d.get('attention', {}).items():
        if block not in ('mos', 'smos'):
            raise SystemExit(f'ERROR: attention block {block} (mos or smos)')
        for s, c, r in specs:
            if s not in sets or c not in CELLS or r not in RATING:
                raise SystemExit(f'ERROR: attention item {[s, c, r]} of {block} is not valid')
            if n < 2:
                raise SystemExit('ERROR: attention items use rank 1 of a cell: n_per_stratum must be at least 2')
    return d


# ── draw ─────────────────────────────────────────────────────

def rows_path(tree, s, mvc_rows, journal_rows):
    if tree == 'mvc':
        return os.path.join(mvc_rows, PART[s], 'metrics', f'{s}_{BUDGET}.jsonl')
    return os.path.join(journal_rows, PART[s] if tree == 'journal' else 't4', 'metrics', f'{s}_{BUDGET}.jsonl')


KEEP = ('pair_id', 'system', 'wav', 'error', 'genders', 'target_speaker', 'sim_ref_wavs', 'parallel_wav', 'source_wav')


def load_rows(conditions, sets, mvc_rows, journal_rows):
    """({condition: {pair_id: row}}, {condition: {set: {path, sha256, system}}}): only the KEEP fields of a row."""
    want = {}
    for c in conditions:
        if c in SYSTEMS:
            for s in sets:
                tree, system = SYSTEMS[c]
                want.setdefault(rows_path(tree, s, mvc_rows, journal_rows), {})[system] = c
    rows = {c: {} for c in conditions if c in SYSTEMS}
    inputs = {c: {} for c in conditions if c in SYSTEMS}
    for path, systems in want.items():
        if not os.path.exists(path):
            raise SystemExit(f'ERROR: {path} missing')
        seen = {c: set() for c in systems.values()}
        with open(path) as f:
            for line in f:
                m = SYSTEM_RE.search(line, 0, 2000)
                if m and m.group(1) not in systems or not line.strip():
                    continue
                r = json.loads(line)
                c = systems.get(r.get('system'))
                if c is None:
                    continue
                if r['pair_id'] in seen[c]:
                    raise SystemExit(f'ERROR: {path}: two rows of {r["system"]} for {r["pair_id"]}')
                seen[c].add(r['pair_id'])
                rows[c][r['pair_id']] = {k: r[k] for k in KEEP if k in r}
        digest = sha256(path)
        for system, c in systems.items():
            s = os.path.basename(path).split('_')[0]
            inputs[c][s] = {'path': shown(path), 'sha256': digest, 'system': system, 'n_rows': len(seen[c])}
    return rows, inputs


def ok(row):
    return row is not None and 'error' not in row and bool(row.get('wav'))


def conversion_ref(pair):
    """Every reference path of the pair, all budgets (the ceai paths)."""
    return sorted({remote_path(p) for b in (pair.get('R') or {}).values() for p in b.get('paths') or []})


def within(x, r):
    return x is not None and r[0] <= x <= r[1]


def natural_and_ref(pair, d):
    """(nat, ref, None) of a pair, each {path, kind, dur, cut_s}, or (nat or None, None, reason)."""
    excluded = set(conversion_ref(pair))
    e = pair.get('E') or {}
    utts = [(remote_path(p), x) for p, x in zip(e.get('paths') or [], e.get('durs') or []) if remote_path(p)
            not in excluded]
    p = remote_path(pair.get('P'))
    if p:
        if p in excluded:
            return None, None, 'P is in the conversion reference'
        if not within(pair.get('P_dur'), d['natural_range']):
            return None, None, 'P duration outside natural_range'
        nat = {'path': p, 'kind': 'P', 'dur': pair['P_dur'], 'cut_s': None}
    else:
        cand = [u for u in utts if within(u[1], d['natural_range'])]
        if not cand:
            return None, None, 'no E utterance within natural_range for the natural clip'
        q, x = min(cand, key=lambda u: (abs(u[1] - pair['source_dur']), u[0]))
        nat = {'path': q, 'kind': 'E', 'dur': x, 'cut_s': None}
    rest = [u for u in utts if u[0] != nat['path']]
    cand = [u for u in rest if within(u[1], d['ref_range'])]
    if cand:
        q, x = min(cand, key=lambda u: (abs(u[1] - d['ref_target_s']), u[0]))
        return nat, {'path': q, 'kind': 'E', 'dur': x, 'cut_s': None}, None
    longer = [u for u in rest if u[1] > d['ref_range'][1]]
    if d['ref_fallback'] and longer:
        q, x = min(longer, key=lambda u: (u[1], u[0]))
        return nat, {'path': q, 'kind': 'E', 'dur': x, 'cut_s': d['ref_range'][1]}, None
    return nat, None, 'no E utterance for the SMOS reference'


def check_row(r, pair, c):
    """A row must describe the pair of the list (genders, target, E = sim_ref_wavs, P = parallel_wav, source)."""
    e = (pair.get('E') or {}).get('paths') or []
    want = (pair['genders'], pair['target_speaker'], sorted(remote_path(p) for p in e), remote_path(pair.get('P')),
            remote_path(pair.get('source_path')))
    got = (r.get('genders'), r.get('target_speaker'), sorted(remote_path(p) for p in r.get('sim_ref_wavs') or []),
           remote_path(r.get('parallel_wav')), remote_path(r.get('source_wav')))
    names = ('genders', 'target_speaker', 'sim_ref_wavs', 'parallel_wav', 'source_wav')
    bad = [n for n, a, b in zip(names, want, got) if a != b]
    if bad:
        raise SystemExit(f'ERROR: the {c} row of {pair["pair_id"]} disagrees with the pair list on {bad}')


def item_audio(pair, rows, conditions, d):
    """({condition: {path, kind, cut_s, ...}, 'ref': ...}, None) of one pair, or (None, reason) when not eligible."""
    if pair['pair_id'] in d.get('exclude_pairs', ()):
        return None, 'in exclude_pairs'
    if not within(pair.get('source_dur'), d['source_range']):
        return None, 'source duration outside source_range'
    nat, ref, why = natural_and_ref(pair, d)
    if why:
        return None, why
    out = {'nat': nat, 'ref': ref}
    if 'src' in conditions:
        src = remote_path(pair.get('source_path'))
        if not src or src in {nat['path'], ref['path']} | set(conversion_ref(pair)):
            return None, 'the source clip is missing, the natural clip, the reference or in R'
        out['src'] = {'path': src, 'kind': 'source', 'dur': pair['source_dur'], 'cut_s': None}
    for c in conditions:
        if c in NATURAL:
            continue
        r = rows[c].get(pair['pair_id'])
        if r is None:
            return None, f'{c}: no row'
        if not ok(r):
            return None, f'{c}: error row'
        check_row(r, pair, c)
        out[c] = {'path': remote_path(r['wav']), 'kind': 'output', 'system': SYSTEMS[c][1], 'cut_s': None}
    return out, None


def sentence(pair):
    return ' '.join(re.findall(r'[a-z0-9]+', (pair.get('source_text') or '').lower())) or pair['pair_id']


def clash(pair, used, d):
    """Why the speaker or sentence rule refuses a pair, given the items taken in its set (None: it passes)."""
    if d['speaker_rule']:
        for role, k in (('source', 'source_speaker'), ('target', 'target_speaker')):
            if pair[k] in used[role]:
                return f'speaker rule: {role} speaker {pair[k]} is in {used[role][pair[k]]}'
    if d['sentence_rule'] and sentence(pair) in used['sentence']:
        return f'sentence rule: the sentence of {used["sentence"][sentence(pair)]}'
    return None


def seeded_order(pairs, seed, s, g):
    ids = sorted(p['pair_id'] for p in pairs if p['set'] == s and p['genders'] == g)
    random.Random(f'{seed}:listening:{s}:{g}').shuffle(ids)
    return ids


def draw(design, pairs, rows, seed=None):
    """The orders, items, practice items and the pairs passed over (see the module text)."""
    d = check_design(design)
    seed = d['seed'] if seed is None else seed
    n, conds, sets = d['n_per_stratum'], all_conditions(d), d.get('sets', list(SETS))
    unknown = sorted(k for k in d.get('exclude_pairs', []) if k.split('__')[0] in sets and k not in pairs)
    if unknown:
        raise SystemExit(f'ERROR: exclude_pairs names {len(unknown)} pairs that are not in the pair lists, '
                         f'e.g. {unknown[0]}')
    order, items, practice, passed, notes = {}, [], [], [], []
    for s in sets:
        order[s], used = {}, {'source': {}, 'target': {}, 'sentence': {}}
        for g in CELLS:
            o = seeded_order(pairs.values(), seed, s, g)
            if d.get('prefer_parallel', {}).get(s):
                o = [k for k in o if pairs[k].get('P')] + [k for k in o if not pairs[k].get('P')]
            order[s][g] = o
            pos, chosen = 0, []
            while len(chosen) < n:
                if pos >= len(o):
                    raise SystemExit(f'ERROR: {s} {g}: {len(chosen)} eligible pairs, {n} needed')
                k, p = o[pos], pairs[o[pos]]
                audio, why = item_audio(p, rows, conds, d)
                why = why or clash(p, used, d)
                if why:
                    passed.append({'set': s, 'genders': g, 'rank': len(chosen), 'position': pos, 'pair_id': k,
                                   'reason': why})
                else:
                    chosen.append({'set': s, 'genders': g, 'rank': len(chosen), 'position': pos, 'pair_id': k,
                                   'source_speaker': p['source_speaker'], 'target_speaker': p['target_speaker'],
                                   'source_utt': p.get('source_utt'), 'has_P': bool(p.get('P')),
                                   'conversion_ref': conversion_ref(p), 'audio': audio})
                    used['source'][p['source_speaker']] = used['target'][p['target_speaker']] = k
                    used['sentence'][sentence(p)] = k
                pos += 1
            if d.get('prefer_parallel', {}).get(s) and not all(it['has_P'] for it in chosen):
                notes.append(f'{s} {g}: {sum(not it["has_P"] for it in chosen)} items without P')
            items += chosen
        texts = [sentence(pairs[it['pair_id']]) for it in items if it['set'] == s]
        if len(set(texts)) < len(texts):
            notes.append(f'{s}: {len(texts) - len(set(texts))} items repeat the sentence of another item')
    for s, g, c in d.get('practice', []):
        cell = [it for it in items if (it['set'], it['genders']) == (s, g)]
        taken = {it['pair_id'] for it in items + practice}
        targets = {it['target_speaker'] for it in items + practice if it['set'] == s}
        for k in order[s][g][max(it['position'] for it in cell) + 1:]:
            p = pairs[k]
            if k in taken or p['target_speaker'] in targets:
                continue
            audio, why = item_audio(p, rows, [c], d)
            if why is None:
                practice.append({'set': s, 'genders': g, 'condition': c, 'pair_id': k,
                                 'source_speaker': p['source_speaker'], 'target_speaker': p['target_speaker'],
                                 'conversion_ref': conversion_ref(p), 'audio': audio})
                break
        else:
            raise SystemExit(f'ERROR: no practice pair left in {s} {g}')
    return {'order': order, 'items': items, 'practice': practice, 'passed_over': passed, 'notes': notes}


def compute_draw(design_path, vctk_pairs, ltc_pairs, mvc_rows, journal_rows, throwaway_seed=None):
    """The items.json document (as JSON reads it back)."""
    raw = open(design_path, 'rb').read()
    design = check_design(json.loads(raw))
    sets = design.get('sets', list(SETS))
    lists = {'vctk2k': vctk_pairs, 'ltc2k': ltc_pairs}
    pairs, pair_inputs = {}, {}
    for s in sets:
        for r in read_jsonl(lists[s]):
            if r['set'] != s:
                raise SystemExit(f'ERROR: {lists[s]} holds a pair of {r["set"]}')
            pairs[r['pair_id']] = r
        pair_inputs[s] = {'path': shown(lists[s]), 'sha256': sha256(lists[s])}
    rows, row_inputs = load_rows(all_conditions(design), sets, mvc_rows, journal_rows)
    res = draw(design, pairs, rows, throwaway_seed)
    doc = {'version': 2, 'kind': 'mvc_v1 listening items', 'design': design, 'design_file': shown(design_path),
           'design_sha256': sha256_bytes(raw), 'throwaway': throwaway_seed is not None,
           'seed_used': str(design['seed'] if throwaway_seed is None else throwaway_seed),
           'args': {'vctk_pairs': shown(vctk_pairs), 'ltc_pairs': shown(ltc_pairs), 'mvc_rows': shown(mvc_rows),
                    'journal_rows': shown(journal_rows)},
           'rule': 'per set and gender combination: the pair_ids sorted, shuffled with random.Random('
                   '"<seed>:listening:<set>:<genders>"), pairs with P first where prefer_parallel; cells filled in the '
                   'order of SETS x CELLS; the items are the first n_per_stratum pairs of the order that are eligible '
                   'and pass the speaker and sentence rules; every pair passed over is reported with its reason',
           'inputs': {'pairs': pair_inputs, 'rows': row_inputs}, **res}
    return json.loads(json.dumps(doc))


def run_draw(a):
    design = check_design(json.load(open(a.design)))
    lists = {'vctk2k': a.vctk_pairs, 'ltc2k': a.ltc_pairs}
    real = [s for s in design.get('sets', list(SETS)) if sha256(lists[s]) in REAL_PAIR_LISTS]
    freeze = None
    if a.throwaway_seed is not None:
        if str(a.throwaway_seed) == str(design['seed']):
            raise SystemExit('ERROR: --throwaway_seed must differ from the seed of the design')
    elif real:
        freeze = check_freeze(a.addendum, a.addendum_sha256, a.design)
    doc = compute_draw(a.design, a.vctk_pairs, a.ltc_pairs, a.mvc_rows, a.journal_rows, a.throwaway_seed)
    if freeze:
        doc['freeze'] = freeze
    with open(a.out, 'w') as f:
        json.dump(doc, f, indent=1)
        f.write('\n')
    digest = sha256(a.out)
    with open(a.out + '.sha256', 'w') as f:
        f.write(f'{digest}  {os.path.basename(a.out)}\n')
    print(f'draw{" (throwaway seed)" if doc["throwaway"] else ""}: {len(doc["items"])} items, '
          f'{len(doc["practice"])} practice, {len(doc["passed_over"])} pairs passed over -> {a.out} (sha256 {digest})')
    for note in doc['notes']:
        print(f'  note: {note}')
    return doc, digest


# ── the freeze ───────────────────────────────────────────────

def git(*args):
    """(exit code, stdout bytes) of a read-only git command in ROOT."""
    r = subprocess.run(['git', '-C', ROOT, *args], capture_output=True)
    return r.returncode, r.stdout


def remote_tag_commit(listing):
    """The commit of refs/tags/TAG in a git ls-remote listing (the peeled line of an annotated tag), or None."""
    plain = peeled = None
    for line in listing.splitlines():
        sha, _, ref = line.strip().partition('\t')
        if ref == f'refs/tags/{TAG}^{{}}':
            peeled = sha
        elif ref == f'refs/tags/{TAG}':
            plain = sha
    return peeled or plain


def check_freeze(addendum, want_sha, design):
    """The freeze (see the module text); returns what items.json, the key and the manifest record of it."""
    if not addendum or not want_sha:
        raise SystemExit('ERROR: needs --addendum and --addendum_sha256 (the frozen listening addendum)')
    want_sha = want_sha.strip().lower()
    if not os.path.isfile(addendum):
        raise SystemExit(f'ERROR: addendum {addendum} not found')
    if sha256(addendum) != want_sha:
        raise SystemExit(f'ERROR: addendum {addendum} has sha256 {sha256(addendum)}, not {want_sha}')
    code, top = git('rev-parse', '--show-toplevel')
    if code or os.path.realpath(top.decode().strip()) != os.path.realpath(ROOT):
        raise SystemExit(f'ERROR: {ROOT} is not the top of a git repository')
    code, commit = git('rev-parse', '--verify', '--quiet', f'refs/tags/{TAG}^{{commit}}')
    if code:
        raise SystemExit(f'ERROR: the tag {TAG} does not exist: freeze the addendum first')
    commit = commit.decode().strip()
    frozen = {}
    for name, path in (('addendum', addendum), ('builder', SELF), ('design', design)):
        if not inside(path, ROOT):
            raise SystemExit(f'ERROR: the {name} {path} is not inside the repository {ROOT}')
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(ROOT)).replace(os.sep, '/')
        code, blob = git('cat-file', 'blob', f'{commit}:{rel}')
        if code:
            raise SystemExit(f'ERROR: the {name} {rel} is not in the commit {commit[:12]} of the tag {TAG}')
        disk = sha256(path)
        if sha256_bytes(blob) != disk:
            raise SystemExit(f'ERROR: the {name} {rel} differs from its copy at the tag {TAG} (sha256 {disk} on '
                             f'disk, {sha256_bytes(blob)} at the tag)')
        frozen[name] = {'path': rel, 'sha256': disk}
    code, listing = git('ls-remote', '--tags', 'origin')
    if code:
        raise SystemExit(f'ERROR: git ls-remote --tags origin failed (exit {code}): the tag must be on the public '
                         f'remote')
    remote = remote_tag_commit(listing.decode())
    if remote != commit:
        raise SystemExit(f'ERROR: the tag {TAG} on origin is {remote or "missing"}, not {commit[:12]}: push the tag')
    return {'tag': TAG, 'commit': commit, **frozen}


# ── build: checks ────────────────────────────────────────────

def audio_paths(doc):
    return sorted({a['path'] for it in doc['items'] + doc['practice'] for a in it['audio'].values()})


def check_dry_run(doc):
    real = [s for s, v in doc['inputs']['pairs'].items() if v['sha256'] in REAL_PAIR_LISTS]
    remote = [p for p in audio_paths(doc) if p.startswith(REMOTE_PREFIX)]
    if real or remote:
        raise SystemExit(f'ERROR: --dry_run is for synthetic inputs only (real pair lists {real}, '
                         f'{len(remote)} /datastore paths)')


def check_redraw(doc, items_path):
    """Draws again from the design, pair lists and rows named in items.json; refuses unless the result is the same."""
    design = resolve(doc['design_file'])
    if not os.path.isfile(design) or sha256(design) != doc['design_sha256']:
        raise SystemExit(f'ERROR: the design file {design} is missing or changed since the draw')
    again = compute_draw(design, **{k: resolve(v) for k, v in doc['args'].items()})
    diff = [k for k in sorted(set(doc) | set(again)) if k != 'freeze' and doc.get(k) != again.get(k)]
    if diff:
        raise SystemExit(f'ERROR: drawing again from {doc["design_file"]} does not give {items_path} (differs in '
                         f'{diff})')
    return design


def check_places(out, key_dir, cache, manifest):
    for name, p in (('--out', out), ('--key_dir', key_dir), ('--cache', cache)):
        if p and inside(p, ROOT):
            raise SystemExit(f'ERROR: {name} {p} is inside the repository {ROOT}, which is public')
    for name, p in (('--key_dir', key_dir), ('--cache', cache)):
        if p and (inside(p, out) or inside(out, p)):
            raise SystemExit(f'ERROR: {name} {p} and the package {out}, which raters receive, must not contain one '
                             f'another')
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise SystemExit(f'ERROR: {out} already holds a build or other files: move it away first')
    for p in (os.path.join(key_dir, 'key.json'), manifest):
        if os.path.exists(p):
            raise SystemExit(f'ERROR: {p} already exists: move it away first')


def fetch(paths, cache, remote):
    """rsync the absolute ceai paths into cache, each kept below it (read-only on ceai)."""
    os.makedirs(cache, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(''.join(p.lstrip('/') + '\n' for p in paths))
        lst = f.name
    try:
        r = subprocess.run(['rsync', '-a', f'--files-from={lst}', f'{remote}:/', cache.rstrip('/') + '/'],
                           stdout=subprocess.DEVNULL)
    finally:
        os.unlink(lst)
    if r.returncode != 0:
        raise SystemExit(f'ERROR: rsync exit {r.returncode}: nothing built')
    missing = [p for p in paths if not os.path.exists(cache.rstrip('/') + p)]
    if missing:
        raise SystemExit(f'ERROR: {len(missing)} files not copied from ceai, e.g. {missing[0]}: nothing built')


# ── build: trials ────────────────────────────────────────────

def new_id(used):
    while True:
        i = secrets.token_hex(6)
        if i not in used:
            used.add(i)
            return i


def trials(doc):
    """The trial lists of the page (ids only), the clips and the trials of the key. Trial kinds: gold (MOS, nat),
    gold_src (SMOS, src of a cross-gender pair), main, attention, practice."""
    d = doc['design']
    used, by, clips, key_trials = set(), {}, {}, {}

    def clip(item, cond, role):
        k = (item['pair_id'], cond, role)
        if k not in by:
            by[k] = c = new_id(used)
            clips[c] = {'set': item['set'], 'genders': item['genders'], 'pair_id': item['pair_id'],
                        'condition': cond, 'role': role, **item['audio'][cond]}
        return by[k]

    def tid(info):
        t = new_id(used)
        key_trials[t] = info
        return t
    mos, smos, practice = [], [], []
    for it in doc['items']:
        where = {'pair_id': it['pair_id'], 'set': it['set'], 'genders': it['genders'], 'rank': it['rank']}
        half = {'half': 'A' if it['rank'] % 2 == 0 else 'B'} if d['halves'] else {}
        for c in d['mos_conditions']:
            kind = 'gold' if c == 'nat' else 'main'
            mos.append({'tid': tid(dict(block='mos', kind=kind, condition=c, **where)), 'stim': clip(it, c, 'stim'),
                        **half})
        for c in d['smos_conditions']:
            kind = 'gold_src' if c == 'src' and it['genders'] in CROSS else 'main'
            smos.append({'tid': tid(dict(block='smos', kind=kind, condition=c, **where)), 'stim': clip(it, c, 'stim'),
                         'ref': clip(it, 'ref', 'ref'), **half})
    cell = {(it['set'], it['genders'], it['rank']): it for it in doc['items']}
    for block, specs in d.get('attention', {}).items():
        for n, (s, g, r) in enumerate(specs):
            it = cell[(s, g, 1)]
            t = tid({'block': block, 'kind': 'attention', 'expected': r, 'pair_id': it['pair_id'], 'set': s,
                     'genders': g, 'condition': 'nat'})
            entry = {'tid': t, 'stim': clip(it, 'nat', f'att{block}{n}'), 'attention': r}
            if block == 'smos':
                entry['ref'] = clip(it, 'ref', 'ref')
            (mos if block == 'mos' else smos).append(entry)
    for pr in doc['practice']:
        t = tid({'block': 'practice', 'kind': 'practice', 'pair_id': pr['pair_id'], 'set': pr['set'],
                 'genders': pr['genders'], 'condition': pr['condition']})
        practice.append({'tid': t, 'stim': clip(pr, pr['condition'], 'practice')})
    shuffle = random.SystemRandom().shuffle
    shuffle(mos)
    shuffle(smos)
    return {'practice': practice, 'mos': mos, 'smos': smos, 'cmos': []}, clips, key_trials


# ── build: audio ─────────────────────────────────────────────

def load(path, sr=SR):
    """(mono float32 at sr, native rate, native channels): channels averaged, soxr HQ, as librosa.load does."""
    import numpy as np
    import soundfile as sf
    import soxr
    x, sr0 = sf.read(path, dtype='float32', always_2d=True)
    channels = x.shape[1]
    x = x.mean(axis=1)
    if sr0 != sr:
        x = soxr.resample(x, sr0, sr, quality='HQ')
    return np.ascontiguousarray(x, dtype=np.float32), sr0, channels


def active_db(x, sr=SR):
    """Active-speech RMS level (dBFS) of a mono float signal: 20 ms frames within ACTIVE_DB of the loudest."""
    import numpy as np
    n = max(1, int(FRAME_S * sr))
    frames = x[:len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    e = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-20)
    act = frames[e >= e.max() - ACTIVE_DB]
    return float(10 * np.log10(np.mean(act.astype(np.float64) ** 2) + 1e-20))


def normalise(x, sr=SR):
    """(signal at TARGET_DBFS active RMS, gain dB); the gain is lowered so that the peak stays at PEAK_DBFS."""
    import numpy as np
    g = TARGET_DBFS - active_db(x, sr)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0 and 20 * math.log10(peak) + g > PEAK_DBFS:
        g = PEAK_DBFS - 20 * math.log10(peak)
    return (x * 10 ** (g / 20)).astype(np.float32), g


def cut(x, max_s, sr=SR, fade_s=FADE_S):
    import numpy as np
    n = int(max_s * sr)
    if len(x) <= n:
        return x
    y = x[:n].copy()
    f = int(fade_s * sr)
    y[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return y


def pad(x, ms, sr=SR):
    import numpy as np
    return np.concatenate([x, np.zeros(int(round(ms * sr / 1000)), np.float32)])


def browser_dur(seconds):
    return max(DUR_STEP, round(seconds, 1))


def write_private(path, obj):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, indent=1)
        f.write('\n')


def build(doc, items_sha, out, key_dir, cache, freeze, dry_run, manifest_path, config=None):
    """Writes the stimuli and lt_data.json into a temporary folder renamed to out on success, key.json and the
    manifest; cache is the local root of the ceai paths (None: the paths as given)."""
    import soundfile as sf
    d = doc['design']
    for it in doc['items'] + doc['practice']:
        a = it['audio']
        natural = [a[k]['path'] for k in ('nat', 'ref', 'src') if k in a]
        if len(set(natural)) < len(natural) or set(natural) & set(it.get('conversion_ref') or []):
            raise SystemExit(f'ERROR: {it["pair_id"]}: the natural clip, the SMOS reference and the source clip must '
                             f'differ and lie outside R')
        need = all_conditions(d) if 'rank' in it else [it['condition']]
        if not set(need) <= set(a):
            raise SystemExit(f'ERROR: {it["pair_id"]} lacks audio of {sorted(set(need) - set(a))}')
    pages, clips, key_trials = trials(doc)
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    os.makedirs(key_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='.mos-build-', dir=parent)
    key_part, key_path = os.path.join(key_dir, '.key.json.part'), os.path.join(key_dir, 'key.json')
    lo, hi = d['pad_ms']
    pads = {}       # copies of one clip (attention items) get different pads, so their files differ
    try:
        os.mkdir(os.path.join(tmp, 'stimuli'))
        stimuli = {}
        for c in sorted(clips):
            a = clips[c]
            src = cache.rstrip('/') + a['path'] if cache else a['path']
            x, sr0, channels = load(src)
            if a['cut_s']:
                x = cut(x, a['cut_s'])
            y, g = normalise(x)
            taken = pads.setdefault((a['path'], a['cut_s']), set())
            if len(taken) > hi - lo:
                raise SystemExit(f'ERROR: pad_ms {[lo, hi]} has too few values for the copies of {a["path"]}')
            ms = lo + secrets.randbelow(hi - lo + 1)
            while ms in taken:
                ms = lo + secrets.randbelow(hi - lo + 1)
            taken.add(ms)
            y = pad(y, ms)
            f = os.path.join(tmp, 'stimuli', f'{c}.flac')
            sf.write(f, y, SR, subtype='PCM_16', format='FLAC')
            a.update(native_sr=sr0, native_channels=channels, source_sha256=sha256(src), gain_db=round(g, 2),
                     pad_ms=ms, dur=round(len(y) / SR, 3), sha256=sha256(f))
            stimuli[c] = {'src': f'stimuli/{c}.flac', 'dur': browser_dur(len(y) / SR)}
        order = list(stimuli)
        random.SystemRandom().shuffle(order)
        cfg = dict(DEFAULT_CONFIG, **(config or {}), halves=d['halves'])
        if d.get('title'):
            cfg['title'] = d['title']
        data = {'version': 1, 'dry_run': bool(dry_run), 'config': cfg, 'stimuli': {c: stimuli[c] for c in order},
                **pages}
        with open(os.path.join(tmp, 'lt_data.json'), 'w') as f:
            json.dump(data, f, indent=1)
            f.write('\n')
        key = {'version': 2, 'dry_run': bool(dry_run), 'built': datetime.datetime.now().astimezone().isoformat(
            timespec='seconds'), 'items_sha256': items_sha, 'freeze': freeze, 'design': d,
            'design_sha256': doc['design_sha256'], 'inputs': doc['inputs'], 'passed_over': doc['passed_over'],
            'items': [{k: v for k, v in it.items() if k not in ('audio', 'conversion_ref')} for it in doc['items']],
            'practice': [{k: v for k, v in it.items() if k not in ('audio', 'conversion_ref')}
                         for it in doc['practice']],
            'trials': key_trials, 'clips': clips, 'lt_data_sha256': sha256(os.path.join(tmp, 'lt_data.json')),
            'normalisation': {'target_dbfs': TARGET_DBFS, 'peak_dbfs': PEAK_DBFS, 'active_db': ACTIVE_DB,
                              'frame_s': FRAME_S, 'fade_s': FADE_S, 'sr': SR, 'format': 'FLAC PCM_16',
                              'pad_ms': [lo, hi], 'browser_dur_step_s': DUR_STEP}}
        write_private(key_part, key)
        if os.path.isdir(out):
            os.rmdir(out)
        os.rename(tmp, out)
        os.replace(key_part, key_path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(key_part):
            os.unlink(key_part)
        raise
    manifest = {'version': 1, 'kind': 'mvc_v1 listening build', 'dry_run': bool(dry_run), 'built': key['built'],
                'freeze': freeze, 'items_sha256': items_sha, 'design_sha256': doc['design_sha256'],
                'key_sha256': sha256(key_path), 'lt_data_sha256': key['lt_data_sha256'], 'n_clips': len(clips),
                'n_trials': {b: len(pages[b]) for b in ('practice', 'mos', 'smos')},
                'clip_sha256': sorted(a['sha256'] for a in clips.values())}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=1)
        f.write('\n')
    return data, key, manifest


def run_build(a):
    items_sha = sha256(a.items)
    doc = json.load(open(a.items))
    if doc.get('throwaway'):
        raise SystemExit(f'ERROR: {a.items} was drawn with --throwaway_seed: it cannot be built')
    check_design(doc['design'])
    if a.dry_run:
        check_dry_run(doc)
        freeze = None
    else:
        freeze = check_freeze(a.addendum, a.addendum_sha256, resolve(doc['design_file']))
    check_redraw(doc, a.items)
    manifest = a.manifest or (os.path.join(a.key_dir, 'build_manifest.json') if a.dry_run
                              else os.path.join(ROOT, 'mvc', 'LISTENING_BUILD_mvc_v1.json'))
    cache = None if a.dry_run else a.cache
    check_places(a.out, a.key_dir, cache, manifest)
    if cache:
        fetch(audio_paths(doc), cache, a.remote)
    data, key, man = build(doc, items_sha, a.out, a.key_dir, cache, freeze, a.dry_run, manifest)
    print(f'build{" (dry run)" if a.dry_run else ""}: {len(data["stimuli"])} clips, MOS {len(data["mos"])}, SMOS '
          f'{len(data["smos"])}, practice {len(data["practice"])} -> {a.out}; key -> {a.key_dir}/key.json; '
          f'manifest -> {manifest}')
    return data, key, man


def main(argv=None):
    data_dir = os.path.expanduser(os.environ.get('MOS_DATA_DIR', '~/mos_data'))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('draw')
    d.add_argument('--design', required=True, help='design JSON (see the module text)')
    d.add_argument('--vctk_pairs', default=os.path.join(ROOT, 'paper/journal_extension/results/v1/pairs/vctk2k.jsonl'))
    d.add_argument('--ltc_pairs', default=os.path.join(ROOT, 'paper/journal_extension/results/v1/pairs/ltc2k.jsonl'))
    d.add_argument('--mvc_rows', default=os.path.join(ROOT, 'paper/mimi_vc/results/m'))
    d.add_argument('--journal_rows', default=os.path.join(ROOT, 'paper/journal_extension/results/v1/m'))
    d.add_argument('--out', required=True, help='items.json')
    d.add_argument('--addendum', default=None)
    d.add_argument('--addendum_sha256', default=None)
    d.add_argument('--throwaway_seed', default=None, help='another seed string, for a feasibility check only')
    b = sub.add_parser('build')
    b.add_argument('--items', required=True)
    b.add_argument('--addendum', default=None)
    b.add_argument('--addendum_sha256', default=None)
    b.add_argument('--out', default=os.path.join(data_dir, 'package'), help='the package raters receive')
    b.add_argument('--key_dir', default=os.path.join(data_dir, 'private'), help='key.json goes here, never in --out')
    b.add_argument('--cache', default=os.path.join(data_dir, 'private', 'cache'),
                   help='local copies of the ceai audio (their paths name the systems: kept like the key)')
    b.add_argument('--manifest', default=None)
    b.add_argument('--remote', default=REMOTE)
    b.add_argument('--dry_run', action='store_true', help='synthetic inputs only: no freeze check, no fetch')
    a = ap.parse_args(argv)
    if a.cmd == 'draw':
        run_draw(a)
    else:
        run_build(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
