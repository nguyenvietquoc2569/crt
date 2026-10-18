#!/usr/bin/env python3
"""
H1 — Build the Dev, VCTK-2k and LTC-2k pair lists of benchmark/ANALYSIS_PLAN.md §1.

Fixed seed 20260928, VCTK mic1 only, VCTK genders from speaker-info.txt.
  - Dev     : the 20 SPIE conference speakers (read from eval_pairs.json), 10 source utterances each
              (half shared-text, half speaker-specific), 5 gender-balanced targets per source utterance.
  - VCTK-2k : 20 source + 20 target speakers (10F/10M each), disjoint from each other and from Dev; p315 and s5
              excluded; 20 source utterances per source speaker (up to 10 from the shared-text block, the rest
              speaker-specific) x 5 gender-balanced targets = 2,000 pairs.
  - LTC-2k  : LibriTTS-R test-clean, 5 utterances of 3-10 s per speaker x 10 gender-balanced targets.
Per target speaker three disjoint sets: R (reference pool, nested budgets; each budget lists whole utterances until
>= budget plus keep_s, the seconds of each utterance to use, so the concatenation is cut to exactly budget_s),
E (SIM enrollment), P (parallel recordings of the shared-text sentences used as sources). R and E never contain a
shared-block recording, a recording of any sentence sent to that target, or any source recording of the set.
E is drawn first from the shuffled pool and stops at n_e utterances or e_max_s seconds, whichever comes first; R
takes the rest. A target whose pool cannot fill a budget has R[b]['shortfall'] = True; load_reference() raises on it,
so a budget column (E16 ~180 s on LTC) covers only targets without a shortfall (listed in summary.json).
Each row carries wer_primary: True for speaker-specific sources, False for VCTK shared-text sources. M3 restricts VCTK
primary WER to wer_primary rows. ANALYSIS_PLAN §4.1 does not say whether the Dev τ-rule WER uses wer_primary rows only
or all Dev rows; that choice is recorded in v1.
Writes <set>.jsonl (one row per pair) and summary.json. Systems read R through load_reference(), never R[b]['paths'].
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics
import time

import soundfile as sf

D = '/datastore/nvquoc/crt/mimi_vc'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vctk_dir', default=f'{D}/benchmark/vctk')
    p.add_argument('--conf_pairs_json', default=f'{D}/benchmark/vctk/eval_pairs.json')
    p.add_argument('--ltc_root', default=f'{D}/data_storage/dataset_raw/LibriTTS_R_eval')
    p.add_argument('--ltc_speakers', default=f'{D}/data_storage/dataset_raw/LibriTTS_R/train-clean-100/speakers.tsv',
                   help='LibriTTS speakers.tsv or SPEAKERS.txt (covers all subsets); used if none is found under ltc_root')
    p.add_argument('--output_dir', default=f'{D}/benchmark/journal/pairs')
    p.add_argument('--sets', default='dev,vctk2k,ltc2k')
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--exclude', default='p315,s5')
    p.add_argument('--shared_min_frac', type=float, default=0.8,
                   help='a normalised text read by >= this fraction of speakers belongs to the shared-text block')
    p.add_argument('--budgets', default='3,30,180', help='R budgets in seconds (30 = main tables)')
    p.add_argument('--n_e', type=int, default=10, help='max SIM enrollment utterances per target (extension_plan §4.1)')
    p.add_argument('--e_max_s', type=float, default=30.0,
                   help='E stops at n_e utterances or once its total reaches this many seconds, whichever comes first')
    p.add_argument('--min_ref_dur', type=float, default=1.0, help='min duration of an R or E utterance (s)')
    p.add_argument('--dev_utts', type=int, default=10)
    p.add_argument('--dev_shared', type=int, default=5)
    p.add_argument('--dev_targets', type=int, default=5)
    p.add_argument('--vctk_spk_per_gender', type=int, default=10)
    p.add_argument('--vctk_utts', type=int, default=20)
    p.add_argument('--vctk_shared', type=int, default=10)
    p.add_argument('--vctk_targets', type=int, default=5)
    p.add_argument('--ltc_utts', type=int, default=5)
    p.add_argument('--ltc_targets', type=int, default=10)
    p.add_argument('--ltc_min_dur', type=float, default=3.0)
    p.add_argument('--ltc_max_dur', type=float, default=10.0)
    return p.parse_args()


def rng(seed, *tags):
    return random.Random(':'.join(str(t) for t in (seed,) + tags))


def norm_text(s):
    s = re.sub(r'[^a-z0-9\s]', ' ', s.lower())
    return ' '.join(s.split())


_DUR = {}


def dur(path):
    if path not in _DUR:
        _DUR[path] = round(sf.info(path).duration, 3)
    return _DUR[path]


def stats(xs):
    xs = list(xs)
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'min': round(min(xs), 3), 'mean': round(statistics.mean(xs), 3),
            'median': round(statistics.median(xs), 3), 'max': round(max(xs), 3), 'total_h': round(sum(xs) / 3600, 3)}


# ── VCTK ─────────────────────────────────────────────────────

def read_vctk_genders(path):
    g = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and re.fullmatch(r'[ps]\d+', p[0]):
            g[p[0]] = p[2]
    return g


def load_vctk(vctk_dir, speakers):
    """speaker -> list of utterances {uid, path, text, norm} having both a transcript and a mic1 recording."""
    cat, bad = {}, {}
    for spk in sorted(speakers):
        tdir, wdir = os.path.join(vctk_dir, 'txt', spk), os.path.join(vctk_dir, 'wav48_silence_trimmed', spk)
        if not os.path.isdir(tdir):
            bad[spk] = 'no txt'
            continue
        mic1 = {f[:-len('_mic1.flac')] for f in os.listdir(wdir) if f.endswith('_mic1.flac')} if os.path.isdir(wdir) else set()
        if not mic1:
            bad[spk] = 'no mic1 audio'
            continue
        utts = []
        for f in sorted(os.listdir(tdir)):
            stem = f[:-4]
            if f.endswith('.txt') and stem in mic1:
                text = open(os.path.join(tdir, f), errors='replace').read().strip()
                utts.append({'uid': stem, 'path': os.path.join(wdir, stem + '_mic1.flac'), 'text': text, 'norm': norm_text(text)})
        if not utts:
            bad[spk] = 'no utterance with both txt and mic1'
            continue
        cat[spk] = utts
    return cat, bad


def detect_shared_block(cat, min_frac):
    """Texts read by >= min_frac of the speakers. Returns the block description and tags every utterance with
    'shared' (block sentence id if the text is the canonical block text, else None) and 'in_block' (its number is a
    block number or its text is a block text; such recordings are never sources of speaker-specific sentences,
    never R and never E)."""
    spk_by_text, ids_by_text = collections.defaultdict(set), collections.defaultdict(collections.Counter)
    for spk, utts in cat.items():
        for u in utts:
            spk_by_text[u['norm']].add(spk)
            ids_by_text[u['norm']][u['uid'].split('_', 1)[1]] += 1
    n, thr = len(cat), min_frac * len(cat)
    block = {t: ids_by_text[t].most_common(1)[0][0] for t, s in spk_by_text.items() if len(s) >= thr}
    block_nums = set(block.values())
    assert len(block_nums) == len(block), 'two block texts share a modal utterance number'
    for utts in cat.values():
        for u in utts:
            num = u['uid'].split('_', 1)[1]
            u['shared'] = block.get(u['norm'])
            u['in_block'] = u['shared'] is not None or num in block_nums
    sentences = sorted(({'id': i, 'text': t, 'n_speakers': len(spk_by_text[t]),
                         'numbers_seen': dict(ids_by_text[t])} for t, i in block.items()), key=lambda x: x['id'])
    return {'n_speakers_indexed': n, 'min_frac': min_frac, 'n_sentences': len(block), 'ids': sorted(block_nums),
            'next_largest_text_speaker_count': max((len(s) for s in spk_by_text.values() if len(s) < thr), default=0),
            'n_in_block_recordings_non_canonical_text': sum(1 for us in cat.values() for u in us if u['in_block'] and u['shared'] is None),
            'sentences': sentences}


# ── LibriTTS-R ───────────────────────────────────────────────

def read_libritts_genders(paths):
    for path in paths:
        if not os.path.exists(path):
            continue
        g = {}
        for line in open(path):
            if line.startswith(';') or line.startswith('READER'):
                continue
            p = [x.strip() for x in (line.split('|') if '|' in line else line.split('\t'))]
            if len(p) >= 2 and p[0].isdigit():
                g[p[0]] = p[1]
        return g, path
    return {}, None


def load_ltc(ltc_dir):
    cat = {}
    for spk in sorted(os.listdir(ltc_dir)):
        utts = []
        for dirpath, _, files in sorted(os.walk(os.path.join(ltc_dir, spk))):
            for f in sorted(files):
                if not f.endswith('.wav'):
                    continue
                stem = f[:-4]
                tpath = os.path.join(dirpath, stem + '.normalized.txt')
                if not os.path.exists(tpath):
                    continue
                text = open(tpath, errors='replace').read().strip()
                utts.append({'uid': stem, 'path': os.path.join(dirpath, f), 'text': text, 'norm': norm_text(text),
                             'shared': None, 'in_block': False})
        if utts:
            cat[spk] = utts
    return cat


# ── Pairing ──────────────────────────────────────────────────

class TargetRotation:
    """Cyclic pointer per gender over a seeded permutation of the targets; skips the source speaker."""

    def __init__(self, targets, genders, seed, tag):
        self.pools, self.ptr = {}, {}
        for g in sorted({genders[t] for t in targets}):
            pool = sorted(t for t in targets if genders[t] == g)
            rng(seed, tag, 'targets', g).shuffle(pool)
            self.pools[g], self.ptr[g] = pool, 0

    def pick(self, src, n, parity):
        assert sorted(self.pools) == ['F', 'M'], f'target genders are {sorted(self.pools)}, not F/M'
        n_f = n // 2 + (n % 2 if parity % 2 == 0 else 0)
        want = [('F', n_f), ('M', n - n_f)]
        out = []
        for g, k in want:
            pool = self.pools[g]
            for _ in range(2 * len(pool)):
                if k == 0:
                    break
                t = pool[self.ptr[g] % len(pool)]
                self.ptr[g] += 1
                if t != src and t not in out:
                    out.append(t)
                    k -= 1
            assert k == 0, f'not enough {g} targets for {src}'
        return out


def make_sources_and_pairs(name, sources, targets, genders, cat, n_utts, n_shared, n_targets, seed, src_ok=None):
    rot = TargetRotation(targets, genders, seed, name)
    shared_idx = {s: {u['shared']: u for u in cat[s] if u['shared'] is not None} for s in targets}
    pairs, shared_fallback = [], 0
    for i, s in enumerate(sorted(sources)):
        r = rng(seed, name, 'source', s)
        specific = sorted((u for u in cat[s] if not u['in_block']), key=lambda u: u['uid'])
        shared = sorted((u for u in cat[s] if u['shared'] is not None), key=lambda u: u['uid'])
        r.shuffle(specific)
        r.shuffle(shared)
        specific = iter(specific)
        for k in range(n_utts):
            tg = rot.pick(s, n_targets, i + k)
            u = None
            if k < n_shared:
                u = next((x for x in shared if all(x['shared'] in shared_idx[t] for t in tg)), None)
                if u is None:
                    shared_fallback += 1
                else:
                    shared.remove(u)
            if u is None:
                u = next(x for x in specific if src_ok is None or src_ok(x))
            for t in tg:
                p = shared_idx[t][u['shared']] if u['shared'] is not None else None
                pairs.append({'pair_id': f'{name}__{u["uid"]}__{t}', 'set': name,
                              'source_speaker': s, 'source_gender': genders[s], 'source_utt': u['uid'],
                              'source_path': u['path'], 'source_text': u['text'], 'source_norm': u['norm'],
                              'source_dur': dur(u['path']), 'source_is_shared_text': u['shared'] is not None,
                              'shared_sentence_id': u['shared'], 'wer_primary': u['shared'] is None,
                              'target_speaker': t, 'target_gender': genders[t], 'genders': f'{genders[s]}2{genders[t]}',
                              'P': p['path'] if p else None, 'P_text': p['text'] if p else None,
                              'P_dur': dur(p['path']) if p else None})
    return pairs, shared_fallback


def build_refs(name, cat, pairs, budgets, n_e, e_max_s, min_dur, seed):
    """Per target: E (n_e utterances or e_max_s seconds, whichever first) then R (nested prefixes until >= each
    budget) from the eligible pool."""
    src_paths = {p['source_path'] for p in pairs}
    by_t = collections.defaultdict(list)
    for p in pairs:
        by_t[p['target_speaker']].append(p)
    refs = {}
    for t, ps in sorted(by_t.items()):
        src_norms = {p['source_norm'] for p in ps}
        pool, dropped = [], collections.Counter()
        for u in sorted(cat[t], key=lambda u: u['uid']):
            if u['in_block']:
                dropped['shared_block'] += 1
            elif u['path'] in src_paths:
                dropped['is_a_source'] += 1
            elif u['norm'] in src_norms:
                dropped['same_text_as_a_source'] += 1
            else:
                pool.append(u)
        rng(seed, name, 'refs', t).shuffle(pool)
        E, R, tot, e_tot = [], [], 0.0, 0.0
        for u in pool:
            if tot >= max(budgets):
                break
            if dur(u['path']) < min_dur:
                dropped['too_short'] += 1
                continue
            if len(E) < n_e and e_tot < e_max_s:
                E.append(u)
                e_tot += dur(u['path'])
            else:
                R.append(u)
                tot += dur(u['path'])
        assert len(E) == n_e or e_tot >= e_max_s, f'{name}: target {t} has only {len(E)} E utterances ({e_tot:.1f} s)'
        rb = {}
        for b in budgets:
            sel, keep, acc = [], [], 0.0
            for u in R:
                if acc >= b:
                    break
                sel.append(u)
                keep.append(round(min(dur(u['path']), b - acc), 3))
                acc += dur(u['path'])
            rb[str(b)] = {'budget_s': b, 'paths': [u['path'] for u in sel], 'durs': [dur(u['path']) for u in sel],
                          'keep_s': keep, 'total_s': round(sum(keep), 3), 'uncut_total_s': round(acc, 3),
                          'shortfall': acc < b}
        refs[t] = {'R': rb, 'E': {'paths': [u['path'] for u in E], 'durs': [dur(u['path']) for u in E],
                                  'total_s': round(sum(dur(u['path']) for u in E), 3)},
                   'P_all': sorted({p['P'] for p in ps if p['P']}), 'dropped': dict(dropped), '_R': R, '_E': E}
    return refs


def check_set(name, pairs, refs, cat, expected_pairs=None):
    """Assert every disjointness property of ANALYSIS_PLAN §1 (reference rule)."""
    text_of = {u['path']: u['norm'] for us in cat.values() for u in us}
    block_of = {u['path']: u['in_block'] for us in cat.values() for u in us}
    src_paths = {p['source_path'] for p in pairs}
    assert len({p['pair_id'] for p in pairs}) == len(pairs), 'duplicate pair_id'
    if expected_pairs is not None:
        assert len(pairs) == expected_pairs, f'{name}: {len(pairs)} pairs, expected {expected_pairs}'
    for t, rf in refs.items():
        R = [u['path'] for u in rf['_R']]
        E = {u['path'] for u in rf['_E']}
        P = set(rf['P_all'])
        assert len(set(R)) == len(R) and not (set(R) & E) and not (set(R) & P) and not (E & P), f'{t}: R/E/P overlap'
        assert not ((set(R) | E) & src_paths), f'{t}: a source recording is in R or E'
        assert not any(block_of[x] for x in set(R) | E), f'{t}: shared-block recording in R or E'
        b = sorted(rf['R'], key=float)
        for r in rf['R'].values():
            assert r['keep_s'][:-1] == r['durs'][:-1] and 0 < r['keep_s'][-1] <= r['durs'][-1], f'{t}: bad keep_s'
            assert abs(r['total_s'] - min(r['budget_s'], r['uncut_total_s'])) < 0.01, f'{t}: R not cut to budget'
        for lo, hi in zip(b, b[1:]):
            assert rf['R'][hi]['paths'][:len(rf['R'][lo]['paths'])] == rf['R'][lo]['paths'], f'{t}: budgets not nested'
    for p in pairs:
        rf = refs[p['target_speaker']]
        assert p['source_speaker'] != p['target_speaker']
        assert os.path.basename(p['source_path']).split('_')[0] == p['source_speaker']
        ref_paths = [x for r in rf['R'].values() for x in r['paths']] + rf['E']['paths']
        assert all(text_of[x] != p['source_norm'] for x in ref_paths), f'{p["pair_id"]}: same sentence in R or E'
        assert p['wer_primary'] == (not p['source_is_shared_text'])
        if p['source_is_shared_text']:
            assert p['P'] and text_of[p['P']] == p['source_norm'] and p['P'] not in ref_paths, f'{p["pair_id"]}: bad P'
        else:
            assert p['P'] is None


def summarize(name, pairs, refs, extra):
    cnt = lambda key: dict(sorted(collections.Counter(p[key] for p in pairs).items()))
    src_utts = {p['source_path']: p for p in pairs}
    per_tgt = collections.Counter(p['target_speaker'] for p in pairs)
    g_src = {p['source_speaker']: p['source_gender'] for p in pairs}
    g_tgt = {p['target_speaker']: p['target_gender'] for p in pairs}
    out = {'n_pairs': len(pairs), 'n_source_utts': len(src_utts),
           'n_source_speakers': len(g_src), 'source_speakers': dict(sorted(g_src.items())),
           'source_speaker_genders': dict(collections.Counter(g_src.values())),
           'n_target_speakers': len(g_tgt), 'target_speakers': dict(sorted(g_tgt.items())),
           'target_speaker_genders': dict(collections.Counter(g_tgt.values())),
           'gender_pairs': cnt('genders'),
           'pairs_per_target': {'min': min(per_tgt.values()), 'max': max(per_tgt.values())},
           'n_pairs_shared_text': sum(p['source_is_shared_text'] for p in pairs),
           'n_pairs_wer_primary': sum(p['wer_primary'] for p in pairs),
           'n_distinct_source_texts_wer_primary': len({p['source_norm'] for p in pairs if p['wer_primary']}),
           'n_source_utts_shared_text': sum(p['source_is_shared_text'] for p in src_utts.values()),
           'shared_utts_per_source_speaker': dict(sorted(collections.Counter(
               p['source_speaker'] for p in src_utts.values() if p['source_is_shared_text']).items())),
           'n_E_utts_per_target': stats(len(r['E']['paths']) for r in refs.values()),
           'durations_s': {'source': stats(p['source_dur'] for p in src_utts.values()),
                           'P': stats(p['P_dur'] for p in pairs if p['P']),
                           'E_total_per_target': stats(r['E']['total_s'] for r in refs.values()),
                           'E_utt': stats(d for r in refs.values() for d in r['E']['durs'])},
           'R_per_target': {b: {'total_s': stats(r['R'][b]['total_s'] for r in refs.values()),
                                'uncut_total_s': stats(r['R'][b]['uncut_total_s'] for r in refs.values()),
                                'n_utts': stats(len(r['R'][b]['paths']) for r in refs.values()),
                                'n_shortfall': sum(r['R'][b]['shortfall'] for r in refs.values()),
                                'shortfall_targets': {t: r['R'][b]['uncut_total_s'] for t, r in sorted(refs.items())
                                                      if r['R'][b]['shortfall']}}
                            for b in next(iter(refs.values()))['R']},
           'ref_pool_dropped_total': dict(sum((collections.Counter(r['dropped']) for r in refs.values()), collections.Counter()))}
    out.update(extra)
    return out


def load_reference(entry, sr, allow_shortfall=False):
    """The one loader for an R[b] entry: each utterance read for its keep_s seconds, concatenated, cut to budget_s.
    Raises on a shortfall entry (pool shorter than budget_s) unless allow_shortfall; budget columns skip those targets."""
    if entry['shortfall'] and not allow_shortfall:
        raise ValueError(f'reference shortfall: {entry["uncut_total_s"]} s available for budget {entry["budget_s"]} s')
    import librosa
    import numpy as np
    wav = np.concatenate([librosa.load(p, sr=sr, mono=True, duration=k)[0] for p, k in zip(entry['paths'], entry['keep_s'])])
    wav = wav[:int(round(entry['budget_s'] * sr))]
    check_reference_len(len(wav), sr, entry['budget_s'])
    return wav


def check_reference_len(n_samples, sr, budget_s, hop=0):
    """Conversion harnesses call this on the reference actually fed to a system (hop = that system's frame size)."""
    assert n_samples <= budget_s * sr + hop, f'reference {n_samples / sr:.3f} s exceeds budget {budget_s} s'


def write_set(name, pairs, refs, out_dir):
    path = os.path.join(out_dir, f'{name}.jsonl')
    with open(path, 'w') as f:
        for p in pairs:
            rf = refs[p['target_speaker']]
            row = {k: v for k, v in p.items() if k != 'source_norm'}
            row['R'] = rf['R']
            row['E'] = rf['E']
            f.write(json.dumps(row) + '\n')
    return path, hashlib.sha256(open(path, 'rb').read()).hexdigest()


def run_set(name, sources, targets, genders, cat, n_utts, n_shared, n_targets, args, budgets, extra, src_ok=None,
            expected=None):
    t0 = time.time()
    pairs, fallback = make_sources_and_pairs(name, sources, targets, genders, cat, n_utts, n_shared, n_targets,
                                             args.seed, src_ok)
    refs = build_refs(name, cat, pairs, budgets, args.n_e, args.e_max_s, args.min_ref_dur, args.seed)
    check_set(name, pairs, refs, cat, expected)
    path, sha = write_set(name, pairs, refs, args.output_dir)
    extra = dict(extra, shared_slots_fallen_back_to_specific=fallback, jsonl=path, sha256=sha)
    s = summarize(name, pairs, refs, extra)
    assert sorted(s['gender_pairs']) == ['F2F', 'F2M', 'M2F', 'M2M'] and len(set(s['gender_pairs'].values())) == 1, \
        f'{name}: gender pairs not balanced {s["gender_pairs"]}'
    print(f'[{name}] {s["n_pairs"]} pairs, {s["n_source_utts"]} source utts ({s["n_source_utts_shared_text"]} shared-text), '
          f'{s["n_pairs_wer_primary"]} wer_primary pairs, {s["n_source_speakers"]} src spk {s["source_speaker_genders"]}, {s["n_target_speakers"]} tgt spk '
          f'{s["target_speaker_genders"]}, genders {s["gender_pairs"]}, pairs/target {s["pairs_per_target"]}, '
          f'R30 {s["R_per_target"]["30"]["total_s"]} (uncut max {s["R_per_target"]["30"]["uncut_total_s"]["max"]}), '
          f'E utts {s["n_E_utts_per_target"]}, E total {s["durations_s"]["E_total_per_target"]}, shortfalls '
          f'{ {b: v["shortfall_targets"] for b, v in s["R_per_target"].items()} } '
          f'-> {path} ({time.time() - t0:.0f}s)', flush=True)
    return pairs, s


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sets = args.sets.split(',')
    budgets = [int(b) for b in args.budgets.split(',')]
    exclude = set(args.exclude.split(','))
    summary = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'args': vars(args), 'sets': {}}

    genders = read_vctk_genders(os.path.join(args.vctk_dir, 'speaker-info.txt'))
    cat, bad = load_vctk(args.vctk_dir, [s for s in genders if s not in exclude])
    block = detect_shared_block(cat, args.shared_min_frac)
    summary['vctk'] = {'n_speakers_in_info': len(genders), 'excluded_by_rule': sorted(exclude),
                       'excluded_missing_data': bad, 'n_indexed': len(cat), 'shared_text_block': block}
    print(f'VCTK: {len(genders)} speakers in speaker-info, excluded {sorted(exclude)} by rule, {bad} for missing data, '
          f'{len(cat)} indexed', flush=True)
    print(f'Shared-text block: {block["n_sentences"]} sentences (read by >= {args.shared_min_frac:.0%} of speakers; '
          f'next most common text is read by {block["next_largest_text_speaker_count"]}), ids {block["ids"]}, '
          f'speakers per sentence {[x["n_speakers"] for x in block["sentences"]]}', flush=True)

    conf = json.load(open(args.conf_pairs_json))
    dev_spk = sorted({x['source_speaker'] for x in conf} | {x['target_speaker'] for x in conf})
    assert all(s in cat for s in dev_spk), 'a conference speaker lacks data'
    assert collections.Counter(genders[s] for s in dev_spk) == {'F': 10, 'M': 10}, 'Dev is not 10F/10M'

    if 'dev' in sets:
        n = len(dev_spk) * args.dev_utts * args.dev_targets
        _, summary['sets']['dev'] = run_set('dev', dev_spk, dev_spk, genders, cat, args.dev_utts, args.dev_shared,
                                            args.dev_targets, args, budgets, {'speakers_from': args.conf_pairs_json},
                                            expected=n)

    if 'vctk2k' in sets:
        k = args.vctk_spk_per_gender
        src, tgt, pool_sizes = [], [], {}
        for g in ('F', 'M'):
            pool = sorted(s for s in cat if genders[s] == g and s not in dev_spk)
            pool_sizes[g] = len(pool)
            rng(args.seed, 'vctk2k', 'speakers', g).shuffle(pool)
            assert len(pool) >= 2 * k, f'only {len(pool)} {g} speakers'
            src += pool[:k]
            tgt += pool[k:2 * k]
        assert not set(src) & set(tgt) and not (set(src) | set(tgt)) & set(dev_spk) and not (set(src) | set(tgt)) & exclude
        n = len(src) * args.vctk_utts * args.vctk_targets
        pairs, s = run_set('vctk2k', src, tgt, genders, cat, args.vctk_utts, args.vctk_shared, args.vctk_targets, args,
                           budgets, {'eligible_speaker_pool_non_dev': pool_sizes}, expected=n)
        assert s['pairs_per_target']['min'] == s['pairs_per_target']['max'] == n // len(tgt)
        summary['sets']['vctk2k'] = s

    if 'ltc2k' in sets:
        ltc_dir = os.path.join(args.ltc_root, 'LibriTTS_R', 'test-clean')
        if not (os.path.isdir(ltc_dir) and os.path.exists(os.path.join(args.ltc_root, 'test_clean.done'))):
            print(f'[ltc2k] skipped: not downloaded yet ({ltc_dir})', flush=True)
            summary['sets']['ltc2k'] = {'skipped': 'not downloaded yet', 'ltc_dir': ltc_dir}
        else:
            lcat = load_ltc(ltc_dir)
            cands = [os.path.join(args.ltc_root, 'LibriTTS_R', f) for f in ('speakers.tsv', 'SPEAKERS.txt')] + [args.ltc_speakers]
            lg_all, gsrc = read_libritts_genders(cands)
            lg = {s: lg_all.get(s, 'unknown') for s in lcat}
            ok = lambda u: args.ltc_min_dur <= dur(u['path']) <= args.ltc_max_dur
            n_ok = {s: sum(ok(u) for u in us) for s, us in lcat.items()}
            too_few = {s: {'n_utts': len(lcat[s]), 'n_in_duration_range': n_ok[s]} for s in lcat if n_ok[s] < args.ltc_utts}
            spk = sorted((s for s in lcat if s not in too_few), key=int)
            unknown = [s for s in spk if lg[s] not in ('F', 'M')]
            assert not unknown, f'[ltc2k] no F/M gender for {unknown} in {gsrc}'
            print(f'[ltc2k] {len(lcat)} speakers indexed; excluded (fewer than {args.ltc_utts} utterances of '
                  f'{args.ltc_min_dur:g}-{args.ltc_max_dur:g} s): {too_few}', flush=True)
            lpairs, summary['sets']['ltc2k'] = run_set(
                'ltc2k', spk, spk, lg, lcat, args.ltc_utts, 0, args.ltc_targets, args, budgets,
                {'ltc_dir': ltc_dir, 'gender_source': gsrc, 'n_utts_indexed': sum(len(v) for v in lcat.values()),
                 'n_speakers_indexed': len(lcat), 'excluded_too_few_source_utts': too_few},
                src_ok=ok, expected=len(spk) * args.ltc_utts * args.ltc_targets)
            assert all(args.ltc_min_dur <= p['source_dur'] <= args.ltc_max_dur for p in lpairs)

    all_dev = set(summary['sets'].get('dev', {}).get('source_speakers', {}))
    all_v2k = set(summary['sets'].get('vctk2k', {}).get('source_speakers', {})) | \
        set(summary['sets'].get('vctk2k', {}).get('target_speakers', {}))
    assert not all_dev & all_v2k, 'Dev and VCTK-2k share a speaker'
    out = os.path.join(args.output_dir, 'summary.json')
    json.dump(summary, open(out, 'w'), indent=1)
    print(f'Summary -> {out}', flush=True)


if __name__ == '__main__':
    main()
