"""Synthetic inputs for the stats.py and select_crt_al.py tests: pair lists shaped like make_pairs.py rows (Dev,
VCTK-2k, LTC-2k), metrics.py rows for every system select_crt_al.SYSTEMS names (with <system>@src rows), and the
evidence files of a selection run. Built in memory or under the test's tmp_path; nothing else is read."""
import json
import os
import zlib

import numpy as np

import select_crt_al as S
import stats

SEED = 7


def wer_counts(e, n, hyp=None):
    """A metrics.py wer_large_v3 entry: e edits on n words (edits beyond n are insertions, as in a Whisper loop);
    characters: 3 e edits on 5 n."""
    s = min(e, n)
    return {'S': s, 'D': 0, 'I': e - s, 'N': n, 'cS': 3 * e, 'cD': 0, 'cI': 0, 'cN': 5 * n,
            'hyp': f'h{e}' if hyp is None else hyp}


def speakers(n_f, n_m, prefix=''):
    return [(f'{prefix}f{i}', 'F') for i in range(n_f)] + [(f'{prefix}m{i}', 'M') for i in range(n_m)]


def _pair(name, pid, src, tgt, prim, par):
    (s, gs), (t, gt) = src, tgt
    return {'pair_id': pid, 'set': name, 'source_speaker': s, 'source_gender': gs, 'target_speaker': t,
            'target_gender': gt, 'genders': f'{gs}2{gt}', 'wer_primary': prim, 'P': par}


def dev_pairs(n=112, n_f=4, n_m=4):
    """Dev: the ordered cells (source != target) of one speaker pool filled in turn up to n pairs; wer_primary and a
    parallel recording P alternate between rounds, never both on a pair (as in pairs/dev.jsonl)."""
    spk = speakers(n_f, n_m)
    cells = [(a, b) for a in spk for b in spk if a != b]
    out = []
    for k in range(n):
        a, b = cells[k % len(cells)]
        prim = (k + k // len(cells)) % 2 == 0
        out.append(_pair('dev', f'dev__{a[0]}_{k:04d}__{b[0]}', a, b, prim, None if prim else f'/x/{b[0]}_{k}.flac'))
    return out


def vctk_pairs(n_targets=5, n_src=(4, 4), n_tgt=(5, 5), utts=4):
    """VCTK-2k: disjoint source and target speakers; each source utterance goes to n_targets targets (the first 5 of
    the 10-target variant are the 5-target list); utterances alternate wer_primary and shared text with P."""
    src, tgt = speakers(*n_src, 's'), speakers(*n_tgt, 't')
    out = []
    for i, a in enumerate(src):
        for u in range(utts):
            prim = u % 2 == 0
            for j in range(n_targets):
                b = tgt[(i + u + j) % len(tgt)]
                out.append(_pair('vctk2k', f'vctk2k__{a[0]}_{u:03d}__{b[0]}', a, b, prim,
                                 None if prim else f'/x/{b[0]}_{u}.flac'))
    return out


def ltc_pairs(n_f=6, n_m=6, utts=2, n_targets=10):
    """LTC-2k: one speaker pool, each utterance to n_targets other speakers, all wer_primary, no parallel recording."""
    spk = speakers(n_f, n_m, 'l')
    out = []
    for i, a in enumerate(spk):
        others = spk[i + 1:] + spk[:i]
        for u in range(utts):
            for j in range(n_targets):
                b = others[(u + j) % len(others)]
                out.append(_pair('ltc2k', f'ltc2k__{a[0]}_{u:03d}__{b[0]}', a, b, True, None))
    return out


# ── metrics rows ─────────────────────────────────────────────

def profiles(scenario='eligible'):
    """{system: (WER %, SIM, UTMOS, dB-MCD, SIM_src same-gender, SIM_src cross-gender)} for every system SYSTEMS names
    (SIM_src None: no @src rows). eligible: C1-C3 meet E1-E4, C4 fails E3 (cross), C5 fails E1; C3 has the lowest WER
    and the highest SIM of C1-C3. f1: every SIM below the 0.80 floor (C5 highest, C4 still fails E3). f2: every
    cross-gender SIM_src above FAISS + 0.01 (C1 the least). f3: every candidate's WER is above S2's: none meets E4."""
    sy = S.SYSTEMS
    p = {sy['S1']: (120, 0.91, 1.7, 8.8, 0.80, 0.62), sy['A5']: (110, 0.91, 1.7, 8.7, 0.80, 0.62)}
    for t, s in sy['S2'].items():
        p[s] = (100 + 20 * t, 0.85 + 0.07 * t, 1.6, 9.0, None, None)
    cand = {'C1': (20, 0.82, 2.2, 9.4, 0.78, 0.605), 'C2': (21, 0.83, 2.6, 9.4, 0.78, 0.606),
            'C3': (15, 0.84, 2.3, 9.5, 0.79, 0.607), 'C4': (22, 0.85, 2.3, 9.2, 0.79, 0.66),
            'C5': (23, 0.78, 2.0, 8.9, 0.79, 0.607)}
    if scenario == 'f1':
        cand = {c: v[:1] + (0.75 + 0.005 * i,) + v[2:] for i, (c, v) in enumerate(cand.items())}
    elif scenario == 'f2':
        cand = {c: v[:5] + (0.66 + 0.01 * i,) for i, (c, v) in enumerate(cand.items())}
    elif scenario == 'f3':
        cand = {c: (170,) + v[1:] for c, v in cand.items()}
    p.update({sy[c]: v for c, v in cand.items()})
    p[sy['C1_anchor']] = cand['C1']     # replaced by an exact copy of C1's rows in system_rows
    p.update({sy['S1u'][0]: (98, 0.93, 2.1, 8.6, None, None), sy['S1u'][1]: (99, 0.94, 2.5, 8.7, None, None),
              sy['E17']: (95, 0.92, 1.8, 8.6, None, None)})
    p.update({'source_copy': (0, 0.74, 4.0, 9.0, 0.95, 0.95), 'target_copy': (2, 0.98, 3.9, 0.5, 0.70, 0.30),
              'mimi_copy_src': (3, 0.73, 3.5, 9.1, 0.93, 0.93), 'mimi_copy_P': (4, 0.95, 3.4, 1.0, 0.70, 0.30)})
    return p


def system_rows(pairs, prof, seed=SEED):
    """{system: {pair_id: row}} with metrics.py's keys; a profile with SIM_src values also gets <system>@src rows.
    Speaker effects are shared by all systems; each system has its own seeded noise. C1's anchor-lab twin is an exact
    copy of C1."""
    rs = np.random.default_rng(seed)
    eff = {s: rs.normal(0, 1) for s in sorted({p[k] for p in pairs for k in ('source_speaker', 'target_speaker')})}
    out = {}
    for name, (w, sim, ut, mcd, ss, sc) in prof.items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        tgt, src = {}, {}
        for k, p in enumerate(pairs):
            n = 10 + k % 7
            e = max(0, int(round(rng.normal(w / 100 * n, 0.3 * n))))
            base = {x: p[x] for x in ('pair_id', 'set', 'wer_primary', 'genders', 'source_speaker', 'target_speaker')}
            r = dict(base, system=name, budget=30, wer_large_v3=wer_counts(e, n, f'h{k}-{e}'),
                     sim_wavlm=sim + 0.02 * eff[p['target_speaker']] + rng.normal(0, 0.03),
                     utmos=ut + 0.1 * eff[p['source_speaker']] + rng.normal(0, 0.2))
            r['secs_resemblyzer'] = r['sim_wavlm'] - 0.1
            if p.get('P'):
                r['mcd_db'] = mcd + rng.normal(0, 0.5)
            tgt[p['pair_id']] = r
            if ss is not None:
                v = (ss if stats.stratum(p['genders']) == 'same' else sc) + 0.02 * eff[p['source_speaker']]
                src[p['pair_id']] = dict(base, system=name + S.SRC, budget=30, sim_wavlm=v + rng.normal(0, 0.03),
                                         target_speaker=p['source_speaker'])
        out[name] = tgt
        if src:
            out[name + S.SRC] = src
    c1, twin = S.SYSTEMS['C1'], S.SYSTEMS['C1_anchor']
    for suf in ('', S.SRC):
        if c1 + suf in out and twin + suf in out:
            out[twin + suf] = {k: dict(r, system=twin + suf) for k, r in out[c1 + suf].items()}
    return out


def set_wer(R, system, pid, e):
    """Replace one row's edit count, keeping its N."""
    r = R[system][pid]
    R[system][pid] = dict(r, wer_large_v3=wer_counts(e, r['wer_large_v3']['N']))


def fail(R, system, pid):
    """Turn one row into a failed conversion (metrics.py writes an error row without metrics)."""
    r = R[system][pid]
    R[system][pid] = {k: r[k] for k in ('pair_id', 'system', 'set', 'budget', 'wer_primary', 'genders')} | {
        'error': 'RuntimeError: synthetic failure'}


# ── files of a selection run ─────────────────────────────────

def _jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return path


def _json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)
    return path


def write_run(root, pairs, R):
    """The files of a complete selection run under root: the Dev pair list, the metrics rows in three files
    (anchor lab, decode lab, controls), the VCTK-2k 5- and 10-target lists (summary.json and vctk2k_source_E.json
    beside each), LTC-2k, sacct output, reproduction checks (passed and failed) and the baselines. Returns the paths
    and the select_crt_al.py options of a complete run, as {option: value}."""
    root = str(root)
    groups = {'d1': lambda s: s.startswith('anc_'), 'ctl': lambda s: s.split('@')[0] in S.SYSTEMS['controls']}
    files = {g: [] for g in ('d1', 'd2', 'ctl')}
    for s, rows in R.items():
        g = next((g for g, f in groups.items() if f(s)), 'd2')
        files[g] += list(rows.values())
    p = {'pairs': _jsonl(os.path.join(root, 'pairs', 'dev.jsonl'), pairs),
         'metrics': [_jsonl(os.path.join(root, 'metrics', f'{g}.jsonl'), rows) for g, rows in files.items()]}
    for key, nt in (('vctk5', 5), ('vctk10', 10)):
        v = vctk_pairs(nt)
        d = os.path.join(root, key)
        p[key] = _jsonl(os.path.join(d, 'vctk2k.jsonl'), v)
        _json(os.path.join(d, 'summary.json'), {'sets': {'vctk2k': {'n_pairs_wer_primary': sum(
            x['wer_primary'] for x in v)}}})
        srcs = sorted({x['source_speaker'] for x in v})
        _json(os.path.join(d, 'vctk2k_source_E.json'), {s: {'paths': [f'/x/{s}_{i}.flac' for i in range(3)],
                                                             'durs': [10.0] * 3, 'total_s': 30.0} for s in srcs[1:]})
    p['ltc'] = _jsonl(os.path.join(root, 'ltc', 'ltc2k.jsonl'), ltc_pairs())
    p['sacct'] = os.path.join(root, 'sacct.txt')
    with open(p['sacct'], 'w') as f:
        f.write('JobID|JobName|Submit|End|State\n2634|crt_anchor|2026-10-11T06:31:00|2026-10-11T06:58:00|COMPLETED\n'
                '3001|d0|2026-10-20T10:00:00|2026-10-20T11:00:00|COMPLETED\n'
                '3010_0|d1|2026-10-21T10:00:00|2026-10-21T20:00:00|COMPLETED\n'
                '3010_1|d1|2026-10-21T10:00:00|2026-10-21T21:00:00|COMPLETED\n'
                '3020|d2|2026-10-22T10:00:00|2026-10-22T14:00:00|COMPLETED\n')
    p['repro_ok'] = _json(os.path.join(root, 'repro_ok.json'), {'passed': True, 'summary': 'synthetic: reproduced'})
    p['repro_bad'] = _json(os.path.join(root, 'repro_bad.json'), {'passed': False, 'summary': 'synthetic mismatch'})
    p['baselines'] = _json(os.path.join(root, 'baselines.json'), [
        {'id': 'S5', 'system': 'kNN-VC', 'repo_commit': 'abc123', 'checkpoint_sha256': 'deadbeef',
         'sample_rate': 16000, 'command': 'python knnvc.py --k 4'}])
    p['out'] = os.path.join(root, 'out', 'RECORD.md')
    opts = {'--pairs': p['pairs'], '--metrics': ','.join(p['metrics']), '--vctk_pairs': p['vctk5'],
            '--vctk10_pairs': p['vctk10'], '--ltc_pairs': p['ltc'], '--repro': p['repro_ok'], '--sacct': p['sacct'],
            '--jobs': '2634=2634,D0=3001,D1=3010,D2=3020', '--baselines': p['baselines'], '--out': p['out']}
    return p, opts


def argv(opts, **over):
    """select_crt_al.py's argv from {option: value}; over: option without the dashes -> value, None drops it."""
    o = dict(opts, **{'--' + k: v for k, v in over.items()})
    return ['select_crt_al.py'] + [x for k, v in o.items() if v is not None for x in (k, v)]
