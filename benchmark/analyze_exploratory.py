#!/usr/bin/env python3
"""
Exploratory addition X1 (DEVIATIONS_v1.md item 5), decided on 21 Oct 2026 after the test results (TEST_RESULTS_v1.md):
CRT-AL against each baseline at the 3 s reference budget, on the six endpoints of family S (capped WER, SIM and UTMOS
on VCTK-2k and LTC-2k). It is not part of the pre-registered analysis (ANALYSIS_PLAN_v1.md §5, §7): it decides
nothing, it is reported whatever it shows, and every table says that it is exploratory. kNN-VC's 3 s outputs (job T4,
the E16 budget sweep) were already seen when X1 was decided; FreeVC, Seed-VC and FACodec at 3 s come from job X1.

The statistics are analyze_tests.py's, imported unchanged (TestSet, load_set, check_done, the quantity table Q, the
S endpoints, Holm and the decision flags), with the budget set to 3 s: Δ = CRT-AL − baseline over the endpoint's pairs
where neither system failed, crossed bootstrap B = 10,000, seed 20260928, VCTK-2k independent, LTC-2k linked. Holm over
the 24 comparisons is printed as a guide, not as the error control of a pre-registered family. Sensitivity: the same
comparisons with uncapped WER. Descriptive: every system's 3 s levels (WER, SIM, UTMOS, SIM_src, M9 with the share
D > 0) and EER.

Inputs: --m_dir (job M: parts t1-t4 done; t3 holds CRT-AL, FAISS and the gated CRT at 3 s, t4 kNN-VC at 3 s), --x1_dir
(job X1: m/metrics/<set>_3{,_src}.jsonl and x1.done, written under the same selection.env as job M), the two pair
lists, the VCTK-2k source-E file and the selection record. CRT-AL and every baseline need one row per pair at 3 s.
Writes --out (.md) and <out stem>.json with the sha256 of every input.
"""
import argparse
import datetime
import json
import os
import re
import sys

import analyze_tests as at
import stats

BUDGET = 3
X1_SYSTEMS = ('freevc', 'seedvc', 'facodec')
HEAD = ('| Comparison | Set | Endpoint | CRT-AL | Baseline | Δ = CRT-AL − baseline [95 % CI] | p | Holm over 24 '
        '(guide only) | n pairs | Wilcoxon p (r_rb) |\n|---|---|---|---|---|---|---|---|---|---|')
NOTES = ['**Exploratory.** Decided on 21 Oct 2026, after the test results; not in ANALYSIS_PLAN_v1.md, so it changes '
         'no §7 decision (DEVIATIONS_v1.md item 5).',
         'kNN-VC at 3 s is the E16 budget-sweep output of job T4, seen before X1 was decided; FreeVC, Seed-VC and '
         'FACodec at 3 s are job X1, converted with the adapters\' fixed settings as in T4.',
         'At 3 s every system sees the same 3 s reference R; Seed-VC\'s 25 s cut does not act.',
         'FreeVC was trained on VCTK ("VCTK-seen"); its VCTK-2k rows carry that flag.',
         'Every number comes from H200 MIG 2g.35gb slices with Mimi in float16.']


def x1_done(x1_dir, m_done):
    """x1.done of job X1, written under the selection.env of every part of job M."""
    path = os.path.join(x1_dir, 'x1.done')
    if not os.path.exists(path):
        raise SystemExit(f'ERROR: {path} missing: job X1 has not finished')
    line = open(path).read().strip()
    m = re.search(r'selection sha256 ([0-9a-f]{64})', line)
    shas = {re.search(r'selection sha256 ([0-9a-f]{64})', v).group(1) for v in m_done.values()}
    if not m or {m.group(1)} != shas:
        raise SystemExit(f'ERROR: {path} ({line!r}) was not written under the selection.env of job M ({sorted(shas)})')
    return line


def load(m_dir, x1_dir, name, pairs):
    """{3: {system: {pair_id: row}}} from job M (t3, t4) and job X1; a system in both is an error."""
    R, inputs = at.load_set(m_dir, name, [('t3', BUDGET), ('t4', BUDGET)], pairs)
    X, x_inputs = at.load_set(x1_dir, name, [('m', BUDGET)], pairs)
    for system, rows in X.get(BUDGET, {}).items():
        if system in R.get(BUDGET, {}):
            raise SystemExit(f'ERROR: {name}: {system} at {BUDGET} s comes from both job M and job X1')
        R.setdefault(BUDGET, {})[system] = rows
    return R, inputs + x_inputs


def required(sets, crt_al, baselines):
    """CRT-AL and every baseline: target rows one per pair, and @src rows, at 3 s; returns the problems."""
    out = []
    for s, ts in sets.items():
        for n in (crt_al,) + tuple(baselines):
            k = len(ts.rows(n, BUDGET))
            if k != len(ts.pairs):
                out.append(f'{s} {n}: {k} rows for {len(ts.pairs)} pairs')
            if not ts.has(n + at.SRC, BUDGET):
                out.append(f'{s} {n}{at.SRC}: no rows')
    return out


def comparison(ts, hid, q, a, b, wer='wer_capped'):
    """analyze_tests.hypothesis at the 3 s budget."""
    metric = wer if q == 'wer' else at.Q[q][0]
    res = stats.compare(ts.quantity_rows(a, q, BUDGET), ts.quantity_rows(b, q, BUDGET), ts.subset(at.Q[q][1]),
                        ts.spk, metric, mode=ts.mode, worst=at.Q[q][3], n_ref=ts.n_ref)
    label = at.LABEL[{'wer_capped': 'wer', 'wer': 'wer_u'}.get(metric, q)]
    return {'id': hid, 'set': ts.name, 'quantity': q, 'label': label, 'sys_a': a, 'sys_b': b, **res}


def x1_family(sets, crt_al, baselines, wer='wer_capped'):
    hyps = []
    for bl in baselines:
        for i, (s, q) in enumerate(at.S_ENDPOINTS, 1):
            h = comparison(sets[s], f'X1-{bl}-{i}', q, crt_al, bl, wer)
            if bl == 'freevc' and s == 'vctk2k':
                h['flag'] = 'VCTK-seen'
            hyps.append(h)
    flags, _ = at.apply_holm(hyps)
    at.set_flags(hyps, flags)
    return {'family': 'X1', 'a': crt_al, 'b': f'baselines at {BUDGET} s', 'hypotheses': hyps,
            'holm': f'over {len(hyps)} (guide only, exploratory)'}


def analyse(sets, systems, baselines):
    a = systems['crt_al']
    X = x1_family(sets, a, baselines)
    U = x1_family(sets, a, baselines, wer='wer')
    return {'x1': X, 'sensitivity': {'wer_definition': {'changed': at.changed(X, U), 'family': U}},
            'levels': {s: at.levels(sets[s], systems, budget=BUDGET,
                                    quantities=('wer', 'wer_u', 'cer', 'sim', 'utmos', 'sim_src', 'm9'))
                       for s in at.SETS},
            'eer': {s: at.eer_table(sets[s], budget=BUDGET) for s in at.SETS}}


def render(res):
    sy = res['systems']
    L = [f'# Exploratory addition X1: CRT-AL vs the baselines at {BUDGET} s (DEVIATIONS_v1.md item 5)', '',
         f'Written {res["written"]} by `benchmark/analyze_exploratory.py` (sha256 {res["script_sha256"]}; '
         f'analyze_tests.py {res["analyze_tests_sha256"]}; stats.py {res["stats_sha256"]}). CRT-AL = {sy["choice"]} '
         f'(`{sy["crt_al"]}`); baselines {", ".join(res["baselines"])}. Crossed bootstrap B = {stats.B}, seed '
         f'{stats.SEED}: VCTK-2k independent, LTC-2k linked.', ''] + [f'- {x}' for x in NOTES] + ['']
    X = res['x1']
    L += [f'## CRT-AL − each baseline at {BUDGET} s (exploratory)', '', f'Holm {X["holm"]}.', '', HEAD]
    L += [at.fmt_h(h) for h in X['hypotheses']]
    ch = res['sensitivity']['wer_definition']['changed']
    L += ['', f'Uncapped WER instead of capped: decisions that change: {at.fmt_changed(ch)}.', '']
    for s in at.SETS:
        L += [f'### Levels, {s}, {BUDGET} s (descriptive)', '',
              '| System | rows (failed) | capped WER % | WER % | capped CER % | SIM | UTMOS | SIM_src | M9 D | '
              'share D > 0 | EER |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for n, row in res['levels'][s].items():
            flag = f' [{row["flag"]}]' if row.get('flag') else ''
            L.append(f'| {n}{flag} | {row["n_rows"]} ({row["fail"]}) | '
                     + ' | '.join(at.ci(row.get(q), at.digits(q)) for q in ('wer', 'wer_u', 'cer', 'sim', 'utmos',
                                                                           'sim_src', 'm9'))
                     + f' | {at.f((row.get("share_d_pos") or {}).get("share"))} | {at.f(res["eer"][s].get(n))} |')
        L.append('')
    L += ['## Inputs', '', f'- job X1: `{res["x1_done"]}`'] + \
         [f'- job M part {k}: `{v}`' for k, v in res['m_done'].items()] + \
         [f'- `{x["path"]}` sha256 {x["sha256"]}' + (f' ({x["rows"]} rows)' if 'rows' in x else '')
          for x in res['inputs']]
    return '\n'.join(L) + '\n'


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    v1 = os.path.join(here, '..', 'paper', 'journal_extension', 'results', 'v1')
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--m_dir', default=os.path.join(v1, 'm'))
    ap.add_argument('--x1_dir', default=os.path.join(v1, 'x1'))
    ap.add_argument('--vctk_pairs', default=os.path.join(v1, 'pairs', 'vctk2k.jsonl'))
    ap.add_argument('--ltc_pairs', default=os.path.join(v1, 'pairs', 'ltc2k.jsonl'))
    ap.add_argument('--source_e', default=os.path.join(v1, 'pairs', 'vctk2k_source_E.json'))
    ap.add_argument('--record', default=os.path.join(here, 'DEV_SELECTION_RECORD_v1.json'))
    ap.add_argument('--out', default=os.path.join(here, 'EXPLORATORY_X1_v1.md'))
    a = ap.parse_args(argv)
    systems = at.systems_from_record(json.load(open(a.record)))
    missing = [s for s in ('knnvc',) + X1_SYSTEMS if s not in systems['baselines']]
    if missing:
        raise SystemExit(f'ERROR: the record does not list {missing}')
    baselines = ['knnvc', *X1_SYSTEMS]
    vp, lp = at.read_pairs(a.vctk_pairs), at.read_pairs(a.ltc_pairs)
    src_e = {k for k, v in json.load(open(a.source_e)).items() if (v or {}).get('paths')}
    inputs = [{'path': os.path.abspath(p), 'sha256': at.sha256(p)}
              for p in (a.record, a.vctk_pairs, a.ltc_pairs, a.source_e)]
    m_done = at.check_done(a.m_dir, ['t1', 't2', 't3', 't4'])
    done = x1_done(a.x1_dir, m_done)
    sets = {}
    for s, pairs in (('vctk2k', vp), ('ltc2k', lp)):
        R, ins = load(a.m_dir, a.x1_dir, s, pairs)
        inputs += ins
        sets[s] = at.TestSet(s, pairs, R, src_e if s == 'vctk2k' else None)
    problems = required(sets, systems['crt_al'], baselines)
    if problems:
        raise SystemExit('ERROR: ' + '; '.join(problems))
    res = analyse(sets, systems, baselines)
    res.update(written=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
               script_sha256=at.sha256(os.path.abspath(__file__)), analyze_tests_sha256=at.sha256(at.__file__),
               stats_sha256=at.sha256(stats.__file__), systems=systems, baselines=baselines, inputs=inputs,
               m_done=m_done, x1_done=done)
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
