#!/usr/bin/env python3
"""Dev quick-look for Mimi-VC: capped WER, CER, SIM, SIM_src, UTMOS of a system on the converted pairs, beside the
published comparison systems (CRT-AL = cand_pm18_rq, FAISS = anc_faiss on Dev) restricted to exactly the same pairs.

capped WER = sum_p min(S+D+I, N) / sum_p N over wer_primary pairs (stats.py's 'wer_capped'); SIM / SIM_src = mean
WavLM-SV cosine to the target's / the source's enrolment centroid; UTMOS mean. Point estimates only: the bootstrap and
Holm decisions stay with stats.py for the final runs.

    python summarize.py --metrics m/dev_30.jsonl --src m/dev_30_src.jsonl \
        --compare cand_pm18_rq=$J/v1/d2/lab_metrics.jsonl:$J/v1/d2/lab_metrics_src.jsonl \
        --compare anc_faiss=$J/v1/d1/metrics.jsonl:$J/v1/d1/metrics_src.jsonl --out summary.json
"""
import argparse
import json
import sys


def load(path, system=None):
    """Last row per (pair_id, system); optionally one system."""
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if system is None or r.get('system', '').split('@')[0] == system:
                out[(r['pair_id'], r['system'])] = r
    return out


def by_pair(rows, src=False):
    return {pid: r for (pid, s), r in rows.items() if s.endswith('@src') == src}


STRATA = {'F2F': 'same', 'M2M': 'same', 'F2M': 'cross', 'M2F': 'cross'}


def strata(tgt, src, pids, wer_key='wer_large_v3'):
    """score() per gender combination (genders field) and pooled same / cross gender."""
    out = {}
    for name, keep in [(k, {k}) for k in STRATA] + [('same', {'F2F', 'M2M'}), ('cross', {'F2M', 'M2F'})]:
        sub = [p for p in pids if p in tgt and tgt[p].get('genders') in keep]
        if sub:
            out[name] = score(tgt, src, sub, wer_key, with_strata=False)
    return out


def score(tgt, src, pids, wer_key='wer_large_v3', with_strata=True):
    E = N = cE = cN = 0
    sims, utm, ssrc, n_wer = [], [], [], 0
    for p in pids:
        r = tgt.get(p)
        if r is None or 'error' in r:
            continue
        w = r.get(wer_key)
        if r.get('wer_primary') and w:
            e = w['S'] + w['D'] + w['I']
            E += min(e, w['N'])
            N += w['N']
            ce = w['cS'] + w['cD'] + w['cI']
            cE += min(ce, w['cN'])
            cN += w['cN']
            n_wer += 1
        if r.get('sim_wavlm') is not None:
            sims.append(r['sim_wavlm'])
        if r.get('utmos') is not None:
            utm.append(r['utmos'])
        s = src.get(p)
        if s is not None and s.get('sim_wavlm') is not None:
            ssrc.append(s['sim_wavlm'])
    mean = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
    res = dict(n=len([p for p in pids if p in tgt]), n_wer=n_wer, wer_capped=round(100 * E / N, 2) if N else None,
               cer_capped=round(100 * cE / cN, 2) if cN else None, sim=mean(sims), sim_src=mean(ssrc), utmos=mean(utm),
               margin=round(mean(sims) - mean(ssrc), 4) if sims and ssrc else None)
    if with_strata:
        res['strata'] = strata(tgt, src, pids, wer_key)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--compare', action='append', default=[], help='name=target_metrics.jsonl:src_metrics.jsonl')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    tgt, src = by_pair(load(a.metrics)), by_pair(load(a.src), src=True)
    pids = sorted(p for p, r in tgt.items() if 'error' not in r)
    res = {'_system': score(tgt, src, pids)}
    for spec in a.compare:
        name, files = spec.split('=', 1)
        tf, sf = files.split(':', 1)
        res[name] = score(by_pair(load(tf, name)), by_pair(load(sf, name), src=True), pids)
    json.dump(res, open(a.out, 'w'), indent=1)
    for k, v in res.items():
        print(f"{k:14s} n={v['n']:4d} WERc={v['wer_capped']} CERc={v['cer_capped']} SIM={v['sim']} SIM_src={v['sim_src']} "
              f"UTMOS={v['utmos']} | same SIM_src {v['strata'].get('same', {}).get('sim_src')} "
              f"cross SIM_src {v['strata'].get('cross', {}).get('sim_src')}", flush=True)
    tc = res.get('target_copy', {}).get('strata', {})
    if tc:                                   # leakage against the natural impostor level, per stratum (PLAN 4 P6)
        for k in ('same', 'cross'):
            if k in tc and k in res['_system']['strata'] and tc[k]['sim_src'] is not None:
                d = res['_system']['strata'][k]['sim_src'] - tc[k]['sim_src']
                res['_system']['strata'][k]['sim_src_minus_target_copy'] = round(d, 4)
                print(f"leakage {k}: SIM_src - SIM_src(target_copy) = {d:+.4f}", flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
