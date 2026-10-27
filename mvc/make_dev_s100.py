#!/usr/bin/env python3
"""Dev-S100 (PLAN.md §5.2): the fixed 100-pair subset of the VCTK Dev pair list used for iteration.

Rule, seed 20260928: 25 pairs per gender combination (F2F, F2M, M2F, M2M, the `genders` field), at least 50
wer_primary pairs, at least 50 pairs with a parallel recording P, every Dev target speaker used at least 3 times, and
at most one pair per source utterance (Dev converts each source utterance to 5 targets; distinct sources keep the
capped WER, computed on the wer_primary pairs only, from resting on a few source texts). Pairs that are both
wer_primary and P count for both and are taken first. The remaining wer_primary and P quotas are then placed one pair
at a time, each in the gender combination with the most free slots and the fewest pairs of that kind so far, under a
max-flow check (on the distinct source utterances of each cell) that keeps the rest satisfiable. Free slots left over
are filled to balance wer_primary against P. Each (gender combination, kind) cell is filled from its pair_ids, sorted
and then shuffled with random.Random('<seed>:dev_s100:<genders>:<kind>') (the make_pairs.rng convention), skipping
pairs whose source utterance is already taken; cells are filled in the order F2F, F2M, M2F, M2M. A deterministic
repair then swaps pairs until every target reaches its minimum: inside the cell where possible, otherwise inside the
gender combination when the totals stay at or above their minimums, otherwise as two swaps, the second (in any gender
combination) mending the total, the target or the source utterance that the first alone would break. Every order used
is a total order on pair_id, so the subset depends only on the set of pairs, not on their order in the file. In Dev
the wer_primary pairs and the P pairs are disjoint, so the subset holds exactly 50 of each. Errors that say
'infeasible' are proofs that no subset meets the rule; a failed fill or repair says so and proves nothing.

The selection reads only pair_id, genders, wer_primary, P, target_speaker and source_utt. The Mac copy and the ceai
copy of dev.jsonl differ only in their paths, so both give the same subset; fields_sha256 in the summary checks this.

Outputs: <out> (one pair_id per line, sorted) and the .json of the same stem beside it (rule, input hashes, realised
counts, quotas, repair swaps, ids). An existing identical subset whose .json records the same ids and seed is left
untouched; a different subset is an error unless --force.

    python make_dev_s100.py --pairs $J/pairs/dev.jsonl
    python run_mvc.py --journal $J --pairs $J/pairs/dev.jsonl --set dev ... \
        --pair_ids "$(python make_dev_s100.py --print_ids)"
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from itertools import product
from pathlib import Path

SEED = 20260928
GENDERS = ('F2F', 'F2M', 'M2F', 'M2M')
KINDS = ('wp', 'w', 'p', 'n')
LABEL = {'wp': 'wer_primary+P', 'w': 'wer_primary', 'p': 'P', 'n': 'neither'}
OUT = Path(__file__).resolve().parent / 'pairs' / 'dev_s100.txt'
PAIRS = '/datastore/ceai/crt/mimi_vc/benchmark/journal/pairs/dev.jsonl'


def load_ids(path=OUT):
    return [l.strip() for l in open(path) if l.strip()]


def ids_arg(path=OUT):
    """The comma-joined pair_ids of a subset file, the format of run_mvc.py --pair_ids."""
    return ','.join(load_ids(path))


def kind(r):
    w, p = bool(r['wer_primary']), bool(r.get('P'))
    return 'wp' if w and p else 'w' if w else 'p' if p else 'n'


def shuffled(ids, seed, *tags):
    ids = sorted(ids)
    random.Random(':'.join(str(t) for t in (seed, 'dev_s100') + tags)).shuffle(ids)
    return ids


def shortfall(need, avail, free):
    """None if the remaining wer_primary / P needs fit the free slots (min cut over the two kinds), else a message."""
    for ks in (('w',), ('p',), ('w', 'p')):
        want = sum(need[k] for k in ks)
        cap = sum(min(sum(avail[g][k] for k in ks), free[g]) for g in free)
        if want > cap:
            return f'{" and ".join(LABEL[k] for k in ks)}: {want} more needed, at most {cap} can be chosen'
    return None


def quotas(cells, per_gender, min_wer, min_p, seed):
    """cells {(g, kind): [pair_id]} -> {g: {kind: count}} with per_gender pairs per g and the two minimums met."""
    n = {g: {k: len(cells.get((g, k), ())) for k in KINDS} for g in GENDERS}
    c = {g: {k: 0 for k in KINDS} for g in GENDERS}
    for g in GENDERS:
        c[g]['wp'] = min(n[g]['wp'], per_gender)
    free = {g: per_gender - c[g]['wp'] for g in GENDERS}
    avail = {g: {k: n[g][k] - c[g][k] for k in KINDS} for g in GENDERS}
    both = sum(c[g]['wp'] for g in GENDERS)
    need = {'w': max(0, min_wer - both), 'p': max(0, min_p - both)}
    err = shortfall(need, avail, free)
    if err:
        raise ValueError(f'infeasible with {per_gender} pairs per gender combination ({both} pairs are both): {err}')
    order = shuffled(GENDERS, seed, 'quota')
    while need['w'] or need['p']:
        k = 'w' if need['w'] >= need['p'] else 'p'
        for g in sorted(order, key=lambda g: (-free[g], c[g][k])):
            if not (free[g] and avail[g][k]):
                continue
            free[g], avail[g][k], need[k] = free[g] - 1, avail[g][k] - 1, need[k] - 1
            if shortfall(need, avail, free) is None:
                c[g][k] += 1
                break
            free[g], avail[g][k], need[k] = free[g] + 1, avail[g][k] + 1, need[k] + 1
    for g in order:
        while free[g]:
            ks = [k for k in ('w', 'p') if avail[g][k]] or ['n']
            k = min(ks, key=lambda k: (c[g][k], sum(c[h][k] for h in GENDERS)))
            free[g], avail[g][k], c[g][k] = free[g] - 1, avail[g][k] - 1, c[g][k] + 1
    return c


def repair(chosen, by, order, targets, min_target, min_wer, min_p):
    """Swap pairs into `chosen` (a set, changed in place) until every target has min_target uses, keeping the gender
    quotas, the two minimums and one pair per source utterance. Returns the swaps [out, in]; the two swaps of a
    compensated move are listed one after the other."""
    rank = {pid: i for ids in order.values() for i, pid in enumerate(ids)}
    use = Counter(by[q]['target_speaker'] for q in chosen)
    srcs = {by[q]['source_utt'] for q in chosen}
    tot = {k: sum(kind(by[q]) in ('wp', k) for q in chosen) for k in ('w', 'p')}
    mins = (('w', min_wer), ('p', min_p))
    swaps = []

    def totals_ok(kins, kouts):
        return all(tot[k] + sum(x in ('wp', k) for x in kins) - sum(x in ('wp', k) for x in kouts) >= m
                   for k, m in mins)

    def ok(t, ins, outs):
        """t gains a use, no other target falls below min(its uses, min_target), sources stay distinct, totals hold."""
        d = Counter(by[x]['target_speaker'] for x in ins)
        d.subtract(by[x]['target_speaker'] for x in outs)
        if d[t] < 1 or any(use[u] + v < min(use[u], min_target) for u, v in d.items()):
            return False
        new = [by[x]['source_utt'] for x in ins]
        if len(set(new)) < len(new) or set(new) & (srcs - {by[x]['source_utt'] for x in outs}):
            return False
        return totals_ok([kind(by[x]) for x in ins], [kind(by[x]) for x in outs])

    def prefer(q):
        return -use[by[q]['target_speaker']], -rank[q], q

    def single(t, cands):
        for same_kind in (True, False):
            for p in cands:
                outs = [q for q in chosen if by[q]['genders'] == by[p]['genders']
                        and (kind(by[q]) == kind(by[p])) == same_kind and ok(t, [p], [q])]
                if outs:
                    return [(min(outs, key=prefer), p)]
        return None

    def compensated(t, cands):
        """(q out, p in) in p's gender combination plus (y out, x in) in any, when the first swap alone breaks a total,
        takes a target below its minimum or reuses the source utterance of a chosen pair, and the second mends it."""
        ranked = sorted(chosen, key=prefer)
        holder = {by[y]['source_utt']: y for y in chosen}
        pool = {key: [x for x in ids if x not in chosen] for key, ids in order.items()}
        for p in cands:
            z = holder.get(by[p]['source_utt'])
            for q in (q for q in ranked if by[q]['genders'] == by[p]['genders']):
                for y in ([z] if z not in (None, q) else [y for y in ranked if y != q]):
                    d = Counter([t])
                    d.subtract([by[q]['target_speaker'], by[y]['target_speaker']])
                    lack = {u for u, v in d.items() if use[u] + v < min(use[u], min_target) or (u == t and v < 1)}
                    if len(lack) > 1:
                        continue
                    for k in KINDS:
                        if not totals_ok([kind(by[p]), k], [kind(by[q]), kind(by[y])]):
                            continue
                        for x in pool.get((by[y]['genders'], k), ()):
                            if x != p and (not lack or by[x]['target_speaker'] in lack) and ok(t, [p, x], [q, y]):
                                return [(q, p), (y, x)]
        return None

    while True:
        short = sorted((use[t], t) for t in targets if use[t] < min_target)
        if not short:
            return swaps
        t = short[0][1]
        cands = sorted((q for q in by if q not in chosen and by[q]['target_speaker'] == t), key=lambda q: (rank[q], q))
        move = single(t, cands) or compensated(t, cands)
        if not move:
            raise ValueError(f'repair failed: target {t} reaches only {use[t]} of {min_target} uses and no single or '
                             f'compensated swap keeps the gender, wer_primary and P quotas, one pair per source '
                             f'utterance and the other targets at {min_target}; another subset may still meet the rule')
        for q, p in move:
            chosen.discard(q)
            chosen.add(p)
            use[by[q]['target_speaker']] -= 1
            use[by[p]['target_speaker']] += 1
            for k in ('w', 'p'):
                tot[k] += (kind(by[p]) in ('wp', k)) - (kind(by[q]) in ('wp', k))
            swaps.append([q, p])
        srcs.clear()
        srcs.update(by[q]['source_utt'] for q in chosen)


def select(rows, seed=SEED, per_gender=25, min_wer=50, min_p=50, min_target=3):
    """rows: the Dev pair list -> (sorted pair_ids, {'quotas', 'swaps'}); ValueError when the rule cannot be met."""
    by = {r['pair_id']: r for r in rows}
    if len(by) != len(rows):
        raise ValueError('duplicate pair_id in the pair list')
    bad = sorted({r['genders'] for r in rows} - set(GENDERS))
    if bad:
        raise ValueError(f'unknown genders {bad}; expected {list(GENDERS)}')
    cells = {}
    for r in rows:
        cells.setdefault((r['genders'], kind(r)), []).append(r['pair_id'])
    have = Counter(r['genders'] for r in rows)
    for g in GENDERS:
        if have[g] < per_gender:
            raise ValueError(f'infeasible: {g} has {have[g]} pairs, {per_gender} needed')
    src_t = {}
    for r in rows:
        src_t.setdefault(r['target_speaker'], set()).add(r['source_utt'])
    few = sorted(t for t in src_t if len(src_t[t]) < min_target)
    if few:
        raise ValueError(f'infeasible: targets {few} have fewer than {min_target} pairs with distinct source '
                         'utterances in the list')
    for x in 'FM':
        tx = {r['target_speaker'] for r in rows if r['genders'][-1] == x}
        slots = per_gender * sum(g[-1] == x for g in GENDERS)
        if len(tx) * min_target > slots:
            raise ValueError(f'infeasible: {len(tx)} {x} targets x {min_target} uses > {slots} slots with {x} targets')
    c = quotas({key: {by[q]['source_utt'] for q in ids} for key, ids in cells.items()},
               per_gender, min_wer, min_p, seed)
    order = {key: shuffled(ids, seed, *key) for key, ids in cells.items()}
    chosen, srcs = set(), set()
    for g, k in product(GENDERS, KINDS):
        take = []
        for q in order.get((g, k), ()):
            if len(take) < c[g][k] and by[q]['source_utt'] not in srcs:
                take.append(q)
                srcs.add(by[q]['source_utt'])
        if len(take) < c[g][k]:
            raise ValueError(f'{g} {LABEL[k]}: only {len(take)} of {c[g][k]} pairs have a source utterance not taken '
                             'by an earlier cell; another subset may still meet the rule')
        chosen.update(take)
    swaps = repair(chosen, by, order, sorted(src_t), min_target, min_wer, min_p)
    return sorted(chosen), {'quotas': c, 'swaps': swaps}


def counts(rows, ids):
    by = {r['pair_id']: r for r in rows}
    sel = [by[i] for i in ids]
    tg = Counter(r['target_speaker'] for r in sel)
    cells = Counter((r['genders'], kind(r)) for r in sel)
    return {'n': len(sel), 'genders': {g: sum(r['genders'] == g for r in sel) for g in GENDERS},
            'wer_primary': sum(bool(r['wer_primary']) for r in sel), 'P': sum(bool(r.get('P')) for r in sel),
            'cells': {f'{g} {LABEL[k]}': cells[(g, k)] for g in GENDERS for k in KINDS if cells[(g, k)]},
            'n_targets': len(tg), 'n_targets_in_list': len({r['target_speaker'] for r in rows}),
            'target_uses_min': min(tg.values()), 'target_uses_max': max(tg.values()),
            'targets': dict(sorted(tg.items())),
            'n_sources': len({r.get('source_speaker') for r in sel}),
            'n_distinct_source_utts': len({r['source_utt'] for r in sel}),
            'n_distinct_wer_primary_texts': len({r['source_text'] for r in sel if r['wer_primary']})}


def fields_sha256(rows):
    keep = sorted([r['pair_id'], r['genders'], bool(r['wer_primary']), bool(r.get('P')), r['target_speaker'],
                   r['source_utt']] for r in rows)
    return hashlib.sha256(json.dumps(keep).encode()).hexdigest()


def write_atomic(path, text):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', default=PAIRS, help='the Dev pair list ($J/pairs/dev.jsonl)')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--force', action='store_true', help='replace an existing, different subset')
    ap.add_argument('--print_ids', action='store_true', help='print the ids of --out comma-joined (for --pair_ids)')
    a = ap.parse_args(argv)
    if a.print_ids:
        print(ids_arg(a.out))
        return 0
    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    try:
        ids, info = select(rows, a.seed)
    except ValueError as e:
        raise SystemExit(f'ERROR: {e}')
    text = ''.join(i + '\n' for i in ids)
    out, js = Path(a.out), Path(a.out).with_suffix('.json')
    old = out.read_text() if out.exists() else None
    if old not in (None, text) and not a.force:
        raise SystemExit(f'ERROR: {out} holds a different subset; pass --force to replace it')
    cnt = counts(rows, ids)
    print(f'dev_s100: {cnt["n"]} of {len(rows)} pairs, seed {a.seed}; '
          + ', '.join(f'{g} {cnt["genders"][g]}' for g in GENDERS)
          + f'; wer_primary {cnt["wer_primary"]}, P {cnt["P"]}; targets {cnt["n_targets"]}/{cnt["n_targets_in_list"]} '
          f'used {cnt["target_uses_min"]}-{cnt["target_uses_max"]} times; {cnt["n_sources"]} source speakers, '
          f'{cnt["n_distinct_source_utts"]} source utterances, {cnt["n_distinct_wer_primary_texts"]} wer_primary '
          f'texts; {len(info["swaps"])} repair swaps', flush=True)
    print('  ' + ', '.join(f'{k} {v}' for k, v in cnt['cells'].items()), flush=True)
    prev = json.loads(js.read_text()) if js.exists() else {}
    if old == text and (prev.get('ids'), prev.get('seed')) == (ids, a.seed):
        print(f'unchanged: {out}', flush=True)
        return 0
    doc = {'rule': 'PLAN.md §5.2: 25 pairs per gender combination, >= 50 wer_primary, >= 50 with P, every Dev target '
                   '>= 3 uses, one pair per source utterance; quotas, seeded cell shuffles and repair as in '
                   'make_dev_s100.py',
           'seed': a.seed, 'pairs': os.path.abspath(a.pairs),
           'pairs_sha256': hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest(), 'fields_sha256': fields_sha256(rows),
           'n_pairs_in_list': len(rows), 'ids_file': str(out.resolve()),
           'ids_sha256': hashlib.sha256(text.encode()).hexdigest(), 'counts': cnt,
           'quotas': {g: {LABEL[k]: v for k, v in q.items() if v} for g, q in info['quotas'].items()},
           'repair_swaps': [{'out': q, 'in': p} for q, p in info['swaps']], 'ids': ids}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(js, json.dumps(doc, indent=1) + '\n')
    write_atomic(out, text)
    print(f'-> {out}, {js}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
