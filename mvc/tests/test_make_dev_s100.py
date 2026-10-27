"""Tests of mvc/make_dev_s100.py on synthetic Dev-like pair lists: every constraint of PLAN.md §5.2 holds, each source
utterance is used at most once, the subset is deterministic and independent of row order, pairs that are both
wer_primary and P are used, the single and compensated target repairs work, infeasible lists fail with a clear
message, a failed repair never claims infeasibility for a feasible list, and the CLI writes, keeps and guards its
outputs. The committed mvc/pairs/dev_s100.txt is checked against the Mac copy of the Dev list when that copy is
present."""
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_dev_s100 as ds  # noqa: E402

DEV = Path('/Users/harry/duplex/paper/journal_extension/results/v1/pairs/dev.jsonl')
F = [f'f{i:02d}' for i in range(10)]
M = [f'm{i:02d}' for i in range(10)]


def row(i, src, tgt, w, p, utt=None):
    gs, gt = src[0].upper(), tgt[0].upper()
    utt = utt or f'{src}_{i:04d}'
    return {'pair_id': f'dev__{utt}__{tgt}', 'set': 'dev', 'source_speaker': src, 'source_gender': gs,
            'source_utt': utt, 'source_text': f'text of {utt}', 'target_speaker': tgt, 'target_gender': gt,
            'genders': f'{gs}2{gt}', 'wer_primary': w, 'P': f'/x/{tgt}_{i}.flac' if p else None}


def dev_like(n=1000, seed=0, both=0.0, per_utt=5):
    """Like VCTK Dev: 10 F and 10 M speakers, each source utterance converted to per_utt distinct targets, wer_primary
    and P disjoint and fixed per source utterance (a fraction `both` of the wer_primary pairs also has P)."""
    rng = random.Random(seed)
    spk = F + M
    rows = []
    for u in range(n // per_utt):
        src = spk[u % 20]
        w = u // 20 % 2 == 0
        for tgt in rng.sample([s for s in spk if s != src], per_utt):
            rows.append(row(len(rows), src, tgt, w, (not w) or rng.random() < both, utt=f'{src}_{u:03d}'))
    return rows


def check(rows, ids, per=25, min_wer=50, min_p=50, min_target=3):
    c = ds.counts(rows, ids)
    assert len(ids) == len(set(ids)) == 4 * per and ids == sorted(ids)
    assert c['genders'] == {g: per for g in ds.GENDERS}
    assert c['wer_primary'] >= min_wer and c['P'] >= min_p
    assert c['n_targets'] == c['n_targets_in_list'] and c['target_uses_min'] >= min_target
    assert c['n_distinct_source_utts'] == 4 * per
    return c


def tiny(spec, utts=None):
    """spec [(genders, target, 'w' | 'p')] -> rows p000, p001, ... with their own source utterances unless given."""
    return [{'pair_id': f'p{i:03d}', 'genders': g, 'target_speaker': t, 'wer_primary': k == 'w',
             'P': 'x' if k == 'p' else None, 'source_utt': (utts or {}).get(i, f'u{i}'), 'source_text': f's{i}'}
            for i, (g, t, k) in enumerate(spec)]


def run_repair(rows, chosen, min_target, min_wer, min_p):
    by = {r['pair_id']: r for r in rows}
    order = {}
    for r in rows:
        order.setdefault((r['genders'], ds.kind(r)), []).append(r['pair_id'])
    chosen = set(chosen)
    swaps = ds.repair(chosen, by, order, sorted({r['target_speaker'] for r in rows}), min_target, min_wer, min_p)
    return sorted(chosen), swaps


def test_dev_like_list_meets_every_constraint():
    rows = dev_like()
    assert set(Counter(r['source_utt'] for r in rows).values()) == {5}
    ids, info = ds.select(rows)
    c = check(rows, ids)
    assert c['wer_primary'] == c['P'] == 50                         # disjoint kinds: exactly half each
    assert c['n_distinct_wer_primary_texts'] == 50
    for g in ds.GENDERS:                                            # stratified: 12 or 13 of each kind per combination
        assert {c['cells'][f'{g} wer_primary'], c['cells'][f'{g} P']} == {12, 13}
    assert all(q['wp'] == q['n'] == 0 for q in info['quotas'].values())


@pytest.mark.parametrize('seed', range(10))
def test_deterministic_and_independent_of_row_order(seed):
    rows = dev_like(seed=seed)
    a, _ = ds.select(rows)
    assert ds.select(list(reversed(rows)))[0] == a
    for k in range(5):
        shuf = rows[:]
        random.Random(100 * seed + k).shuffle(shuf)
        assert ds.select(shuf)[0] == a
    d, _ = ds.select(rows, seed=ds.SEED + 1)
    check(rows, d)
    assert d != a


def test_pairs_with_both_wer_primary_and_p_are_taken_first():
    rows = dev_like(seed=1, both=0.1, per_utt=1)
    nb = Counter(r['genders'] for r in rows if r['wer_primary'] and r['P'])
    ids, info = ds.select(rows)
    c = check(rows, ids)
    assert all(info['quotas'][g]['wp'] == min(nb[g], 25) for g in ds.GENDERS)
    sel = {r['pair_id']: r for r in rows if r['pair_id'] in set(ids)}
    assert sum(bool(r['wer_primary'] and r['P']) for r in sel.values()) == sum(min(nb[g], 25) for g in ds.GENDERS)
    assert c['wer_primary'] + c['P'] - 100 == sum(min(nb[g], 25) for g in ds.GENDERS)


def test_repair_brings_a_rare_target_to_its_minimum():
    rows = [r for r in dev_like(seed=2) if r['target_speaker'] != 'f09']
    rows += [row(5000 + i, s, 'f09', i % 2 == 0, i % 2 == 1) for i, s in enumerate(['f00', 'm01', 'f02'])]
    ids, info = ds.select(rows)
    check(rows, ids)
    assert {r['pair_id'] for r in rows if r['target_speaker'] == 'f09'} <= set(ids)
    assert info['swaps']


def test_compensated_swap_mends_a_total():
    """The reviewer's case: mb's only pair p002 (F2M, wer_primary) can enter only for a P pair, which leaves P one short
    of its minimum; the M2M swap p007 (wer_primary) out, p008 (P) in restores it."""
    rows = tiny([('F2F', 'fc', 'w'), ('F2F', 'fc', 'w'), ('F2M', 'mb', 'w'), ('F2M', 'ma', 'p'), ('F2M', 'ma', 'p'),
                 ('M2F', 'fc', 'p'), ('M2F', 'fc', 'w'), ('M2M', 'ma', 'w'), ('M2M', 'ma', 'p'), ('M2M', 'mc', 'p')])
    ids, swaps = run_repair(rows, ['p000', 'p001', 'p003', 'p004', 'p005', 'p006', 'p007', 'p009'], 1, 4, 4)
    assert swaps == [['p004', 'p002'], ['p007', 'p008']]
    assert ids == ['p000', 'p001', 'p002', 'p003', 'p005', 'p006', 'p008', 'p009']
    for seed in range(10):                                          # the single-swap repair failed for 6 of these
        ids, _ = ds.select(rows, seed=seed, per_gender=2, min_wer=4, min_p=4, min_target=1)
        c = ds.counts(rows, ids)
        assert c['genders'] == {g: 2 for g in ds.GENDERS} and c['wer_primary'] >= 4 and c['P'] >= 4
        assert c['target_uses_min'] >= 1 and c['n_targets'] == 4


def test_compensated_swap_restores_a_target():
    """fc's only pair p000 (F2F) can enter only for p002, fb's only F2F pair; the M2F swap p007 (fa) out, p006 (fb) in
    keeps fb."""
    rows = tiny([('F2F', 'fc', 'w'), ('F2F', 'fa', 'p'), ('F2F', 'fb', 'w'), ('F2M', 'ma', 'p'), ('F2M', 'ma', 'w'),
                 ('M2F', 'fa', 'p'), ('M2F', 'fb', 'w'), ('M2F', 'fa', 'w'), ('M2M', 'ma', 'p'), ('M2M', 'mb', 'p'),
                 ('M2M', 'mc', 'w'), ('M2M', 'mb', 'p')])
    ids, swaps = run_repair(rows, ['p001', 'p002', 'p003', 'p004', 'p005', 'p007', 'p008', 'p010'], 1, 4, 4)
    assert swaps[:2] == [['p002', 'p000'], ['p007', 'p006']]
    c = ds.counts(rows, ids)
    assert c['wer_primary'] >= 4 and c['P'] >= 4 and c['n_targets'] == 6


def test_compensated_swap_frees_a_source_utterance():
    """ma's only pair p002 (F2M) shares its source utterance with the chosen F2F pair p000; swapping p000 for p001
    frees it. p002 then holds that utterance, so mc must come from p007, not from p006 with the same utterance."""
    rows = tiny([('F2F', 'fa', 'w'), ('F2F', 'fa', 'w'), ('F2M', 'ma', 'w'), ('F2M', 'mb', 'w'), ('M2F', 'fa', 'w'),
                 ('M2M', 'mb', 'w'), ('M2M', 'mc', 'w'), ('M2M', 'mc', 'w'), ('M2M', 'mb', 'w')],
                utts={0: 'shared', 2: 'shared', 6: 'shared'})
    ids, swaps = run_repair(rows, ['p000', 'p003', 'p004', 'p005', 'p008'], 1, 0, 0)
    assert swaps == [['p003', 'p002'], ['p000', 'p001'], ['p008', 'p007']]
    assert ids == ['p001', 'p002', 'p004', 'p005', 'p007']


def test_source_utterances_are_not_repeated():
    rows = dev_like(n=2000, seed=3, per_utt=10)
    ids, _ = ds.select(rows)
    c = check(rows, ids)
    assert c['n_distinct_wer_primary_texts'] == c['wer_primary'] == 50


def brute_force(rows, per, min_wer, min_p, min_target):
    byg = {g: [r for r in rows if r['genders'] == g] for g in ds.GENDERS}
    targets = {r['target_speaker'] for r in rows}
    for combo in itertools.product(*[itertools.combinations(byg[g], per) for g in ds.GENDERS]):
        sel = [r for c in combo for r in c]
        u = Counter(r['target_speaker'] for r in sel)
        if (len({r['source_utt'] for r in sel}) == len(sel) and sum(bool(r['wer_primary']) for r in sel) >= min_wer
                and sum(bool(r['P']) for r in sel) >= min_p and all(u[t] >= min_target for t in targets)):
            return True
    return False


@pytest.mark.parametrize('shared', [False, True])
def test_small_random_lists_never_claim_false_infeasibility(shared):
    """Every subset returned meets the rule and 'infeasible' is said only when no subset does, also when sibling
    gender combinations share source utterances. With distinct sources the repair solves nearly every feasible list
    (the single-swap repair failed on about 1 in 15 of them)."""
    rng = random.Random(1)
    feasible = solved = 0
    for _ in range(400):
        spec = [(g, rng.choice(('fa', 'fb', 'fc') if g[-1] == 'F' else ('ma', 'mb', 'mc')), rng.choice('wp'))
                for g in ds.GENDERS for _ in range(rng.randint(2, 4))]
        rows = tiny(spec, {i: f'{g[0]}{rng.randint(0, 5)}' for i, (g, _, _) in enumerate(spec)} if shared else None)
        ok = brute_force(rows, 2, 4, 4, 1)
        feasible += ok
        try:
            ids, _ = ds.select(rows, per_gender=2, min_wer=4, min_p=4, min_target=1)
        except ValueError as e:
            assert not ok or 'infeasible' not in str(e), str(e)
            continue
        assert ok
        solved += 1
        c = ds.counts(rows, ids)
        assert c['genders'] == {g: 2 for g in ds.GENDERS} and c['wer_primary'] >= 4 and c['P'] >= 4
        assert c['n_targets'] == c['n_targets_in_list'] and c['n_distinct_source_utts'] == 8
    assert feasible > 50 and (shared or solved >= feasible - 2)


def test_skewed_cells_are_still_solved():
    """F2F holds only wer_primary pairs and F2M only P pairs, so M2F and M2M must supply the rest of each kind."""
    rows = [r for r in dev_like(seed=4, per_utt=1) if not (r['genders'] == 'F2F' and r['P']) and
            not (r['genders'] == 'F2M' and r['wer_primary'])]
    ids, info = ds.select(rows)
    c = check(rows, ids)
    assert c['cells']['F2F wer_primary'] == 25 and c['cells']['F2M P'] == 25
    assert c['wer_primary'] == c['P'] == 50


def test_quota_placement_keeps_the_rest_feasible(monkeypatch):
    """M2F and M2M hold only wer_primary pairs, F2M only P pairs: the 50 P pairs need all of F2F, so a greedy
    wer_primary placement in F2F must be refused by the max-flow check."""
    rows = [r for r in dev_like(seed=4, per_utt=1) if not (r['genders'] in ('M2F', 'M2M') and r['P']) and
            not (r['genders'] == 'F2M' and r['wer_primary'])]
    orig, refused = ds.shortfall, []
    monkeypatch.setattr(ds, 'shortfall', lambda *a: refused.append(orig(*a)) or refused[-1])
    ids, info = ds.select(rows)
    c = check(rows, ids)
    assert any(refused)
    assert c['cells'] == {'F2F P': 25, 'F2M P': 25, 'M2F wer_primary': 25, 'M2M wer_primary': 25}


def test_fill_balances_when_minimums_are_low():
    rows = dev_like(seed=5)
    ids, _ = ds.select(rows, min_wer=20, min_p=20)
    c = check(rows, ids, min_wer=20, min_p=20)
    assert c['wer_primary'] == c['P'] == 50


def first_per_utt(rows):
    return list({r['source_utt']: r for r in rows}.values())


@pytest.mark.parametrize('edit, match', [
    (lambda rows: [r for r in rows if r['genders'] != 'F2M'] + [r for r in rows if r['genders'] == 'F2M'][:24],
     'F2M has 24 pairs, 25 needed'),
    (lambda rows: [r for r in rows if not r['wer_primary']] + first_per_utt([r for r in rows if r['wer_primary']])[:40],
     'wer_primary: 50 more needed, at most 40 can be chosen'),
    (lambda rows: [r for r in rows if r['target_speaker'] != 'm03'] + [row(9000, 'f00', 'm03', True, False),
                                                                       row(9001, 'f01', 'm03', False, True)],
     r"targets \['m03'\] have fewer than 3 pairs with distinct source utterances"),
    (lambda rows: [r for r in rows if r['target_speaker'] != 'm03'] + [
        row(9000, 'f00', 'm03', True, False), row(9001, 'f01', 'm03', False, True),
        dict(row(9002, 'f01', 'm03', False, True), pair_id='dev__x__m03', source_utt='f01_9001')],
     r"targets \['m03'\] have fewer than 3 pairs with distinct source utterances"),
    (lambda rows: rows + [row(9100 + 3 * j + i, 'm00', f'fx{j}', i == 0, i != 0) for j in range(8) for i in range(3)],
     'infeasible: 18 F targets x 3 uses > 50 slots'),
    (lambda rows: rows + [rows[0]], 'duplicate pair_id'),
    (lambda rows: rows + [dict(rows[0], pair_id='x', genders='F2X')], 'unknown genders'),
])
def test_infeasible_lists_fail_with_a_clear_message(edit, match):
    with pytest.raises(ValueError, match=match):
        ds.select(edit(dev_like(seed=6)))


def test_repair_failure_names_the_target_without_claiming_infeasibility():
    """Target fx's only pairs are neither wer_primary nor P, and the 50/50 minimums leave no room for such a pair."""
    rows = dev_like(seed=7) + [row(9200 + i, 'm00', 'fx', False, False) for i in range(3)]
    with pytest.raises(ValueError, match='repair failed: target fx reaches only 0 of 3') as e:
        ds.select(rows)
    assert 'infeasible' not in str(e.value) and 'may still meet the rule' in str(e.value)


def test_load_ids_and_ids_arg(tmp_path):
    p = tmp_path / 'ids.txt'
    p.write_text('dev__a\n\ndev__b\n  \ndev__c\n')
    assert ds.load_ids(p) == ['dev__a', 'dev__b', 'dev__c']
    assert ds.ids_arg(p) == 'dev__a,dev__b,dev__c'


def test_cli_writes_keeps_and_guards(tmp_path, capsys):
    rows = dev_like(seed=8)
    pairs = tmp_path / 'dev.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = tmp_path / 'pairs' / 'dev_s100.txt'
    assert ds.main(['--pairs', str(pairs), '--out', str(out)]) == 0
    ids = ds.load_ids(out)
    check(rows, ids)
    doc = json.loads(out.with_suffix('.json').read_text())
    assert doc['ids'] == ids and doc['seed'] == ds.SEED and doc['counts']['n'] == 100
    assert doc['counts']['n_distinct_source_utts'] == 100
    assert doc['fields_sha256'] == ds.fields_sha256(rows) and len(doc['pairs_sha256']) == 64
    assert 'wer_primary 50, P 50' in capsys.readouterr().out
    # the same input with other paths gives the same subset and the same fields hash
    moved = [dict(r, P=r['P'].replace('/x/', '/y/') if r['P'] else None) for r in rows]
    assert ds.fields_sha256(moved) == doc['fields_sha256']
    mtime = out.stat().st_mtime_ns
    assert ds.main(['--pairs', str(pairs), '--out', str(out)]) == 0
    assert 'unchanged' in capsys.readouterr().out and out.stat().st_mtime_ns == mtime
    with pytest.raises(SystemExit, match='holds a different subset'):
        ds.main(['--pairs', str(pairs), '--out', str(out), '--seed', '1'])
    assert ds.load_ids(out) == ids
    assert ds.main(['--pairs', str(pairs), '--out', str(out), '--seed', '1', '--force']) == 0
    assert ds.load_ids(out) != ids and json.loads(out.with_suffix('.json').read_text())['seed'] == 1
    capsys.readouterr()
    assert ds.main(['--out', str(out), '--print_ids']) == 0
    assert capsys.readouterr().out.strip() == ','.join(ds.load_ids(out))
    assert not list(tmp_path.rglob('*.tmp'))


def test_cli_rewrites_a_stale_json(tmp_path, capsys):
    """A crash between the two writes leaves a .json describing another subset; the next run must not keep it."""
    rows = dev_like(seed=8)
    pairs = tmp_path / 'dev.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out, js = tmp_path / 's.txt', tmp_path / 's.json'
    assert ds.main(['--pairs', str(pairs), '--out', str(out)]) == 0
    good = json.loads(js.read_text())
    ds.main(['--pairs', str(pairs), '--out', str(out), '--seed', '1', '--force'])
    stale = json.loads(js.read_text())
    assert stale['ids'] != good['ids']
    out.write_text(''.join(i + '\n' for i in good['ids']))           # .txt back at the seed-20260928 subset
    capsys.readouterr()
    assert ds.main(['--pairs', str(pairs), '--out', str(out)]) == 0
    assert 'unchanged' not in capsys.readouterr().out
    doc = json.loads(js.read_text())
    assert doc['ids'] == good['ids'] == ds.load_ids(out) and doc['seed'] == ds.SEED
    js.write_text(json.dumps(dict(doc, seed=1)))                    # same ids, other seed recorded
    assert ds.main(['--pairs', str(pairs), '--out', str(out)]) == 0
    assert 'unchanged' not in capsys.readouterr().out and json.loads(js.read_text())['seed'] == ds.SEED


def test_cli_infeasible_exits_with_error(tmp_path):
    pairs = tmp_path / 'dev.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in dev_like()[:80]))
    with pytest.raises(SystemExit, match='ERROR: infeasible'):
        ds.main(['--pairs', str(pairs), '--out', str(tmp_path / 's.txt')])
    assert not (tmp_path / 's.txt').exists()


def test_committed_subset_file():
    ids = ds.load_ids(ds.OUT)
    assert len(ids) == len(set(ids)) == 100 and all(i.startswith('dev__') for i in ids) and ids == sorted(ids)
    doc = json.loads(ds.OUT.with_suffix('.json').read_text())
    assert doc['ids'] == ids and doc['seed'] == ds.SEED
    assert doc['counts']['n_distinct_source_utts'] == 100


@pytest.mark.skipif(not DEV.exists(), reason='Mac copy of the Dev pair list not present')
def test_committed_subset_matches_the_dev_list():
    rows = [json.loads(l) for l in open(DEV) if l.strip()]
    ids, _ = ds.select(rows)
    assert ids == ds.load_ids(ds.OUT)
    assert ds.select(list(reversed(rows)))[0] == ids
    for k in range(3):
        shuf = rows[:]
        random.Random(k).shuffle(shuf)
        assert ds.select(shuf)[0] == ids
    c = check(rows, ids)
    assert c['n_targets'] == 20 and c['n_distinct_wer_primary_texts'] == 50
    assert json.loads(ds.OUT.with_suffix('.json').read_text())['fields_sha256'] == ds.fields_sha256(rows)
