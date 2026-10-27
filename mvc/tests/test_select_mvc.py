"""Tests of mvc/analysis/select_mvc.py, the Dev selection R2 of mvc/ANALYSIS_PLAN_mvc_v1.md §4, on synthetic rows:
the rules of R2.0-R2.4 and B1 on numbers (boundaries, ties, every fallback), whole selections through the CLI on a
synthetic 1,000-pair Dev list (the main path; trigger i, fallback F, trigger ii, B1 kept with its second R2; B1 not
kept), the re-derivation stop, the record and mvc_selection.env, and the refusals (duplicate rows, missing rows, wrong
row counts 100 / 1,000, wrong or missing sha256, describe() commit, settings, target_copy, stats.py)."""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import select_mvc as sm  # noqa: E402
import stats  # noqa: E402

SPK = [f'f{i:02d}' for i in range(10)] + [f'm{i:02d}' for i in range(10)]
B1_DIR = sm.MVC_DS + '/runs/b1'
K = {'k150': 'ema_0150000.pt', 'k200': 'ema_0200000.pt', 'b1': 'ema_0020000.pt'}


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def n2(k):
    return 0.03 * (((k * 104729) % 17) - 8) / 8


def tc_value(k):
    return 0.50 + 0.05 * (((k * 31) % 7) - 3) / 3


# ── the synthetic world ──────────────────────────────────────

class World:
    """A 1,000-pair Dev list over 20 speakers (Dev-S100 = its first 100 pairs), target_copy@src on the pairs with P,
    a FREEZE file, and eval runs written as eval_dev.sh writes them, each file registered with its sha256."""

    def __init__(self, tmp, mp):
        self.tmp, self.root, self.reg = tmp, tmp / 'eval', {}
        self.pairs = []
        for k in range(1000):
            s = k % 20
            t = (s + 1 + (k // 20) % 19) % 20
            g = 'F' if s < 10 else 'M'
            gt_ = 'F' if t < 10 else 'M'
            self.pairs.append({'pair_id': f'dev__{SPK[s]}_{k:04d}__{SPK[t]}', 'set': 'dev', 'source_speaker': SPK[s],
                               'target_speaker': SPK[t], 'genders': f'{g}2{gt_}', 'wer_primary': (k // 2) % 2 == 0,
                               'P': f'/p/{k}.flac' if k % 2 == 0 else None})
        self.mp = mp
        self.k = {p['pair_id']: i for i, p in enumerate(self.pairs)}
        self.s100 = [p['pair_id'] for p in self.pairs[:100]]
        (tmp / 'dev.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in self.pairs))
        (tmp / 'dev_s100.txt').write_text('\n'.join(self.s100) + '\n')
        tc = [{'pair_id': p['pair_id'], 'system': 'target_copy@src', 'set': 'dev', 'budget': 30,
               'sim_wavlm': tc_value(i)} for i, p in enumerate(self.pairs) if p['P']]
        tc += [{'pair_id': p['pair_id'], 'system': 'source_copy@src', 'set': 'dev', 'budget': 30, 'sim_wavlm': 0.9}
               for p in self.pairs]
        self.write_tc(tc)
        code = ''.join(f'{sm.sha256(f)}  mvc/{f.name}\n' for f in sorted(sm.MVC.glob('*.py')))   # the code as is
        (tmp / 'FREEZE').write_text(f'# synthetic\n{sm.sha256(tmp / "dev_s100.txt")}  {sm.S100_KEY}\n'
                                    f'{sm.sha256(tmp / "dev.jsonl")}  {sm.CEAI_DEV_PAIRS}\n{code}')
        mp.setattr(sm, 'FREEZE_PATH', tmp / 'FREEZE')
        mp.setattr(sm, 'S100_PATH', tmp / 'dev_s100.txt')
        mp.setattr(sm, 'ANALYZE_MVC', tmp / 'analyze_mvc.py')
        mp.setattr(sm, 'ANALYZE_MVC_TEST', tmp / 'test_analyze_mvc.py')
        mp.setattr(sm, 'DEVIATIONS', tmp / 'DEVIATIONS.md')
        self.commit = sm.describe_commit()
        mp.setattr(sm, 'FROZEN_COMMIT', self.commit)

    def write_tc(self, rows):
        p = self.tmp / 'tc.jsonl'
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.mp.setattr(sm, 'TARGET_COPY_SRC_SHA256', {sm.sha256(p): 'synthetic target_copy'})

    def _write(self, rel, rows, raw=None):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw if raw is not None else ''.join(json.dumps(r) + '\n' for r in rows))
        self.reg[rel] = sm.sha256(p)

    def ids(self, pairset):
        return self.s100 if pairset == 's100' else [p['pair_id'] for p in self.pairs]

    def run(self, rel, conf, ckpt=K['k200'], pairset='s100', budget=30, run_dir=sm.R1_DIR, wer=2.0, wer_s100=None,
            sim=0.90, utmos=3.70, nl=(0.0, 0.0), ecapa=0.0, settings=None, commit=None, ecapa_rows=True, tgt_hook=None):
        name = rel.rsplit('/', 1)[-1]
        name = name[:-len(f'_b{budget}')] if budget != 30 else name
        sha = h(rel)[:16]
        ids = self.ids(pairset)
        tgt, src, et, es = [], [], [], []
        wer_ids = [k for k in ids if self.pairs[self.k[k]]['wer_primary']]
        pos = {k: i for i, k in enumerate(wer_ids)}
        for k in ids:
            i = self.k[k]
            p = self.pairs[i]
            g = stats.stratum(p['genders'])
            rate = (wer if wer_s100 is None or k not in self.s100 else wer_s100) / 10   # errors per 10-word pair
            j = pos.get(k, 0)
            e = int((j + 1) * rate) - int(j * rate)
            tgt.append({'pair_id': k, 'system': name, 'set': 'dev', 'budget': budget, 'genders': p['genders'],
                        'wer_primary': p['wer_primary'], 'conv_cfg_sha': sha, 'sim_wavlm': sim, 'utmos': utmos,
                        'wer_large_v3': {'S': e, 'D': 0, 'I': 0, 'N': 10, 'cS': e, 'cD': 0, 'cI': 0, 'cN': 50}})
            base = (tc_value(i) if p['P'] else 0.45) + n2(i)
            src.append({'pair_id': k, 'system': name + '@src', 'set': 'dev', 'budget': budget, 'conv_cfg_sha': sha,
                        'sim_wavlm': base + nl[0 if g == 'same' else 1]})
            et.append({'pair_id': k, 'system': name, 'set': 'dev', 'budget': budget, 'sim_ecapa': 0.6 - ecapa})
            es.append({'pair_id': k, 'system': name + '@src', 'set': 'dev', 'budget': budget,
                       'sim_ecapa': 0.2 + n2(i) + ecapa})
        if tgt_hook:
            tgt_hook(tgt, src)
        self._write(f'{rel}/metrics/dev_{budget}.jsonl', tgt)
        self._write(f'{rel}/metrics/dev_{budget}_src.jsonl', src)
        if ecapa_rows:
            self._write(f'{rel}/ecapa/dev_{budget}.jsonl', et)
            self._write(f'{rel}/ecapa/dev_{budget}_src.jsonl', es)
        m = re.fullmatch(r'ema_(\d{7})\.pt', ckpt)                  # R1: --steps 200000; B1 here: 20k steps
        step = int(m[1]) if m else 20000
        s = dict(sm.mvc_json(run_dir, ckpt, conf), **(settings or {}), train_step=step,
                 train_args={'steps': 200000 if run_dir == sm.R1_DIR else 20000})
        ev = {'event': 'start', 'conv_cfg_sha': sha, 'conv_cfg': {'system': name, 'describe': {
            'commit': commit or self.commit, 'checkpoints': {f'{run_dir}/{ckpt}': h(f'{run_dir}/{ckpt}')},
            'settings': s}}}
        self._write(f'{rel}/conv/dev/{budget}/{name}/runs.jsonl', [ev])
        return rel

    def recon(self, rel, wer):
        rows = []
        for j in range(300):
            e = int((j + 1) * wer / 10) - int(j * wer / 10)
            rows.append({'pair_id': f'recon__u{j:03d}', 'system': rel.rsplit('/', 1)[-1], 'wer_primary': True,
                         'wer_large_v3': {'S': e, 'D': 0, 'I': 0, 'N': 10, 'cS': e, 'cD': 0, 'cI': 0, 'cN': 50}})
        self._write(f'{rel}/metrics.jsonl', rows)
        return rel

    def sha_file(self):
        p = self.tmp / 'eval.sha256'
        p.write_text(''.join(f'{v}  {sm.EVAL_PREFIX + k if i % 2 else "./" + k}\n'
                             for i, (k, v) in enumerate(sorted(self.reg.items()))))
        return str(p)

    def args(self, stage, runs, *extra, rnd='r1'):
        a = ['--stage', stage, '--round', rnd, '--eval_root', str(self.root), '--sha256', self.sha_file(),
             '--pairs', str(self.tmp / 'dev.jsonl'), '--target_copy_src', str(self.tmp / 'tc.jsonl'),
             '--state_dir', str(self.tmp / 'state'), '--out', str(self.tmp / 'rec' / 'RECORD.md'),
             '--env', str(self.tmp / 'rec' / 'mvc_selection.env'), '--cfg_dir', str(self.tmp / 'rec' / 'cfg')]
        for flag, rels in runs.items():
            for r in rels:
                a += [f'--{flag}', r]
        return a + list(extra)

    def main(self, stage, runs, *extra, rnd='r1'):
        return sm.main(self.args(stage, runs, *extra, rnd=rnd))

    def state(self, sid):
        return json.load(open(sm.state_path(self.tmp / 'state', sid)))


def C(k):
    return sm.from_key(k)


# The main path: R2.0 keeps 200k; the screen's plain CFG 0 w 0.75 fails condition 3 only, so its interval variant
# replaces it and is the D-mvc-2 pick; CFG 1 w 1 fails condition 5 (ECAPA against WavLM); four finalists.
SCREEN_A = {
    'cfg0_w0': dict(wer=2.0, sim=0.890, utmos=3.70, nl=(0.08, 0.10)),
    'cfg0_w0.5': dict(wer=2.4, sim=0.900, utmos=3.70, nl=(0.06, 0.08), ecapa=-0.015),
    'cfg0_w0.75': dict(wer=2.4, sim=0.900, utmos=3.55, nl=(-0.03, -0.02), ecapa=-0.08),
    'cfg0_w1': dict(wer=2.0, sim=0.902, utmos=3.71, nl=(-0.05, -0.04), ecapa=-0.10),
    'cfg0_w0.5_u0.5': dict(wer=2.4, sim=0.900, utmos=3.70, nl=(-0.03, -0.02), ecapa=-0.08),
    'cfg0_w0.75_u0.5': dict(wer=2.4, sim=0.899, utmos=3.68, nl=(-0.03, -0.02), ecapa=-0.08),
    'cfg0_w1_u0.5': dict(wer=2.0, sim=0.902, utmos=3.72, nl=(-0.04, -0.03), ecapa=-0.09),
    'cfg1_w0': dict(wer=1.6, sim=0.905, utmos=3.65, nl=(0.06, 0.08)),
    'cfg1_w0.5': dict(wer=1.8, sim=0.908, utmos=3.67, nl=(-0.02, -0.02), ecapa=-0.08),
    'cfg1_w0.75': dict(wer=2.0, sim=0.908, utmos=3.67, nl=(-0.03, -0.03), ecapa=-0.09),
    'cfg1_w1': dict(wer=1.8, sim=0.906, utmos=3.67, nl=(-0.04, -0.04), ecapa=+0.02),
}
# Full Dev: the pick fails the WER check, CFG 1 w 0.5 fails L-pass; E3 = CFG 0 w 1 (lower WER on 1,000 pairs) and
# CFG 1 w 0.75 (first on Dev-900, where CFG 0 w 1's errors sit).
FULL_A = {
    'cfg0_w0': dict(wer=2.0, sim=0.890, utmos=3.70, nl=(0.08, 0.10)),
    'cfg0_w0.75_u0.5': dict(wer=3.5, sim=0.900, utmos=3.70, nl=(-0.03, -0.03)),
    'cfg1_w0.5': dict(wer=2.2, sim=0.912, utmos=3.68, nl=(0.02, 0.06)),
    'cfg1_w0.75': dict(wer=2.4, sim=0.908, utmos=3.67, nl=(-0.03, -0.02)),
    'cfg0_w1': dict(wer=2.5, wer_s100=0.0, sim=0.903, utmos=3.71, nl=(-0.05, -0.04)),
}
# R2.4: CFG 0 w 1 loses 0.07 SIM at 3 s (fails); CFG 1 w 0.75 passes.
BUDGET_A = {'cfg0_w1': (0.903 - 0.07, 0.905), 'cfg1_w0.75': (0.908 - 0.03, 0.906)}


def build_a(w, snap='k200', run_dir=sm.R1_DIR, ckpt=None, prefix='', budgets=('cfg0_w1', 'cfg1_w0.75'), commit=None):
    ckpt = ckpt or K[snap]
    kw = dict(ckpt=ckpt, run_dir=run_dir, commit=commit)
    runs = {'screen': [w.run(f'{prefix}r2_1/{k}', C(k), **kw, **v) for k, v in SCREEN_A.items()],
            'full': [w.run(f'{prefix}r2_full/{k}', C(k), pairset='full', **kw, **v) for k, v in FULL_A.items()],
            'budget': []}
    for k in budgets:
        s3, s180 = BUDGET_A[k]
        for b, s in ((3, s3), (180, s180)):
            runs['budget'].append(w.run(f'{prefix}r2_4/{k}_b{b}', C(k), pairset='full', budget=b, sim=s, **kw))
    return runs


def r2_0_runs(w, k150=None, k200=None, recon=(3.0, 3.4)):
    a = dict(dict(wer=2.0, sim=0.895, utmos=3.70), **(k150 or {}))
    b = dict(dict(wer=2.2, sim=0.897, utmos=3.71), **(k200 or {}))
    return {'ref': [w.run('r2_0/r2_0_k150_ref', sm.REF_CONF, ckpt=K['k150'], **a),
                    w.run('r2_0/r2_0_k200_ref', sm.REF_CONF, ckpt=K['k200'], **b)],
            'recon': [f'k150={w.recon("r2_0rec/r2_0_k150_rec", recon[0])}',
                      f'k200={w.recon("r2_0rec/r2_0_k200_rec", recon[1])}']}


def merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def census_line(w, rel, pairset):
    b = re.search(r'_b(\d+)$', rel)
    ids = w.s100 if pairset == 's100' else w.pairs[0::20] if pairset == 'screen' else [p['pair_id'] for p in w.pairs]
    ids = [x if isinstance(x, str) else x['pair_id'] for x in ids]
    return f'{rel}/summary.json\t{b[1] if b else 30}\t{len(ids)}\t' \
           f'{hashlib.sha256(chr(10).join(sorted(ids)).encode()).hexdigest()}\n'


def operator_files(w, runs, b1=False, extra_census=(), extra=()):
    """sacct, jobs, contamination, R1 hashes, census and the analyze_mvc files for --final. The census holds every Dev
    run given (as on ceai), three runs before the freeze and extra_census = [(rel, pairset)]."""
    tags = sorted(({r.split('=', 1)[-1].split('/', 1)[0] for v in runs.values() for r in v} |
                   {rel.split('/', 1)[0] for rel, _ in extra_census}) - sm.PRE_R2_TAGS) + (['b1_train'] if b1 else [])
    sacct = w.tmp / 'sacct.txt'
    sacct.write_text('JobID|JobName|Submit|End|State\n' + ''.join(
        f'{7000 + i}|mvc_eval|2026-10-28T0{i % 10}:00:00|2026-10-28T1{i % 10}:00:00|COMPLETED\n'
        for i in range(len(tags))))
    con = w.tmp / 'contamination.json'
    con.write_text(json.dumps({'passed': True, 'manifest': '/x/train_ids.txt', 'manifest_sha256': '0' * 64,
                               'command': 'pytest mvc/tests/test_train.py'}))
    r1h = w.tmp / 'r1.sha256'
    r1h.write_text(''.join(f'{h(f"{sm.R1_DIR}/{n}") if n in (K['k150'], K['k200']) else "1" * 64}  {sm.R1_DIR}/{n}\n'
                           for n in sm.R1_FILES))
    cen = w.tmp / 'census.tsv'
    lines = [census_line(w, 'e3/a', 's100'), census_line(w, 'e3/a_b3', 's100'), census_line(w, 'pilot/x', 'screen')]
    for flag, rels in runs.items():
        if flag != 'recon':
            lines += [census_line(w, rel, 's100' if flag in ('ref', 'screen') else 'full') for rel in rels]
    lines += [census_line(w, rel, ps) for rel, ps in extra_census]
    cen.write_text(''.join(lines))
    (w.tmp / 'analyze_mvc.py').write_text('# analyze\n')
    (w.tmp / 'test_analyze_mvc.py').write_text('# tests\n')
    out = ['--sacct', str(sacct), '--contamination', str(con), '--r1_hashes', str(r1h), '--census', str(cen)]
    for i, t in enumerate(tags):
        out += ['--jobs', f'{t}={7000 + i}']
    return out + list(extra)


# ── rules on numbers ─────────────────────────────────────────

def nums_of(**kw):
    """{key: numbers} for the eleven configurations; kw overrides per key."""
    out = {}
    for c in sm.GRID:
        k = sm.ckey(c)
        x = {'wer': 2.0, 'sim': 0.90, 'utmos': 3.70, 'nl': {'same': {'est': 0.0, 'se': 0.01},
                                                           'cross': {'est': 0.0, 'se': 0.01}}}
        for kk, v in kw.get(k, {}).items():
            if kk == 'nl':
                x['nl'] = {'same': {'est': v[0], 'se': v[2] if len(v) > 2 else 0.01},
                           'cross': {'est': v[1], 'se': v[2] if len(v) > 2 else 0.01}}
            else:
                x[kk] = v
        out[k] = x
    return out


def d5_all(met=True, **kw):
    return {sm.ckey(c): {'met': kw.get(sm.ckey(c), met)} for c in sm.GRID if c[1] > 0}


def test_r2_0_rule_each_condition_and_r1x():
    a = {'wer': 2.0, 'sim': 0.900, 'utmos': 3.70, 'recon_wer': 3.0}
    cond, snap, r1x = sm.r2_0_rule(a, dict(a))
    assert snap == 'k200' and not any(cond.values()) and r1x == {'sim_rose': False, 'utmos_rose': False,
                                                                  'runs': False}
    for key, val, which in (('recon_wer', 4.01, 'recon_wer'), ('sim', 0.8899, 'sim'), ('utmos', 3.6499, 'utmos'),
                            ('wer', 4.01, 'wer')):
        cond, snap, _ = sm.r2_0_rule(a, dict(a, **{key: val}))
        assert snap == 'k150' and [k for k, v in cond.items() if v] == [which]
    for key, val in (('recon_wer', 4.0), ('sim', 0.890), ('utmos', 3.65), ('wer', 4.0)):   # boundaries: not beyond
        assert sm.r2_0_rule(a, dict(a, **{key: val}))[1] == 'k200'
    assert sm.r2_0_rule(a, dict(a, sim=0.9001))[2]['runs'] and sm.r2_0_rule(a, dict(a, utmos=3.71))[2]['utmos_rose']


def test_screen_conditions_boundaries_and_eligibility():
    base = {'cfg0_w0': dict(nl=(0.05, 0.05)), 'cfg1_w0': dict(nl=(0.05, 0.05))}
    r = sm.screen_rule(nums_of(**base), d5_all())
    assert r['eligible'] == ['cfg0_w0.5', 'cfg0_w0.75', 'cfg0_w1', 'cfg1_w0.5', 'cfg1_w0.75', 'cfg1_w1']
    assert r['pick'] == 'cfg0_w0.5' and not r['trigger_i']
    # exactly on every threshold: NL +0.03, SIM c0 - 0.005, UTMOS c0 - 0.08, WER c0 + 2.0 -> met
    edge = dict(base, **{'cfg0_w0.5': dict(nl=(0.03, 0.03), sim=0.895, utmos=3.62, wer=4.0)})
    r = sm.screen_rule(nums_of(**edge), d5_all())
    assert all(r['flags']['cfg0_w0.5'].values())
    for k, v, cond in (('nl', (0.0301, 0.0), '1'), ('nl', (0.0, 0.0301), '1'), ('sim', 0.8949, '2'),
                       ('utmos', 3.6199, '3'), ('wer', 4.01, '4')):
        r = sm.screen_rule(nums_of(**dict(base, **{'cfg0_w0.5': {k: v}})), d5_all())
        assert [i for i, x in r['flags']['cfg0_w0.5'].items() if not x] == [cond]
        assert 'cfg0_w0.5' not in r['eligible']
    r = sm.screen_rule(nums_of(**base), d5_all(**{'cfg0_w0.5': False}))
    assert not r['flags']['cfg0_w0.5']['5'] and r['pick'] == 'cfg0_w0.75'
    # CFG 1 cells are judged against the CFG 1 w 0 cell
    r = sm.screen_rule(nums_of(**dict(base, cfg1_w0=dict(nl=(0.05, 0.05), sim=0.95))), d5_all())
    assert not r['flags']['cfg1_w0.5']['2'] and r['flags']['cfg0_w0.5']['2']


def test_interval_variant_clause():
    base = {'cfg0_w0': dict(nl=(0.05, 0.05)), 'cfg1_w0': dict(nl=(0.05, 0.05))}
    fails3 = dict(base, **{'cfg0_w0.5': dict(utmos=3.5), 'cfg0_w0.5_u0.5': dict(utmos=3.69)})
    r = sm.screen_rule(nums_of(**fails3), d5_all())
    assert 'cfg0_w0.5' not in r['eligible'] and 'cfg0_w0.5_u0.5' in r['eligible'] and r['pick'] == 'cfg0_w0.5_u0.5'
    # the plain cell fails 3 and another condition: the variant is not eligible, whatever it shows
    r = sm.screen_rule(nums_of(**dict(fails3, **{'cfg0_w0.5': dict(utmos=3.5, wer=5.0)})), d5_all())
    assert 'cfg0_w0.5_u0.5' not in r['eligible'] and r['pick'] == 'cfg0_w0.75'
    # the plain cell passes: the variant is never eligible (and is not the pick)
    r = sm.screen_rule(nums_of(**base), d5_all())
    assert not any(k.endswith('_u0.5') for k in r['eligible'])
    # the variant replaces the plain cell but must meet 1-5 itself against the same c0
    r = sm.screen_rule(nums_of(**dict(fails3, **{'cfg0_w0.5_u0.5': dict(utmos=3.69, nl=(0.04, 0.0))})), d5_all())
    assert 'cfg0_w0.5_u0.5' not in r['eligible'] and 'replaces' in r['eligibility']['cfg0_w0.5_u0.5']['why']
    r = sm.screen_rule(nums_of(**fails3), d5_all(**{'cfg0_w0.5_u0.5': False}))
    assert 'cfg0_w0.5_u0.5' not in r['eligible']


def test_w0_cells_pick_and_trigger_i():
    r = sm.screen_rule(nums_of(), d5_all())          # every NL 0: w 0 cells eligible by condition 1 alone
    assert 'cfg0_w0' in r['eligible'] and 'cfg1_w0' in r['eligible'] and r['pick'] == 'cfg0_w0'
    only1 = {sm.ckey(c): dict(nl=(0.05, 0.05)) for c in sm.GRID if c[0] == 0.0}
    only1['cfg1_w0'] = dict(nl=(0.05, 0.05))
    r = sm.screen_rule(nums_of(**only1), d5_all())
    assert r['eligible'] == ['cfg1_w0.5', 'cfg1_w0.75', 'cfg1_w1'] and r['pick'] is None and not r['trigger_i']
    none = {sm.ckey(c): dict(nl=(0.0, 0.031)) for c in sm.GRID}
    r = sm.screen_rule(nums_of(**none), d5_all())
    assert r['eligible'] == [] and r['pick'] is None and r['trigger_i']


def test_finalists_bounds_ties_and_fill():
    n = nums_of(**{'cfg0_w0.5': dict(sim=0.910, utmos=3.75),           # the pick
                   'cfg0_w0.75': dict(wer=2.004, sim=0.905),             # ties cfg1_w0.5 at two decimals, higher SIM
                   'cfg1_w0.5': dict(wer=2.001, sim=0.904),
                   'cfg1_w0.75': dict(wer=1.0, sim=0.899),               # below best SIM - 0.010: only as filler
                   'cfg1_w1': dict(wer=1.5, utmos=3.69),                 # below best UTMOS - 0.05
                   'cfg0_w1': dict(wer=3.0)})
    el = ['cfg0_w0.5', 'cfg0_w0.75', 'cfg0_w1', 'cfg1_w0.5', 'cfg1_w0.75', 'cfg1_w1']
    r = sm.finalists_rule(n, el, 'cfg0_w0.5')
    assert r['finalists'] == ['cfg0_w0.5', 'cfg0_w0.75', 'cfg1_w0.5', 'cfg0_w1'] and r['taken_by_wer'] == []
    assert r['best_sim'] == 0.910 and r['best_utmos'] == 3.75
    r = sm.finalists_rule(n, ['cfg0_w0.5', 'cfg0_w0.75', 'cfg1_w0.75', 'cfg1_w1'], 'cfg0_w0.5')
    assert r['finalists'] == ['cfg0_w0.5', 'cfg0_w0.75', 'cfg1_w0.75', 'cfg1_w1'] and r['taken_by_wer'] == \
        ['cfg1_w0.75', 'cfg1_w1']                                    # fewer than three in bounds: the rest by WER
    r = sm.finalists_rule(n, ['cfg1_w0.5', 'cfg1_w0.75'], None)      # no pick: only the other eligible ones
    assert r['finalists'] == ['cfg1_w0.75', 'cfg1_w0.5']
    eq = nums_of(**{'cfg0_w0.5': dict(sim=0.95), 'cfg0_w1': dict(), 'cfg1_w0.5': dict(), 'cfg0_w1_u0.5': dict(),
                    'cfg1_w1': dict(), 'cfg0_w0.75': dict()})
    r = sm.finalists_rule(eq, ['cfg0_w0.5', 'cfg0_w1', 'cfg1_w0.5', 'cfg0_w1_u0.5', 'cfg1_w1', 'cfg0_w0.75'],
                          'cfg0_w0.5')
    # all tie on WER, SIM, UTMOS -> smaller w, smaller CFG, plain before interval; the 0.95 pick sets best SIM
    assert r['within_bounds'] == [] and r['finalists'] == ['cfg0_w0.5', 'cfg1_w0.5', 'cfg0_w0.75', 'cfg0_w1']


def test_finalists_without_eligible_take_the_smallest_nl():
    n = nums_of(**{'cfg0_w1': dict(nl=(0.05, 0.04)), 'cfg1_w1': dict(nl=(0.04, 0.045)),
                   'cfg0_w1_u0.5': dict(nl=(0.045, 0.045)), 'cfg1_w0.75': dict(nl=(0.05, 0.05)),
                   'cfg0_w0.75': dict(nl=(0.05, 0.05))})
    n = {k: (v if k in ('cfg0_w1', 'cfg1_w1', 'cfg0_w1_u0.5', 'cfg1_w0.75', 'cfg0_w0.75') else
             dict(v, nl={'same': {'est': 0.2, 'se': 0.01}, 'cross': {'est': 0.2, 'se': 0.01}})) for k, v in n.items()}
    r = sm.finalists_rule(n, [], None)
    # max NL 0.045 twice (equal w -> smaller CFG first), then 0.05 three times (smaller w, then smaller CFG)
    assert r['finalists'] == ['cfg0_w1_u0.5', 'cfg1_w1', 'cfg0_w0.75', 'cfg1_w0.75'] and 'trigger i' in r['how']


def full_nums(**kw):
    base = {'wer': 2.0, 'sim': 0.90, 'utmos': 3.70}
    out = {}
    for k, v in kw.items():
        x = dict(base, **{a: b for a, b in v.items() if a != 'nl'})
        nlv = v.get('nl', (-0.05, -0.05, 0.01))
        x['nl'] = {'same': {'est': nlv[0], 'se': nlv[2]}, 'cross': {'est': nlv[1], 'se': nlv[2]}}
        out[k] = x
    return out


def test_choice_l_pass_best_sets_wer_check_and_order():
    n = full_nums(**{'cfg0_w1': dict(wer=2.5), 'cfg1_w0.5': dict(sim=0.95, nl=(0.0, 0.0201, 0.01)),
                     'cfg0_w0.75': dict(wer=2.3, sim=0.889), 'cfg1_w0.75': dict(wer=3.01),
                     'cfg0_w0.5': dict(nl=(0.02, 0.02, 0.01))})
    fin = list(n)
    r = sm.choice_rule(n, fin, 2.0)
    p = r['per']
    assert not p['cfg1_w0.5']['l_pass'] and p['cfg0_w0.5']['l_pass']          # 0.0201 > 0.03 - 0.01; 0.02 = 0.02
    assert r['best_sim'] == 0.90                                              # the non-L-passing 0.95 does not count
    assert not p['cfg0_w0.75']['sim_ok'] and not p['cfg1_w0.75']['wer_check'] and p['cfg0_w1']['wer_check']
    assert r['order'] == ['cfg0_w0.5', 'cfg0_w1'] and r['fallback_f'] is None and r['first'] == 'cfg0_w0.5'
    # WER ties at two decimals -> higher SIM -> higher UTMOS -> smaller w -> smaller CFG -> plain before interval
    t = full_nums(**{'cfg0_w1': dict(wer=2.004), 'cfg1_w0.5': dict(wer=2.001), 'cfg0_w0.75': dict(wer=2.0, sim=0.901),
                     'cfg0_w0.75_u0.5': dict(wer=2.0, sim=0.901), 'cfg1_w1': dict(utmos=3.701)})
    assert sm.choice_rule(t, list(t), 1.5)['order'] == ['cfg0_w0.75', 'cfg0_w0.75_u0.5', 'cfg1_w1', 'cfg1_w0.5',
                                                        'cfg0_w1']
    # unrounded, cfg1_w1 has the lower WER and the higher SIM; at two / three decimals they tie -> smaller CFG
    u = full_nums(**{'cfg0_w1': dict(wer=2.004, sim=0.8996), 'cfg1_w1': dict(wer=2.0, sim=0.9001)})
    assert sm.choice_rule(u, list(u), 1.5)['order'] == ['cfg0_w1', 'cfg1_w1']


def test_fallback_f():
    n = full_nums(**{'cfg0_w1': dict(sim=0.900, nl=(0.01, 0.05, 0.01)), 'cfg1_w1': dict(sim=0.880, nl=(0.0, 0.03,
                                                                                                   0.01)),
                     'cfg1_w0.75': dict(sim=0.905, nl=(0.03, 0.04, 0.01)), 'cfg0_w0.75': dict(sim=0.894,
                                                                                               nl=(0.0, 0.035, 0.01))})
    r = sm.choice_rule(n, list(n), 2.0)
    # within 0.010 of the top SIM 0.905: cfg0_w1 (max NL 0.05) and cfg1_w0.75 (0.04); cfg1_w1 (0.03) is too low
    assert r['order'] == [] and r['fallback_f'] == 'cfg1_w0.75' and r['first'] == 'cfg1_w0.75'
    t = full_nums(**{'cfg0_w1': dict(sim=0.9004, nl=(0.0, 0.05, 0.01)), 'cfg1_w0.5': dict(sim=0.9001,
                                                                                         nl=(0.05, 0.0, 0.01)),
                     'cfg0_w0.5': dict(sim=0.902, nl=(0.05, 0.0, 0.01)), 'cfg0_w0.5_u0.5': dict(sim=0.902,
                                                                                             nl=(0.05, 0.0, 0.01))})
    # equal max NL 0.05 -> the higher SIM (three decimals) -> smaller w -> smaller CFG -> plain first
    assert sm.choice_rule(t, list(t), 2.0)['fallback_f'] == 'cfg0_w0.5'
    t2 = {k: v for k, v in t.items() if k not in ('cfg0_w0.5', 'cfg0_w0.5_u0.5')}
    assert sm.choice_rule(t2, list(t2), 2.0)['fallback_f'] == 'cfg1_w0.5'


def test_budget_rule_paths():
    full = {'order': ['a', 'b', 'c'], 'fallback_f': None, 'numbers': {k: {'sim': 0.90} for k in 'abcF'}}
    ok, low3, low180 = {3: 0.84, 180: 0.895}, {3: 0.8399, 180: 0.90}, {3: 0.90, 180: 0.8949}
    r = sm.budget_rule(full, {'a': ok})
    assert r['status'] == 'complete' and r['choice'] == 'a' and not r['r2_4_failed'] and len(r['checked']) == 1
    r = sm.budget_rule(full, {'a': low3})
    assert r['status'] == 'pending' and r['needs'] == ['b'] and not r['checked'][0]['pass']
    r = sm.budget_rule(full, {'a': low3, 'b': ok, 'c': ok})
    assert r['choice'] == 'b' and [c['config'] for c in r['checked']] == ['a', 'b']
    r = sm.budget_rule(full, {'a': low3, 'b': low180, 'c': ok})        # at most two checked
    assert r['choice'] == 'a' and r['r2_4_failed'] and len(r['checked']) == 2
    r = sm.budget_rule(dict(full, order=['a']), {'a': low180})
    assert r['choice'] == 'a' and r['r2_4_failed']
    r = sm.budget_rule(full, {})
    assert r['status'] == 'pending' and r['needs'] == ['a']
    f = dict(full, order=[], fallback_f='F')
    assert sm.budget_rule(f, {})['needs'] == ['F']
    r = sm.budget_rule(f, {'F': low3})
    assert r['choice'] == 'F' and r['source'] == 'F' and r['r2_4_failed']


def test_b1_keep_rule():
    good = {'sim_src_same': -0.02, 'sim_src_cross': -0.03, 'sim': -0.005, 'utmos': -0.05, 'wer': 0.5}
    assert sm.b1_rule(good)[1]
    for k, v in (('sim_src_same', -0.0199), ('sim_src_cross', -0.0199), ('sim', -0.0051), ('utmos', -0.0501),
                 ('wer', 0.51)):
        fl, kept = sm.b1_rule(dict(good, **{k: v}))
        assert not kept and [x for x, y in fl.items() if not y] == [k]


def test_mvc_json_has_every_default_key_in_order():
    j = sm.mvc_json(sm.R1_DIR, K['k200'], (1.0, 0.75, 1.0))
    assert list(j) == list(sm.vc_adapter.DEFAULTS)
    assert j['temp'] == 0.0 and j['relax'] == 0.15 and j['relax_k'] == 2 and j['cfg'] == 1.0 and j['neg_src'] == 0.75
    assert j['neg_src_until'] == 1.0 and j['stats_path'] is None and j['decode'] == 'latent'
    assert sm.from_key(sm.ckey((0.0, 0.75, 0.5))) == (0.0, 0.75, 0.5) and len(sm.GRID) == 11


# ── whole selections through the CLI ─────────────────────────

def b1_code(tmp_path, monkeypatch):
    """B1's code: a copy of mvc/*.py with train.py changed; returns its describe() commit."""
    code = tmp_path / 'code_b1'
    code.mkdir()
    for f in sm.MVC.glob('*.py'):
        shutil.copy(f, code / f.name)
    (code / 'train.py').write_text((code / 'train.py').read_text() + '\n# B1: shifted content, p_shift 0.5\n')
    monkeypatch.setattr(sm, 'CODE_DIR', code)
    return sm.describe_commit()


def test_main_path_record_and_env(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(stats, 'gate_l_strata', lambda *a, **k: calls.append(1))      # NL never goes through it
    runs = merge(r2_0_runs(w), build_a(w, budgets=('cfg0_w1',)))
    w.main('r2_0', runs)
    s0 = w.state('r1/r2_0')
    assert s0['snapshot']['ckpt'] == K['k200'] and s0['r1x']['runs'] and s0['commit'] == w.commit
    w.main('r2_1', runs)
    s1 = w.state('r1/r2_1')
    assert s1['eligible'] == ['cfg0_w1', 'cfg1_w0.5', 'cfg1_w0.75', 'cfg0_w0.75_u0.5']
    assert s1['pick'] == 'cfg0_w0.75_u0.5' and not s1['trigger_i']
    fl = s1['flags']
    assert [i for i, v in fl['cfg0_w0.75'].items() if not v] == ['3'] and not fl['cfg1_w1']['5']
    assert not fl['cfg0_w0.5']['1'] and s1['d5']['cfg0_w1']['met']
    n_cross = sum(stats.stratum(w.pairs[w.k[k]]['genders']) == 'cross' for k in w.s100)
    assert s1['d5']['cfg0_w1']['cross']['n_wavlm'] == s1['d5']['cfg0_w1']['cross']['n_ecapa'] == n_cross
    assert all(s1['numbers'][k]['nl'][g]['fail_b'] == 0 for k in s1['numbers'] for g in sm.STRATA)
    w.main('r2_2', runs)
    assert w.state('r1/r2_2')['finalists'] == ['cfg0_w0.75_u0.5', 'cfg1_w0.5', 'cfg1_w0.75', 'cfg0_w1']
    w.main('r2_3', runs)
    s3 = w.state('r1/r2_3')
    per = s3['full']['per']
    assert not per['cfg0_w0.75_u0.5']['wer_check'] and not per['cfg1_w0.5']['l_pass']
    assert s3['full']['order'] == ['cfg0_w1', 'cfg1_w0.75'] and s3['d900']['order'] == ['cfg1_w0.75', 'cfg0_w1']
    assert not s3['d900_agrees']
    nl = s3['full']['numbers']['cfg0_w1']['nl']
    assert nl['same']['n'] == sum(1 for p in w.pairs if p['P'] and stats.stratum(p['genders']) == 'same')
    w.main('r2_4', runs)                               # only the first's budget runs: it fails, the second is needed
    s4 = w.state('r1/r2_4')
    assert s4['status'] == 'pending' and s4['needs'] == ['cfg1_w0.75']
    with pytest.raises(SystemExit, match='pending'):
        w.main('record', runs)
    runs['budget'] += [w.run(f'r2_4/cfg1_w0.75_b{b}', C('cfg1_w0.75'), pairset='full', budget=b, sim=s)
                       for b, s in zip((3, 180), BUDGET_A['cfg1_w0.75'])]
    w.main('r2_4', runs)
    s4 = w.state('r1/r2_4')
    assert s4['status'] == 'complete' and s4['choice'] == 'cfg1_w0.75' and s4['source'] == 'E3'
    assert [c['pass'] for c in s4['checked']] == [False, True] and not s4['b1_triggered'] and not s4['r5']
    with pytest.raises(SystemExit, match='not triggered'):
        w.main('b1', runs)
    w.main('record', runs)                                                              # draft
    md = (tmp_path / 'rec' / 'RECORD.md').read_text()
    assert 'DRAFT' in md and 'PLACEHOLDER' in md and not (tmp_path / 'rec' / 'mvc_selection.env').exists()
    with pytest.raises(SystemExit, match='final record needs'):
        w.main('record', runs, '--final')
    ops = operator_files(w, runs)
    w.main('record', runs, '--final', *ops)
    rec = json.load(open(tmp_path / 'rec' / 'RECORD.json'))
    assert rec['status'] == 'complete' and rec['mvc_v1']['config'] == 'cfg1_w0.75' and calls == []
    env = dict(x.split('=', 1) for x in (tmp_path / 'rec' / 'mvc_selection.env').read_text().split())
    assert env['RECORD_SHA256'] == sm.sha256(tmp_path / 'rec' / 'RECORD.md')
    assert env['SNAPSHOT'] == f'{sm.R1_DIR}/{K["k200"]}' and env['SNAPSHOT_SHA256'] == h(env['SNAPSHOT'])
    assert env['DESCRIBE_COMMIT'] == w.commit
    cfg = tmp_path / 'rec' / 'cfg'
    for name, key in (('CFG_MVC_V1', 'mvc_v1'), ('CFG_MVC_V1_Q32', 'mvc_v1_q32'),
                      ('CFG_MVC_COPY_LATENT', 'mvc_copy_latent')):
        assert env[name] == f'{key}.json' and sm.sha256(cfg / env[name]) == env[name + '_SHA256']
    v1 = json.load(open(cfg / 'mvc_v1.json'))
    assert v1 == sm.mvc_json(sm.R1_DIR, K['k200'], (1.0, 0.75, 1.0))
    assert json.load(open(cfg / 'mvc_v1_q32.json')) == dict(v1, decode='codes')
    assert json.load(open(cfg / 'mvc_copy_latent.json')) == dict(sm.vc_adapter.DEFAULTS, mode='copy_latent')
    c = rec['operator']['census']
    assert c['before_r2'] == {'Dev-S100': {'30': 1, '3': 1}, 'screen 0/20': {'30': 1}}
    # the R2 runs are the 2 + 11 + 5 + 4 Dev runs; the two reconstructions are not Dev configurations
    assert len(c['r2_runs']) == 22 and len(rec['dev_runs_used']) == 22 and len(rec['recon_runs_used']) == 2
    assert 'plus the R2 runs (summary.json files under the R2 and B1 TAGs, used or not): 22, of which 22 used' in \
        (tmp_path / 'rec' / 'RECORD.md').read_text()
    assert rec['code']['changed_since_freeze'] == {} and rec['mvc_v1']['describe_commit'] == w.commit
    rb = rec['rb_dev_counterparts']
    assert set(rb) == {'3', '30', '180'} and abs(rb['3']['sim_wavlm']['est'] - BUDGET_A['cfg1_w0.75'][0]) < 1e-9
    t = rec['rounds']['r1']['tables']
    assert set(t['full']) == {'cfg0_w0', 'cfg0_w0.75_u0.5', 'cfg1_w0.5', 'cfg1_w0.75', 'cfg0_w1'}
    assert set(t['d900']['cfg0_w1']['gender']) == set(sm.GENDER_PAIRS) and 'ecapa' in t['full']['cfg0_w1']
    assert rec['triggers']['r1'] == {'trigger_i': False, 'trigger_ii': False, 'b1_triggered': False, 'r5': False}


def test_rederivation_stops_on_a_difference(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    runs = merge(r2_0_runs(w), build_a(w))
    with pytest.raises(SystemExit, match='no saved output'):
        w.main('r2_1', runs)                                    # R2.0 first
    w.main('r2_0', runs)
    w.main('r2_1', runs)
    w.main('r2_1', runs)                                        # same files, same output: allowed
    p = sm.state_path(tmp_path / 'state', 'r1/r2_1')
    s = json.load(open(p))
    s['pick'] = 'cfg0_w1'
    json.dump(s, open(p, 'w'))
    with pytest.raises(SystemExit, match=r'STOP .*r1/r2_1.*/pick'):
        w.main('r2_2', runs)
    with pytest.raises(SystemExit, match=r'STOP .*r1/r2_1'):
        w.main('r2_1', runs)                                    # a stage's own earlier output differs: stop
    os.remove(p)
    w.main('r2_1', runs)
    # a re-scored input, re-listed with its new sha256: the numbers change and the earlier stage no longer matches
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), **dict(SCREEN_A['cfg0_w1'], sim=0.80))
    with pytest.raises(SystemExit, match=r'STOP .*r1/r2_1'):
        w.main('r2_2', runs)


def test_trigger_i_fallback_f_trigger_ii_and_b1_kept(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    r0 = r2_0_runs(w, recon=(3.0, 4.5))                          # reconstruction +1.5 at 200k -> 150k
    # stats_path is allowed on the 150k reference run only (job e5's run differs in it alone)
    r0['ref'][0] = w.run('r2_0/r2_0_k150_ref', sm.REF_CONF, ckpt=K['k150'], wer=2.0, sim=0.895, utmos=3.70,
                         settings={'stats_path': '/x/stats.jsonl'})
    nl = {'cfg1_w1': 0.055, 'cfg0_w1': 0.06, 'cfg1_w0.75': 0.065, 'cfg0_w1_u0.5': 0.07}
    screen = [w.run(f'r2_1/{sm.ckey(c)}', c, ckpt=K['k150'], nl=(-0.05, nl.get(sm.ckey(c), 0.10)))
              for c in sm.GRID]
    full_eff = {'cfg0_w0': dict(nl=(0.08, 0.10)), 'cfg1_w1': dict(sim=0.880, nl=(-0.05, 0.055)),
                'cfg0_w1': dict(sim=0.900, nl=(-0.05, 0.06)), 'cfg1_w0.75': dict(sim=0.905, nl=(-0.05, 0.065)),
                'cfg0_w1_u0.5': dict(sim=0.899, nl=(-0.05, 0.07))}
    full = [w.run(f'r2_full/{k}', C(k), ckpt=K['k150'], pairset='full', **v) for k, v in full_eff.items()]
    budget = [w.run(f'r2_4/cfg0_w1_b{b}', C('cfg0_w1'), ckpt=K['k150'], pairset='full', budget=b, sim=s)
              for b, s in ((3, 0.85), (180, 0.90))]
    runs = merge(r0, {'screen': screen, 'full': full, 'budget': budget})
    for st in sm.STAGES:
        w.main(st, runs)
    assert w.state('r1/r2_0')['snapshot']['ckpt'] == K['k150'] and w.state('r1/r2_0')['conditions']['recon_wer']
    s1, s2 = w.state('r1/r2_1'), w.state('r1/r2_2')
    assert s1['trigger_i'] and s1['pick'] is None
    assert s2['finalists'] == ['cfg1_w1', 'cfg0_w1', 'cfg1_w0.75', 'cfg0_w1_u0.5']
    s3, s4 = w.state('r1/r2_3'), w.state('r1/r2_4')
    assert s3['full']['order'] == [] and s3['full']['fallback_f'] == 'cfg0_w1'        # cfg1_w1 is 0.025 below in SIM
    assert s4['choice'] == 'cfg0_w1' and s4['source'] == 'F' and not s4['r2_4_failed']
    assert s4['trigger_i'] and s4['trigger_ii'] and s4['b1_triggered']
    with pytest.raises(SystemExit, match='--b1_run'):
        w.main('b1', runs)
    # B1 at the parent setting of trigger i (CFG 0, SNG 0): SIM_src -0.04 in both strata -> kept. B1 changed train.py,
    # so B1 and its second R2 carry another describe() commit than R1's frozen one.
    b1c = b1_code(tmp_path, monkeypatch)
    runs['b1_run'] = [w.run('b1eval/b1_nosng', sm.NOSNG_CONF, ckpt=K['b1'], run_dir=B1_DIR, pairset='full',
                            wer=2.2, utmos=3.69, nl=(0.04, 0.06), commit=b1c)]
    w.main('b1', runs)
    b = w.state('b1')
    assert b['parent']['config'] == 'cfg0_w0' and b['kept'] and abs(b['deltas']['sim_src_same']['est'] + 0.04) < 1e-9
    assert b['snapshot']['commit'] == b1c and b['snapshot']['train_step'] == 20000
    with pytest.raises(SystemExit, match='R2.1 needs'):
        w.main('r2_1', runs, rnd='b1')
    runs = merge(runs, build_a(w, run_dir=B1_DIR, ckpt=K['b1'], prefix='b1r2_', commit=b1c))
    w.run('b1r2_r2_1/cfg0_w1', C('cfg0_w1'), ckpt=K['b1'], run_dir=B1_DIR, **SCREEN_A['cfg0_w1'])   # frozen commit
    with pytest.raises(SystemExit, match='one code for the whole round'):
        w.main('r2_1', runs, rnd='b1')
    w.run('b1r2_r2_1/cfg0_w1', C('cfg0_w1'), ckpt=K['b1'], run_dir=B1_DIR, commit=b1c, **SCREEN_A['cfg0_w1'])
    for st in sm.STAGES[1:]:
        w.main(st, runs, rnd='b1')
    assert w.state('b1/r2_4')['choice'] == 'cfg1_w0.75' and not w.state('b1/r2_4')['b1_triggered']
    ops = operator_files(w, runs, b1=True)
    with pytest.raises(SystemExit, match=r'final record needs.*without a deviation entry.*mvc/train.py'):
        w.main('record', runs, '--final', *ops)
    new = sm.sha256(sm.CODE_DIR / 'train.py')
    (tmp_path / 'DEVIATIONS.md').write_text(f'- B1: mvc/train.py, new sha256 {new}\n')
    w.main('record', runs, '--final', *ops)
    rec = json.load(open(tmp_path / 'rec' / 'RECORD.json'))
    assert rec['final_round'] == 'b1' and rec['mvc_v1']['config'] == 'cfg1_w0.75'
    assert rec['mvc_v1']['snapshot']['run_dir'] == B1_DIR and rec['mvc_v1']['snapshot']['ckpt'] == K['b1']
    assert json.load(open(tmp_path / 'rec' / 'cfg' / 'mvc_v1.json'))['run_dir'] == B1_DIR
    assert rec['triggers']['r1']['b1_triggered'] and 'b1' in rec['triggers']
    assert rec['code']['changed_since_freeze'] == {'mvc/train.py': new} and rec['mvc_v1']['describe_commit'] == b1c
    env = dict(x.split('=', 1) for x in (tmp_path / 'rec' / 'mvc_selection.env').read_text().split())
    assert env['DESCRIBE_COMMIT'] == b1c


def test_b1_not_kept_keeps_r1s_fallback(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    screen = [w.run(f'r2_1/{sm.ckey(c)}', c, nl=(-0.05, 0.10 - 0.01 * c[1])) for c in sm.GRID]   # nothing eligible
    s2 = sm.finalists_rule({sm.ckey(c): {'nl': {'same': {'est': -0.05}, 'cross': {'est': 0.10 - 0.01 * c[1]}},
                                         'wer': 2, 'sim': 0.9, 'utmos': 3.7} for c in sm.GRID}, [], None)['finalists']
    full = [w.run(f'r2_full/{k}', C(k), pairset='full', utmos=3.519, nl=(-0.05, 0.10 - 0.01 * C(k)[1]))
            for k in dict.fromkeys(s2 + ['cfg0_w0'])]                  # UTMOS below 3.82 - 0.3: the R5 trigger
    f = sm.choice_rule({k: {'wer': 2.0, 'sim': 0.9, 'utmos': 3.7, 'nl': {'same': {'est': -0.05, 'se': 0.01},
                                                                       'cross': {'est': 0.10 - 0.01 * C(k)[1],
                                                                                 'se': 0.01}}} for k in s2},
                       s2, 2.0)['fallback_f']
    budget = [w.run(f'r2_4/{f}_b{b}', C(f), pairset='full', budget=b, sim=0.88) for b in (3, 180)]
    runs = merge(r2_0_runs(w), {'screen': screen, 'full': full, 'budget': budget})
    for st in sm.STAGES:
        w.main(st, runs)
    s4 = w.state('r1/r2_4')
    assert s4['source'] == 'F' and s4['choice'] == f and s4['r2_4_failed'] and s4['b1_triggered'] and s4['r5']
    # §4.7: B1's final EMA snapshot, ema_<--steps>.pt; neither ema_last.pt nor an earlier snapshot
    for ckpt in ('ema_last.pt', 'ema_0010000.pt'):
        runs['b1_run'] = [w.run('b1eval/b1_nosng', sm.NOSNG_CONF, ckpt=ckpt, run_dir=B1_DIR, pairset='full',
                                nl=(-0.10, -0.10))]
        with pytest.raises(SystemExit, match='final EMA snapshot'):
            w.main('b1', runs)
    # B1 changed mvc/*.py: its runs carry another commit; R1's runs keep the frozen one
    b1c = b1_code(tmp_path, monkeypatch)
    runs['b1_run'] = [w.run('b1eval/b1_nosng', sm.NOSNG_CONF, ckpt=K['b1'], run_dir=B1_DIR, pairset='full',
                            utmos=3.519, nl=(-0.06, 0.095), commit=b1c)]      # ΔSIM_src -0.01 / -0.005: not kept
    w.main('b1', runs)
    assert not w.state('b1')['kept'] and w.state('b1')['snapshot']['commit'] == b1c
    with pytest.raises(SystemExit, match='not kept'):
        w.main('r2_1', runs, rnd='b1')
    w.main('record', runs)
    rec = json.load(open(tmp_path / 'rec' / 'RECORD.json'))
    assert rec['final_round'] == 'r1' and rec['mvc_v1']['config'] == f and 'not kept' in rec['mvc_v1']['lineage']
    assert rec['mvc_v1']['describe_commit'] == w.commit != b1c
    # mvc_v1 is R1's: the record must be written with the frozen code, which T1 then runs (§2.1)
    with pytest.raises(SystemExit, match=r"final record needs.*give describe\(\) commit " + b1c):
        w.main('record', runs, '--final', *operator_files(w, runs, b1=True))
    monkeypatch.setattr(sm, 'CODE_DIR', sm.MVC)
    w.main('record', runs, '--final', *operator_files(w, runs, b1=True))
    env = dict(x.split('=', 1) for x in (tmp_path / 'rec' / 'mvc_selection.env').read_text().split())
    assert env['DESCRIBE_COMMIT'] == w.commit


# ── refusals ─────────────────────────────────────────────────

@pytest.fixture
def screen_world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    runs = merge(r2_0_runs(w), {'screen': [w.run(f'r2_1/{k}', C(k), **v) for k, v in SCREEN_A.items()]})
    w.main('r2_0', runs)
    return w, runs


def test_refuses_duplicate_rows(screen_world):
    w, runs = screen_world
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), tgt_hook=lambda t, s: t.append(dict(t[0])), **SCREEN_A['cfg0_w1'])
    with pytest.raises(SystemExit, match='occurs twice'):
        w.main('r2_1', runs)


def test_refuses_missing_rows_and_wrong_counts(screen_world):
    w, runs = screen_world
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), tgt_hook=lambda t, s: (t.pop(), s.pop()), **SCREEN_A['cfg0_w1'])
    with pytest.raises(SystemExit, match='99 rows.*incomplete'):
        w.main('r2_1', runs)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), tgt_hook=lambda t, s: s.pop(), **SCREEN_A['cfg0_w1'])
    with pytest.raises(SystemExit, match='@src rows cover 99'):
        w.main('r2_1', runs)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), pairset='full', **SCREEN_A['cfg0_w1'])       # 1,000 rows given as a screen run
    with pytest.raises(SystemExit, match='screen runs Dev-S100'):
        w.main('r2_1', runs)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), **SCREEN_A['cfg0_w1'])
    runs2 = dict(runs, screen=[r for r in runs['screen'] if not r.endswith('cfg1_w1')])
    with pytest.raises(SystemExit, match=r"R2.1 needs .*cfg1_w1"):
        w.main('r2_1', runs2)
    w.run('r2_full/cfg0_w0', C('cfg0_w0'), pairset='full', tgt_hook=lambda t, s: (t.pop(), s.pop()))
    with pytest.raises(SystemExit, match='999 rows'):
        w.main('r2_1', dict(runs, full=['r2_full/cfg0_w0']))
    w.run('r2_full/cfg0_w0', C('cfg0_w0'), pairset='s100')                             # 100 rows given as full Dev
    w.main('r2_1', dict(runs, full=['r2_full/cfg0_w0']))                               # not used before R2.3
    w.main('r2_2', dict(runs, full=['r2_full/cfg0_w0']))
    with pytest.raises(SystemExit, match='full Dev at 30 s'):
        w.main('r2_3', dict(runs, full=['r2_full/cfg0_w0']))


def test_refuses_sha256_problems(screen_world):
    w, runs = screen_world
    shalist = w.sha_file()
    p = w.root / 'r2_1/cfg0_w1/metrics/dev_30.jsonl'
    p.write_text(p.read_text().replace('"utmos": 3.71', '"utmos": 3.91'))
    a = w.args('r2_1', runs)
    a[a.index('--sha256') + 1] = shalist
    with pytest.raises(SystemExit, match='differs from the ceai value'):
        sm.main(a)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), **SCREEN_A['cfg0_w1'])
    del w.reg['r2_1/cfg0_w1/ecapa/dev_30_src.jsonl']
    with pytest.raises(SystemExit, match='not in any --sha256 list'):
        w.main('r2_1', runs)


def test_refuses_commit_settings_snapshot_and_conv_cfg(screen_world):
    w, runs = screen_world
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), commit='2abe5842326f3dbd', **SCREEN_A['cfg0_w1'])
    with pytest.raises(SystemExit, match='describe\\(\\) commit 2abe5842326f3dbd'):
        w.main('r2_1', runs)
    for bad in ({'temp': 1.0}, {'stats_path': '/x/s.jsonl'}, {'decode': 'codes'}):
        w.run('r2_1/cfg0_w1', C('cfg0_w1'), settings=bad, **SCREEN_A['cfg0_w1'])
        with pytest.raises(SystemExit, match='settings differ'):
            w.main('r2_1', runs)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), ckpt=K['k150'], **SCREEN_A['cfg0_w1'])          # not the R2.0 snapshot
    with pytest.raises(SystemExit, match="round's snapshot"):
        w.main('r2_1', runs)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), tgt_hook=lambda t, s: t[5].update(conv_cfg_sha='0' * 16),
          **SCREEN_A['cfg0_w1'])
    with pytest.raises(SystemExit, match='conv_cfg_sha'):
        w.main('r2_1', runs)
    w.run('r2_1/cfg0_w1', C('cfg0_w1'), **SCREEN_A['cfg0_w1'])
    w.run('r2_1/extra', (0.0, 0.25, 1.0))
    with pytest.raises(SystemExit, match='not one of the eleven'):
        w.main('r2_1', dict(runs, screen=runs['screen'] + ['r2_1/extra']))
    w.run('r2_1/dup', C('cfg0_w1'), **SCREEN_A['cfg0_w1'])
    with pytest.raises(SystemExit, match='same configuration'):
        w.main('r2_1', dict(runs, screen=runs['screen'] + ['r2_1/dup']))


def test_refuses_r2_0_inputs(tmp_path, monkeypatch, capsys):
    w = World(tmp_path, monkeypatch)
    runs = r2_0_runs(w)
    w.run('r2_0/r2_0_k200_ref', sm.REF_CONF, ckpt=K['k200'], settings={'stats_path': '/x/s.jsonl'})
    with pytest.raises(SystemExit, match='settings differ'):
        w.main('r2_0', runs)                                       # stats_path: 150k only
    w.run('r2_0/r2_0_k200_ref', sm.REF_CONF, ckpt=K['k200'], commit='2abe5842326f3dbd')
    with pytest.raises(SystemExit, match='run it again with the frozen code'):
        w.main('r2_0', runs)                                       # e5's commit differs: 150k / 200k run again
    w.run('r2_0/r2_0_k200_ref', sm.REF_CONF, ckpt=K['k200'])
    with pytest.raises(SystemExit, match='reconstruction k200'):
        w.main('r2_0', dict(runs, recon=runs['recon'][:1]))
    rows = [json.loads(x) for x in (w.root / 'r2_0rec/r2_0_k200_rec/metrics.jsonl').read_text().split('\n') if x]
    w._write('r2_0rec/r2_0_k200_rec/metrics.jsonl', rows[:299])
    with pytest.raises(SystemExit, match='299 rows'):
        w.main('r2_0', runs)
    w.recon('r2_0rec/r2_0_k200_rec', 3.4)
    w.main('r2_0', runs)
    with pytest.raises(SystemExit):
        w.main('r2_0', runs, rnd='b1')
    assert 'not repeated' in capsys.readouterr().err


def test_refuses_target_copy_pair_list_and_stats(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    runs = r2_0_runs(w)
    good = [json.loads(x) for x in (tmp_path / 'tc.jsonl').read_text().split('\n') if x]
    bad = [dict(r, error='RuntimeError: x') if r['system'] == 'target_copy@src' and r['pair_id'] == w.s100[0] else r
           for r in good]
    w.write_tc(bad)
    with pytest.raises(SystemExit, match='target_copy@src has no usable row on 1'):
        w.main('r2_0', runs)
    w.write_tc([r for r in good if r['pair_id'] != w.s100[2] or r['system'] != 'target_copy@src'])
    with pytest.raises(SystemExit, match='target_copy@src has no usable row'):
        w.main('r2_0', runs)
    w.write_tc(good + good[:1])
    with pytest.raises(SystemExit, match='twice'):
        w.main('r2_0', runs)
    w.write_tc(good)
    monkeypatch.setattr(sm, 'TARGET_COPY_SRC_SHA256', {})
    with pytest.raises(SystemExit, match='not the journal'):
        w.main('r2_0', runs)
    w.write_tc(good)
    (tmp_path / 'dev.jsonl').write_text((tmp_path / 'dev.jsonl').read_text() + '\n')
    with pytest.raises(SystemExit, match='neither the ceai Dev list'):
        w.main('r2_0', runs)
    monkeypatch.setattr(sm, 'STATS_SHA256', '0' * 64)
    with pytest.raises(SystemExit, match='stats.py has sha256'):
        w.main('r2_0', runs)


def test_nl_target_copy_never_fails_on_the_p_pairs(tmp_path, monkeypatch):
    """§3.3: NL is stats.compare against target_copy@src on the pairs with P; fail_b = 0, so a worst case can come
    only from failed conversions of X (here two error rows of X)."""
    w = World(tmp_path, monkeypatch)
    dev, _, _ = sm.load_dev(str(tmp_path / 'dev.jsonl'), str(tmp_path / 'tc.jsonl'))

    def two_errors(t, s):
        for i in (0, 2):
            s[i] = {'pair_id': s[i]['pair_id'], 'system': s[i]['system'], 'set': 'dev', 'budget': 30,
                    'conv_cfg_sha': s[i]['conv_cfg_sha'], 'error': 'RuntimeError: x'}
    w.run('x/run', C('cfg0_w1'), tgt_hook=two_errors, nl=(0.01, 0.02))
    inp = sm.Inputs(str(w.root), [w.sha_file()])
    r = sm.load_run(inp, dev, 'x/run', True)
    assert r.counts['metrics/dev_30_src.jsonl'] == {'rows': 100, 'errors': 2}
    out = {g: sm.nl(dev, r, dev.sets['s100'], g) for g in sm.STRATA}
    assert all(o['fail_b'] == 0 for o in out.values())
    assert sum(o['fail_a'] for o in out.values()) == 2 and any('worst_case' in o for o in out.values())
    g = stats.stratum(w.pairs[0]['genders'])
    par = dev.par(dev.sets['s100'], g)
    d = stats.compare(r.src, dev.tc, par, dev.spk, 'sim_wavlm', mode=stats.LINKED, worst='max')
    assert out[g]['est'] == d['est'] and out[g]['se'] == d['se'] and d['n'] == len(par) - 2   # both in this stratum


# ── the review's findings ────────────────────────────────────

def fake_run(ids, src, esrc):
    return SimpleNamespace(src=src, esrc=esrc, tgt={k: {'sim_wavlm': 0.9} for k in ids},
                           etgt={k: {'sim_ecapa': 0.6} for k in ids})


def test_condition5_as_job_e3_computed_it_and_its_boundaries(tmp_path, monkeypatch):
    """§4.1 / §4.3 condition 5 'as job e3 computed it': e3's rule5.py averaged WavLM over the pairs where both WavLM @src
    rows exist and ECAPA over the pairs where both ECAPA @src rows exist, separately."""
    World(tmp_path, monkeypatch)
    dev, _, _ = sm.load_dev(str(tmp_path / 'dev.jsonl'), str(tmp_path / 'tc.jsonl'))
    ids = dev.sets['s100']
    cross = dev.stratum(ids, 'cross')
    src0, e0 = {k: {'sim_wavlm': 0.5} for k in ids}, {k: {'sim_ecapa': 0.2} for k in ids}
    # one cross pair whose WavLM @src row of c has a scoring error, and whose ECAPA Δ is -2.0; elsewhere -0.08 / -0.03
    bad = cross[0]
    src = dict({k: {'sim_wavlm': 0.42} for k in ids}, **{bad: {'error': 'RuntimeError: x'}})
    es = dict({k: {'sim_ecapa': 0.17} for k in ids}, **{bad: {'sim_ecapa': 0.2 - 2.0}})
    d = sm.delta5(dev, fake_run(ids, src, es), fake_run(ids, src0, e0), ids)
    c = d['cross']
    assert c['n_wavlm'] == len(cross) - 1 and c['n_ecapa'] == len(cross)
    assert abs(c['wavlm'] + 0.08) < 1e-12 and abs(c['ecapa'] - (-0.03 * (len(cross) - 1) - 2.0) / len(cross)) < 1e-12
    assert d['met']                    # on the joint pairs ECAPA would be -0.03 against -0.08: not met

    def met(dw, de):
        s = {k: {'sim_wavlm': 0.5 + dw} for k in ids}
        e = {k: {'sim_ecapa': 0.2 + de} for k in ids}
        return sm.delta5(dev, fake_run(ids, s, e), fake_run(ids, src0, e0), ids)['met']
    assert met(-0.08, -0.04) and met(-0.08, -0.2) and met(0.02, 0.01)            # exactly half; more; both positive
    assert not met(-0.08, -0.0399) and not met(-0.08, 0.05) and not met(0.02, -0.01)   # under half; opposite signs
    assert met(0.0, 0.0) and not met(0.0, -0.01)          # a zero WavLM Δ: only a zero ECAPA Δ has its sign


def test_recon_runs_are_bound_to_their_snapshot_by_name(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    runs = r2_0_runs(w, recon=(3.0, 4.5))                           # 200k reconstructs 1.5 points worse -> 150k
    swapped = dict(runs, recon=['k150=r2_0rec/r2_0_k200_rec', 'k200=r2_0rec/r2_0_k150_rec'])
    with pytest.raises(SystemExit, match="system 'r2_0_k200_rec', not 'r2_0_k150_rec'"):
        w.main('r2_0', swapped)
    w.main('r2_0', runs)
    assert w.state('r1/r2_0')['snapshot']['ckpt'] == K['k150']


def test_trigger_ii_alone_b1_parent_is_the_r2_choice(tmp_path, monkeypatch):
    """Eligible cells exist (no trigger i); on full Dev E3 is empty and F leaks (trigger ii): B1's parent is F's
    setting, not CFG 0 SNG 0."""
    w = World(tmp_path, monkeypatch)
    screen = [w.run(f'r2_1/{k}', C(k), **v) for k, v in SCREEN_A.items()]
    full_eff = {'cfg0_w0': dict(nl=(0.08, 0.10)), 'cfg0_w0.75_u0.5': dict(sim=0.900, nl=(0.01, 0.05)),
                'cfg1_w0.5': dict(sim=0.905, nl=(0.02, 0.04)), 'cfg1_w0.75': dict(sim=0.904, nl=(0.03, 0.045)),
                'cfg0_w1': dict(sim=0.902, nl=(0.0, 0.06))}
    full = [w.run(f'r2_full/{k}', C(k), pairset='full', **v) for k, v in full_eff.items()]
    budget = [w.run(f'r2_4/cfg1_w0.5_b{b}', C('cfg1_w0.5'), pairset='full', budget=b, sim=s)
              for b, s in ((3, 0.86), (180, 0.905))]
    runs = merge(r2_0_runs(w), {'screen': screen, 'full': full, 'budget': budget})
    for st in sm.STAGES:
        w.main(st, runs)
    s4 = w.state('r1/r2_4')
    assert s4['source'] == 'F' and s4['choice'] == 'cfg1_w0.5' and not s4['trigger_i'] and s4['trigger_ii']
    runs['b1_run'] = [w.run('b1eval/b1_nosng', sm.NOSNG_CONF, ckpt=K['b1'], run_dir=B1_DIR, pairset='full',
                            nl=(-0.01, 0.01))]
    with pytest.raises(SystemExit, match='own run_dir on full Dev at 30 s at cfg1_w0.5'):
        w.main('b1', runs)
    runs['b1_run'] = [w.run('b1eval/b1_f', C('cfg1_w0.5'), ckpt=K['b1'], run_dir=B1_DIR, pairset='full', sim=0.905,
                            nl=(-0.01, 0.01))]
    w.main('b1', runs)
    b = w.state('b1')
    assert b['parent'] == {'config': 'cfg1_w0.5', 'run': 'r2_full/cfg1_w0.5'} and b['kept']
    assert all(abs(b['deltas'][g]['est'] + 0.03) < 1e-9 for g in ('sim_src_same', 'sim_src_cross'))


def test_triggers_rule_boundaries():
    def x(same, cross, ut):
        return {'nl': {'same': {'est': same}, 'cross': {'est': cross}}, 'utmos': ut}
    assert sm.triggers_rule(False, x(0.03, 0.03, 3.52)) == {'trigger_i': False, 'trigger_ii': False,
                                                            'b1_triggered': False, 'r5': False}
    assert sm.triggers_rule(False, x(0.0301, 0.0, 3.6))['b1_triggered']
    assert sm.triggers_rule(False, x(0.0, 0.0301, 3.6))['trigger_ii']
    assert sm.triggers_rule(True, x(0.0, 0.0, 3.6))['b1_triggered']
    assert sm.triggers_rule(False, x(0.0, 0.0, 3.5199))['r5'] and f'{sm.R5_LEVEL:.2f}' == '3.52'


def test_rederivation_ignores_where_the_inputs_lie(tmp_path, monkeypatch):
    """§4.1: only a difference in the files stops the selection; the same Dev list and target_copy rows read from
    another folder give the same stage output."""
    w = World(tmp_path, monkeypatch)
    runs = r2_0_runs(w)
    w.main('r2_0', runs)
    assert set(w.state('r1/r2_0')['dev_inputs']['pairs']) == {'sha256'}
    (tmp_path / 'copy').mkdir()
    for n in ('dev.jsonl', 'tc.jsonl'):
        shutil.copy(tmp_path / n, tmp_path / 'copy' / n)
    a = w.args('r2_0', runs)
    a[a.index('--pairs') + 1] = str(tmp_path / 'copy' / 'dev.jsonl')
    a[a.index('--target_copy_src') + 1] = str(tmp_path / 'copy' / 'tc.jsonl')
    sm.main(a)


def test_r2_runs_need_a_new_tag(tmp_path, monkeypatch):
    """§4.1: every R2 job uses a TAG no earlier job used; job e5's r1k150_ns1.0 is the one run R2.0 may take (§4.2)."""
    w = World(tmp_path, monkeypatch)
    runs = merge(r2_0_runs(w), {'screen': [w.run(f'e3/{k}', C(k), **v) for k, v in SCREEN_A.items()]})
    with pytest.raises(SystemExit, match='new TAG'):
        w.main('r2_1', runs)
    r0 = r2_0_runs(w)
    r0['ref'][0] = w.run(sm.E5_REF, sm.REF_CONF, ckpt=K['k150'], settings={'stats_path': '/x/stats.jsonl'})
    w.main('r2_0', r0)
    assert w.state('r1/r2_0')['numbers']['k150']['run'] == sm.E5_REF
    r0['ref'][1] = w.run('e5/r1k200_ns1.0', sm.REF_CONF, ckpt=K['k200'])
    with pytest.raises(SystemExit, match='new TAG'):
        w.main('r2_0', r0)
    r0 = r2_0_runs(w)
    r0['recon'][0] = f'k150={w.recon("recon/r2_0_k150_rec", 3.0)}'
    with pytest.raises(SystemExit, match='new TAG'):
        w.main('r2_0', r0)


def test_operator_items_jobs_census_r1_files_contamination(tmp_path, monkeypatch):
    """§4.8: job IDs of every R2 and B1 job (also of runs refused or repeated), the census with the R2 runs apart and
    without the reconstructions, every ema_*.pt of R1, and the contamination result recorded whatever it is."""
    w = World(tmp_path, monkeypatch)
    dev, _, _ = sm.load_dev(str(tmp_path / 'dev.jsonl'), str(tmp_path / 'tc.jsonl'))
    runs = {'ref': [sm.E5_REF, 'r2_0/r2_0_k200_ref'], 'screen': ['r2_1/cfg0_w0'],
            'recon': ['k150=r2_0rec/r2_0_k150_rec', 'k200=r2_0rec/r2_0_k200_rec']}
    ctx = SimpleNamespace(dev=dev, used={sm.E5_REF, 'r2_0/r2_0_k200_ref', 'r2_1/cfg0_w0'},
                          recon={k: {'rel': x.split('=')[1]} for k, x in zip(sm.SNAP, runs['recon'])})
    d = {'r1/r2_0': {'numbers': {k: {'run': '-', 'snapshot_sha256': h(f'{sm.R1_DIR}/{v}')} for k, v in K.items()
                                 if k in sm.SNAP}}}

    def ns(**kw):
        return sm.parse_args(w.args('record', {}, *operator_files(w, runs, **kw)))
    it, ph, pr = sm.operator_items(ns(), ctx, d)
    assert ph == [] and pr == []
    assert sorted(it['jobs']) == ['r2_0', 'r2_0rec', 'r2_1'] and 'e5' not in it['jobs']
    c = it['census']
    assert c['before_r2'] == {'Dev-S100': {'30': 2, '3': 1}, 'screen 0/20': {'30': 1}}       # e5's run is pre-R2
    assert sorted(x['run'] for x in c['r2_runs']) == ['r2_0/r2_0_k200_ref', 'r2_1/cfg0_w0']
    # a refused run under its own R2 TAG: counted with the R2 runs, and its job is required
    a = ns(extra_census=[('r2_1a/cfg0_w0', 'full')])
    it, ph, pr = sm.operator_items(a, ctx, d)
    assert ph == [] and pr == [] and len(it['census']['r2_runs']) == 3 and 'r2_1a' in it['jobs']
    a.jobs = [j for j in a.jobs if not j.startswith('r2_1a=')]
    _, ph, _ = sm.operator_items(a, ctx, d)
    assert any("TAGs after the freeze in the census that no run used here and no option declares: ['r2_1a']" in x
               for x in ph)
    # R1x's Dev-S100 evals after the freeze: declared with --other_tag, counted apart
    a = ns(extra_census=[('r1xev/r1x250', 's100')])
    a.jobs = [j for j in a.jobs if not j.startswith('r1xev=')]
    a.other_tag = ['r1xev=R1x at 250k and 300k (§4.10)']
    it, ph, pr = sm.operator_items(a, ctx, d)
    assert ph == [] and pr == [] and it['census']['after_freeze_not_r2'] == {'r1xev': {'Dev-S100': {'30': 1}}}
    assert len(it['census']['r2_runs']) == 2
    # a census made before the last R2 run; a TAG given with two jobs; a pre-freeze TAG as an R2 job
    a = ns()
    cen = Path(a.census)
    cen.write_text(''.join(x for x in cen.read_text().splitlines(True) if not x.startswith('r2_1/')))
    a.jobs += ['r2_1=1', 'e3=2']
    _, _, pr = sm.operator_items(a, ctx, d)
    assert any('census lacks the runs used' in x for x in pr) and any('TAG r2_1 is given twice' in x for x in pr)
    assert any("TAGs of the jobs before the freeze as R2 or B1 jobs: ['e3']" in x for x in pr)
    # every ema_*.pt of R1
    ops = operator_files(w, runs)
    r1h = Path(ops[ops.index('--r1_hashes') + 1])
    r1h.write_text(''.join(x for x in r1h.read_text().splitlines(True) if 'ema_0010000' not in x))
    _, _, pr = sm.operator_items(sm.parse_args(w.args('record', {}, *ops)), ctx, d)
    assert pr == ["--r1_hashes lacks $MVC/runs/r1/['ema_0010000.pt'] (last.pt, args.json, stats.npz and every "
                  "ema_*.pt, §4.10)"]
    assert len(sm.R1_FILES) == 24 and sm.R1_FILES[-2:] == ('ema_0200000.pt', 'ema_last.pt')
    # the contamination test's result is recorded, not a refusal
    ops = operator_files(w, runs)
    con = Path(ops[ops.index('--contamination') + 1])
    con.write_text(json.dumps(dict(json.loads(con.read_text()), passed=False)))
    it, ph, pr = sm.operator_items(sm.parse_args(w.args('record', {}, *ops)), ctx, d)
    assert ph == [] and pr == [] and it['contamination']['passed'] is False
    con.write_text(json.dumps({'passed': False}))
    _, _, pr = sm.operator_items(sm.parse_args(w.args('record', {}, *ops)), ctx, d)
    assert any('--contamination lacks' in x for x in pr)


def test_frozen_commit_is_the_describe_of_the_frozen_files():
    """FROZEN_COMMIT is vc_adapter.describe()'s hash of the 24 mvc/*.py listed in FREEZE_mvc_v1.sha256 (§2.1)."""
    frozen = [k for k in sm.read_freeze(sm.FREEZE_PATH) if re.fullmatch(r'mvc/[^/]+\.py', k)]
    assert len(frozen) == 24
    ci = sm.code_identity()
    if ci['changed_since_freeze']:
        pytest.skip(f'mvc/*.py changed since the freeze: {sorted(ci["changed_since_freeze"])}')
    assert ci['current'] == sm.FROZEN_COMMIT == 'ba09a8efc7678e38'


def test_code_identity_lists_changes_and_checks_the_constant(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    b1c = b1_code(tmp_path, monkeypatch)
    code = sm.CODE_DIR
    ci = sm.code_identity()
    assert ci == {'current': b1c, 'frozen': w.commit, 'changed_since_freeze': {'mvc/train.py':
                                                                               sm.sha256(code / 'train.py')}}
    (code / 'extra.py').write_text('# new\n')
    os.remove(code / 'perturb.py')
    assert set(sm.code_identity()['changed_since_freeze'].items()) >= {('mvc/extra.py', sm.sha256(code / 'extra.py')),
                                                                       ('mvc/perturb.py', None)}
    monkeypatch.setattr(sm, 'CODE_DIR', sm.MVC)
    monkeypatch.setattr(sm, 'FROZEN_COMMIT', '0' * 16)
    with pytest.raises(SystemExit, match='not FROZEN_COMMIT'):
        sm.code_identity()
