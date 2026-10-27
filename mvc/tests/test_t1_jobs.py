"""Tests of the T1 jobs of mvc/ANALYSIS_PLAN_mvc_v1.md §8 (mvc/jobs/job_t1_conv.slurm, job_t1_metrics.slurm,
job_t1_ecapa.slurm, t1_lib.sh, t1_check.py):
  - t1_check.py's checks on a synthetic ceai tree (selection env, frozen files, manifests, metric rows, drift checks,
    ECAPA and guard rows), and that its env names and record lines are the ones select_mvc.py writes;
  - t1_lib.sh's helpers with a python that records its command lines (as test_jobs.py);
  - the three job scripts end to end, env.sh replaced by a fake one: python records every call, runs t1_check.py and
    benchmark/lab_rows.py for real and simulates run_mvc.py, metrics.py, score_ecapa.py and acoustic_guard.py (which
    writes its summary of means beside its output, as the real one does), whose values all start with 0.42424, a
    string no job may print (§8.3: counts only);
  - submit_t1.sh with slot_wait.sh, submit_ceai.sh and squeue replaced by fakes that record their calls."""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / 'jobs'
MVC = JOBS.parent
REPO = MVC.parent
_spec = importlib.util.spec_from_file_location('t1_check', JOBS / 't1_check.py')
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

ENV_LINE = 'source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh'
SENTINEL = '0.42424'
BASE = ('knnvc', 'freevc', 'seedvc', 'facodec')
X2SYS = ('cand_pm18_rq', 'faiss', 'crt_tau0.4', 'source_copy') + BASE
JSHA = '9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105'
PUSH_TODO = '- Push of the selection record (`mvc-selection-record-v1`): (to be filled in).'
PUSH_DONE = ('- 30 Oct 2026 09:00:00: the selection record was pushed to the public repository with the annotated tag '
             '`mvc-selection-record-v1`.')
VAL = '''def val(*k):
    return 0.42424 + int(hashlib.sha256('|'.join(map(str, k)).encode()).hexdigest()[:6], 16) * 1e-12
'''
exec(VAL)
SIM = 'import hashlib, json, os, sys\n' + VAL + r'''
def arg(name, default=None, many=False):
    a = sys.argv
    out = [a[i + 1] for i, x in enumerate(a) if x == name]
    return out if many else (out[-1] if out else default)
def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()]
def wj(p, rows, mode='w'):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
tool = os.path.basename(sys.argv[1])
if tool == 'run_mvc.py':
    s, b, name, cfg = arg('--set'), arg('--budget'), arg('--name'), json.loads(arg('--cfg'))
    d = os.path.join(arg('--out_dir'), s, b, name)
    os.makedirs(d, exist_ok=True)
    fail = set(os.environ.get('SIM_FAIL', '').split(','))
    rows = []
    for p in rj(arg('--pairs')):
        r = dict(pair_id=p['pair_id'], set=s, budget=int(b), system=name, wav=os.path.join(d, p['pair_id'] + '.wav'),
                 source_wav=p['source_path'], sim_ref_wavs=p['E']['paths'], ref_text='text', parallel_wav=p['P'],
                 source_speaker=p['source_speaker'], target_speaker=p['target_speaker'], conv_cfg_sha='c0ffee')
        if p['R'][b]['shortfall']:
            r.update(error='ValueError: reference shortfall: 75.0 s available for budget 180 s', shortfall=True)
        elif p['pair_id'] in fail:
            r.update(error='RuntimeError: boom', shortfall=False)
        else:
            open(r['wav'], 'w').close()
        rows.append(r)
    wj(os.path.join(d, 'manifest.jsonl'), rows)
    ck = {os.environ['SIM_SNAPSHOT']: os.environ['SIM_SNAPSHOT_SHA']} if cfg['mode'] == 'convert' else {}
    wj(os.path.join(d, 'runs.jsonl'), [{'event': 'start', 'conv_cfg_sha': 'c0ffee', 'conv_cfg': {'describe': {
        'commit': os.environ['SIM_COMMIT'], 'checkpoints': ck, 'settings': dict(cfg, train_step=1, train_args={})}}}],
       'a')
    sys.exit(1 if any('error' in r and not r['shortfall'] for r in rows) else 0)
elif tool == 'metrics.py':
    rows, n, ms = rj(arg('--in_jsonl')), int(arg('--max_rows', 0)), arg('--metrics').split(',')
    out = []
    for r in rows[:n] if n else rows:
        o = {k: v for k, v in r.items() if k != 'R'}
        if 'error' in r:
            o['error'] = 'FileNotFoundError: no wav'
        else:
            o['sim_wavlm'] = val(r['pair_id'], r['system'], 'w')
            if 'utmos' in ms:
                o['utmos'] = val(r['pair_id'], r['system'], 'u')
            if 'wer' in ms:
                o['wer_large_v3'], o['wer_base'] = {'hyp': 'h ' + r['pair_id']}, {'hyp': 'b ' + r['pair_id']}
        out.append(o)
    wj(arg('--out_jsonl'), out)
elif tool == 'score_ecapa.py':
    if '--smoke' in sys.argv:
        sys.exit(0)
    systems, out = arg('--systems').split(','), []
    done = {(r['pair_id'], r['system']) for r in rj(arg('--out'))} if os.path.exists(arg('--out')) else set()
    for r in [x for f in arg('--rows', many=True) for x in rj(f)]:
        if r['system'].split('@')[0] in systems and (r['pair_id'], r['system']) not in done:   # it resumes
            o = {k: r.get(k) for k in ('pair_id', 'system', 'set', 'budget')}
            if 'error' in r:
                o['error'] = 'input row: ' + r['error']
            else:
                o['sim_ecapa'] = val(r['pair_id'], r['system'], 'e')
            out.append(o)
    wj(arg('--out'), out, 'a')
    sys.exit(1 if any('error' in o for o in out) else 0)
elif tool == 'acoustic_guard.py':
    rows, out = [r for r in rj(arg('--rows')) if r.get('wav') and r.get('source_wav')], arg('--out')
    wj(out, [dict(pair_id=r['pair_id'], system=r['system'], hnr=val(r['pair_id'], r['system'], 'h')) for r in rows])
    s = json.dumps({'n': len(rows), 'hnr': val('mean')})
    open(out[:-len('.jsonl')] + '.summary.json', 'w').write(s)      # as acoustic_guard.py:104
    print(s)
else:
    sys.exit('sim: unknown tool ' + tool)
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wj(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def rj(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def pair(set_, i, src, tgt, short180=False, par=None):
    R = {b: {'budget_s': int(b), 'paths': [f'/d/{tgt}_r{b}.wav'], 'keep_s': [float(b)], 'uncut_total_s': 75.0,
             'shortfall': short180 and b == '180'} for b in ('3', '30', '180')}
    return {'pair_id': f'{set_}__{src}_{i:03d}__{tgt}', 'set': set_, 'source_speaker': src, 'target_speaker': tgt,
            'source_path': f'/d/{src}_{i:03d}.wav', 'source_text': 'text', 'P': par, 'E': {'paths': [f'/d/{tgt}_e.wav']},
            'R': R, 'wer_primary': True, 'genders': 'F2F', 'source_is_shared_text': False}


class World:
    """A synthetic ceai tree: $WORK_DIR/mvc (the code, the record, the env, FREEZE), $J (pair lists, job M's rows and
    metrics, X2's ECAPA rows, done files), $DATA_STORAGE (snapshot, ECAPA site); VCTK-2k has 4 pairs, LTC-2k 103 of
    which 101 have a 180 s reference shortfall (§1)."""

    def __init__(self, tmp):
        self.tmp, self.n_job = tmp, 100
        self.W = tmp / 'w'
        self.C, self.J, self.DS = self.W / 'mvc', self.W / 'benchmark' / 'journal', self.W / 'data_storage'
        self.M = self.DS / 'mvc'
        self.T = self.M / 'test'
        (self.C / 'jobs').mkdir(parents=True)
        for f in ('t1_lib.sh', 't1_check.py'):
            (self.C / 'jobs' / f).symlink_to(JOBS / f)
        for f in ('run_mvc.py', 'acoustic_guard.py', 'vc_adapter.py'):
            (self.C / f).write_text(f'# {f} stub\n')
        self.J.mkdir(parents=True)
        (self.J / 'lab_rows.py').symlink_to(REPO / 'benchmark' / 'lab_rows.py')
        self.vctk = [pair('vctk2k', i, s, t, par=f'/d/{t}_{i:03d}.wav' if i == 1 else None)
                     for i, (s, t) in enumerate([('p1', 'p3'), ('p1', 'p4'), ('p2', 'p3'), ('p2', 'p4')])]
        self.ltc = [pair('ltc2k', i, 'L1', 'L0', short180=True) for i in range(101)] + \
                   [pair('ltc2k', 101 + i, 'L0', 'L1') for i in range(2)]
        self.pairs = {'vctk2k': self.J / 'pairs' / 'vctk2k.jsonl', 'ltc2k': self.J / 'pairs' / 'ltc2k.jsonl'}
        wj(self.pairs['vctk2k'], self.vctk)
        wj(self.pairs['ltc2k'], self.ltc)
        (self.J / 'pairs' / 'vctk2k_source_E.json').write_text(json.dumps({'p1': ['/d/p1_e.wav'], 'p2': ['/d/p2_e.wav']}))
        self.selection()
        self.write_deviations(PUSH_DONE)
        self.journal()
        (self.DS / 'site' / 'x2_ecapa' / 'speechbrain').mkdir(parents=True)
        (tmp / 'sim.py').write_text(SIM)
        self.calls_file = tmp / 'calls.txt'
        (tmp / 'env.sh').write_text(f'''
WORK_DIR={self.W}
J={self.J}
DATA_STORAGE={self.DS}
MVC={self.M}
RAW=/raw
SLURM_CPUS_PER_TASK=2
export SIM_COMMIT={self.commit} SIM_SNAPSHOT={self.snap} SIM_SNAPSHOT_SHA={self.snap_sha}
python() {{
  local a
  a=("$@")
  [ "${{a[0]}}" = -u ] && a=("${{a[@]:1}}")
  echo "${{a[*]}}" >> {self.calls_file}
  case "${{a[0]}}" in
    -c) return 0 ;;
    */t1_check.py|*/lab_rows.py) {sys.executable} "${{a[@]}}" ;;
    *) {sys.executable} {tmp}/sim.py "${{a[@]}}" ;;
  esac
}}
cd $WORK_DIR/mvc
''')

    def selection(self):
        snap = self.M / 'runs' / 'r1' / 'ema_0200000.pt'
        snap.parent.mkdir(parents=True)
        snap.write_bytes(b'weights')
        self.snap, self.snap_sha = str(snap), sha(snap)
        v1 = dict(run_dir=str(snap.parent), ckpt='ema_0200000.pt', steps=16, cfg=1.0, mode='convert', decode='latent',
                  neg_src=0.75, neg_src_until=1.0, relax=0.15, stats_path=None)
        self.cfgs = {'mvc_v1': v1, 'mvc_v1_q32': dict(v1, decode='codes'),
                     'mvc_copy_latent': dict(run_dir=None, ckpt='ema_last.pt', steps=16, cfg=0.7, mode='copy_latent',
                                             decode='latent', neg_src=0.0, neg_src_until=1.0, relax=0.0, stats_path=None)}
        self.cfg_dir = self.C / 'MVC_SELECTION_RECORD_v1_cfg'
        self.cfg_dir.mkdir()
        for k, v in self.cfgs.items():
            (self.cfg_dir / f'{k}.json').write_text(json.dumps(v) + '\n')
        self.commit = tc.describe_commit(self.C)
        self.record = self.C / 'MVC_SELECTION_RECORD_v1.md'
        self.write_record()
        fr = [f'{"0" * 64}  mvc/run_mvc.py'] + [f'{sha(p)}  {p}' for p in (*self.pairs.values(),
                                                                         self.J / 'pairs' / 'vctk2k_source_E.json')]
        (self.C / 'FREEZE_mvc_v1.sha256').write_text('# Section 1\n' + '\n'.join(fr) + '\n')

    def write_deviations(self, push):
        (self.C / 'DEVIATIONS_mvc_v1.md').write_text('# MimiFlow-VC: deviations and freeze log (analysis plan v1)\n\n'
                                                     '## Freeze log\n\n' + (push + '\n' if push else '') +
                                                     '- Submit time of the first T1 job: (to be filled in).\n')

    def write_record(self, status='complete'):
        L = ['# MimiFlow-VC selection record v1 (ANALYSIS_PLAN_mvc_v1.md §4)', '',
             f'Status: **{status}**. Written 2026-10-30 10:00:00 by `mvc/analysis/select_mvc.py`.', '', '## mvc_v1', '',
             f'- Snapshot: `{self.snap}`, sha256 {self.snap_sha}.',
             f'- describe() commit: {self.commit} (frozen code: {self.commit}).', '', '| File | sha256 | JSON |',
             '|---|---|---|']
        L += [f'| `{k}.json` | {sha(self.cfg_dir / f"{k}.json")} | `{json.dumps(v)}` |' for k, v in self.cfgs.items()]
        self.record.write_text('\n'.join(L) + '\n')
        self.write_env()

    def env(self):
        e = {'RECORD_SHA256': sha(self.record), 'SNAPSHOT': self.snap, 'SNAPSHOT_SHA256': self.snap_sha}
        for k, v in tc.CFG_VARS.items():
            e[v], e[v + '_SHA256'] = f'{k}.json', sha(self.cfg_dir / f'{k}.json')
        e['DESCRIBE_COMMIT'] = self.commit
        return e

    def write_env(self, **over):
        e = {k: v for k, v in dict(self.env(), **over).items() if v is not None}
        (self.C / 'mvc_selection.env').write_text(''.join(f'{k}={v}\n' for k, v in e.items()))

    def jrow(self, p, s):
        return {'pair_id': p['pair_id'], 'set': p['set'], 'budget': 30, 'system': s, 'wav': f'/jw/{s}/{p["pair_id"]}.wav',
                'source_wav': p['source_path'], 'sim_ref_wavs': p['E']['paths'], 'source_speaker': p['source_speaker'],
                'target_speaker': p['target_speaker']}

    def journal(self):
        v1 = self.J / 'v1'
        t1 = [self.jrow(p, s) for p in self.vctk for s in ('front_a', 'cand_pm18_rq', 'faiss', 'crt_tau0.4',
                                                          'source_copy', 'front_b', 'mimi_copy_src', 'anc_x')]
        wj(v1 / 'm' / 't1' / 'rows' / 'vctk2k_30.jsonl', t1)
        wj(v1 / 'm' / 't1' / 'metrics' / 'vctk2k_30.jsonl',
           [dict(r, sim_wavlm=val(r['pair_id'], r['system'], 'w'), utmos=val(r['pair_id'], r['system'], 'u'),
                 wer_large_v3={'hyp': 'h ' + r['pair_id']}, wer_base={'hyp': 'b ' + r['pair_id']}) for r in t1])
        t4v = [self.jrow(p, s) for p in self.vctk for s in BASE]
        wj(v1 / 'm' / 't4' / 'rows' / 'vctk2k_30.jsonl', t4v)
        wj(v1 / 'x2' / 'ecapa' / 'vctk2k_30.jsonl',
           [{'pair_id': r['pair_id'], 'system': r['system'], 'set': 'vctk2k', 'budget': 30,
             'sim_ecapa': val(r['pair_id'], r['system'], 'e')} for r in t1 + t4v if r['system'] in X2SYS])
        wj(v1 / 'm' / 't2' / 'rows' / 'ltc2k_30.jsonl', [self.jrow(p, s) for p in self.ltc for s in ('cand_pm18_rq',
                                                                                                    'faiss')])
        wj(v1 / 'm' / 't4' / 'rows' / 'ltc2k_30.jsonl', [self.jrow(p, s) for p in self.ltc for s in BASE])
        for f in ('m/t1/m.done', 'm/t2/m.done', 'm/t4/m.done', 'x2/x2.done'):
            (v1 / f).parent.mkdir(parents=True, exist_ok=True)
            (v1 / f).write_text(f'job 4653 2026-10-20 06:45:22 selection sha256 {JSHA}\n')

    # ── running ──
    def run(self, job, **env):
        text = (JOBS / job).read_text()
        assert text.count(ENV_LINE) == 1
        script = self.tmp / f'run_{job}.sh'
        script.write_text(text.replace(ENV_LINE, f'source {self.tmp}/env.sh'))
        self.n_job += 1
        e = {k: v for k, v in os.environ.items() if k not in ('PART', 'ONLY', 'ECAPA_RESCORE', 'SIM_FAIL', 'PYTHONPATH',
                                                              'SLURM_RESTART_COUNT')}
        e.update(env, SLURM_JOB_ID=str(self.n_job))
        p = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=e)
        return p.returncode, p.stdout + p.stderr

    def calls(self, tool=None):
        lines = self.calls_file.read_text().splitlines() if self.calls_file.exists() else []
        self.calls_file.unlink(missing_ok=True)
        return [x for x in lines if tool is None or x.split()[0].endswith(tool)]

    def convert_all(self, fail=''):
        """The ten conversions through the simulator, the cfg JSONs installed (for the unit tests)."""
        shutil.copytree(self.cfg_dir, self.T / 'cfg')
        e = dict(os.environ, SIM_COMMIT=self.commit, SIM_SNAPSHOT=self.snap, SIM_SNAPSHOT_SHA=self.snap_sha,
                 SIM_FAIL=fail)
        for s in tc.SETS:
            for name, b in tc.COMBOS:
                subprocess.run([sys.executable, str(self.tmp / 'sim.py'), 'run_mvc.py', '--pairs', str(self.pairs[s]),
                                '--set', s, '--budget', str(b), '--name', name, '--cfg',
                                json.dumps(self.cfgs[name]), '--out_dir', str(self.T / 'conv')], env=e)

    def manifest(self, s, b, name):
        return self.T / 'conv' / s / str(b) / name / 'manifest.jsonl'


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


def problems(fn, *args):
    P = tc.Problems()
    fn(*args, P)
    return P


def sel(w, cfg_dir=None, install=''):
    return problems(tc.check_selection, str(w.C / 'mvc_selection.env'), str(w.record), str(cfg_dir or w.cfg_dir),
                    str(w.C), install)


def conv(w):
    return problems(tc.check_conv, str(w.T / 'conv'), {k: str(v) for k, v in w.pairs.items()},
                    str(w.C / 'mvc_selection.env'), str(w.T / 'cfg'))


# ── t1_check.py: selection ───────────────────────────────────

def test_selection_strings_are_those_select_mvc_writes():
    src = (MVC / 'analysis' / 'select_mvc.py').read_text()
    for needle in ("f'RECORD_SHA256={sha256(a.out)}'", """f'SNAPSHOT={m["snapshot"]["path"]}'""",
                   """f'SNAPSHOT_SHA256={m["snapshot"]["sha256"]}'""",
                   "(('mvc_v1', 'CFG_MVC_V1'), ('mvc_v1_q32', 'CFG_MVC_V1_Q32'),", "('mvc_copy_latent', 'CFG_MVC_COPY_LATENT')",
                   """env += [f'{name}={c[k]["file"]}', f'{name}_SHA256={c[k]["sha256"]}']""",
                   """env.append(f'DESCRIBE_COMMIT={m["describe_commit"]}')""", "'file': f'{k}.json'",
                   """f'Status: **{rec["status"]}**.""", "rec['status'] = 'complete'",
                   """f'- Snapshot: `{m["snapshot"]["path"]}`, sha256 {m["snapshot"]["sha256"]}.'""",
                   """f'- describe() commit: {m["describe_commit"]} (""", """f'| `{x["file"]}` | {x["sha256"]} | """,
                   "default=str(MVC / 'mvc_selection.env')", "default=str(MVC / 'MVC_SELECTION_RECORD_v1.md')",
                   "default=str(MVC / 'MVC_SELECTION_RECORD_v1_cfg')"):
        assert needle in src, needle
    assert sorted(tc.ENV_KEYS) == sorted(['RECORD_SHA256', 'SNAPSHOT', 'SNAPSHOT_SHA256', 'DESCRIBE_COMMIT'] +
                                         [v + x for v in tc.CFG_VARS.values() for x in ('', '_SHA256')])
    lib = (JOBS / 't1_lib.sh').read_text()
    for x in ('SEL_ENV=$C/mvc_selection.env', 'RECORD=$C/MVC_SELECTION_RECORD_v1.md',
              'RECORD_CFG=$C/MVC_SELECTION_RECORD_v1_cfg'):
        assert x in lib


def test_describe_commit_is_vc_adapters():
    sys.path.insert(0, str(MVC))
    import vc_adapter
    ad = vc_adapter.MimiVC()
    ad.ckpt_path, ad.step, ad.train_args = None, None, {}
    assert ad.describe()['commit'] == tc.describe_commit(MVC)


def test_selection_passes_and_installs(w):
    dst = w.T / 'cfg'
    assert sel(w, install=str(dst)) == []
    assert all((dst / f'{k}.json').read_bytes() == (w.cfg_dir / f'{k}.json').read_bytes() for k in w.cfgs)
    assert sel(w, cfg_dir=dst, install=str(dst)) == []
    (dst / 'mvc_v1.json').write_text('{}\n')
    assert any('mvc_v1.json has sha256' in p for p in sel(w, cfg_dir=dst))
    assert any('exists and differs' in p for p in sel(w, install=str(dst)))


@pytest.mark.parametrize('over, msg', [
    (dict(DESCRIBE_COMMIT=None), 'lacks'), (dict(EXTRA='1'), 'which select_mvc.py does not write'),
    (dict(SNAPSHOT_SHA256='abc'), 'is not a sha256'), (dict(CFG_MVC_V1='v1.json'), 'not mvc_v1.json'),
    (dict(DESCRIBE_COMMIT='0' * 16), 'not DESCRIBE_COMMIT'), (dict(RECORD_SHA256='0' * 64), 'not RECORD_SHA256'),
    (dict(SNAPSHOT_SHA256='0' * 64), 'not SNAPSHOT_SHA256'), (dict(CFG_MVC_V1_Q32_SHA256='0' * 64), 'not CFG_MVC_V1_Q32'),
    (dict(SNAPSHOT='/x/ema.pt'), 'missing')])
def test_selection_refuses_a_missing_or_wrong_value(tmp_path, over, msg):
    w = World(tmp_path)
    w.write_env(**over)
    assert any(msg in p for p in sel(w)), sel(w)


def test_selection_refuses_a_changed_file(w):
    (w.C / 'run_mvc.py').write_text('# changed\n')
    assert any('mvc/*.py differ from the code of the record' in p for p in sel(w))
    w = World(w.tmp / 'b')
    w.write_record(status='DRAFT: not the selection record')
    assert any('Status: **complete**.' in p for p in sel(w))
    w = World(w.tmp / 'c')
    Path(w.snap).write_bytes(b'other weights')
    assert any('not SNAPSHOT_SHA256' in p for p in sel(w))
    w = World(w.tmp / 'd')
    w.cfgs['mvc_v1_q32'] = dict(w.cfgs['mvc_v1'], decode='codes', relax=0.0)
    (w.cfg_dir / 'mvc_v1_q32.json').write_text(json.dumps(w.cfgs['mvc_v1_q32']) + '\n')
    w.write_record()
    assert sel(w) == ['mvc_v1_q32.json is not mvc_v1.json with "decode": "codes" (§2.2)']
    w = World(w.tmp / 'e')
    w.cfgs['mvc_v1']['ckpt'] = w.cfgs['mvc_v1_q32']['ckpt'] = 'ema_0150000.pt'
    for k in ('mvc_v1', 'mvc_v1_q32'):
        (w.cfg_dir / f'{k}.json').write_text(json.dumps(w.cfgs[k]) + '\n')
    w.write_record()
    assert any('not SNAPSHOT' in p for p in sel(w))


def test_frozen_checks_the_ceai_lines_only(w):
    fr = w.C / 'FREEZE_mvc_v1.sha256'
    assert problems(tc.check_frozen, str(fr)) == []
    w.pairs['ltc2k'].write_text(w.pairs['ltc2k'].read_text() + '\n')
    assert problems(tc.check_frozen, str(fr)) == [f'{w.pairs["ltc2k"]} differs from its frozen sha256 '
                                                  f'{fr.read_text().split()[-4]}']


# ── t1_check.py: conversions, metrics, drift, ECAPA, guards ──

def test_conv_passes_and_counts(w, capsys):
    w.convert_all()
    assert conv(w) == []
    out = capsys.readouterr().out
    assert 'conv ltc2k 180 s mvc_v1: 103 rows for 103 pairs, 101 shortfall, 0 failed' in out
    assert 'conv vctk2k 3 s mvc_v1: 4 rows for 4 pairs, 0 shortfall' in out and out.count('  conv ') == 10


def test_conv_counts_failed_conversions_once_every_pair_has_a_row(w, capsys):
    w.convert_all(fail=w.vctk[2]['pair_id'])                          # §5.4: errors that remain are failures
    assert conv(w) == []
    out = capsys.readouterr().out
    assert 'conv vctk2k 30 s mvc_v1: 4 rows for 4 pairs, 0 shortfall, 1 failed' in out
    assert 'conv: 5 failed conversions in all' in out


def test_pushed(w, capsys):
    dev = str(w.C / 'DEVIATIONS_mvc_v1.md')
    assert problems(tc.check_pushed, dev) == [] and 'pushed: - 30 Oct 2026' in capsys.readouterr().out
    w.write_deviations(PUSH_TODO)
    assert any('still "(to be filled in)"' in p for p in problems(tc.check_pushed, dev))
    w.write_deviations(None)
    assert any('has no line on the push' in p for p in problems(tc.check_pushed, dev))
    os.remove(dev)
    assert any('missing' in p for p in problems(tc.check_pushed, dev))


def test_conv_refuses_wrong_rows(w):
    w.convert_all()
    m = w.manifest('ltc2k', 180, 'mvc_v1')
    rows = rj(m)
    rows[0].pop('error'), rows[0].pop('shortfall')                     # a shortfall pair converted
    rows[-1].update(error='ValueError: reference shortfall: 1 s', shortfall=True)
    wj(m, rows)
    assert any('shortfall rows on 101 pairs, 100 of them among' in p for p in conv(w))
    wj(m, rows[1:])
    assert any('rows on 102 of the 103 pairs' in p for p in conv(w))


@pytest.mark.parametrize('edit, msg', [
    (lambda d: d.update(commit='f' * 16), 'describe() commit ffff'),
    (lambda d: d['settings'].update(neg_src=1.0), "differ from mvc_v1.json in ['neg_src']"),
    (lambda d: d.update(checkpoints={'/x/ema.pt': '0' * 64}), 'converted with the checkpoints')])
def test_conv_refuses_another_code_setting_or_snapshot(tmp_path, edit, msg):
    w = World(tmp_path)
    w.convert_all()
    runs = w.manifest('vctk2k', 30, 'mvc_v1').parent / 'runs.jsonl'
    ev = rj(runs)
    edit(ev[0]['conv_cfg']['describe'])
    wj(runs, ev)
    assert any(msg in p for p in conv(w)), conv(w)


def test_conv_refuses_rows_of_another_conv_cfg_and_missing_wavs(w):
    w.convert_all()
    m = w.manifest('vctk2k', 30, 'mvc_v1_q32')
    rows = rj(m)
    rows[0]['conv_cfg_sha'] = 'other'
    wj(m, rows)
    os.remove(rows[1]['wav'])
    P = conv(w)
    assert any('1 rows not of the conv_cfg c0ffee' in p for p in P) and any('WAV is missing' in p for p in P)


def metrics_dir(w, part, g, systems, edit=None):
    """lab_rows.py (real) and the simulated metrics.py on the converted manifests of one part."""
    s, b = g.rsplit('_', 1)
    o = w.T / 'm' / part
    ins = [x for n in systems for x in ('--manifest', str(w.manifest(s, b, n)))]
    se = ['--source_e', str(w.J / 'pairs' / 'vctk2k_source_E.json')] if s == 'vctk2k' else []
    subprocess.run([sys.executable, str(REPO / 'benchmark' / 'lab_rows.py'), '--pairs', str(w.pairs[s]), *se, *ins,
                    '--out', str(o / 'rows' / f'{g}.jsonl')], check=True, capture_output=True)
    for k, ms in (('', 'mcd,wer,sim,secs,utmos'), ('_src', 'sim,secs')):
        subprocess.run([sys.executable, str(w.tmp / 'sim.py'), 'metrics.py', '--in_jsonl', str(o / 'rows' / f'{g}{k}.jsonl'),
                        '--out_jsonl', str(o / 'metrics' / f'{g}{k}.jsonl'), '--metrics', ms], check=True)
    return o


def test_metrics_check(w, capsys):
    w.convert_all(fail=w.ltc[101]['pair_id'])
    o = metrics_dir(w, 't3', 'ltc2k_180', ['mvc_v1'])
    assert problems(tc.check_metrics, str(o), 'ltc2k_180', str(w.pairs['ltc2k']), ['mvc_v1']) == []
    assert 'ltc2k_180: 103 metric rows (1 systems x 103 pairs), 101 shortfall rows, 1 failed conversions, 0 scoring ' \
           'errors; @src 1 metric rows, 0 errors' in capsys.readouterr().out
    o = metrics_dir(w, 't1', 'vctk2k_30', tc.CFG_VARS)
    f = o / 'metrics' / 'vctk2k_30.jsonl'
    rows = rj(f)
    rows[0]['error'] = 'CUDA error'
    wj(f, rows)
    P = problems(tc.check_metrics, str(o), 'vctk2k_30', str(w.pairs['vctk2k']), list(tc.CFG_VARS))
    assert len(P) == 1 and '1 scoring errors' in P[0]
    wj(f, rows[1:])
    P = problems(tc.check_metrics, str(o), 'vctk2k_30', str(w.pairs['vctk2k']), list(tc.CFG_VARS))
    assert any('1 rows without a metric row' in p for p in P)
    src = o / 'metrics' / 'vctk2k_30_src.jsonl'
    wj(src, rj(src)[1:])
    P = problems(tc.check_metrics, str(o), 'vctk2k_30', str(w.pairs['vctk2k']), ['mvc_v1'])
    assert any("rows of ['mvc_copy_latent', 'mvc_v1', 'mvc_v1_q32'], not of ['mvc_v1']" in p for p in P)
    assert any('1 @src rows without a metric row' in p for p in P)
    assert not any(SENTINEL in p for p in P)


def drift_files(w, edits=()):
    new = w.tmp / 'new.jsonl'
    subprocess.run([sys.executable, str(w.tmp / 'sim.py'), 'metrics.py', '--in_jsonl',
                    str(w.J / 'v1/m/t1/rows/vctk2k_30.jsonl'), '--out_jsonl', str(new), '--max_rows', '32',
                    '--metrics', 'mcd,wer,sim,secs,utmos'], check=True)
    rows = rj(new)
    for i, k, v in edits:
        if k in ('wer_large_v3', 'wer_base'):
            rows[i][k] = {'hyp': v}
        else:
            rows[i][k] = v
    wj(new, rows)
    return problems(tc.drift_m, str(new), str(w.J / 'v1/m/t1/metrics/vctk2k_30.jsonl'), 32, 31)


def test_drift_m(w, capsys):
    assert drift_files(w) == []
    assert 'sim_wavlm differs on 0, utmos on 0; hypotheses identical on 32' in capsys.readouterr().out
    assert drift_files(w, [(3, 'wer_base', 'other')]) == []                         # 1 of 32 may differ (§8.3)
    assert drift_files(w, [(3, 'wer_base', 'x'), (4, 'wer_large_v3', 'y')]) == [
        'identical large-v3 and base hypotheses on 30 rows; §8.3 requires at least 31 of 32']
    assert drift_files(w, [(0, 'sim_wavlm', val('x'))]) == [
        'sim_wavlm differs on 1 rows and utmos on 0; §8.3 requires max |Δ| = 0']
    assert drift_files(w, [(0, 'utmos', float('nan'))]) == [
        'sim_wavlm differs on 0 rows and utmos on 1; §8.3 requires max |Δ| = 0']
    assert any('1 re-scored rows have an error' in p for p in drift_files(w, [(5, 'error', 'oom')]))
    assert SENTINEL not in capsys.readouterr().out


def test_ecapa_drift(w, capsys):
    rows_in, new = w.tmp / 'in.jsonl', w.tmp / 'e.jsonl'
    x2 = str(w.J / 'v1/x2/ecapa/vctk2k_30.jsonl')
    assert problems(tc.ecapa_drift_rows, x2, [str(w.J / 'v1/m/t1/rows/vctk2k_30.jsonl'),
                                              str(w.J / 'v1/m/t4/rows/vctk2k_30.jsonl')], 32, str(rows_in)) == []
    assert len(rj(rows_in)) == 32
    assert any('have no input row' in p for p in problems(tc.ecapa_drift_rows, x2, [
        str(w.J / 'v1/m/t1/rows/vctk2k_30.jsonl')], 32, str(w.tmp / 'x.jsonl')))
    subprocess.run([sys.executable, str(w.tmp / 'sim.py'), 'score_ecapa.py', '--systems', ','.join(X2SYS), '--rows',
                    str(rows_in), '--out', str(new)], check=True)
    assert problems(tc.drift_ecapa, str(new), x2, 32, 0.001) == []
    rows = rj(new)
    rows[0]['sim_ecapa'] += 0.0005
    rows[1]['sim_ecapa'] += 0.002
    wj(new, rows)
    assert problems(tc.drift_ecapa, str(new), x2, 32, 0.001) == ['|Δ sim_ecapa| > 0.001 on 1 rows (§8.3)']
    assert SENTINEL not in capsys.readouterr().out


def test_ecapa_and_guard_checks(w):
    w.convert_all()
    o = metrics_dir(w, 't1', 'vctk2k_30', tc.CFG_VARS)
    e = w.T / 'x2' / 'ecapa'
    for k in ('', '_src'):
        subprocess.run([sys.executable, str(w.tmp / 'sim.py'), 'score_ecapa.py', '--systems', ','.join(tc.CFG_VARS),
                        '--rows', str(o / 'rows' / f'vctk2k_30{k}.jsonl'), '--out', str(e / f'vctk2k_30{k}.jsonl')])
    args = (str(o / 'rows'), str(e), 'vctk2k_30', str(w.pairs['vctk2k']), list(tc.CFG_VARS))
    assert problems(tc.check_ecapa, *args) == []
    rows = rj(e / 'vctk2k_30.jsonl')
    rows[0] = {k: rows[0][k] for k in ('pair_id', 'system', 'set', 'budget')} | {'error': 'RuntimeError: x'}
    wj(e / 'vctk2k_30.jsonl', rows[:-1])
    P = problems(tc.check_ecapa, *args)
    assert any('1 rows not scored' in p for p in P) and any('1 error rows on converted rows' in p for p in P)
    gi = w.T / 'guard' / 'input' / 'vctk2k_30.jsonl'
    src = [f'{o}/rows/vctk2k_30.jsonl=mvc_v1,mvc_v1_q32', f'{w.J}/v1/m/t1/rows/vctk2k_30.jsonl=cand_pm18_rq',
           f'{w.J}/v1/m/t4/rows/vctk2k_30.jsonl={",".join(BASE)}']
    assert problems(tc.guard_rows, str(gi), src) == [] and len(rj(gi)) == 28
    assert any('holds no row of' in p for p in problems(tc.guard_rows, str(w.tmp / 'g.jsonl'), src + [
        f'{w.J}/v1/m/t1/rows/vctk2k_30.jsonl=seedvc']))
    g = w.T / 'guard' / 'vctk2k_30.jsonl'
    subprocess.run([sys.executable, str(w.tmp / 'sim.py'), 'acoustic_guard.py', '--rows', str(gi), '--out', str(g)],
                   capture_output=True, check=True)
    gsys = ['mvc_v1', 'mvc_v1_q32', 'cand_pm18_rq', *BASE]
    summ = w.T / 'guard' / 'vctk2k_30.summary.json'
    assert summ.exists()                                               # what acoustic_guard.py writes beside --out
    P = problems(tc.check_guard, str(g), str(w.pairs['vctk2k']), gsys)
    assert len(P) == 1 and str(summ) in P[0] and 'summaries of means' in P[0] and SENTINEL not in P[0]
    summ.unlink()
    assert problems(tc.check_guard, str(g), str(w.pairs['vctk2k']), gsys) == []
    wj(g, rj(g)[1:])
    assert problems(tc.check_guard, str(g), str(w.pairs['vctk2k']), gsys) == [
        f'{g} mvc_v1: rows on 3 of the 4 pairs, 0 outside']


# ── t1_lib.sh helpers, python recorded (as test_jobs.py) ─────

def run_lib(tmp_path, call, fail=''):
    log = tmp_path / 'calls.txt'
    log.unlink(missing_ok=True)
    (tmp_path / 'm' / 'test' / 'cfg').mkdir(parents=True, exist_ok=True)
    for k in tc.CFG_VARS:
        (tmp_path / 'm' / 'test' / 'cfg' / f'{k}.json').write_text(json.dumps({'mode': k, 'cfg': 1.0}) + '\n')
    script = f'''
WORK_DIR=/w J=/j DATA_STORAGE=/ds MVC={tmp_path}/m SLURM_JOB_ID=7 SLURM_CPUS_PER_TASK=8
python() {{
  local a o= prev=
  echo "${{PYTHONPATH:-}}|$*" >> {log}
  if [[ "$*" == *acoustic_guard.py* ]]; then    # writes --out and, beside it, its summary of means (acoustic_guard.py:104)
    for a in "$@"; do [ "$prev" = --out ] && o=$a; prev=$a; done
    mkdir -p "$(dirname "$o")" && echo '{{}}' > "$o" && echo "{{\\"hnr\\": {SENTINEL}1}}" > "${{o%.jsonl}}.summary.json"
    echo "{{\\"hnr\\": {SENTINEL}1}}"
  fi
  [[ -n "{fail}" && "$*" == *"{fail}"* ]] && return 1
  return 0
}}
source {JOBS / 't1_lib.sh'}
{call}
echo "FAILED=$FAILED"
'''
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=True, env=env).stdout
    calls = [x.split('|', 1) for x in log.read_text().splitlines()] if log.exists() else []
    return [c for _, c in calls], [p for p, _ in calls], out


def test_convert_runs_every_system_and_budget_with_the_installed_cfg(tmp_path):
    calls, _, out = run_lib(tmp_path, 't1_convert ltc2k', fail='--budget 180')
    assert [(c.split('--name ')[1].split()[0], c.split('--budget ')[1].split()[0]) for c in calls] == [
        ('mvc_v1', '30'), ('mvc_v1_q32', '30'), ('mvc_copy_latent', '30'), ('mvc_v1', '3'), ('mvc_v1', '180')]
    assert all(c.startswith('-u /w/mvc/run_mvc.py --journal /j --pairs /j/pairs/ltc2k.jsonl --set ltc2k') for c in calls)
    assert all(c.endswith(f'--out_dir {tmp_path}/m/test/conv --device cuda') for c in calls)
    assert '--cfg {"mode": "mvc_v1_q32", "cfg": 1.0} --out_dir' in calls[1]
    assert out.strip().splitlines()[-1] == 'FAILED= conv_ltc2k_180_mvc_v1'


def test_score_runs_lab_rows_metrics_and_checks(tmp_path):
    calls, _, out = run_lib(tmp_path, 't1_score /o vctk2k 30 mvc_v1,mvc_v1_q32,mvc_copy_latent')
    T = f'{tmp_path}/m/test'
    assert calls[0] == (f'-u /j/lab_rows.py --pairs /j/pairs/vctk2k.jsonl --source_e /j/pairs/vctk2k_source_E.json '
                        f'--manifest {T}/conv/vctk2k/30/mvc_v1/manifest.jsonl --manifest '
                        f'{T}/conv/vctk2k/30/mvc_v1_q32/manifest.jsonl --manifest '
                        f'{T}/conv/vctk2k/30/mvc_copy_latent/manifest.jsonl --out /o/rows/vctk2k_30.jsonl')
    assert calls[1] == ('-u /j/metrics.py --in_jsonl /o/rows/vctk2k_30.jsonl --out_jsonl /o/metrics/vctk2k_30.jsonl '
                        '--metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model '
                        '/j/models/wavlm-base-plus-sv --device cuda')
    assert calls[2] == ('-u /j/metrics.py --in_jsonl /o/rows/vctk2k_30_src.jsonl --out_jsonl '
                        '/o/metrics/vctk2k_30_src.jsonl --metrics sim,secs --wavlm_model /j/models/wavlm-base-plus-sv '
                        '--device cuda')
    assert calls[3] == ('-u /w/mvc/jobs/t1_check.py metrics --dir /o --g vctk2k_30 --pairs /j/pairs/vctk2k.jsonl '
                        '--systems mvc_v1,mvc_v1_q32,mvc_copy_latent')
    assert len(calls) == 4 and out.strip().splitlines()[-1] == 'FAILED='
    calls, _, _ = run_lib(tmp_path, 't1_score /o ltc2k 180 mvc_v1')
    assert '--source_e' not in calls[0] and calls[0].endswith(
        f'--manifest {T}/conv/ltc2k/180/mvc_v1/manifest.jsonl --out /o/rows/ltc2k_180.jsonl')


def test_score_stops_when_lab_rows_fails_and_records_check_failures(tmp_path):
    calls, _, out = run_lib(tmp_path, 't1_score /o ltc2k 30 mvc_v1', fail='lab_rows')
    assert len(calls) == 1 and out.strip().splitlines()[-1] == 'FAILED= rows_ltc2k_30'
    calls, _, out = run_lib(tmp_path, 't1_score /o ltc2k 30 mvc_v1', fail='t1_check.py metrics')
    assert len(calls) == 4 and out.strip().splitlines()[-1] == 'FAILED= checks_ltc2k_30'


def test_drift_m_rescores_32_journal_rows_into_a_new_file(tmp_path):
    calls, _, out = run_lib(tmp_path, 't1_drift_m t2; echo rc=$?')
    new = f'{tmp_path}/m/test/drift/m_t2_job7_r0_vctk2k_30.jsonl'
    assert calls == [f'-u /j/metrics.py --in_jsonl /j/v1/m/t1/rows/vctk2k_30.jsonl --out_jsonl {new} --max_rows 32 '
                     '--metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model '
                     '/j/models/wavlm-base-plus-sv --device cuda',
                     f'-u /w/mvc/jobs/t1_check.py drift_m --new {new} --ref /j/v1/m/t1/metrics/vctk2k_30.jsonl --n 32 '
                     '--min_same 31']
    assert 'rc=0' in out
    calls, _, out = run_lib(tmp_path, 't1_drift_m t2; echo rc=$?', fail='metrics.py')
    assert len(calls) == 1 and 'rc=1' in out
    Path(new).write_text('')
    calls, _, out = run_lib(tmp_path, 't1_drift_m t2; echo rc=$?')
    assert calls == [] and 'rc=1' in out and 'exists' in out
    calls, _, out = run_lib(tmp_path, 'SLURM_RESTART_COUNT=1; t1_drift_m t2; echo rc=$?')   # a requeue: a new file
    assert 'rc=0' in out and calls[0].split('--out_jsonl ')[1].split()[0] == new.replace('_r0_', '_r1_')


def test_drift_ecapa_rescores_32_x2_rows_into_a_new_file(tmp_path):
    calls, paths, out = run_lib(tmp_path, 't1_drift_ecapa t1; echo rc=$?')
    T = f'{tmp_path}/m/test'
    rows, new = f'{T}/drift/ecapa_t1_job7_r0_rows.jsonl', f'{T}/drift/ecapa_t1_job7_r0_vctk2k_30.jsonl'
    assert calls == [f'-u /w/mvc/jobs/t1_check.py ecapa_drift_rows --x2 /j/v1/x2/ecapa/vctk2k_30.jsonl --rows '
                     f'/j/v1/m/t1/rows/vctk2k_30.jsonl --rows /j/v1/m/t4/rows/vctk2k_30.jsonl --n 32 --out {rows}',
                     f'-u /j/score_ecapa.py --model_dir /ds/models/spkrec-ecapa-voxceleb --savedir {T}/x2/ecapa_model '
                     f'--device cuda --systems faiss,crt_tau0.4,cand_pm18_rq,knnvc,freevc,seedvc,facodec,source_copy,'
                     f'target_copy --rows {rows} --out {new}',
                     f'-u /w/mvc/jobs/t1_check.py drift_ecapa --new {new} --ref /j/v1/x2/ecapa/vctk2k_30.jsonl --n 32 '
                     f'--tol 0.001']
    assert paths == ['', '/ds/site/x2_ecapa', ''] and 'rc=0' in out
    Path(new).write_text('')
    calls, _, out = run_lib(tmp_path, 't1_drift_ecapa t1; echo rc=$?')
    assert calls == [] and 'rc=1' in out and 'exists' in out
    calls, _, out = run_lib(tmp_path, 'SLURM_RESTART_COUNT=2; ECAPA_SAVEDIR=/sd; t1_drift_ecapa t1; echo rc=$?')
    assert 'rc=0' in out and f'--out {new.replace("_r0_", "_r2_")}' in calls[1] and '--savedir /sd ' in calls[1]


def test_ecapa_uses_the_x2_site_and_the_guard_keeps_no_summary(tmp_path):
    T = tmp_path / 'm' / 'test'
    (T / 'guard' / 'tmp').mkdir(parents=True)
    (T / 'guard' / 'tmp' / 'ltc2k_30.summary.json').write_text('{}')   # a stopped run's
    calls, paths, out = run_lib(tmp_path, 't1_ecapa ltc2k t2; t1_guard ltc2k t2')
    assert calls[0] == (f'-u /j/score_ecapa.py --model_dir /ds/models/spkrec-ecapa-voxceleb --savedir {T}/x2/ecapa_model '
                        f'--device cuda --systems mvc_v1,mvc_v1_q32,mvc_copy_latent --rows {T}/m/t2/rows/ltc2k_30.jsonl '
                        f'--out {T}/x2/ecapa/ltc2k_30.jsonl')
    assert calls[1].endswith(f'--rows {T}/m/t2/rows/ltc2k_30_src.jsonl --out {T}/x2/ecapa/ltc2k_30_src.jsonl')
    assert paths[:2] == ['/ds/site/x2_ecapa', '/ds/site/x2_ecapa'] and paths[2:] == [''] * (len(paths) - 2)
    assert calls[3] == (f'-u /w/mvc/jobs/t1_check.py guard_rows --out {T}/guard/input/ltc2k_30.jsonl --src '
                        f'{T}/m/t2/rows/ltc2k_30.jsonl=mvc_v1,mvc_v1_q32 --src /j/v1/m/t2/rows/ltc2k_30.jsonl=cand_pm18_rq '
                        f'--src /j/v1/m/t4/rows/ltc2k_30.jsonl=knnvc,freevc,seedvc,facodec')
    assert calls[4] == (f'-u /w/mvc/acoustic_guard.py --rows {T}/guard/input/ltc2k_30.jsonl --out '
                        f'{T}/guard/tmp/ltc2k_30.jsonl --workers 8')
    assert calls[5] == (f'-u /w/mvc/jobs/t1_check.py guard --guard {T}/guard/ltc2k_30.jsonl --pairs /j/pairs/ltc2k.jsonl '
                        f'--systems mvc_v1,mvc_v1_q32,cand_pm18_rq,knnvc,freevc,seedvc,facodec')
    assert SENTINEL not in out and out.strip().splitlines()[-1] == 'FAILED='
    assert [str(p.relative_to(T)) for p in (T / 'guard').rglob('*') if p.is_file()] == ['guard/ltc2k_30.jsonl']
    calls, _, out = run_lib(tmp_path, 't1_guard ltc2k t2', fail='acoustic_guard.py')   # failed: no output kept
    assert out.strip().splitlines()[-1] == 'FAILED= guard_ltc2k' and not list((T / 'guard').rglob('*.json*'))


def test_t1_jobs_follow_the_ceai_rules():
    for job, limit in (('job_t1_conv.slurm', 24), ('job_t1_metrics.slurm', 12), ('job_t1_ecapa.slurm', 4)):
        s = (JOBS / job).read_text()
        for line in ('#SBATCH --account=uit', '#SBATCH --partition=normal', '#SBATCH --gres=gpu:2g.35gb:1',
                     f'#SBATCH --time={limit:02d}:00:00', ENV_LINE, 'source $WORK_DIR/mvc/jobs/t1_lib.sh'):
            assert line in s, (job, line)
        assert re.search(r'#SBATCH --job-name=mvc_t1_', s) and not re.search(r'^[^#]*summarize\.py', s, re.M)


# ── the jobs end to end ──────────────────────────────────────

def kinds(calls):
    """The steps of a job's python calls, in order."""
    out = []
    for x in calls:
        tool, rest = os.path.basename(x.split()[0]), x.split()[1:]
        if tool == 'score_ecapa.py':
            out.append('ecapa_smoke' if '--smoke' in rest else 'score_ecapa')
        elif tool == 'metrics.py':
            out.append('drift_metrics' if '--max_rows' in rest else 'metrics')
        elif tool == 't1_check.py':
            out.append(rest[0])
        elif tool in ('lab_rows.py', 'run_mvc.py', 'acoustic_guard.py'):
            out.append(tool[:-3])
    return out


def test_jobs_end_to_end(w):
    rc, out = w.run('job_t1_metrics.slurm', PART='t1')                       # no conversion yet
    assert rc == 1 and 'nothing scored' in out and not w.calls('metrics.py')
    rc, out = w.run('job_t1_conv.slurm')
    assert rc == 0, out
    assert kinds(w.calls())[:3] == ['pushed', 'selection', 'frozen']
    assert SENTINEL not in out and 'conv: 0 failed conversions in all' in out
    assert (w.T / 'conv' / 'conv.done').read_text().strip().endswith(f'selection sha256 {sha(w.record)}')
    assert all((w.T / 'cfg' / f'{k}.json').read_bytes() == (w.cfg_dir / f'{k}.json').read_bytes() for k in w.cfgs)
    assert 'conv ltc2k 180 s mvc_v1: 103 rows for 103 pairs, 101 shortfall, 0 failed' in out
    rc, out = w.run('job_t1_ecapa.slurm')                                    # metrics not done
    assert rc == 1 and 'm/t1/m.done missing' in out
    for part in ('t1', 't2', 't3'):
        rc, out = w.run('job_t1_metrics.slurm', PART=part)
        assert rc == 0, out
        assert SENTINEL not in out and 'drift (WavLM-SV, UTMOS, Whisper): 32 rows' in out and '-> pass' in out
        assert 'drift (ECAPA): 32 rows' in out
        assert (w.T / 'm' / part / 'm.done').read_text().startswith('job ')
        c = w.calls()
        k = [x for x in kinds(c) if x not in ('selection', 'frozen', 'conv')]
        assert k[:7] == ['ecapa_smoke', 'ecapa_drift_rows', 'score_ecapa', 'drift_ecapa', 'drift_metrics', 'drift_m',
                         'lab_rows'], k                               # §8.1, §8.3: both drift checks before any row
        assert 'score_ecapa' not in k[7:] and 'ecapa_smoke' not in k[7:]
        assert all(f'--savedir {w.T}/drift/ecapa_model_{part} ' in x for x in c if 'score_ecapa.py' in x)
        assert sorted(p.name for p in (w.T / 'drift').glob(f'*_{part}_job{w.n_job}_r0_*')) == [
            f'ecapa_{part}_job{w.n_job}_r0_rows.jsonl', f'ecapa_{part}_job{w.n_job}_r0_vctk2k_30.jsonl',
            f'm_{part}_job{w.n_job}_r0_vctk2k_30.jsonl']
    assert sorted(p.name for p in (w.T / 'm' / 't3' / 'metrics').iterdir()) == sorted(
        f'{s}_{b}{k}.jsonl' for s in ('vctk2k', 'ltc2k') for b in (3, 180) for k in ('', '_src'))
    rc, out = w.run('job_t1_ecapa.slurm')
    assert rc == 0, out
    assert SENTINEL not in out and 'drift (ECAPA): 32 rows' in out
    assert kinds(w.calls())[:6] == ['selection', 'frozen', 'ecapa_smoke', 'ecapa_drift_rows', 'score_ecapa',
                                    'drift_ecapa']
    for f in ('x2/x2.done', 'guard/guard.done'):
        assert (w.T / f).read_text().strip().endswith(f'selection sha256 {sha(w.record)}')
    assert sorted(p.name for p in (w.T / 'x2' / 'ecapa').iterdir()) == [
        'ltc2k_30.jsonl', 'ltc2k_30_src.jsonl', 'vctk2k_30.jsonl', 'vctk2k_30_src.jsonl']
    assert sorted(p.name for p in (w.T / 'guard').glob('*_30*.jsonl')) == ['ltc2k_30.jsonl', 'vctk2k_30.jsonl']
    assert not list(w.T.rglob('*.summary.json')) and not (w.T / 'guard' / 'tmp').exists()   # §8.3: deleted unread
    rc, out = w.run('job_t1_conv.slurm')                                     # every pair attempted: no conversion
    assert rc == 1 and 'conv.done exists' in out and not w.calls('run_mvc.py')
    (w.T / 'conv' / 'conv.done').unlink()
    rc, out = w.run('job_t1_conv.slurm')                                     # metrics exist: no conversion
    assert rc == 1 and '/m exists' in out and not w.calls('run_mvc.py')


def test_conv_job_refusals_and_failed_conversions(w):
    w.write_deviations(PUSH_TODO)                                           # §8.3: the record not pushed yet
    rc, out = w.run('job_t1_conv.slurm')
    assert rc == 1 and 'still "(to be filled in)"' in out and 'nothing converted' in out and not w.calls('run_mvc.py')
    w.write_deviations(PUSH_DONE)
    w.write_env(DESCRIBE_COMMIT='0' * 16)
    rc, out = w.run('job_t1_conv.slurm')
    assert rc == 1 and 'not DESCRIBE_COMMIT' in out and not w.calls('run_mvc.py')
    w.write_env()
    rc, out = w.run('job_t1_conv.slurm', ONLY='vctk')
    assert rc == 1 and 'neither vctk2k nor ltc2k' in out
    fail = w.vctk[0]['pair_id']
    rc, out = w.run('job_t1_conv.slurm', ONLY='vctk2k', SIM_FAIL=fail)     # one set: no conv.done yet
    assert rc == 1 and len(w.calls('run_mvc.py')) == 5 and not (w.T / 'conv' / 'conv.done').exists()
    assert 'conv vctk2k 30 s mvc_v1: 4 rows for 4 pairs, 0 shortfall, 1 failed' in out and 'has a row per pair: no' in out
    rc, out = w.run('job_t1_conv.slurm', ONLY='ltc2k')                       # the second set: every pair has a row
    assert rc == 0 and 'conv: 5 failed conversions in all' in out and (w.T / 'conv' / 'conv.done').exists()
    assert len(w.calls('run_mvc.py')) == 5
    rc, out = w.run('job_t1_conv.slurm', ONLY='vctk2k')                      # §5.4: no retry once every pair has a row
    assert rc == 1 and 'conv.done exists' in out and not w.calls('run_mvc.py')
    rc, out = w.run('job_t1_metrics.slurm', PART='t1')                       # failed conversions are scored as failures
    assert rc == 0 and 'vctk2k_30: 12 metric rows (3 systems x 4 pairs), 0 shortfall rows, 3 failed conversions' in out
    rc, out = w.run('job_t1_metrics.slurm', PART='t4')
    assert rc == 1 and 'is not t1, t2 or t3' in out


def test_metrics_job_scores_nothing_when_the_drift_check_fails(w):
    assert w.run('job_t1_conv.slurm')[0] == 0
    f = w.J / 'v1/m/t1/metrics/vctk2k_30.jsonl'
    rows = rj(f)
    rows[7]['utmos'] += 1e-9
    wj(f, rows)
    w.calls()
    rc, out = w.run('job_t1_metrics.slurm', PART='t2')
    assert rc == 1 and 'utmos on 1' in out and 'nothing scored' in out and not w.calls('lab_rows.py')
    assert not (w.T / 'm' / 't2' / 'm.done').exists()
    (w.T / 'conv' / 'conv.done').write_text(f'job 1 2026-10-30 10:00:00 selection sha256 {"0" * 64}\n')
    rc, out = w.run('job_t1_metrics.slurm', PART='t2')
    assert rc == 1 and 'written under another record' in out


def test_metrics_job_scores_nothing_when_the_ecapa_drift_check_fails(w):
    assert w.run('job_t1_conv.slurm')[0] == 0
    f = w.J / 'v1/x2/ecapa/vctk2k_30.jsonl'
    rows = rj(f)
    rows[3]['sim_ecapa'] += 0.002
    wj(f, rows)
    w.calls()
    rc, out = w.run('job_t1_metrics.slurm', PART='t1')
    assert rc == 1 and '|Δ sim_ecapa| > 0.001 on 1' in out and 'nothing scored' in out
    k = kinds(w.calls())
    assert k[-1] == 'drift_ecapa' and not {'drift_metrics', 'lab_rows', 'metrics'} & set(k)
    assert not (w.T / 'm' / 't1' / 'm.done').exists()
    (w.J / 'v1/x2/x2.done').unlink()                                        # the drift checks read X2's and job M's rows
    rc, out = w.run('job_t1_metrics.slurm', PART='t1')
    assert rc == 1 and 'x2/x2.done missing or not written under the journal selection' in out
    assert not {'ecapa_smoke', 'score_ecapa'} & set(kinds(w.calls()))


def test_ecapa_job_rescores_after_an_error_row_on_request_only(w):
    assert w.run('job_t1_conv.slurm')[0] == 0
    for part in ('t1', 't2', 't3'):
        assert w.run('job_t1_metrics.slurm', PART=part)[0] == 0
    e = w.T / 'x2' / 'ecapa' / 'vctk2k_30.jsonl'
    r = rj(w.T / 'm' / 't1' / 'rows' / 'vctk2k_30.jsonl')[0]
    wj(e, [{k: r[k] for k in ('pair_id', 'system', 'set', 'budget')} | {'error': 'RuntimeError: CUDA error'}])
    rc, out = w.run('job_t1_ecapa.slurm')                                    # score_ecapa.py keeps the error row
    assert rc == 1 and '1 error rows on converted rows' in out and 'ECAPA_RESCORE=1' in out
    assert not (w.T / 'x2' / 'x2.done').exists() and (w.T / 'guard' / 'guard.done').exists()
    rc, out = w.run('job_t1_ecapa.slurm')                                    # resubmitted unchanged: still refused
    assert rc == 1 and '1 error rows on converted rows' in out
    rc, out = w.run('job_t1_ecapa.slurm', ECAPA_RESCORE='1')
    assert rc == 0, out
    assert out.count('unread (log it as an operational change)') == 4
    assert (w.T / 'x2' / 'x2.done').exists() and not any('error' in x for x in rj(e)) and SENTINEL not in out
    rc, out = w.run('job_t1_ecapa.slurm', ECAPA_RESCORE='1')
    assert rc == 1 and 'x2.done exists: nothing deleted' in out and len(rj(e)) == 12
    rc, out = w.run('job_t1_ecapa.slurm', ECAPA_RESCORE='yes')
    assert rc == 1 and 'is not 1' in out


# ── submit_t1.sh ─────────────────────────────────────────────

class Queue:
    """submit_t1.sh with its ceai prefix moved under tmp: slot_wait.sh and submit_ceai.sh are fakes that log their
    calls; submit_ceai.sh hands out job ids and queues the job; squeue lists the queue for `polls` calls (failing first
    `fails` times), then empties it and writes the done file of every job not named in no_done."""

    def __init__(self, tmp, polls=0, fails=0, no_done=()):
        self.tmp, self.R = tmp, tmp / 'root'
        self.T = self.R / 'data_storage' / 'mvc' / 'test'
        for d in ('mvc/jobs', 'benchmark/journal', 'data_storage/mvc/test', 'bin'):
            (self.R / d).mkdir(parents=True, exist_ok=True)
        self.log = tmp / 'log.txt'
        (tmp / 'squeue_n').write_text(str(polls))
        (tmp / 'squeue_fail').write_text(str(fails))
        (tmp / 'no_done').write_text(''.join(f'{self.T}/{x}\n' for x in no_done))
        (self.R / 'mvc/jobs/slot_wait.sh').write_text(f'echo slot_wait >> {self.log}\n')
        (self.R / 'benchmark/journal/submit_ceai.sh').write_text(f'''
n=$(( $(cat {tmp}/jid 2>/dev/null || echo 100) + 1 )); echo $n > {tmp}/jid
echo "submit $(basename $1) ${{*:2}} in $(basename $PWD)" >> {self.log}
echo "waiting to submit $(basename $1): the 1 h gap"
case "$*" in
  *ONLY=ltc2k*) d={self.T}/conv/conv.done ;;
  *PART=*) d={self.T}/m/${{2##*PART=}}/m.done ;;
  *) d= ;;
esac
[ -n "$d" ] && ! grep -qx "$d" {tmp}/no_done && echo "$d" >> {tmp}/pending
echo $n >> {tmp}/queue
printf 'submitted: 2026-10-30 10:00:00\\t%s\\t2026-10-30T10:00:00\\t%s\\t%s\\n' $n "$1" "${{*:2}}"
''')
        sq = self.R / 'bin' / 'squeue'
        sq.write_text(f'''#!/bin/bash
echo "squeue $*" >> {self.log}
f=$(cat {tmp}/squeue_fail)
[ $f -gt 0 ] && {{ echo $((f - 1)) > {tmp}/squeue_fail; exit 1; }}
n=$(cat {tmp}/squeue_n)
if [ $n -gt 0 ]; then echo $((n - 1)) > {tmp}/squeue_n; cat {tmp}/queue; exit 0; fi
[ -f {tmp}/pending ] && while read d; do mkdir -p $(dirname $d); echo "job 1 done" > $d; done < {tmp}/pending
rm -f {tmp}/pending {tmp}/queue
''')
        sq.chmod(0o755)

    def run(self, *args, **env):
        script = self.tmp / 'submit_t1.sh'
        script.write_text((JOBS / 'submit_t1.sh').read_text().replace('/datastore/ceai/crt/mimi_vc', str(self.R)))
        e = {k: v for k, v in os.environ.items() if k != 'ECAPA_RESCORE'}
        e.update(env, POLL='0', PATH=f'{self.R}/bin:{os.environ["PATH"]}')
        p = subprocess.run(['bash', str(script), *args], capture_output=True, text=True, env=e)
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        self.log.unlink(missing_ok=True)
        return p.returncode, p.stdout + p.stderr, lines


def test_submit_t1_runs_every_stage_in_order_through_slot_wait(tmp_path):
    q = Queue(tmp_path, polls=2, fails=1)
    rc, out, log = q.run()
    assert rc == 0, out
    sub = [x for x in log if not x.startswith('squeue')]
    assert sub == ['slot_wait', 'submit job_t1_conv.slurm --export=ALL,ONLY=vctk2k in journal',
                   'slot_wait', 'submit job_t1_conv.slurm --export=ALL,ONLY=ltc2k in journal'] + [
        x for p in ('t1', 't2', 't3') for x in ('slot_wait', f'submit job_t1_metrics.slurm --export=ALL,PART={p} in journal')
    ] + ['slot_wait', 'submit job_t1_ecapa.slurm --export=ALL in journal']
    first_metrics = log.index('submit job_t1_metrics.slurm --export=ALL,PART=t1 in journal')
    assert [x for x in log[:first_metrics] if x.startswith('squeue')] == ['squeue -h -u ceai -o %i'] * 4   # 1 failed, 2 busy
    ecapa = log.index('submit job_t1_ecapa.slurm --export=ALL in journal')
    assert log[ecapa - 2].startswith('squeue') and 'is job 101' in out and 'is job 106' in out
    assert 'waiting to submit job_t1_conv.slurm' in out                  # submit_ceai.sh's lines are shown


def test_submit_t1_stops_when_a_stage_leaves_no_done_file(tmp_path):
    q = Queue(tmp_path, no_done=['conv/conv.done'])
    rc, out, log = q.run()
    assert rc == 1 and 'have left the queue without' in out and 'conv.done' in out
    assert not [x for x in log if 'metrics' in x or 'ecapa' in x]
    q = Queue(tmp_path / 'b', no_done=['m/t2/m.done'])
    rc, out, log = q.run()
    assert rc == 1 and 'm/t2/m.done' in out and not [x for x in log if 'job_t1_ecapa' in x]
    rc, out, log = q.run('metrics', 't2')                                   # resume: only t2, then ECAPA
    assert rc == 1 and 'm/t2/m.done' in out                                 # (still no m.done in this fake)
    (tmp_path / 'b' / 'no_done').write_text('')
    rc, out, log = q.run('metrics', 't2')
    assert rc == 0 and [x for x in log if x.startswith('submit')] == [
        'submit job_t1_metrics.slurm --export=ALL,PART=t2 in journal', 'submit job_t1_ecapa.slurm --export=ALL in journal']


def test_submit_t1_refusals_and_ecapa_rescore(tmp_path):
    q = Queue(tmp_path)
    for args, msg in ((('bogus',), 'is not conv, metrics or ecapa'), (('conv', 't1'), 'is not a set'),
                      (('metrics', 'ltc2k'), 'is not a set')):
        rc, out, log = q.run(*args)
        assert rc == 1 and msg in out and log == []
    (q.T / 'm' / 't1').mkdir(parents=True)
    rc, out, log = q.run()
    assert rc == 1 and '/m exists: no conversion' in out and log == []
    (q.T / 'm' / 't3').mkdir(parents=True)
    (q.T / 'm' / 't3' / 'm.done').write_text('job 1\n')
    rc, out, log = q.run('metrics')                                         # t3 is done: no part submitted
    assert rc == 1 and 'm/t3/m.done exists' in out and log == []
    rc, out, log = q.run('ecapa', ECAPA_RESCORE='1')
    assert rc == 0 and log == ['slot_wait', 'submit job_t1_ecapa.slurm --export=ALL,ECAPA_RESCORE=1 in journal']
