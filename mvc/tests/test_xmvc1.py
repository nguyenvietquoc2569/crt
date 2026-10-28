"""Tests of X-mvc-1 (mvc/ANALYSIS_PLAN_mvc_v1.md §4.10; DEVIATIONS_mvc_v1.md operational note 4):
  - mvc/jobs/job_xmvc1.slurm end to end on test_t1_jobs.World's synthetic ceai tree (env.sh replaced; python records
    every call, runs t1_check.py, xmvc1_check.py and lab_rows.py for real and simulates run_mvc.py, metrics.py,
    score_ecapa.py and acoustic_guard.py, whose values all start with 0.42424, a string no job may print): it writes
    only under $X, refuses before T1 is done, builds the cfg as mvc_v1.json with only run_dir and ckpt changed, runs
    both drift checks before any row is scored, runs the T1 scorer commands with its own ECAPA savedir, converts
    again only until every pair has been attempted (conv.done), lists its result files by sha256 and prints counts
    only;
  - xmvc1_check.py's cfg, snapshot (with R1's frozen stats.npz), conv and files checks;
  - submit_xmvc1.sh's order (slot_wait.sh before each submit_ceai.sh) and refusals, with test_t1_jobs.py's fakes;
  - mvc/analysis/dnsmos_xmvc1.py with test_dnsmos_t1.py's stand-in for dnsmos.py, and its refusals;
  - mvc/analysis/analyze_xmvc1.py on synthetic rows: levels, paired differences and NL recomputed independently, no p
    value, Holm or decision in the output, the header text and the refusals (the copy checked against the job's
    lists); where the T1 rows are on this Mac, main() on mvc_v1's rows relabelled mvc_r1x300 gives every Δ = 0 and
    mvc_v1's levels and NL as TEST_RESULTS_mvc_v1.json reports them."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MVC_DIR = HERE.parent
REPO = MVC_DIR.parent
for p in (HERE, MVC_DIR / 'analysis', MVC_DIR / 'jobs'):
    sys.path.insert(0, str(p))
import analyze_mvc as am  # noqa: E402
import analyze_tests as at  # noqa: E402
import analyze_xmvc1 as ax  # noqa: E402
import dnsmos_t1 as dt  # noqa: E402
import dnsmos_xmvc1 as dx  # noqa: E402
import stats  # noqa: E402
import test_analyze_mvc as tam  # noqa: E402
import test_dnsmos_t1 as tdt  # noqa: E402
import test_t1_jobs as t1t  # noqa: E402
import xmvc1_check as xc  # noqa: E402

SENTINEL, sha, wj, rj = t1t.SENTINEL, t1t.sha, t1t.wj, t1t.rj
X, MVC = ax.X, am.MVC
JOB = 'job_xmvc1.slurm'
T1_DONE = ('m/t1/m.done', 'm/t2/m.done', 'm/t3/m.done', 'x2/x2.done', 'guard/guard.done')
SIM_CK = "    ck = {os.environ['SIM_SNAPSHOT']: os.environ['SIM_SNAPSHOT_SHA']} if cfg['mode'] == 'convert' else {}\n"
SIM_CK_X = ("    snap = os.path.join(cfg['run_dir'], cfg['ckpt']) if cfg['mode'] == 'convert' else None\n"
            "    ck = {snap: hashlib.sha256(open(snap, 'rb').read()).hexdigest()} if snap else {}\n")
SIM_PAIRS = "    for p in rj(arg('--pairs')):\n"          # SIM_STOP=n: run_mvc.py stops after n pairs
SIM_PAIRS_X = "    for p in rj(arg('--pairs'))[:int(os.environ.get('SIM_STOP') or 10 ** 9)]:\n"
ROUTE = '*/t1_check.py|*/lab_rows.py)'
# the job's python calls in order (test_t1_jobs.kinds; xmvc1_check.py as x_<subcommand>)
STEPS = ['selection', 'frozen', 'x_cfg', 'x_snapshot', 'run_mvc', 'x_conv', 'ecapa_smoke', 'ecapa_drift_rows',
         'score_ecapa', 'drift_ecapa', 'drift_metrics', 'drift_m', 'lab_rows', 'metrics', 'metrics', 'metrics',
         'score_ecapa', 'score_ecapa', 'ecapa', 'guard_rows', 'acoustic_guard', 'guard', 'x_files']
LISTED = ('cfg/{x}.json', 'snapshot.sha256', 'conv/{s}.conv.done', 'm/{p}/rows/{s}_30.jsonl', 'm/{p}/rows/{s}_30_src.jsonl',
          'm/{p}/metrics/{s}_30.jsonl', 'm/{p}/metrics/{s}_30_src.jsonl', 'x2/ecapa/{s}_30.jsonl',
          'x2/ecapa/{s}_30_src.jsonl', 'guard/{s}_30.jsonl')


class XWorld(t1t.World):
    """test_t1_jobs.World with T1 done under the record (the files named in t1_done), R1x's 300k snapshot and
    stats.npz (R1's, listed in FREEZE_mvc_v1.sha256), xmvc1_check.py run for real and a run_mvc.py simulator that names
    the snapshot of its cfg, as describe() does, and stops after SIM_STOP pairs when set."""

    def __init__(self, tmp, t1_done=T1_DONE):
        super().__init__(tmp)
        (self.C / 'jobs' / 'xmvc1_check.py').symlink_to(t1t.JOBS / 'xmvc1_check.py')
        self.X = self.M / 'xmvc1'
        self.snapx = self.M / 'runs' / 'r1x' / 'ema_0300000.pt'
        self.snapx.parent.mkdir(parents=True)
        self.snapx.write_bytes(b'r1x weights at 300k')
        r1_stats = Path(self.snap).parent / 'stats.npz'
        for p in (r1_stats, self.snapx.parent / 'stats.npz'):
            p.write_bytes(b'mean and std of R1')
        fr = self.C / 'FREEZE_mvc_v1.sha256'
        fr.write_text(fr.read_text() + f'{sha(r1_stats)}  {r1_stats}\n')
        sim = (tmp / 'sim.py').read_text()
        assert sim.count(SIM_CK) == 1 and sim.count(SIM_PAIRS) == 1
        (tmp / 'sim.py').write_text(sim.replace(SIM_CK, SIM_CK_X).replace(SIM_PAIRS, SIM_PAIRS_X))
        env = (tmp / 'env.sh').read_text()
        assert ROUTE in env
        (tmp / 'env.sh').write_text(env.replace(ROUTE, '*/t1_check.py|*/xmvc1_check.py|*/lab_rows.py)'))
        for f in t1_done:
            p = self.T / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f'job 6841 2026-10-28 16:36:06 selection sha256 {sha(self.record)}\n')

    def runx(self, **env):
        return self.run(JOB, PYTHONDONTWRITEBYTECODE='1', **env)

    def tree(self):
        """sha256 of every file outside $X."""
        return {str(p): sha(p) for p in self.W.rglob('*') if p.is_file() and self.X not in p.parents}


def kinds(calls):
    out = []
    for x in calls:
        if os.path.basename(x.split()[0]) == 'xmvc1_check.py':
            out.append('x_' + x.split()[1])
        else:
            out += t1t.kinds([x])
    return out


def problems(fn, *args):
    P = xc.tc.Problems()
    fn(*args, P)
    return P


# ── the job end to end ───────────────────────────────────────

def tool_calls(w, s, part, job):
    """The job's run_mvc.py, lab_rows.py, metrics.py, score_ecapa.py and acoustic_guard.py command lines on set s:
    T1's commands (test_t1_jobs.py pins them for T1: t1_score, t1_drift_m, t1_drift_ecapa, t1_score_ecapa, t1_guard)
    with $X's paths, and score_ecapa.py with the set's own savedir under $X."""
    J, x, wavlm = w.J, w.X, f'{w.J}/models/wavlm-base-plus-sv'
    ecapa = (f'{J}/score_ecapa.py --model_dir {w.DS}/models/spkrec-ecapa-voxceleb --savedir {x}/x2/ecapa_model_{s} '
             f'--device cuda')
    drift = f'{x}/drift/{{}}_x_{s}_job{job}_r0_{{}}.jsonl'
    se = f' --source_e {J}/pairs/vctk2k_source_E.json' if s == 'vctk2k' else ''
    cfg = (x / 'cfg' / f'{X}.json').read_text().strip()
    return [f'{w.C}/run_mvc.py --journal {J} --pairs {J}/pairs/{s}.jsonl --set {s} --budget 30 --name {X} --cfg {cfg} '
            f'--out_dir {x}/conv --device cuda',
            f'{ecapa} --smoke',
            f'{ecapa} --systems faiss,crt_tau0.4,cand_pm18_rq,knnvc,freevc,seedvc,facodec,source_copy,target_copy '
            f'--rows {drift.format("ecapa", "rows")} --out {drift.format("ecapa", "vctk2k_30")}',
            f'{J}/metrics.py --in_jsonl {J}/v1/m/t1/rows/vctk2k_30.jsonl --out_jsonl {drift.format("m", "vctk2k_30")} '
            f'--max_rows 32 --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model {wavlm} '
            f'--device cuda',
            f'{J}/lab_rows.py --pairs {J}/pairs/{s}.jsonl{se} --manifest {x}/conv/{s}/30/{X}/manifest.jsonl --out '
            f'{x}/m/{part}/rows/{s}_30.jsonl',
            f'{J}/metrics.py --in_jsonl {x}/m/{part}/rows/{s}_30.jsonl --out_jsonl {x}/m/{part}/metrics/{s}_30.jsonl '
            f'--metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model {wavlm} --device cuda',
            f'{J}/metrics.py --in_jsonl {x}/m/{part}/rows/{s}_30_src.jsonl --out_jsonl '
            f'{x}/m/{part}/metrics/{s}_30_src.jsonl --metrics sim,secs --wavlm_model {wavlm} --device cuda',
            f'{ecapa} --systems {X} --rows {x}/m/{part}/rows/{s}_30.jsonl --out {x}/x2/ecapa/{s}_30.jsonl',
            f'{ecapa} --systems {X} --rows {x}/m/{part}/rows/{s}_30_src.jsonl --out {x}/x2/ecapa/{s}_30_src.jsonl',
            f'{w.C}/acoustic_guard.py --rows {x}/guard/input/{s}_30.jsonl --out {x}/tmp/guard_{s}/{s}_30.jsonl '
            f'--workers 2']


TOOLS = ('run_mvc.py', 'lab_rows.py', 'metrics.py', 'score_ecapa.py', 'acoustic_guard.py')


def test_job_end_to_end_writes_only_under_x(tmp_path):
    w = XWorld(tmp_path)
    before = w.tree()
    for s, part, n in (('vctk2k', 't1', 4), ('ltc2k', 't2', 103)):
        rc, out = w.runx(ONLY=s)
        assert rc == 0, out
        assert SENTINEL not in out                                         # counts only (§8.3)
        calls = w.calls()
        assert kinds(calls) == STEPS                                       # both drift checks before any row
        # T1's scorer commands; every score_ecapa.py call (smoke, drift, target, @src) with the set's savedir in $X
        assert [c for c in calls if os.path.basename(c.split()[0]) in TOOLS] == tool_calls(w, s, part, w.n_job)
        assert f'conv {s} 30 s {X}: {n} rows for {n} pairs, 0 failed' in out
        assert f'{s}_30: {n} metric rows (1 systems x {n} pairs), 0 shortfall rows, 0 failed conversions' in out
        assert f'ECAPA {s}_30: {n} rows (1 systems x {n} pairs), 0 errors' in out
        assert 'drift (ECAPA): 32 rows' in out and 'drift (WavLM-SV, UTMOS, Whisper): 32 rows' in out
        tail = f'snapshot sha256 {sha(w.snapx)} selection sha256 {sha(w.record)}'
        for f in (f'{s}.done', f'conv/{s}.conv.done'):
            assert re.fullmatch(rf'job {w.n_job} \d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d {tail}\n', (w.X / f).read_text()), f
        for f in (f'm/{part}/rows/{s}_30.jsonl', f'm/{part}/rows/{s}_30_src.jsonl', f'm/{part}/metrics/{s}_30.jsonl',
                  f'm/{part}/metrics/{s}_30_src.jsonl', f'x2/ecapa/{s}_30.jsonl', f'x2/ecapa/{s}_30_src.jsonl',
                  f'guard/{s}_30.jsonl'):
            assert {r['system'].split('@')[0] for r in rj(w.X / f)} == {X}, f
        runs = rj(w.X / 'conv' / s / '30' / X / 'runs.jsonl')
        assert runs[0]['conv_cfg']['describe']['checkpoints'] == {str(w.snapx): sha(w.snapx)}
        # the copy list: the set's result files, relative to $X, with their sha256
        want = [f.format(x=X, s=s, p=part) for f in LISTED]
        assert (w.X / f'{s}.sha256').read_text() == ''.join(f'{sha(w.X / f)}  {f}\n' for f in want)
    assert w.tree() == before                                              # nothing written outside $X
    assert not any((w.T / d).exists() for d in ('cfg', 'conv', 'drift', 'xmvc1'))
    assert not list(w.X.rglob('*.summary.json')) and not list((w.X / 'tmp').iterdir())   # §8.3: deleted unread
    assert sorted(p.name for p in (w.X / 'drift').iterdir()) == sorted(
        f'{k}_x_{s}_job{j}_r0_{f}' for s, j in (('vctk2k', 101), ('ltc2k', 102))
        for k, f in (('ecapa', 'rows.jsonl'), ('ecapa', 'vctk2k_30.jsonl'), ('m', 'vctk2k_30.jsonl')))
    cfg, v1 = json.loads((w.X / 'cfg' / f'{X}.json').read_text()), w.cfgs['mvc_v1']
    assert {k for k in set(cfg) | set(v1) if cfg.get(k) != v1.get(k)} == {'run_dir', 'ckpt'}
    assert (cfg['run_dir'], cfg['ckpt']) == (str(w.M / 'runs' / 'r1x'), 'ema_0300000.pt')
    st = w.snapx.parent / 'stats.npz'
    assert (w.X / 'snapshot.sha256').read_text() == f'{sha(w.snapx)}  {w.snapx}\n{sha(st)}  {st}\n'


@pytest.mark.parametrize('missing', T1_DONE)
def test_job_refuses_before_t1_is_done(tmp_path, missing):
    w = XWorld(tmp_path, t1_done=[f for f in T1_DONE if f != missing])
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and f'{missing} missing or written under another record' in out and 'nothing run' in out
    assert not w.calls('run_mvc.py') and not w.X.exists()


def test_job_other_refusals(tmp_path):
    w = XWorld(tmp_path)
    (w.T / 'x2' / 'x2.done').write_text(f'job 1 2026-10-28 16:16:44 selection sha256 {"0" * 64}\n')
    rc, out = w.runx(ONLY='ltc2k')                                         # T1 done under another record
    assert rc == 1 and 'x2/x2.done missing or written under another record' in out
    w = XWorld(tmp_path / 'b')
    for env, msg in ((dict(ONLY='vctk'), 'is not vctk2k or ltc2k'), (dict(), 'is not vctk2k or ltc2k'),
                     (dict(ONLY='vctk2k', ECAPA_RESCORE='yes'), 'is not 1')):
        rc, out = w.runx(**env)
        assert rc == 1 and msg in out and not w.calls('run_mvc.py')
    (w.C / 'run_mvc.py').write_text('# changed\n')                         # not the code of the record
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and 'mvc/*.py differ from the code of the record' in out and 'nothing run' in out
    w = XWorld(tmp_path / 'c')
    w.snapx.unlink()
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and 'ema_0300000.pt is missing' in out and not w.X.exists()
    w.snapx.write_bytes(b'r1x weights at 300k')
    (w.J / 'v1' / 'm' / 't4' / 'm.done').unlink()                          # the drift checks read job M's rows
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and 'm/t4/m.done missing or not written under the journal selection' in out and not w.X.exists()
    w = XWorld(tmp_path / 'd')
    (w.snapx.parent / 'stats.npz').write_bytes(b'mean and std of R1x')    # not R1's normalisation statistics
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and "not that of R1's" in out and 'nothing converted' in out and not w.calls('run_mvc.py')
    assert not (w.X / 'snapshot.sha256').exists()


def test_job_failed_conversion_resubmission_and_done(tmp_path):
    w = XWorld(tmp_path)
    rc, out = w.runx(ONLY='vctk2k', SIM_FAIL=w.vctk[2]['pair_id'])
    assert rc == 0, out                                                    # §5.4: counted, not retried
    assert f'conv vctk2k 30 s {X}: 4 rows for 4 pairs, 1 failed' in out
    assert 'vctk2k_30: 4 metric rows (1 systems x 4 pairs), 0 shortfall rows, 1 failed conversions' in out
    assert 'ECAPA vctk2k_30: 4 rows (1 systems x 4 pairs), 1 errors (1 failed conversions)' in out
    assert f'guard {w.X}/guard/vctk2k_30.jsonl: 4 rows (1 systems x 4 pairs)' in out and SENTINEL not in out
    w.calls()
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and 'vctk2k.done exists' in out and not w.calls('run_mvc.py')
    (w.X / 'vctk2k.done').unlink()
    rc, out = w.runx(ONLY='vctk2k')                                        # every pair attempted: no conversion
    assert rc == 0 and 'conv.done exists: every pair has been attempted, no more conversion' in out
    assert 'run_mvc' not in kinds(w.calls()) and f'conv vctk2k 30 s {X}: 4 rows for 4 pairs, 1 failed' in out
    done = w.X / 'conv' / 'vctk2k.conv.done'
    good = done.read_text()
    (w.X / 'vctk2k.done').unlink()
    done.write_text(good.replace(sha(w.snapx), '0' * 64))                  # written under another snapshot
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and 'written under another snapshot or record' in out and not w.calls('run_mvc.py')
    done.unlink()                                                          # a metric exists: never convert again
    rc, out = w.runx(ONLY='vctk2k')
    assert rc == 1 and f'{w.X}/m/t1 exists without {done}' in out and not w.calls('run_mvc.py')


def test_job_converts_again_only_until_every_pair_is_attempted(tmp_path):
    """§5.4: a stopped conversion is resumed by resubmitting; once every pair has a row (conv.done), a resubmission
    (here after a failed drift check, before any metric) never converts again and the failure stays counted."""
    w = XWorld(tmp_path)
    done = w.X / 'conv' / 'ltc2k.conv.done'
    rc, out = w.runx(ONLY='ltc2k', SIM_STOP='50')                          # run_mvc.py stopped after 50 pairs
    assert rc == 1 and 'rows on 50 of the 103 pairs' in out and 'nothing scored' in out
    assert kinds(w.calls())[-2:] == ['run_mvc', 'x_conv'] and not done.exists() and not (w.X / 'm').exists()
    f = w.J / 'v1/m/t1/metrics/vctk2k_30.jsonl'
    good = f.read_text()
    rows = rj(f)
    rows[7]['utmos'] += 1e-9
    wj(f, rows)
    rc, out = w.runx(ONLY='ltc2k', SIM_FAIL=w.ltc[102]['pair_id'])        # resubmitted: converts the rest, 1 fails
    assert rc == 1 and f'conv ltc2k 30 s {X}: 103 rows for 103 pairs, 1 failed' in out
    assert 'utmos on 1' in out and 'nothing scored' in out and SENTINEL not in out
    k = kinds(w.calls())
    assert k.count('run_mvc') == 1 and k[-1] == 'drift_m' and 'lab_rows' not in k and not (w.X / 'm').exists()
    marker = done.read_text()
    assert marker.endswith(f'snapshot sha256 {sha(w.snapx)} selection sha256 {sha(w.record)}\n')
    f.write_text(good)
    e = w.J / 'v1/x2/ecapa/vctk2k_30.jsonl'
    good = e.read_text()
    rows = rj(e)
    rows[3]['sim_ecapa'] += 0.002
    wj(e, rows)
    rc, out = w.runx(ONLY='ltc2k')
    assert rc == 1 and '|Δ sim_ecapa| > 0.001 on 1' in out and 'nothing scored' in out
    k = kinds(w.calls())
    assert k[-1] == 'drift_ecapa' and 'run_mvc' not in k and not (w.X / 'm').exists()
    e.write_text(good)
    rc, out = w.runx(ONLY='ltc2k')                                         # no conversion again: the failure stays
    assert rc == 0, out
    assert kinds(w.calls()) == [x for x in STEPS if x != 'run_mvc'] and done.read_text() == marker
    assert f'conv ltc2k 30 s {X}: 103 rows for 103 pairs, 1 failed' in out
    assert 'ltc2k_30: 103 metric rows (1 systems x 103 pairs), 0 shortfall rows, 1 failed conversions' in out


def test_job_rescores_ecapa_on_request_only(tmp_path):
    w = XWorld(tmp_path)
    assert w.runx(ONLY='vctk2k')[0] == 0
    (w.X / 'vctk2k.done').unlink()
    e = w.X / 'x2' / 'ecapa' / 'vctk2k_30.jsonl'
    r = rj(w.X / 'm' / 't1' / 'rows' / 'vctk2k_30.jsonl')[0]
    wj(e, [{k: r[k] for k in ('pair_id', 'system', 'set', 'budget')} | {'error': 'RuntimeError: CUDA error'}])
    rc, out = w.runx(ONLY='vctk2k')                                        # score_ecapa.py keeps the error row
    assert rc == 1 and '1 error rows on converted rows' in out and not (w.X / 'vctk2k.done').exists()
    rc, out = w.runx(ONLY='vctk2k', ECAPA_RESCORE='1')
    assert rc == 0, out
    assert out.count('unread (log it as an operational change)') == 2 and SENTINEL not in out
    assert len(rj(e)) == 4 and not any('error' in x for x in rj(e))


def test_job_follows_the_ceai_rules():
    s = (t1t.JOBS / JOB).read_text()
    for line in ('#SBATCH --account=uit', '#SBATCH --partition=normal', '#SBATCH --gres=gpu:2g.35gb:1',
                 '#SBATCH --time=08:00:00', '#SBATCH --job-name=mvc_xmvc1', 'source $WORK_DIR/mvc/jobs/t1_lib.sh'):
        assert line in s, line
    code = '\n'.join(x for x in s.splitlines() if not x.lstrip().startswith('#'))
    assert s.count(t1t.ENV_LINE) == 1 and not re.search(r'summarize\.py', code)
    assert '/test' not in code and '$J/v1' not in code and 'T=$X' in code       # writes under $X only


# ── xmvc1_check.py ───────────────────────────────────────────

def test_cfg_is_mvc_v1_with_run_dir_and_ckpt_only(tmp_path, capsys):
    src = tmp_path / 'mvc_v1.json'
    v1 = dict(run_dir='/d/runs/r1', ckpt='ema_0200000.pt', steps=16, cfg=1.0, mode='convert', decode='latent',
              neg_src=1.0, relax=0.15, stats_path=None)
    src.write_text(json.dumps(v1) + '\n')
    out = tmp_path / 'x' / 'cfg' / f'{X}.json'
    assert problems(xc.build_cfg, str(src), '/d/runs/r1x', 'ema_0300000.pt', str(out)) == []
    assert json.loads(out.read_text()) == dict(v1, run_dir='/d/runs/r1x', ckpt='ema_0300000.pt')
    assert 'the other 7 keys unchanged' in capsys.readouterr().out
    assert problems(xc.build_cfg, str(src), '/d/runs/r1x', 'ema_0300000.pt', str(out)) == []   # a resubmission
    assert any('exists and differs' in p for p in problems(xc.build_cfg, str(src), '/d/runs/r1x', 'ema_0290000.pt',
                                                           str(out)))
    assert problems(xc.build_cfg, str(src), '/d/runs/r1', 'ema_0300000.pt', str(tmp_path / 'o.json')) == [
        f"the X-mvc-1 cfg differs from {src} in ['ckpt'], not exactly in ['ckpt', 'run_dir']"]
    src.write_text(json.dumps(dict(v1, mode='copy_latent')) + '\n')
    assert any('not a mode convert cfg' in p for p in problems(xc.build_cfg, str(src), '/r1x', 'e.pt',
                                                                 str(tmp_path / 'o.json')))


def test_snapshot_and_stats_sha256_must_not_change(tmp_path):
    runs, out = tmp_path / 'runs', tmp_path / 'snapshot.sha256'
    snap, st, r1 = runs / 'r1x' / 'ema_0300000.pt', runs / 'r1x' / 'stats.npz', runs / 'r1' / 'stats.npz'
    for p, b in ((snap, b'w'), (st, b'stats'), (r1, b'stats')):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    fr = tmp_path / 'FREEZE_mvc_v1.sha256'
    fr.write_text(f'# Section 2\n{sha(r1)}  {r1}\n')

    def P(*a):
        return problems(xc.snapshot, str(snap), str(st), *a, str(out))
    assert P(str(r1), str(fr)) == [] and out.read_text() == f'{sha(snap)}  {snap}\n{sha(st)}  {st}\n'
    assert P(str(r1), str(fr)) == []                                           # a resubmission
    assert xc.read_snapshot(str(out), xc.tc.Problems()) == (str(snap), sha(snap))
    if shutil.which('sha256sum'):                                              # sha256sum's own format
        assert subprocess.run(['sha256sum', '-c', '--status', str(out)]).returncode == 0
    assert any('gives 0 sha256 of' in p for p in P(str(runs / 'r0' / 'stats.npz'), str(fr)))   # not frozen
    st.write_bytes(b'other stats')                                             # not R1's normalisation statistics
    assert any("not that of R1's" in p for p in P(str(r1), str(fr)))
    st.write_bytes(b'stats')
    snap.write_bytes(b'other')
    assert any('a file changed' in p for p in P(str(r1), str(fr)))
    snap.unlink()
    assert any('missing' in p for p in P(str(r1), str(fr)))


def test_files_lists_relative_paths_with_sha256(tmp_path):
    for f in ('a/x.jsonl', 'b.json'):
        (tmp_path / f).parent.mkdir(exist_ok=True)
        (tmp_path / f).write_text(f)
    out = tmp_path / 'vctk2k.sha256'
    assert problems(xc.files, str(tmp_path), ['a/x.jsonl', 'b.json'], str(out)) == []
    assert out.read_text() == f'{sha(tmp_path / "a/x.jsonl")}  a/x.jsonl\n{sha(tmp_path / "b.json")}  b.json\n'
    assert any('2 files missing' in p for p in problems(xc.files, str(tmp_path), ['c.json', str(tmp_path / 'b.json')],
                                                        str(tmp_path / 'o.sha256')))
    assert not (tmp_path / 'o.sha256').exists()


@pytest.fixture(scope='module')
def converted(tmp_path_factory):
    """A World after the vctk2k job: its manifest, cfg and snapshot file."""
    w = XWorld(tmp_path_factory.mktemp('conv'))
    assert w.runx(ONLY='vctk2k')[0] == 0
    return w


def conv_problems(w, tmp, edit=None):
    d = tmp / 'conv'
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(w.X / 'conv' / 'vctk2k' / '30' / X, d)
    if edit:
        edit(d)
    return problems(xc.check_conv, str(d / 'manifest.jsonl'), str(w.pairs['vctk2k']), 'vctk2k', X,
                    str(w.X / 'cfg' / f'{X}.json'), str(w.X / 'snapshot.sha256'), w.commit)


def edit_runs(fn):
    def edit(d):
        ev = rj(d / 'runs.jsonl')
        fn(ev[0]['conv_cfg']['describe'])
        wj(d / 'runs.jsonl', ev)
    return edit


def edit_manifest(fn):
    def edit(d):
        rows = rj(d / 'manifest.jsonl')
        wj(d / 'manifest.jsonl', fn(rows))
    return edit


@pytest.mark.parametrize('edit, msg', [
    (None, None),
    (edit_runs(lambda d: d.update(commit='f' * 16)), "not the record's DESCRIBE_COMMIT"),
    (edit_runs(lambda d: d['settings'].update(cfg=0.0)), "differ from"),
    (edit_runs(lambda d: d['settings'].update(run_dir='/x/runs/r1', ckpt='ema_0200000.pt')), "in ['ckpt', 'run_dir']"),
    (edit_runs(lambda d: d.update(checkpoints={'/x/runs/r1/ema_0200000.pt': '0' * 64})), 'converted with the checkpoints'),
    (edit_manifest(lambda rows: rows[1:]), 'rows on 3 of the 4 pairs'),
    (edit_manifest(lambda rows: rows + [dict(rows[0], system=MVC)]), 'a row of mvc_v1, vctk2k, 30 s'),
    (edit_manifest(lambda rows: [dict(rows[0], error='ValueError: reference shortfall', shortfall=True)] + rows[1:]),
     '1 shortfall rows'),
    (edit_manifest(lambda rows: [dict(rows[0], conv_cfg_sha='other')] + rows[1:]), '1 rows not of the conv_cfg'),
    (lambda d: os.remove(rj(d / 'manifest.jsonl')[0]['wav']), 'WAV is missing')])
def test_conv_check(converted, tmp_path, edit, msg):
    w = converted
    if msg == 'WAV is missing':                    # the manifest names $X's WAVs: work on a copy of them
        wavs = [Path(r['wav']) for r in rj(w.X / 'conv' / 'vctk2k' / '30' / X / 'manifest.jsonl')]
        saved = {p: p.read_bytes() for p in wavs}
        try:
            P = conv_problems(w, tmp_path, edit)
        finally:
            for p, b in saved.items():
                p.write_bytes(b)
    else:
        P = conv_problems(w, tmp_path, edit)
    assert (P == []) if msg is None else any(msg in p for p in P), P


# ── submit_xmvc1.sh ──────────────────────────────────────────

class XQueue(t1t.Queue):
    """test_t1_jobs.Queue's fakes, with T1's done files, running submit_xmvc1.sh."""

    def __init__(self, tmp, t1_done=T1_DONE):
        super().__init__(tmp)
        self.X = self.R / 'data_storage' / 'mvc' / 'xmvc1'
        for f in t1_done:
            (self.T / f).parent.mkdir(parents=True, exist_ok=True)
            (self.T / f).write_text('job 1\n')

    def run(self, *args, **env):
        script = self.tmp / 'submit_xmvc1.sh'
        script.write_text((t1t.JOBS / 'submit_xmvc1.sh').read_text().replace('/datastore/ceai/crt/mimi_vc', str(self.R)))
        e = {k: v for k, v in os.environ.items() if k != 'ECAPA_RESCORE'}
        e.update(env, POLL='0', PATH=f'{self.R}/bin:{os.environ["PATH"]}')
        p = subprocess.run(['bash', str(script), *args], capture_output=True, text=True, env=e)
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        self.log.unlink(missing_ok=True)
        return p.returncode, p.stdout + p.stderr, lines


def test_submit_xmvc1_submits_each_set_after_slot_wait(tmp_path):
    q = XQueue(tmp_path)
    rc, out, log = q.run()
    assert rc == 0, out
    assert log == ['slot_wait', 'submit job_xmvc1.slurm --export=ALL,ONLY=vctk2k in journal',
                   'slot_wait', 'submit job_xmvc1.slurm --export=ALL,ONLY=ltc2k in journal']
    assert 'ONLY=vctk2k is job 101' in out and 'ONLY=ltc2k is job 102' in out
    assert 'waiting to submit job_xmvc1.slurm' in out                    # submit_ceai.sh's lines are shown
    rc, out, log = q.run('ltc2k', ECAPA_RESCORE='1')
    assert rc == 0 and log == ['slot_wait', 'submit job_xmvc1.slurm --export=ALL,ONLY=ltc2k,ECAPA_RESCORE=1 in journal']


def test_submit_xmvc1_refusals(tmp_path):
    q = XQueue(tmp_path)
    rc, out, log = q.run('t1')
    assert rc == 1 and 'is not a set' in out and log == []
    q.X.mkdir(parents=True)
    (q.X / 'ltc2k.done').write_text('job 1\n')
    rc, out, log = q.run()
    assert rc == 1 and 'ltc2k.done exists' in out and log == []
    rc, out, log = q.run('vctk2k')
    assert rc == 0 and log == ['slot_wait', 'submit job_xmvc1.slurm --export=ALL,ONLY=vctk2k in journal']
    for missing in T1_DONE:
        q = XQueue(tmp_path / missing.replace('/', '_'), t1_done=[f for f in T1_DONE if f != missing])
        rc, out, log = q.run()
        assert rc == 1 and f'{missing} missing: T1 is not done' in out and log == []


# ── dnsmos_xmvc1.py ──────────────────────────────────────────

MARK = tdt.MARK


def dnsmos_env(tmp_path, monkeypatch, other=None):
    xr, remote = tmp_path / 'xr', tmp_path / 'remote'
    for s in dt.SETS:
        rows = []
        for i in range(3):
            pid = f'{s}__s{i}__t'
            r = dict(pair_id=pid, system=X, utmos=float(MARK + '5'),
                     wav=f'/datastore/ceai/crt/mimi_vc/data_storage/mvc/xmvc1/conv/{s}/30/{X}/{pid}.wav')
            if i == 2:
                r['error'] = 'FileNotFoundError: no wav'               # a failed conversion: its WAV does not exist
            else:
                real = remote / r['wav'].lstrip('/')
                real.parent.mkdir(parents=True, exist_ok=True)
                real.write_bytes(b'RIFF')
            rows.append(r)
        if other:
            rows.append(dict(rows[0], system=other))
        rel = f'm/{dt.PART[s]}/metrics/{s}_30.jsonl'
        wj(xr / rel, rows)
        (xr / f'{s}.sha256').write_text(f'{sha(xr / rel)}  {rel}\n')                 # the job's list (part)
        (xr / f'{s}.done').write_text(f'job 7001 2026-10-30 10:00:00 snapshot sha256 {"ab" * 32} selection sha256 '
                                      f'{ax.RECORD_SHA256}\n')
    models = tmp_path / 'models'
    models.mkdir()
    for k in dt.MODEL_SHA256:
        (models / k).write_bytes(b'onnx')
    stub = tmp_path / 'stub_dnsmos.py'
    stub.write_text(tdt.STUB)
    monkeypatch.setattr(dt, 'MODEL_SHA256', {k: dt.sha256(models / k) for k in dt.MODEL_SHA256})
    fetched = []

    def fetch(paths, wav_root, remote_host):
        fetched.extend(paths)
        for p in paths:
            src = remote / p.lstrip('/')
            if src.exists():
                Path(wav_root + p).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, wav_root + p)
        return 0

    monkeypatch.setattr(dt, 'fetch', fetch)
    return xr, ['--xr', str(xr), '--models', str(models), '--python', f'{sys.executable} {stub}'], fetched


def test_dnsmos_scores_both_sets_and_keeps_no_value_or_wav(tmp_path, monkeypatch, capsys):
    xr, argv, fetched = dnsmos_env(tmp_path, monkeypatch)
    assert dx.main(argv) == 0
    out = capsys.readouterr().out
    assert MARK not in out and '2 batches, 6 rows' in out
    assert sorted(p.name for p in (xr / 'dnsmos').iterdir()) == ['input', f'ltc2k_30_{X}.jsonl', f'vctk2k_30_{X}.jsonl']
    assert not list(xr.rglob('*.summary.json')) and not [p for p in (xr / 'wav').rglob('*') if p.is_file()]
    assert len(fetched) == 4 and all(p.startswith('/datastore/ceai/crt/mimi_vc/data_storage/mvc/xmvc1/') for p in fetched)
    rows = rj(xr / 'dnsmos' / f'vctk2k_30_{X}.jsonl')
    assert len(rows) == 3 and [r['pair_id'] for r in rows if 'error' in r] == ['vctk2k__s2__t']
    assert all(set(r) <= {'pair_id', 'system', 'wav'} for r in rj(xr / 'dnsmos' / 'input' / f'ltc2k_30_{X}.jsonl'))
    assert f'vctk2k {X}: 3 rows, 2 WAVs, 1 error rows' in out
    fetched.clear()
    assert dx.main(argv) == 0 and not fetched and capsys.readouterr().out.count('already scored, skipped') == 2


def test_dnsmos_refusals_and_dnsmos_t1_unchanged(tmp_path, monkeypatch):
    xr, argv, fetched = dnsmos_env(tmp_path, monkeypatch, other=MVC)
    with pytest.raises(SystemExit, match=f'holds a row of {MVC}'):
        dx.main(argv)
    xr, argv, fetched = dnsmos_env(tmp_path / 'b', monkeypatch)
    f = xr / 'm' / 't2' / 'metrics' / 'ltc2k_30.jsonl'
    good = f.read_text()
    f.write_text(good + '\n')                                               # not the file the job listed
    with pytest.raises(SystemExit, match='ltc2k_30.jsonl is not the file .*ltc2k.sha256 lists'):
        dx.main(argv)
    f.write_text(good)
    done = xr / 'vctk2k.done'
    done.write_text(done.read_text().replace(ax.RECORD_SHA256, '0' * 64))   # X-mvc-1 not done under the record
    with pytest.raises(SystemExit, match='vctk2k.done missing or not written under the selection record'):
        dx.main(argv)
    done.unlink()
    with pytest.raises(SystemExit, match='vctk2k.done missing'):
        dx.main(argv)
    real = dx.DNSMOS_T1_SHA256
    monkeypatch.setattr(dx, 'DNSMOS_T1_SHA256', '0' * 64)                   # the T1 driver changed
    with pytest.raises(SystemExit, match='differs from the sha256 0000'):
        dx.main(argv)
    monkeypatch.setattr(dx, 'DNSMOS_T1_SHA256', real)
    monkeypatch.setattr(dt, 'DNSMOS_SHA256', '0' * 64)
    with pytest.raises(SystemExit, match='frozen sha256'):
        dx.main(argv)
    assert not fetched
    # the T1 driver, imported unchanged (DEVIATIONS_mvc_v1.md, after the T1 analysis, note 1)
    assert dt.sha256(dt.__file__) == 'bfd93a3d1c6c823c2a4afa534e4f28df8cb398cbd98ed9efc7ecb6b5069d746d' == \
        dx.DNSMOS_T1_SHA256
    assert dx.RECORD_SHA256 == ax.RECORD_SHA256


# ── analyze_xmvc1.py: synthetic rows ─────────────────────────

LEVEL_PARAMS = {X: dict(rate=0.02, sim=0.91, utmos=3.9, src=0.67, e=0.63, e_src=0.21, ovrl=3.25, seed=1),
                MVC: dict(rate=0.03, sim=0.90, utmos=3.8, src=0.66, e=0.62, e_src=0.20, ovrl=3.20, seed=2),
                'target_copy': dict(rate=0.0, sim=0.95, utmos=4.0, src=0.65, e=0.8, e_src=0.12, ovrl=3.3, seed=3)}
XFAIL = 'vctk2k__s01_000__t01'          # a failed mvc_r1x300 conversion on a wer_primary VCTK-2k pair


def system_rows(pairs, system, fail=()):
    """{kind: rows} of one system at 30 s: metrics.py target and @src rows, ECAPA target and @src rows, DNSMOS and guard
    rows; a failed conversion gets an error row and no @src row (lab_rows.py, score_ecapa.py), dnsmos.py's error row
    and an acoustic_guard.py error row. target_copy only on the pairs with P."""
    c = LEVEL_PARAMS[system]
    rng = np.random.default_rng(c['seed'] + (0 if pairs[0]['set'] == 'vctk2k' else 10))
    out = {k: [] for k in ('m', 'm_src', 'e', 'e_src', 'd', 'g')}
    for i, p in enumerate(pairs):
        if system == 'target_copy' and not p['P']:
            continue
        base = {'pair_id': p['pair_id'], 'system': system, 'set': p['set'], 'budget': 30}
        if p['pair_id'] in fail:
            out['m'].append(dict(base, error='FileNotFoundError: no wav'))
            out['e'].append(dict(base, error='input row: FileNotFoundError: no wav'))
            out['d'].append({'pair_id': p['pair_id'], 'system': system, 'error': "KeyError: 'wav'"})
            out['g'].append({'pair_id': p['pair_id'], 'system': system, 'error': 'LibsndfileError: no file'})
            continue
        k, ins = int(rng.binomial(12, c['rate'])), (20 if i % 9 == 4 else 0)       # some rows above the cap
        out['m'].append(dict(base, wer_large_v3={'S': k, 'D': 0, 'I': ins, 'N': 12, 'cS': k, 'cD': 0, 'cI': ins,
                                                 'cN': 60},
                             sim_wavlm=c['sim'] + 0.02 * rng.standard_normal(),
                             utmos=c['utmos'] + 0.2 * rng.standard_normal()))
        out['m_src'].append(dict(base, system=system + '@src', sim_wavlm=c['src'] + 0.02 * rng.standard_normal()))
        out['e'].append(dict(base, sim_ecapa=c['e'] + 0.03 * rng.standard_normal()))
        out['e_src'].append(dict(base, system=system + '@src', sim_ecapa=c['e_src'] + 0.03 * rng.standard_normal()))
        o = c['ovrl'] + 0.1 * rng.standard_normal()
        out['d'].append({'pair_id': p['pair_id'], 'system': system, 'SIG': o + 0.3, 'BAK': 3.9, 'OVRL': o,
                         'P808': o + 0.4 + 0.05 * rng.standard_normal()})
        out['g'].append({'pair_id': p['pair_id'], 'system': system, 'hnr': 12 + rng.standard_normal(), 'hnr_src': 13.0,
                         'voiced_kept': 0.9, 'f0_pcc': 0.8, 'st_sd_ratio': 1.0, 'rms_db_diff': rng.standard_normal()})
    return out


def write_world(tmp, xsys=X, x_fail=(XFAIL,)):
    """The --xr, --mr and --v1 trees, TEST_RESULTS_mvc_v1.json naming every --mr and --v1 file, and the argv."""
    xr, mr, v1 = tmp / 'xr', tmp / 'mr', tmp / 'v1'
    pairs = {'vctk2k': tam.vctk_pairs(), 'ltc2k': tam.ltc_pairs()}
    data = {}
    for s, ps in pairs.items():
        part = ax.PART[s]
        data[s] = {X: system_rows(ps, X, x_fail), MVC: system_rows(ps, MVC)}
        rx = {k: [dict(r, system=r['system'].replace(X, xsys)) for r in v] for k, v in data[s][X].items()}
        rm = data[s][MVC]
        wj(v1 / 'pairs' / f'{s}.jsonl', ps)
        for d, rows, done in ((xr, rx, None), (mr, rm, 'm.done')):
            wj(d / 'm' / part / 'metrics' / f'{s}_30.jsonl', rows['m'])
            wj(d / 'm' / part / 'metrics' / f'{s}_30_src.jsonl', rows['m_src'])
            wj(d / 'x2' / 'ecapa' / f'{s}_30.jsonl', rows['e'])
            wj(d / 'x2' / 'ecapa' / f'{s}_30_src.jsonl', rows['e_src'])
        (mr / 'm' / part / 'm.done').write_text(f'job 6724 2026-10-28 12:13:43 selection sha256 {ax.RECORD_SHA256}\n')
        wj(xr / 'dnsmos' / f'{s}_30_{X}.jsonl', rx['d'])
        wj(xr / 'dnsmos' / 'input' / f'{s}_30_{X}.jsonl', [{'pair_id': 'x', 'system': X}])   # dnsmos_t1's inputs
        wj(mr / 'dnsmos' / f'{s}_30_{MVC}.jsonl', rm['d'])
        wj(mr / 'dnsmos' / f'{s}_30_knnvc.jsonl', [dict(r, system='knnvc') for r in rm['d']])
        wj(xr / 'guard' / f'{s}_30.jsonl', rx['g'])
        wj(xr / 'guard' / 'input' / f'{s}_30.jsonl', rx['m'])
        wj(mr / 'guard' / f'{s}_30.jsonl', rm['g'] + [dict(r, system='knnvc') for r in rm['g']])
    tc_rows = system_rows(pairs['vctk2k'], 'target_copy')
    data['vctk2k']['target_copy'] = tc_rows
    cand = system_rows(pairs['vctk2k'], MVC)
    wj(v1 / 'm' / 't1' / 'metrics' / 'vctk2k_30.jsonl', tc_rows['m'] + [dict(r, system='cand_pm18_rq')
                                                                         for r in cand['m']])
    wj(v1 / 'm' / 't1' / 'metrics' / 'vctk2k_30_src.jsonl', tc_rows['m_src'] + [
        dict(r, system='cand_pm18_rq@src') for r in cand['m_src']])
    (v1 / 'm' / 't1' / 'm.done').write_text(f'job 4653 2026-10-20 06:45:22 selection sha256 '
                                            f'{am.JOURNAL_SELECTION_SHA256}\n')
    (v1 / 'pairs' / 'vctk2k_source_E.json').write_text(json.dumps({p['source_speaker']: {'paths': ['/e.flac']}
                                                                   for p in pairs['vctk2k']}))
    rec_cfg = MVC_DIR / 'MVC_SELECTION_RECORD_v1_cfg' / 'mvc_v1.json'
    cfg = dict(json.loads(rec_cfg.read_text()), run_dir='/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x',
               ckpt='ema_0300000.pt')
    (xr / 'cfg').mkdir(parents=True)
    (xr / 'cfg' / f'{X}.json').write_text(json.dumps(cfg) + '\n')
    snap_sha = 'ab' * 32
    (xr / 'snapshot.sha256').write_text(f'{snap_sha}  {cfg["run_dir"]}/{cfg["ckpt"]}\n'
                                        f'{ax.R1_STATS_SHA256}  {cfg["run_dir"]}/stats.npz\n')
    for s in ax.SETS:
        (xr / f'{s}.done').write_text(f'job 7001 2026-10-30 10:00:00 snapshot sha256 {snap_sha} selection sha256 '
                                      f'{ax.RECORD_SHA256}\n')
    write_lists(xr)
    t1 = tmp / 'TEST_RESULTS_mvc_v1.json'
    t1.write_text(json.dumps({'script_sha256': ax.ANALYZE_MVC_SHA256, 'inputs': [
        {'path': str(p), 'sha256': sha(p)} for d in (mr, v1) for p in sorted(d.rglob('*.json*'))]}))
    argv = ['--xr', str(xr), '--mr', str(mr), '--v1', str(v1), '--t1', str(t1), '--record_cfg', str(rec_cfg),
            '--out', str(tmp / 'out' / 'X.md')]
    return argv, pairs, data


def write_lists(xr):
    """<set>.sha256 as the job writes them (xmvc1_check.py files), for the files the analysis reads."""
    for s in ax.SETS:
        p = ax.PART[s]
        fs = ['cfg/mvc_r1x300.json', 'snapshot.sha256', f'm/{p}/metrics/{s}_30.jsonl', f'm/{p}/metrics/{s}_30_src.jsonl',
              f'x2/ecapa/{s}_30.jsonl', f'x2/ecapa/{s}_30_src.jsonl', f'guard/{s}_30.jsonl']
        (xr / f'{s}.sha256').write_text(''.join(f'{sha(xr / f)}  {f}\n' for f in fs))


@pytest.fixture(scope='module')
def analysed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('xa')
    argv, pairs, data = write_world(tmp)
    assert ax.main(argv) == 0
    return (json.loads((tmp / 'out' / 'X.json').read_text()), (tmp / 'out' / 'X.md').read_text(), pairs, data,
            argv, tmp)


def ok(rows):
    return {r['pair_id']: r for r in rows if 'error' not in r}


def capped_wer(rows, pids):
    e = [r['wer_large_v3'] for k, r in rows.items() if k in pids]
    return 100.0 * sum(min(x['S'] + x['D'] + x['I'], x['N']) for x in e) / sum(x['N'] for x in e)


def test_levels_recomputed_independently(analysed):
    res, _, pairs, data, _, _ = analysed
    for s in ax.SETS:
        wer_pids = {p['pair_id'] for p in pairs[s] if p['wer_primary']}
        for n in (X, MVC):
            row, d = res['sets'][s]['levels'][n], data[s][n]
            m, e, dn = ok(d['m']), ok(d['e']), ok(d['d'])
            assert row['n_rows'] == len(pairs[s]) and row['fail'] == (1 if (n, s) == (X, 'vctk2k') else 0)
            want = {'wer': capped_wer(m, wer_pids), 'sim': np.mean([r['sim_wavlm'] for r in m.values()]),
                    'utmos': np.mean([r['utmos'] for r in m.values()]),
                    'sim_src': np.mean([r['sim_wavlm'] for r in d['m_src']]),
                    'ecapa_sim': np.mean([r['sim_ecapa'] for r in e.values()]),
                    'dnsmos_ovrl': np.mean([r['OVRL'] for r in dn.values()]),
                    'dnsmos_p808': np.mean([r['P808'] for r in dn.values()])}
            for q, v in want.items():
                x = row[q]
                assert x['est'] == pytest.approx(v, abs=1e-12), (s, n, q)
                assert x['lo'] <= x['est'] <= x['hi'] and x['B'] == stats.B and x['mode'] == at.MODE[s]
            g = res['sets'][s]['guards'][n]
            assert g['hnr']['est'] == pytest.approx(np.mean([r['hnr'] for r in d['g'] if 'error' not in r]), abs=1e-12)
            assert g['n_rows'] == len(pairs[s]) and g['fail'] == row['fail']
    assert 'target_copy' not in res['sets']['vctk2k']['levels']


def test_differences_recomputed_independently(analysed):
    res, _, pairs, data, _, _ = analysed
    for s in ax.SETS:
        diffs = {h['quantity']: h for h in res['sets'][s]['differences']}
        assert list(diffs) == list(ax.DIFFS) and all((h['sys_a'], h['sys_b']) == (X, MVC) for h in diffs.values())
        a, b = data[s][X], data[s][MVC]
        ma, mb = ok(a['m']), ok(b['m'])
        both = set(ma) & set(mb)
        wer_pids = {p['pair_id'] for p in pairs[s] if p['wer_primary']} & both
        want = {'wer': capped_wer(ma, wer_pids) - capped_wer(mb, wer_pids),
                'sim': np.mean([ma[k]['sim_wavlm'] - mb[k]['sim_wavlm'] for k in both]),
                'utmos': np.mean([ma[k]['utmos'] - mb[k]['utmos'] for k in both]),
                'ecapa_sim': np.mean([ok(a['e'])[k]['sim_ecapa'] - ok(b['e'])[k]['sim_ecapa'] for k in both]),
                'dnsmos_ovrl': np.mean([ok(a['d'])[k]['OVRL'] - ok(b['d'])[k]['OVRL'] for k in both])}
        for q, v in want.items():
            h = diffs[q]
            assert h['est'] == pytest.approx(v, abs=1e-12), (s, q)
            assert h['lo'] <= h['est'] <= h['hi'] and h['mode'] == at.MODE[s] and h['B'] == stats.B
            n_fail = 1 if s == 'vctk2k' else 0                     # XFAIL is a wer_primary pair
            assert (h['fail_a'], h['fail_b']) == (n_fail, 0)
            assert ('worst_case' in h) == bool(n_fail)
    # the CIs are stats.compare's on the same pairs, metric and bootstrap mode (VCTK-2k independent, LTC-2k linked)
    s = 'ltc2k'
    ra, rb = by_pid(data[s][X]['m']), by_pid(data[s][MVC]['m'])
    ref = stats.compare(ra, rb, [p['pair_id'] for p in pairs[s]], {p['pair_id']: (p['source_speaker'],
                        p['target_speaker']) for p in pairs[s]}, 'sim_wavlm', mode=stats.LINKED)
    h = next(h for h in res['sets'][s]['differences'] if h['quantity'] == 'sim')
    assert (h['est'], h['lo'], h['hi'], h['se']) == pytest.approx((ref['est'], ref['lo'], ref['hi'], ref['se']))


def by_pid(rows):
    return {r['pair_id']: r for r in rows}


def test_nl_as_analyze_mvc_defines_it(analysed):
    res, _, pairs, data, _, _ = analysed
    tc = by_pid(data['vctk2k']['target_copy']['m_src'])
    src = {x: by_pid(data['vctk2k'][x]['m_src']) for x in (X, MVC)}
    for g, r in res['nl'].items():
        pids = [p['pair_id'] for p in pairs['vctk2k'] if p['P'] and stats.stratum(p['genders']) == g]
        assert r['n_pairs_P'] == len(pids) and r['n_no_source_E'] == 0
        for x in (X, MVC):
            use = [k for k in pids if k in src[x]]
            assert r[x]['est'] == pytest.approx(np.mean([src[x][k]['sim_wavlm'] - tc[k]['sim_wavlm'] for k in use]),
                                                abs=1e-12)
            assert r[x]['target_copy_failures'] == 0 and r[x]['lo'] <= r[x]['est'] <= r[x]['hi']
        common = [k for k in pids if k in src[X] and k in src[MVC]]
        assert r['diff']['est'] == pytest.approx(np.mean([src[X][k]['sim_wavlm'] - src[MVC][k]['sim_wavlm']
                                                          for k in common]), abs=1e-12)
        if len(common) == len(pids):                   # no failure: Δ NL is exactly NL(mvc_r1x300) − NL(mvc_v1)
            assert r['diff']['est'] == pytest.approx(r[X]['est'] - r[MVC]['est'], abs=1e-12)
    # exactly analyze_mvc.nl_family (§3.3)
    _, _, _, _, argv, _ = analysed
    a = dict(zip(argv[::2], argv[1::2]))
    v1 = Path(a['--v1'])
    ps = at.read_pairs(v1 / 'pairs' / 'vctk2k.jsonl')
    R = am.merge([('x', at.load_set(Path(a['--xr']) / 'm', 'vctk2k', [('t1', 30)], ps)[0]),
                  ('m', at.load_set(Path(a['--mr']) / 'm', 'vctk2k', [('t1', 30)], ps)[0]),
                  ('j', ax.keep_systems(at.load_set(v1 / 'm', 'vctk2k', [('t1', 30)], ps)[0], ('target_copy',)))])
    ts = am.make_set('vctk2k', ps, R, {p['source_speaker'] for p in ps}, keep=None)
    for x in (X, MVC):
        fam = am.nl_family(ts, x=x, beside=False)
        for g in stats.STRATA:
            assert {k: res['nl'][g][x][k] for k in ('est', 'lo', 'hi', 'se', 'n')} == {
                k: fam['strata'][g][k] for k in ('est', 'lo', 'hi', 'se', 'n')}


FORBIDDEN_KEYS = {'p', 'p_ni', 'wilcoxon', 'reject', 'rejects', 'all_reject', 'favours', 'holm', 'non_inferior',
                  'worst_case_all_reject'}


def keys(x):
    if isinstance(x, dict):
        return set(x) | {k for v in x.values() for k in keys(v)}
    if isinstance(x, list):
        return {k for v in x for k in keys(v)}
    return set()


def test_no_p_value_holm_or_decision_and_the_header(analysed):
    res, md, _, _, _, tmp = analysed
    assert not keys(res) & FORBIDDEN_KEYS
    low = md.lower()
    for word in ('holm', 'p value', 'p-value', '| p |', 'reject', 'favour', 'better', 'worse', 'significan',
                 'improv'):
        assert word not in low, word
    head = md.split('## vctk2k')[0]
    for x in ('# Exploratory addition X-mvc-1', '§4.10', '**Exploratory (§4.10).**', '**Descriptive only.**',
              'nothing here is a test or a decision', 'never replaces or relabels a mvc_v1 result',
              "R1x's 250k snapshot was unusable", 'operational note 3', 'operational note 4',
              f'B = {stats.B}, seed {stats.SEED}: VCTK-2k independent, LTC-2k linked',
              '/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/ema_0300000.pt',
              f"R1's normalisation statistics (`/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/stats.npz`, "
              f"sha256 {ax.R1_STATS_SHA256}"):
        assert x in head, x
    assert f"so it equals NL({X}) − NL({MVC}) when neither system failed on the stratum's pairs" in md
    for x in ('## vctk2k, 30 s', '## ltc2k, 30 s', f'### {X} − {MVC}, paired (95 % CIs)', '### Acoustic guards',
              '## Natural-level leakage NL on VCTK-2k', '## Inputs'):
        assert x in md, x
    assert res['changed'] == {'ckpt': ['ema_0200000.pt', 'ema_0300000.pt'],
                              'run_dir': ['/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1',
                                          '/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x']}
    assert res['analyze_mvc_sha256'] == ax.ANALYZE_MVC_SHA256 and res['frozen'] == am.FROZEN
    # per set: X-mvc-1 2 metrics, 2 ECAPA, 1 DNSMOS, 1 guard, 1 list; T1 the same but the list; plus cfg and
    # snapshot.sha256 (X-mvc-1), 3 pair files and journal job M part t1's 2 files (T1), the record's cfg and the T1
    # results (other)
    assert (len(res['inputs']['x_mvc_1']), len(res['inputs']['t1']), len(res['inputs']['other'])) == (16, 17, 2)
    assert (res['stats_npz'], res['stats_npz_sha256']) == (
        '/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/stats.npz', ax.R1_STATS_SHA256)
    fr = (MVC_DIR / 'FREEZE_mvc_v1.sha256').read_text()            # R1's stats.npz as frozen
    assert f'{ax.R1_STATS_SHA256}  /datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1/stats.npz\n' in fr
    assert all(len(x['sha256']) == 64 for g in res['inputs'].values() for x in g)
    assert set(res['done']) == {'X-mvc-1 vctk2k', 'X-mvc-1 ltc2k', 'MimiFlow-VC T1 part t1', 'MimiFlow-VC T1 part t2',
                                'journal job M part t1'}


def test_refusals(tmp_path, monkeypatch, analysed):
    _, _, _, _, argv, _ = analysed
    with pytest.raises(SystemExit, match='an earlier output is never overwritten'):
        ax.main(argv)

    def run(edit=None, **kw):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        a, _, _ = write_world(d, **kw)
        if edit:
            edit(d)
        return ax.main(a)
    with pytest.raises(SystemExit, match=f'holds rows of .*{MVC}'):        # an mvc_v1 row in the X-mvc-1 tree
        run(xsys=MVC)
    with pytest.raises(SystemExit, match='are not inputs of'):              # a T1 file the T1 analysis did not read
        run(lambda d: wj(d / 'mr' / 'dnsmos' / f'ltc2k_30_{MVC}.jsonl', rj(d / 'mr' / 'dnsmos' /
                                                                         f'ltc2k_30_{MVC}.jsonl')[1:]))
    with pytest.raises(SystemExit, match=r"differs from mvc_v1.json in \['cfg', 'ckpt', 'run_dir'\]"):
        run(lambda d: (d / 'xr' / 'cfg' / f'{X}.json').write_text(json.dumps(dict(json.loads(
            (d / 'xr' / 'cfg' / f'{X}.json').read_text()), cfg=0.0))))
    with pytest.raises(SystemExit, match='does not name the snapshot sha256'):
        run(lambda d: (d / 'xr' / 'snapshot.sha256').write_text(
            (d / 'xr' / 'snapshot.sha256').read_text().replace('ab' * 32, 'cd' * 32)))
    with pytest.raises(SystemExit, match='ltc2k.done missing'):
        run(lambda d: (d / 'xr' / 'ltc2k.done').unlink())
    with pytest.raises(SystemExit, match=f'vctk2k ECAPA {X}: 95 rows for 96 pairs'):
        run(lambda d: (wj(d / 'xr' / 'x2' / 'ecapa' / 'vctk2k_30.jsonl',
                          rj(d / 'xr' / 'x2' / 'ecapa' / 'vctk2k_30.jsonl')[1:]), write_lists(d / 'xr')))
    with pytest.raises(SystemExit, match=r"differs from mvc_v1.json in \['ckpt', 'run_dir'\]"):   # no KeyError
        run(lambda d: (d / 'xr' / 'cfg' / f'{X}.json').write_text(json.dumps({k: v for k, v in json.loads(
            (d / 'xr' / 'cfg' / f'{X}.json').read_text()).items() if k != 'ckpt'})))
    with pytest.raises(SystemExit, match="stats.npz has sha256 0000.*not R1's frozen stats.npz"):
        run(lambda d: (d / 'xr' / 'snapshot.sha256').write_text(
            (d / 'xr' / 'snapshot.sha256').read_text().replace(ax.R1_STATS_SHA256, '0' * 64)))
    with pytest.raises(SystemExit, match='ltc2k_30.jsonl is not the file .*ltc2k.sha256 lists'):   # the copy
        run(lambda d: wj(d / 'xr' / 'guard' / 'ltc2k_30.jsonl', rj(d / 'xr' / 'guard' / 'ltc2k_30.jsonl')[::-1]))
    with pytest.raises(SystemExit, match='guard/vctk2k_30.jsonl.*are not listed with their sha256'):
        run(lambda d: (d / 'xr' / 'vctk2k.sha256').write_text(''.join(
            x for x in open(d / 'xr' / 'vctk2k.sha256') if 'guard/' not in x)))
    with pytest.raises(SystemExit, match='vctk2k.sha256 missing'):
        run(lambda d: (d / 'xr' / 'vctk2k.sha256').unlink())
    monkeypatch.setattr(ax, 'ANALYZE_MVC_SHA256', '0' * 64)
    with pytest.raises(SystemExit, match='not the 0000'):
        run()
    monkeypatch.setitem(am.FROZEN, 'stats', '0' * 64)
    with pytest.raises(SystemExit, match='frozen journal code changed'):
        run()


# ── analyze_xmvc1.py: the T1 rows on this Mac ────────────────

MR = REPO / 'paper' / 'mimi_vc' / 'results'
V1 = REPO / 'paper' / 'journal_extension' / 'results' / 'v1'
T1_RESULTS = MVC_DIR / 'TEST_RESULTS_mvc_v1.json'


def real_xr(xr):
    """An X-mvc-1 tree made of T1's mvc_v1 rows (metrics, ECAPA, DNSMOS, guards) relabelled mvc_r1x300, with the
    job's cfg, snapshot.sha256, done files and lists."""
    def relabel(rows):
        return [dict(r, system=r['system'].replace(MVC, X, 1)) for r in rows if r['system'].split('@')[0] == MVC]
    for s in ax.SETS:
        p = ax.PART[s]
        for k in ('', '_src'):
            wj(xr / 'm' / p / 'metrics' / f'{s}_30{k}.jsonl', relabel(rj(MR / 'm' / p / 'metrics' / f'{s}_30{k}.jsonl')))
            wj(xr / 'x2' / 'ecapa' / f'{s}_30{k}.jsonl', relabel(rj(MR / 'x2' / 'ecapa' / f'{s}_30{k}.jsonl')))
        wj(xr / 'dnsmos' / f'{s}_30_{X}.jsonl', relabel(rj(MR / 'dnsmos' / f'{s}_30_{MVC}.jsonl')))
        wj(xr / 'guard' / f'{s}_30.jsonl', relabel(rj(MR / 'guard' / f'{s}_30.jsonl')))
    rec_cfg = MVC_DIR / 'MVC_SELECTION_RECORD_v1_cfg' / 'mvc_v1.json'
    cfg = dict(json.loads(rec_cfg.read_text()), run_dir='/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x',
               ckpt='ema_0300000.pt')
    (xr / 'cfg').mkdir()
    (xr / 'cfg' / f'{X}.json').write_text(json.dumps(cfg) + '\n')
    (xr / 'snapshot.sha256').write_text(f'{"ab" * 32}  {cfg["run_dir"]}/{cfg["ckpt"]}\n'
                                        f'{ax.R1_STATS_SHA256}  {cfg["run_dir"]}/stats.npz\n')
    for s in ax.SETS:
        (xr / f'{s}.done').write_text(f'job 7001 2026-10-30 10:00:00 snapshot sha256 {"ab" * 32} selection sha256 '
                                      f'{ax.RECORD_SHA256}\n')
    write_lists(xr)


@pytest.mark.skipif(not (MR / 'guard' / 'ltc2k_30.jsonl').exists() or not T1_RESULTS.exists(),
                    reason='the T1 rows are not on this machine')
def test_real_t1_rows_relabelled_give_zero_differences_and_t1s_numbers(tmp_path, capsys):
    """main() on the T1 rows of mvc_v1 relabelled mvc_r1x300 (and the T1 files themselves, which pass the
    TEST_RESULTS inputs check): every Δ is 0 [0, 0], and both systems' levels, guards and NL are those
    TEST_RESULTS_mvc_v1.json reports for mvc_v1, so X-mvc-1 shows mvc_v1 exactly as T1 reported it."""
    real_xr(tmp_path / 'xr')
    out = tmp_path / 'out' / 'X.md'
    assert ax.main(['--xr', str(tmp_path / 'xr'), '--mr', str(MR), '--v1', str(V1), '--t1', str(T1_RESULTS),
                    '--out', str(out)]) == 0
    capsys.readouterr()
    res, t1 = json.loads(out.with_suffix('.json').read_text()), json.loads(T1_RESULTS.read_text())
    ks = ('est', 'lo', 'hi', 'n')
    for s in ax.SETS:
        lv = res['sets'][s]['levels']
        want = {'wer': t1['levels'][s][MVC]['wer'], 'sim': t1['levels'][s][MVC]['sim'],
                'utmos': t1['levels'][s][MVC]['utmos'], 'sim_src': t1['levels'][s][MVC]['sim_src'],
                'ecapa_sim': t1['ecapa']['levels'][s][MVC]['sim'], 'dnsmos_ovrl': t1['dnsmos']['levels'][s][MVC]['OVRL'],
                'dnsmos_p808': t1['dnsmos']['levels'][s][MVC]['P808']}
        for n in (X, MVC):
            for q, x in want.items():
                assert {k: lv[n][q][k] for k in ks} == {k: x[k] for k in ks}, (s, n, q)
            assert (lv[n]['n_rows'], lv[n]['fail']) == (t1['levels'][s][MVC]['n_rows'], t1['levels'][s][MVC]['fail'])
            gd = res['sets'][s]['guards'][n]
            for k in am.GUARD_FIELDS:
                assert {j: gd[k][j] for j in ks[:3]} == {j: t1['guards'][s][MVC][k][j] for j in ks[:3]}, (s, n, k)
        for h in res['sets'][s]['differences']:
            assert (h['est'], h['lo'], h['hi'], h['fail_a'], h['fail_b']) == (0, 0, 0, 0, 0), (s, h['quantity'])
        wer = next(h for h in res['sets'][s]['differences'] if h['quantity'] == 'wer')
        assert wer['n'] == t1['levels'][s][MVC]['wer']['n']                  # the wer_primary pairs
    for g, r in res['nl'].items():
        for n in (X, MVC):
            assert {k: r[n][k] for k in ks + ('se',)} == {k: t1['nl']['strata'][g][k] for k in ks + ('se',)}, (g, n)
        assert (r['diff']['est'], r['diff']['lo'], r['diff']['hi'], r['diff']['n']) == (0, 0, 0, r['n_pairs_P'])
