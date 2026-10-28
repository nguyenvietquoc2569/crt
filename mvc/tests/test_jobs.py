"""Tests of mvc/jobs: every job script parses, and eval_dev.sh's helpers call the scripts with the expected arguments
(python is replaced by a shell function that records its command lines)."""
import subprocess
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / 'jobs'


@pytest.mark.parametrize('script', sorted(p.name for p in JOBS.iterdir() if p.suffix in ('.sh', '.slurm')))
def test_job_scripts_parse(script):
    subprocess.run(['bash', '-n', str(JOBS / script)], check=True)


def run_helper(tmp_path, call, fail=''):
    log = tmp_path / 'calls.txt'
    script = f'''
MVC=/m J=/j RAW=/raw FAILED=
python() {{ echo "$*" >> {log}; [[ -n "{fail}" && "$*" == *{fail or 'x'}* ]] && return 1; return 0; }}
source {JOBS / 'eval_dev.sh'}
{call}
echo "FAILED=$FAILED"
'''
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=True).stdout
    return log.read_text().splitlines() if log.exists() else [], out.strip().splitlines()[-1]


def test_recon_eval_runs_probe_metrics_and_summary(tmp_path):
    calls, failed = run_helper(tmp_path, 'recon_eval p6_V_rec \'{"run_dir": "/m/runs/p6_V"}\'')
    assert len(calls) == 3 and failed == 'FAILED='
    assert calls[0].startswith('probe_recon.py --shards /m/mimi/dev-clean --raw /raw --name p6_V_rec')
    assert '--cfg {"run_dir": "/m/runs/p6_V"} --out /m/eval/recon --n 300' in calls[0]
    assert calls[1].startswith('/j/metrics.py --in_jsonl /m/eval/recon/p6_V_rec/rows.jsonl')
    assert '--metrics wer,sim,utmos --whisper_models large-v3' in calls[1]
    assert calls[2].startswith('summarize.py --metrics /m/eval/recon/p6_V_rec/metrics.jsonl')
    assert calls[2].endswith('--out /m/eval/recon/p6_V_rec/summary.json')


def test_recon_eval_records_failure_and_stops(tmp_path):
    calls, failed = run_helper(tmp_path, 'recon_eval r \'{}\'', fail='probe_recon')
    assert len(calls) == 1 and failed == 'FAILED= r:recon'


def test_score_dev_creates_the_metrics_folder(tmp_path):
    o = tmp_path / 'eval' / 'x'
    calls, failed = run_helper(tmp_path, f'score_dev {o} sys 30 /m/manifest.jsonl')
    assert (o / 'metrics').is_dir() and failed == 'FAILED='
    assert any(c.startswith('/j/metrics.py') and f'--out_jsonl {o}/metrics/dev_30.jsonl' in c for c in calls)


def test_recon_eval_subset_length_limit_and_output_folder(tmp_path):
    calls, failed = run_helper(tmp_path, 'recon_eval tr_V \'{"run_dir": "/m/runs/p6_V"}\' train-clean-100 200 10 /m/eval/p7')
    assert len(calls) == 3 and failed == 'FAILED='
    assert calls[0].startswith('probe_recon.py --shards /m/mimi/train-clean-100 --raw /raw --name tr_V')
    assert calls[0].endswith('--out /m/eval/p7 --n 200 --max_s 10')
    assert calls[1].startswith('/j/metrics.py --in_jsonl /m/eval/p7/tr_V/rows.jsonl')
    assert calls[2].endswith('--out /m/eval/p7/tr_V/summary.json')


def test_p7_ladder_runs_the_rungs_in_order(tmp_path):
    jobs = str(JOBS)
    calls, failed = run_helper(tmp_path, f'WORK_DIR={jobs}/../..; mkdir -p /tmp/none; '
                                         f'EXTRA=\'dv_x={{"run_dir": "/m/runs/x"}}\'; INIT_TS=0.5,0.9; source {jobs}/p7_ladder.sh')
    probes = [c for c in calls if c.startswith('probe_recon.py')]
    names = [c.split('--name ')[1].split()[0] for c in probes]
    assert names == ['tr_copy', 'tr_V', 'dv_copy', 'dv_V', 'dv_V_t0.5', 'dv_V_t0.9', 'dv_x'] and failed == 'FAILED='
    assert all(c.endswith('--out /m/eval/p7 --n 200 --max_s 10') for c in probes)
    assert '--shards /m/mimi/train-clean-100' in probes[1] and '--shards /m/mimi/dev-clean' in probes[3]
    assert '"init_t": 0.9' in probes[5] and '"mode": "copy_latent"' in probes[0]


def test_eval_job_sources_the_after_hook():
    s = (JOBS / 'job_eval.slurm').read_text()
    assert 'source $WORK_DIR/mvc/jobs/after_$TAG.sh' in s and (JOBS / 'after_p6sweep.sh').exists()


@pytest.mark.parametrize('evals', sorted(p.name for p in JOBS.glob('*_evals.txt')))
def test_eval_lists_parse_with_unique_names_and_known_settings(evals):
    import json
    import sys
    sys.path.insert(0, str(JOBS.parent))
    import vc_adapter
    items = (JOBS / evals).read_text().strip().split(';')
    names = [it.split('=', 1)[0] for it in items]
    assert len(names) == len(set(names))
    for it in items:
        cfg = json.loads(it.split('=', 1)[1])
        assert set(cfg) <= set(vc_adapter.DEFAULTS), set(cfg) - set(vc_adapter.DEFAULTS)
        if cfg.get('stats_path'):                           # one stats file per configuration
            assert cfg['stats_path'].endswith(f"/{it.split('=', 1)[0]}/stats.jsonl")


def test_r1_follows_plan_2_6_and_sets_tokens_once():
    s = (JOBS / 'job_r1.slurm').read_text()
    r1 = s.split('R1="', 1)[1].split('"', 1)[0]
    for flag in ('--tau_dist uniform', '--lr 1.5e-4', '--warmup 5000', '--sched const_tail', '--beta2 0.99',
                 '--ema 0.9999', '--d 768 --layers 16 --heads 12', '--whisper 1', 'audit_v1/exclude.txt'):
        assert flag in r1, flag
    assert '--tokens' not in r1 and s.count('--tokens $TOK') == 1


def test_r1x_continues_r1_without_writing_it():
    s = (JOBS / 'job_r1x.slurm').read_text()
    r1 = (JOBS / 'job_r1.slurm').read_text().split('R1="', 1)[1].split('"', 1)[0]
    rx = s.split('R1="', 1)[1].split('"', 1)[0]
    assert rx.replace('--steps 300000', '--steps $STEPS').replace(' --rewarm 2000 --rewarm_from 200000', '') == r1
    assert 'OUT=$MVC/runs/r1x' in s and "c['step'] == 200000" in s and '--out $OUT' in s and '--out $SRC' not in s
    h = s.index('sha256sum $SRC/last.pt $SRC/args.json $SRC/stats.npz $SRC/ema_*.pt > $MVC/runs/r1_hashes.sha256.tmp')
    assert h < s.index('mkdir -p $OUT')                 # absolute names, as select_mvc.py --r1_hashes reads them


def test_r1x_evals_are_mvc_v1_on_the_r1x_snapshot():
    """§4.10: the R1x Dev-S100 runs at 250k and 300k use mvc_v1's frozen settings (and CFG 0 for the reference) on
    R1x's snapshot; their hooks add ECAPA and the guards; the submit script waits for the snapshot."""
    import json
    v1 = json.loads((JOBS.parent / 'MVC_SELECTION_RECORD_v1_cfg' / 'mvc_v1.json').read_text())
    for k in (250, 300):
        items = dict(it.split('=', 1) for it in (JOBS / f'r1x{k}_evals.txt').read_text().strip().split(';'))
        assert set(items) == {f'r1x_k{k}_ref', f'r1x_k{k}_mvc'}
        for name, cfg in ((f'r1x_k{k}_mvc', 1.0), (f'r1x_k{k}_ref', 0.0)):
            c = json.loads(items[name])
            assert c == dict(v1, run_dir=v1['run_dir'] + 'x', ckpt=f'ema_0{k}000.pt', cfg=cfg)
        h = (JOBS / f'after_r1x{k}.sh').read_text()
        assert '$MVC/eval/$TAG/' in h and 'score_ecapa.py' in h and 'acoustic_guard.py' in h
    s = (JOBS / 'submit_r1x_evals.sh').read_text()
    assert 'TAG=r1x$K,BUDGETS=30,EVALS_FILE=$M/jobs/r1x${K}_evals.txt' in s
    code = '\n'.join(x for x in s.splitlines() if not x.lstrip().startswith('#'))
    assert code.index('f=$R/ema_0${K}000.pt') < code.index('until [ -f $f ]') < code.index('slot_wait.sh') \
        < code.index('submit_ceai.sh')
