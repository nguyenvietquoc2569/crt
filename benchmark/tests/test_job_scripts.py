"""Static checks of the plan-v1 Slurm job layer (benchmark/job_*.slurm) against the account rules of
ANALYSIS_PLAN_v1.md §8.2 and RUNBOOK_v1.md: syntax, resources, paths, caches, Whisper models and output dirs."""
import glob
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = sorted(glob.glob(os.path.join(BENCH, 'job_*.slurm')))
ROOT = '/datastore/ceai/crt/'          # CRT's home since 19 Oct 2026 (DEVIATIONS_v1.md item 4); before: /datastore/nvquoc/crt
DEV_JOBS = ('job_d0.slurm', 'job_d1_metrics.slurm', 'job_d2.slurm', 'job_ot_screen.slurm', 'job_otf_screen.slurm',
            'job_move_ceai.slurm')
OLD_HOMES = r'/datastore/(nvquoc|eaic)|/datastore/ceai/nvquoc'


def sbatch(path):
    """#SBATCH options of a script as {name: value} (flags without a value map to '')."""
    out = {}
    for line in open(path):
        m = re.match(r'#SBATCH\s+--([\w-]+)(?:=(\S+))?', line)
        if m:
            out[m.group(1)] = m.group(2) or ''
    return out


def body(path):
    """The script without comment lines."""
    return ''.join(l for l in open(path) if not l.lstrip().startswith('#'))


def hours(t):
    """Slurm --time [D-]HH:MM:SS -> hours."""
    d, _, hms = t.rpartition('-')
    h, m, s = (int(x) for x in hms.split(':'))
    return int(d or 0) * 24 + h + m / 60 + s / 3600


def test_job_layer_present():
    names = {os.path.basename(p) for p in JOBS}
    for n in ('job_d0.slurm', 'job_d1_conv.slurm', 'job_d1_metrics.slurm', 'job_d2.slurm', 'job_t1.slurm',
              'job_t2.slurm', 'job_t3.slurm', 'job_t4.slurm', 'job_m.slurm', 'job_baselines_smoke.slurm',
              'job_move_ceai.slurm'):
        assert n in names


@pytest.mark.parametrize('path', JOBS, ids=os.path.basename)
def test_bash_syntax(path):
    r = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('path', JOBS, ids=os.path.basename)
def test_account_rules(path):
    o = sbatch(path)
    assert o.get('account') == 'uit'
    assert o.get('partition') == 'normal'
    assert o.get('gres') == 'gpu:2g.35gb:1'
    assert 0 < hours(o['time']) <= 72
    assert o.get('job-name') and o['job-name'] != 'qwen-serve'
    assert 'dependency' not in o
    if 'array' in o:
        assert o['array'].endswith('%1')


@pytest.mark.parametrize('path', JOBS, ids=os.path.basename)
def test_logs_and_env_under_ceai(path):
    o = sbatch(path)
    for k in ('output', 'error'):
        assert o[k].startswith(ROOT), o[k]
    b = body(path)
    envs = re.search(r'^ENVS=(\S+)', b, re.M)       # the smoke job selects its envs through $ENVS
    b_env = b.replace('$ENVS', envs.group(1)) if envs else b
    assert re.search(r'source /datastore/ceai/crt/envs/\w+/bin/activate', b_env)
    assert re.search(r'export HF_HUB_OFFLINE=1', b)
    for var in ('XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME'):
        m = re.search(rf'export {var}=(\S+)', b)
        assert m and ('$DATA_STORAGE' in m.group(1) or m.group(1).startswith(ROOT)), var


@pytest.mark.parametrize('path', JOBS, ids=os.path.basename)
def test_no_old_accounts_or_service(path):
    for i, line in enumerate(body(path).splitlines(), 1):
        if re.search(OLD_HOMES, line):
            # only the guards that refuse lists still holding an old home's paths may name them
            assert 'grep -q' in line or 'ERROR' in line, f'line {i}: {line}'
        assert not re.search(r'/datastore/(nvquoc|ceai)/qwen', line), f'line {i}: {line}'
        assert not re.search(r'\bscancel\b', line), f'line {i}: {line}'


@pytest.mark.parametrize('path', JOBS, ids=os.path.basename)
def test_whisper_models(path):
    """Dev jobs score with Whisper-large-v3 only, test-set scoring keeps v0's secondary Whisper-base (§3.2)."""
    calls = [c for c in re.findall(r'metrics\.py[^\n]*(?:\\\n[^\n]*)*', body(path)) if 'wer' in c]
    name = os.path.basename(path)
    for c in calls:
        m = re.search(r'--whisper_models\s+(\S+)', c)
        assert m, f'{name}: metrics.py with wer but no --whisper_models'
        want = 'large-v3' if name in DEV_JOBS or name == 'job_baselines_smoke.slurm' else 'large-v3,base'
        assert m.group(1).strip('"\'') == want, f'{name}: {m.group(1)}'


def test_output_dirs_distinct():
    """Every job writes into its own $J/v1/<dir>; the two D1 jobs share d1 by design (metrics read the conversions)."""
    outs = {}
    for p in JOBS:
        m = re.search(r'^OUT=(\S+)', body(p), re.M)
        assert m, os.path.basename(p)
        outs[os.path.basename(p)] = m.group(1)
    shared = {k: v for k, v in outs.items() if list(outs.values()).count(v) > 1}
    assert set(shared) <= {'job_d1_conv.slurm', 'job_d1_metrics.slurm'}, shared
    assert all(v.startswith('$J/v1/') for v in outs.values()), outs
