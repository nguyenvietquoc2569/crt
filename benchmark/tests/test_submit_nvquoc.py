"""Tests of benchmark/submit_nvquoc.sh and of benchmark/submit_ceai.sh (the same wrapper for the account ceai, with
its own log and lock), each run against the stand-ins of its account as `bash <wrapper> <script> [args]` with PATH
starting at a temporary directory of stand-ins for date, sleep, id, flock, squeue, sacct and sbatch (plus scancel and scontrol, which
must never be called). Nothing real is submitted or queried.

The stand-ins share a state directory (STUB_STATE): a fake clock (`now`, epoch s) that `date` reads and `sleep`
advances, so waiting hours takes no real time; the queue ({id, name, submit, until}: a job is in squeue while submit
<= now < until, and in sacct from its submit on); Submit times known only to sacct (finished jobs); Submit times that
sacct does not show yet (`lag`); and calls.log, one line per call. `date` is GNU date as the wrapper calls it (+FMT,
-d @N, -d '-N days', -d <ISO time>), computed by the system date at the fake clock, in the zone of TZ or, with TZ
unset, of the login node (STUB_LOCALTIME, default Asia/Ho_Chi_Minh); STUB_NO_TZDATA makes every zone UTC, as glibc
does without the zone file. `flock 9` locks the descriptor it inherits with flock(2) and exits, leaving it locked, as
util-linux flock does. The wrapper writes its log and lock under tmp_path (SUBMIT_NVQUOC_LOG and SUBMIT_NVQUOC_LOCK,
or SUBMIT_CEAI_*) and never sees the caller's environment. squeue and sacct answer only for the account under test
(STUB_ACCOUNT), and id prints it unless STUB_USER says otherwise.
"""
import fcntl
import marshal
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNTS = {'nvquoc': ('submit_nvquoc', 'SUBMIT_NVQUOC'), 'ceai': ('submit_ceai', 'SUBMIT_CEAI')}   # name, env prefix
BASH = shutil.which('bash')
SYSTEM_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'
SYSTEM_DATE = shutil.which('date', path=SYSTEM_PATH)
ICT = ZoneInfo('Asia/Ho_Chi_Minh')
REAL_SLURM = [c for c in ('sbatch', 'squeue', 'sacct') if shutil.which(c) or shutil.which(c, path=SYSTEM_PATH)]

pytestmark = pytest.mark.skipif(
    BASH is None or SYSTEM_DATE is None or bool(REAL_SLURM) or os.path.exists('/etc/profile.d/modules.sh'),
    reason='needs bash, date and a machine without Slurm: the wrapper sources /etc/profile.d/modules.sh and runs '
           '`module load slurm`, which could put the real squeue/sacct/sbatch ahead of the stand-ins')

LOG_CALL = r'''S=$STUB_STATE
now=$(cat "$S/now")
IFS=$'\x1f' args="$*"
printf '%s\t%s\t%s\t%s\n' "${0##*/}" "$now" "${TZ-<unset>}" "$args" >> "$S/calls.log"
'''

# GNU date at the fake clock. Strict input check first: the system date (BSD on macOS) ignores trailing junk.
DATE = r'''#!/bin/bash
''' + LOG_CALL + r'''zone=${TZ-${STUB_LOCALTIME:-Asia/Ho_Chi_Minh}}
[ -z "$STUB_NO_TZDATA" ] && [ -n "$zone" ] || zone=UTC
t=$now fmt='%a %b %e %H:%M:%S %Z %Y' spec= has_spec=
while [ $# -gt 0 ]; do
    case $1 in
        -d|--date) spec=$2 has_spec=1; shift;;
        --date=*) spec=${1#--date=} has_spec=1;;
        +*) fmt=${1#+};;
        *) echo "date stub: unsupported argument '$1'" >&2; exit 2;;
    esac
    shift
done
re_at='^@(-?[0-9]+)$'
re_days='^[[:space:]]*([+-]?[0-9]+)[[:space:]]+days?[[:space:]]*$'
re_iso='^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}$'
invalid() { echo "date: invalid date '$spec'" >&2; exit 1; }
if [ -n "$has_spec" ]; then
    if [[ $spec =~ $re_at ]]; then
        t=${BASH_REMATCH[1]}
    elif [[ $spec =~ $re_days ]]; then
        t=$(( now + 86400 * ${BASH_REMATCH[1]} ))
    elif [[ $spec =~ $re_iso ]]; then
        t=$(TZ=$zone to_epoch "${spec/ /T}" 2>/dev/null) || invalid
    else
        invalid
    fi
fi
TZ=$zone from_epoch "$t" "+$fmt"
'''
DATE_BSD = r'''to_epoch() { /bin/date -j -f '%Y-%m-%dT%H:%M:%S' "$1" +%s; }
from_epoch() { /bin/date -r "$1" "$2"; }
'''
DATE_GNU = r'''to_epoch() { /bin/date -d "$1" +%s; }
from_epoch() { /bin/date -d "@$1" "$2"; }
'''

SLEEP = r'''#!/bin/bash
''' + LOG_CALL + r'''echo $(( now + ${1%.*} )) > "$S/now"
if [ "$(grep -c '^sleep' "$S/calls.log")" -gt "${STUB_MAX_SLEEPS:-200}" ]; then   # a wrapper that never stops waiting
    kill -TERM $PPID
    exit 1
fi
'''

ID = r'''#!/bin/bash
''' + LOG_CALL + r'''[ "$args" = -un ] || { echo "id stub: unexpected arguments $*" >&2; exit 2; }
echo "${STUB_USER:-$STUB_ACCOUNT}"
'''

NOOP = '#!/bin/bash\n' + LOG_CALL   # scancel, scontrol: recorded, so a test can check they were never called

# squeue, sacct, sbatch and flock; the queue state is a marshal file (no imports beyond fcntl).
SLURM = r'''
import fcntl, marshal, os, sys

S = os.environ['STUB_STATE']


def log(name, argv, now):
    with open(os.path.join(S, 'calls.log'), 'a') as f:
        f.write(f"{name}\t{now}\t{os.environ.get('TZ', '<unset>')}\t{chr(31).join(argv)}\n")


def fail(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def zone():
    from datetime import timezone
    name = os.environ.get('TZ')
    if name is None:
        name = os.environ.get('STUB_LOCALTIME', 'Asia/Ho_Chi_Minh')
    if not name or os.environ.get('STUB_NO_TZDATA'):
        return timezone.utc
    from zoneinfo import ZoneInfo
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def iso(t):   # Submit as sacct -o Submit and squeue -o %V print it: local time of the node
    from datetime import datetime
    return datetime.fromtimestamp(t, zone()).strftime('%Y-%m-%dT%H:%M:%S')


def parse_iso(s):
    digits = s[:4] + s[5:7] + s[8:10] + s[11:13] + s[14:16] + s[17:]
    if len(s) != 19 or s[4] + s[7] + s[10] + s[13] + s[16] != '--T::' or not digits.isdigit():
        fail(f'sacct: error: invalid time {s!r}')
    from datetime import datetime
    return int(datetime(int(s[:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:]),
                        tzinfo=zone()).timestamp())


def squeue(argv, d, now):
    if os.environ.get('SQUEUE_FAIL'):
        fail('squeue: error: Unable to contact slurm controller (connect failure)')
    jobs = [j for j in d['queue'] if j['submit'] <= now < j['until']]
    if argv == ['-h', '-u', os.environ['STUB_ACCOUNT'], '-o', '%i|%j|%T|%V']:
        for j in jobs:
            print(f"{j['id']}|{j['name']}|{'RUNNING' if now >= j['submit'] + 60 else 'PENDING'}|{iso(j['submit'])}")
        for line in d['squeue_extra']:
            print(line)
    elif len(argv) == 5 and argv[:2] == ['-h', '-j'] and argv[3:] == ['-o', '%V']:
        for j in jobs:
            if str(j['id']) == argv[2]:
                print(iso(j['submit']))
    else:
        fail(f'squeue stub: unexpected arguments {argv}', 2)


def sacct(argv, d, now):
    if os.environ.get('SACCT_FAIL'):
        fail('sacct: error: Problem talking to the database: Connection refused')
    if len(argv) == 9 and argv[:3] == ['-u', os.environ['STUB_ACCOUNT'], '-S'] \
            and argv[4:] == ['-X', '-n', '-P', '-o', 'Submit']:
        start = parse_iso(argv[3])
        times = {t for t in d['sacct'] if start <= t <= now}
        times |= {j['submit'] for j in d['queue'] if j['submit'] <= now and j['until'] >= start}
        for t in sorted(times - set(d['lag'])):
            print(iso(t))
        for line in d['sacct_extra']:
            print(line)
    elif len(argv) == 7 and argv[0] == '-j' and argv[2:] == ['-X', '-n', '-P', '-o', 'Submit']:
        for j in d['queue']:
            if str(j['id']) == argv[1] and j['submit'] <= now:
                print(iso(j['submit']))
    else:
        fail(f'sacct stub: unexpected arguments {argv}', 2)


def sbatch(argv, d, now):
    rec = {'argv': argv, 'now': now, 'env': {k: v for k, v in os.environ.items() if k.startswith('SBATCH_')}}
    if os.environ.get('STUB_SBATCH_FAIL'):
        d['sbatch'].append(dict(rec, failed=True))
        fail('sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified')
    jid = d['next_id']
    d['next_id'] += 1
    name = os.path.basename(argv[-1])
    for line in open(argv[-1]):
        if line.startswith('#SBATCH --job-name='):
            name = line.split('=', 1)[1].strip()
    if not d['hide_new']:
        d['queue'].append({'id': jid, 'name': name, 'submit': now, 'until': now + d['job_s']})
    d['sbatch'].append(dict(rec, id=jid))
    print(d['sbatch_out'].replace('{jid}', str(jid)))


def main(name):
    argv = sys.argv[1:]
    with open(os.path.join(S, 'now')) as f:
        now = int(f.read())
    log(name, argv, now)
    if name == 'flock':   # util-linux "flock <fd>": lock the descriptor inherited from the shell; it stays locked
        fcntl.flock(int(argv[-1]), fcntl.LOCK_EX)
        log('flock-acquired', argv, now)
        return
    with open(os.path.join(S, '.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(os.path.join(S, 'state'), 'rb') as f:
            d = marshal.load(f)
        try:
            {'squeue': squeue, 'sacct': sacct, 'sbatch': sbatch}[name](argv, d, now)
        finally:
            with open(os.path.join(S, 'state.tmp'), 'wb') as f:
                marshal.dump(d, f)
            os.replace(os.path.join(S, 'state.tmp'), os.path.join(S, 'state'))
'''

QWEN = {'id': 100, 'name': 'qwen-serve', 'submit': '2026-10-16T04:00:00', 'until': '2026-10-20T04:30:00'}


def T(s):
    """Epoch of an Asia/Ho_Chi_Minh wall-clock time."""
    return int(datetime.strptime(s, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=ICT).timestamp())


def ict(t):
    return datetime.fromtimestamp(t, ICT).strftime('%Y-%m-%dT%H:%M:%S')


def write_stubs(bin_dir):
    bsd = subprocess.run([SYSTEM_DATE, '--version'], capture_output=True).returncode != 0
    bin_dir.mkdir()
    (bin_dir / '_slurmstub.py').write_text(SLURM)
    scripts = {'date': DATE.replace('#!/bin/bash\n', '#!/bin/bash\n' + (DATE_BSD if bsd else DATE_GNU), 1)
               .replace('/bin/date', SYSTEM_DATE), 'sleep': SLEEP, 'id': ID, 'scancel': NOOP, 'scontrol': NOOP}
    for name in ('squeue', 'sacct', 'sbatch', 'flock'):
        scripts[name] = f'#!{sys.executable} -S\nfrom _slurmstub import main\nmain({name!r})\n'
    for name, text in scripts.items():
        (bin_dir / name).write_text(text.replace('#!/bin/bash', f'#!{BASH}', 1))
        (bin_dir / name).chmod(0o755)


class Slurm:
    def __init__(self, root, bin_dir, acct='nvquoc'):
        self.acct = acct
        self.name, prefix = ACCOUNTS[acct]
        self.wrapper = os.path.join(BENCH, self.name + '.sh')
        self.log_var, self.lock_var = prefix + '_LOG', prefix + '_LOCK'
        self.root = root
        self.bin = bin_dir
        self.state_dir = root / 'state'
        self.jobs = root / 'jobs'
        self.log = root / 'logs' / 'submit.log'
        self.lock = root / 'submit.lock'
        for d in (self.state_dir, self.jobs, root / 'home'):
            d.mkdir()
        self.set(now='2026-10-18T10:00:00', queue=[QWEN])

    def set(self, now, queue=(), sacct=(), lag=(), **kw):
        """A fresh state: the clock at `now`, the queue, Submit times of finished jobs, the calls log emptied."""
        t = lambda x: T(x) if isinstance(x, str) else x
        st = {'queue': [dict(j, submit=t(j['submit']), until=t(j['until'])) for j in queue],
              'sacct': [t(x) for x in sacct], 'lag': [t(x) for x in lag], 'squeue_extra': [], 'sacct_extra': [],
              'sbatch': [], 'next_id': 9000, 'job_s': 3600, 'sbatch_out': '{jid}', 'hide_new': False}
        st.update(kw)
        (self.state_dir / 'state').write_bytes(marshal.dumps(st))
        (self.state_dir / 'now').write_text(str(t(now)))
        (self.state_dir / 'calls.log').write_text('')

    def state(self):
        return marshal.loads((self.state_dir / 'state').read_bytes())

    def calls(self, cmd=None):
        out = []
        for line in (self.state_dir / 'calls.log').read_text().splitlines():
            name, now, tz, args = line.split('\t')
            if cmd is None or name == cmd:
                out.append({'cmd': name, 'now': int(now), 'TZ': None if tz == '<unset>' else tz,
                            'argv': args.split('\x1f') if args else []})
        return out

    def script(self, name='crt_x', account='uit', partition='normal', gres='gpu:2g.35gb:1', time='72:00:00',
               extra=(), fname=None):
        lines = ['#!/bin/bash', f'#SBATCH --job-name={name}']
        lines += [f'#SBATCH --{k}={v}' for k, v in (('account', account), ('partition', partition), ('gres', gres),
                                                    ('time', time)) if v is not None]
        lines += list(extra) + ['echo hi']
        p = self.jobs / (fname or f'job{len(list(self.jobs.iterdir()))}.slurm')
        p.write_text('\n'.join(lines) + '\n')
        return str(p)

    def env(self, **kw):
        e = {'PATH': f'{self.bin}:{SYSTEM_PATH}', 'HOME': str(self.root / 'home'), 'LANG': 'C', 'LC_ALL': 'C',
             'STUB_STATE': str(self.state_dir), 'STUB_ACCOUNT': self.acct, self.log_var: str(self.log),
             self.lock_var: str(self.lock), 'POLL': '300'}
        e.update(kw)
        return {k: v for k, v in e.items() if v is not None}

    def popen(self, *args, **env):
        return subprocess.Popen([BASH, self.wrapper, *args], env=self.env(**env), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def run(self, *args, **env):
        p = subprocess.run([BASH, self.wrapper, *args], env=self.env(**env), capture_output=True, text=True, timeout=120)
        assert not self.calls('scancel') and not self.calls('scontrol'), 'the wrapper must never cancel or change a job'
        return p

    def submitted(self):
        return [s for s in self.state()['sbatch'] if not s.get('failed')]

    def sleeps(self):
        return [c['argv'] for c in self.calls('sleep')]

    def log_lines(self):
        return [line.split('\t') for line in self.log.read_text().splitlines()] if self.log.exists() else []


@pytest.fixture(scope='module')
def stub_bin(tmp_path_factory):
    """The stand-ins, written once per module: macOS vets every new executable on its first run (about 1 s)."""
    d = tmp_path_factory.mktemp('slurm_stubs') / 'bin'
    write_stubs(d)
    return d


@pytest.fixture(params=sorted(ACCOUNTS))
def slurm(request, tmp_path, stub_bin):
    return Slurm(tmp_path, stub_bin, request.param)


def submitted_once(slurm, p):
    """The run submitted exactly one job; returns its sbatch record."""
    assert p.returncode == 0, p.stdout + p.stderr
    sub = slurm.submitted()
    assert len(sub) == 1, sub
    return sub[0]


def refused(slurm, p, why):
    assert p.returncode == 1, p.stdout + p.stderr
    assert 'REFUSED, nothing submitted' in p.stderr and why in p.stderr, p.stderr
    assert slurm.state()['sbatch'] == [] and not slurm.log.exists()


# ── The stand-in date itself ─────────────────────────────────

def test_date_stand_in_behaves_as_gnu_date_at_the_fake_clock(slurm):
    def date(*args, **env):
        p = subprocess.run([str(slurm.bin / 'date'), *args], env=slurm.env(**env), capture_output=True, text=True)
        return p.returncode, p.stdout.strip()
    now = T('2026-10-18T10:00:00')
    assert date('+%s') == (0, str(now))
    assert date('+%F %T') == (0, '2026-10-18 10:00:00')
    assert date('+%z', TZ='Asia/Ho_Chi_Minh') == (0, '+0700')
    assert date('+%z', TZ='Asia/Ho_Chi_Minh', STUB_NO_TZDATA='1') == (0, '+0000')
    assert date('-d', f'@{now + 120}', '+%H%M', TZ='Asia/Ho_Chi_Minh') == (0, '1002')
    assert date('-d', '-3 days', '+%Y-%m-%dT%H:%M:%S') == (0, '2026-10-15T10:00:00')
    assert date('-d', '-3 days', '+%Y-%m-%dT%H:%M:%S', STUB_LOCALTIME='UTC') == (0, '2026-10-15T03:00:00')
    assert date('-d', '2026-10-16T04:00:00', '+%s') == (0, str(T('2026-10-16T04:00:00')))
    for junk in ('Unknown', 'N/A', 'garbage'):        # (sacct / squeue junk that GNU date rejects too)
        assert date('-d', junk, '+%s')[0] == 1, junk


# ── Submitting, the log line ─────────────────────────────────

def test_submits_at_once_when_all_rules_hold(slurm):
    script = slurm.script()
    p = slurm.run(script)
    sub = submitted_once(slurm, p)
    real = os.path.realpath(script)
    assert sub['argv'] == ['--parsable', real] and sub['now'] == T('2026-10-18T10:00:00')
    assert slurm.sleeps() == []
    line = ['2026-10-18 10:00:00', '9000', '2026-10-18T10:00:00', real, '']
    assert slurm.log_lines() == [line]
    assert 'submitted: ' + '\t'.join(line) in p.stdout
    # the lock first, then the queue of the account, then sacct over the last 3 days
    cmds = [c['cmd'] for c in slurm.calls() if c['cmd'] not in ('date', 'id')]
    assert cmds == ['flock', 'flock-acquired', 'squeue', 'sacct', 'sbatch', 'squeue'], cmds
    assert slurm.calls('flock')[0]['argv'] == ['9']
    assert slurm.calls('sacct')[0]['argv'] == ['-u', slurm.acct, '-S', '2026-10-15T10:00:00', '-X', '-n', '-P', '-o',
                                               'Submit']


def test_log_appends_one_line_per_submission_and_creates_its_directory(slurm):
    assert not slurm.log.parent.exists()
    submitted_once(slurm, slurm.run(slurm.script(), '--export=ALL,PART=t1'))
    slurm.set(now='2026-10-18T12:00:00', queue=[QWEN], sacct=['2026-10-18T10:00:00'], next_id=9001)
    script = slurm.script(fname='job_m.slurm')
    assert slurm.run(script).returncode == 0
    lines = slurm.log_lines()
    assert [x[1] for x in lines] == ['9000', '9001'] and lines[0][4] == '--export=ALL,PART=t1' and lines[1][4] == ''
    assert lines[1][:4] == ['2026-10-18 12:00:00', '9001', '2026-10-18T12:00:00', os.path.realpath(script)]


def test_parsable_cluster_suffix_and_unknown_submit_time(slurm):
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN], sbatch_out='{jid};hpc', hide_new=True)
    submitted_once(slurm, slurm.run(slurm.script()))
    assert slurm.log_lines()[0][1:3] == ['9000', 'unknown']   # neither squeue -j nor sacct -j knows the job yet
    assert slurm.calls('sacct')[-1]['argv'] == ['-j', '9000', '-X', '-n', '-P', '-o', 'Submit']


def test_sbatch_failure_is_refused_without_a_log_line(slurm):
    p = slurm.run(slurm.script(), STUB_SBATCH_FAIL='1')
    assert p.returncode == 1 and 'REFUSED' in p.stderr and 'sbatch failed' in p.stderr, p.stderr
    assert slurm.submitted() == [] and not slurm.log.exists()


# ── Rule 1: fewer than 3 jobs of the account (any name) running or pending ──

def test_two_jobs_in_the_queue_do_not_block_only_the_gap_from_their_submit(slurm):
    crt = {'id': 101, 'name': 'crt_d0', 'submit': '2026-10-18T09:30:00', 'until': '2026-10-18T12:10:00'}
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, crt])
    p = slurm.run(slurm.script())
    sub = submitted_once(slurm, p)
    assert T('2026-10-18T10:31:00') <= sub['now'] <= T('2026-10-18T10:31:00') + 300, ict(sub['now'])
    assert f'jobs of {slurm.acct} are in the queue' not in p.stdout
    assert f'the last Submit of a job of {slurm.acct} was at 2026-10-18 09:30:00' in p.stdout


def test_three_jobs_of_any_name_block_until_one_leaves(slurm):
    jobs = [QWEN, {'id': 7, 'name': 'interactive', 'submit': '2026-10-18T08:00:00', 'until': '2026-10-18T10:20:00'},
            {'id': 8, 'name': 'qwen-serve-test', 'submit': '2026-10-18T08:00:00', 'until': '2026-10-18T10:40:00'}]
    slurm.set(now='2026-10-18T10:00:00', queue=jobs)
    p = slurm.run(slurm.script(), POLL='1200')
    sub = submitted_once(slurm, p)
    assert T('2026-10-18T10:20:00') <= sub['now'] < T('2026-10-18T10:40:00'), ict(sub['now'])
    assert (f'3 or more jobs of {slurm.acct} are in the queue: 100 qwen-serve RUNNING;7 interactive RUNNING;'
            '8 qwen-serve-test RUNNING;') in p.stdout


def test_pending_jobs_count_toward_the_three(slurm):
    run = {'id': 101, 'name': 'crt_d1_conv', 'submit': '2026-10-18T07:00:00', 'until': '2026-10-18T12:00:00'}
    pend = {'id': 102, 'name': 'crt_d1', 'submit': '2026-10-18T09:59:30', 'until': '2026-10-18T10:04:00'}   # stub: pending
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, run, pend])                                    # for its first 60 s
    p = slurm.run(slurm.script(), POLL='1200')
    sub = submitted_once(slurm, p)
    assert f'3 or more jobs of {slurm.acct} are in the queue' in p.stdout and '102 crt_d1 PENDING' in p.stdout
    assert sub['now'] >= T('2026-10-18T09:59:30') + 3660                     # then the gap from its Submit


def test_qwen_serve_jobs_count_like_any_other(slurm):
    rot = {'id': 103, 'name': 'qwen-serve', 'submit': '2026-10-17T04:00:00', 'until': '2026-10-21T04:30:00'}
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, rot])
    assert submitted_once(slurm, slurm.run(slurm.script()))['now'] == T('2026-10-18T10:00:00')   # two: no wait
    third = {'id': 104, 'name': 'qwen-serve', 'submit': '2026-10-17T05:00:00', 'until': '2026-10-18T10:30:00'}
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, rot, third])
    p = slurm.run(slurm.script(), POLL='600')
    assert submitted_once(slurm, p)['now'] >= T('2026-10-18T10:30:00') and '3 or more jobs' in p.stdout


# ── Rule 3: >= 3660 s since the last Submit ──────────────────

def test_gap_from_a_finished_job_known_only_to_sacct(slurm):
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN], sacct=['2026-10-18T09:30:00'])
    p = slurm.run(slurm.script())
    sub = submitted_once(slurm, p)
    assert sub['now'] == T('2026-10-18T10:35:00') and slurm.sleeps() == [['300']] * 7, ict(sub['now'])
    assert f'the last Submit of a job of {slurm.acct} was at 2026-10-18 09:30:00, 1800 s ago (< 3660 s)' in p.stdout


@pytest.mark.parametrize('ago, waits', [(3660, False), (3659, True), (3661, False), (60, True)])
def test_gap_boundary(slurm, ago, waits):
    now = T('2026-10-18T10:00:00')
    slurm.set(now=now, queue=[QWEN], sacct=[now - ago])
    sub = submitted_once(slurm, slurm.run(slurm.script(), POLL='1800'))
    assert (sub['now'] > now) == waits
    assert sub['now'] - (now - ago) >= 3660


def test_gap_counts_the_squeue_submit_time_when_sacct_lags(slurm):
    lagging = {'id': 104, 'name': 'qwen-serve', 'submit': '2026-10-18T09:50:00', 'until': '2026-10-25T00:00:00'}
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, lagging], lag=['2026-10-18T09:50:00'])
    sub = submitted_once(slurm, slurm.run(slurm.script(), POLL='600'))
    assert sub['now'] == T('2026-10-18T11:00:00'), ict(sub['now'])
    assert ['-d', '2026-10-18T09:50:00', '+%s'] in [c['argv'] for c in slurm.calls('date')]   # read from squeue %V


def test_qwen_serve_submits_count_for_the_gap(slurm):
    rot = {'id': 105, 'name': 'qwen-serve', 'submit': '2026-10-18T09:40:00', 'until': '2026-10-22T04:30:00'}
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, rot])
    sub = submitted_once(slurm, slurm.run(slurm.script(), POLL='600'))
    assert sub['now'] == T('2026-10-18T10:50:00'), ict(sub['now'])


# ── Rule 2: no submission 03:00-04:10 Asia/Ho_Chi_Minh ───────

def test_window_then_the_0400_qwen_serve_rotation(slurm):
    rot = {'id': 106, 'name': 'qwen-serve', 'submit': '2026-10-18T04:00:00', 'until': '2026-10-22T04:30:00'}
    slurm.set(now='2026-10-18T03:30:00', queue=[QWEN, rot])
    p = slurm.run(slurm.script(), POLL='600')
    sub = submitted_once(slurm, p)
    assert sub['now'] == T('2026-10-18T05:10:00'), ict(sub['now'])
    assert 'the 03:00-04:10 window (Asia/Ho_Chi_Minh)' in p.stdout and 'was at 2026-10-18 04:00:00' in p.stdout
    assert slurm.sleeps()[0] == ['600']


@pytest.mark.parametrize('start, waits', [
    ('2026-10-18T02:57:00', False), ('2026-10-18T02:57:59', False), ('2026-10-18T02:58:00', True),
    ('2026-10-18T02:58:30', True), ('2026-10-18T03:00:00', True), ('2026-10-18T04:09:59', True),
    ('2026-10-18T04:10:00', False), ('2026-10-18T04:30:00', False)])
def test_window_boundaries(slurm, start, waits):
    slurm.set(now=start, queue=[QWEN])
    p = slurm.run(slurm.script(), POLL='900')
    sub = submitted_once(slurm, p)
    assert (sub['now'] > T(start)) == waits, ict(sub['now'])
    if waits:
        assert sub['now'] >= T('2026-10-18T04:10:00') and 'the 03:00-04:10 window' in p.stdout
        assert not ('0258' <= datetime.fromtimestamp(sub['now'], ICT).strftime('%H%M') < '0410')


@pytest.mark.parametrize('localtime', ['UTC', 'Europe/Paris'])
def test_window_is_in_asia_ho_chi_minh_whatever_the_node_zone(slurm, localtime):
    # 03:30 in Ho Chi Minh City is 20:30 UTC (22:30 in Paris) of the day before
    slurm.set(now='2026-10-18T03:30:00', queue=[QWEN])
    p = slurm.run(slurm.script(), STUB_LOCALTIME=localtime, POLL='1200')
    sub = submitted_once(slurm, p)
    assert sub['now'] == T('2026-10-18T04:10:00') and 'the 03:00-04:10 window' in p.stdout, ict(sub['now'])
    window = [c for c in slurm.calls('date') if c['argv'][-1:] == ['+%H%M']]
    assert window and all(c['TZ'] == 'Asia/Ho_Chi_Minh' for c in window)
    # sacct -S is written in the node's zone, as sacct reads it
    first = slurm.calls('sacct')[0]
    assert first['argv'][3] == datetime.fromtimestamp(first['now'] - 3 * 86400, ZoneInfo(localtime)).strftime(
        '%Y-%m-%dT%H:%M:%S')


def test_gap_is_exact_when_the_node_zone_is_not_ict(slurm):
    now = T('2026-10-18T10:00:00')
    slurm.set(now=now, queue=[QWEN], sacct=[now - 3600])
    sub = submitted_once(slurm, slurm.run(slurm.script(), STUB_LOCALTIME='UTC', POLL='60'))
    assert sub['now'] == now + 60


# ── Refusals: squeue / sacct failures and unparsable output ──

@pytest.mark.parametrize('env, state', [
    ({'SQUEUE_FAIL': '1'}, {}),
    ({'SACCT_FAIL': '1'}, {}),
    ({}, {'squeue_extra': ['garbage line']}),
    ({}, {'squeue_extra': ['1|a|RUNNING']}),
    ({}, {'squeue_extra': ['1|a|b|RUNNING|2026-10-18T09:00:00']}),
    ({}, {'squeue_extra': ['1|qwen-serve|RUNNING|N/A']}),
    ({}, {'sacct_extra': ['Unknown']}),
    ({}, {'sacct_extra': ['garbage']}),
], ids=['squeue_fails', 'sacct_fails', 'squeue_junk', 'squeue_3_fields', 'squeue_5_fields', 'squeue_bad_time',
        'sacct_unknown', 'sacct_junk'])
def test_refuses_when_squeue_or_sacct_fail_or_print_junk(slurm, env, state):
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN], **state)
    p = slurm.run(slurm.script(), **env)
    refused(slurm, p, 'squeue or sacct failed, or printed a line that could not be parsed')


def test_blank_lines_from_squeue_and_sacct_are_not_junk(slurm):
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN], squeue_extra=[''], sacct_extra=[''])
    submitted_once(slurm, slurm.run(slurm.script()))


def test_refuses_when_sacct_fails_while_waiting(slurm):
    # three jobs are queued until 10:10; once one has left, sacct prints junk: the wrapper stops instead of submitting
    crt = {'id': 101, 'name': 'crt_d0', 'submit': '2026-10-18T09:30:00', 'until': '2026-10-18T10:10:00'}
    crt2 = {'id': 102, 'name': 'crt_d2', 'submit': '2026-10-18T08:30:00', 'until': '2026-10-18T12:00:00'}
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN, crt, crt2], sacct_extra=['Unknown'])
    refused(slurm, slurm.run(slurm.script()), 'squeue or sacct failed')
    assert len(slurm.sleeps()) == 2


# ── Refusals: the account's job rules in the script ──────────

@pytest.mark.parametrize('kw, why', [
    ({'account': 'other'}, 'lacks #SBATCH --account=uit'),
    ({'account': None}, 'lacks #SBATCH --account=uit'),
    ({'partition': 'debug'}, 'lacks #SBATCH --account=uit, --partition=normal'),
    ({'partition': None}, '--partition=normal'),
    ({'gres': 'gpu:1'}, '--gres=gpu:2g.35gb:1'),
    ({'gres': 'gpu:2g.35gb:2'}, '--gres=gpu:2g.35gb:1'),
    ({'gres': None}, '--gres=gpu:2g.35gb:1'),
    ({'time': '96:00:00'}, '--time=96:00:00 is not HH:MM:SS <= 72:00:00'),
    ({'time': '72:00:01'}, 'is not HH:MM:SS <= 72:00:00'),
    ({'time': '3-00:00:00'}, 'is not HH:MM:SS <= 72:00:00'),
    ({'time': '4320'}, 'is not HH:MM:SS <= 72:00:00'),
    ({'time': '10:60:00'}, 'is not HH:MM:SS <= 72:00:00'),
    ({'time': None}, '--time=? is not HH:MM:SS'),
    ({'extra': ['#SBATCH --array=0-3']}, 'can run more than one task at a time'),
    ({'extra': ['#SBATCH --array=0-3%2']}, 'can run more than one task at a time'),
    ({'extra': ['#SBATCH --array=0-3%10']}, 'can run more than one task at a time'),
    ({'name': 'qwen-serve'}, 'is named qwen-serve'),
    ({'extra': ['#SBATCH -t 100:00:00']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH -A other']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH -p debug']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH -a 0-3']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH -J qwen-serve']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH --time 99:00:00']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH --partition debug']}, 'in a form this check does not read'),
    ({'extra': ['#SBATCH --time=96:00:00']}, 'is not HH:MM:SS <= 72:00:00'),   # the last --time wins, as in sbatch
    ({'extra': ['#SBATCH --account=other']}, 'lacks #SBATCH --account=uit'),
], ids=lambda x: x if isinstance(x, str) else repr(x))
def test_refuses_scripts_that_break_the_job_rules(slurm, kw, why):
    refused(slurm, slurm.run(slurm.script(**kw)), why)
    assert not slurm.calls('squeue') and not slurm.calls('flock')   # refused before the lock and the queue


@pytest.mark.parametrize('kw', [{'time': '1:00:00'}, {'time': '00:30:00'}, {'time': '071:59:59'},
                                {'extra': ['#SBATCH --array=0-19%1']}, {'extra': ['#SBATCH --time=06:00:00']},
                                {'extra': ['#SBATCH --mem=32G', '#SBATCH --cpus-per-task=8']}], ids=repr)
def test_accepts_scripts_within_the_rules(slurm, kw):
    submitted_once(slurm, slurm.run(slurm.script(**kw)))


def test_repository_job_scripts_pass_the_rule_checks(slurm):
    names = sorted(f for f in os.listdir(BENCH) if f.startswith('job_') and f.endswith('.slurm'))
    assert names
    for f in names:
        slurm.set(now='2026-10-18T10:00:00', queue=[QWEN])
        p = slurm.run(os.path.join(BENCH, f))
        assert p.returncode == 0 and len(slurm.submitted()) == 1, (f, p.stderr)


# ── Refusals: extra sbatch arguments ─────────────────────────

@pytest.mark.parametrize('args', [
    ['--time=1:00:00'], ['--time-min=10'], ['--array=0-1'], ['--job-name=qwen-serve'], ['--account=x'],
    ['--partition=debug'], ['--gres=gpu:1'], ['--gres-flags=disable-binding'],
    ['-t', '5'], ['-J', 'qwen-serve'], ['-p', 'debug'], ['-A', 'x'], ['-a', '0-3'], ['-t5'], ['-pdebug'],
    # abbreviations: sbatch, like getopt_long, accepts any unambiguous prefix of a long option
    ['--part=debug'], ['--acco=x'], ['--arr=0-3'], ['--job-n=qwen-serve'], ['--ti=99:00:00'], ['--gre=gpu:1'],
], ids=' '.join)
def test_refuses_extra_args_that_set_a_job_rule(slurm, args):
    refused(slurm, slurm.run(slurm.script(), *args), 'the extra argument')


@pytest.mark.parametrize('args', [
    ['--export', 'ALL,PART=t1'],     # the value would be taken as the script
    ['OTHER.slurm'],                 # a second script
    ['--hold'], ['--exclusive'], ['-Q'], ['-n4'], ['-dafterok:123'], ['--=x'], ['-'],   # not --name=value
], ids=' '.join)
def test_refuses_extra_args_that_are_not_name_value_options(slurm, args):
    refused(slurm, slurm.run(slurm.script(), *args), 'the extra argument')


def test_passes_name_value_options_to_sbatch_in_order(slurm):
    script = slurm.script()
    sub = submitted_once(slurm, slurm.run(script, '--export=ALL,PART=t1', '--mail-type=END'))
    assert sub['argv'] == ['--parsable', '--export=ALL,PART=t1', '--mail-type=END', os.path.realpath(script)]
    assert slurm.log_lines()[0][4] == '--export=ALL,PART=t1 --mail-type=END'


# ── Refusals: user, script, zone data ────────────────────────

def test_refuses_when_not_the_wrappers_account(slurm):
    other = 'ceai' if slurm.acct == 'nvquoc' else 'nvquoc'
    refused(slurm, slurm.run(slurm.script(), STUB_USER=other), f'this is {other}, not {slurm.acct}')
    assert not slurm.calls('squeue') and not slurm.calls('sacct')


def test_refuses_a_missing_script_and_no_arguments(slurm):
    refused(slurm, slurm.run(str(slurm.jobs / 'nope.slurm')), 'no script')
    refused(slurm, slurm.run(), f'usage: bash {slurm.name}.sh <script.slurm>')
    refused(slurm, slurm.run(str(slurm.jobs)), 'no script')


def test_refuses_without_zone_data(slurm):
    refused(slurm, slurm.run(slurm.script(), STUB_NO_TZDATA='1'), 'no zone data for Asia/Ho_Chi_Minh')


def test_zone_check_does_not_depend_on_the_callers_tz(slurm):
    assert submitted_once(slurm, slurm.run(slurm.script(), TZ='UTC'))['now'] == T('2026-10-18T10:00:00')


# ── SBATCH_* variables, POLL ─────────────────────────────────

def test_sbatch_variables_are_unset(slurm):
    env = {'SBATCH_PARTITION': 'large', 'SBATCH_JOB_NAME': 'qwen-serve', 'SBATCH_TIMELIMIT': '7-00:00:00',
           'SBATCH_ACCOUNT': 'other', 'SBATCH_ARRAY_INX': '0-9'}
    p = slurm.run(slurm.script(), **env)
    assert submitted_once(slurm, p)['env'] == {}
    for k, v in env.items():
        assert f'{slurm.name}: ignoring {k}={v}' in p.stdout


def test_poll_defaults_to_300_s(slurm):
    slurm.set(now='2026-10-18T10:00:00', queue=[QWEN], sacct=['2026-10-18T09:00:00'])
    sub = submitted_once(slurm, slurm.run(slurm.script(), POLL=None))
    assert slurm.sleeps() == [['300']] and sub['now'] == T('2026-10-18T10:05:00')


# ── The lock ─────────────────────────────────────────────────

def test_waits_for_the_lock_of_another_wrapper(slurm):
    with open(slurm.lock, 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = slurm.popen(slurm.script())
        deadline = time.time() + 20
        while not slurm.calls('flock') and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        assert proc.poll() is None, proc.communicate()
        assert slurm.calls('flock') and not slurm.calls('flock-acquired')
        assert not slurm.calls('squeue') and slurm.submitted() == []
    out, err = proc.communicate(timeout=60)   # the lock was released on leaving the with block
    assert proc.returncode == 0, out + err
    assert len(slurm.submitted()) == 1


def test_refuses_when_the_lock_cannot_be_opened(slurm):
    p = slurm.run(slurm.script(), **{slurm.lock_var: str(slurm.root / 'no' / 'such' / 'dir' / 'lock')})
    refused(slurm, p, 'cannot lock')


def test_two_wrappers_never_both_submit_into_an_empty_queue(slurm):
    procs = [slurm.popen(slurm.script(name=n), POLL='1200') for n in ('crt_a', 'crt_b')]
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, out + err
    first, second = slurm.submitted()
    assert second['now'] - first['now'] >= 3660, (ict(first['now']), ict(second['now']))
    assert len(slurm.log_lines()) == 2


# ── The ceai wrapper ─────────────────────────────────────────

def test_ceai_wrapper_is_the_nvquoc_wrapper_with_its_account_log_and_lock():
    def body(name):
        text = open(os.path.join(BENCH, name)).read()
        text = text[text.index('\nUSR='):]
        for a, b in (('_transfer_logs/submit_ceai.log', '_transfer_logs/submit.log'), ('submit_nvquoc', 'submit_X'),
                     ('submit_ceai', 'submit_X'), ('SUBMIT_NVQUOC', 'SUBMIT_X'),
                     ('SUBMIT_CEAI', 'SUBMIT_X'), ('USR=nvquoc', 'USR=X'), ('USR=ceai', 'USR=X'),
                     ('/datastore/nvquoc/crt/', '/datastore/X/crt/'), ('/datastore/ceai/crt/', '/datastore/X/crt/')):
            text = text.replace(a, b)
        return text
    assert body('submit_ceai.sh') == body('submit_nvquoc.sh')
    ceai = open(os.path.join(BENCH, 'submit_ceai.sh')).read()
    assert '\nUSR=ceai\nGAP=3660\nMAXJOBS=3\n' in ceai
    assert 'LOG=${SUBMIT_CEAI_LOG:-/datastore/ceai/crt/_transfer_logs/submit_ceai.log}' in ceai
    assert 'LOCK=${SUBMIT_CEAI_LOCK:-/tmp/submit_ceai.lock}' in ceai
    assert 'nvquoc' not in ceai[ceai.index('\nUSR='):].replace('test_submit_nvquoc.py', '')
