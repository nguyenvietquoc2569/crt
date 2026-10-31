"""Tests of mvc/tools/tunnel_supervise.sh against a fake cloudflared (a bash script that logs its start, its signals and
its process group) and fake local and public health endpoints (http.server on free 127.0.0.1 ports), with short
intervals: the restart on exit, the end-to-end watchdog (only while the local server is up, only on the tunnel's 530),
the cap and its rolling window, the pause after a restart, the stop by signal, by Ctrl-C and by closing the terminal,
and that no other cloudflared is ever signalled."""
import fcntl
import os
import re
import shlex
import signal
import subprocess
import termios
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'tunnel_supervise.sh'

FAKE_CF = r'''#!/bin/bash
# fake cloudflared: logs "start <args> pgid <pgid>", each SIGTERM / SIGINT; exits after $FAKE_TERMS SIGTERMs (default
# 1), or with status 3 when the file $FAKE_CRASH appears; writes one line to its terminal, if it has one, as
# cloudflared logs to the tab
log=${FAKE_LOG:?}
echo "start $* pgid $(ps -o pgid= -p $$ | tr -d ' ') pid $$" >> "$log"
{ echo "fake cloudflared $$ up" > /dev/tty; } 2>/dev/null
echo "tty ok pid $$" >> "$log"
n=0
on_term() { n=$((n + 1)); echo "TERM $n pid $$" >> "$log"; [ $n -ge "${FAKE_TERMS:-1}" ] && exit 0; }
trap on_term TERM
trap 'echo "INT pid $$" >> "$log"' INT
while :; do
  if [ -n "${FAKE_CRASH:-}" ] && [ -e "$FAKE_CRASH" ]; then rm -f "$FAKE_CRASH"; echo "crash pid $$" >> "$log"; exit 3; fi
  sleep 0.05
done
'''
LINE = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ supervise (\S+): \S')


class Health:
    """A health endpoint on 127.0.0.1 answering `code`: a status, or a list of them served in turn from its first
    (from each assignment on); 0 closes the connection without an answer (curl's 000). Records each request's path and
    header names, and its time and code."""

    def __init__(self, code=200):
        self.lock, self.seen, self.served = threading.Lock(), [], []
        self.code = code
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.path, sorted(k.lower() for k in self.headers.keys())))
                with outer.lock:
                    codes = outer.code if isinstance(outer.code, list) else [outer.code]
                    code = codes[outer.k % len(codes)]
                    outer.k += 1
                    outer.served.append((time.monotonic(), code))
                if code == 0:
                    self.close_connection = True
                    return
                body = b'{"ok": true}' if code == 200 else b'error code: 1033'
                self.send_response(code)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/api/health'
        threading.Thread(target=self.httpd.serve_forever, kwargs=dict(poll_interval=0.05), daemon=True).start()

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, code):
        with self.lock:
            self._code, self.k = code, 0

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def wait_until(pred, timeout=10.0):
    t = time.monotonic()
    while not pred():
        if time.monotonic() - t > timeout:
            raise AssertionError('condition not reached')
        time.sleep(0.02)


class Setup:
    """A fake cloudflared, a config, local and public endpoints and the environment of short intervals."""

    def __init__(self, tmp_path, **env):
        self.dir = tmp_path
        self.cf = tmp_path / 'bin' / 'cloudflared'
        self.cf.parent.mkdir()
        self.cf.write_text(FAKE_CF)
        self.cf.chmod(0o755)
        self.cfg = tmp_path / 'mos.yml'
        self.cfg.write_text('tunnel: fake\n')
        self.log = tmp_path / 'cf.log'
        self.local, self.public = Health(), Health()
        self.env = dict(os.environ, CLOUDFLARED=str(self.cf), FAKE_LOG=str(self.log), WATCH_INTERVAL_S='0.2',
                        WATCH_FAILS='3', WATCH_GRACE_S='0.5', WATCH_TIMEOUT_S='2', WATCH_MAX_PER_HOUR='2',
                        WATCH_RESTART_S='0.3', WATCH_STOP_S='1', WATCH_TICK_S='0.05')
        self.env.update(env)
        self.decoys = []

    def args(self):
        return ['bash', str(SCRIPT), str(self.cfg), self.local.url, self.public.url]

    def decoy(self, cfg):
        """Another cloudflared (another tunnel, or one started by hand): must never be signalled."""
        log = self.dir / f'decoy{len(self.decoys)}.log'
        p = subprocess.Popen([str(self.cf), 'tunnel', '--config', str(cfg), '--no-autoupdate', 'run'],
                             env=dict(self.env, FAKE_LOG=str(log)), start_new_session=True)
        wait_until(lambda: log.exists())
        self.decoys.append((p, log))
        return p

    def lines(self, kind):
        return [ln for ln in self.log.read_text().splitlines() if ln.startswith(kind)] if self.log.exists() else []

    def close(self):
        for p, _ in self.decoys:
            p.kill()
            p.wait()
        self.local.close()
        self.public.close()


@pytest.fixture
def setup(tmp_path):
    made = []

    def make(**env):
        s = Setup(tmp_path, **env)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()
    subprocess.run(['pkill', '-KILL', '-f', str(tmp_path / 'bin' / 'cloudflared')])   # only this test's fakes


def run(s):
    out = open(s.dir / 'out.txt', 'w')
    p = subprocess.Popen(s.args(), env=s.env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return p, out, lambda: (s.dir / 'out.txt').read_text()


def test_watchdog_restarts_the_tunnel_only_when_the_public_url_fails_with_the_local_server_up(setup):
    s = setup()
    (s.dir / 'exam-dev.yml').write_text('tunnel: other\n')
    other = s.decoy(s.dir / 'exam-dev.yml')                       # another tunnel of this Mac
    same = s.decoy(s.cfg)                                         # one already running with this config
    p, out, text = run(s)
    try:
        wait_until(lambda: len(s.lines('start')) == 1 and len(s.public.seen) >= 2)
        assert f'note: another cloudflared runs with this config (pid {same.pid}); it is left alone' in text()
        assert s.lines('TERM') == []
        s.local.code = 503                                        # the server's own problem: the tunnel is left alone
        time.sleep(0.5)
        s.public.code, n0 = 530, len(s.public.seen)
        time.sleep(1.5)                                           # about 7 probes of the local server only
        assert s.lines('TERM') == [] and len(s.public.seen) == n0
        assert text().count("the server's problem, the tunnel is left alone") == 1
        s.local.code = 200                                        # server up, public 530: restart after 3 failures
        wait_until(lambda: len(s.lines('TERM')) == 1)
        wait_until(lambda: len(s.lines('start')) == 2)
        wait_until(lambda: len(s.lines('TERM')) == 2)             # still 530 after the grace: the second restart
        wait_until(lambda: len(s.lines('start')) == 3)
        wait_until(lambda: 'cap of 2 restarts per 3600s reached' in text())      # the third: over the cap
        time.sleep(1.5)
        assert len(s.lines('TERM')) == 2 and len(s.lines('start')) == 3 and text().count('cap of 2') == 1
        s.public.code = 200
        wait_until(lambda: 'answers 200 again' in text().split('cap of 2')[1])
        for d, log in s.decoys:                                   # nothing else was signalled
            assert d.poll() is None and 'TERM' not in log.read_text() and 'INT' not in log.read_text()
        assert other.poll() is None
        p.terminate()                                             # SIGTERM: stop cloudflared, then exit
        assert p.wait(10) == 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        out.close()
    own = [ln.split()[-1] for ln in s.lines('start')]             # the pids of the cloudflareds it started
    term_pids = [ln.split()[3] for ln in s.lines('TERM')]
    assert set(term_pids) == set(own) and len(term_pids) == 3     # each own one once (the last at the stop)
    lines = text().splitlines()
    assert all(LINE.match(ln) and LINE.match(ln).group(1) == 'mos' for ln in lines), lines   # only its own lines
    restarts = [ln for ln in lines if ': restart ' in ln]
    assert len(restarts) == 2 and 'restart 1/2 of the last 3600s: SIGTERM to cloudflared' in restarts[0]
    assert sum(f'public {s.public.url} answers 530 while the local one is ok (' in ln for ln in lines) == 9
    assert 'local ' + s.local.url + ' answers 200 again' in text() and lines[-1].endswith('stopped')
    paths = {path for path, _ in s.public.seen + s.local.seen}
    headers = {h for _, hs in s.public.seen + s.local.seen for h in hs}
    assert paths == {'/api/health'} and headers <= {'host', 'user-agent', 'accept'}   # no credentials


def gap_after_restart(served):
    """The time from the probe of the first 3 public 530s in a row to the next probe."""
    codes = [c for _, c in served]
    i = next(i for i in range(2, len(codes)) if codes[i - 2:i + 1] == [530] * 3)
    return served[i + 1][0] - served[i][0]


def test_only_the_tunnels_530_counts_and_any_other_answer_resets_the_count(setup):
    """No answer (000), a 403 (Cloudflare's own rules) or another status is not the tunnel's error 1033: logged once
    per streak, never a restart, and the count of 530s in a row starts again; a 200 between 530s resets it too. Then
    3 x 530: one restart, after which the probes pause for WATCH_GRACE_S."""
    s = setup(WATCH_GRACE_S='1')
    p, out, text = run(s)
    try:
        wait_until(lambda: len(s.lines('start')) == 1 and len(s.public.seen) >= 2)
        url = s.public.url
        for code in (403, 0):
            s.public.code, n0 = code, len(s.public.served)
            wait_until(lambda: len(s.public.served) >= n0 + 6)
        assert text().count(f'public {url} answers 403, not 530: no restart') == 1
        assert text().count(f'public {url} answers 000, not 530: no restart') == 1
        for code in ([403, 530, 530], [200, 530, 530], [0, 530, 530], [502, 530, 530]):   # 2 x 530, then a reset
            s.public.code, n0 = code, len(s.public.served)
            wait_until(lambda: len(s.public.served) >= n0 + 9)
        wait_until(lambda: text().count(f'public {url} answers 530 while the local one is ok (2/3)') >= 12)
        assert s.lines('TERM') == [] and len(s.lines('start')) == 1
        assert text().count(f'public {url} answers 530 while the local one is ok (3/3)') == 0
        s.public.code = 530
        wait_until(lambda: len(s.lines('TERM')) == 1 and len(s.lines('start')) == 2)
        n0 = len(s.public.served)
        wait_until(lambda: len(s.public.served) >= n0 + 1)                    # the first probe after the restart
        p.terminate()
        assert p.wait(10) == 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        out.close()
    assert gap_after_restart(s.public.served) >= 1.0 + 0.3                  # WATCH_GRACE_S + WATCH_RESTART_S
    assert text().count(': restart ') == 1 and len(s.lines('TERM')) == 2    # the second: at the stop


def test_the_cap_is_a_rolling_window(setup):
    """WATCH_MAX_PER_HOUR=1 per WATCH_WINDOW_S=3 s, the public URL always 530: one restart, the cap (one line), and a
    second restart once the first is older than the window."""
    s = setup(WATCH_MAX_PER_HOUR='1', WATCH_WINDOW_S='3')
    s.public.code = 530
    p, out, text = run(s)
    try:
        wait_until(lambda: text().count(': restart 1/1 of the last 3s') == 2, 15)
        assert text().index('cap of 1 restarts per 3s reached') < text().rindex(': restart 1/1')
        assert text().count('cap of 1 restarts per 3s reached') == 1
        p.terminate()
        assert p.wait(10) == 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        out.close()


def test_restart_on_exit_and_sigkill_when_sigterm_is_ignored(setup):
    s = setup(FAKE_TERMS='99', FAKE_CRASH='')
    crash = s.dir / 'crash'
    s.env['FAKE_CRASH'] = str(crash)
    p, out, text = run(s)
    try:
        wait_until(lambda: len(s.lines('start')) == 1 and len(s.public.seen) >= 1)
        crash.write_text('')
        wait_until(lambda: len(s.lines('start')) == 2)
        assert 'exited with status 3: starting it again in 0.3s' in text()
        s.public.code = 530
        wait_until(lambda: 'still runs 1s after SIGTERM: SIGKILL' in text())
        wait_until(lambda: len(s.lines('start')) == 3)
        s.public.code = 200
        wait_until(lambda: 'answers 200 again' in text())
        p.send_signal(signal.SIGHUP)                              # the tab closed: SIGTERM, SIGKILL 1 s later, exit
        assert p.wait(10) == 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        out.close()
    assert text().count('SIGKILL') == 2 and text().splitlines()[-1].endswith('stopped')
    wait_until(lambda: subprocess.run(['pgrep', '-f', str(s.cf)], capture_output=True).returncode == 1, 5)


def in_a_terminal(s, tostop=False, reader=True, args=None):
    """The script in a pty, as in a Terminal tab (session leader, the pty its controlling terminal, in its foreground
    process group); with tostop, stty tostop; with reader, a thread reads the terminal (none when the test closes the
    master end: on macOS a read in progress would keep it open). -> the process, the master end, what was read."""
    master, slave = os.openpty()
    if tostop:
        attrs = termios.tcgetattr(slave)
        attrs[3] |= termios.TOSTOP
        termios.tcsetattr(slave, termios.TCSANOW, attrs)

    def tty():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    p = subprocess.Popen(args or s.args(), env=s.env, stdin=slave, stdout=slave, stderr=slave, preexec_fn=tty)
    os.close(slave)
    got = []

    def read():
        while True:
            try:
                b = os.read(master, 4096)
            except OSError:
                return
            if not b:
                return
            got.append(b)
    if reader:
        threading.Thread(target=read, daemon=True).start()
    return p, master, got


def test_ctrl_c_in_the_terminal_reaches_cloudflared_once(setup):
    """In a terminal (a pty, the script in its foreground process group) a Ctrl-C reaches cloudflared only through the
    script, as one SIGTERM (its grace period); a second Ctrl-C sends the second, after which cloudflared stops. With
    stty tostop, cloudflared (in its own, background, process group) still writes to the terminal."""
    s = setup(FAKE_TERMS='2', WATCH_STOP_S='30')
    p, master, got = in_a_terminal(s, tostop=True)
    try:
        wait_until(lambda: len(s.lines('start')) == 1 and len(s.public.seen) >= 1)
        assert len(s.lines('tty ok')) == 1                                    # not stopped by SIGTTOU
        start = s.lines('start')[0].split()
        assert start[start.index('pgid') + 1] != str(os.getpgid(p.pid))       # its own process group
        os.write(master, b'\x03')
        wait_until(lambda: len(s.lines('TERM')) == 1)
        time.sleep(0.5)
        assert s.lines('INT') == [] and len(s.lines('TERM')) == 1 and p.poll() is None   # in its grace period
        os.write(master, b'\x03')
        assert p.wait(10) == 0
        assert s.lines('INT') == [] and len(s.lines('TERM')) == 2
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(0.1)
        os.close(master)
    text = b''.join(got).decode()
    assert 'stopping: SIGTERM to cloudflared' in text and 'second signal: SIGTERM again' in text
    assert 'Terminated' not in text and 'Killed' not in text and f'fake cloudflared {start[-1]} up' in text


def test_ctrl_c_with_the_output_piped_still_stops_cloudflared(setup):
    """Run as `... | cat` (or tee) in a terminal, a Ctrl-C also ends cat: the script's next line then meets a closed
    pipe, which must not end the script before it has sent cloudflared its SIGTERM and seen it stop."""
    s = setup()
    p, master, got = in_a_terminal(s, args=['bash', '-c', shlex.join(s.args()) + ' | cat'])
    try:
        wait_until(lambda: len(s.lines('start')) == 1 and len(s.public.seen) >= 1)
        pid = int(s.lines('start')[0].split()[-1])
        os.write(master, b'\x03')
        wait_until(lambda: len(s.lines('TERM')) == 1)
        p.wait(10)
        mine = ['pgrep', '-f', f'tunnel_supervise.sh {s.cfg}']                  # this test's script only
        wait_until(lambda: subprocess.run(mine, capture_output=True).returncode == 1, 5)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(0.1)
        os.close(master)
    assert s.lines('INT') == [] and len(s.lines('TERM')) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_closing_the_terminal_gives_cloudflared_its_grace_period(setup):
    """The tab closed (the pty's master end closed: SIGHUP, and every later write to the terminal fails): one SIGTERM to
    cloudflared, which keeps its grace period (WATCH_STOP_S) before the SIGKILL, then the script exits."""
    s = setup(FAKE_TERMS='2', WATCH_STOP_S='2')
    p, master, _ = in_a_terminal(s, reader=False)
    try:
        wait_until(lambda: len(s.lines('start')) == 1 and len(s.public.seen) >= 2)
        pid = int(s.lines('start')[0].split()[-1])
        os.close(master)
        master = None
        wait_until(lambda: len(s.lines('TERM')) == 1)
        time.sleep(1)
        os.kill(pid, 0)                                                       # still in its grace period
        assert p.wait(10) == 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        if master is not None:
            os.close(master)
    assert len(s.lines('TERM')) == 1 and len(s.lines('start')) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)                                                       # SIGKILL after the grace period


@pytest.mark.parametrize('args, msg', [
    ([], 'usage:'),
    (['/nonexistent.yml', 'http://127.0.0.1:1/x', 'https://example.invalid/x'], 'no such config file'),
    (['CFG', '127.0.0.1:8790/api/info', 'https://example.invalid/x'], 'not an http(s) URL'),
])
def test_usage_errors(tmp_path, args, msg):
    cfg = tmp_path / 'c.yml'
    cfg.write_text('')
    r = subprocess.run(['bash', str(SCRIPT)] + [str(cfg) if a == 'CFG' else a for a in args], capture_output=True,
                       text=True, env=dict(os.environ, CLOUDFLARED='/usr/bin/true'))
    assert r.returncode == 2 and msg in r.stderr
