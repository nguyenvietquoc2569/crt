#!/bin/bash
# tunnel_supervise.sh CONFIG LOCAL_HEALTH_URL PUBLIC_HEALTH_URL: runs one Cloudflare tunnel of this Mac
# (cloudflared tunnel --config CONFIG --no-autoupdate run, the command of the Terminal tabs) and keeps it working end to
# end. Run it in the tunnel's Terminal tab instead of that command:
#   mvc/tools/tunnel_supervise.sh ~/.cloudflared/mos.yml http://127.0.0.1:8790/api/info https://mos.betaschool.edu.vn/api/info
#   mvc/tools/tunnel_supervise.sh ~/.cloudflared/vc.yml http://127.0.0.1:8791/api/health https://vc.betaschool.edu.vn/api/health
# Why (31 Oct 2026): after a network event cloudflared can stay alive and silent while Cloudflare has no working
# connection of the tunnel (the public URL answers 530, error 1033); its own readiness still says ready, and only a
# restart helps. So the check goes through the public URL.
# - cloudflared is started here, and started again 10 s after it exits (WATCH_RESTART_S).
# - Every 60 s (WATCH_INTERVAL_S) GET LOCAL_HEALTH_URL and, only when it answers 200, PUBLIC_HEALTH_URL (20 s timeout
#   each, WATCH_TIMEOUT_S; no credentials, no extra headers). 3 public answers 530 in a row (WATCH_FAILS; error 1033:
#   Cloudflare has no connection of the tunnel) while the local one is 200: SIGTERM to the cloudflared started here
#   (SIGKILL if it still runs 40 s later, WATCH_STOP_S), started again 10 s later, then 90 s without probes
#   (WATCH_GRACE_S). Any other answer, or none (this Mac's network, DNS or TLS, Cloudflare's own rules, an edge error: a
#   restart mends none of them), is logged once and resets the count. At most 6 restarts per hour (WATCH_MAX_PER_HOUR;
#   WATCH_WINDOW_S=3600): past that one line, and the probes go on. A local failure is the server's problem: the
#   tunnel is left alone.
# - Only the cloudflared started here is ever signalled, by its pid. It runs in its own process group (ignoring
#   SIGTTOU, so it writes to the tab even with stty tostop), so a Ctrl-C in the tab reaches it once, through this
#   script: one SIGTERM (cloudflared's grace period), then this script exits when it has stopped; a second Ctrl-C stops
#   it at once. SIGTERM and SIGHUP (tab closed) do the same. Other cloudflared processes (other configs, or one already
#   running with this config, which is reported) are never touched.
# - One line per event on stdout; cloudflared's own log (level info, no client addresses) goes to the tab as before.
# CLOUDFLARED (default: cloudflared on PATH) and WATCH_TICK_S (1 s, the step of the sleeps) exist for the tests
# (mvc/tests/test_tunnel_supervise.py).
set -u
[ $# -eq 3 ] || { echo "usage: $0 CONFIG LOCAL_HEALTH_URL PUBLIC_HEALTH_URL" >&2; exit 2; }
[ -f "$1" ] || { echo "$1: no such config file" >&2; exit 2; }
CFG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
LOCAL=$2 PUBLIC=$3
for u in "$LOCAL" "$PUBLIC"; do
  case $u in http://*|https://*) ;; *) echo "$u: not an http(s) URL" >&2; exit 2 ;; esac
done
CF_BIN=${CLOUDFLARED:-cloudflared}
command -v "$CF_BIN" >/dev/null || { echo "$CF_BIN not found" >&2; exit 2; }
INTERVAL=${WATCH_INTERVAL_S:-60} FAILS=${WATCH_FAILS:-3} GRACE=${WATCH_GRACE_S:-90} TIMEOUT=${WATCH_TIMEOUT_S:-20}
MAXH=${WATCH_MAX_PER_HOUR:-6} WINDOW=${WATCH_WINDOW_S:-3600} RESTART=${WATCH_RESTART_S:-10} STOPW=${WATCH_STOP_S:-40}
TICK=${WATCH_TICK_S:-1}
NAME=$(basename "$CFG" .yml)

# /bin/echo, not the builtin: after the tab is closed, a failed builtin echo leaves its text in bash's buffer, and the
# next $(...) gets it in its output (stop_cf would then skip cloudflared's grace period); a closed pipe kills only it
log() { /bin/echo "$(date -u +%FT%TZ) supervise $NAME: $*"; }
probe() { curl -s -o /dev/null -w '%{http_code}' --max-time "$TIMEOUT" "$1" 2>/dev/null; }
ticks() { awk -v s="$1" -v t="$TICK" 'BEGIN { n = int(s / t); if (n * t < s) n++; print n }'; }
# nap SECONDS [cf]: sleeps in steps of TICK; returns 1 early when a stop was asked or (with cf) cloudflared has exited
nap() {
  local i=0 n
  n=$(ticks "$1")
  while [ $i -lt $n ]; do
    [ -n "$STOPPING" ] && return 1
    [ $# -gt 1 ] && ! kill -0 "$CF" 2>/dev/null && return 1
    sleep "$TICK"
    i=$((i + 1))
  done
  [ -z "$STOPPING" ]
}
start_cf() {
  set -m                                       # its own process group: a Ctrl-C in the tab reaches it only from here
  ( trap '' TTOU; exec "$CF_BIN" tunnel --config "$CFG" --no-autoupdate run </dev/null ) &
  CF=$!
  set +m
  log "cloudflared started (pid $CF)"
}
# stop_cf: SIGTERM to the cloudflared started here (unless the signal handler sent it), SIGKILL after STOPW
stop_cf() {
  [ "$TERMED" = "$CF" ] || kill -TERM "$CF" 2>/dev/null
  TERMED=$CF
  local i=0 n
  n=$(ticks "$STOPW")
  while kill -0 "$CF" 2>/dev/null && [ $i -lt $n ]; do sleep "$TICK"; i=$((i + 1)); done
  if kill -0 "$CF" 2>/dev/null; then
    log "cloudflared (pid $CF) still runs ${STOPW}s after SIGTERM: SIGKILL"
    kill -KILL "$CF" 2>/dev/null
  fi
  wait "$CF" 2>/dev/null
  CF=                                          # never signal a pid that is no longer this script's child
}
on_signal() {
  if [ -z "$STOPPING" ]; then
    STOPPING=1
    log "stopping${CF:+: SIGTERM to cloudflared (pid $CF)}"
  else
    log "second signal${CF:+: SIGTERM again (cloudflared stops at once)}"
  fi
  [ -n "$CF" ] && kill -TERM "$CF" 2>/dev/null && TERMED=$CF
}
finish() {
  [ -n "$CF" ] && kill -0 "$CF" 2>/dev/null && stop_cf
  log "stopped"
  exit 0
}

CF= TERMED= STOPPING=
trap on_signal INT TERM HUP
others=$(pgrep -u "$(id -u)" -f -- "tunnel --config $CFG " | tr '\n' ' ')
[ -n "$others" ] && log "note: another cloudflared runs with this config (pid ${others% }); it is left alone"
log "public $PUBLIC, local $LOCAL: probe every ${INTERVAL}s, restart after $FAILS public 530s in a row, at most" \
  "$MAXH per ${WINDOW}s"
start_cf
fails=0 restarts= capped= ldown= bad= delay=$INTERVAL
while :; do
  if ! nap "$delay" cf; then
    [ -n "$STOPPING" ] && finish
    wait "$CF" 2>/dev/null
    log "cloudflared (pid $CF) exited with status $?: starting it again in ${RESTART}s"
    CF=
    nap "$RESTART" || finish
    start_cf
    fails=0 delay=$INTERVAL
    continue
  fi
  delay=$INTERVAL
  code=$(probe "$LOCAL")
  [ -n "$STOPPING" ] && finish
  if [ "$code" != 200 ]; then
    [ -z "$ldown" ] && log "local $LOCAL answers ${code:-nothing}: the server's problem, the tunnel is left alone"
    ldown=1 fails=0
    continue
  fi
  [ -n "$ldown" ] && log "local $LOCAL answers 200 again"
  ldown=
  code=$(probe "$PUBLIC")
  [ -n "$STOPPING" ] && finish
  if [ "$code" = 200 ]; then
    [ -n "$bad" ] && log "public $PUBLIC answers 200 again"
    bad= fails=0 capped=
    continue
  fi
  if [ "$code" != 530 ]; then                  # not the tunnel's error: a restart would not mend it
    [ "$bad" != "$code" ] && log "public $PUBLIC answers $code, not 530: no restart"
    bad=$code fails=0
    continue
  fi
  fails=$((fails + 1)) bad=530
  [ -z "$capped" ] && log "public $PUBLIC answers $code while the local one is ok ($fails/$FAILS)"
  [ $fails -lt "$FAILS" ] && continue
  now=$(date +%s) keep= n=0
  for t in $restarts; do
    [ $((now - t)) -lt "$WINDOW" ] && keep="$keep $t" && n=$((n + 1))
  done
  restarts=$keep
  if [ $n -ge "$MAXH" ]; then
    [ -z "$capped" ] && log "cap of $MAXH restarts per ${WINDOW}s reached: no restart, still probing"
    capped=1
    continue
  fi
  capped=
  log "restart $((n + 1))/$MAXH of the last ${WINDOW}s: SIGTERM to cloudflared (pid $CF)"
  restarts="$restarts $now" fails=0
  stop_cf
  [ -n "$STOPPING" ] && finish
  log "cloudflared stopped: starting it again in ${RESTART}s"
  nap "$RESTART" || finish
  start_cf
  delay=$GRACE
done
