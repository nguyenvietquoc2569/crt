#!/bin/bash
# Submits one CRT job from the UIT HPC account nvquoc. Until 18 Oct 2026 the account also served the live Qwen model
# (job name qwen-serve, rotated by a 04:00 cron); that service has moved off the account, and the corresponding author
# allowed CRT up to the account's 3 running jobs. Rules for this account:
#   1. fewer than MAXJOBS (3) jobs of nvquoc, of any name (qwen-serve included), running or pending, so that a new
#      job can start at once (the account runs at most 3);
#   2. no submission between 03:00 and 04:10 Asia/Ho_Chi_Minh (nor in the 2 min before 03:00), kept in case the
#      qwen-serve rotation returns;
#   3. >= 3660 s since the last Submit of any job of nvquoc, qwen-serve included (sacct -X over the last 3 days, plus
#      the submit times that squeue shows, in case sacct lags).
# Run it inside tmux on the login node: it polls every POLL s (default 300) until all three hold, then submits
# <script.slurm> with the extra sbatch args and appends "<time> <job id> <Submit> <script> <args>" (tab-separated) to
# /datastore/nvquoc/crt/_transfer_logs/submit.log. It refuses (exit 1, nothing submitted) when squeue or sacct fail
# or print a line it cannot parse, when it does not run as nvquoc, when date has no zone data for Asia/Ho_Chi_Minh,
# and when the job would break the account's job rules: the script must have #SBATCH --account=uit,
# --partition=normal, --gres=gpu:2g.35gb:1 and --time=HH:MM:SS <= 72:00:00, an --array only with %1 and no
# --job-name=qwen-serve, and the extra args must be options --name=value that set none of these, not even by an
# abbreviated name (--part=debug). SBATCH_* variables, which sbatch would let override the #SBATCH lines, are unset.
# Wrappers run one at a time (flock on /tmp/submit_nvquoc.lock), so two of them cannot both see an empty queue and
# submit. It never cancels or changes a job.
#   bash submit_nvquoc.sh job_d0.slurm
#   bash submit_nvquoc.sh job_m.slurm --export=ALL,PART=t1

USR=nvquoc
GAP=3660
MAXJOBS=3
POLL=${POLL:-300}
# SUBMIT_NVQUOC_LOG and SUBMIT_NVQUOC_LOCK exist only so that benchmark/tests/test_submit_nvquoc.py can keep the log
# and the lock under its temporary directory; never set them on the HPC.
LOG=${SUBMIT_NVQUOC_LOG:-/datastore/nvquoc/crt/_transfer_logs/submit.log}
LOCK=${SUBMIT_NVQUOC_LOCK:-/tmp/submit_nvquoc.lock}

refuse() { echo "$(date '+%F %T') submit_nvquoc: REFUSED, nothing submitted: $*" >&2; exit 1; }

[ $# -ge 1 ] || refuse 'usage: bash submit_nvquoc.sh <script.slurm> [extra sbatch args]'
SCRIPT=$(readlink -f "$1") && [ -f "$SCRIPT" ] || refuse "no script $1"
shift
[ "$(id -un)" = $USR ] || refuse "this is $(id -un), not $USR"
[ "$(TZ=Asia/Ho_Chi_Minh date +%z)" = +0700 ] \
    || refuse 'no zone data for Asia/Ho_Chi_Minh (date would place the 03:00-04:10 window in UTC)'
source /etc/profile.d/modules.sh 2>/dev/null
module load slurm >/dev/null 2>&1

opt() { sed -n "s/^#SBATCH --$1=//p" "$SCRIPT" | tail -1; }    # the script's last #SBATCH --<name>=<value>
! grep -qE '^#SBATCH[[:space:]]+(-[ApatJ]|--(account|partition|gres|time|array|job-name)[[:space:]])' "$SCRIPT" \
    || refuse "$SCRIPT sets account, partition, gres, time, array or job name in a form this check does not read"
[ "$(opt account)" = uit ] && [ "$(opt partition)" = normal ] && [ "$(opt gres)" = gpu:2g.35gb:1 ] \
    || refuse "$SCRIPT lacks #SBATCH --account=uit, --partition=normal or --gres=gpu:2g.35gb:1"
t=$(opt time)
[[ $t =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] \
    && (( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} <= 259200 )) \
    || refuse "$SCRIPT: #SBATCH --time=${t:-?} is not HH:MM:SS <= 72:00:00"
a=$(opt array)
[ -z "$a" ] || [[ $a == *%1 ]] || refuse "$SCRIPT: #SBATCH --array=$a can run more than one task at a time (use %1)"
[ "$(opt job-name)" != qwen-serve ] || refuse "$SCRIPT is named qwen-serve"
for x in "$@"; do   # sbatch takes a bare word as the script and any unambiguous prefix of a long option (--part=)
    [[ $x == --?*=* ]] || refuse "the extra argument $x is not an option of the form --name=value"
    for r in --account --partition --gres --time --array --job-name; do
        [[ $r == "${x%%=*}"* || ${x%%=*} == "$r"* ]] && refuse "the extra argument $x sets a fixed job rule"
    done
done
for v in ${!SBATCH_*}; do   # sbatch lets SBATCH_* variables override the #SBATCH lines checked above
    echo "submit_nvquoc: ignoring $v=${!v}"
    unset "$v"
done

exec 9>>"$LOCK" && flock 9 || refuse "cannot lock $LOCK"

blocked() {  # prints why a submission must wait now, nothing if it may go; status 1 if squeue or sacct failed or
             # printed a line that cannot be parsed
    local now t hm q s last=0
    now=$(date +%s)
    for t in $now $(( now + 120 )); do
        hm=$(TZ=Asia/Ho_Chi_Minh date -d @"$t" +%H%M) || return 1
        if (( 10#$hm >= 300 && 10#$hm < 410 )); then
            echo 'the 03:00-04:10 window (Asia/Ho_Chi_Minh)'
            return 0
        fi
    done
    q=$(squeue -h -u $USR -o '%i|%j|%T|%V') || return 1
    [ -z "$(awk -F'|' 'NF && NF != 4' <<< "$q")" ] || return 1
    s=$(awk -F'|' 'NF {print $1, $2, $3}' <<< "$q")
    if (( $(grep -c . <<< "$s") >= MAXJOBS )); then
        echo "$MAXJOBS or more jobs of $USR are in the queue: $(tr '\n' ';' <<< "$s")"
        return 0
    fi
    s=$(sacct -u $USR -S "$(date -d '-3 days' +%Y-%m-%dT%H:%M:%S)" -X -n -P -o Submit) || return 1
    while read -r t; do
        [ -n "$t" ] || continue
        t=$(date -d "$t" +%s 2>/dev/null) || return 1
        (( t > last )) && last=$t
    done <<< "$s
$(awk -F'|' 'NF {print $4}' <<< "$q")"
    if (( now - last < GAP )); then
        echo "the last Submit of a job of $USR was at $(date -d @"$last" '+%F %T'), $(( now - last )) s ago (< $GAP s)"
    fi
    return 0
}

while :; do
    why=$(blocked) || refuse 'squeue or sacct failed, or printed a line that could not be parsed'
    [ -n "$why" ] || break
    echo "$(date '+%F %T') waiting to submit $(basename "$SCRIPT"): $why"
    sleep "$POLL"
done
out=$(sbatch --parsable "$@" "$SCRIPT") || refuse "sbatch failed: $out"
jid=${out%%;*}
sub=$(squeue -h -j "$jid" -o %V 2>/dev/null | head -1)
[ -n "$sub" ] || sub=$(sacct -j "$jid" -X -n -P -o Submit 2>/dev/null | head -1)
mkdir -p "$(dirname "$LOG")"
line=$(printf '%s\t%s\t%s\t%s\t%s' "$(date '+%F %T')" "$jid" "${sub:-unknown}" "$SCRIPT" "$*")
echo "$line" >> "$LOG"
echo "submitted: $line"
