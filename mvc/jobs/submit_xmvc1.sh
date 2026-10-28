#!/bin/bash
# submit_xmvc1.sh [SET ...]: submits job_xmvc1.slurm, X-mvc-1 of mvc/ANALYSIS_PLAN_mvc_v1.md §4.10 (DEVIATIONS_mvc_v1.md
# operational note 4), once per SET (default vctk2k ltc2k) with ONLY=<set>, each after slot_wait.sh (the CRT/P1 split)
# through submit_ceai.sh (the account's 3 jobs, the 1 h gap, the 03:00-04:10 blackout), as submit_t1.sh does. It
# refuses (nothing submitted) when T1 is not done (the m.done of $MVC/test/m/t1, t2, t3, x2/x2.done, guard/guard.done)
# or when a named set's $X/<set>.done exists; the job checks again what it needs and refuses otherwise. It does not
# wait for the jobs: resubmit a set that stopped by naming it, e.g. "submit_xmvc1.sh ltc2k". The script only tests that
# done files exist; it reads no metric file. ECAPA_RESCORE=1 bash submit_xmvc1.sh <set> passes ECAPA_RESCORE=1.
# DNSMOS runs on the Mac after the copy of the files listed in $X/<set>.sha256 (mvc/analysis/dnsmos_xmvc1.py), then
# mvc/analysis/analyze_xmvc1.py; both check the copy against the lists.
# Tell the interpreter session before running it. Run it in tmux on login01:
#   bash -l -c "bash /datastore/ceai/crt/mimi_vc/mvc/jobs/submit_xmvc1.sh 2>&1 | tee -a /datastore/ceai/crt/mimi_vc/mvc/logs/submit_xmvc1.log"
M=/datastore/ceai/crt/mimi_vc/mvc
J=/datastore/ceai/crt/mimi_vc/benchmark/journal
T=/datastore/ceai/crt/mimi_vc/data_storage/mvc/test
X=/datastore/ceai/crt/mimi_vc/data_storage/mvc/xmvc1

say() { echo "$(date '+%F %T') submit_xmvc1: $*"; }

for x in "$@"; do
  case $x in
    vctk2k|ltc2k) ;;
    *) say "ERROR: $x is not a set (vctk2k, ltc2k)"; exit 1 ;;
  esac
done
for f in m/t1/m.done m/t2/m.done m/t3/m.done x2/x2.done guard/guard.done; do
  [ -e $T/$f ] || { say "ERROR: $T/$f missing: T1 is not done (§4.10); nothing submitted"; exit 1; }
done
for SET in ${@:-vctk2k ltc2k}; do
  [ ! -e $X/$SET.done ] || { say "ERROR: $X/$SET.done exists: X-mvc-1 on $SET is done; nothing submitted"; exit 1; }
done
for SET in ${@:-vctk2k ltc2k}; do
  bash $M/jobs/slot_wait.sh || exit 1
  out=$(cd $J && bash submit_ceai.sh $M/jobs/job_xmvc1.slurm \
          --export=ALL,ONLY=$SET${ECAPA_RESCORE:+,ECAPA_RESCORE=$ECAPA_RESCORE} | tee /dev/stderr)
  JID=$(awk -F'\t' '/^submitted: /{print $2}' <<< "$out")
  [ -n "$JID" ] || { say "ERROR: no job id from submit_ceai.sh for job_xmvc1.slurm ONLY=$SET"; exit 1; }
  say "job_xmvc1.slurm ONLY=$SET is job $JID"
done
say "X-mvc-1 jobs submitted; when $X/<set>.done exists for both sets, copy the files listed in $X/<set>.sha256, the" \
    "two lists and the done files to the Mac (dnsmos_xmvc1.py's header), then run dnsmos_xmvc1.py and analyze_xmvc1.py"
