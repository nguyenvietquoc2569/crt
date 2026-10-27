#!/bin/bash
# chain_after_download.sh <script.slurm> [--export=...]: waits for fetch_v2.sh to finish (stops if a subset failed),
# then for slot_wait.sh, then submits through submit_ceai.sh.
F=/datastore/ceai/crt/_transfer_logs/fetch_v2.log
until grep -q finished $F; do sleep 300; done
grep -q failed $F && { echo "$(date '+%F %T') a download failed; nothing submitted"; exit 1; }
bash /datastore/ceai/crt/mimi_vc/mvc/jobs/slot_wait.sh
cd /datastore/ceai/crt/mimi_vc/benchmark/journal && bash submit_ceai.sh "$@"
