# Sourced at the end of the e2 job_eval.slurm (26 Oct): acoustic guards for the R1-snapshot leakage probes.
for m in $MVC/eval/e2/*/metrics/dev_30.jsonl; do
  g=$(dirname $(dirname $m))/guard/dev_30.jsonl
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
