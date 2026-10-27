# Sourced at the end of the e5 job_eval.slurm (27 Oct): ECAPA and acoustic guards for the R1 150k snapshot configs.
for d in $MVC/eval/e5/*; do
  n=$(basename $d); [ -d $d/rows ] || continue
  for r in $d/rows/dev_*.jsonl; do
    o=$d/ecapa/$(basename $r); [ -f $o ] && continue; mkdir -p $d/ecapa
    PYTHONPATH=$DATA_STORAGE/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DATA_STORAGE/models/spkrec-ecapa-voxceleb \
        --systems $n --rows $r --out $o || FAILED="$FAILED ecapa:$n"
  done
done
for m in $MVC/eval/e5/*/metrics/dev_30.jsonl; do
  g=$(dirname $(dirname $m))/guard/dev_30.jsonl
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
