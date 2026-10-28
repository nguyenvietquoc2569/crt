# Sourced at the end of the r1x300 job_eval.slurm (ANALYSIS_PLAN_mvc_v1 §4.10, exploratory R1x on Dev-S100): ECAPA on
# the target and @src rows of every configuration, then the acoustic guards, as after_r2_1.sh does for R2.1.
for d in $MVC/eval/$TAG/*; do
  n=$(basename $d); [ -d $d/rows ] || continue
  for r in $d/rows/dev_*.jsonl; do
    o=$d/ecapa/$(basename $r); [ -f $o ] && continue; mkdir -p $d/ecapa
    PYTHONPATH=$DATA_STORAGE/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DATA_STORAGE/models/spkrec-ecapa-voxceleb \
        --systems $n --rows $r --out $o || FAILED="$FAILED ecapa:$n"
  done
done
for m in $MVC/eval/$TAG/*/metrics/dev_30.jsonl; do
  g=$(dirname $(dirname $m))/guard/dev_30.jsonl
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
