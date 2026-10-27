# Sourced at the end of the r2_4 job_eval.slurm (ANALYSIS_PLAN_mvc_v1 §4.6, R2.4): ECAPA and guards on the 3 s and 180 s
# full-Dev runs of the chosen configuration.
for d in $MVC/eval/r2_4/*; do
  n=$(basename $d); n=${n%_b*}; [ -d $d/rows ] || continue
  for r in $d/rows/dev_*.jsonl; do
    o=$d/ecapa/$(basename $r); [ -f $o ] && continue; mkdir -p $d/ecapa
    PYTHONPATH=$DATA_STORAGE/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DATA_STORAGE/models/spkrec-ecapa-voxceleb \
        --systems $n --rows $r --out $o || FAILED="$FAILED ecapa:$n"
  done
done
for m in $MVC/eval/r2_4/*/metrics/dev_*.jsonl; do
  case $m in *_src.jsonl) continue;; esac
  g=$(dirname $(dirname $m))/guard/$(basename $m)
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
