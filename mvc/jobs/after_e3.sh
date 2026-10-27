# Sourced at the end of the e3 job_eval.slurm (27 Oct, D-mvc-2): reference budgets 3 s and 180 s for SNG 0 / 0.75 / 1.0,
# then ECAPA on target and @src rows of every e3 configuration (and the p6_V reference e1/T0_w0_rx15) for rule 5, and the
# acoustic guards.
V="\"run_dir\": \"$MVC/runs/r1\", \"ckpt\": \"ema_0100000.pt\", \"temp\": 0.0, \"cfg\": 0.0, \"relax\": 0.15, \"relax_k\": 2"
for b in 3 180; do
  eval_dev e3 @$WORK_DIR/mvc/pairs/dev_s100.txt "r1k100_ns0={$V};r1k100_ns0.75={$V, \"neg_src\": 0.75};r1k100_ns1.0={$V, \"neg_src\": 1.0}" $b
done
echo "=== ECAPA: $(date)"
for d in $MVC/eval/e3/* $MVC/eval/e1/T0_w0_rx15; do
  n=$(basename $d); [ -d $d/rows ] || continue
  for r in $d/rows/dev_*.jsonl; do
    o=$d/ecapa/$(basename $r)
    [ -f $o ] && continue
    mkdir -p $d/ecapa
    PYTHONPATH=$DATA_STORAGE/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DATA_STORAGE/models/spkrec-ecapa-voxceleb \
        --systems $n --rows $r --out $o || FAILED="$FAILED ecapa:$n"
  done
done
echo "=== acoustic guards: $(date)"
for m in $MVC/eval/e3/*/metrics/dev_*.jsonl; do
  case $m in *_src.jsonl) continue;; esac
  g=$(dirname $(dirname $m))/guard/$(basename $m)
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
for f in $MVC/eval/e3/*/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); v = d['_system']; st = v.get('strata', {}); print(json.dumps(dict(n=v.get('n'), wer=v.get('wer_capped'), sim=v.get('sim'), utmos=v.get('utmos'), same=st.get('same', {}).get('sim_src_minus_target_copy'), cross=st.get('cross', {}).get('sim_src_minus_target_copy'))))"; done
