# Sourced at the end of the e4 job_eval.slurm (27 Oct): R1 100k same-speaker reconstructions (SNG off: source and
# target are the same speaker) at <= 10 s and at any length (LTC-2k has many sources above the 10 s training crop),
# against the copies of the same utterances (p8rec/al_copy, p7/dv_copy); then ECAPA and acoustic guards for e4.
O=$MVC/eval/e4rec
V="{\"run_dir\": \"$MVC/runs/r1\", \"ckpt\": \"ema_0100000.pt\", \"temp\": 0.0, \"cfg\": 0.0, \"relax\": 0.15, \"relax_k\": 2}"
recon_eval rc_r1k100 "$V" dev-clean 200 10 $O
recon_eval al_r1k100 "$V" dev-clean 300 0 $O
for d in $MVC/eval/e4/*; do
  n=$(basename $d); [ -d $d/rows ] || continue
  for r in $d/rows/dev_*.jsonl; do
    o=$d/ecapa/$(basename $r); [ -f $o ] && continue; mkdir -p $d/ecapa
    PYTHONPATH=$DATA_STORAGE/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DATA_STORAGE/models/spkrec-ecapa-voxceleb \
        --systems $n --rows $r --out $o || FAILED="$FAILED ecapa:$n"
  done
done
for m in $MVC/eval/e4/*/metrics/dev_30.jsonl $O/*/metrics.jsonl; do
  case $m in */e4rec/*) g=$(dirname $m)/guard.jsonl;; *) g=$(dirname $(dirname $m))/guard/dev_30.jsonl;; esac
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
