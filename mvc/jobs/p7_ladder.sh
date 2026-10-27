# Sourced by job_p7.slurm and after_p6sweep.sh (needs env.sh and eval_dev.sh): the P7 naturalness bottleneck ladder.
# RUN (p6_V), N (200), MAX_S (10), INIT_TS (0.3,0.6,0.8,0.9,0.95) and EXTRA ("NAME=CFGJSON;...") as in job_p7.slurm.
RUN=${RUN:-p6_V}; N=${N:-200}; MAXS=${MAX_S:-10}; OUT=$MVC/eval/p7
V="{\"run_dir\": \"$MVC/runs/$RUN\"}"
recon_eval tr_copy '{"mode": "copy_latent"}' train-clean-100 $N $MAXS $OUT
recon_eval tr_V "$V" train-clean-100 $N $MAXS $OUT
recon_eval dv_copy '{"mode": "copy_latent"}' dev-clean $N $MAXS $OUT
recon_eval dv_V "$V" dev-clean $N $MAXS $OUT
for t in $(echo ${INIT_TS:-0.3,0.6,0.8,0.9,0.95} | tr , " "); do
  recon_eval dv_V_t$t "{\"run_dir\": \"$MVC/runs/$RUN\", \"init_t\": $t}" dev-clean $N $MAXS $OUT
done
if [ -n "${EXTRA:-}" ]; then
  IFS=';' read -ra ITEMS <<< "$EXTRA"
  for it in "${ITEMS[@]}"; do recon_eval "${it%%=*}" "${it#*=}" dev-clean $N $MAXS $OUT; done
fi
for f in $OUT/*/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); print(json.dumps({k: {x: v.get(x) for x in ('n', 'wer_capped', 'sim', 'utmos')} for k, v in d.items()}))"; done
