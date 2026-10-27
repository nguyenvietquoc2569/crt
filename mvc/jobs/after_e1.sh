# Sourced at the end of the e1 job_eval.slurm (26 Oct, keep-Mimi plan E1): the relax at the 180 s budget (3 x 60 s
# chunk ensemble vs one 60 s chunk), same-speaker reconstructions of the base and both relax settings, and the
# acoustic guards of everything (DNSMOS runs locally on the copied wavs).
V=$MVC/runs/p6_V; O=$MVC/eval/e1rec
eval_dev e1 @$WORK_DIR/mvc/pairs/dev_s100.txt "T0_w0_rx15={\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0, \"relax\": 0.15, \"relax_k\": 2};T0_w0_rx15_p60={\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0, \"relax\": 0.15, \"relax_k\": 2, \"prompt_max\": 750}" 180
recon_eval rc_T0_w0 "{\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0}" dev-clean 200 10 $O
recon_eval rc_T0_w0_rx15 "{\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0, \"relax\": 0.15, \"relax_k\": 2}" dev-clean 200 10 $O
recon_eval rc_T0_w0_rx25 "{\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0, \"relax\": 0.25, \"relax_k\": 2}" dev-clean 200 10 $O
echo "=== acoustic guards: $(date)"
for m in $MVC/eval/e1/*/metrics/dev_*.jsonl; do
  case $m in *_src.jsonl) continue;; esac
  g=$(dirname $(dirname $m))/guard/$(basename $m)
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
for m in $O/*/metrics.jsonl; do
  g=$(dirname $m)/guard.jsonl
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
for f in $O/*/summary.json $MVC/eval/e1/*/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); print(json.dumps({k: {x: v.get(x) for x in ('n', 'wer_capped', 'sim', 'sim_src', 'utmos')} for k, v in d.items()}))"; done
for g in $MVC/eval/e1/*/guard/*.summary.json $O/*/guard.summary.json; do echo "== $g $(cat $g | tr -d "\n ")"; done
