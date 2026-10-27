# Sourced at the end of the p8 job_eval.slurm (26 Oct, J1 of the bottleneck plan): reconstructions at the two
# pre-declared candidates D2 (temp 0.4, w 0.7) and D5 (temp 0, w 0), the any-length check, and 3 s / 180 s references.
O=$MVC/eval/p8rec; V=$MVC/runs/p6_V
recon_eval dv_T0.4 "{\"run_dir\": \"$V\", \"temp\": 0.4}" dev-clean 200 10 $O
recon_eval dv_T0_w0 "{\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0}" dev-clean 200 10 $O
recon_eval tr_T0_w0 "{\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0}" train-clean-100 200 10 $O
recon_eval dv_C_T0_w0 "{\"run_dir\": \"$MVC/runs/pilot_C\", \"temp\": 0.0, \"cfg\": 0.0}" dev-clean 200 10 $O
recon_eval al_copy '{"mode": "copy_latent"}' dev-clean 300 0 $O
recon_eval al_T0.4 "{\"run_dir\": \"$V\", \"temp\": 0.4}" dev-clean 300 0 $O
for b in 3 180; do
  eval_dev p8 @$WORK_DIR/mvc/pairs/dev_s100.txt "T0.4_w0.7={\"run_dir\": \"$V\", \"temp\": 0.4};T0_w0={\"run_dir\": \"$V\", \"temp\": 0.0, \"cfg\": 0.0}" $b
done
echo "=== acoustic guards: $(date)"
for m in $MVC/eval/p8/*/metrics/dev_*.jsonl $MVC/eval/p6sweep/{V_base,V_t0.8,V_q32}/metrics/dev_30.jsonl \
         $MVC/eval/floors/{knnvc,freevc,seedvc,facodec,copy_latent}/metrics/dev_30.jsonl; do
  case $m in *_src.jsonl) continue;; esac
  g=$(dirname $(dirname $m))/guard/$(basename $m)
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
for m in $O/*/metrics.jsonl $MVC/eval/p7/{dv_copy,dv_V,tr_V}/metrics.jsonl; do
  g=$(dirname $m)/guard.jsonl
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
for f in $O/*/summary.json $MVC/eval/p8/*/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); print(json.dumps({k: {x: v.get(x) for x in ('n', 'wer_capped', 'sim', 'sim_src', 'utmos')} for k, v in d.items()}))"; done
for g in $MVC/eval/p8/*/guard/*.summary.json $MVC/eval/{p6sweep,floors}/*/guard/*.summary.json $O/*/guard.summary.json; do echo "== $g $(cat $g | tr -d "\n ")"; done
