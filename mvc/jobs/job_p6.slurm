#!/bin/bash
#SBATCH --job-name=mvc_p6
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_p6_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_p6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=60G
#SBATCH --time=20:00:00
# Mimi-VC P6 (PLAN.md 4): the v1 recipe at pilot scale, then throughput of the main-run configuration.
#   arms (ARMS, ';'-separated NAME:extra-args). Default after the P5 gate record (PLAN.md, 25 Oct): V = v1 recipe with
#   perturbed CB0 + the perturbed Whisper-small stream + CAM++; Vf = V trained without the utterances of the audit's
#   exclusion list (A0; the list for train-clean-100 + dev-clean is recomputed first with the current RULES);
#   training data TRAIN (default train-clean-100, add train-clean-360 when encoded), 60k steps, 40M (d512 L8 H8);
#   Dev-S100 at 30, 3 and 180 s for every arm (NFE 16, w 0.7, continuous decode), and dev-clean reconstruction;
#   1,000 steps of the 121M main-run configuration (d768 L16 H12, 32k-token batches) for s/step and memory.
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/eval_dev.sh
FAILED=
AUD=$WORK_DIR/mvc/results/audit_tc100
echo " audit pass 2 for train-clean-100 + dev-clean: $(date)"
python audit_data.py --shards $(dirs train-clean-100,dev-clean) --summarize --force --out $AUD || FAILED="$FAILED audit"
TRAIN=${TRAIN:-train-clean-100}
TRAIN=${TRAIN//+/,}          # sbatch --export splits its value at commas: pass TRAIN=a+b there
STEPS=${STEPS:-60000}
V1="--sampler prompt --tokens 16000 --pos segment_random --seg_len 125 --spk 192 --p_drop_cond 0.15 --p_drop_spk 0.10 \
--p_drop_content 0.05 --corrupt 0.1 --content pcb0 --ada_single --val_every 2000 --sample_every 20000 --snap_every 20000 --exclude_pairs $J/pairs/dev.jsonl,$J/pairs/vctk2k.jsonl,$J/pairs/ltc2k.jsonl"
ARMS=${ARMS:-"V:--whisper=1;Vf:--whisper=1 --exclude_utts=$AUD/exclude.txt"}
EVALS= NAMES=
IFS=';' read -ra AS <<< "$ARMS"
for arm in "${AS[@]}"; do
  name=p6_${arm%%:*}; extra=${arm#*:}
  echo " train $name ($extra): $(date)"
  python train.py --train_dirs $(dirs $TRAIN) --val_dirs $(dirs dev-clean) --out $MVC/runs/$name --d 512 --layers 8 \
      --heads 8 --steps $STEPS --max_hours 6 $V1 $extra || FAILED="$FAILED $name"
  EVALS="$EVALS;$name={\"run_dir\": \"$MVC/runs/$name\"}"
  NAMES="$NAMES $name"
done
EVALS=${EVALS#;}
for b in 30 3 180; do
  eval_dev p6 @$WORK_DIR/mvc/pairs/dev_s100.txt "$EVALS" $b
done
eval_dev p6 @$WORK_DIR/mvc/pairs/dev_s100.txt "pilot_C={\"run_dir\": \"$MVC/runs/pilot_C\"}" 30   # P5 arm C on Dev-S100
for name in $NAMES; do
  recon_eval ${name}_rec "{\"run_dir\": \"$MVC/runs/$name\"}"
done
echo " throughput 121M: $(date)"
python train.py --train_dirs $(dirs $TRAIN) --val_dirs $(dirs dev-clean) --out $MVC/runs/p6_throughput --d 768 \
    --layers 16 --heads 12 --steps 1000 --tokens 32000 --log_every 50 --val_every 1000 --n_samples 0 $V1 --whisper=1 \
    || FAILED="$FAILED throughput"
python - <<PY
import json
recs = [json.loads(l) for l in open('$MVC/runs/p6_throughput/train.log') if '"steps_per_s"' in l]
if recs:
    sps = [r['steps_per_s'] for r in recs[2:]] or [recs[-1]['steps_per_s']]
    print('121M throughput: %.3f steps/s, %.2f s/step, peak memory %s GB, data wait %.3f s per log window' % (
        sum(sps) / len(sps), len(sps) / sum(sps), recs[-1].get('peak_mem_gb'), recs[-1].get('data_wait_s', 0)))
PY
for f in $MVC/eval/p6/*/summary.json $MVC/eval/recon/p6_*/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); print(json.dumps({k: {x: v.get(x) for x in ('n', 'wer_capped', 'sim', 'sim_src', 'utmos')} for k, v in d.items()}))"; done
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
