#!/bin/bash
#SBATCH --job-name=mvc_r1
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_r1_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_r1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
# Mimi-VC R1 (PLAN.md R1, 26 Oct): the main run. 121M (d768 L16 H12, adaLN-single, PLAN 2.4) on 555 h of LibriTTS-R
# (train-clean-100 + 360 + other-500) without the audit_v1 exclusions (A0), the v1 recipe with the Whisper-small stream,
# and the PLAN 2.6 optimisation that p6_V did not use: tau ~ U(0,1), lr 1.5e-4, 5k warm-up, constant then cosine to
# 1.5e-5 from 80 %, AdamW beta2 0.99, EMA 0.9999. 200k steps, snapshots every 10k, resumable (resubmit to continue).
# Token budget: 300 steps at 32k tokens in a scratch run first; if that fails or peaks above 32 GB, 24k tokens.
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
RUN=${RUN:-r1}
STEPS=${STEPS:-200000}
TRAIN=$(dirs train-clean-100,train-clean-360,train-other-500)
R1="--d 768 --layers 16 --heads 12 --steps $STEPS --sampler prompt --pos segment_random --seg_len 125 --spk 192 \
--p_drop_cond 0.15 --p_drop_spk 0.10 --p_drop_content 0.05 --corrupt 0.1 --content pcb0 --ada_single --whisper 1 \
--tau_dist uniform --lr 1.5e-4 --warmup 5000 --sched const_tail --beta2 0.99 --ema 0.9999 \
--val_every 2000 --sample_every 20000 --snap_every 10000 --exclude_utts $WORK_DIR/mvc/results/audit_v1/exclude.txt \
--exclude_pairs $J/pairs/dev.jsonl,$J/pairs/vctk2k.jsonl,$J/pairs/ltc2k.jsonl"
TOK=${TOKENS:-}
if [ -z "$TOK" ] && [ ! -f $MVC/runs/$RUN/last.pt ]; then
  echo " token probe at 32k: $(date)"
  rm -rf $MVC/runs/${RUN}_tokprobe
  if python train.py --train_dirs $TRAIN --val_dirs $(dirs dev-clean) --out $MVC/runs/${RUN}_tokprobe $R1 --steps 300 \
       --tokens 32000 --log_every 50 --val_every 1000 --n_samples 0 --max_hours 1; then
    peak=$(python -c "import json; r=[json.loads(l) for l in open('$MVC/runs/${RUN}_tokprobe/train.log') if 'peak_mem_gb' in l]; print(max(x['peak_mem_gb'] for x in r))")
    sps=$(python -c "import json; r=[json.loads(l)['steps_per_s'] for l in open('$MVC/runs/${RUN}_tokprobe/train.log') if 'steps_per_s' in l][2:]; print(round(len(r)/sum(r),3))")
    echo " 32k tokens: peak $peak GB, $sps s/step"
    TOK=$(python -c "print(32000 if $peak <= 32 else 24000)")
  else
    echo " 32k tokens failed; using 24k"; TOK=24000
  fi
fi
[ -n "$TOK" ] || TOK=$(python -c "import json; print(json.load(open('$MVC/runs/$RUN/args.json'))['tokens'])")
echo " R1 $RUN with $TOK tokens: $(date)"
python train.py --train_dirs $TRAIN --val_dirs $(dirs dev-clean) --out $MVC/runs/$RUN --max_hours ${HOURS:-70} $R1 --tokens $TOK
rc=$?
echo "finished rc=$rc: $(date)"
exit $rc
