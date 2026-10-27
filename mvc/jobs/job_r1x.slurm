#!/bin/bash
#SBATCH --job-name=mvc_r1x
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_r1x_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_r1x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=64G
#SBATCH --time=40:00:00
# Mimi-VC R1x (ANALYSIS_PLAN_mvc_v1 §4.10, exploratory, author's request at the freeze): R1 continued from its 200k
# state to 300k in its own directory. runs/r1 is never written: the first start copies r1's last.pt, args.json and
# stats.npz into runs/r1x; train.py then resumes at step 200k with R1's arguments, --steps 300000 (const_tail: 1.5e-4
# to 240k, cosine to 1.5e-5 at 300k) and a 2,000-step linear re-warm. Never eligible to become mvc_v1.
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
SRC=$MVC/runs/r1; OUT=$MVC/runs/r1x
if [ ! -f $OUT/last.pt ]; then
  mkdir -p $OUT
  python -c "import torch; c = torch.load('$SRC/last.pt', map_location='cpu', weights_only=False); assert c['step'] == 200000, c['step']" \
    || { echo "r1/last.pt is not the 200k state"; exit 1; }
  cp $SRC/last.pt $SRC/stats.npz $OUT/ && cp $SRC/args.json $OUT/args_r1.json && echo " copied R1's 200k state: $(date)"
fi
TOK=$(python -c "import json; print(json.load(open('$SRC/args.json'))['tokens'])")
TRAIN=$(dirs train-clean-100,train-clean-360,train-other-500)
R1="--d 768 --layers 16 --heads 12 --steps 300000 --sampler prompt --pos segment_random --seg_len 125 --spk 192 \
--p_drop_cond 0.15 --p_drop_spk 0.10 --p_drop_content 0.05 --corrupt 0.1 --content pcb0 --ada_single --whisper 1 \
--tau_dist uniform --lr 1.5e-4 --warmup 5000 --sched const_tail --beta2 0.99 --ema 0.9999 \
--val_every 2000 --sample_every 20000 --snap_every 10000 --exclude_utts $WORK_DIR/mvc/results/audit_v1/exclude.txt \
--exclude_pairs $J/pairs/dev.jsonl,$J/pairs/vctk2k.jsonl,$J/pairs/ltc2k.jsonl --rewarm 2000 --rewarm_from 200000"
echo " R1x with $TOK tokens: $(date)"
python train.py --train_dirs $TRAIN --val_dirs $(dirs dev-clean) --out $OUT --max_hours ${HOURS:-38} $R1 --tokens $TOK
rc=$?
echo "finished rc=$rc: $(date)"
exit $rc
