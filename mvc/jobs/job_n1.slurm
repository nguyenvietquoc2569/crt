#!/bin/bash
#SBATCH --job-name=mvc_n1
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_n1_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_n1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
# Mimi-VC N1 (26 Oct, keep-Mimi plan): can a guidance branch that sees no voice at all (prompt frames = pure noise)
# move the UTMOS-SIM trade-off line instead of sliding along it? Two 20k-step fine-tunes from p6_V's EMA with p6_V's
# recipe (lr 3e-5, 500 warm-up, constant then cosine, EMA 0.999): n1_np (--null_noise_prompt 1) and n1_ctrl (the same
# extra training without it). Dev-S100 at 30 s for both: temp 0 / w 0, temp 0.1 / w 0.7 (standard branch), and
# temp 0.1 with the noise-prompt branch at w 0.5 and 1.0, all with the Mimi relax RELAX (default 0.15 x 2).
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/eval_dev.sh
FAILED=
if [ -n "${PRE_EVALS_FILE:-}" ]; then          # short Dev-S100 evals first (e.g. an R1 snapshot), results in $MVC/eval/n1pre
  eval_dev n1pre @$WORK_DIR/mvc/pairs/dev_s100.txt "$(cat $PRE_EVALS_FILE)" 30
fi
RX=${RELAX:-0.15}
V1="--sampler prompt --tokens 16000 --pos segment_random --seg_len 125 --spk 192 --p_drop_cond 0.15 --p_drop_spk 0.10 \
--p_drop_content 0.05 --corrupt 0.1 --content pcb0 --ada_single --whisper 1 --val_every 2000 --sample_every 20000 \
--snap_every 10000 --exclude_pairs $J/pairs/dev.jsonl,$J/pairs/vctk2k.jsonl,$J/pairs/ltc2k.jsonl"
FT="--init_ckpt $MVC/runs/p6_V/ema_last.pt --steps 20000 --lr 3e-5 --warmup 500 --sched const_tail --ema 0.999"
EVALS=
for arm in np:1.0 ctrl:0.0; do
  name=n1_${arm%%:*}
  echo " train $name (null_noise_prompt ${arm#*:}): $(date)"
  python train.py --train_dirs $(dirs train-clean-100) --val_dirs $(dirs dev-clean) --out $MVC/runs/$name --d 512 \
      --layers 8 --heads 8 --max_hours 3 $V1 $FT --null_noise_prompt ${arm#*:} || { FAILED="$FAILED $name"; continue; }
  R="\"run_dir\": \"$MVC/runs/$name\", \"relax\": $RX, \"relax_k\": 2"
  EVALS="$EVALS;${name}_T0_w0_rx={$R, \"temp\": 0.0, \"cfg\": 0.0};${name}_T0.1_w0.7_rx={$R, \"temp\": 0.1, \"cfg\": 0.7}"
  EVALS="$EVALS;${name}_T0.1_np0.5_rx={$R, \"temp\": 0.1, \"cfg\": 0.5, \"cfg_branch\": \"noiseprompt\"}"
  EVALS="$EVALS;${name}_T0.1_np1.0_rx={$R, \"temp\": 0.1, \"cfg\": 1.0, \"cfg_branch\": \"noiseprompt\"}"
done
[ -n "$EVALS" ] && eval_dev n1 @$WORK_DIR/mvc/pairs/dev_s100.txt "${EVALS#;}" 30
for m in $MVC/eval/n1/*/metrics/dev_30.jsonl; do
  g=$(dirname $(dirname $m))/guard/dev_30.jsonl
  [ -f $g ] || python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard"
done
for f in $MVC/eval/n1/*/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); print(json.dumps({k: {x: v.get(x) for x in ('n', 'wer_capped', 'sim', 'sim_src', 'utmos')} for k, v in d.items()}))"; done
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
