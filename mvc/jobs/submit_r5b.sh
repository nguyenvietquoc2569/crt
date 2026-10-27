#!/bin/bash
# submit_r5b.sh (26 Oct): the R5b pilot (job_r5.slurm) once CRT has a slot (slot_wait.sh; E1 running) and the 1 h gap
# allows (submit_ceai.sh): GTA at temp 0 / CFG 0 + Mimi relax 0.15 x 2 (every 6th train-clean-100 utterance), 6k
# decoder steps with pitch-safe losses (mel + FM on real rows only, anchor to the stock decoder on GTA rows, prefix
# context, D warm-up 1.5k), checkpoints every 1.5k, Dev-S100 at 3k / 4.5k / 6k.
M=/datastore/ceai/crt/mimi_vc/mvc
bash $M/jobs/slot_wait.sh
cd /datastore/ceai/crt/mimi_vc/benchmark/journal && RUN=p6_V GTA_CFG_FILE=$M/jobs/gta_T0_w0_rx15.json GTA_TAG=T0w0rx15 \
  EVERY=6 STEPS=6000 DEC_EVAL_STEPS="003000 004500 006000" \
  DEC_ARGS="--pair_losses real --w_anchor 1 --prefix --cache_wav --d_warmup 1500 --w_mel 0.1 --ckpt_every 1500" \
  bash submit_ceai.sh $M/jobs/job_r5.slurm --export=ALL
