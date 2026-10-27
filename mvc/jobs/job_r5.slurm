#!/bin/bash
#SBATCH --job-name=mvc_r5
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_r5_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_r5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=48G
#SBATCH --time=14:00:00
# Mimi-VC R5 pilot (PLAN.md 2.8; triggered: Dev UTMOS 2.49 < Mimi copy 3.83 - 0.3): ground-truth-aligned fine-tune of
# Mimi's decoder on MimiFlow-VC latents.
#   1. gta_latents.py: RUN (default p6_V) converts every EVERY-th (default 3rd) train-clean-100 utterance in
#      reconstruction mode with the
#      inference settings in GTA_CFG_FILE (default: the adapter defaults), audit exclusions left out;
#   2. decoder_ft.py: STEPS (default 30000) of mel + MS-STFT GAN + feature matching, half the examples real latents;
#      (DEC_ARGS: extra decoder_ft.py arguments, e.g. "--w_mel 0.1");
#   0. PRE_EVALS_FILE (optional): Dev-S100 probes run before the GTA step (results under $MVC/eval/r5pre);
#   3. Dev-S100 at 30 s: the same settings with the fine-tuned decoder (checkpoints DEC_EVAL_STEPS, default
#      "010000 last"; decoder_<step>.pt), temp 1 with the last checkpoint, copy_latent with the
#      fine-tuned decoder (decoding real latents must not get worse), and dev-clean reconstruction.
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/eval_dev.sh
FAILED=
RUN=${RUN:-p6_V}
STEPS=${STEPS:-30000}
CFG=$( [ -n "${GTA_CFG_FILE:-}" ] && cat $GTA_CFG_FILE || echo '{}' )
BASE=$(python -c "import json,sys; d=json.loads(sys.argv[1]); d['run_dir']='$MVC/runs/$RUN'; print(json.dumps(d))" "$CFG")
NAME=${RUN}${GTA_TAG:+_$GTA_TAG}      # GTA_TAG names the inference setting, so GTA sets of different settings never mix
G=$MVC/gta/$NAME; D=$MVC/decoder/$NAME
if [ -n "${PRE_EVALS_FILE:-}" ]; then          # short Dev-S100 probes first (a file avoids quoting JSON through --export)
  eval_dev r5pre @$WORK_DIR/mvc/pairs/dev_s100.txt "$(cat $PRE_EVALS_FILE)" 30
fi
echo " gta $BASE: $(date)"
python -u gta_latents.py --shards $(dirs train-clean-100) --raw $RAW --every ${EVERY:-3} --cfg "$BASE" \
    --exclude $WORK_DIR/mvc/results/audit_tc100/exclude.txt --out $G || FAILED="$FAILED gta"
echo " decoder fine-tune: $(date)"
python -u decoder_ft.py --gta $G --shards $(dirs train-clean-100) --raw $RAW --out $D --steps $STEPS ${DEC_ARGS:-} \
    || FAILED="$FAILED decoder"
withdec() { python -c "import json,sys; d=json.loads(sys.argv[1]); d['decoder_ckpt']=sys.argv[2]; print(json.dumps(d))" "$1" "$2"; }
echo " eval: $(date)"
T1="{\"run_dir\": \"$MVC/runs/$RUN\"}"
EV=
for s in ${DEC_EVAL_STEPS:-010000 last}; do      # decoder checkpoints to score (decoder_<step>.pt, or last)
  EV="$EV;${NAME}_dec$s=$(withdec "$BASE" $D/decoder_$s.pt)"
done
eval_dev r5 @$WORK_DIR/mvc/pairs/dev_s100.txt "${EV#;};${NAME}_t1_dec=$(withdec "$T1" $D/decoder_last.pt);${NAME}_copy_latent_dec={\"mode\": \"copy_latent\", \"decoder_ckpt\": \"$D/decoder_last.pt\"}" 30
recon_eval ${NAME}_dec_rec "$(withdec "$BASE" $D/decoder_last.pt)" dev-clean 200 10
for m in $MVC/eval/r5/${NAME}_*/metrics/dev_30.jsonl $MVC/eval/recon/${NAME}_dec_rec/metrics.jsonl; do
  g=$(dirname $m)/guard_$(basename $m)
  python acoustic_guard.py --rows $m --out $g --workers $SLURM_CPUS_PER_TASK || FAILED="$FAILED guard:$m"
done
for f in $MVC/eval/r5/${NAME}_*/summary.json $MVC/eval/recon/${NAME}_dec_rec/summary.json; do echo "== $f"; python -c "import json; d = json.load(open('$f')); print(json.dumps({k: {x: v.get(x) for x in ('n', 'wer_capped', 'sim', 'sim_src', 'utmos')} for k, v in d.items()}))"; done
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
