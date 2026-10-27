# Sourced at the end of the pilot job (needs jobs/env.sh and jobs/eval_dev.sh): the J-probe battery of PLAN.md 4.
#   P0b content-controlled leakage on the VCTK Dev shared sentences (probe_parallel_text.py)
#   P1  decoder tolerance (probe_decoder.py: run -> metrics.py -> summarize)
#   P2  content ceiling at 12.5 Hz, CTC character probes (probe_content_ctc.py)
#   P3  training-free Mimi floors (knn_latent, ot_latent) and the four baselines on Dev-S100 at 30 s
#   P4  CAM++ sanity (probe_campplus.py)
# Each probe is independent: a failure is recorded in $FAILED and the next probe runs.
source $WORK_DIR/mvc/jobs/eval_dev.sh        # the current helpers, even if the calling job sourced an older copy
R=$WORK_DIR/mvc/results
mkdir -p $R $MVC/probes/p1/m
echo " P4 campplus: $(date)"
python -u probe_campplus.py --raw $RAW --pairs $J/pairs/dev.jsonl --out $R --device cuda || FAILED="$FAILED P4"
echo " P0b parallel text: $(date)"
python -u probe_parallel_text.py --pairs $J/pairs/dev.jsonl --out $R/p0b --K 2 --seed 20261025 \
    --workers $SLURM_CPUS_PER_TASK --device cuda || FAILED="$FAILED P0b"
echo " P1 decoder: $(date)"
P1=$MVC/probes/p1
python -u probe_decoder.py run --shards_dev $MVC/mimi/dev-clean --raw $RAW --pairs $J/pairs/dev.jsonl --journal $J \
    --out $P1 --local_disk /tmp --device cuda \
  && python -u $J/metrics.py --in_jsonl $P1/rows_all.jsonl --out_jsonl $P1/m/all.jsonl --metrics wer,sim,utmos \
      --whisper_models large-v3 --wavlm_model $J/models/wavlm-base-plus-sv --device cuda \
  && python -u probe_decoder.py summarize --metrics $P1/m --json $R/p1_summary.json || FAILED="$FAILED P1"
echo " P2 content CTC: $(date)"
python -u probe_content_ctc.py --out $R/p2 --vctk_pairs $J/pairs/dev.jsonl || FAILED="$FAILED P2"
echo " P3 floors and baselines on Dev-S100: $(date)"
python make_dev_s100.py --pairs $J/pairs/dev.jsonl || FAILED="$FAILED dev_s100_check"
eval_dev floors @$WORK_DIR/mvc/pairs/dev_s100.txt \
    "knn_latent={\"mode\": \"knn_latent\"};ot_latent={\"mode\": \"ot_latent\"};copy_latent={\"mode\": \"copy_latent\"};copy_codes={\"mode\": \"copy_codes\"}"
export MPLCONFIGDIR=$DATA_STORAGE/cache/matplotlib
for SYS in knnvc freevc seedvc facodec; do
  [ $SYS = seedvc ] && source /datastore/ceai/crt/envs/seedvc/bin/activate
  O=$MVC/eval/floors/$SYS
  python -u $J/run_baselines.py --pairs $J/pairs/dev.jsonl --set dev --budget 30 --systems $SYS --out_dir $O/conv \
      --pair_ids $(paste -sd, $WORK_DIR/mvc/pairs/dev_s100.txt) --device cuda || FAILED="$FAILED P3:$SYS"
  [ $SYS = seedvc ] && source /datastore/ceai/crt/envs/mimi_vc/bin/activate
  [ -f $O/conv/dev/30/$SYS/manifest.jsonl ] && score_dev $O $SYS 30 $O/conv/dev/30/$SYS/manifest.jsonl
done
for f in $R/p4_campplus.json $R/p0b/p0b_parallel_text.json $R/p1_summary.json $R/p2/p2_content.json; do
  [ -f $f ] && { echo "== $f"; head -c 3000 $f; echo; }
done
