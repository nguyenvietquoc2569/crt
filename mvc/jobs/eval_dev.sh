# Sourced by mvc job scripts.
# eval_dev TAG SEL "NAME=CFGJSON;NAME=CFGJSON;..." [BUDGET]: for each item, run_mvc.py converts the Dev pairs selected
#   by SEL ("i/n" = shard, "@file" = pair-id list such as pairs/dev_s100.txt) at BUDGET seconds (default 30), then
#   score_dev scores them. Results: $MVC/eval/TAG/NAME[_bBUDGET]/summary.json.
# score_dev O NAME BUDGET MANIFEST: lab_rows.py builds target and @src rows, metrics.py scores WER (Whisper large-v3),
#   WavLM-SV SIM and UTMOS (@src: SIM), summarize.py compares with CRT-AL, FAISS and the controls on the same pairs
#   (gender strata, leakage against target_copy). Failures are appended to $FAILED.
# recon_eval NAME CFGJSON: same-speaker reconstruction of 300 dev-clean utterances (probe_recon.py, PLAN 5.2), scored
#   with capped WER, SIM and UTMOS. Results: $MVC/eval/recon/NAME/summary.json.
sel_arg() { if [[ $1 == @* ]]; then echo "--pair_ids $(paste -sd, ${1#@})"; else echo "--shard $1"; fi; }
score_dev() {
  local O=$1 NAME=$2 BUDGET=$3 MAN=$4
  mkdir -p $O/metrics                  # metrics.py appends to its output and creates no folder (job 5644)
  python $J/lab_rows.py --pairs $J/pairs/dev.jsonl --manifest $MAN --out $O/rows/dev_$BUDGET.jsonl \
      || { FAILED="$FAILED $NAME:rows"; return 1; }
  python $J/metrics.py --in_jsonl $O/rows/dev_$BUDGET.jsonl --out_jsonl $O/metrics/dev_$BUDGET.jsonl \
      --metrics wer,sim,utmos --whisper_models large-v3 --wavlm_model $J/models/wavlm-base-plus-sv --device cuda \
      || { FAILED="$FAILED $NAME:m"; return 1; }
  python $J/metrics.py --in_jsonl $O/rows/dev_${BUDGET}_src.jsonl --out_jsonl $O/metrics/dev_${BUDGET}_src.jsonl \
      --metrics sim --wavlm_model $J/models/wavlm-base-plus-sv --device cuda || { FAILED="$FAILED $NAME:msrc"; return 1; }
  python summarize.py --metrics $O/metrics/dev_$BUDGET.jsonl --src $O/metrics/dev_${BUDGET}_src.jsonl \
      --compare cand_pm18_rq=$J/v1/d2/lab/metrics.jsonl:$J/v1/d2/lab/metrics_src.jsonl \
      --compare anc_faiss=$J/v1/d1/anc/dev/30/metrics.jsonl:$J/v1/d1/anc/dev/30/metrics_src.jsonl \
      --compare target_copy=$J/v1/d2/controls/metrics.jsonl:$J/v1/d2/controls/metrics_src.jsonl \
      --compare mimi_copy_src=$J/v1/d2/controls/metrics.jsonl:$J/v1/d2/controls/metrics_src.jsonl \
      --out $O/summary.json || { FAILED="$FAILED $NAME:sum"; return 1; }
}
recon_eval() {   # NAME CFG [SUBSET (dev-clean)] [N (300)] [MAX_S (0 = any length)] [OUT ($MVC/eval/recon)]
  local NAME=$1 CFG=$2 SUBSET=${3:-dev-clean} N=${4:-300} MAXS=${5:-0} OUT=${6:-$MVC/eval/recon}
  local O=$OUT/$1
  echo "=== recon $NAME on $SUBSET (n $N, max_s $MAXS) $CFG: $(date)"
  python probe_recon.py --shards $MVC/mimi/$SUBSET --raw $RAW --name $NAME --cfg "$CFG" --out $OUT \
      --n $N $([ "$MAXS" = 0 ] || echo --max_s $MAXS) || { FAILED="$FAILED $NAME:recon"; return 1; }
  python $J/metrics.py --in_jsonl $O/rows.jsonl --out_jsonl $O/metrics.jsonl --metrics wer,sim,utmos \
      --whisper_models large-v3 --wavlm_model $J/models/wavlm-base-plus-sv --device cuda \
      || { FAILED="$FAILED $NAME:recon_m"; return 1; }
  python summarize.py --metrics $O/metrics.jsonl --src $O/metrics.jsonl --out $O/summary.json \
      || { FAILED="$FAILED $NAME:recon_sum"; return 1; }
}
eval_dev() {
  local TAG=$1 SEL=$2 EVALS=$3 BUDGET=${4:-30} it NAME CFG O
  IFS=';' read -ra ITEMS <<< "$EVALS"
  for it in "${ITEMS[@]}"; do
    NAME=${it%%=*}; CFG=${it#*=}
    O=$MVC/eval/$TAG/$NAME$([ "$BUDGET" = 30 ] || echo _b$BUDGET)
    mkdir -p $O
    echo "=== eval $TAG/$NAME budget $BUDGET $CFG: $(date)"
    python run_mvc.py --journal $J --pairs $J/pairs/dev.jsonl --set dev --budget $BUDGET --name $NAME --cfg "$CFG" \
        --out_dir $O/conv $(sel_arg $SEL) || { FAILED="$FAILED $NAME:conv"; continue; }
    score_dev $O $NAME $BUDGET $(ls $O/conv/dev/$BUDGET/$NAME/manifest*.jsonl | head -1)
  done
}
