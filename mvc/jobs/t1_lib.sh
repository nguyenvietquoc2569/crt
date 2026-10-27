# t1_lib.sh: sourced by the T1 jobs of mvc/ANALYSIS_PLAN_mvc_v1.md §8 (job_t1_conv.slurm, job_t1_metrics.slurm,
# job_t1_ecapa.slurm) after env.sh (WORK_DIR, J, DATA_STORAGE, MVC). It holds the T1 paths of §8.2, the scorer commands
# of §3.1 and §8.2 (job M's, benchmark/job_m.slurm, with --wavlm_model on every metrics.py call), the drift checks of
# §8.3 and the row checks of every step (t1_check.py). Nothing here prints a metric value of a T1 system: the checks
# print counts; acoustic_guard.py's printed means go to /dev/null and the summary file of means it writes beside its
# output is deleted unread (§8.3).
C=$WORK_DIR/mvc                                   # the MimiFlow-VC code on ceai (run_mvc.py, acoustic_guard.py, *.py)
T=$MVC/test                                       # the T1 tree of §8.2, never under $J/v1
SEL_ENV=$C/mvc_selection.env                      # select_mvc.py --stage record --final, copied to ceai with the record
DEVIATIONS=$C/DEVIATIONS_mvc_v1.md                # copied to ceai with the push of the record filled in (§8.4 steps 6-7)
RECORD=$C/MVC_SELECTION_RECORD_v1.md              # its sha256 is RECORD_SHA256
RECORD_CFG=$C/MVC_SELECTION_RECORD_v1_cfg         # select_mvc.py --cfg_dir: mvc_v1.json, mvc_v1_q32.json, ...
T1_CHECK=$C/jobs/t1_check.py
WAVLM=$J/models/wavlm-base-plus-sv                # §3.1: every metrics.py call passes it
ECAPA_MODEL=$DATA_STORAGE/models/spkrec-ecapa-voxceleb
ECAPA_SITE=$DATA_STORAGE/site/x2_ecapa            # X2's speechbrain site folder (§3.5)
ECAPA_SAVEDIR=$T/x2/ecapa_model                   # where speechbrain links the model files; one per concurrent job
T1_SETS="vctk2k ltc2k"
T1_SYSTEMS=mvc_v1,mvc_v1_q32,mvc_copy_latent      # §2; at 3 s and 180 s only mvc_v1
X2_SYSTEMS=faiss,crt_tau0.4,cand_pm18_rq,knnvc,freevc,seedvc,facodec,source_copy,target_copy   # job_x2.slurm
GUARD_SYSTEMS=mvc_v1,mvc_v1_q32,cand_pm18_rq,knnvc,freevc,seedvc,facodec                      # §3.8
JOURNAL_SELECTION_SHA256=9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105     # job M, X2 (§2.3)
FAILED=
stamp() { echo " $1: $(date)"; [ -z "${TIMES:-}" ] || echo "$(date +%s) $1" >> $TIMES; }
st() { [ "$1" -eq 0 ] || FAILED="$FAILED $2"; }
t1_done_line() { echo "job $SLURM_JOB_ID $(date '+%F %T') selection sha256 $RECORD_SHA256"; }
t1_done_ok() { grep -q "selection sha256 $RECORD_SHA256\$" "$1" 2>/dev/null; }   # <done file> written under the record
t1_run() { echo "job${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}"; }   # names a drift file: a requeue keeps the job id

t1_pushed() {  # §8.3 first bullet: no T1 conversion before the selection record is pushed (DEVIATIONS_mvc_v1.md)
  python -u $T1_CHECK pushed --deviations $DEVIATIONS
}

t1_selection() {  # t1_selection <cfg dir> [<install dir>]: mvc_selection.env against the record, the cfg JSONs in <cfg
                  # dir>, the snapshot and the describe() commit of $C/*.py (§4.8), then the frozen ceai files
                  # (FREEZE_mvc_v1.sha256 Section 2: pair lists, scorers, ECAPA and DNSMOS models); sources the env
  python -u $T1_CHECK selection --env $SEL_ENV --record $RECORD --cfg_dir $1 --code_dir $C ${2:+--install $2} \
    && python -u $T1_CHECK frozen --freeze $C/FREEZE_mvc_v1.sha256 \
    && source $SEL_ENV
}

t1_convert() {  # t1_convert <set>: run_mvc.py for every (system, budget) of §2 and §8.2 on <set>, cfg from $T/cfg
  local set=$1 nb name b
  for nb in mvc_v1:30 mvc_v1_q32:30 mvc_copy_latent:30 mvc_v1:3 mvc_v1:180; do
    name=${nb%:*}
    b=${nb#*:}
    stamp "$set $b s: $name"
    python -u $C/run_mvc.py --journal $J --pairs $J/pairs/$set.jsonl --set $set --budget $b --name $name \
        --cfg "$(cat $T/cfg/$name.json)" --out_dir $T/conv --device cuda
    st $? conv_${set}_${b}_$name
  done
}

t1_conv_check() {  # every T1 manifest of both sets has a row per pair (§8.2); error rows are counted as failures (§5.4)
  python -u $T1_CHECK conv --conv_dir $T/conv --cfg_dir $T/cfg --env $SEL_ENV --pairs vctk2k=$J/pairs/vctk2k.jsonl \
      --pairs ltc2k=$J/pairs/ltc2k.jsonl
}

t1_journal_done() {  # job M's parts t1, t2, t4 and X2 done under the journal selection: the drift checks, the guards
                     # and the comparators read their rows
  local f
  for f in $J/v1/m/t1/m.done $J/v1/m/t2/m.done $J/v1/m/t4/m.done $J/v1/x2/x2.done; do
    grep -q "selection sha256 $JOURNAL_SELECTION_SHA256" $f 2>/dev/null \
      || { echo "ERROR: $f missing or not written under the journal selection"; return 1; }
  done
}

t1_ecapa_env() {  # X2's speechbrain site folder imports (§3.5)
  [ -d $ECAPA_SITE/speechbrain ] || { echo "ERROR: $ECAPA_SITE/speechbrain missing (job_x2.slurm installed it)"; return 1; }
  PYTHONPATH=$ECAPA_SITE${PYTHONPATH:+:$PYTHONPATH} python -c \
    "import speechbrain, hyperpyyaml; print('speechbrain', speechbrain.__version__, 'from', speechbrain.__file__)"
}

t1_drift_m() {  # t1_drift_m <tag>: §8.3 drift check of WavLM-SV, UTMOS and Whisper: the T1 command on the first 32 rows
                # of job M's VCTK-2k 30 s rows into a new file, against job M's metrics of those rows
  local out=$T/drift/m_$1_$(t1_run)_vctk2k_30.jsonl
  mkdir -p $T/drift
  [ ! -e $out ] || { echo "ERROR: $out exists (metrics.py would reuse its rows)"; return 1; }
  python -u $J/metrics.py --in_jsonl $J/v1/m/t1/rows/vctk2k_30.jsonl --out_jsonl $out --max_rows 32 \
      --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model $WAVLM --device cuda \
    && python -u $T1_CHECK drift_m --new $out --ref $J/v1/m/t1/metrics/vctk2k_30.jsonl --n 32 --min_same 31
}

t1_score() {  # t1_score <part dir> <set> <budget> <systems>: lab_rows.py on the systems' manifests (VCTK-2k with the
              # source-E file, §3.1), metrics.py on the target rows and on the @src rows, then the row checks
  local o=$1 set=$2 b=$3 sys=$4 g=$2_$3 x ins= e= s
  for x in ${sys//,/ }; do ins="$ins --manifest $T/conv/$set/$b/$x/manifest.jsonl"; done
  [ $set != vctk2k ] || e="--source_e $J/pairs/vctk2k_source_E.json"
  stamp "$g: rows"
  python -u $J/lab_rows.py --pairs $J/pairs/$set.jsonl $e $ins --out $o/rows/$g.jsonl
  s=$?
  st $s rows_$g
  [ $s -eq 0 ] || return 1
  stamp "$g: metrics"
  python -u $J/metrics.py --in_jsonl $o/rows/$g.jsonl --out_jsonl $o/metrics/$g.jsonl \
      --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model $WAVLM --device cuda
  st $? metrics_$g
  stamp "$g: metrics, source speaker"
  python -u $J/metrics.py --in_jsonl $o/rows/${g}_src.jsonl --out_jsonl $o/metrics/${g}_src.jsonl \
      --metrics sim,secs --wavlm_model $WAVLM --device cuda
  st $? metrics_src_$g
  python -u $T1_CHECK metrics --dir $o --g $g --pairs $J/pairs/$set.jsonl --systems $sys
  st $? checks_$g
}

t1_score_ecapa() {  # score_ecapa.py with X2's site folder and model (§3.5); the arguments are appended
  PYTHONPATH=$ECAPA_SITE${PYTHONPATH:+:$PYTHONPATH} python -u $J/score_ecapa.py --model_dir $ECAPA_MODEL \
      --savedir $ECAPA_SAVEDIR --device cuda "$@"
}

t1_drift_ecapa() {  # t1_drift_ecapa <tag>: §8.3 ECAPA drift check: X2's first 32 VCTK-2k rows re-scored into a new
                    # file, |Δ sim_ecapa| <= 0.001
  local in=$T/drift/ecapa_$1_$(t1_run)_rows.jsonl out=$T/drift/ecapa_$1_$(t1_run)_vctk2k_30.jsonl
  mkdir -p $T/drift
  [ ! -e $out ] || { echo "ERROR: $out exists (score_ecapa.py would reuse its rows)"; return 1; }
  python -u $T1_CHECK ecapa_drift_rows --x2 $J/v1/x2/ecapa/vctk2k_30.jsonl --rows $J/v1/m/t1/rows/vctk2k_30.jsonl \
      --rows $J/v1/m/t4/rows/vctk2k_30.jsonl --n 32 --out $in \
    && t1_score_ecapa --systems $X2_SYSTEMS --rows $in --out $out \
    && python -u $T1_CHECK drift_ecapa --new $out --ref $J/v1/x2/ecapa/vctk2k_30.jsonl --n 32 --tol 0.001
}

t1_ecapa() {  # t1_ecapa <set> <part>: score_ecapa.py on the part's 30 s target and @src rows of the T1 systems, then the
              # checks; score_ecapa.py exits 1 on the error rows of failed conversions, so the check decides
  local set=$1 p=$2 k
  for k in '' _src; do
    stamp "ECAPA ${set}_30$k"
    t1_score_ecapa --systems $T1_SYSTEMS --rows $T/m/$p/rows/${set}_30$k.jsonl --out $T/x2/ecapa/${set}_30$k.jsonl
    echo "  score_ecapa.py exit $?"
  done
  python -u $T1_CHECK ecapa --rows_dir $T/m/$p/rows --ecapa_dir $T/x2/ecapa --g ${set}_30 --pairs $J/pairs/$set.jsonl \
      --systems $T1_SYSTEMS
  st $? ecapa_$set
}

t1_guard() {  # t1_guard <set> <part>: acoustic_guard.py (§3.8) on the 30 s rows of mvc_v1 and mvc_v1_q32 (this part),
              # cand_pm18_rq (job M's part of the set) and the four baselines (job M t4), read only. Its printed means go
              # to /dev/null; it writes into guard/tmp/, where the <set>_30.summary.json of means it writes beside its
              # output is deleted unread and only <set>_30.jsonl is moved to guard/ (§8.3). The input goes to
              # guard/input/, outside analyze_mvc.py's guard/<set>_30*.jsonl
  local set=$1 p=$2 in=$T/guard/input/$1_30.jsonl tmp=$T/guard/tmp s
  stamp "guards $set"
  python -u $T1_CHECK guard_rows --out $in --src $T/m/$p/rows/${set}_30.jsonl=mvc_v1,mvc_v1_q32 \
      --src $J/v1/m/$p/rows/${set}_30.jsonl=cand_pm18_rq --src $J/v1/m/t4/rows/${set}_30.jsonl=knnvc,freevc,seedvc,facodec \
    || { st 1 guard_input_$set; return 1; }
  rm -rf $tmp $T/guard/${set}_30.jsonl           # a stopped or earlier run's output, unread
  python -u $C/acoustic_guard.py --rows $in --out $tmp/${set}_30.jsonl --workers ${SLURM_CPUS_PER_TASK:-8} > /dev/null
  s=$?
  rm -f $tmp/${set}_30.summary.json               # acoustic_guard.py's means: deleted unread (§8.3)
  [ $s -ne 0 ] || mv $tmp/${set}_30.jsonl $T/guard/${set}_30.jsonl || s=1
  rm -rf $tmp
  st $s guard_$set
  python -u $T1_CHECK guard --guard $T/guard/${set}_30.jsonl --pairs $J/pairs/$set.jsonl --systems $GUARD_SYSTEMS
  st $? guard_rows_$set
}
