# Sourced at the end of the r2_0 job_eval.slurm (ANALYSIS_PLAN_mvc_v1 §4.2, R2.0): same-speaker reconstruction capped
# WER on the 300 dev-clean utterances of PLAN §5.2 (temp 0, relax 0.15 x 2, CFG 0, no SNG) for both snapshots.
O=$MVC/eval/r2_0rec
for k in k150 k200; do
  recon_eval r2_0_${k}_rec "$(python -c "import json; print(json.load(open('$WORK_DIR/mvc/jobs/r2_0_recon.json'))['$k'])")" dev-clean 300 0 $O
done
