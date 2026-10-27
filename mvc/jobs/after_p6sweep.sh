# Sourced at the end of the p6sweep job_eval.slurm (26 Oct): run the P7 bottleneck ladder in the same slot, plus the
# 20k snapshot and pilot C on the same dev-clean reconstructions.
EXTRA="dv_V_ck20={\"run_dir\": \"$MVC/runs/p6_V\", \"ckpt\": \"ema_0020000.pt\"};dv_C={\"run_dir\": \"$MVC/runs/pilot_C\"}"
source $WORK_DIR/mvc/jobs/p7_ladder.sh
