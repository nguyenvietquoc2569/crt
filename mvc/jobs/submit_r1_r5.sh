#!/bin/bash
# submit_r1_r5.sh (26 Oct): after the peer's pending submission (after_peer_submit.sh), submit CRT's R1 and the R5
# decoder fine-tune J2, one at a time through slot_wait.sh + submit_ceai.sh (which keeps the 1 h gap): R1 first if
# mvc_s3 (whose Whisper features and audit_v1 R1 needs) has left the queue by then, otherwise J2 first and R1 after S3.
# Run in tmux on login01: bash -l -c "bash submit_r1_r5.sh" >> logs/submit_r1_r5.log 2>&1
M=/datastore/ceai/crt/mimi_vc/mvc
source /etc/profile.d/modules.sh 2>/dev/null
module load slurm >/dev/null 2>&1
S3=${S3_JOB:-5853}
r5() {
  bash $M/jobs/slot_wait.sh
  (cd /datastore/ceai/crt/mimi_vc/benchmark/journal && GTA_CFG_FILE=$M/jobs/gta_T0_w0.json GTA_TAG=T0w0 EVERY=6 \
      STEPS=30000 DEC_ARGS="--w_mel 0.1" PRE_EVALS_FILE=$M/jobs/r5pre_evals.txt bash submit_ceai.sh $M/jobs/job_r5.slurm --export=ALL)
}
r1() {
  while squeue -h -j "$S3" -o %T 2>/dev/null | grep -q .; do sleep 300; done
  [ -f $M/results/audit_v1/exclude.txt ] || { echo "$(date '+%F %T') no audit_v1/exclude.txt: R1 not submitted"; return 1; }
  bash $M/jobs/slot_wait.sh
  (cd /datastore/ceai/crt/mimi_vc/benchmark/journal && bash submit_ceai.sh $M/jobs/job_r1.slurm --export=ALL)
}
bash $M/jobs/after_peer_submit.sh 07:21 07:50
if squeue -h -j "$S3" -o %T 2>/dev/null | grep -q .; then r5; r1; else r1; r5; fi
echo "$(date '+%F %T') submit_r1_r5 done"
