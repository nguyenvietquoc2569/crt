#!/bin/bash
#SBATCH --job-name=mvc_pilot
#SBATCH --output=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_pilot_%j.out
#SBATCH --error=/datastore/ceai/crt/mimi_vc/mvc/logs/mvc_pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=60G
#SBATCH --time=30:00:00
# Mimi-VC pilot (probes P1 and P3 of the research plan), on train-clean-100 (held-out: dev-clean):
#   0. the rest of S0 (jobs 5639/5640 hung in Praat and were cancelled): perturbed codes of train-clean-100 and
#      dev-clean (resumes at the first missing shard), and probe P0 on dev-clean; in the background from step 3 on,
#      S2 = jobs/s2_encode.sh (train-other-500 encode, perturbed codes and CAM++ of train-clean-360 + other-500), log
#      logs/s2_<job>.log, waited for at the end;
#   1. CAM++ speaker embeddings of every utterance (spk_embed.py -> shard.spk.npy);
#   2. Whisper-small features of perturbed draw 0 (whisper_feats.py -> shard.pwhisper0.npy);
#   3. three 30M MimiFlow pilots, 60k steps each: arm A content = clean CB0, arm B = CB0 of perturbed copies,
#      arm C = B plus the perturbed Whisper stream;
#   4. the Dev screen (shard 0/20 = 50 VCTK pairs, 30 s): A, B, C (Whisper of the clean / of a perturbed source), and
#      the decode controls copy_latent (the source's own continuous latent through the unquantised path) and
#      copy_codes (its 32 codes), scored with the journal metrics; arm B's dev-clean reconstruction;
#   5. the J-probe battery P0b, P1, P2, P3, P4 (jobs/jprobe.sh).
source /datastore/ceai/crt/mimi_vc/mvc/jobs/env.sh
source $WORK_DIR/mvc/jobs/eval_dev.sh
FAILED=
STEPS=${STEPS:-60000}
echo " S0 perturbed: $(date)"
python encode_perturbed.py --shards $(dirs train-clean-100,dev-clean) --raw $RAW --K 2 --workers $SLURM_CPUS_PER_TASK \
    || FAILED="$FAILED S0"
echo " spk: $(date)"
python spk_embed.py --shards $(dirs train-clean-100,dev-clean) --raw $RAW || FAILED="$FAILED spk"
echo " P0: $(date)"
mkdir -p $WORK_DIR/mvc/results
python probe_speaker_info.py --shards $MVC/mimi/dev-clean --K 2 --out $WORK_DIR/mvc/results/p0_speaker_info.json \
    || FAILED="$FAILED P0"
echo " whisper: $(date)"
python whisper_feats.py --shards $(dirs train-clean-100,dev-clean) --raw $RAW --K 1 --workers $SLURM_CPUS_PER_TASK \
    || FAILED="$FAILED whisper"
echo " S2 in the background: $(date)"
bash $WORK_DIR/mvc/jobs/s2_encode.sh > $WORK_DIR/mvc/logs/s2_$SLURM_JOB_ID.log 2>&1 &
S2=$!
COMMON="--d 512 --layers 8 --heads 8 --batch 32 --steps $STEPS --spk 192 --val_every 2000 --sample_every 20000 --snap_every 20000 --exclude_pairs $J/pairs/dev.jsonl,$J/pairs/vctk2k.jsonl,$J/pairs/ltc2k.jsonl"
for arm in A:cb0: B:pcb0: C:pcb0:--whisper=1; do
  IFS=: read -r tag content extra <<< "$arm"
  name=pilot_$tag
  echo " train $name ($content $extra): $(date)"
  python train.py --train_dirs $(dirs train-clean-100) --val_dirs $(dirs dev-clean) --out $MVC/runs/$name \
      --content $content --max_hours 6 $COMMON $extra || FAILED="$FAILED $name"
done
echo " eval: $(date)"
eval_dev pilot 0/20 "copy_latent={\"mode\": \"copy_latent\"};copy_codes={\"mode\": \"copy_codes\"};pilot_A={\"run_dir\": \"$MVC/runs/pilot_A\"};pilot_B={\"run_dir\": \"$MVC/runs/pilot_B\"};pilot_C={\"run_dir\": \"$MVC/runs/pilot_C\"};pilot_C_pw={\"run_dir\": \"$MVC/runs/pilot_C\", \"whisper_src\": \"perturbed\"}"
recon_eval pilot_B_rec "{\"run_dir\": \"$MVC/runs/pilot_B\"}"
for f in $MVC/eval/pilot/*/summary.json $MVC/eval/recon/pilot_B_rec/summary.json; do echo "== $f"; cat $f; done
echo " J-probe: $(date)"
source $WORK_DIR/mvc/jobs/jprobe.sh
echo " waiting for S2: $(date)"
wait $S2 || FAILED="$FAILED S2"
tail -3 $WORK_DIR/mvc/logs/s2_$SLURM_JOB_ID.log
echo "finished, failed:${FAILED:- none}: $(date)"
[ -z "$FAILED" ]
