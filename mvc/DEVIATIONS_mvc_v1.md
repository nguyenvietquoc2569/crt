# MimiFlow-VC: deviations and freeze log (analysis plan v1)

Records required by `mvc/ANALYSIS_PLAN_mvc_v1.md` §8.4 step 7 and §10. Times are +07:00.

## Freeze log

- 27 Oct 2026 16:28: the analysis plan was frozen by the corresponding author, Quoc Nguyen. His decisions are in §11 of the plan.
- 27 Oct 2026 16:30:29: the plan was pushed to the public repository https://github.com/nguyenvietquoc2569/crt as commit 1cf96fc on branch `mvc-v1-freeze`, with the annotated tag `analysis-plan-mvc-v1`. The plan's sha256 is 46f2b586640fb5a7fee6654343a5403a4f93e96d5e7b1558219f701755bfcd86.
- 28 Oct 2026 01:51:42: the first R2 job was submitted (R2.0, job 6625, TAG r2_0). This was after the push of the analysis-plan tag at 27 Oct 16:30:29.
- 28 Oct 2026 06:21:09: the selection record was pushed to https://github.com/nguyenvietquoc2569/crt as commit 3a2b9b6 on branch `mvc-v1-freeze`, with the annotated tag `mvc-selection-record-v1`. `MVC_SELECTION_RECORD_v1.md` has sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42. mvc_v1 is cfg1_w1 (fallback F). The R1 run files were hashed by R1x (job 6658) at 06:19:29, before it wrote anything.
- 28 Oct 2026 07:22:10: the first T1 job was submitted (job 6663, `job_t1_conv.slurm` with ONLY=vctk2k, through `submit_t1.sh`). This was after the push of the selection record at 06:21:09. Its checks of the push line, `mvc_selection.env` against the record, the snapshot, the describe() commit and the 21 frozen ceai files passed.

## Deviations

None so far.

## Operational notes (not deviations)

Choices on points the plan leaves open, fixed before any T1 metric exists (28 Oct 2026, about 08:45).

1. **T1 jobs** (`mvc/jobs/job_t1_{conv,metrics,ecapa}.slurm`, `t1_lib.sh`, `t1_check.py`, `submit_t1.sh`; commit 3a2b9b6).
   - `conv.done` is `$T/conv/conv.done`. It is written once every manifest has a row per pair. Remaining error rows are failed conversions (§5.4), and the job then converts nothing more.
   - Scorer drift checks (§8.3) run at the start of each metrics part (ECAPA, then WavLM-SV, UTMOS and Whisper) and again in the ECAPA job. Each writes its own file under `$T/drift/`, named by job and restart count.
     - WavLM-SV and UTMOS must be equal to the last digit.
     - "At least 31 of 32" is read as at least 31 rows on which both Whisper hypotheses (large-v3 and base) are identical.
     - The 32 ECAPA rows are X2's first 32 VCTK-2k rows without error.
   - All three jobs check `DESCRIBE_COMMIT` and the frozen files, not only the conversion job. The ECAPA job waits for the three `m.done` files (the §8.1 order).
   - `acoustic_guard.py`'s printed means go to /dev/null, and the summary file of means it writes is deleted unread.
2. **DNSMOS on the Mac** (`mvc/analysis/dnsmos_t1.py`, with tests).
   - It reads only `pair_id`, `system` and `wav` of the 30 s metric rows.
   - The journal rows on the Mac still name `/datastore/nvquoc/crt`, which moved to `/datastore/ceai/crt` on 21 Oct (benchmark/DEVIATIONS_v1.md item 4), so that prefix is mapped.
   - It copies each (set, system) batch of WAVs read-only from ceai into `MR/wav/` below its absolute ceai path, runs the frozen `dnsmos.py --norm_lufs -26`, deletes the summary file of means unread, and deletes the batch's WAVs. The Mac has about 17 GiB free, not enough for all 38,000 WAVs; ceai keeps the originals.
3. **R1x at 250k (§4.10), exploratory: the snapshot is unusable** (28 Oct 2026, 21:30).
   - **What happened.** Training had one transient loss spike at step ~239.4k: loss 1.80 over 100 steps, gradient norm 251 before clipping (normally about 0.2). The live model recovered within about 300 steps, and its validation loss stayed normal (0.320–0.334).
   - **The EMA was corrupted.** Its validation loss was 0.3147 at 238k (R1 at 200k: 0.3158), then 0.78 at 244k, 41.6 at 248k and 40.4 at 250k. It has been falling since: 0.47 at 260k, 0.35 at 266k.
   - **Result.** The Dev-S100 run of `ema_0250000.pt` (job 6885) gives WER 94–96 %, SIM 0.59–0.60 and UTMOS 1.26 at both settings. This measures the corrupted average, not longer training. It is reported as such.
   - **What stays as planned.** The 300k evaluation runs as planned. Its `val_loss_ema` is reported beside it, to show whether the EMA has recovered. No other snapshot is evaluated, and mvc_v1 is not affected: R1 had no such spike, its largest gradient norm being 2.46 during warm-up.
