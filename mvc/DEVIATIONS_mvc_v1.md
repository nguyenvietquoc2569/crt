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

4. **R1x at 300k (§4.10), exploratory, and the X-mvc-1 decision** (29 Oct 2026, committed 04:18:08 as 7d4054f).
   - **Run.** R1x finished at step 300k (job 6658). Its EMA recovered from the spike of item 3: `val_loss_ema` 0.3101 at 300k (R1 at 200k: 0.3158). The Dev-S100 runs of `ema_0300000.pt` (TAG r1x300, submitted 04:10:23) give, against R1 at 200k:
     - reference setting (CFG 0, SNG 1.0): WER 2.34 → 1.75, SIM 0.903 → 0.901, UTMOS 3.874 → 3.904, SIM_src 0.689 → 0.694;
     - mvc_v1's setting (CFG 1, SNG 1.0): WER 1.46 → 2.92, SIM 0.909 → 0.905, UTMOS 3.887 → 3.949, SIM_src 0.683 → 0.686.
   - **Decision on X-mvc-1, made before any other R1x Dev number is read.** X-mvc-1 runs: R1x's 300k snapshot on VCTK-2k and LTC-2k at 30 s, with mvc_v1's frozen inference settings, after T1 is complete and recorded (done: de81e24).
     - **Why.** At the freeze the author asked to see the most this training reaches. The Dev-S100 numbers show higher UTMOS and flat SIM, and only a test-set run can tell whether that holds there.
     - **How it is reported.** Descriptively, with CIs only and in its own section. It never replaces or relabels a mvc_v1 result, as §4.10 fixes.

5. **X-mvc-1 jobs and analysis** (`mvc/jobs/job_xmvc1.slurm`, `xmvc1_check.py`, `submit_xmvc1.sh`; `mvc/analysis/dnsmos_xmvc1.py`, `analyze_xmvc1.py`; tests `mvc/tests/test_xmvc1.py`). They were written, reviewed and fixed before any X-mvc-1 output existed.
   - **Jobs.** One job per set (`ONLY=vctk2k|ltc2k`), writing only under `$MVC/xmvc1`. Each job refuses unless T1's five done files name the selection record.
   - **Settings.** The cfg is mvc_v1.json with only `run_dir` (→ `runs/r1x`) and `ckpt` (→ `ema_0300000.pt`) changed. `snapshot.sha256` holds the snapshot and `runs/r1x/stats.npz`, which must equal R1's frozen `stats.npz` (f6aae232…).
   - **Conversions.** These follow §5.4 through `$X/conv/<set>.conv.done`: retries only until every pair has been attempted.
   - **Checks before scoring.** The ECAPA smoke test, the ECAPA drift check and the WavLM/UTMOS/Whisper drift check all run first, as in T1. The scorer commands are T1's.
   - **Outputs.**
     - Per-set ECAPA savedirs and guard scratch folders; the guard means are deleted unread; `ECAPA_RESCORE=1` as in T1.
     - A copy list `$X/<set>.sha256` is written before the done line, which reads `job <id> <date> snapshot sha256 <sha> selection sha256 <record>`.
   - **On the Mac.** DNSMOS runs with `dnsmos_t1.py`'s code, checked by sha256, and its rows are copied back to ceai. The analysis is descriptive, with CIs only and no p values or Holm, and writes `mvc/EXPLORATORY_XMVC1_v1.{md,json}`.

## After the T1 analysis (28 Oct 2026, 23:20)

The analysis ran once, 22:42:57–22:43:30 (log `analyze_mvc_run.log`, sha256 cf406e19…). Its output was committed unchanged as de81e24 (`TEST_RESULTS_mvc_v1.md` d2a7cace…, `.json` 7674852b…).

An independent check followed. It recomputed every level and difference from the rows, re-derived every decision from the plan, and audited the inputs; each candidate problem then had an adversarial re-check. It found no difference in any number or decision, and re-running `analyse()` in memory reproduces the JSON exactly. The results file is not regenerated. Its notes are recorded here.

1. **Order of DNSMOS (§8.1, §8.2).** DNSMOS ran on the Mac, as §3.7 and the selection record fix, so it ran after the copy of the T1 tree to the Mac and not before it as the §8.1 list has it.
   - Copy: 21:05, 74 files, list `t1_copy.sha256` sha256 47e667f3f667a1dedef804d2a67c1c8b7b494ee8ecd7548b0492b578e26d65f3, every entry verified.
   - Driver `dnsmos_t1.py` (sha256 bfd93a3d…), frozen `dnsmos.py` (330b1d35…), `--norm_lufs -26`, 21:06–22:40; 38,000 rows, 0 errors (log sha256 6dd8ce17…).
   - Environment: uvx, Python 3.13.13, numpy 2.5.4, soundfile 0.14.0, soxr 1.1.0, librosa 1.0.0, onnxruntime 1.31.1, pyloudnorm 0.2.0. These versions were read back at 23:15 from the same cached environment.
   - The scored WAVs were deleted on the Mac after each batch; ceai keeps them.
   - The 40 DNSMOS files (rows and inputs) were copied back to ceai `$T/dnsmos/` with identical sha256 (list `dnsmos.sha256`, sha256 fd6ce481743fa5127497949b60011bb3611b03716c6e412ab110d9474a97fec0, in `$T/mac_logs/` with the two logs and the copy list).
2. **Done markers.** `analyze_mvc.py` checks the `m.done` files of parts t1–t3. It does not read the MimiFlow-VC `x2/x2.done` or `guard/guard.done`. Both name the record sha256 9f939ae6… (job 6841, 16:16:44 and 16:36:06).
3. **Dev counterparts of family RB (§4.6).** The results file gives no Dev value beside the RB rows. §4.6 names the full-Dev numbers at 3, 30 and 180 s as RB's Dev counterparts, while §4.9 asks for Dev-900 values beside test numbers, and Dev-900 exists only at 30 s. `analyze_mvc.py` left them out by design (its tests, commit bb925c7). They are in the selection record (`rb_dev_counterparts`), cfg1_w1 on full Dev at 3 / 30 / 180 s:
   - WER 1.90 / 1.84 / 2.05;
   - SIM 0.879 / 0.912 / 0.918;
   - UTMOS 3.801 / 3.865 / 3.861.
   The paper shows them labelled "full Dev, selected on Dev".
4. **Wording for the paper (§7.2, §5.2).**
   - The leakage sentence names the strata as test IDs; the paper writes "same-gender and cross-gender".
   - As §7.2 requires, angle 4 of PLAN §6.2 is presented as a measurement.
   - The Wilcoxon p and r_rb of NL, L and NI are in the JSON only.
5. **Erratum of times.** Operational notes 1–2 say "about 08:45"; they were committed at 08:34:59 (e9eab38), after the first T1 jobs had started (07:22) and before any T1 metric existed. Item 3 of the operational notes says "21:30"; it was committed at 21:05:32 (b724a21).

## X-mvc-1 (29 Oct 2026, exploratory, §4.10)

- **Jobs.** On ceai: job 6996 (VCTK-2k, submitted 05:12:55, done 06:27:35) and job 7002 (LTC-2k, submitted 06:17:57, done 07:30:13).
  - 3,800 conversions, 0 failed, 0 scoring errors.
  - Every ECAPA and WavLM/UTMOS/Whisper drift check passed exactly.
  - Snapshot sha256 e668c87e…, stats.npz R1's (f6aae232…).
- **Copy and DNSMOS.** The copy to the Mac (22 files) matches both per-set sha256 lists. DNSMOS: 3,800 rows, 0 errors. The rows were copied back to ceai `$MVC/xmvc1/dnsmos/` (list `dnsmos.sha256` 16e679d5…, verified there).
- **Analysis.** `analyze_xmvc1.py` ran at 07:59:15 and wrote `EXPLORATORY_XMVC1_v1.md` (sha256 5d9beb4e74a82f56983ab1cf0fec9b96c6f77ba4171a3ba3761e171f254a5afb) and `.json` (21100ecaeff5613b3597c058d443038dd37d613564087d5814cd7184b14af6ad), committed unchanged. Its UTMOS, SIM and DNSMOS OVRL differences were recomputed independently from the rows and match.

## Listening test (mvc/LISTENING_ADDENDUM_mvc_v1.md, §13 step 6)

- **30 Oct 2026, 03:02:36: freeze.** The addendum was frozen and pushed to https://github.com/nguyenvietquoc2569/crt as commit 2b82187 on branch `mvc-v1-freeze`, with the annotated tag `listening-addendum-mvc-v1`. The tag on the remote peels to the same commit.
  - Addendum sha256: 45f5b428dd622b508199289ffcefcfd6109f97a81bbf6bff1ccc1aa9b8486356. Design sha256: 6a1cb8fa…3b5. The 36 frozen hashes are in `mvc/LISTENING_FREEZE_mvc_v1.sha256`.
  - The author delegated the remaining §16 decisions to Claude on 30 Oct 2026. Ethics stays pending until UIT answers, and no rater takes part before that.
- **03:02:52–03:02:54: the one draw** (§12 step 4), with the tagged builder and design and seed string `'20260928:mvc_v1'`.
  - 32 items, 3 practice items and 167 pairs passed over.
  - `$D/private/items.json`, sha256 9cc329548e1a224225aae3a209129330652df499bf9ecbf51556ca30805d4094. Nobody has opened it.
- **03:03:18–03:03:39: the build** (§12 step 5).
  - 231 clips; 162 MOS, 194 SMOS and 3 practice trials.
  - The package is in `$D/package` (30 MB). The key is in `$D/private/`.
  - The manifest is `mvc/LISTENING_BUILD_mvc_v1.json`, sha256 a60994fdc872064d81cb47c8d6263e75c9f51b585ba715d7687d545a57f2ce92, committed with this entry (§12 step 6).
  - The manifest holds no condition mapping. `lt_data.json` names only opaque clip ids. The key and the item list have not been opened.
- **03:10: build check, then the cache deleted.** All 231 clip hashes and the hashes of `lt_data.json`, `key.json` and `items.json` match the manifest. The audio cache `$D/private/cache` (60 MB, paths naming the systems) was then deleted (§11.5). The originals stay on ceai.
- **31 Oct 2026: deviation of the consent template (§14).**
  - The author opened a public demo of mvc_v1 at vc.betaschool.edu.vn, with its API at vc-api.betaschool.edu.vn, and decided that people who use it are excluded from rating. Its pages label outputs as MimiFlow-VC.
  - So the exclusion sentence of `mvc/mos/consent_template_v1.html` ("Who can take part", VI and EN) now also names the demo's users. Screening and analysis are unchanged.
  - The template's frozen sha256 in `LISTENING_FREEZE_mvc_v1.sha256` is superseded by this commit; the filled consent's sha256 is logged before the pilot, as §13 says.
- **Next: §12 step 7.** It waits for UIT's answer (Decision 26). Until then the live site stays in TEST DATA mode and the package is not imported.
