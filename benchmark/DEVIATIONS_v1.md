# Deviations from analysis plan v1 and the timestamps of §8.4 step 8

`benchmark/ANALYSIS_PLAN_v1.md` was frozen on 18 Oct 2026 in commit db9eedaf028e21d70dea1691f1ef2ccbdfb976ed. Its sha256 is 64430ad401ec2c512d6977820f2dfb3ae43aaa20bd603ed7d379f801f8ebe63e, and `FREEZE_v1.sha256` has sha256 d24fcb034bc694604a6f31672d4cc816e824254eed609389bb58adcdc2a0ca47. Every change made after the freeze is listed below with its date and reason.

## Timestamps (§8.4 steps 7–8)

| Event | Evidence | Time (+07:00) |
|---|---|---|
| Plan v1 public | Tag `analysis-plan-v1` (tag object cdff77ee20678108f8bcc66c0abf7be6466c0701) pushed to github.com/nguyenvietquoc2569/crt; GitHub `pushed_at` 2026-10-18T04:42:31Z | 2026-10-18 11:42:31 |
| D1 submitted | Job 4344 (`job_d1_conv.slurm`), `sacct` Submit | 2026-10-18 11:43:03 |
| Selection record public | Commit c9bc2aa, tag `dev-selection-record-v1` (tag object 097119be96490e5868f7423d3b4f62c52c3a2240) pushed to github.com/nguyenvietquoc2569/crt; GitHub `pushed_at` 2026-10-18T17:39:22Z | 2026-10-19 00:39:22 |
| T1 submitted | Job 4487 (`job_t1.slurm`), `sacct` Submit | 2026-10-19 01:55:43 |

## Deviations

1. **18 Oct 2026, about 15:00 (+07:00), operational: storage budget lifted.**
   - §8.2 says the project keeps its own total on `nvquoc` under about 90 GiB until the quota is known.
   - The corresponding author confirmed that storage on `nvquoc` is no longer limited, so that self-limit no longer applies.
   - Effects: the logit dumps of D1 and the test sets are kept instead of being considered for deletion after their decodes are checked, and the held LibriTTS-R train-clean-360 download resumes.
   - No setting, system, rule or result is affected.

2. **19 Oct 2026, about 01:00 (+07:00): the C1 identity check of `select_crt_al.py` now takes WAV identity first (§4.4, §8.1 item 9).**
   - **What happened:** run 1 of the selection with the frozen script (sha256 8ac8eec3a4c37343383556e8a88c14cd587f2b8c74eafa3bd5835fe34eeaad9f) ended INCOMPLETE for one reason only. `cand_pm_rq_T1` (C1 from the decoding lab) and its anchor-lab twin `anc_R3_s0_pm_rq_T1` had identical Whisper hypotheses on 999 of 1,000 pairs, while max |ΔSIM| = max |ΔUTMOS| = 0.
   - **Why this is not a mismatch:**
     - The WAVs of the two decodes are byte-identical on all 1,000 pairs: sha256 of every file, `results/v1/d2/c1_cand_wavs.sha256` and `results/v1/d1/c1_anchor_wavs.sha256`.
     - The one differing hypothesis (`dev__p236_154__p230`) comes from Whisper's batched decoding of identical audio in two different batches.
     - §8.1 item 9 checks sample identity first and uses the metrics tolerance only when the WAVs are not identical. The frozen script could only apply the metrics tolerance, because it never saw the WAVs.
   - **The fix:**
     - `select_crt_al.py` gains `--c1_wavs` (the two sha256 lists). When every common pair is byte-identical, the check passes by identity; otherwise the metrics tolerance applies as before.
     - Eight unit tests were added in `tests/test_select_crt_al.py` (sha256 1ca0f3421180e9e5d7436e43346b14db7cb99721b0472b1a10f2e0173a68c614); the suite now has 668 tests, all passing.
     - New `select_crt_al.py` sha256: 9405fb8bf0b73339bed21bfbfd42494a15d910c07a9cd9f7f5df0a28028bff46.
     - No rule, threshold, statistic or input changed.
   - **Both outputs are kept**, as §4.4 requires:
     - run 1, frozen script: `DEV_SELECTION_RECORD_v1_run1_frozen.md` (sha256 781b1acc5a8585b0d8298ec2899c7599d77f9d416015316f66614786387ed114), status INCOMPLETE;
     - run 2, fixed script: `DEV_SELECTION_RECORD_v1.md` (sha256 dafcf3004da07f9dad6b5b6e2b2ed906b8fb333ecc3d61643a47d747b0f106ea), status complete.
     - Every decision is identical in the two runs: τ_dev 0.9, S1u `us32_w0.2_0.2_0.1`, E17 enters, CRT-AL = C2 (950-pair check C2), VCTK-2k 5 targets, the same 32 listening items. Only the status and the C1-check fields differ.

3. **19 Oct 2026, operational: nvquoc hit a disk quota; logit dumps deleted with the author's approval; T3 re-run.**
   - **What happened:** from 04:56 (+07:00) every write on `nvquoc` failed with "Disk quota exceeded" (EDQUOT).
     - The account then held about 332,836 MiB: CRT 160,210, the interpreter project 139,770, Qwen 32,797, home 59. So the account is capped at about 325 GiB. The "no longer limited" of item 1 was wrong.
     - T3 (job 4490) died in the LTC-2k 3 s anchor lab inside `np.savez_compressed`; its `.err` stops at exactly 8,192 bytes. Every later LTC-2k step of that job failed. The LibriTTS-R train-clean-360 download failed at the same moment.
   - **Space freed:**
     - 05:31: the partial train-clean-360 archive, 11,106,287,700 bytes. That download is not used by this plan.
     - About 08:33: the logit dumps of D1, T1 and T2, 9,600 files, 41.8 GiB. §8.2 allows this once the decodes that read them have been checked, but only with the user's approval; the corresponding author gave it that morning.
     - The files are listed with sizes and times in `results/v1/deleted_dumps_20261019.tsv` (sha256 72e2ee5f87c8938e3e30279a8e78de34175d6236c3d0f63e5bc8985ad2ed71b1).
     - Every decode that read them was complete. D2: C1–C5, S1u, E17. T1 and T2: 2,000 and 1,800 WAVs for each of `pm18_rq`, `lmbr_k2_g1_T1` and `us32_w0.2_0.2_0.1`. Each run's `anchor_summary.json` exists.
     - The decoded WAVs, manifests and metrics are kept. The dumps can be regenerated from the frozen code and weights.
     - The author also approved deleting T3's dumps once its decodes are checked.
   - **T3 re-runs** use `job_t3.slurm` unchanged. RUNBOOK_v1.md, "Rules for the account": the conversion and anchor steps resume and `crt_decode_lab.py` reruns whole.
     - Job 4506 (Submit 08:34:41) completed VCTK-2k at 3 s and 180 s, and LTC-2k at 180 s: 1,699 decodes plus the 101 expected reference shortfalls.
     - It failed again in the LTC-2k 3 s anchor lab on EDQUOT at 08:45:57. That was brief: the job wrote about 8.5 GiB right after. A likely cause is that the 08:33 deletion was credited to the quota late.
     - Job 4617 (Submit 15:37:47) re-runs it.
     - Slurm planned it for 17:25, because the free MIG slices were held for a higher-priority job and a 72 h limit cannot backfill. Its limit was therefore lowered to 1:15:00 (`scontrol update`; the script is unchanged), and it started at 15:58:05. The remaining work is about 30 min; job 4506 did the comparable steps in 32 min.
     - Job 4617 completed at 16:38:10 with no failed step. All decodes are complete: VCTK-2k 2,000 at 3 s and 2,000 at 180 s; LTC-2k 1,800 at 3 s, and 1,699 at 180 s plus the 101 shortfalls. `t3.done` carries selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105.
     - T3's anchor-lab logit dumps were then deleted, with the author's approval as above: 7,499 files, 34.1 GiB, listed in `results/v1/deleted_dumps_t3_20261019.tsv` (sha256 7545b7c827520239dee5269985587de7cebb11010af7399881a6f2f947f990d2). T3's run_conversions dumps (codes, no logits) are kept.
     - T4 (job 4626) was submitted at 17:33:23.
   - No setting, system, rule or result is affected.

4. **19 Oct 2026, operational, in progress: CRT moves from `nvquoc` to `ceai`.**
   - **The author's instruction, the same morning:** "move everything to ceai account which has just been raised to 1.2TB. Then delete everything on nvquoc. Nvquoc from now will be used to run gwen and other support model to support Q2 journal target".
     - A 50 GiB write test on ceai passed at 08:38.
     - CRT's new home is `/datastore/ceai/crt`. The interpreter project moves to `/datastore/ceai/p1`.
   - **T3 and T4 run on `nvquoc` with the frozen scripts.** They cannot move first: `crt_anchor_lab.py`'s resume check (`run_cfg`: `weights`, `conv_dir`) and the baselines' `conv_cfg_sha` hold absolute paths, and T4's rows must keep the smoke test's `conv_cfg_sha` (RUNBOOK_v1.md §7). Job M runs on `ceai`.
   - **The copy:** a tar relay through the author's Mac, nvquoc → Mac → ceai, because the two accounts cannot read each other's folders. It is checked by comparing the (path, type, size, mtime, link target) lists of both sides.
   - **New files:**
     - `move_paths.py`, with 23 tests: rewrites `/datastore/nvquoc/crt` → `/datastore/ceai/crt` in every text file and symlink, logs each change, and checks the logged hashes and the section-2 hashes of `FREEZE_v1.sha256` mapped back.
     - `job_move_ceai.slurm`: runs that rewrite and check, compares `pip freeze` with `FREEZE_v1_pip_freeze.txt`, and re-scores the first 32 rows of the D1 anchor-lab manifest on ceai against D1.
     - `submit_ceai.sh`: `submit_nvquoc.sh` with the account, log and lock changed, shared with the interpreter project. The tests run on both wrappers and check that they do not drift apart.
   - **Changed frozen files**, only to name the new home:
     - In every `job_*.slurm` and in `setup_baselines.sh`, `/datastore/nvquoc/crt` becomes `/datastore/ceai/crt`.
     - The stale-path guards now refuse pair lists that hold `/datastore/nvquoc`, `/datastore/ceai/nvquoc` or `/datastore/eaic` paths, instead of `/datastore/ceai`. Their messages and the "submit with" comments change accordingly.
     - `tests/test_job_scripts.py` follows.
     - `RUNBOOK_v1.md` gains §8.
     - No setting, system, rule or statistic changes.
   - **Also new:** `analyze_tests.py`, the one-shot test-set statistics of §5 and §7, with 48 tests. It was checked against this plan by an independent review before any test metric existed.
   - **The copy and the move job on ceai, done:**
     - Phase 1 (everything T3/T4 did not write) and then T3 and T4 were relayed. `relay_copy.sh verify` showed no difference: 441,030 and 22,725 entries.
     - `job_move_ceai.slurm` (job 4654, 21:17–21:37) rewrote 285 text files (`v1/move/move_paths_4654.tsv`). The check found 0 problems: every logged hash held, mapped back, and all 16 section-2 files of `FREEZE_v1.sha256` have their frozen hashes.
     - `pip freeze` of the copied `envs/mimi_vc` is identical to `FREEZE_v1_pip_freeze.txt`, and CUDA runs on an H200 MIG 2g.35gb slice.
     - The first 32 rows of the D1 anchor-lab manifest, scored again on ceai, equal the D1 rows exactly: the largest |difference| is 0 for dB-MCD, the F0 metrics, SIM, SECS, UTMOS and every Whisper count, and all 32 hypotheses are identical.
   - **Change of plan the same evening, by the author:** the Qwen server stayed on `ceai`, where it holds one of the account's 3 job slots.
     - Job M therefore runs on `nvquoc`'s free slots with the frozen scripts, instead of on `ceai` with one slot.
     - The parts are chained one submit per hour through `submit_nvquoc.sh` (M t1: job 4653, Submit 20:56:20).
     - Their time limits were lowered to 14 h (t1) and 12 h (t2–t4) by `scontrol update`, so that backfill can start them next to reservations. `metrics.py` resumes, so a part that hits its limit is resubmitted unchanged.
     - The move job runs on `ceai` before M ends. T4's outputs are copied first, and M's outputs and the job logs follow, with a second rewrite run (`move_paths.py` takes several logs).
   - **Listening test (§5.4).** The author recruits the raters and chose an offline web page.
     - `listening_stimuli.py` (18 tests) and `job_listening_stimuli.slurm` (ceai) build the stimuli. They use the record's 32 items with its replacement rule. Natural target speech is P, else the target's E utterance closest in length to the source; the SMOS reference is another E utterance. They also make the Mimi copies (float16 on the GPU), set the loudness to an active-speech RMS of −26 dBFS, and write the clips under ids that hide the condition, with a separate key.
     - `listening/page/` is the rating page. `listening_analysis.py` (10 tests) applies the pre-registered screening, then computes MOS / SMOS / CMOS with a crossed rater × item bootstrap; it writes `cmos.json` for `analyze_tests.py --cmos` and gives the pilot's rater count.
     - The loudness normalisation and the E-utterance choice are protocol details §5.4 leaves open. They are fixed here, before any rating.
   - The relay's final check, the rewrite and environment checks, and the deletion of `/datastore/nvquoc/crt` (never `/datastore/nvquoc/qwen`) are recorded here once done.
   - **21 Oct 2026: a gap in the rewrite, found by job X1 and fixed.**
     - `move_paths.py` listed `.pth` among the binary suffixes, meant for torch checkpoints. It therefore skipped Python's site-packages path files, and `check` skipped them too.
     - One such file kept the old home: `envs/seedvc/lib/python3.12/site-packages/mimi_vc_base.pth` still named `/datastore/nvquoc/crt/envs/mimi_vc/lib/python3.12/site-packages`, so the Seed-VC env on ceai could not import librosa.
     - A scan of every `.pth` file under 64 KiB in `/datastore/ceai/crt` (3 files) found no other.
     - That file was rewritten by hand at 04:52 (sha256 c7b50949… → 0a69fa8d…).
     - `.pth` was removed from the binary suffixes: torch checkpoints still count as binary through the NUL test. A test covers both cases (`move_paths.py` sha256 474d7eb0…, pushed to ceai).
     - No completed job used the Seed-VC env on ceai: T4 ran on nvquoc, and M t4, the listening stimuli and the move job use `envs/mimi_vc`. So no result is affected.

5. **21 Oct 2026, exploratory addition X1, decided after the test results: FreeVC, Seed-VC and FACodec at the 3 s budget.**
   - **The author's decision.** Asked whether to run the other three baselines at 3 s as a clearly labelled exploratory addition, the corresponding author answered "You suggest it" (21 Oct 2026).
   - **Why.** In the budget sweep (E16), CRT-AL's capped WER is flat from 3 s to 180 s, while kNN-VC rises to 60.27 % (VCTK-2k) and 34.57 % (LTC-2k) at 3 s. A reader of that figure will ask how the other baselines behave with a 3 s reference, and the plan does not answer it.
   - **What runs.** `job_x1.slurm` converts FreeVC, Seed-VC and FACodec at 3 s on VCTK-2k (the record's 5-target list) and LTC-2k. It uses the adapters' fixed settings, exactly as T4 converts kNN-VC at 3 s, and scores the outputs as job M part t4 does, into the new directory `v1/x1`. It runs only after `t4.done` and `m/t4/m.done`, under the current selection.env.
   - **What is computed, fixed before any X1 output exists.** `analyze_exploratory.py` (10 tests, plus a job-script test) runs the comparison of `analyze_tests.py`, imported unchanged, at 3 s:
     - Δ = CRT-AL − baseline on the six endpoints of family S (capped WER, SIM and UTMOS on both sets), for kNN-VC, FreeVC, Seed-VC and FACodec: 24 comparisons, with the plan's crossed bootstrap.
     - Holm over the 24 is printed as a guide only.
     - Sensitivity: uncapped WER.
     - Descriptive: every system's 3 s levels (WER, CER, SIM, UTMOS, SIM_src, M9 with the share D > 0) and EER.
     - Output: `EXPLORATORY_X1_v1.{md,json}`.
   - **Status of the result: exploratory.**
     - kNN-VC's 3 s outputs had already been seen.
     - X1 changes no §7 decision and is reported whatever it shows. None of its numbers is presented as a pre-registered result.
     - It adds no CRT-AL system, decoder or setting, and produces no CRT output.
   - **Submitted.** Job 4955 on ceai through `submit_ceai.sh` (Submit 2026-10-21 04:34:05 +07, the interpreter session told before). `job_x1.slurm` sha256 62e74151c7b70c57efa3a6d2df881832b14ddfd4833c1741c65608745d7127dc on both sides. It started at once on h200node01, with selection sha256 9c7275f5…bc105, the same as `t4.done` and `m/t4/m.done`.
   - **To check after the run.** On ceai, the adapters' `conv_cfg_sha` differ from T4's, because their configs hold the absolute home path. The `conv_cfg` of X1's `runs.jsonl` must equal T4's once `/datastore/ceai/crt` is mapped back to `/datastore/nvquoc/crt`.
   - **First run, 21 Oct 2026.** Job 4955 converted FreeVC and FACodec at 3 s on both sets: 2,000 and 1,800 pairs, 0 errors.
     - Seed-VC stopped at import (`No module named 'librosa'`), because of the stale `.pth` path described in item 4. The job's completeness check then stopped it before any scoring, so no X1 metric exists from it.
     - After the fix, `job_x1.slurm` is resubmitted unchanged (tmux `crt_x1_resubmit` on login01, through `submit_ceai.sh`). `run_baselines.py` resumes, so only Seed-VC is converted again before all three systems are scored.
     - The first wrapper refused at 05:35:57 and submitted nothing: its tmux shell had no Slurm on its PATH. It was restarted at 05:44 under a login shell and waits for the 1 h gap after the interpreter's job 4958 (Submit 05:35:57).

6. **21 Oct 2026, listening test (§5.4), before any rating: background questions and the consent step.**
   - **The author's instruction.** On the listening test: "Make the reviewers happy".
   - **Background questions on the page.** After the last rating and before the download, the page asks five questions (`BACKGROUND` in `listening/page/lt_core.js`): age group, first language, understanding of spoken English, known hearing problem, and listening device.
     - Every question offers "Prefer not to say", and there is no free text, so nothing identifies a rater.
     - The answers go into the results file as `background`. `finished` stays the time of the last rating, so the 60-minute rule of §5.4 is unchanged.
     - `listening_analysis.py` reports the counts over the kept raters, descriptively, in the final and pilot reports.
     - No screening rule, condition, item, scale or statistic changes.
   - **Consent text.** `listening/consent_v1.html` is a draft in English and Vietnamese for the author to approve.
     - It keeps four draft fields to fill or delete: contact, compensation, ethics, and their Vietnamese twins.
     - `listening_stimuli.py config` writes the approved text and the contact into a built package's `lt_data.js` without touching the clips or trials. It refuses the text while a draft field is left.
   - **Tests.**
     - The Node tests of the page: 18 pass.
     - The pytest tests of `listening_analysis.py` and `listening_stimuli.py`: 38 pass, including the new ones for the background counts, the key list shared with the page, and `config` with its refusals.
     - The questions screen was checked in a browser: Continue stays disabled until all five are answered, and the answers are saved.
