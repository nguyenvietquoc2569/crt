# Deviations from analysis plan v1 and the timestamps of §8.4 step 8

`benchmark/ANALYSIS_PLAN_v1.md` was frozen on 18 Oct 2026 in commit db9eedaf028e21d70dea1691f1ef2ccbdfb976ed. Its sha256 is 64430ad401ec2c512d6977820f2dfb3ae43aaa20bd603ed7d379f801f8ebe63e, and `FREEZE_v1.sha256` has sha256 d24fcb034bc694604a6f31672d4cc816e824254eed609389bb58adcdc2a0ca47. Every change made after the freeze is listed below with its date and reason.

## Timestamps (§8.4 steps 7–8)

| Event | Evidence | Time (+07:00) |
|---|---|---|
| Plan v1 public | Tag `analysis-plan-v1` (tag object cdff77ee20678108f8bcc66c0abf7be6466c0701) pushed to github.com/nguyenvietquoc2569/crt; GitHub `pushed_at` 2026-10-18T04:42:31Z | 2026-10-18 11:42:31 |
| D1 submitted | Job 4344 (`job_d1_conv.slurm`), `sacct` Submit | 2026-10-18 11:43:03 |
| Selection record public | – | – |
| T1 submitted | – | – |

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
