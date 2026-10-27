# MimiFlow-VC (paper 3): analysis plan v1

Status: frozen on 27 Oct 2026, 16:28 (+07:00) by Quoc Nguyen, the corresponding author, who approved the text and the decisions in §11. Before the freeze it was drafted on 27 Oct 2026 and revised after a statistics review and a consistency review. The freeze comes before the first R2 job is submitted; R1 ends at about 02:00 on 28 Oct 2026. Nothing in this plan has been run. From the freeze on, every change is a deviation (§10).

File: `mvc/ANALYSIS_PLAN_mvc_v1.md`. `mvc/PLAN.md` applies where this plan says nothing. Where the two differ, this plan applies, and §0.3 lists the difference. The journal's frozen statistics are reused without change: `benchmark/stats.py` (sha256 a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6) and `benchmark/analyze_tests.py` (sha256 03f2bbf76f79338a46c0ad191b59979c6358dd1f6c7aaee908508a927cf2a933).

Paths. On ceai: `J=/datastore/ceai/crt/mimi_vc/benchmark/journal`, `DS=/datastore/ceai/crt/mimi_vc/data_storage`, `MVC=$DS/mvc`. On the Mac: `V1=paper/journal_extension/results/v1` (journal results) and `MR=paper/mimi_vc/results` (new, for this plan).

## 0. Lineage

### 0.1 Relation to PLAN.md and to the journal plan

PLAN.md §5.3 asks for this file and requires it to be frozen before R2. It follows `benchmark/ANALYSIS_PLAN_v1.md` (the journal plan): the same pair lists, metrics, crossed bootstrap, Holm procedure, failure rules and one-shot test. The journal's test rows (job M, X1, X2) are the comparator rows. They are read and never re-scored or re-converted.

### 0.2 What was seen before this plan

1. Every Dev-S100 result of MimiFlow-VC: the gate record P0–P6, deviations D-mvc-1 and D-mvc-2, "D-mvc-2 applied" (job e3, R1 at 100k), job e4 (100k). Job e5 (R1 at 150k, submitted 15:54) was running at the freeze and none of its numbers had been read; training tips C26–C37. The job lists `mvc/jobs/*_evals.txt` alone name at least 83 Dev configurations: e1 8, e2 9, e3 7, e4 4, e5 4, p6 sweep 18, p8 22, pilot 6, r1snap40 3, r5pre 2. The P5, P6, N1 and R5 jobs add more. The selection record gives the exact count (§4.8).
2. The four baselines and CRT-AL on Dev-S100 (P3).
3. Leakage at R1 100k under both definitions of §3.3, computed for this plan from the job e3 rows (Dev-S100, 25 pairs with P per stratum, linked bootstrap):

   | SNG w (CFG 0) | summarize.py value, same / cross | paired NL, same / cross (SE) |
   |---|---|---|
   | 0 | +0.050 / +0.096 | +0.078 (0.026) / +0.097 (0.048) |
   | 0.5 | +0.000 / +0.047 | +0.023 (0.027) / +0.038 (0.052) |
   | 0.75 | −0.022 / +0.031 | +0.002 (0.026) / +0.019 (0.051) |
   | 1.0 | −0.042 / +0.016 | −0.017 (0.026) / −0.001 (0.051) |
   | 1.0, t < 0.5 | −0.033 / +0.022 | −0.009 (0.027) / +0.007 (0.050) |

   Under the paired definition, w 0.75 would have met D-mvc-2 condition 1 at 100k; under the summarize.py value it failed by 0.001. The switch to the paired definition (§0.3 row 4) is therefore not neutral for the D-mvc-2 pick, and it was made with this table known. It is made for the duration confound of §3.3 and before R1's final snapshot is scored.
4. The journal's test results (`TEST_RESULTS_v1.md`, `EXPLORATORY_X1_v1.md`, `EXPLORATORY_X2_v1.md`). The standard errors of §5.5, the baselines' paired NL and the ECAPA coverages of §3.6 were computed from those rows for this plan. They contain no MimiFlow-VC output.
5. The author's blind A/B of relax (one listener, the author; relax preferred 9 times, never the other way, 11 ties; possible order effect; C32). It is Dev evidence and is never cited as a listening result (§6.4).

No MimiFlow-VC output exists on full Dev, VCTK-2k or LTC-2k.

### 0.3 Deviations from PLAN.md

| No. | PLAN.md | This plan | Reason |
|---|---|---|---|
| 1 | §5.1 example: `"ckpt":"ema_200000.pt"` | `ema_0200000.pt`, or `ema_0150000.pt` by R2.0 (§4.2); path and sha256 in the selection record | Erratum: `train.py:417` writes `ema_{step:07d}.pt` (`ema_last.pt` at l. 424, the adapter's default) |
| 2 | §2.7 and the §5.1 command: temp 1, CFG w 0.7, no relax | temp 0, Mimi relax 0.15 × 2, CFG w ∈ {0, 1}, source-negative guidance (SNG); every adapter key written out (§2.1) | C28, C30 (temp 0: Dev-S100 UTMOS +1.14), C32 (relax: +0.15), D-mvc-2 and C36 |
| 3 | R2 step 1: w ∈ {0.5, 1, 2}; Q32; source perturbation K = 2; NFE 32; 180 s full + scaling | The D-mvc-2 addendum grid, SNG {0, 0.5, 0.75, 1} × CFG {0, 1}, plus the interval variant (§4.3) | CFG ≥ 2 is banned by the addendum. Source perturbation raised leakage in e2 (cross +0.119 and +0.129 against +0.090; C35), and K-draw averaging does not exist (`perturb_src` is one draw, `vc_adapter.py:176`). No logit scaling exists in `model.py`, `train.py` or `vc_adapter.py`. NFE 32 moved UTMOS by +0.02 at temp 0 (C30). Q32 is a T1 system of its own (§2.2) |
| 4 | D-mvc-2 condition 1 uses `summarize.py`'s `sim_src_minus_target_copy` | The paired natural-level leakage NL (§3.3) in condition 1, in R2 and on the test sets; the summarize.py value is reported and decides nothing | Duration confound (§3.3). Known when written: §0.2 item 3 |
| 5 | D-mvc-2 decides the SNG weight on the final snapshot | D-mvc-2 conditions 1–5 become the Dev-S100 screen of R2; condition 6 is checked on full Dev (R2.4); R2 decides on full Dev | The two rules can pick different weights. The Dev-S100 leakage SE is 0.026–0.052 |
| 6 | R2 rule: leakage ≤ +0.03 per stratum, "best" sets undefined | NL ≤ +0.03 − SE_g per stratum on full Dev; the "best" sets, the tie-breaks and the binding WER check are defined; the rule is repeated on Dev-900 (§4.5) | Winner's curse (§4.9) |
| 7 | R4 if leakage > +0.03 or "the gate-L margin is worse than kNN-VC's"; risk 2 order: source perturbation → R4 → E1 → E2 | B1 (D-mvc-2's fallback) is the only training-side remedy before T1 (§4.7). The kNN-VC part of the trigger is dropped. R4, E1 and E2 are not run before T1 | kNN-VC itself leaks (VCTK-2k paired NL +0.049 / +0.055) and has no full-Dev rows. The leak runs through the Whisper stream (C35), which B1 changes and R4 does not |
| 8 | R5 if Dev UTMOS < mimi_copy_src − 0.3 | Level named: the journal's `mimi_copy_src` on full Dev, 3.82. R5 is not run whatever the trigger shows | R5b failed (C33) |
| 9 | §2.6: extend to 300k if Dev-S100 still improves between 150k and 200k | No extension of the run that defines mvc_v1. At the author's request, an exploratory continuation R1x to 300k (§4.10) runs if SIM or UTMOS rose from 150k to 200k; it is never eligible to become mvc_v1 | The learning rate ends its cosine tail at 200k (1.5e-5), so 200k–300k does not continue the same schedule; R1x restarts it (§4.10). The 150k numbers were not read at the freeze |
| 10 | §5.3 H2: four baselines, same endpoints plus SIM_src, "Holm is a guide" | Family S: 24 hypotheses without SIM_src, Holm over 24 binding; SIM_src and M9 against baselines are descriptive | Claims against strong baselines need error control. A lower SIM_src can come from a degraded voice |
| 11 | §5.3 H3: leakage per stratum, gate L against kNN-VC, M9, ECAPA coverage | NL test on VCTK-2k (§5.1), family L relative to kNN-VC, M9 descriptive, coverage formula (§3.6) | Definitions were missing |
| 12 | §5.3 H4 and S | Families RB (budgets) and Q (decode route), and three non-inferiority checks (§5.1) | Not defined |
| 13 | §6.1 tiers, "Seed-VC parity", P(must) ≈ 0.85 | Tiers are point-estimate benchmarks; "parity" only from family NI; P(must) withdrawn (§7.3) | §5.5 |
| 14 | §5.1: `mvc_copy_latent` is "the codec ceiling of the deployed path" | Continuous fp32 decode of the source's own latent, without relax | In control modes `convert()` returns before `convert_latent` (`vc_adapter.py:147–157`), and relax runs only at l. 204. Relax on real latents costs 0.07 UTMOS (C32) |
| 15 | §3.1 and R1: ≈ 555 h | R1 trained on 501 h of LibriTTS-R, 2,274 speakers | As trained (C34) |
| 16 | D-mvc-2: "R1 is always also reported without SNG" | Full Dev only (R2), not a T1 system | Keeps T1 to the systems of §5.4 |
| 17 | §6.2 item 5: MOS/SMOS listening test | Not pre-registered here (§6.4) | – |
| 18 | §5.1: `hypothesis` for q ∈ {wer, sim, utmos, mcd, sim_src, m9} | dB-MCD, SIM_src and M9 are descriptive | H1 is six tests (§5.1) |

## 1. Sets

All lists come from the journal (`make_pairs.py`, seed 20260928). The VCTK-2k list is the selection record's 5-target variant.

| Set | Pairs | `wer_primary` | Same / cross | Pairs with P, same / cross | Speakers | Cells | Bootstrap |
|---|---|---|---|---|---|---|---|
| Dev-S100 (`mvc/pairs/dev_s100.txt`) | 100, 25 per gender pair | 50 | 50 / 50 | 25 / 25 | 20, both roles | – | linked |
| Full Dev (`$J/pairs/dev.jsonl`) | 1,000 | 500 | 500 / 500 | 252 / 248 | 20, both roles | 380 | linked |
| Dev-900 (full Dev minus Dev-S100) | 900 | 450 | 450 / 450 | 227 / 223 | 20 | – | linked |
| VCTK-2k (`$J/pairs/vctk2k.jsonl`) | 2,000 | 1,020 | 1,000 / 1,000 | 491 / 489 (242 / 243 distinct P recordings, 24 sentences) | 20 sources × 20 targets, disjoint | 400 | independent |
| LTC-2k (`$J/pairs/ltc2k.jsonl`) | 1,800; 1,699 at 180 s | 1,800 | 900 / 900 | none | 36, both roles | 1,260 | linked; independent as a sensitivity analysis |

- The Mac copies have sha256 bdbf36a3… (VCTK-2k), 32fc9205… (LTC-2k) and ccbd53df… (VCTK-2k source-E file), as listed in `TEST_RESULTS_v1.md`. The ceai copies differ only in the path prefix that `move_paths.py` rewrote (DEVIATIONS_v1 item 4). The selection record lists the ceai sha256 values.
- Budgets of 3, 30 and 180 s are nested and cut to exactly that length (`make_pairs.load_reference`). 30 s is primary.
- LTC-2k at 180 s: 101 pairs, those of targets 3729 (75.04 s available) and 6930 (105.0 s), have a reference shortfall. They are removed from every 180 s LTC-2k pair list before any comparison, for every system, so they are neither failures nor imputed (§5.4).
- Gender strata: same = F2F ∪ M2M, cross = F2M ∪ M2F, from the `genders` field.
- Dev-S100 is a subset of full Dev. VCTK-2k and LTC-2k have never been converted by MimiFlow-VC.

## 2. Systems

### 2.1 mvc_v1, the primary system

- Model: the R1 run in `$MVC/runs/r1`. It is a 121M DiT (its `args.json`), trained to 200k steps on 501 h of LibriTTS-R (train-clean-100, train-clean-360 and train-other-500, without the 20 validation speakers and the A0 audit exclusions). The EMA snapshot is chosen by R2.0 (§4.2). If B1 is kept (§4.7), the model is B1's. The row name is `mvc_v1` whatever the lineage; the record states it.
- Conversion: `mvc/run_mvc.py` through the journal harness (`run_baselines.convert_pair`, the reference `make_pairs.load_reference(R[budget])`, the per-pair seed `seed_all(int(sha256(pair_id)[:8], 16))`). The ODE noise seed is drawn from that seeded generator, so every pair has its own fixed seed. temp 0 zeroes only the target frames' initial noise (C28). Bitwise reproduction across GPU kernels is not claimed.
- Inference settings, written out with every key of `vc_adapter.DEFAULTS` so that a later change of a default cannot alter the system. Values in angle brackets are fixed by R2 and written into the selection record:

  ```
  {"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "<ema_0200000.pt | ema_0150000.pt>",
   "steps": 16, "cfg": <0.0 | 1.0>, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false,
   "decode": "latent", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler",
   "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null,
   "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std",
   "neg_src": <0.0 | 0.5 | 0.75 | 1.0>, "neg_src_until": <1.0 | 0.5>}
  ```

  `run_dir` becomes B1's run directory if B1 is kept. Test runs never set `stats_path`.
- References: the whole budget is the prompt, cut into chunks of at most 750 frames whose velocities are averaged. So 3 s and 30 s are one chunk, and 180 s (2,250 frames) is three 60 s chunks. No other 180 s mode is run. The CAM++ vector is the renormalised mean of 10 s windows.
- Mimi `kyutai/mimi` commit 89091b3e466eb6a9d11e537bf26b144f194978f7: encoding in fp32, continuous decoding in fp32. The journal's Mimi copies decode 32 codes in fp16; `mvc_copy_latent` and the Q32 route measure the difference (§2.2).
- Code identity: `describe()` records a hash of every `mvc/*.py` file (`commit`). The selection record stores it, and every T1 manifest's `conv_cfg` must carry the same value. `mvc/*.py` does not change between the selection record and the end of the T1 conversions. For that reason the new analysis scripts live in `mvc/analysis/` (§5.8), outside that hash.

### 2.2 The other T1 systems (30 s only)

- `mvc_v1_q32` (secondary): the JSON of mvc_v1 with `"decode": "codes"`. The relaxed latent is re-quantised to 32 Mimi codes and decoded by `MimiModel.decode` in fp16, as the journal's Mimi copies are, so the output is 32 genuine Mimi tokens.
- `mvc_copy_latent` (descriptive ceiling): every key at its default except `"mode": "copy_latent"`. It is the continuous fp32 decode of the source's own latent, without relax and without the model. It is not the ceiling of the deployed path (§0.3 row 14).

### 2.3 Comparators (journal rows, read only)

| Row name | System | Rows (Mac `V1/...`; ceai `$J/v1/...`) | Role |
|---|---|---|---|
| `cand_pm18_rq` | CRT-AL (C2 of the journal record); never called `crt_al` in rows | `m/t1` (VCTK-2k 30 s), `m/t2` (LTC-2k 30 s), `m/t3` (3 s, 180 s) | H1; descriptive at 3 and 180 s |
| `knnvc` | kNN-VC | `m/t4` (30, 3 and 180 s) | S, NI, L; descriptive at 3 and 180 s |
| `freevc` | FreeVC, trained on VCTK ("VCTK-seen") | `m/t4` (30 s), `x1/m` (3 s) | S; its VCTK-2k rows carry the flag |
| `seedvc` | Seed-VC v1 small; sees only the first 25 s of R at 30 and 180 s | `m/t4` (30 s), `x1/m` (3 s) | S, NI |
| `facodec` | FACodec | `m/t4` (30 s), `x1/m` (3 s) | S |
| `faiss` | FAISS (published front end) | `m/t1`, `m/t2`, `m/t3` | descriptive |
| `source_copy`, `target_copy` | natural source; natural target recording P (VCTK-2k pairs with P only) | `m/t1`, `m/t2` | `target_copy@src` defines NL; levels |
| `mimi_copy_src`, `mimi_copy_P` | Mimi Q32 copies, fp16 | `m/t1`, `m/t2` | ceilings, descriptive |
| ECAPA rows of `cand_pm18_rq`, `faiss`, the four baselines, `source_copy` and `target_copy`, 30 s | speechbrain ECAPA-TDNN | `x2/ecapa/<set>_30{,_src}.jsonl` | §3.5, §6.1 |

On ceai, X2's ECAPA rows are under `$J/v1/x2/ecapa/`. Job M's rows carry the journal selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105 in every `m.done`, and X1's `x1.done` carries the same value.

### 2.4 Dev-only systems

These never run on VCTK-2k or LTC-2k: `mvc_r1_nosng` (the chosen snapshot with CFG 0 and SNG 0, run on full Dev in R2.3 as D-mvc-2's ablation and as the reference of the binding WER check); every R2 configuration that is not chosen; B1 if it is not kept; the R3 ablations.

## 3. Metrics

### 3.1 Scorers, identical to job M

- `benchmark/metrics.py` with sha256 22daec636e14cdee41e5af3328bd77fde4b4765c7d1bc1243a4ee28f2b71f1d4, its `FREEZE_v1.sha256` entry. Every call passes `--wavlm_model $J/models/wavlm-base-plus-sv`, because the default path (`metrics.py:79`) points to the deleted `/datastore/nvquoc` tree. UTMOS22-strong comes from job M's torch hub cache. `metrics.py` checks the Whisper large-v3 and base checkpoints by sha256. The environment is `/datastore/ceai/crt/envs/mimi_vc`, whose `pip freeze` equals `benchmark/FREEZE_v1_pip_freeze.txt`.
- Test-set calls: `--metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base` on target rows, and `--metrics sim,secs` on `@src` rows. Dev calls use `--whisper_models large-v3`.
- `benchmark/lab_rows.py` (sha256 135516ac5960c32d3cc6d152b0baad70fc7950ede58591a165c6b9ed70cec163) builds the target rows and their `@src` twins. VCTK-2k calls add `--source_e $J/pairs/vctk2k_source_E.json`; LTC-2k and Dev calls do not, because there every source speaker is also a target and its target E set serves.
- Every new scoring writes to a new `--out_jsonl`, because `metrics.py` reuses a row whose input fields are unchanged (RUNBOOK_v1 §0).
- Scorer drift check (§8.3) before any MimiFlow-VC test row is scored.

### 3.2 Quantities

- Capped WER (M3c): 100 · Σ_p min(S_p + D_p + I_p, N_p) / Σ_p N_p with Whisper-large-v3, over the `wer_primary` pairs (`stats` metric `wer_capped`). It is the WER of every rule and endpoint of this plan. Uncapped corpus WER is reported beside it and tested as a sensitivity analysis. Whisper-base WER is descriptive. Capped CER is descriptive.
- SIM: WavLM-base-plus-SV cosine between the output and the mean embedding of the target's E set, all pairs.
- SIM_src: the same against the source speaker's E set (`@src` rows).
- UTMOS: UTMOS22-strong, all pairs.
- dB-MCD and the F0 metrics against P (VCTK-2k pairs with P): descriptive.
- M9: D = SIM_src − SIM per pair, and the share of pairs with D > 0: descriptive.
- EER (`metrics.sv_trials`, WavLM-SV): descriptive.

### 3.3 Natural-level leakage NL

For a system X and stratum g:

NL_g(X) = mean over the stratum's pairs with a parallel recording P of [SIM_src(X) − SIM_src(`target_copy`)], paired per pair.

Both terms are WavLM-SV cosines to the same source-speaker E centroid, and `target_copy` is the target speaker's natural recording of the same sentence. NL is the excess of the output's similarity to the source over that of natural speech by the target speaker. Its CI and SE come from the crossed bootstrap over those pairs: independent on VCTK-2k, linked on Dev.

- Why paired. `summarize.py:97–103` subtracts `target_copy`'s mean over the P pairs from X's mean over all stratum pairs. On VCTK-2k the P pairs have longer sources (mean 6.13 s against 3.12 s), and every system's SIM_src is 0.021–0.040 higher on them. The summarize.py value therefore reads low. On the journal rows, Seed-VC's value is +0.039 / +0.055 (same / cross) against a paired NL of +0.051 / +0.065. On Dev-S100 the two definitions differ in both directions (§0.2 item 3).
- The summarize.py value is reported beside NL wherever NL appears, and it decides nothing.
- NL is used in D-mvc-2 condition 1 from the final-snapshot re-application on (R2.1), in R2 (R2.3) and on the test sets (§5.1). §0.3 row 4 records the change.
- Implementation: `stats.compare(rows['X@src'], rows['target_copy@src'], P pairs of g, spk, 'sim_wavlm', mode, worst='max')`, and the decision `stats.gate_l(d, margin=0.03)`. NL is never computed with `stats.gate_l_strata` against `target_copy`: that function runs over every stratum pair and has no margin argument, so on VCTK-2k the 509 same-gender and 511 cross-gender pairs without P would count as `target_copy` failures and produce a meaningless worst case. A unit test checks that `target_copy` has no failure on the P pairs (`fail_b` = 0), so a worst case can come only from failed conversions of X.
- NL exists only on VCTK-2k and Dev. LTC-2k has no P, so no NL. On LTC-2k leakage is tested only by family L against kNN-VC and described by M9. No substitute for NL on LTC-2k is defined, before or after unblinding.

Natural-level reference values on VCTK-2k (paired NL, same / cross, computed for this plan from the journal rows): kNN-VC +0.049 / +0.055, FreeVC +0.070 / +0.100, Seed-VC +0.051 / +0.065, FACodec +0.095 / +0.096, CRT-AL +0.001 / −0.014.

### 3.4 Gate-L quantity

Δ_src,g = SIM_src(mvc_v1) − SIM_src(`knnvc`) over all pairs of stratum g (VCTK-2k: pairs whose source has an E set; all 2,000 have one), paired. Both systems have every row, so `stats.gate_l_strata` (margin +0.01 from `stats.MARGIN_L`) applies as in the journal.

### 3.5 ECAPA

`benchmark/score_ecapa.py` (sha256 1f4d2aeeeeaaeb5a78362f5f8d740841fd5e0b93f70906fed59b80ccf3422b46) with speechbrain `spkrec-ecapa-voxceleb` at revision 0f99f2d0ebe89ac095bcc5903c4dd8f72b367286 in `$DS/models/spkrec-ecapa-voxceleb`, in the X2 site folder `$DS/site/x2_ecapa`. `sim_ecapa` is the cosine between the output's embedding and the mean of the L2-normalised embeddings of the row's E set, the rule `metrics.py` applies to WavLM-SV. Scored at 30 s on both sets, target and `@src` rows, for `mvc_v1`, `mvc_v1_q32` and `mvc_copy_latent`. The comparators' ECAPA rows are X2's, scored by the same script, model and environment.

### 3.6 ECAPA coverage

Coverage = (E_out − E_src) / (E_tgt − E_src), with E_out, E_src and E_tgt the mean ECAPA SIM to the target of mvc_v1, `source_copy` and `target_copy`. All three are taken over the same 980 VCTK-2k pairs with P, at 30 s.

- The CI comes from the independent crossed bootstrap: in each replicate the three weighted sums are formed from the cell sums (`stats.bootstrap`, stat (S0 − S1) / (S2 − S1)).
- The journal's 6 % for CRT-AL came from mixed pair sets ((0.161 − 0.121) / (0.795 − 0.121): output and source over all pairs, target over the P pairs), which carry the duration confound of §3.3. On the 980 pairs CRT-AL's coverage is 0.070, and that is the value reported next to mvc_v1's. The baselines on the same pairs, for reference: kNN-VC 0.83, FreeVC 0.75, Seed-VC 0.94, FACodec 0.56, FAISS 0.72.
- Coverage is reported only for VCTK-2k at 30 s.

### 3.7 DNSMOS

`mvc/dnsmos.py` (sha256 330b1d35ba6441f3aa3369d8dacd7c5e99eef9852134c37e10e5244a35b66a16), DNSMOS P.835 and P.808 of the Microsoft DNS Challenge, fields `SIG`, `BAK`, `OVRL` and `P808`, with `--norm_lufs -26`. The ONNX models are kept in two durable folders, `$DS/models/dnsmos/` on ceai and `paper/mimi_vc/models/dnsmos/` on the Mac, both listed in `FREEZE_mvc_v1.sha256`, with sha256 269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd (`sig_bak_ovr.onnx`) and 9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91 (`model_v8.onnx`). DNSMOS runs on the 30 s WAVs of every system of §5.1 on both sets: the MimiFlow-VC systems, `cand_pm18_rq`, the four baselines, `mimi_copy_src` and `source_copy`. The journal WAVs are scored by DNSMOS for the first time; nothing is converted again. It runs on the Mac: the WAVs named by the metric rows are copied read-only from ceai to `MR/wav/` with their absolute path kept below that prefix, and the rows are rewritten to point there (no audio is converted or edited). DNSMOS is descriptive, apart from the flag of §6.2.

### 3.8 Acoustic guards

`mvc/acoustic_guard.py` (sha256 12fafe9b0eeabf29885039578214727caa85b2d2c7ab04d0953b201e011c7c90): HNR, share of the source's voiced frames kept, F0 correlation with the source, semitone-SD ratio and level difference, for mvc_v1, `mvc_v1_q32`, `cand_pm18_rq` and the four baselines at 30 s. Descriptive.

## 4. Dev selection (R2)

### 4.1 Common rules

- Every R2 configuration is the JSON of §2.1 with only the values of the step changed, at 30 s unless the step says otherwise.
- Every R2 job uses a new `TAG` under `$MVC/eval/` that no earlier job used. `eval_dev.sh:50` scores the first `manifest*.jsonl` it lists, so a full-Dev run (`SEL=0/1`, which writes `manifest.shard0of1.jsonl`) in a directory that already holds a Dev-S100 `manifest.jsonl` would silently score 100 pairs. Before a run's numbers are used, its target and `@src` metrics files are counted: one row per pair, 100 or 1,000. A run with fewer rows is incomplete and is completed or repeated under a new `TAG`. Error rows are counted in the record.
- Dev quantities of a configuration c:
  - WER(c): capped WER over the `wer_primary` pairs;
  - SIM(c), UTMOS(c): means over all pairs;
  - NL_g(c) (§3.3), against the journal's Dev `target_copy@src` rows (`$J/v1/d2/controls/metrics_src.jsonl`), with its SE;
  - the ECAPA ΔSIM_src of condition 5 (as job e3 computed it).
  CIs and SEs come from the linked crossed bootstrap (B = 10,000, seed 20260928).
- Ties: WER is compared at two decimals (in %), SIM and UTMOS at three.
- `mvc/analysis/select_mvc.py` (§5.8) carries out the rules from the R2 metrics files copied to the Mac (sha256 checked) and prints every step. It is committed, with its tests, before the R2.1 screen is decided. Its last run re-derives every earlier stage from the files, and any difference from an earlier stage's output stops the selection.
- Cost: about 13–17 slice-h (PLAN.md says 6): R2.0 about 0.8, R2.1 about 1.7, five full-Dev configurations about 7.5, R2.4 about 3.5 per configuration checked.

### 4.2 R2.0: snapshot

- Candidates: `ema_0150000.pt` and `ema_0200000.pt` of R1. `ema_last.pt` holds the same weights as the 200k snapshot and is never named.
- Measured on both at the reference setting (temp 0, relax 0.15 × 2, CFG 0, SNG 1.0). Job e5's `r1k150_ns1.0` ran this setting at 150k on Dev-S100 and is used for 150k, because its JSON differs only in `stats_path`, which does not change the output, provided its manifest's `describe()` commit equals the current one; otherwise 150k is run again:
  - Dev-S100 WER, SIM and UTMOS;
  - same-speaker reconstruction capped WER on the 300 dev-clean utterances of PLAN §5.2 (`recon_eval`, temp 0, relax 0.15 × 2, CFG 0, no SNG).
- Rule: `ema_0200000.pt`, unless at least one of these holds, in which case `ema_0150000.pt`:
  - reconstruction WER(200k) > reconstruction WER(150k) + 1.0 point (risk 9, overfitting);
  - SIM(200k) < SIM(150k) − 0.010;
  - UTMOS(200k) < UTMOS(150k) − 0.05;
  - WER(200k) > WER(150k) + 2.0 points.
- Training and validation loss are never used (C5). There is no extension to 300k (§0.3 row 9).
- For the record: R1's 100k gate (PLAN R1: Dev-S100 WER ≤ 8 %, SIM ≥ 0.87) was passed with 2.34 % and 0.888 (CFG 0, SNG 0).

### 4.3 R2.1: Dev-S100 screen (the re-application of D-mvc-2)

- Grid on the chosen snapshot, eleven configurations run in one job, with ECAPA on target and `@src` rows:
  - eight plain cells c = (CFG, w) for CFG ∈ {0, 1} and SNG w ∈ {0, 0.5, 0.75, 1.0};
  - three interval variants at CFG 0 with `neg_src_until` 0.5, for w ∈ {0.5, 0.75, 1.0}.
- The D-mvc-2 conditions of a cell c with w > 0 are judged against the plain w 0 cell with the same CFG (c0):
  1. NL_g(c) ≤ +0.03 in both strata (paired, point estimate; §0.3 row 4);
  2. SIM(c) ≥ SIM(c0) − 0.005;
  3. UTMOS(c) ≥ UTMOS(c0) − 0.08;
  4. WER(c) − WER(c0) ≤ +2.0 points;
  5. the cross-gender ECAPA ΔSIM_src (c − c0) has the sign of the WavLM ΔSIM_src and at least half its size.
- Eligibility:
  - a plain cell with w > 0 is eligible when it meets conditions 1–5;
  - if a plain CFG 0 cell fails condition 3 and no other, its interval variant replaces it and is eligible when it meets 1–5 against the same c0; an interval variant is never eligible otherwise;
  - a w 0 cell is eligible when it meets condition 1 (2–5 then hold trivially).
- The D-mvc-2 pick is the smallest w with an eligible CFG 0 entry, plain or interval. It is reported as D-mvc-2's result on the final snapshot. The choice itself is made by R2.3.
- Condition 6 of D-mvc-2 (3 s and 180 s) is checked on full Dev in R2.4.
- If no cell is eligible, B1 is triggered (§4.7, trigger i), and R2.2 takes as finalists the four configurations with the smallest Dev-S100 value of max_g NL_g, so that R1 still has a defined choice.

### 4.4 R2.2: finalists

- The D-mvc-2 pick, if there is one, and up to three other eligible configurations. These are found by ordering the eligible configurations on their Dev-S100 numbers with the rule of R2.3: among those with SIM ≥ best − 0.010 and UTMOS ≥ best − 0.05 (bests over the eligible configurations), the lowest WER first, ties as in §4.1, then the next. If fewer than three meet the SIM and UTMOS bounds, the rest are taken by lowest WER.
- At most four finalists. Each runs on full Dev (1,000 pairs, 30 s, target and `@src` rows, ECAPA), and so does `mvc_r1_nosng` (§2.4).

### 4.5 R2.3: the choice on full Dev

For each finalist c, on full Dev:

- L-pass: NL_g(c) ≤ +0.03 − SE_g(c) in both strata, with SE_g(c) the linked-bootstrap SE over the Dev pairs with P (252 same, 248 cross). For scale, the paired full-Dev SEs are 0.006 / 0.012 (same / cross) for FAISS and 0.021 / 0.027 for CRT-AL, against 0.022–0.056 on Dev-S100 (computed for this plan from the journal Dev rows).
- Best SIM and best UTMOS: the maxima over the L-passing finalists.
- WER check, the "binding WER check on full Dev" that D-mvc-2 condition 4 refers to: WER(c) − WER(`mvc_r1_nosng`) ≤ +1.0 point.
- The eligible set E3: L-pass, SIM ≥ best SIM − 0.010, UTMOS ≥ best UTMOS − 0.05 and the WER check.
- Order of E3: the lowest WER first; ties → the higher SIM → the higher UTMOS → the smaller w → the smaller CFG → plain before interval.
- The choice is the first configuration of that order that passes R2.4 (at most two are checked; §4.6).
- Fallback F, when E3 is empty: among the finalists with SIM ≥ (the highest SIM of all finalists) − 0.010, the one with the smallest max_g NL_g; ties → the higher SIM → the smaller w → the smaller CFG. F also runs R2.4, for the record.
- The Dev-900 check: the whole of R2.3 is repeated on the 900 Dev pairs outside Dev-S100, with the same finalists. If it orders another configuration first, both are reported, and the choice made on 1,000 pairs stands.

### 4.6 R2.4: reference budgets

- The first configuration of E3's order runs at 3 s and 180 s on full Dev. It passes when, as point estimates, SIM(3 s) ≥ SIM(30 s) − 0.06 and SIM(180 s) ≥ SIM(30 s) − 0.005 (D-mvc-2 condition 6).
- If it fails, the second configuration of the order is checked the same way. At most two are checked. If neither passes, the first of the order is the choice, and the failure is recorded.
- The chosen configuration's full-Dev numbers at 3, 30 and 180 s are the Dev counterparts of family RB.

### 4.7 B1 and the other triggers

- B1 is triggered by (i) no eligible configuration in R2.1, or (ii) NL_g > +0.03 (point estimate, full Dev) in either stratum for the R2 choice or for fallback F. Trigger (ii) replaces PLAN R2's R4 trigger (§0.3 row 7).
- B1 is D-mvc-2's fallback: a shifted-content fine-tune of the R2.0 snapshot, 20k steps, with the Whisper features and CB0 taken from p6_V conversions of the training utterances to other speakers, p_shift 0.5.
- B1 keep rule (D-mvc-2), on full Dev, paired against its R1 parent at the R2 choice's inference setting (under trigger i: CFG 0, SNG 0): ΔSIM_src ≤ −0.02 in both strata, ΔSIM ≥ −0.005, ΔUTMOS ≥ −0.05 and capped WER change ≤ +0.5 point.
  - Kept: R2.1–R2.4 run once more with B1's final EMA snapshot (file and sha256 in the record; R2.0 is not repeated), and mvc_v1 is that choice, with B1's `run_dir`.
  - Not kept: mvc_v1 is R1's R2 choice, or fallback F.
- B1 runs at most once. If the second R2 meets a trigger again, nothing more starts: its choice, or its fallback F, is mvc_v1, and §7.2 reports the leakage. R4, E1, E2 and any setting outside the R2 grid are not tried before T1, unless a deviation is logged before any test conversion.
- R5 trigger, reported only: the chosen configuration's full-Dev UTMOS below 3.82 − 0.3 = 3.52, where 3.82 is the journal's `mimi_copy_src` on full Dev (`DEV_SELECTION_RECORD_v1.md`, "Controls"). R5 is not run (§0.3 row 8).

### 4.8 The selection record

`select_mvc.py` writes `mvc/MVC_SELECTION_RECORD_v1.{md,json}`. It holds:

- the snapshot file and its sha256, R2.0's numbers and decision, B1's trace and keep decision if it ran, and the model lineage of mvc_v1;
- the three frozen JSONs (mvc_v1, `mvc_v1_q32`, `mvc_copy_latent`) with their sha256, and the `describe()` commit hash;
- every R2.1 configuration's numbers with its condition flags, and the D-mvc-2 pick;
- for the finalists and `mvc_r1_nosng`, the full-Dev and Dev-900 tables: WER, SIM and UTMOS with CIs; NL with SE and the summarize.py value per stratum; ECAPA; M9 per stratum; the gender-pair table;
- the R2.3 and R2.4 traces, fallback F if used, and the B1 and R5 triggers;
- the number of Dev configurations ever scored: the distinct `summary.json` files under `$MVC/eval/` whose pairs are Dev-S100 or the Dev screen shard, by budget, plus the R2 runs;
- the sha256 of `mvc/analysis/analyze_mvc.py` and of its tests, which are written and reviewed before the record (§5.8);
- the result of the contamination test (PLAN §3.1; `mvc/tests/test_train.py`) on the training ID manifest of mvc_v1's model, with the manifest's sha256;
- where DNSMOS runs and how WAV paths are mapped;
- the ceai sha256 of the pair lists and of the source-E file;
- the job IDs and `sacct` Submit and End times of every R2 and B1 job, and the sha256 of every metrics file read.

The record is committed with its sha256 in the commit message and pushed before the first T1 job is submitted (§8.4). `mvc_selection.env` is written from it: `RECORD_SHA256`, the snapshot path and sha256, the three JSON files and their sha256, and `DESCRIBE_COMMIT`. The T1 jobs refuse to start without it, or when any of these values does not match.

### 4.9 Why these rules

- Winner's curse. Dev-S100 carried dozens of configurations (§0.2 item 1), and full Dev contains Dev-S100. Picking the smallest weight that passes makes the Dev leakage estimate biased low. Therefore the choice is made on full Dev, L-pass keeps one SE of distance from the margin, the choice is repeated on Dev-900, and every Dev number shown beside a test number is the Dev-900 value, labelled "selected on Dev". No Dev number is pooled with test data or used to explain a test result. Test leakage is expected to be higher than Dev leakage.
- D-mvc-2 as a screen. Its conditions were written before job e3 and are kept. Their Dev-S100 leakage estimates have SEs of 0.026–0.052 against a margin of 0.03, which is why they only screen.
- B1 before R4. The leak runs through the Whisper-small content stream (C35). B1 shifts that stream in training; R4's self-transformation replaces CB0 only.

### 4.10 R1x: exploratory continuation of R1 to 300k (author's request at the freeze)

- Purpose: to see how far the same model goes with more steps. R1x is exploratory. It never takes part in R2, never becomes mvc_v1 and is in no hypothesis family of §5.
- Snapshot first: `$MVC/runs/r1` is not written again after step 200k. Its `last.pt`, `args.json`, `stats.npz` and `ema_*.pt` stay as they are, and their sha256 go into the selection record. R1x runs in its own directory `$MVC/runs/r1x`, started from a copy of R1's `last.pt` (step 200k).
- Trigger, decided from R2.0's numbers at the reference setting (Dev-S100, temp 0, relax 0.15 × 2, CFG 0, SNG 1.0): SIM(200k) > SIM(150k) or UTMOS(200k) > UTMOS(150k), as point estimates. If neither rose, R1x does not run, and the record says so.
- Schedule: R1's arguments with `--steps 300000` (const_tail: 1.5e-4 constant to 240k, cosine to 1.5e-5 at 300k), resumed at step 200k with a linear re-warm of 2,000 steps (`--rewarm 2000 --rewarm_from 200000`). Snapshots every 10k as in R1.
- Evaluation: Dev-S100 at 250k and 300k, at the reference setting and at the inference setting R2 chose for mvc_v1, with ECAPA and the acoustic guards. Reported as exploratory, beside R1 at 200k.
- Test sets: only after T1 of mvc_v1 is complete and its results are recorded, R1x's 300k snapshot may run on VCTK-2k and LTC-2k at 30 s with mvc_v1's frozen inference settings, as the exploratory addition X-mvc-1. It is descriptive (CIs only), reported in its own section and never replaces or re-labels a mvc_v1 result. Whether to run it is decided before any of its Dev numbers are read beyond those of the 250k/300k Dev-S100 runs above.

## 5. Endpoints and statistics (T1)

### 5.1 Families

Δ = A − B per pair, over the endpoint's pairs where neither system failed. The six endpoints, in this order, are: 1 capped WER on VCTK-2k (1,020 `wer_primary` pairs), 2 capped WER on LTC-2k (1,800), 3 SIM on VCTK-2k (2,000), 4 SIM on LTC-2k (1,800), 5 UTMOS on VCTK-2k, 6 UTMOS on LTC-2k (the order of `analyze_tests.S_ENDPOINTS`). Budget 30 s unless stated.

| Family | Hypotheses | Test | Multiplicity | Role |
|---|---|---|---|---|
| H1 (primary) | mvc_v1 − `cand_pm18_rq` on endpoints 1–6: H1-1 … H1-6 | two-sided, α 0.05 | Holm over 6 | outcome (§7.1) |
| NL (natural level) | NL1s, NL1c: NL_g(mvc_v1) on VCTK-2k (491 / 489 pairs with P). H0: NL_g ≥ +0.03 | one-sided, α 0.025 each: rejected when the upper limit of the two-sided 95 % CI is below +0.03 | none; the claim needs both (intersection–union) | leakage row (§7.2) |
| L (relative to kNN-VC) | Δ_src,g = SIM_src(mvc_v1) − SIM_src(`knnvc`) (§3.4): L1s, L1c on VCTK-2k (independent), L2s, L2c on LTC-2k (linked). H0: Δ_src,g ≥ +0.01 | one-sided, α 0.025 each, CI rule with margin +0.01 | none; the claim needs all four | wording (§7.2) |
| S (confirmatory secondary) | mvc_v1 − b on endpoints 1–6, b ∈ {`knnvc`, `freevc`, `seedvc`, `facodec`}: S-b-1 … S-b-6 | two-sided, α 0.05 | Holm over 24 | claims against baselines (§7.3) |
| NI (non-inferiority) | mvc_v1 − b on endpoints 1–6, b ∈ {`seedvc`, `knnvc`}; margins: WER +1.0 point, SIM −0.010, UTMOS −0.10 | one-sided, p of §5.2 | Holm over 12 at α 0.025 | the only basis for "comparable", "matches" or "parity" |
| RB (reference budget) | mvc_v1(3 s) − mvc_v1(30 s) and mvc_v1(180 s) − mvc_v1(30 s) on endpoints 1–6, paired on common pairs: RB-3-1 … RB-3-6, RB-180-1 … RB-180-6. LTC-2k at 180 s and its 30 s comparator use the same 1,699 pairs | two-sided, α 0.05 | Holm over 12 | secondary |
| RBN (budget checks) | per set: RBN-1 SIM(3 s) − SIM(30 s) > −0.06; RBN-2 SIM(180 s) − SIM(30 s) > −0.005; RBN-3 WER(3 s) − WER(30 s) < +1.5 points | CI rule: lower limit above −0.06, lower limit above −0.005, upper limit below +1.5 | none; each reported | D-mvc-2 condition 6 and the Should tier (§7.3) |
| Q (decode route) | `mvc_v1_q32` − mvc_v1 on endpoints 1–6: Q-1 … Q-6 | two-sided, α 0.05 | Holm over 6 | secondary |
| Descriptive | everything in §5.7 | CIs only | – | – |

- The direction of a rejection follows `analyze_tests.favours`: lower WER, higher SIM and higher UTMOS favour a system.
- FreeVC's VCTK-2k rows carry "VCTK-seen". They stay in family S, so that m stays 24, but no claim of beating or matching FreeVC on VCTK-2k is made from them.
- SIM_src and M9 against the baselines are descriptive and are not in S, because a lower SIM_src can come from a degraded voice.
- Error control is per family. H1 decides the outcome; NL decides the leakage row; S, NI and L decide the wording against the baselines.

### 5.2 Tests and p-values

- Two-sided tests: p = `stats.p_two_sided` over the B replicates, p = min(1, 2 min(1 + #{Δ* ≤ 0}, 1 + #{Δ* ≥ 0}) / (B + 1)). Holm (`stats.holm`): sort the m p-values and reject the i-th smallest while p_(i) ≤ α / (m − i + 1). A hypothesis that cannot be computed counts in its m and is not rejected.
- Non-inferiority: for endpoint q with s_q = +1 (SIM, UTMOS) or −1 (WER) and margin δ_q (0.010, 0.10, 1.0 point), H0 is s_q Δ ≤ −δ_q, and p_NI = (1 + #{s_q Δ* ≤ −δ_q}) / (B + 1). Holm over the 12 at α 0.025. For a single test this equals the CI rule: the 2.5th percentile of Δ* above −δ_q for SIM and UTMOS, the 97.5th percentile below +δ_q for WER.
- NL, L and RBN are decided by the stated CI limit and are in no Holm family.
- The smallest attainable two-sided p is 2 / 10,001 = 0.0002, which is enough for Holm with m up to 250 at α 0.05; the smallest one-sided p is 1 / 10,001.
- Secondary statistics, as in the journal: the Wilcoxon signed-rank test and the matched-pairs rank-biserial r of every hypothesis, labelled possibly anti-conservative because pairs share speakers.

### 5.3 Bootstrap

`stats.py`, unchanged: crossed speaker bootstrap, B = 10,000, seed 20260928, 95 % percentile CIs, SE the standard deviation of the replicates.

- VCTK-2k: independent (20 source speakers and 20 target speakers drawn separately; 400 cells).
- LTC-2k and Dev: linked (one pool of 36, or 20, speakers whose draw counts serve both roles). LTC-2k with the independent bootstrap is a sensitivity analysis.
- A replicate holds every pair whose source and target were both drawn, weighted by the product of the draw counts. A replicate without a pair of positive weight, or without reference words, is redrawn, and the redraws are counted. Every call starts from the seed.
- Corpus WER differences are formed per replicate from the summed counts; the cap applies per pair, before the sums.
- Ratios (ECAPA coverage) are formed per replicate from the same weighted sums.

### 5.4 Pair sets and missing data

- A failed conversion is a missing row or a row with an error (`stats.failed`). It is left out of every comparison it enters, and failures are counted per system and set in the main tables. A row without error whose value is missing or not finite is left out as undefined.
- When two systems' failure counts differ, a worst-case analysis imputes the failing system's worst case: for WER every reference word of the pair deleted (E = N); for a continuous metric the failing system's worst finite value on that set (lowest SIM or UTMOS, highest SIM_src). Decisions that change are listed.
- LTC-2k at 180 s: the 101 shortfall pairs are removed from the pair list before `compare()`, for both systems and both budgets of a contrast, so 1,699 pairs enter and nothing is imputed for them.
- NL: only pairs with P. `target_copy@src` has a row on all 980 (checked by a test).
- VCTK-2k pairs whose source speaker has no source-E set are left out of every SIM_src quantity and counted (none are expected; all 20 sources have one).
- A T1 conversion job may be resubmitted unchanged (same code, JSON and snapshot) until every pair has been attempted, before any metric exists. `run_mvc.py` keeps successful conversions and retries error rows, so no successful conversion is ever made again. Errors that remain are failed conversions.

### 5.5 Power

The multiplier is z_{1−α/(2m)} + z_{0.80} for the two-sided families and z_{1−α/m} + z_{0.80} for NI: 3.48 for H1 and Q (m = 6), 3.92 for S (m = 24), 3.71 for NI (m = 12, one-sided, α 0.025) and for RB (m = 12, two-sided), and 2.80 for a single one-sided test at 0.025 (NL, L). `stats.MDE_Z` = 3.53 (Holm over 7) is not used.

The REML prediction of ANALYSIS_PLAN_v1 §5.3 is not used, because on the journal it came out too small against the realised bootstrap SEs (L1c: 0.0186 predicted, 0.028 realised; P3: 0.0125 against 0.0156). The SEs below are realised crossed-bootstrap SEs on the journal's test rows, computed for this plan.

| Family | Endpoint | SE proxy | VCTK-2k SE | VCTK-2k MDE | LTC-2k SE | LTC-2k MDE |
|---|---|---|---|---|---|---|
| H1 (× 3.48) | capped WER (points) | CRT-AL − each baseline | 1.68–1.78 | 5.9–6.2 | 0.77–0.80 | 2.7–2.8 |
| | SIM | | 0.015–0.020 | 0.051–0.070 | 0.005–0.009 | 0.016–0.032 |
| | UTMOS | | 0.035–0.040 | 0.12–0.14 | 0.036–0.046 | 0.13–0.16 |
| S (× 3.92) | capped WER (points) | between two baselines | 0.47–0.96 | 1.8–3.8 | 0.21–0.28 | 0.8–1.1 |
| | SIM | | 0.006–0.010 | 0.024–0.040 | 0.003–0.007 | 0.012–0.026 |
| | UTMOS | | 0.025–0.048 | 0.10–0.19 | 0.022–0.034 | 0.09–0.13 |

- NI (× 3.71, the S proxies). For 80 % power the true Δ must be at most −0.7 (SE 0.47) to −2.6 (SE 0.96) WER points on VCTK-2k and +0.22 to −0.04 on LTC-2k; at least +0.013 to +0.027 SIM on VCTK-2k and +0.001 to +0.015 on LTC-2k; at least −0.007 to +0.078 UTMOS on VCTK-2k and −0.018 to +0.026 on LTC-2k. "Parity" in VCTK-2k SIM therefore requires mvc_v1 to be truly ahead of the baseline.
- NL (× 2.80; the SE of the baselines' paired NL on VCTK-2k is 0.009–0.016). For 80 % power the true NL must be at most +0.005 (SE 0.009) to −0.015 (SE 0.016).
- L (× 2.80; the SE of a between-baseline Δ_src is 0.006–0.014 on VCTK-2k and 0.004–0.011 on LTC-2k). For 80 % power the true Δ_src must be at most −0.006 to −0.030 on VCTK-2k and −0.002 to −0.020 on LTC-2k.
- H1-4 (LTC-2k SIM; CRT-AL 0.911) has 80 % power only if mvc_v1 reaches about 0.93–0.94 on LTC-2k. The Dev evidence is VCTK only (0.898 at 100k). From VCTK-2k to LTC-2k, kNN-VC, Seed-VC and FACodec gain +0.027 to +0.034 in SIM, and FreeVC +0.010. The Must tier (§7.3) can therefore fail on this test alone, and PLAN §6.1's P(must) ≈ 0.85 is withdrawn.
- RB and Q compare a system with itself and have smaller SEs; they are not predicted.
- The proxies are other systems' SEs. mvc_v1's own SEs are known only after T1, and the table describes the design, not a promise.

### 5.6 Sensitivity analyses

Each is reported with the decisions (reject, system favoured) that change:

- uncapped corpus WER in every WER hypothesis of H1, S, NI, RB and Q, with the same Holm sets;
- LTC-2k with the independent bootstrap: H1, S, NI, RB, Q and L2s, L2c;
- failed conversions at their worst case (§5.4): every family, NL and L;
- H1 ∪ S with Holm over 30;
- NL beside the summarize.py value (descriptive; it decides nothing).

### 5.7 Descriptive analyses (CIs only)

- The levels of every system per set at 30 s: capped and uncapped WER (large-v3 and base), capped CER, SIM, SECS, UTMOS, SIM_src, SECS_src, M9 with the share D > 0, failures and EER; dB-MCD and the F0 metrics on VCTK-2k pairs with P.
- SIM_src and M9 per stratum: mvc_v1 minus each baseline and minus CRT-AL.
- The gender-pair table (F2F, F2M, M2F, M2M) of mvc_v1, CRT-AL and the baselines.
- At 3 s: mvc_v1 minus each baseline (kNN-VC from job M part t4, FreeVC, Seed-VC and FACodec from X1), 24 comparisons with Holm printed as a guide. They are labelled descriptive, because the baselines' 3 s outputs were seen before this plan. At 180 s: mvc_v1 minus kNN-VC (6). At 3 s and 180 s: mvc_v1 minus CRT-AL (12).
- The budget table of mvc_v1, CRT-AL and kNN-VC at 3, 30 and 180 s (as the journal's E16).
- `mvc_copy_latent`: its levels, mvc_v1 minus `mvc_copy_latent`, and `mvc_copy_latent` minus `mimi_copy_src` (fp32 continuous against fp16 Q32).
- The unpaired NL, ECAPA NL per stratum, ECAPA coverage with its CI, DNSMOS, the acoustic guards.

### 5.8 Implementation

- `mvc/analysis/analyze_mvc.py` (Mac, `.venv-stats`), built on `analyze_exploratory.py`:
  - it imports `analyze_tests` and `stats` unchanged and checks their sha256 at start;
  - journal rows: `analyze_tests.load_set` on `V1/m` with (t1, 30), (t3, 3), (t3, 180), (t4, 30), (t4, 3), (t4, 180) for VCTK-2k and t2 in place of t1 for LTC-2k, and on `V1/x1` with [('m', 3)];
  - MimiFlow-VC rows: the same function on `MR/m` with parts t1 (VCTK-2k, 30 s), t2 (LTC-2k, 30 s) and t3 (both sets, 3 and 180 s);
  - the merge stops on any (pair_id, system, budget) found twice, as `analyze_exploratory.load` does;
  - `analyze_tests.check_done` on the journal parts t1–t4 (selection sha256 9c7275f5…bc105), the checks of `x1.done` and `x2.done`, and `check_done` on the MimiFlow-VC parts t1–t3, whose `m.done` must name the sha256 of the committed MimiFlow-VC selection record;
  - its own `required()`: one row per pair at 30 s for every family system, target and `@src`; mvc_v1 at 3 s and 180 s (LTC-2k at 180 s: 1,699 rows plus the 101 shortfall rows); `target_copy@src` on the 980 VCTK-2k pairs with P; ECAPA and DNSMOS rows;
  - 30 s hypotheses through `analyze_tests.hypothesis`; 3 s and 180 s through a comparison with a budget argument, as `analyze_exploratory.comparison`; `levels()` is given a systems dict with an `s2` key (`crt_tau0.9`), which `system_order` needs;
  - NL as §3.3, L through `stats.gate_l_strata`, NI p-values as §5.2, coverage as §3.6;
  - it runs once and writes `mvc/TEST_RESULTS_mvc_v1.{md,json}` with every number and the sha256 of every input.
- `mvc/analysis/select_mvc.py` carries out §4.
- Unit tests in `mvc/tests/test_analyze_mvc.py` and `mvc/tests/test_select_mvc.py`, run with the rest of `mvc/tests`, cover at least: NL never through `gate_l_strata`, `fail_b` = 0 for `target_copy` and no worst case from it; the removal of the 180 s shortfall pairs; the NI p-value and its agreement with the CI rule; the Holm sizes 6, 24, 12, 12 and 6; the outcome table of §7.1; the coverage statistic against a direct computation; the R2 rules on synthetic rows, with ties and every fallback; and refusals on duplicate rows, a missing `m.done` or a wrong sha256.
- Both scripts live in `mvc/analysis/` so that they do not enter the `describe()` hash of §2.1. Each is reviewed against this plan by an independent reviewer before it is used, as `analyze_tests.py` was (DEVIATIONS_v1 item 4).

## 6. Second-model checks

### 6.1 ECAPA (secondary, pre-specified)

- ECAPA SIM: mvc_v1 minus each baseline and minus CRT-AL, both sets (10 comparisons).
- ECAPA SIM_src per stratum, the same comparisons (20), and the ECAPA NL per stratum on VCTK-2k (§3.3 with `sim_ecapa`).
- ECAPA coverage (§3.6).
- Agreement with WavLM-SV: Spearman's ρ over all (pair, system) values per set, and Kendall's τ of the system means, as X2 computed them.
- Flags, printed beside the WavLM-SV result they qualify:
  - a family-S rejection that favours mvc_v1 on SIM is reported as "not confirmed by a second speaker model" when the matching ECAPA SIM difference has the opposite sign and its 95 % CI excludes 0;
  - a family-L rejection is flagged in the same way when the ECAPA Δ_src is above 0 with a CI that excludes 0;
  - an NL rejection in a stratum is reported as "not confirmed by ECAPA" when the lower limit of the ECAPA NL CI in that stratum is above +0.03. For scale, the baselines' ECAPA NL on VCTK-2k is +0.035 to +0.18 and CRT-AL's −0.048 / −0.027 (computed for this plan from the X2 rows).

### 6.2 DNSMOS (descriptive, with one flag)

- OVRL and P808: mvc_v1 minus each baseline and minus CRT-AL, both sets (10 comparisons each), with the crossed bootstrap.
- A family-S or H1 rejection that favours mvc_v1 on UTMOS is reported as "not confirmed by a second naturalness predictor" when the matching OVRL difference has the opposite sign and its 95 % CI excludes 0.
- SNG, temp and relax were chosen on WavLM-SV and UTMOS, so these checks are the guard against tuning to the two primary predictors.

### 6.3 Acoustic guards

HNR, voiced frames kept, F0 correlation, semitone-SD ratio and loudness difference (§3.8) are descriptive.

### 6.4 Listening

No listening test is pre-registered in this plan. A MOS/SMOS test (PLAN §6.2 item 5) needs its own addendum (conditions, items drawn by seed from the test lists, scales, rater screening, analysis), frozen before any stimulus is built. Without one, any listening result is descriptive. The author's A/B of relax (§0.2 item 5) is never cited as evidence.

## 7. Outcome framing, fixed in advance

### 7.1 The H1 outcome

| Outcome | Condition | Framing |
|---|---|---|
| A | All six H1 tests reject in favour of mvc_v1 | mvc_v1 improves on CRT-AL in intelligibility, speaker similarity and naturalness on both test sets |
| A− | No H1 rejection favours CRT-AL, at least one test does not reject, and every non-rejected point estimate favours mvc_v1 | As A, with each non-rejected endpoint reported as "not significant" with its CI. No sentence claims improvement on every endpoint |
| B | At least one H1 rejection favours CRT-AL | Trade-off: the abstract names the endpoint and set on which CRT-AL is better |
| C | No H1 rejection favours CRT-AL, and at least one non-rejected point estimate favours CRT-AL | Improvement is claimed only on the rejected endpoints; the others read "no improvement shown" |

### 7.2 Leakage

- NL1s and NL1c both reject: "no source-speaker leakage beyond the natural other-speaker level (+0.03, WavLM-SV) on VCTK-2k", with any ECAPA flag of §6.1. NL is always shown beside mvc_v1's SIM on the same pairs and `target_copy`'s, because a voice degraded toward neither speaker also lowers SIM_src.
- Otherwise: "leakage above the natural level not excluded", naming the stratum, in the abstract, whatever H1, S or L show. The paper then makes no claim of a converter without source leakage, and angle 4 of PLAN §6.2 is presented as a measurement.
- L1s, L1c, L2s and L2c all reject: "not leakier than kNN-VC (margin +0.01)". This is a relative statement and is never worded as an absence of leakage; kNN-VC's own NL is +0.049 / +0.055. Otherwise: "leakage relative to kNN-VC not shown to be within +0.01", naming the strata.
- LTC-2k has only L2s, L2c and M9.
- Every table of family S carries the NL result in its caption.

### 7.3 Wording against the baselines, and the tiers of PLAN §6.1

- "Better than b on q" comes only from a family-S rejection that favours mvc_v1, and "worse" only from one that favours b. Otherwise the text gives the CI.
- "Comparable", "matches" and "parity" come only from a family-NI rejection for that baseline, endpoint and set. Without it the text gives the CI.
- No claim of beating or matching FreeVC on VCTK-2k is made.
- The Must, Should and Stretch thresholds of PLAN §6.1 are point-estimate benchmarks for the discussion and are never presented as tests:
  - Must: outcome A, ECAPA coverage ≥ 0.50 as a point estimate (CI reported), and leakage reported per stratum;
  - Should: capped WER ≤ 5.0 / 3.5 %, SIM ≥ 0.89 / 0.93, UTMOS ≥ 3.6 / 3.9 (VCTK-2k / LTC-2k), and WER at 3 s within 1.5 points of 30 s, read from RBN-3;
  - Stretch: WER ≤ 3.0 / 2.5 %, SIM ≥ 0.91 / 0.945.
- PLAN §6.1's probabilities are not part of this plan.

## 8. Execution order, jobs and unblinding

### 8.1 Order

Freeze and public timestamp (§8.4) → R2.0 → `select_mvc.py` committed → R2.1 → R2.2 and R2.3 on full Dev, with `mvc_r1_nosng` → R2.4 → B1 and a second R2, if triggered → `analyze_mvc.py` written, tested and reviewed → selection record committed and pushed → T1 conversions → scorer drift checks → T1 metrics t1, t2, t3 → ECAPA → DNSMOS and guards → copy to the Mac with sha256 → `analyze_mvc.py`, once.

Every job goes through `submit_ceai.sh`, and the interpreter session is told before each submission.

### 8.2 T1 layout and commands

On ceai, under `$MVC/test/` (never under `$J/v1/`):

- `cfg/`: the three JSON files of the record.
- `conv/<set>/<budget>/<name>/`: `run_mvc.py` output. `conv.done` is written when mvc_v1 (3, 30 and 180 s), `mvc_v1_q32` and `mvc_copy_latent` (30 s) have a manifest row for every pair of both sets.
- `m/<part>/rows/` and `m/<part>/metrics/<set>_<budget>{,_src}.jsonl`, with part t1 = VCTK-2k at 30 s (the three systems), t2 = LTC-2k at 30 s (the three systems) and t3 = mvc_v1 at 3 and 180 s on both sets. Each part writes `m/<part>/m.done` = `job <id> <date> selection sha256 <sha256 of the MimiFlow-VC selection record>` only when its row checks pass. About 19,000 target rows (6,000 + 5,400 + 7,600).
- `x2/ecapa/<set>_30{,_src}.jsonl`, `dnsmos/`, `guard/`, `drift/`.

On the Mac, the same tree under `MR/`, copied with identical sha256.

Commands (`C=/datastore/ceai/crt/mimi_vc/mvc`):

```
python -u $C/run_mvc.py --journal $J --pairs $J/pairs/vctk2k.jsonl --set vctk2k --budget 30 --name mvc_v1 \
    --cfg "$(cat $MVC/test/cfg/mvc_v1.json)" --out_dir $MVC/test/conv --device cuda
python -u $J/lab_rows.py --pairs $J/pairs/vctk2k.jsonl --source_e $J/pairs/vctk2k_source_E.json \
    --manifest $MVC/test/conv/vctk2k/30/mvc_v1/manifest.jsonl --manifest $MVC/test/conv/vctk2k/30/mvc_v1_q32/manifest.jsonl \
    --manifest $MVC/test/conv/vctk2k/30/mvc_copy_latent/manifest.jsonl --out $MVC/test/m/t1/rows/vctk2k_30.jsonl
python -u $J/metrics.py --in_jsonl $MVC/test/m/t1/rows/vctk2k_30.jsonl --out_jsonl $MVC/test/m/t1/metrics/vctk2k_30.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model $J/models/wavlm-base-plus-sv --device cuda
python -u $J/metrics.py --in_jsonl $MVC/test/m/t1/rows/vctk2k_30_src.jsonl --out_jsonl $MVC/test/m/t1/metrics/vctk2k_30_src.jsonl \
    --metrics sim,secs --wavlm_model $J/models/wavlm-base-plus-sv --device cuda
PYTHONPATH=$DS/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DS/models/spkrec-ecapa-voxceleb --smoke
PYTHONPATH=$DS/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DS/models/spkrec-ecapa-voxceleb \
    --systems mvc_v1,mvc_v1_q32,mvc_copy_latent --rows $MVC/test/m/t1/rows/vctk2k_30.jsonl \
    --out $MVC/test/x2/ecapa/vctk2k_30.jsonl
```

- `--set` is exactly `vctk2k` or `ltc2k`. LTC-2k calls of `lab_rows.py` have no `--source_e`.
- `score_ecapa.py` runs once per set and per row file (target, `_src`); it strips `@src` when it matches `--systems`.
- DNSMOS: `python mvc/dnsmos.py --models <durable folder> --rows <metrics rows> --out <file> --norm_lufs -26`.

### 8.3 Unblinding

- No T1 conversion starts before the selection record is committed and pushed.
- Scorer drift checks, before any MimiFlow-VC test row is scored:
  - WavLM-SV, UTMOS and Whisper: `metrics.py --max_rows 32` re-scores the first 32 rows of `$J/v1/m/t1/rows/vctk2k_30.jsonl` into `$MVC/test/drift/` with the T1 command. Against `$J/v1/m/t1/metrics/vctk2k_30.jsonl` it requires max |Δ| = 0 for `sim_wavlm` and `utmos`, and identical large-v3 and base hypotheses on at least 31 of 32 rows (Whisper's batched decoding changed 1 of 1,000 hypotheses on identical audio; DEVIATIONS_v1 item 2).
  - ECAPA: 32 rows of X2 re-scored, max |Δ `sim_ecapa`| ≤ 0.001.
  - If a check fails, nothing is scored until the cause is found and logged as a deviation. Journal rows are read, never replaced.
- No MimiFlow-VC test metric is computed before every T1 conversion exists: the metrics job requires `conv.done`.
- No test metric file (metrics, ECAPA, DNSMOS, guards) is opened, summarised or plotted before `analyze_mvc.py` runs. The T1 jobs print row counts, not values, and `summarize.py` is never run on test rows.
- `analyze_mvc.py` runs once. A refusal before any output (a missing input) is fixed and the script re-run. Any change to its code after it has produced numbers is a deviation, and both outputs are kept.

### 8.4 Freeze procedure

1. The corresponding author reads, edits and approves this text. Only the author freezes it.
2. The approved text is saved as `mvc/ANALYSIS_PLAN_mvc_v1.md`, with the status line "frozen on (date, time +07:00) by Quoc Nguyen".
3. `mvc/FREEZE_mvc_v1.sha256` holds:
   - on the Mac (`shasum -a 256`): the plan, `mvc/PLAN.md`, every `mvc/*.py`, `mvc/jobs/eval_dev.sh`, `mvc/jobs/job_eval.slurm`, `mvc/pairs/dev_s100.txt`, the durable DNSMOS models, and the journal files used unchanged (`stats.py`, `analyze_tests.py`, `analyze_exploratory.py`, `metrics.py`, `lab_rows.py`, `score_ecapa.py`);
   - on ceai (`sha256sum`, by the author): `$MVC/runs/r1/args.json` and `stats.npz`, the pair lists, the source-E file and the ECAPA model files.
   The snapshot files are hashed in the selection record, because the 200k snapshot does not exist before about 02:00.
4. `mvc/` is not under version control today (`git status` lists `?? mvc/`). The author commits the plan, `PLAN.md`, `FREEZE_mvc_v1.sha256` and the MimiFlow-VC code and tests, staged by name, with the plan's sha256 in the commit message, and creates the annotated tag `analysis-plan-mvc-v1`. Claude does this only on the author's explicit instruction.
5. Public timestamp: the tag is pushed to the public GitHub repository that holds `analysis-plan-v1`, before the Submit time of the first R2 job. Without that push, the paper calls the plan "internally frozen", not "pre-registered".
6. The selection record follows the same steps (tag `mvc-selection-record-v1`) before the Submit time of the first T1 job.
7. The push times and the Submit times of the first R2 and the first T1 job go into `mvc/DEVIATIONS_mvc_v1.md`.
8. At the freeze, before R2.0, a dated line under D-mvc-2 in `mvc/PLAN.md` points to §0.3 rows 4–6 of this plan, so that the change of condition 1 is on record before R1's final snapshot is scored.

## 9. Rules against rescuing results, and what the paper reports

Kept from ANALYSIS_PLAN_v1 §9 and PLAN §5.3:

- No metric, setting, snapshot, subset or reference budget is chosen after test scores are seen.
- The test sets are converted and scored once per frozen system: mvc_v1, `mvc_v1_q32` and `mvc_copy_latent`. There are no rescue runs.
- Null and negative results stay in the tables. Failed conversions are counted and reported, never dropped silently.

Added:

- The R2 grid is closed at the freeze.
- The configurations R2 does not choose, `mvc_r1_nosng`, B1 if it is not kept, and the R3 ablations never run on the test sets under this plan.
- Any later test-set run of another MimiFlow-VC variant (R5, R6, another SNG or CFG value) is exploratory, like X1 and X2: it is decided and logged in `mvc/DEVIATIONS_mvc_v1.md` before any of its output exists, it changes no decision of §7, and no such variant is promoted to the headline.
- No output is chosen per utterance with Whisper, WavLM-SV, ECAPA, UTMOS or DNSMOS, and no speaker-verification model or MOS predictor enters training (PLAN risk 12).

Reported whatever the outcome:

- every family (H1, NL, L, S, NI, RB, Q) with estimates, CIs, p-values, Holm decisions, pair counts and failures, and the sensitivity analyses with the decisions they change;
- the outcome and the leakage rows of §7, in the abstract where §7 requires it;
- NL per stratum, beside the summarize.py value and the ECAPA NL; family L per stratum; M9;
- the second-model checks and their flags;
- the R2 trace: R2.0, all eleven R2.1 configurations, the finalists on full Dev and Dev-900, `mvc_r1_nosng`, the choice and R2.4, B1 if it ran, and the number of Dev configurations ever scored;
- next to any test number, mvc_v1's Dev-900 value, labelled "selected on Dev";
- the deviation table of §0.3 and every later deviation;
- the limits: FreeVC has seen VCTK; Seed-VC uses only the first 25 s of the reference; MimiFlow-VC decodes in fp32 and the journal's Mimi copies in fp16; the hardware (H200 MIG 2g.35gb on ceai); the 101 LTC-2k shortfall pairs at 180 s; the decisions rest on one speaker model (WavLM-SV), with ECAPA as the second; the NL margin of +0.03 was set on Dev; LTC-2k has no natural-level check.

## 10. Deviations

- After the freeze, every change goes into `mvc/DEVIATIONS_mvc_v1.md` with its date and time (+07:00), its reason, what had been seen when it was decided (which R2, B1 or T1 outputs existed), the files changed with their new sha256, and whether a rule, threshold or result is affected. Where possible it is written before the job it affects is submitted.
- A rule of §4 is not changed after the R2 results it applies to exist, and §5–§7 are not changed after any T1 conversion exists. In those cases the frozen rule's result is reported, and any other analysis is labelled post hoc.
- A code fix needed to carry out a frozen rule (as DEVIATIONS_v1 item 2) is logged with its tests and new hashes. The affected stage is re-run once on the fixed code, and both outputs are reported.
- Operational changes (paths, jobs, storage, accounts) are logged and do not affect rules.
- The paper reports every deviation.

## 11. The author's decisions at the freeze (27 Oct 2026, 16:28)

1. D-mvc-2 condition 1 and R2 use the paired NL (§0.3 row 4), accepted with §0.2 item 3 known.
2. The run that defines mvc_v1 is not extended to 300k (§0.3 row 9). An exploratory continuation R1x to 300k runs under §4.10.
3. B1 is the only training-side remedy before T1; R4, E1 and E2 are dropped from the pre-T1 ladder; R5 is not run (§0.3 rows 7–8).
4. The cost of R2, about 13–17 slice-h against PLAN.md's 6, is accepted (§4.1).
5. DNSMOS runs on the Mac with WAVs copied read-only from ceai; its models are kept in `$DS/models/dnsmos/` and `paper/mimi_vc/models/dnsmos/` (§3.7).
6. `mvc/` is committed to git, staged by name, on the branch `mvc-v1-freeze`, with the annotated tag `analysis-plan-mvc-v1`. The branch and the tag are pushed to the public GitHub repository before the first R2 job (§8.4).
7. No listening test is pre-registered (§6.4).
