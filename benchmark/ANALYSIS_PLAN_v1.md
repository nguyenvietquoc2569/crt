# CRT journal extension: analysis plan v1 (frozen 18 Oct 2026)

Status: frozen on 18 Oct 2026 by Quoc Nguyen (corresponding author), after job D0 passed and before any full-Dev selection result exists. Drafted on 11 Oct 2026 and revised on 18 Oct 2026 with the author's decisions (§10) and the corrections found while building the tooling. From the freeze on, every change is a deviation (§8.4 step 9, `benchmark/DEVIATIONS_v1.md`).

File: `benchmark/ANALYSIS_PLAN_v1.md`. The v0 file, `benchmark/ANALYSIS_PLAN.md`, is not changed. Where v1 says nothing, v0 applies. In this document `results/` means `paper/journal_extension/results/`, and "the idea plan" means `paper/journal_extension/crt_math_ideas_verified.md`.

## 0. Lineage

### 0.1 Relation to v0

v0 says that changes made after 28 Sep 2026 go into v1, and that v1 is dated and frozen at the end of Dev selection, before any test-set conversion is scored. This v1 is frozen earlier, before the result of the full-Dev selection run is read. The frozen rules of §4 then carry out the Dev selection, and their output goes into a separate dated selection record (§4.6). v1 itself is not edited after the Dev numbers exist.

The items that v0 §7 listed as "pending for v1" become binding at the freeze. They are restated where they apply (§1.3, §3.2, §4.2). Row 19 of §0.3 corrects one of them.

v0 is committed unchanged in commit bd1f40ad801bc5b099361872be6ba37c1fa6c9b2, made on 18 Oct 2026; it was written on 28 Sep 2026 according to its file modification time (28 Sep 2026 21:36:54). The paper does not describe v0 as externally time-stamped. The scripts this plan relies on were not under version control either before the freeze. They are committed with this plan, and their sha256 values are in `benchmark/FREEZE_v1.sha256`.

### 0.2 What was seen before the freeze

The following results were known before this plan was frozen (items 1–7 before the draft of 11 Oct 2026, item 8 after it). They shaped v1.

1. The published-pipeline Dev results and the decoder exploration on a 50-pair Dev screen. The screen consists of the Dev pair rows k with k mod 20 = 0 of `pairs/dev.jsonl` (`--shard 0/20`), with 30 s target references. It ran as HPC jobs 2624 to 2661 on 11 Oct 2026, on H200 MIG 2g.35gb slices (`results/SUMMARY.md`; `docs/training_tips.md` C12–C17; raw files in `results/{lab_r1, lab_r2, anchor_lab, lab_r3, round4, round5, round6, budget_check}`).
2. The size of the exploration. The nine screen metrics files (`lab_r1/metrics.jsonl`, `lab_r2/metrics.jsonl`, `anchor_lab/metrics.jsonl`, `lab_r3/metrics.jsonl`, `round4/lab_r4_metrics.jsonl`, `round4/anchor_us32/metrics.jsonl`, `round5/lab_r5_metrics.jsonl`, `round6/lab_r6_metrics.jsonl`, `budget_check/metrics.jsonl`) hold 176 system rows. Some rows are the same output under another name; FAISS, for example, appears in three files. Counted by identical per-pair Whisper hypotheses and SIM values, the 176 rows hold 162 distinct outputs, and 156 once the two natural-speech controls and the four 3 s / 180 s budget rows are removed. The truth-set analysis compared a further 71 decoding rules (the rows of the main table of `results/round5/truth_fit_v2.md`). These counts were made for this plan from the named files.
3. Every screen number in this document is exploratory. It has no multiplicity correction, it covers 50 pairs, and all its targets are female. The screen has 20 source speakers and 10 target speakers (26 F→F and 24 M→F pairs, counted from the `genders` field of `results/budget_check/metrics.jsonl`). No screen number describes conversion into a male voice.
4. The headline WER figures in `SUMMARY.md` are corpus WER over all 50 screen pairs. Only 20 of the 50 are `wer_primary` pairs (speaker-specific sentences); `SUMMARY.md`'s caveats say "about 25", but the rows give 20. The other 30 are shared-text pairs with a parallel recording. The selection rule of §4 uses `wer_primary` pairs only (v0 §7), so §2.6 shows both columns.
5. The self-conversion truth set (307 pairs) and every analysis in `results/round5/truth_fit_v2.md`, `results/token_stability_32.md` and `results/codeword_clustering.md`.
6. The idea plan. Its modification time is 11 Oct 2026 01:39:59. The rows of `results/anchor_lab/metrics.jsonl` carry the run_id `20261011-063126-h200node01-…`, so the first source-anchored run (job 2634) started at 06:31:26 the same day. This is file-system evidence only; the selection record adds the `sacct` Submit and End times of job 2634 (§4.6).
7. Source-speaker leakage by gender pair. Split by gender pair (computed for this plan from `results/anchor_lab/metrics_src.jsonl`), the pooled SIM_src values hide a rise toward the source speaker on cross-gender pairs. On the 24 M→F pairs, C1 has SIM_src 0.670 against 0.631 for FAISS, a rise of 0.039, about four times the 0.01 margin of §4.4. On the 26 F→F pairs it has 0.768 against 0.821. The pooled 0.721 against 0.730 combines the two, and the F→F fall comes with a general loss of speaker identity: C1's target SIM is 0.100 below FAISS's (0.819 against 0.919). The per-pair margin D = SIM_src − SIM_tgt (§3.1, M9) is not lowered by a uniformly degraded voice, and it moves toward the source: its mean is −0.098 for C1 and −0.189 for FAISS, and 10 of 50 C1 outputs (3 of the 24 M→F) are closer to the source centroid than to the target centroid, against 3 of 50 (0 of 24) for FAISS (computed for this plan from `results/anchor_lab/metrics.jsonl` and `metrics_src.jsonl`). The sentence "Source leakage does not rise" in `SUMMARY.md` §4 rests on the pooled values. It is withdrawn and is not reused. C2–C5 have no M→F leakage measurement.

8. Seen between 11 and 18 Oct 2026, before the freeze (`SUMMARY.md` §8). All of it was exploratory or a check:
   - Job D0 (4321) re-ran the screen systems on the `nvquoc` account and reproduced them exactly (§8.1 item 9).
   - The baseline smoke test (job 4322) ran the four baselines on the first 20 screen pairs (female targets only; §2.4 forbids changing a setting on these numbers):
     - capped WER 0.0–1.6 %, SIM 0.914–0.955, UTMOS 3.47–3.96;
     - on the same pairs, C1 15.1 %, 0.822 and 2.32, and C3 13.6 %, 0.836 and 2.42.
   - The Mimi Q = 32 copies (§2.5) showed that the codec is nearly transparent: capped WER 0.16 %, UTMOS 3.82, and SIM to the target 0.976 for the copy of P.
   - Rounds 7 and 7b (jobs 4335 and 4337) tried a closed-form Gaussian optimal-transport map of the latent toward the target reference's statistics. That is 15 system rows in two files, with 14 distinct outputs (§6 T12).
     - Both were negative: SIM to the target stayed at 0.737–0.748 for the source-based maps, and the maps added no similarity to C3's base while costing WER and UTMOS.
     - No candidate was added.

No VCTK-2k or LTC-2k conversion exists, and no system has been run on the full 1,000 Dev pairs (`SUMMARY.md`, caveats).

The screen led to replacing the primary system. With a 30 s reference, the published pipeline was close to unintelligible: uncapped corpus WER over all 50 pairs was 126.02 % for FAISS and 107.97 % for CRT τ = 0.4 (capped: 95.45 % and 94.80 %). Source-anchored CRT with latent decoding reached 17.89 to 19.84 % (no pair above 100 %, so capped and uncapped are equal) (`results/anchor_lab/metrics.jsonl`, `results/lab_r3/lab_report.md`, `results/round4/lab_r4_metrics.jsonl`).

v0 lets changes be recorded in v1 before any test scoring. It reserved Dev for τ and the other hyper-parameters ("nothing else", v0 §1), however, and did not anticipate a change of primary system. Replacing S2 with a decoder family developed on the Dev screen is the largest deviation from v0 (§0.3 row 1). The v0 primary analysis is kept as family V and is reported in the main results table, not only in the supplement (§5.1, §7). The candidate decoders were found by trying many configurations on the same 50 pairs, so their screen numbers are biased in their favour (the winner's curse). §4 therefore selects on the full Dev set and repeats the selection without the 50 screen pairs, and no screen number enters any test.

### 0.3 Deviations from v0

Rows 1–19 were written on 11 Oct 2026 and rows 20–21 on 18 Oct 2026, all after the screen results were known. The last column names any part of an entry whose substance was written down earlier, with its source and time.

| No. | v0 | v1 | Reason | Content fixed earlier (source, time) |
|---|---|---|---|---|
| 1 | §2: the primary system is S2 CRT(τ_dev) | The primary system is CRT-AL, one of five closed candidates C1–C5, chosen on full Dev by §4.4 | Screen: uncapped WER 17.89–19.84 % for the source-anchored latent decoders against 126.02 % (FAISS) and 107.97 % (CRT τ = 0.4), all 50 pairs, exploratory (capped: unchanged for the decoders, 95.45 % and 94.80 %) | none |
| 2 | §4: SIM ≥ FAISS in the τ rule | Not applied to CRT-AL; replaced by E1–E4 (§4.4). The τ rule for S2 is unchanged | Every candidate's screen SIM is below FAISS (0.818–0.857 against 0.919). Both the v0 constraint and the idea plan's looser SIM_target ≥ FAISS − 0.01 (0.909) exclude all five. The SIM cost is tested in P3/P4 instead | The idea plan: SIM ≥ FAISS in §8 item 4 (l. 524); SIM_target ≥ FAISS − 0.01 and SIM_source − FAISS ≤ +0.01 in T1.1 (l. 188) and T2.1 prediction 3 (l. 317); file time 11 Oct 01:39:59 |
| 3 | not in v0 (idea plan §8 item 4) | The idea plan fixed a rule for one primary CRT decoder: lowest `wer_primary` WER subject to SIM ≥ FAISS, ties broken by MCD. v1 replaces it by §4.4 | As row 2: no candidate meets SIM ≥ FAISS on the screen | Idea plan l. 524, 01:39:59 |
| 4 | §5: the primary family is CRT(τ_dev) − FAISS | Family P = CRT-AL − FAISS. New key secondary family K = CRT-AL − CRT τ = 0.4. The v0 family is kept as family V, reported in the main table | Follows from 1 | none |
| 5 | none | Gate L: non-inferiority of SIM to the source speaker, margin +0.01, tested separately in the same-gender and cross-gender strata of each test set | The output CB0 is the source's, and the screen shows a rise on M→F pairs (§0.2 item 7) | Margin: idea plan l. 188 and l. 317 (01:39:59, before the anchor run at 06:31:26). The stratification is new |
| 6 | §6: outcomes A/B/C depend on the primary family alone | A and B also require gate L to reject; a "leakage not excluded" row caps the framing at B | Follows from 5 | none |
| 7 | §5: secondary family CRT(τ_dev) vs baselines | CRT-AL vs baselines | Follows from 1 | none |
| 8 | §5: P8 is CMOS CRT(τ_dev) vs FAISS | P8 is CMOS CRT-AL vs FAISS; K8 is CMOS CRT-AL vs CRT τ = 0.4 | Follows from 1 | none |
| 9 | §1 lists four sets | Adds the truth set (theory only), source-speaker E sets for VCTK-2k, and a pre-hashed 10-target VCTK-2k variant (§5.3) | Needed for §6, gate L and the size check | The truth set was built and analysed before this draft (`results/truth_fit.md`, `results/round5/truth_fit_v2.md`) |
| 10 | §1: Dev is for "τ and every other hyper-parameter; nothing else" | Dev also gives descriptive mechanism tables (§6) and gender-pair tables (§4.4) | They explain results and select nothing | none |
| 11 | §2 lists S1–S8 | Adds the us32 front-end variant S1u (descriptive), ablations A1–A5 and the R3 τ grid, and fixes the E17 slot to one named decoder | Screen | none |
| 12 | Header: v1 is frozen at the end of Dev selection | v1 is frozen before full-Dev selection; the selection output goes into a separate record | Stricter than v0 | none |
| 13 | §4: τ-rule ties are not defined | A tie is equal Dev WER at two decimals (in %) | Makes the rule mechanical | none |
| 14 | §5 size check: "if it is too large" | Variance components, stated thresholds and a stated rebuild rule (§5.3) | v0 gave no threshold | UTMOS 0.05 and MCD 0.10 dB: idea plan §1 (l. 34). SIM 0.010: idea plan l. 188 and l. 317 |
| 15 | §5: crossed bootstrap with source and target speakers resampled independently | On Dev and LTC-2k, where every speaker is both a source and a target, the primary bootstrap resamples each speaker once for both roles; v0's version becomes a sensitivity analysis. VCTK-2k is unchanged | Treating one person's two roles as independent can understate the variance | none |
| 16 | §2: S8 is "chosen at setup … recorded in v1" | S8 is FACodec, or there is no S8 (§2.4) | v0 asks v1 to record the choice | `paper/journal_extension/baselines.md` §4 (28 Sep 2026) |
| 17 | §3: M1–M7 | Adds M8 SIM_src and M9 source–target margin. M7 coverage is undefined for continuous decoders | Needed for gate L and for the latent decoders | The idea plan, T1.1 "Scoring", asks for a `sim_src` column (01:39:59) |
| 18 | §1: the E16 sweep covers FAISS, CRT and kNN-VC | It covers FAISS, S2, S3, CRT-AL and kNN-VC | S3 is the published setting and CRT-AL the new primary | none |
| 19 | §7: "E is 10 utterances per target" | E is up to 10 utterances or 30 s per speaker, whichever comes first | Correction: this is what `make_pairs.py` does (`n_e`, `e_max_s`) | `make_pairs.py` (28 Sep 2026) |
| 20 | §3 M3 and §5: corpus WER, total errors (S + D + I) over total reference words | The WER of every v1 rule and endpoint is capped corpus WER (§3.1, M3c): each pair contributes min(E_p, N_p) errors, so no pair counts above 100 %. This covers E4 and the CRT-AL choice (§4.4), S1u and E17 (§4.3), families P, K and S (§5.1), the size check of P and K (§5.3) and the WER tables. Uncapped corpus WER is reported beside every capped WER and is tested as a sensitivity analysis with the same tests. The parts v1 keeps from v0 keep v0's uncapped WER as their primary: the τ rule of §4.2 and family V, with capped WER reported beside them | One Whisper repetition loop decided a corpus WER on the screen: pair `dev__p226_064__p230` carried 112 of FAISS's 270 errors on the 20 `wer_primary` pairs, and an output's insertions have no upper bound | The corresponding author's decision, 18 Oct 2026 (§10) |
| 21 | §7: "5 LTC targets have less than 180 s available (minimum 57.6 s)" | 2 LTC targets have less than 180 s: 3729 (75.04 s) and 6930 (105.0 s), 101 pairs | Correction: these are the shortfalls of the current `pairs/ltc2k.jsonl` (`R["180"].shortfall`; `summary.json`, `R_per_target`); the v0 sentence described an earlier build of the list | `pairs/ltc2k.jsonl` (9 Oct 2026) |

## 1. Sets

### 1.1 Pair lists

All lists come from `benchmark/make_pairs.py` with seed 20260928, VCTK mic1 recordings only, and genders from `speaker-info.txt` (v0 §1).

| Set | Construction | Size | Use |
|---|---|---|---|
| Dev | The 20 SPIE conference speakers (10 F / 10 M), 10 source utterances each (5 shared-text, 5 speaker-specific), 5 gender-balanced targets per source utterance. Every Dev speaker is both a source and a target | 1,000 pairs; 500 `wer_primary` | The τ rule, the S1u setting, the E17 slot and the CRT-AL selection (§4); the size check (§5.3); full-Dev mechanism tables (§6) |
| Dev screen | Dev rows k with k mod 20 = 0 | 50 pairs: 20 `wer_primary` and 30 with a parallel recording; 20 source and 10 target speakers, all targets female | Exploration, now finished, reported as exploratory. It is part of full Dev, and the selection is repeated without it (§4.4) |
| VCTK-2k (primary test) | v0 §1: 20 source and 20 target speakers, disjoint from each other and from Dev; p315 and s5 excluded; 20 source utterances per source speaker × 5 targets | 2,000 pairs. The `wer_primary` count is in `pairs/summary.json` on the HPC and is copied into the selection record | Main tables |
| VCTK-2k, 10-target variant | The same command with `--vctk_targets 10`, generated before the freeze into a separate directory and hashed (§8.1 item 3) | 4,000 pairs | Replaces VCTK-2k only if §5.3 triggers. It is a different sample, not an extension of the 5-target set: `make_pairs.py`'s target rotation (`TargetRotation.pick`), shared-sentence choice and reference building (`build_refs`) all depend on the number of targets, so the targets, the source utterances, the R and E sets and the `wer_primary` count change |
| LTC-2k (second test) | LibriTTS-R test-clean, utterances of 3–10 s, 10 gender-balanced targets each. Every speaker is both a source and a target (`run_set('ltc2k', spk, spk, …)`, `make_pairs.py` l. 503) | 1,800 pairs (v0 §7). LTC has no shared-text block, so every pair is `wer_primary`. There is no parallel recording, so there is no MCD | Main tables |
| Conference-100 | The 100 SPIE pairs | 100 | Corrections appendix only |
| Truth set | `pairs/self.jsonl`: a Dev target's own held-out shared-text recording, converted to that same target with the 30 s reference. The truth is its Mimi codes (`crt_truth_fit.py` docstring) | 307 pairs | Fitting and theory only (§6). Never used to select or tune a source-anchored system, because there the source CB0 is the true CB0 and the source anchor is an oracle |

### 1.2 References and source-speaker E sets

The v0 reference rule holds. Each target speaker's reference pool R, parallel recordings P and SIM enrolment set E are mutually disjoint, and R never holds a recording of a sentence sent to that target.

SIM_src (M8) needs an E set for each source speaker. On Dev and LTC-2k every source speaker is also a target, and its target E set already excludes every source recording of the set (`make_pairs.py`, `build_refs`, `src_paths`). These existing E sets are used as the source E sets, so SIM_src and SIM_tgt use the same centroid for the same speaker. D0 checks that every source speaker of `dev.jsonl` and `ltc2k.jsonl` appears there as a target.

Only VCTK-2k sources are not targets. For them, `make_pairs.py` gains a step that builds an E set for every source speaker, with the target-E rules (`n_e` = 10 and `e_max_s` = 30 s; never a source recording of the set; never a shared-block recording) and the same seed. It writes a separate file, and the VCTK-2k pair list is not rewritten. The same step runs for the 10-target variant. Pairs whose source speaker gets no E set are left out of gate L and counted.

### 1.3 Reference budgets (v0 §1 and §7, binding)

- Every system gets the same R. The main tables and all of §4 use 30 s.
- The budgets of 3, 30 and 180 s are nested, and each is cut to exactly that length.
- 2 LTC targets have less than 180 s of audio: 3729 (75.04 s) and 6930 (105.0 s), 101 pairs (§0.3 row 21). The 180 s column covers only targets without a shortfall.
- On the test sets, CRT-AL and its ablations run in the R1+R3 mode of `crt_anchor_lab.py` (§8.1 item 4), which needs no source-speaker reference. A source-speaker shortfall at 180 s therefore cannot stop a run; if one occurs anyway, it is counted as a shortfall.
- The sweep over 3/30/180 s (E16) covers FAISS, S2, S3, CRT-AL and kNN-VC, and it is descriptive.
- E is up to 10 utterances or 30 s per speaker, whichever comes first (`make_pairs.py`, `n_e`, `e_max_s`).

## 2. Systems

### 2.1 Published and v0 systems

| ID | System | Definition (code) | Role |
|---|---|---|---|
| S1 | FAISS | Stage 1 only: retrieved codes on the published greedy path (`run_conversions.py`, `retrieve`) | Comparator of family P |
| S2 | CRT(τ_dev) | Published CRT v5.6.5 (slim weights `benchmark/journal/models/crt_v565_best_weights.pt`, sha256 b26d6f94…7542 as recorded in `results/anchor_lab/anchor_summary.shard0of20.json`). The anchor is the retrieved CB0, with a max-softmax gate at the τ_dev chosen by the v0 τ rule (§4.2) | Family V; descriptive elsewhere |
| S3 | CRT τ = 0.4 | As S2 at τ = 0.4, the SPIE setting, re-measured with this harness | Comparator of family K |
| S4 | CRT argmax | Every CB1–18 token is replaced by CRT's argmax (`crt_argmax`) | Descriptive |
| S1u | us32 unit selection | A FAISS-only front-end variant: Viterbi over the top-32 reference frames with a natural-successor reward (`crt_decode_lab.py`, `us32_w<repeat>_<move>_<join>`, no CRT). The setting is chosen by §4.3 | Descriptive |
| E17 | Frame-level latent MBR | `lmbr_k2_g1_T1` of `crt_decode_lab.py`, run with `--cb0 retr` (the default) on the published R1 logits of the `run_conversions.py` dump, without `--logits_from`: latent MBR over the 8 FAISS candidate frames, κ = 2, Viterbi continuity g = 1, T = 1. Its screen row is `lab_lmbr_k2_g1_T1` in `results/lab_r2/`. The source-CB0, R3-logit variant `r3a_lmbr_k2_g1_T1` of `results/lab_r3/` is a different system and is not E17 | v0's optional E17 slot. It enters only by the v0 §4 rule (§4.3) |

On Dev, `crt_anchor_lab.py` decodes S2's τ grid as `anc_R1_r0_tau<τ>` for τ ∈ {0.1, …, 0.9}. Its R1 forward pass reproduces the `run_conversions.py` dumps exactly: the maximum logit difference is 0 on the 50 screen pairs (`results/anchor_lab/anchor_summary.shard0of20.json`, `r1_repro`). For the test sets, S1–S4 come from `run_conversions.py --systems faiss,crt_tau:0.4,τ_dev,crt_argmax --dump_logits`. When τ_dev = 0.4, S2 and S3 are the same system and τ_dev is left out of the list, because `run_conversions.py` exits on a duplicate system (`parse_systems`, l. 171–172). The `--dump_logits` dump is kept because S1u and E17 read it: `crt_decode_lab.py` loads `crt_logits` for every system (`run_pair`, l. 331).

### 2.2 CRT-AL candidates (closed at the freeze)

CRT-AL means CRT with a source anchor and latent decoding. All five candidates share the same CRT forward pass and differ only in the decoder.

The common part is recipe R3 (`crt_anchor_lab.py`, `anchor_pair`):
- CRT input: the anchor CB0 is the source utterance's own CB0. The hints CB1–18 are the tokens retrieved on the published greedy path. The speaker embedding is the target's (Resemblyzer on the first 15 s of the reference, as published).
- The output CB0 is the source's CB0.
- Decoding uses `crt_decode_lab.py` with R3 logits (`--logits_from` the anchor dumps, `--logits_key R3`, `--cb0 src`). The full D2 commands are in §8.2.

The motivation for R3 was CRT's training input, in which the anchor is a clean CB0 and the hints are frames retrieved with that CB0 as the query (the idea plan, checked fact 3); the published inference feeds the retrieved CB0 instead. R3 does not reproduce training exactly, because in training the speaker embedding belongs to the anchor's speaker. The confidence prediction derived from this input-matching hypothesis failed on the screen (§6). v1 therefore makes no mechanism claim for R3 and treats it as an intervention whose parts the ablations A1–A5 separate.

Notation (`crt_decode_lab.py` docstring):
- E_k is the codebook of acoustic level k (CB1–CB31, 256-d), and r_k is the retrieved token.
- p_k = softmax(R3 logits_k / T) for k = 1–18.
- a_F = Σ_{k=1..31} E_k[r_k].
- μ_T = Σ_{k≤18} p_k E_k + Σ_{k=19..31} E_k[r_k].
- RQ(z) is Mimi's own greedy residual quantisation of z over all 31 acoustic levels.
- GV_γ is variance restoration with exponent γ.

| ID | Name in code | Output |
|---|---|---|
| C1 | `pm_rq_T1` | Codes (source CB0, RQ(μ_1)): the posterior-mean latent, re-quantised. This is an approximate joint minimum-Bayes-risk code under squared latent error. On the screen it equals `anc_R3_s0_pm_rq_T1` of `crt_anchor_lab.py` (identical outputs; 18.21 % WER in both files) |
| C2 | `pm18_rq` | Codes (source CB0, RQ(Σ_{k≤18} p_k E_k)) at T = 1. As C1, but the retrieved CB19–31 detail is dropped before re-quantisation |
| C3 | `gv1_pm_cont_T1` | The continuous latent GV_1(μ_1), decoded through Mimi's decoder with the source CB0 (`decode_latent`) |
| C4 | `gv1_shrink0.85_cont` | The continuous latent GV_1(a_F + 0.85 (μ_1 − a_F)) |
| C5 | `gv0.5_shrink0.75_cont` | The continuous latent GV_0.5(a_F + 0.75 (μ_1 − a_F)) |

GV_γ (`crt_decode_lab.py`, branch `gv`, l. 610–624) works per utterance:
- It takes the top 16 principal components of the target reference's decoder-space acoustic latent, computed on speech frames (frames whose norm is above its 20th percentile).
- The gain for each component is clip((v_ref / v_x)^{γ/2}, 1, 3). The statistics are computed on the output's speech frames, with the mask taken from the FAISS latent.
- The mean over speech frames is kept.

The candidates are numbered from the fewest to the most free parameters. That order is the last tie-break in §4.4.

### 2.3 Ablations and descriptive grids

| ID | System (code name) | What it removes or changes |
|---|---|---|
| A1 | `anc_R1_s0_pm_rq_T1` | Published anchor (retrieved CB0), source CB0 out, latent MBR. Tests the anchor |
| A2 | `anc_R3_r0_pm_rq_T1` | R3 with the retrieved CB0 out. Tests the output CB0 |
| A3 | `anc_R3_s0_argmax` | R3 and source CB0, with token argmax instead of latent decoding |
| A4 | `anc_R3_s0_tau0.4` | R3 and source CB0, with the published τ = 0.4 gate |
| A5 | `anc_src0_faiss` | Source CB0 swapped into the FAISS codes, without CRT |

A3 and A4 keep the retrieved CB19–31 tokens under a source CB0. `crt_anchor_lab.py` marks them as not registered, and here they serve only as ablation rows. Two more groups are descriptive on Dev: the R3 τ grid (`anc_R3_r0_tau<τ>`, τ ∈ {0.1, …, 0.9}) and the geometric-pooling rows `anc_G0.5_*` that the Dev job produces. None of these systems is ever selected.

### 2.4 Baselines S5–S8 (conditional)

The baselines follow v0 §2. Their settings are fixed here, and the remaining details (repository commit, checkpoint sha256, sample rate, full command line) are written into the selection record before T1 and not changed afterwards:
- S5: kNN-VC, `prematched=True`, k = 4, with the repository's prematched HiFi-GAN vocoder.
- S6: FreeVC, checkpoint `freevc.pth` (the one `setup_baselines.sh` fetches), flagged "VCTK-seen".
- S7: Seed-VC v1 small (the offline model with the Whisper-small content encoder), 25 diffusion steps (the README default, l. 55 and l. 69; `baselines.md` §2c; the argparse default of `inference.py` is 30), all other settings at the defaults of the pinned commit. Upstream inference keeps only the first 25 s of the reference (`ref[:sr*25]`), so at the 30 s and 180 s budgets Seed-VC sees the first 25 s of R; this is stated wherever Seed-VC appears.
- S8: FACodec (`amphion/naturalspeech3_facodec`, the `FACodecEncoderV2`/`DecoderV2` voice-conversion path of the model card and the authors' Space). In that path the content and prosody codes come from the source utterance, the residual codes are dropped, and R supplies only the timbre vector. Its weights are Apache-2.0 and it decodes in one step (`baselines.md` §4). Vevo's weights are CC-BY-NC-4.0. S8 is FACodec or nothing: if FACodec does not run, there is no S8, and Vevo is not substituted.

A baseline counts as running when it converts 20 screen pairs (the smoke test) and is listed in the selection record. The smoke test checks only that the system runs and produces speech; its numbers are not used to change any setting. A baseline that is not listed when the record is committed is dropped before any test conversion exists, and its hypotheses leave family S (§5.1).

The corresponding author approved the downloads of all four baselines on 18 Oct 2026. They are installed on the HPC at pinned versions: kNN-VC `bshall/knn-vc` c616845, FreeVC `OlaWod/FreeVC` 60032a7 (`freevc.pth` sha256 e2cc2d04…, the file of the authors' Hugging Face Space), Seed-VC `Plachtaa/seed-vc` 51383ef, and FACodec through `open-mmlab/Amphion` 26f6883. All run through one harness, `benchmark/run_baselines.py` with one adapter per system in `benchmark/baselines/`, which gives every system the same R (§1.3) and writes `metrics.py` rows. `benchmark/download_baselines.sh` and `setup_baselines.sh` rebuild the installation and check every sha256. The full record (commits, checkpoint sha256, sample rates, environments and command lines) is written by `run_baselines.py --record` in the smoke test and copied into the selection record.

### 2.5 Controls

- Natural source recording (`source_copy`). This is the WER floor, and its SIM to the target is the different-speaker level.
- Natural target recording (`target_copy`). This is the target's parallel recording P, available on shared-text pairs of Dev and VCTK-2k only.
- Mimi Q = 32 copy-synthesis (the codec ceiling) of the source recording, and of P where it exists: `run_conversions.py` systems `mimi_copy_src` and `mimi_copy_P` (§8.1 item 6), run in D0.

### 2.6 Screen numbers of the candidates (exploratory)

These numbers come from the 50-pair screen with 30 s references.

WER is Whisper-large-v3 corpus WER, given over all 50 pairs and over the 20 `wer_primary` pairs. The first column matches the lab reports exactly. The second column was recomputed for this plan from the named files, with the corpus-WER definition of `metrics.corpus_wer` restricted to `wer_primary` rows. The two capped columns (M3c, §3.1) were computed for this plan from the named files with `benchmark/stats.py` (`corpus_wer(..., capped=True)`).

| ID | System | WER % (50) | WER % (20 wer_primary) | Capped WER % (50) | Capped WER % (20 wer_primary) | SIM tgt | UTMOS | dB-MCD (30) | File |
|---|---|---|---|---|---|---|---|---|---|
| S1 | FAISS | 126.02 | 204.55 | 95.45 | 99.24 | 0.919 | 1.71 | 8.82 | `anchor_lab/` |
| S3 | CRT τ = 0.4 | 107.97 | 115.15 | 94.80 | 98.48 | 0.892 | 1.59 | 9.02 | `anchor_lab/` |
| A5 | src0_faiss | 148.46 | 111.36 | 96.42 | 98.48 | 0.918 | 1.70 | 8.71 | `anchor_lab/` |
| C1 | pm_rq_T1 | 18.21 | 26.52 | 18.21 | 26.52 | 0.819 | 2.19 | 9.38 | `anchor_lab/` |
| C2 | pm18_rq | 19.19 | 25.76 | 19.19 | 25.76 | 0.818 | 2.59 | 9.42 | `lab_r3/` |
| C3 | gv1_pm_cont_T1 | 17.89 | 21.97 | 17.89 | 21.97 | 0.838 | 2.31 | 9.53 | `lab_r3/` |
| C4 | gv1_shrink0.85_cont | 19.84 | 29.55 | 19.84 | 29.55 | 0.853 | 2.28 | 9.22 | `round4/lab_r4_metrics.jsonl` |
| C5 | gv0.5_shrink0.75_cont | 19.84 | 30.30 | 19.84 | 30.30 | 0.857 | 2.04 | 8.86 | `round4/lab_r4_metrics.jsonl` |
| – | Natural source recording | 0.00 | 0.00 | 0.00 | 0.00 | 0.736 | 4.02 | 9.05 | `budget_check/` |

Leakage on the screen, by gender pair. SIM src is the WavLM-SV cosine to the source speaker's E centroid (`results/anchor_lab/metrics_src.jsonl`); D is SIM src − SIM tgt per pair (M9). All values in this table were computed for this plan from `results/anchor_lab/metrics.jsonl` and `metrics_src.jsonl`.

| ID | SIM src, all 50 | SIM src, F→F (26) | SIM src, M→F (24) | Mean D | Pairs with D > 0, all / M→F |
|---|---|---|---|---|---|
| S1 FAISS | 0.730 | 0.821 | 0.631 | −0.189 | 3/50, 0/24 |
| S3 CRT τ = 0.4 | 0.730 | 0.814 | 0.638 | −0.162 | 6/50, 0/24 |
| A5 src0_faiss | 0.727 | 0.820 | 0.627 | −0.191 | 3/50, 0/24 |
| C1 pm_rq_T1 | 0.721 | 0.768 | 0.670 | −0.098 | 10/50, 3/24 |

C2 and C3 were measured on the F→F pairs only (`results/lab_r3/lab_report.md`): SIM src 0.783 for C2 and 0.786 for C3, against 0.820 for `src0_faiss`, the baseline of that report (FAISS has 0.821 in `anchor_lab/`). Paired with C1 on those pairs, C3 is +0.018 [−0.003, +0.036] and C2 +0.015 [−0.012, +0.045] (pair bootstrap; both marked "!" in that report as above the +0.01 limit). C4 and C5 have no leakage measurement at all. Every candidate needs the full-Dev, per-stratum SIM src of §4.4 before it can be selected.

## 3. Metrics

### 3.1 Metrics (`benchmark/metrics.py`, every signal resampled to 16 kHz)

M1–M7 are as in v0 §3, with these changes:

- M3c capped WER (and capped CER, M3c-CER) is the WER of every v1 rule and endpoint (§0.3 row 20). With E_p = S_p + D_p + I_p the Whisper-large-v3 edit count of pair p and N_p its reference word count, both from the `metrics.py` row, capped corpus WER = 100 · Σ_p min(E_p, N_p) / Σ_p N_p over the pairs of the endpoint (for CER the character counts). Uncapped corpus WER (v0 M3, `metrics.corpus_wer`) is reported beside it everywhere. The Whisper hypotheses themselves are not changed, filtered or re-decoded.
- M8 SIM_src is the WavLM-base-plus-SV cosine between the output and the mean embedding of the source speaker's E set. It is produced in two steps. First, each row is copied with `system` renamed to `<system>@src`, `sim_ref_wavs` set to the source's E set and `target_speaker` set to the source speaker. `crt_anchor_lab.py` does this in `main` (l. 441–448, file `manifest_src`); for the `crt_decode_lab.py` outputs a helper does the same (§8.1 item 5). Second, `metrics.py --metrics sim,secs` scores those rows. Resemblyzer SECS_src is secondary.
- M9 source–target margin is D = SIM_src − SIM_tgt per pair (both WavLM-SV), with the indicator D > 0 (the output is closer to the source centroid than to the target centroid). A voice that is degraded toward neither speaker lowers SIM_src and SIM_tgt together, so D is less affected by degradation than SIM_src. M9 is descriptive.
- M7 coverage (the share of changed CB1–18 tokens) is defined only for code-valued systems. For continuous decoders (C3–C5), the latent distance to the FAISS latent and the energy ratio from `crt_decode_lab.py` (`stats_lat`) are reported instead.

### 3.2 Settings carried over from v0 §7 (binding)

- The τ-rule WER uses Dev `wer_primary` pairs only (500 pairs). v1 applies the same restriction to every Dev WER in §4.
- dB-MCD silence rule: frames at or below −20 dB of the utterance's mean WORLD power are removed before DTW.
- F0: Harvest. log-F0 is normalised by each utterance's own voiced mean.
- EER: each output is scored against every target centroid in the set, and its own target is the genuine trial.
- Whisper: greedy decoding, language en, no timestamps, batched. Whisper-large-v3 WER never falls back to Whisper-base. Every Dev `metrics.py` call passes `--whisper_models large-v3` (the default is `large-v3,base`, `metrics.py` l. 77). The test-set calls keep v0's secondary Whisper-base WER (`large-v3,base`).
- WavLM-SV: the safetensors copy of `microsoft/wavlm-base-plus-sv`, with identical weights.
- The τ grid includes 0.1–0.3.
- Speaker embedding for retrieval and CRT: Resemblyzer on the first 15 s of the reference, as published. At 180 s, retrieval uses the whole budget and the embedding still uses 15 s.
- Temperature-scaled gates need full logits, so the calibration sets run with `--dump_logits`.

Hardware: every number in this plan comes from H200 MIG 2g.35gb slices with Mimi in float16 (`anchor_summary.shard0of20.json`, `cfg.mimi_dtype`), while the SPIE results came from an A100. Decisions near a gate threshold depend on dtype and device (`docs/training_tips.md` C9), so the device is reported with every table.

## 4. Dev selection rules

### 4.1 Common inputs

All rules run on the full Dev set at the 30 s budget, after jobs D1 and D2 (§8.2). The common pair set is every Dev pair that has a non-error metrics row for S1, for every S2 τ value, for A5 and for every candidate, and a non-error `@src` row for S1 and for every candidate. Excluded pairs are counted in the selection record.

The gender strata are same-gender (F→F ∪ M→M) and cross-gender (F→M ∪ M→F), from the `genders` field.

The Dev quantities of a system X are:
- WER(X): capped corpus WER (M3c) from Whisper-large-v3 over the `wer_primary` pairs of the common set. The uncapped corpus WER (`metrics.corpus_wer(..., primary_only=True)`) is printed beside it. §4.2 alone uses the uncapped WER as its rule input (§0.3 row 20).
- SIM(X): mean WavLM-SV cosine to the target E centroid, over all common pairs.
- UTMOS(X): mean UTMOS22-strong, over all common pairs.
- SRC_g(X): mean SIM_src (M8) over the common pairs of stratum g.
- MCD(X): mean dB-MCD over the common pairs that have a parallel recording.

Standard errors and confidence intervals in this section use the linked crossed bootstrap of §5.2 (10,000 replicates, seed 20260928). A single script, `benchmark/select_crt_al.py`, carries out the rules. It is written and hashed before the freeze (§8.1), and its printed output is the selection record. The rules run in the order §4.2, §4.3, §4.4.

### 4.2 S2: the v0 τ rule, unchanged

1. Take the global τ ∈ {0.1, 0.2, …, 0.9} with the lowest Dev WER, subject to Dev SIM ≥ FAISS SIM. The WER here is v0's uncapped corpus WER over the `wer_primary` pairs, because this rule is v0's; the capped WER of each τ is reported beside it (§0.3 row 20).
2. Break ties (equal WER at two decimals, in %) by the lower dB-MCD.
3. If no τ meets the SIM constraint, take the τ with the smallest SIM deficit and report that this happened.

No τ in {0.1, 0.2, 0.3, 0.4, 0.6} met the constraint on the screen, and SIM rose with τ, from 0.849 at τ = 0.1 to 0.905 at τ = 0.6, against 0.919 for FAISS (`results/anchor_lab/metrics.jsonl`). If this holds on full Dev, step 3 picks the τ with the highest SIM, probably the largest τ in the grid, and S2 is then close to FAISS. Family V and the comparison with S2 in E4 would then compare CRT-AL with a near-FAISS system. The rule stays as written.

### 4.3 S1u setting and the E17 slot

- S1u: of `us32_w0.2_0.2_0.1` and `us32_w0.5_0.3_0.3`, take the one with the lower Dev WER, subject to SIM ≥ FAISS. Break ties by the lower dB-MCD. If neither setting meets the constraint, take the one with the higher SIM. (The screen gave SIM 0.938 and 0.944, both above FAISS; `results/lab_r2/lab_report.md`.)
- E17 (`lmbr_k2_g1_T1`, as defined in §2.1) enters the test runs as a descriptive secondary system only if its Dev WER is lower than S2's and its Dev SIM ≥ FAISS SIM (v0 §4). (Screen: uncapped WER 94.80 % over all 50 pairs, capped 90.73 %; on the 20 `wer_primary` pairs 107.58 % uncapped and 94.70 % capped, equal to S2's capped 94.70 % at two decimals; SIM 0.917 against 0.919; `results/lab_r2/lab_report.md`.)

### 4.4 CRT-AL: the pre-registered selection rule

Eligibility. A candidate Ck is eligible if it meets all four conditions:

- E1, similarity floor: SIM(Ck) ≥ 0.80.
- E2, naturalness floor: UTMOS(Ck) ≥ UTMOS(S1).
- E3, leakage guard, in each stratum: SRC_g(Ck) ≤ SRC_g(S1) + 0.01 for g = same-gender and g = cross-gender (point estimates).
- E4, intelligibility gain over the published system and over the no-CRT control: the upper limit of the 95 % CI of WER(Ck) − WER(S2) is below 0, and the same holds for WER(Ck) − WER(A5).

Choice among the eligible candidates:

1. Let C_m be the eligible candidate with the lowest Dev WER. If two are equal at two decimals (in %), C_m is the one with the lower candidate number.
2. For each eligible Ck, let s_k be the bootstrap standard error of WER(Ck) − WER(C_m) (s_m = 0). Let T be the set of eligible candidates with WER(Ck) − WER(C_m) ≤ s_k. This is the one-standard-error rule used in model selection by cross-validation (Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, 2nd ed., 2009, §7.10), with "highest SIM" in place of "simplest model".
3. CRT-AL is the candidate in T with the highest SIM.
4. Ties in SIM (equal at three decimals) are broken by the lower dB-MCD (two decimals), then by the higher UTMOS (two decimals), then by the lower candidate number (C1 before C5).

If no candidate is eligible, the following fallbacks apply in order:

- F1. Some candidates meet E3 and E4, but none meets all of E1–E4. CRT-AL is the candidate with the highest SIM among those that meet E3 and E4. Its E1 and E2 deficits are reported. This is the analogue of step 3 of v0 §4.
- F2. No candidate meets E3 and E4 together, but some meet E4. CRT-AL is the E4-passing candidate with the smallest leakage excess, max over the two strata of SRC_g(Ck) − SRC_g(S1); ties are broken by the higher SIM. The selection record states the Dev leakage, and gate L decides on the test sets as usual (§5.1, §7).
- F3. No candidate meets E4. There is no CRT-AL. The primary system reverts to S2 and the families revert to v0 §5: family V becomes primary, with P8 as v0 defined it, and families K and L are dropped (§5.1, §5.4). The candidates' Dev numbers are reported, and no candidate is run on the test sets.

Pre-specified checks. They are reported, and only the reproduction check can change what happens next:

- Without the screen: the same rule (E1–E4, the choice and the fallbacks) is run on the 950 Dev pairs outside the 50-pair screen. If it picks a different candidate, both picks are reported, and CRT-AL stays as chosen on all 1,000 pairs.
- By gender pair: every candidate's Dev WER, SIM, UTMOS, SIM_src and M9 is reported for each gender combination (F→F, F→M, M→F, M→M), because the screen had female targets only.
- Sensitivity of the choice: the record reports, without changing the choice, which candidate fixed WER tolerances of 0, 1 and 2 points in step 2 would select, and which a rule "the 95 % CI of WER(Ck) − WER(C_m) contains 0" would select.
- Reproduction: on the 50 screen pairs, D1 and D2 must reproduce D0, and D0 must reproduce the screen (§8.1 item 9, with the comparisons and the tolerance stated there). A mismatch stops the selection. Any fix is logged in `benchmark/DEVIATIONS_v1.md` with the new file hashes, the selection is re-run once on the fixed code, and both outputs are reported with the reason.

### 4.5 Why these constraints

All thresholds in this section were set with the screen values known.

E1. The similarity floor is absolute because every candidate's target similarity is below FAISS on the screen (0.818–0.857 against 0.919), so a constraint relative to FAISS would exclude the whole family. The floor of 0.80 was placed so that every candidate passes on the screen. It sits 0.018 below the lowest candidate (C2, 0.818) and 0.064 above the similarity of a different natural speaker (the source recording scores 0.736 against the target E). It removes a candidate whose similarity collapses on full Dev; it does not rank the candidates. A floor at the midpoint between the source recording and FAISS (0.828) was considered and rejected: on the screen it would have excluded C1 (0.819) and C2 (0.818), and on full Dev it would separate candidates by noise near the floor. The similarity cost is not hidden: P3 and P4 test it against FAISS, and §7 treats a significant loss as a trade-off outcome.

E2. The UTMOS floor is relative to FAISS on the same pairs. Every candidate exceeds FAISS by at least 0.33 on the screen (2.04–2.59 against 1.71), so the floor removes only a candidate that is less natural than the system it replaces.

E3. The leakage margin of +0.01 was written before the first source-anchored result (§0.3 row 5). E3 applies it to point estimates for the choice; on the test sets it becomes a formal non-inferiority test (§5.1, gate L). E3 is applied per gender stratum because pooling hides the screen's M→F rise (§0.2 item 7). C1 already exceeds the margin on the screen's M→F pairs (0.670 against 0.631), and every candidate keeps the source CB0, so E3 may fail for all five on full Dev. Fallback F2 then keeps the E4-passing candidate with the least leakage instead of reverting to S2. Reverting would drop the intelligibility comparison without ever testing the leakage on test data; under F2 the leakage is tested by gate L, and §7 caps the framing if gate L fails. This choice is flagged for the author in §10.

E4 keeps the part of prediction 3 of the idea plan's T2.1 that does not depend on SIM: a candidate must beat both the published system and the CB0 swap without CRT on intelligibility.

The tolerance in step 2. An earlier version of this draft used a fixed tolerance of 1.0 percentage point, chosen after the screen. That value matters for the screen's 50-pair column: there, any fixed tolerance of 1.95 points or more selects C5 instead of C3 (C3 17.89 %; C4 and C5 both 19.84 %; SIM 0.838, 0.853 and 0.857). On the `wer_primary` column, which the rule uses, C3 is selected for any tolerance below 7.58 points, C4 between 7.58 and 8.33 points, and C5 above (C3 21.97 %, C4 29.55 %, C5 30.30 %). The interval offered as the basis for 1.0 point was the pair-bootstrap CI of differences in corpus WER over all 50 pairs, about ±3 points with 2,000 resamples (`results/lab_r3/lab_report.md`, paired differences against `anc_R3_s0_pm_rq_T1`: C3 −0.33 [−3.62, +2.79], C2 +0.98 [−1.53, +3.55]; method in `benchmark/crt_lab_report.py`). It was not an interval over the 20 `wer_primary` pairs, and scaling it to full Dev by 1/√n ignored speaker clustering. v1 therefore replaces the fixed value with the one-standard-error rule, whose width is measured on full Dev by the crossed bootstrap and cannot have been set by the screen. Within that width the rule prefers similarity, the main weakness of every candidate (`SUMMARY.md` §7). A "CI contains 0" rule was not chosen, because a 95 % interval is about twice as wide and could admit a candidate whose `wer_primary` WER is several points worse; C5 was 8.33 points above C3 on that column of the screen.

Why the choice uses all 1,000 pairs. The plan's requirement fixes selection on full Dev, as v0 defines Dev. The 50 screen pairs are 5 % of Dev (20 of the 500 `wer_primary` pairs). They are the pairs on which C1–C5 were developed, so including them can push the choice toward the screen's favourite; the 950-pair check shows whether it did.

The screen cannot run this rule as written: it has 20 `wer_primary` pairs, female targets only, no M→F measurement for C2–C5, and no full-Dev bootstrap. Running `select_crt_al.py` on the screen files (§8.1 item 2) is a harness check with no standing.

### 4.6 Selection record

`select_crt_al.py` writes `benchmark/DEV_SELECTION_RECORD_v1.md`, which holds:

- the identity of S2 (τ_dev, and which step of §4.2 decided it), the S1u setting and the E17 decision;
- every candidate's Dev WER, CER, SIM, SECS, UTMOS, SIM_src per stratum, M9 per stratum and dB-MCD with CIs, its E1–E4 flags, s_k, the trace of steps 1–4 and, if used, of F1–F3;
- the 950-pair check, the gender-pair table, the tolerance-sensitivity table and the reproduction check;
- the size check of §5.3, the VCTK-2k variant that T1 will use (5 or 10 targets), and the `wer_primary` counts of both variants;
- the baselines that will run (§2.4), each with its repository commit, checkpoint sha256, sample rate and full command line;
- the listening-test item list and its sha256 (§5.4);
- the job IDs and `sacct` Submit and End times of jobs 2634, D0, D1 and D2;
- the sha256 of every metrics file read.

The record is committed, with its sha256 in the commit message, before the first test-set job is submitted, and it is published or deposited as in §8.4 step 8. It does not edit v1.

## 5. Endpoints and statistics

### 5.1 Families

| Family | Hypotheses | Test | Multiplicity | Role |
|---|---|---|---|---|
| P (primary) | Δ = CRT-AL − FAISS: P1 WER VCTK-2k (`wer_primary`); P2 WER LTC-2k; P3 SIM VCTK-2k; P4 SIM LTC-2k; P5 UTMOS VCTK-2k; P6 UTMOS LTC-2k; P7 dB-MCD on the VCTK-2k parallel subset; P8 CMOS CRT-AL vs FAISS | Two-sided, α = 0.05 | Holm over 8, or over 7 if P8 is dropped (§5.4) | Decides the outcome (§7) |
| L (leakage gate) | Δ_src,g = SIM_src(CRT-AL) − SIM_src(FAISS) in stratum g (same-gender, cross-gender): L1s and L1c on VCTK-2k, L2s and L2c on LTC-2k. H0: Δ_src,g ≥ +0.01; H1: Δ_src,g < +0.01 | One-sided, α = 0.025 each | None. "No leakage beyond 0.01" is claimed only if all four tests reject (intersection–union test) | Condition in §7 |
| K (key secondary) | Δ = CRT-AL − CRT τ = 0.4 on K1–K7 (the endpoints of P1–P7); K8 CMOS CRT-AL vs CRT τ = 0.4 | Two-sided, α = 0.05 | Holm over 8, or over 7 without K8 | Required for any claim of improving on the published system |
| V (v0 primary, demoted) | CRT(τ_dev) − FAISS on P1–P7 | Two-sided | Holm over 7 | Reported in full in the main results table, as the analysis v0 registered |
| S (secondary) | CRT-AL vs each baseline that runs (S5–S8), on WER, SIM and UTMOS, on both test sets | Two-sided | Holm within the family (6 per baseline, at most 24) | |
| Sensitivity | P ∪ K | Two-sided | Holm over 16 (14 without CMOS) | Reported, with whether any P or K conclusion changes |
| Sensitivity (WER cap) | P1, P2, K1, K2 and the WER hypotheses of S with uncapped corpus WER; V's WER hypotheses with capped WER | Two-sided | The same Holm sets as the primary versions, with the other WER definition substituted | Reported, with whether any conclusion changes (§0.3 row 20) |
| Descriptive | S2 outside V, S4, S1u, E17 if it enters, A1–A5, the τ grids, the controls, CER, SECS, EER, the F0 metrics, coverage, M9, the budget sweep, the gender-pair tables and every screen number | CIs only | | |

The WER hypotheses of P, K and S use capped corpus WER (M3c); those of V use v0's uncapped corpus WER (§0.3 row 20).

Error control is per family. P is the primary family. K is controlled separately, and the union is reported as a sensitivity analysis, so a reader can see whether a claim depends on how the families are split.

Reported next to gate L, descriptively, for each set and stratum:
- the M9 difference, mean D(CRT-AL) − mean D(FAISS), with its CI, and the share of pairs with D > 0 for both systems;
- the natural other-speaker level: the mean SIM_src of `target_copy` (natural target speech) on the same pairs. It exists only where a parallel recording exists (Dev and VCTK-2k shared-text pairs), so LTC-2k has no such check. If gate L rejects in a VCTK-2k stratum where CRT-AL's mean SIM_src is below that level, the pass is reported as "not interpretable as absence of leakage", because a degraded voice lowers SIM_src.

FreeVC rows on VCTK-2k stay in family S and carry the "VCTK-seen" flag, as in v0.

If §4.4 ends in fallback F3, family V is primary with P8 as v0 defined it, families K and L are dropped, and family S compares S2 with the baselines, as in v0.

### 5.2 Inference

- Continuous metrics (SIM, UTMOS, dB-MCD, SIM_src, D) use per-pair paired differences with a crossed bootstrap of 10,000 replicates, seed 20260928. A replicate contains every pair whose source and target were both drawn, weighted by the product of their draw counts.
  - VCTK-2k, whose source and target speakers are disjoint: source speakers and target speakers are resampled with replacement, independently (v0).
  - Dev and LTC-2k, where the source pool and the target pool are the same speakers: the primary analysis resamples the speaker pool once, and each speaker's draw count serves for both roles (the linked bootstrap). v0's independent version, which treats a speaker's two roles as separate identities, is a sensitivity analysis.
- Corpus WER uses the same bootstrap on the difference in corpus WER (total errors over total reference words, per system and per replicate). For capped WER each pair's errors are min(E_p, N_p) before the sums; the cap is applied per pair, so it does not depend on the replicate.
- A replicate in which an endpoint has no pair of positive weight, or zero reference words, is redrawn. The number of redraws is reported per endpoint.
- 95 % CIs are percentile intervals. The two-sided p-value is p = min(1, 2 · min(1 + #{Δ* ≤ 0}, 1 + #{Δ* ≥ 0}) / (B + 1)).
- Gate L rejects H0 in a stratum when the upper limit of the two-sided 95 % percentile interval of Δ_src,g is below +0.01.
- Holm: sort the m p-values and reject the i-th smallest while p_(i) ≤ α / (m − i + 1).
- Secondary statistics (from v0): Wilcoxon signed-rank on pair means and the matched-pairs rank-biserial effect size, labelled as possibly anti-conservative.
- Failed conversions: the primary analysis leaves out of a comparison every pair with an error row for either system, and failure counts per system and set are reported in the main tables. Whenever the two systems' failure counts differ, a sensitivity analysis imputes the worst case for the failing system: for WER, every reference word of the failed pair is counted as deleted; for a continuous metric, the failing system's worst observed value on that set (lowest SIM or UTMOS, highest dB-MCD, highest SIM_src).
- Implementation: `benchmark/stats.py`, written and tested on the screen files before the freeze (§8.1).

### 5.3 Size check (v0 §5, made concrete)

After the Dev selection and before any test conversion:

1. Endpoints. For P1, P3, P5, P7 and K1, K3, K5, K7, use the Dev paired differences between the systems they compare. For gate L, use Δ_src,g of CRT-AL − FAISS in each stratum.
2. Variance components. For each endpoint, fit d_p = μ + a_source + b_target + c_source×target + e_p to the Dev per-pair differences d_p by REML (statsmodels `MixedLM` in its variance-component form, with one group), giving σ_a², σ_b², σ_c² and σ_e². For WER, d_p is the linearised per-pair difference 100 · (E_A,p − E_B,p) / N̄, where E = min(S + D + I, N_p) is the capped error count of M3c (for family V, which is predicted only under F3, E = S + D + I uncapped, as V's hypotheses use) and N̄ is the mean number of reference words over the Dev `wer_primary` pairs; its mean equals the difference in corpus WER of the same definition. The prediction of step 3 has no term for the covariance between a speaker's source and target effects, which the linked designs (Dev, LTC-2k) have; the predicted SE can therefore be too small there, and the record states this next to each LTC-2k prediction.
3. Prediction. For a test design with n_s source speakers, n_t target speakers, n_c distinct source–target cells and n pairs entering the endpoint, SE² = σ_a²/n_s + σ_b²/n_t + σ_c²/n_c + σ_e²/n. The counts are read from the pair lists. For VCTK-2k, n_s = n_t = 20 under both variants, so adding targets per source reduces only the cell and residual terms.
4. MDE for P and K: MDE = (z_{1−0.05/14} + z_{0.80}) × SE = (2.69 + 0.84) × SE = 3.53 × SE, which gives 80 % power at Holm's strictest two-sided level for 7 hypotheses.
5. Power condition for gate L, per stratum: (z_{0.975} + z_{0.80}) × SE_g = 2.80 × SE_g ≤ 0.01 − Δ_dev,g, where Δ_dev,g is the Dev estimate of Δ_src,g. If Δ_dev,g ≥ 0.01, the gate has no power in that stratum under any design, and the record says so.
6. Thresholds. The WER threshold is 2.0 points, a judgement of the smallest WER difference the paper would discuss. The SIM threshold is 0.010, the change in target SIM that the idea plan treats as negligible (SIM_target ≥ FAISS − 0.01, l. 188 and l. 317). The UTMOS threshold of 0.05 and the dB-MCD threshold of 0.10 dB are the effect sizes the idea plan expects from a decoding rule (§1, l. 34).
7. Rebuild rule. Evaluate the ten VCTK-2k conditions (eight MDEs and the two L1 strata) for the 5-target list and for the pre-hashed 10-target variant. If at least one condition fails with 5 targets and passes with 10, T1 uses the 10-target variant as hashed, without regeneration. Otherwise VCTK-2k stays at 5 targets, and the record lists the endpoints that remain underpowered. LTC-2k already has 10 targets per utterance and is never changed; its predicted MDEs and its L2 conditions are reported.

The prediction ignores imbalance between cells and treats the components as known, so it is approximate. The record reports each predicted MDE with that caveat.

### 5.4 Listening test

Conditions for MOS and SMOS:
- natural target speech: a target recording that is in neither R nor the SMOS reference (P where it exists, otherwise an E utterance not used as the SMOS reference);
- its Mimi Q = 32 copy-synthesis;
- FAISS;
- CRT τ = 0.4;
- CRT-AL.

kNN-VC and Seed-VC are added only if they run and the pilot shows that the session still fits. FreeVC is excluded because it has seen VCTK.

Items: 32 per condition, 16 from VCTK-2k and 16 from LTC-2k, with 4 per gender combination per set. `select_crt_al.py` draws them with seed 20260928 from the test pair lists (the VCTK-2k variant chosen by §5.3) when it writes the selection record, before any test conversion exists. The item list and its sha256 go into the record. If an item's conversion fails for any condition, it is replaced by the next item in the seeded order within the same set and gender combination, and the replacement is reported.

Scales:
- MOS naturalness: 5-point ACR (ITU-T P.808).
- SMOS similarity: 5-point similarity to a target E reference (VCC2020 style).
- CMOS: CCR −3 to +3 (ITU-T P.800), for CRT-AL vs FAISS (32 same-item pairs, P8) and CRT-AL vs CRT τ = 0.4 (32 pairs, K8). The A/B order is randomised and balanced. Eight hidden identical FAISS–FAISS pairs measure bias.

Raters: 15 to 20 after screening. A 5-rater pilot measures session length and variance. The final count targets 80 % power at a CMOS difference of 0.2, which is a design value and not an expected result, within the limit of 15–20 raters. If the pilot calls for more than 20 raters, 20 are used and the achieved MDE is reported. If a full session takes longer than 60 minutes, each rater rates one of two balanced halves of the MOS/SMOS items, and every rater does all the CMOS pairs.

Screening is fixed in advance and applied by script before any analysis. There is a headphone check at the start. A rater is excluded who rates natural target speech ≤ 2 on more than one gold item, fails an attention item, or gives |CMOS| ≥ 2 on 5 or more of the 8 hidden identical pairs.

Analysis: MOS and SMOS per condition, with 95 % CIs from a bootstrap over raters × items (10,000 replicates). For CMOS, the mean preference for CRT-AL with the same bootstrap and a two-sided p-value; these results enter P (P8) and K (K8). Mixed models with rater and item random intercepts are secondary. Range-equalising bias is discussed, and MOS is also reported within the codec systems.

The v0 condition holds for both CMOS blocks. P8 stays in family P only if at least 15 screened raters complete the CRT-AL vs FAISS block, and the same applies to K8 and its block. Otherwise the family has 7 hypotheses and the listening results are descriptive.

Under fallback F3 the conditions are natural target speech, its Mimi copy, FAISS, CRT τ = 0.4 and CRT(τ_dev) (left out if τ_dev = 0.4). The only CMOS block is CRT(τ_dev) vs FAISS, which is P8 as v0 defined it; there is no K8.

Tool: the open P.808 implementation or a web page. `benchmark/listening_test.py` is not reused (`eval_protocol.md` §6). Ethics approval, recruitment and budget at UIT are still open questions (`extension_plan.md` Appendix B).

## 6. Theory and mechanism analyses (descriptive, pre-specified)

These analyses explain the results. None of them enters §4 or §7. Most were run before this draft, so v1 fixes how they are reported, not what they found. Truth-set results measure latent error, and they are claimed to carry over to audio only where an audio measurement exists. Every truth-set nerr is printed with its energy ratio, because a posterior mean can lower squared error by over-smoothing. Proposition numbers follow the idea plan §7.

| No. | Analysis | Statement it supports | Script → output | Status | Reported as |
|---|---|---|---|---|---|
| T1 | Token instability, all 32 codebooks | Mimi residual tokens flip under small edits, mostly as residual cascades. The stable object is the summed latent | `benchmark/token_stability_32.py` → `results/token_stability_32.{md,json}` (3,621 clips from the interpreter project's audio pool) | Done | Per-codebook table, no tests |
| T2 | Codeword clustering of CRT probabilities | CRT's uncertainty clusters on nearby codewords in decoder space, and hardly at all under RVQ cosine | `benchmark/codeword_clustering.py` → `results/codeword_clustering.{md,json}` (screen); re-run in D2 on the full-Dev `run_conversions.py` dumps of D1 | Screen done | Full-Dev table in the text; screen table in the supplement, labelled exploratory |
| T3 | Orphaned overrides | A per-level gate changes a child token while keeping a parent that CRT wanted to change, so it mixes two residual cascades | Same script (orphaned-overrides column); `crt_decode_lab.py` `orphan_share` in `lab_summary.json` | Screen done | Share per level, on full Dev |
| T4 | Gate monotonicity and the Chow reading (Prop. 2) | The max-softmax gate is the Bayes rule under token loss only if the retrieval expert's per-level accuracy equals τ. The measured accuracies give near-zero thresholds, i.e. argmax-all | `crt_truth_fit.py` §d, §e and the cross-fitted `chow` system → `results/round5/truth_fit_v2.{md,json}`, `results/round5/tau_chow.json` | Done (job 2639) | Per-level decile gains, the "upper interval" column and the `chow` row |
| T5 | Shapley attribution over levels (Prop. 5) | Which levels carry CRT's error reduction, compared with a CRT-free codebook-mean control | `crt_truth_fit.py` §c → `truth_fit_v2.md` | Done | φ_k with speaker-bootstrap CIs. The one-sided speaker-level t tests with Holm over 18, already in the script, are the only tests |
| T6 | Calibration | CRT is overconfident about exact tokens | `crt_truth_fit.py` §b | Done | Held-out NLL (speaker-grouped 2 × 5-fold CV). A term is kept only if the pair-bootstrap CI of its ΔNLL is below 0 |
| T7 | Oracle headroom | How much a per-frame selector or a convex blend of candidate frames could gain | `crt_truth_fit.py` §f | Done | Gains with CIs and shuffled-candidate nulls |
| T8 | Decoding rules on the truth set (Prop. 1, Prop. 3) | Under squared decoder-space loss, the Bayes act depends only on the posterior-mean latent. The edit rule falls back to FAISS when λ = 0 | `crt_truth_fit.py` main table | Done | Supplement table: nerr with energy ratio |
| T9 | Input-matching predictions (Prop. 6) | Whether R3 restores the training relation between anchor and hints | `crt_anchor_lab.py` `summarize` → `anchor_summary*.json`. On full Dev, run over all D1 dumps by the summarise-only entry point (§8.1 item 4), because `summarize` covers only the pairs of one invocation's shard | Screen done | Reported, including the failure described below |
| T10 | Retrieval-failure mechanism | With a 30 s reference, the reference does not cover the source's CB0 tokens and the path rarely advances | `crt_decode_lab.py` `diagnostics` → `lab_summary.json` `diag` (full Dev, D2) | Screen done | Dev table |
| T11 | Negative audio checks | The posterior applied to CB1..K only (round 6); speaker guidance (round 5) | `crt_decode_lab.py` (`pmk*`, `--guide_g`) → `results/round6/lab_r6_metrics.jsonl`, `results/round5/lab_r5_metrics.jsonl` | Screen only | Supplement, labelled exploratory; not re-run |
| T12 | Gaussian optimal transport in decoder space (rounds 7 and 7b) | In Mimi's latent space speaker identity is not a second-order statistic of the frames: matching the speech-frame mean and the covariance in up to the 128 leading principal components, of the acoustic projection or of the whole decoder input, leaves the source speaker. Contrast with WavLM features, where a Gaussian map converts speakers (MKL-VC) | `crt_decode_lab.py` `gaussian_ot`, `ot<k>`, `otf<k>` → `$J/v1/ot_screen`, `$J/v1/otf_screen` (`SUMMARY.md` §8) | Screen only | Supplement, labelled exploratory; not re-run |

Two screen results are reported because they limit the theory claims:

- Prediction 1 of the input-matching hypothesis failed on the screen. The median top-1 probability at CB1–4 was 0.236 under R1, 0.256 under R3 and 0.234 under Psim. Psim was therefore not above R1, so the prediction that R3 closes at least half of the R1 → Psim gap was undefined and was recorded as not holding (`results/anchor_lab/anchor_summary.shard0of20.json`). The idea plan's T2.1 states that the hypothesis fails if confidence does not rise. The paper therefore does not explain R3's WER gain by input matching; it reports the gain as an effect of the intervention and discusses its parts only with the ablations A1–A5.
- The truth-set finding that CRT's information sits in CB1 (T5) did not carry over to audio in the source-anchored setting. On the published-path dumps, applying the R3 posterior to CB1–4 only gave an uncapped WER of 34.47 % (capped 34.15 %) when all levels were re-quantised, and 66.34 % (capped 62.76 %) when the retrieved CB5–31 tokens were kept. C1 gave 18.21 % (both definitions) on the same dumps (`results/round6/lab_r6_metrics.jsonl`, rows `r6p_pmk4_rq` and `r6p_pmk4_keep`, all 50 pairs).

## 7. Outcome framing, fixed in advance

Family P and gate L alone decide the outcome. Family K decides whether the paper may say that CRT-AL improves on the published CRT. Family V is reported in the main results table as the v0 analysis, whatever it shows.

| Outcome | Condition | Framing |
|---|---|---|
| A | At least one P endpoint significantly favours CRT-AL, none favours FAISS, and gate L rejects in all four tests | Method paper: a source-anchored CRT forward pass with latent decoding improves retrieval-based codec VC over FAISS. The input-matching explanation is not claimed, because its prediction 1 failed (§6); the mechanism is discussed only with the ablations A1–A5 |
| B | Significant P effects in both directions, and gate L rejects in all four tests | Operating-point paper: CRT-AL is an intelligibility operating point that costs target similarity. The analysis rests on the WER–SIM frontier of the decoders (shrinkage, variance restoration) and on the risk–coverage material of §6 |
| C | No significant P effect, or only effects favouring FAISS | Analysis paper: "when does post-retrieval refinement help?", built on §6 |
| Leakage not excluded | Gate L fails to reject in at least one of the four tests, whatever P shows | No WER gain is presented without the leakage result next to it, and the abstract states the leakage result and names the set and stratum where the gate failed. The framing is at most B |

A gate-L pass that §5.1 reports as "not interpretable as absence of leakage" is stated as such wherever the leakage result appears.

The screen's point estimates resembled outcome B: WER and UTMOS favoured C1, while target SIM and dB-MCD favoured FAISS. The M→F leakage rise of §0.2 item 7 points toward the "leakage not excluded" row. Both patterns come from 50 exploratory pairs with female targets only, and they do not set the framing.

If fallback F3 of §4.4 makes S2 the primary system, v0 §6 applies unchanged.

## 8. Execution order, compute and freeze

### 8.1 Before the freeze

Status on 18 Oct 2026: items 1–10 are done. D0 (job 4321) passed, and the smoke test (job 4322) ran all four baselines. Every item has unit tests in `benchmark/tests/` (660 tests, run with `.venv-stats/bin/python -m pytest benchmark/tests`). Item 11 is open.

No full-Dev or test-set conversion is run or scored in this phase. The test pair lists and the source recordings are read only to build E sets and the 10-target variant. The 50 screen pairs may be re-run, because their results are already known.

1. `benchmark/stats.py`: the linked and independent crossed bootstraps, the corpus-WER bootstrap with redraw counts, Holm, gate L by stratum, the variance components and the size check of §5.3, and the worst-case imputation. Tested on the screen metrics files.
2. `benchmark/select_crt_al.py`: carries out §4.2–§4.4, draws the listening items and writes the selection record. Tested on the screen files; the screen output is a harness check with no standing.
3. `make_pairs.py`: the VCTK-2k source-E step (§1.2), writing a separate file. The 10-target VCTK-2k variant is generated with `--sets vctk2k --vctk_targets 10` into a separate `--output_dir`, where it writes its own `summary.json` (`make_pairs.py` writes the summary afresh, l. 514), together with its source-E file. With `--sets vctk2k` the script's final Dev-overlap assert has no Dev speakers to compare against, so the variant's speakers are checked against Dev's speaker list explicitly.
4. `crt_anchor_lab.py`:
   - an R1+R3 mode, used for every test-set run (VCTK-2k and LTC-2k, every budget). It skips R3s and Psim and needs no source-speaker reference, so it neither exits on VCTK-2k (today the script exits when a source speaker is not a target, `main`, l. 370–372) nor runs unused recipes on LTC-2k, where it would otherwise run R3s and Psim and fail on source-speaker shortfalls at 180 s (`anchor_pair`, l. 163–164, a `RuntimeError` that is counted as an error, not a shortfall);
   - a system-subset option (A1–A5 and, if selected, C1);
   - `sim_src_wavs` taken from the source-E file (VCTK-2k) or from the target E sets (Dev, LTC-2k);
   - a summarise-only entry point that runs `summarize` over all dumps of a run, for T9 when D1 is sharded.
5. `crt_decode_lab.py` outputs: a helper that carries `sim_src_wavs` into the rows and writes `<system>@src` rows with the transformation of `crt_anchor_lab.py` `main` (l. 441–448), plus the merge step that renames `lab_` rows by job tag, as `run_round4.slurm` does. Without it the decoders have no SIM_src: `crt_decode_lab.py` copies its rows from the `run_conversions.py` FAISS row, which has no `sim_src_wavs`, and it writes no `@src` manifest.
6. The Mimi Q = 32 copy-synthesis control (§2.5).
7. The listening-item draw, inside `select_crt_al.py` (§5.4).
8. Slurm scripts for D0, D1, D2, T1–T4 and M, with full command lines (`--whisper_models` included) and a new `--out_dir` per script and job. `crt_anchor_lab.py` refuses to resume into a directory written with another configuration (`main`, l. 397–400), and the current script has 36 systems where job 2634 had 28, so the screen directories cannot be reused. `run_conversions.py` and `crt_anchor_lab.py` share the dump layout `<out_dir>/dumps/<set>/<budget>/<pair_id>.npz`, so they never share an `--out_dir`.
9. Job D0 on the 50 screen pairs, with these checks:
   - S1 and S3: the D0 `run_conversions.py` codes equal the screen dumps in `journal/clustering_check` (retrieved codes and the τ = 0.4 gate).
   - A1–A5 and C1: the `out_codes` of the D0 anchor dumps equal those of job 2634 (`journal/anchor_lab`).
   - C1–C5 through `crt_decode_lab.py`: the WAVs equal, sample for sample, the screen WAVs of `lab_r3` (C1–C3) and `lab_r4p` (C4, C5). If they are not identical, the check passes only if every pair's Whisper hypothesis is identical and every per-pair SIM and UTMOS differs by at most 0.001 and 0.01 from the screen rows; the largest sample difference is reported. The Mimi copy has no screen counterpart and is not part of this check.
   - The measured per-pair conversion time, decode time and dump sizes (`du`), which replace the estimates of §8.2 and set the array sharding.
   - Every source speaker of `dev.jsonl` and `ltc2k.jsonl` appears as a target; every VCTK-2k source speaker has a source-E set.
10. Baselines: approved and installed on 18 Oct 2026 (§2.4); `download_baselines.sh` and `setup_baselines.sh` are rewritten for `/datastore/nvquoc/crt`. Remaining: record their storage and run the smoke test of §2.4 on 20 screen pairs (`benchmark/job_baselines_smoke.slurm`).
11. The listening-test tool, rater recruitment and the ethics question (§5.4).

Items 10 and 11 may finish after the freeze. Item 10 must finish before the selection record is committed, and item 11 before the listening sessions.

### 8.2 Jobs after the HPC quota raise

HPC rules (admins' guide, 9 Oct 2026). On 18 Oct 2026 the project moved, on the corresponding author's instruction, from the shared `ceai` account to the author's own `nvquoc` account, root `/datastore/nvquoc/crt` (the same layout below that root; the copy was verified file by file, and the rebuilt environment's `pip freeze` is identical to the old one). Until 18 Oct 2026 the `nvquoc` account also ran a live service whose cron job resubmitted a job at 04:00 every second day; that service has since moved off the account. The rules:
- Every job uses `--account=uit --partition=normal --gres=gpu:2g.35gb:1`, with `--time` at most 3 days.
- At least 1 h between two submissions on the account, whoever made the last one, and no submission between 03:00 and 04:10 local time (kept in case the service returns).
- Fewer than 3 jobs of the account, of any name, running or pending at a submission. The corresponding author allowed CRT the account's 3 running jobs once the service had left.
- `benchmark/submit_nvquoc.sh` enforces all three rules and refuses to submit when `sacct` or `squeue` fails.
- A job is submitted only after the jobs whose outputs it reads have finished: the D1 metrics and D2 both after the D1 conversions, and T(n+1) after T(n). Arrays use `%1`.
- The interpreter session is still told of every submission.
- Before D1, the HPC copies of the frozen files are checked against `benchmark/FREEZE_v1.sha256`, and `grep -l /datastore/eaic` over the frozen file list must return nothing. The grep is restricted to that list because over all of `benchmark/` it matches 16 files today, among them `download_baselines.sh`, `setup_baselines.sh`, `download_eval_models.sh` and `listening_test.py`; an earlier re-copy also reverted edited scripts. Exception: `pairs/summary.json` keeps `/datastore/eaic` strings in its `args` and `jsonl` metadata fields; no job reads a path from it, so it is hashed but left out of the grep.

| Job | Starts after | Work | New WAVs | Audio (GB) | Logit dumps (GB, bound) | Metric rows | Metric time |
|---|---|---|---|---|---|---|---|
| D0 | Before the freeze | §8.1 item 9 on the 50 screen pairs | 2,350 (47 systems × 50 pairs) | < 1 | < 1 | about 4,700 | ≈ 0.8 h |
| D1 | Freeze and public timestamp (§8.4) | Dev, 1,000 pairs, 30 s: `run_conversions.py --systems faiss --dump_logits`; `crt_anchor_lab.py --dump_logits --dump_recipes R3` (36 systems, including the R1 and R3 τ grids, A1–A5 and C1); `metrics.py` on `manifest` and `manifest_src`; summarise-only over all dumps | 37,000 | 8.7 | 9.0 (R1 and R3) | 74,000 | 12.1 h |
| D2 | D1 finished | `crt_decode_lab.py` for C1–C5, S1u (two settings) and E17 (commands below); Mimi Q = 32 copies; natural controls scored; `codeword_clustering.py` on the D1 R1 dumps; `metrics.py` including `@src` | 9,500 | 2.2 | 0 | 19,500 | 3.5 h |
| Local | D2 finished | Metrics copied to the Mac; `select_crt_al.py` run; selection record committed and timestamped | – | – | – | – | minutes (CPU) |
| T1 | Record committed and timestamped | VCTK-2k (2,000 pairs; 4,000 if §5.3 triggers), 30 s: S1–S4 with `--dump_logits`; the R1+R3 mode for A1–A5 and CRT-AL (R3 dumped only if CRT-AL is one of C2–C5); `crt_decode_lab.py` for S1u, E17 if it entered, and CRT-AL if it is one of C2–C5; Mimi copies. No metrics | 23,000–27,000 (46,000–54,000) | 5.4–6.3 (10.8–12.7) | 9.0 if C1, 18.0 if C2–C5 (doubled for 4,000) | – | – |
| T2 | T1 finished | LTC-2k (1,800 pairs), 30 s: as T1, without parallel recordings. No metrics | 19,800–23,400 | 4.6–5.5, × r_LTC | 8.1 if C1, 16.2 if C2–C5, × r_LTC | – | – |
| T3 | T2 finished | Budget sweep at 3 s and 180 s on both test sets: S1, S2 (if τ_dev ≠ 0.4), S3, CRT-AL. No metrics | 22,800–30,400 (5-target VCTK-2k) | 5.4–7.1 | 0 if C1; up to 34.2 if C2–C5 and the R3 dumps are kept | – | – |
| T4 | T3 finished; only baselines listed in the record | S5–S8 at 30 s on both test sets; kNN-VC also at 3 s and 180 s. No metrics | ≤ 22,800 (5-target VCTK-2k) | ≤ 5.4 | 0 | – | – |
| M | T4 finished (T3 if no baseline is listed) | `metrics.py` on every test output, `@src` rows included | – | – | – | about 147,000 (5-target VCTK-2k) to 238,000 (10-target, every optional system) | about 33–52 h |

D1 commands (paths under `J=/datastore/nvquoc/crt/mimi_vc/benchmark/journal`; `RC` and `ANC` are new directories):
```
python -u $J/run_conversions.py --pairs $J/pairs/dev.jsonl --set dev --budget 30 --systems faiss \
    --dump_logits --out_dir $RC --device cuda
python -u $J/crt_anchor_lab.py --pairs $J/pairs/dev.jsonl --set dev --budget 30 --conv_dir $RC \
    --out_dir $ANC --dump_logits --dump_recipes R3 --device cuda
python -u $J/metrics.py --in_jsonl $ANC/dev/30/manifest.jsonl --out_jsonl $ANC/dev/30/metrics.jsonl \
    --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3 --device cuda
python -u $J/metrics.py --in_jsonl $ANC/dev/30/manifest_src.jsonl --out_jsonl $ANC/dev/30/metrics_src.jsonl \
    --metrics sim,secs --device cuda
```
Both conversion scripts accept `--shard i/n`; if D0's timing calls for sharding, the shards' manifests are merged before `metrics.py`, and T9 uses the summarise-only entry point.

D2 commands. `crt_decode_lab.py` applies `--cb0` to every system of a call, so the source-CB0 candidates and the retrieved-CB0 front-end systems need separate calls and separate output directories (`LAB` is a new directory):
```
python -u $J/crt_decode_lab.py --conv_dir $RC/dev/30 --dumps $RC/dumps/dev/30 --out_dir $LAB/cand \
    --systems pm_rq_T1,pm18_rq,gv1_pm_cont_T1,gv1_shrink0.85_cont,gv0.5_shrink0.75_cont \
    --logits_from $ANC/dumps/dev/30 --logits_key R3 --cb0 src --device cuda
python -u $J/crt_decode_lab.py --conv_dir $RC/dev/30 --dumps $RC/dumps/dev/30 --out_dir $LAB/front \
    --systems us32_w0.2_0.2_0.1,us32_w0.5_0.3_0.3,lmbr_k2_g1_T1 --device cuda
```
Both calls name their rows `lab_<system>`, so the merge step of §8.1 item 5 renames them by tag before `metrics.py --whisper_models large-v3` and the `@src` scoring. `crt_decode_lab.py` runs as one process with no `--shard` and no resume, and it opens `manifest.jsonl` for writing at the start (l. 757), so a killed run loses its manifest. D0's decode time × 1,000 pairs × 8 systems must therefore fit the 3-day limit with a wide margin; if it does not, sharding is added to the script before the freeze.

How the estimates were made:
- Output length: the mean output is 4.89 s and 61.1 Mimi frames long (`results/anchor_lab/metrics.jsonl`, fields `dur_s` and `n_frames`), so one 24 kHz 16-bit WAV is 0.235 MB.
- Logit dumps: one R1 or R3 logits array is 18 × 61.1 × 2,048 × 2 bytes ≈ 4.5 MB per pair at float16. This is an upper bound: both scripts write dumps with `np.savez_compressed` (`run_conversions.py` l. 525, `crt_anchor_lab.py` l. 258), and the compressed size has not been measured. D0 measures it.
- The other dump arrays (top-k, candidate scores, confidence and entropy per recipe, reference codes) are not counted and have not been measured. D0 measures them.
- Scoring time: about 1.1 s per output with `--whisper_models large-v3`. The mean per-row `time_s` totals are 1.07–1.14 s in `anchor_lab/metrics.jsonl`, `lab_r3/metrics.jsonl` and `round4/lab_r4_metrics.jsonl`. Scoring an output against the source speaker (`@src`) takes 0.074 s per row on average (median 0.033 s; `anchor_lab/metrics_src.jsonl`; 0.076 s in `round4/anchor_us32/metrics_src.jsonl`). The test-set job M also runs Whisper-base (§3.2), which these figures do not include, so M's time is a lower bound.
- Metrics rows: about 12.4 KB each (17.3 MB for 1,400 rows in `anchor_lab/metrics.jsonl`).
- LTC-2k: outputs and logits scale with the source duration. r_LTC is the mean LTC source duration (`source_dur` in `ltc2k.jsonl`) divided by 4.89 s, and it is read before the freeze. LTC sources are at most 10 s, so r_LTC ≤ 2.05. The T3 and T4 ranges above are at r_LTC = 1.
- WAV counts per test pair: S1–S4 (3 if τ_dev = 0.4), A1–A5, CRT-AL, S1u, E17 if it entered, and Mimi copies of the source and, on VCTK-2k, of P (at most half the pairs, from the shared-text share in v0 §1).
- Conversion and decoding time was not measured separately and is not included. It can be read from the `date` lines in the HPC logs of jobs 2629, 2634, 2636 and 2638, or measured in D0. Baseline inference time is not included either.

Totals (est., storage):
- About 63 GB in the smallest case considered: 5-target VCTK-2k, CRT-AL = C1, E17 not entering, τ_dev ≠ 0.4, r_LTC = 1, T3 dumps not kept.
- About 81 GB if CRT-AL is one of C2–C5, and about 115 GB if the T3 R3 dumps are also kept.
- About 212 GB in the largest case: 10-target VCTK-2k, CRT-AL one of C2–C5, E17 entering, r_LTC = 2.05, T3 dumps kept.
- Not included: the other dump arrays, and the baseline repositories, weights and environments (§8.1 item 10). None of these has been measured.

Metric time is about 48 to 68 h of single-task time (D1 + D2 + M), run as two array tasks at a time.

On 11 Oct 2026 the `ceai` account was capped at about 43 GB, with about 3 GB free, which could not hold this work. On 18 Oct 2026 a write test on `nvquoc` stored 100 GiB without a quota error; the account's actual quota is not known, and the account's live service shares it. The project therefore keeps its own total under about 90 GiB until the quota is known, and every job prints `du` of its outputs. Logit dumps can be deleted once the decodes that read them have been checked, but only with the user's approval. If CRT-AL is C1, T3 needs no R3 dump, because the R1+R3 mode decodes `pm_rq_T1` directly.

The listening test needs no GPU. Its items are drawn when the selection record is written (§5.4), and the sessions start once T1 and T2 exist.

### 8.3 Order

v0 committed → §8.1 items 1–9 → D0 → freeze and public timestamp → D1 → D2 → selection record (committed and timestamped) → T1 → T2 → T3 → T4 (baselines listed in the record) → M → statistics on the test sets, run once → listening test → paper.

### 8.4 Freeze procedure

1. At the freeze, before the v1 commit, the author commits `benchmark/ANALYSIS_PLAN.md` unchanged, staged by name, in a commit of its own (§0.1). Claude makes this commit only on the author's explicit instruction. (The draft of 11 Oct 2026 placed this commit before D0. On 18 Oct 2026 the author chose to make it at the freeze together with the v1 commit; D0 only re-runs the 50 screen pairs, whose results were already known, and v0 is committed byte for byte as it stands, so the order does not change what v0 says.)
2. The corresponding author, Quoc Nguyen, reads this draft, edits it as needed and approves it. Only the author freezes it.
3. §8.1 items 1–9 are done and D0 passes.
4. The approved text is saved as `benchmark/ANALYSIS_PLAN_v1.md`, with the status line changed to "frozen on (date) by Quoc Nguyen" and the v0 commit hash entered in §0.1.
5. Hashes go into `benchmark/FREEZE_v1.sha256`, computed where each file lives:
   - on the Mac, with `shasum -a 256`: the plan; `make_pairs.py`, `run_conversions.py`, `crt_anchor_lab.py`, `crt_decode_lab.py`, `metrics.py`, `stats.py`, `select_crt_al.py`, `codeword_clustering.py`, `crt_truth_fit.py`, the `@src` and merge helper, `v5.6.5-secs-high/train_crt.py` (from which `run_conversions.py` imports the CRT class), `run_round4.slurm` (the pattern of the D2 calls) and the D0, D1, D2, T1–T4 and M Slurm scripts;
   - on the HPC login node, with `sha256sum`, by the author: the pair lists as they are on the HPC, the source-E files, both VCTK-2k variants with their `summary.json`, the WavLM-SV and UTMOS checkpoints, and the output of `pip freeze` in `/datastore/nvquoc/crt/envs/mimi_vc`. The sha256 values inside `pairs/summary.json` describe the files before their paths were rewritten, first from `/datastore/eaic` to `/datastore/ceai` and then to `/datastore/nvquoc/crt`, with `sed`, so they no longer match the HPC bytes; `FREEZE_v1.sha256` records the bytes actually used;
   - recorded values: the CRT weights b26d6f945974d729675c2e8c35af61ea6feb38c71445500a1cda448e3f8d7542 (the state-dict hash of `run_conversions.state_sha256`: tensor names, dtypes, shapes and bytes; the file `models/crt_v565_best_weights.pt` itself has sha256 ede3497b68ff4ba8b4a348dc2d4ab3ceb3464a20cc324b5e246d179c1cf48dbc) and the Mimi commit 89091b3e466eb6a9d11e537bf26b144f194978f7 (`results/anchor_lab/anchor_summary.shard0of20.json`, `cfg`), both re-checked at run time; the Whisper-large-v3 checkpoint, whose SHA256 `metrics.py` checks before use (`check_whisper_cached`).
6. The author commits the plan, `FREEZE_v1.sha256` and the frozen scripts, staged by name, with the plan's sha256 in the commit message, and creates the annotated tag `analysis-plan-v1`. Claude does this only on the author's explicit instruction.
7. Public timestamp, a precondition of D1: the tag is pushed to a public remote (the author first confirms that `github.com/nguyenvietquoc2569/crt` is public), or the plan and `FREEZE_v1.sha256` are deposited with a third-party timestamp such as OSF or Zenodo. If the author chooses neither, the paper calls the plan "internally frozen", not "pre-registered".
8. The same applies to the selection record before T1. The D1 and T1 job IDs and their `sacct` Submit times go into the selection record and into `benchmark/DEVIATIONS_v1.md`, and each push or deposit time must come before the Submit time of the job that follows it: the plan's before D1, the record's before T1.
9. Any later change is a deviation. It is dated, given with its reason in `benchmark/DEVIATIONS_v1.md`, and reported in the paper.

## 9. Rules against rescuing results

Kept from v0 without change:
- No metric, τ, subset or reference budget is chosen after test scores are seen.
- Test sets are scored once per frozen system.
- Null and negative rows stay in the tables.
- Numbers from the SPIE paper are never reused unless this harness re-measures them. `paper_facts.md` and `lean_plan.md` list the known problems:
  - the source-referenced MFCC "MCD";
  - the untraceable 100-pair WER (13.5 / 25.0) and the untraceable 17.8 % override rate;
  - "1.1 kbps", where 32 codebooks give 4.4 kbps;
  - the hard-coded streaming table;
  - the RVQ dropout that the code does not contain;
  - about 2,340 unique training speakers.
- Conference-100 results appear only in the corrections appendix.

Added in v1:
- The candidate set C1–C5 is closed at the freeze. No candidate, decoder, temperature, shrinkage weight or variance-restoration exponent is added or changed afterwards, whatever full Dev shows.
- No decoder is developed, tuned or added on test data. Candidates that are not selected are not converted on the test sets.
- No test-set conversion starts before the selection record is committed and timestamped.
- No test-set metric is computed or opened before every test conversion has finished, the baselines' included (job M follows T4). Baseline settings are fixed in the selection record, a baseline not listed there is not run, and S8 is FACodec or nothing.
- Every candidate's full-Dev numbers are reported, together with the E1–E4 flags, the selection trace, the tolerance-sensitivity table, the S2 τ grid, the 950-pair check and the gender-pair table.
- Screen numbers are always labelled exploratory and are never pooled with full-Dev or test data in a test. The paper states how much was screened: 176 system rows in nine files, 162 distinct outputs (156 at 30 s without the natural controls), and 71 decoding rules on the truth set; then, on 18 Oct 2026, 15 more system rows in two files (14 distinct outputs; rounds 7 and 7b, §6 T12).
- The truth set never selects or tunes a source-anchored system.
- Resemblyzer SECS never enters a selection, and no output is chosen per utterance with Whisper, WavLM or UTMOS.
- If the outcome is B or C, no configuration found after the freeze and no non-selected candidate is promoted to the headline.
- Failed conversions are counted and reported, never dropped silently.

## 10. Review notes

Notes on review issues that this draft adopts only in part, or handles differently from the reviewer's suggestion:

- Selection on 950 instead of 1,000 pairs: kept on 1,000 pairs, because the plan's requirement fixes full Dev; the 950-pair repetition stays as a reported check (§4.5).
- Tolerance in step 2: replaced by the one-standard-error rule instead of the suggested "CI contains 0" rule, which could admit a candidate several WER points worse (§4.5); the CI rule is reported as a sensitivity row.
- E3 per stratum: adopted, with an added fallback F2 that keeps a leaking but intelligible candidate under gate L instead of reverting to S2. This goes beyond the review; the corresponding author chose F2 on 18 Oct 2026. Ties in F2's leakage excess are judged at three decimals, like the SIM ties of §4.4 step 4.
- Dump size: the suggested "about 3 MB per pair per recipe" is not used, because no local file shows it; the 4.5 MB bound stays until D0 measures the compressed size.
- Whisper models: Dev calls use large-v3 only, as suggested, but the test-set calls keep Whisper-base as v0 M3's secondary WER, so job M's time is a lower bound.
- Deviation table: the suggested "written after the screen" column would read "yes" in every row, so it is stated once above the table instead.
- E16 sweep: S2 is added to the sweep to stay close to v0 (§0.3 row 18), which the review did not ask for; it costs at most one system per budget.

The corresponding author's decisions of 18 Oct 2026:
- Fallback F2 of §4.4 is kept: if every candidate fails E3, the E4-passing candidate with the least leakage is kept, and gate L decides on the test sets.
- The primary WER is capped corpus WER (M3c), with uncapped corpus WER as a sensitivity analysis (§0.3 row 20).
- The four baselines kNN-VC, FreeVC, Seed-VC and FACodec are downloaded and run (§2.4).
- Freeze: once D0 passes and the author approves this text, v0, v1, `FREEZE_v1.sha256` and the frozen scripts are committed (staged by name), tagged `analysis-plan-v1` and pushed to the public GitHub repository as the public timestamp of §8.4 step 7, after the author confirms that the repository is public.
- Still open: whether the checkpoint is released before submission (corrections section of the paper outline).
