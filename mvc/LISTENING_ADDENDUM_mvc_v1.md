# MimiFlow-VC (paper 3): listening test addendum v1

Status: FROZEN on 30 Oct 2026, 03:02 +07:00 by Quoc Nguyen, the corresponding author, who on 30 Oct 2026 delegated the remaining decisions of §16 to Claude ("make it easy, you decide it, let go"). Drafted 29 Oct 2026, revised the same day after a protocol review, revised again the same day to the author's decisions of 29 Oct 2026 (§16 items 1, 3, 17 and 27), aligned the same day with the website server and the builder after an independent review (§0.2 item 11), and revised on 30 Oct 2026 to the decisions made that day under the author's delegation, which settle every item of §16 except UIT's answer on ethics, and revised again the same day after an independent review (§16; §0.2 items 12 and 13). Quoc Nguyen, the corresponding author, edits, approves and freezes it. This is the addendum that `mvc/ANALYSIS_PLAN_mvc_v1.md` §6.4 calls for.

Nothing in it has been run:
- no item has been drawn with the seed string of §3.1;
- no stimulus has been built;
- no rater has been invited, and no rating of this test exists.

From the freeze on, every change is a deviation (§14).

File: `mvc/LISTENING_ADDENDUM_mvc_v1.md`. It adds a listening test to the frozen analysis plan (sha256 46f2b586640fb5a7fee6654343a5403a4f93e96d5e7b1558219f701755bfcd86) and changes nothing in it. No rule, family, threshold or outcome of the plan's §4–§7 depends on a listening result.

- The design adapts the journal's listening test, `benchmark/ANALYSIS_PLAN_v1.md` §5.4 (sha256 64430ad401ec2c512d6977820f2dfb3ae43aaa20bd603ed7d379f801f8ebe63e).
- It reuses the journal's tested page and scripts, copied into `mvc/mos/`. Nothing under `benchmark/` is changed.
- Every rule of §2–§6 that the stimulus builder applies is also written in the design file `mvc/mos/design_mvc_v1.json`, which is frozen with this text (§13).

Paths:
- on ceai: `J=/datastore/ceai/crt/mimi_vc/benchmark/journal` and `MVC=/datastore/ceai/crt/mimi_vc/data_storage/mvc`;
- on the Mac: `V1=paper/journal_extension/results/v1` and `MR=paper/mimi_vc/results`;
- the data directory `D=$MOS_DATA_DIR` (default `~/mos_data`), which is outside the repository.

## 0. Lineage

### 0.1 Relation to the analysis plan

- Plan §6.4: "A MOS/SMOS test (PLAN §6.2 item 5) needs its own addendum (conditions, items drawn by seed from the test lists, scales, rater screening, analysis), frozen before any stimulus is built. Without one, any listening result is descriptive." This text is that addendum.
- PLAN.md §6.2 item 5 asks for "a small listening test (MOS/SMOS), using the existing paper-2 listening tooling".
- The systems are the plan's frozen T1 systems and comparators (plan §2).
  - mvc_v1 is the selection record's choice: R1 `ema_0200000.pt`, CFG 1, SNG 1.0 (fallback F; `mvc/MVC_SELECTION_RECORD_v1.json`, sha256 b84885feb5fa0dc9b8c4cf35ee0f772f0cffe65cf21016f74d70f64c0920b9c8).
  - The stimuli are the T1 outputs at the 30 s budget. Nothing is converted again.
- The confirmatory family LT (§9.1) is new. It holds two hypotheses, both against CRT-AL. It sits beside the plan's families H1, NL, L, S, NI, RB, RBN and Q, and changes none of them.

### 0.2 What was seen before this draft

1. All T1 results: `mvc/TEST_RESULTS_mvc_v1.md`, written 28 Oct 2026, 22:43. These include outcome A, natural-level leakage not excluded, families S and NI, ECAPA and DNSMOS. The comparisons of §9.1, and the author's decision of 29 Oct 2026 to confirm only against CRT-AL (§1), were made with these numbers known:
   - mvc_v1 is ahead of CRT-AL by +1.201 / +1.296 UTMOS, +0.080 / +0.017 SIM and +0.295 / +0.176 ECAPA SIM (VCTK-2k / LTC-2k);
   - Seed-VC is ahead of mvc_v1 on SIM (S-seedvc-3, S-seedvc-4) and on ECAPA SIM (−0.271 / −0.113);
   - kNN-VC is ahead of mvc_v1 on ECAPA SIM (−0.191 / −0.034);
   - ECAPA coverage on the 980 VCTK-2k pairs with P is 0.522 for mvc_v1, 0.070 for CRT-AL, 0.834 for kNN-VC and 0.943 for Seed-VC.
2. The X-mvc-1 results (`mvc/EXPLORATORY_XMVC1_v1.md`, 29 Oct 2026, 07:59). R1x is not part of this test (§2.2).
3. Listening by the author:
   - the Dev pages `paper/mimi_vc/status/mvc_listen.html`, `mvc_listen_v2.html` and `mvc_listen_v3.html`, and the blind A/B of relax (Dev; plan §0.2 item 5);
   - the test-set page `paper/mimi_vc/status/mvc_listen_test.html` (29 Oct 2026, 02:44). It has 84 labelled clips from 12 pairs: source, natural target, mvc_v1, CRT-AL, Seed-VC, kNN-VC and FreeVC. The pairs were chosen by mvc_v1's own WER, SIM and UTMOS.
   These 12 pairs are excluded from the draw (§3.3). Decided 30 Oct 2026 (§16 item 30): only the author has heard this page, and the author does not rate. The page is shared only as a private claude.ai artifact and has never been in the public repository: no file under `paper/mimi_vc/` is tracked or has ever been committed (`git log --all`, checked 30 Oct 2026). `.gitignore` now excludes the labelled pages (`paper/mimi_vc/status/mvc_listen*.html` and `mvc_blind*.html`), so that they cannot be committed by mistake.
4. For the first draft, the pair lists were read to count eligible pairs, and the T1 rows were read to compute per-pair spreads of the UTMOS and DNSMOS differences (§10).
5. For the first draft, the draw procedure was run 1,000 times per set with throwaway seed strings and scratch code, to check the speaker rule. No item list was kept.
6. A development draw. While the builder was being written, `build_stimuli.py draw` ran once on the real pair lists with a draft design:
   - seed `20260928`, that is, the journal's stream `'20260928:listening:<set>:<g>'`;
   - six conditions (FACodec included) and P pairs first in VCTK-2k;
   - none of the duration, speaker, sentence or exclusion rules of §3.2–§3.3.
   Its output (32 items, 3 practice items, 0 replacements) was the scratch file `/private/tmp/claude-501/-Users-harry-duplex/bacaee22-78f6-4537-8e78-4807b7f536d4/scratchpad/mos_draw/items.json`. Only AI agents read it: the builder's agent, which reported only these counts and that all 16 VCTK-2k items had P, and the protocol review of item 7. The folder no longer existed on 29 Oct 2026 at 14:04. Decided 30 Oct 2026 (§16 item 6): the file was deleted on 29 Oct 2026, and no person read its item list.
   This draw is not the test's: §3.1 uses another seed string and other rules. The builder now refuses to draw on the real pair lists before the freeze (§13).
7. The protocol review (29 Oct 2026, about noon), run by an AI agent, read:
   - the pair lists;
   - the T1 rows' fields, among them `utmos` (the spreads of §10) and `dur_s` (output length per system, §4.2);
   - the server code.
   It also simulated the bootstrap of §9.1 under the model of §10, and ran 300 throwaway draws per set with its own code.
8. For this revision, the builder's own `draw()` ran 300 times on the real pair lists with the seed strings `throwaway-feas-0` … `throwaway-feas-299`: once under the rules of §3, and once more without the sentence rule.
   - It read only pair-list fields and whether each condition's T1 row exists without an error. It read no metric.
   - Only counts were printed (§3.2, §3.3, §6.2), and no item list was kept.
   - The estimator of §9.1 and its bootstrap were simulated under the model of §10, without data (§10).
   The seed string of §3.1 has never been used in a draw (see also item 13).
9. The journal's own listening test has not been run. `benchmark/` holds no rating file and no `LISTENING_RESULTS_v1` or `LISTENING_PILOT_v1`. Its 32 items (`benchmark/DEV_SELECTION_RECORD_v1.listening_items.json`, sha256 145aa331c4e8b16b9ce5b45a4aebf07108eac758cbda4c412f5f06e6900fc019) are not reused (§3.1, Decision 6).
10. For the revision to the author's decisions, the protocol review's simulation of §10 was run again, unchanged and with its own seed, to read off the MDE under Holm over 2. It reproduced the SEs of §10 exactly. It used no data and no draw.
11. To align this text with the website, an independent review and the revision that followed (29 Oct 2026, afternoon) read the server, the page, the builder, the design file, the consent template and the README of `mvc/mos/`, the ECAPA table of `mvc/TEST_RESULTS_mvc_v1.md` (for LT-2, §1) and the family table of the plan. They ran the test suites of `mvc/mos/`, which use synthetic data only. No draw was run.
12. For the revision of 30 Oct 2026 to the delegated decisions, and for the revision after the independent review that followed it the same day, the server, page, builder, design file, consent template and README of `mvc/mos/` were read again. The consent template (§11.5), the README, the page's break and end texts (`public/i18n.js`, §6.3, §11.3), the admin page's note on the code cap and `.gitignore` (item 3) were revised, and the test suites of `mvc/mos/` were run (synthetic data only). Read-only git commands showed that no file under `paper/mimi_vc/` has ever been committed. Apart from the checks of item 13, no draw was run, and no T1 row, metric or rating was read.
13. Before the freeze, on 30 Oct 2026, an AI agent ran the checks of §12 step 2 and one feasibility draw:
    - the builder's tests with `MOS_GIT_TESTS=1` (78 passed, the guard test against real git repositories among them) and without it (77 passed, 1 skipped), and the server and page tests (68 and 32 passed), all on synthetic data. After the revisions of item 12, the same suites passed again with the same counts;
    - one draw on the real pair lists with `mvc/mos/design_mvc_v1.json` (sha256 6a1cb8fa5b095fa387a6fa32567f1258aaaa577216bc909b362d283b734a13b5) and the throwaway seed string `feasibility-check:20261030`. It read pair-list fields and whether each condition's T1 row exists without an error, and no metric. Only counts were printed: 32 items, 3 practice items and 76 pairs passed over (speaker rule 61, of which 43 for the source and 18 for the target speaker; sentence rule 6; source duration 5; no natural clip 2; P duration 1; excluded pair 1). All 16 VCTK-2k items had P, and 4 of the 16 LTC-2k items had a cut reference. No item list or pair ID was printed, and the output was deleted the same day;
    - `build` refused those items. Two runs of `draw` without `--throwaway_seed` were refused by the guard before any draw (no addendum given; no tag), so no draw has used the seed string of §3.1.

### 0.3 Differences from the journal design (ANALYSIS_PLAN_v1 §5.4)

| No. | Journal §5.4 | This addendum | Reason |
|---|---|---|---|
| 1 | Conditions: natural target, its Mimi copy, FAISS, CRT τ = 0.4, CRT-AL; kNN-VC and Seed-VC only after the pilot | Natural target, mvc_v1, CRT-AL, Seed-VC, kNN-VC; natural source in SMOS only | §2 |
| 2 | CMOS blocks P8 and K8, with 8 hidden identical pairs | No CMOS | The questions are MOS and SMOS; session length (§6.2) |
| 3 | Listening results enter families P (P8) and K (K8) | Own family LT of 2 hypotheses, MOS and SMOS of mvc_v1 − CRT-AL, Holm over 2 (§9.1); everything against Seed-VC and kNN-VC descriptive (§9.3) | The plan has no listening hypothesis; the author's decision of 29 Oct 2026 (§1) |
| 4 | Items drawn by `select_crt_al.py` with `random.Random('20260928:listening:<set>:<g>')`; no eligibility rule; the list is written into the Dev selection record | Drawn after the freeze by `mvc/mos/build_stimuli.py draw` with `'20260928:mvc_v1:listening:<set>:<g>'`; P pairs first in VCTK-2k; durations of 2–10 s; within a set, one item per source speaker, per target speaker and per sentence; the 12 shown pairs left out (§3) | A stream separate from the journal's; same-sentence natural speech; items spread over speakers and sentences; no cut natural clips; items fixed by frozen code, not by a list chosen with T1 known |
| 5 | Halves only if the pilot passes 60 min | Halves fixed now (§6.2) | A full session is estimated at 95–106 min |
| 6 | Gold rule: natural target ≤ 2 on more than 1 gold item (32 per rater) | More than 2 of 16 (§8) | Each rater has 16 gold items |
| 7 | Attention items ask for 3, 3 (MOS) and 2 (SMOS) | MOS asks for 2 and 4; SMOS asks for 4 and 2 | No constant rating passes either block (§8) |
| 8 | Rater count from the pilot's CMOS variance, 15–20 | 20 screened raters (10 per half) over both access routes, fixed now; LT needs at least 8 per half (§7.1, §10) | §10 |
| 9 | Crossed bootstrap over raters and items, without strata; mean over ratings | Stratified (raters within half, items within set × half); estimand with equal item weights (§9.1); the unstratified version is a sensitivity analysis | The halves form two disjoint blocks; halves may end unequal |
| 10 | Offline page; raters e-mail their results file | Website on the Mac; only answers stored on the server count, and an e-mailed results file is never analysed (§11.3); raters enter with an invite code or through open registration, and the route is recorded (§7.3, §11.2) | The author's request; both routes decided 29 Oct 2026 |
| 11 | Consent draft for an offline page | Consent for a server that stores answers (`mvc/mos/consent_template_v1.html`) | §11.5 |
| 12 | Stimuli keep their own length | 0–250 ms of random trailing silence per clip; durations rounded to 0.1 s for the browser (§4.2) | The output length tells the systems apart |

## 1. Purpose and status of the results

- The test asks two things:
  - whether listeners confirm what T1 measured for mvc_v1 against CRT-AL, the system of the published paper that mvc_v1 is meant to replace (family H1);
  - how listeners judge mvc_v1's similarity to the target speaker against the two baselines where T1 found clear SIM differences.
  The first question is confirmatory. The second is descriptive.
- Confirmatory, family LT (§9.1), with each Δ = mvc_v1 − CRT-AL:
  - LT-1: MOS against CRT-AL;
  - LT-2: SMOS against CRT-AL.
  Both are two-sided, with Holm over 2 at α 0.05.
- How the family was chosen. On 29 Oct 2026 the author decided to confirm only against CRT-AL, the H1 comparator against which the paper's main claim is made, and to keep every comparison against Seed-VC and kNN-VC descriptive. The decision was made with T1 known (§0.2 item 1), before any item was drawn and before any rating existed.
  - The earlier draft chose by a rule: a comparison of mvc_v1 was confirmatory when T1 predicted a listener difference at least the size of the design's minimum detectable effect (MDE), in either direction. By that rule the draft also made SMOS against Seed-VC and SMOS against kNN-VC confirmatory (its LT-3 and LT-4, Holm over 4), with the baselines expected to win (§9.3). The author chose not to confirm them.
  - As a result, both hypotheses of LT are comparisons that T1 predicts mvc_v1 to win. Similarity against the baselines, where T1 predicts the baselines to win and family S has already rejected on SIM in Seed-VC's favour, is not tested. The paper says so, and reports those comparisons beside LT-2 whatever their result (§9.3, §15).
- T1's predictions for LT, against an MDE of 0.44 under design values A and 0.61 under B at 20 raters (§10):
  - LT-1: the UTMOS difference on the eligible pairs is 1.19 / 1.30.
  - LT-2: ECAPA coverage rises from 0.070 (CRT-AL) to 0.522 (mvc_v1). If SMOS moves in proportion over a 2–3 point span from source to natural target, the difference is 0.90–1.36 points on VCTK-2k. Scaled to LTC-2k by the ECAPA SIM gaps (+0.176 against +0.295), it is 0.54–0.81 there, about 0.72–1.08 pooled.
- Descriptive (CIs only: no p-value and no Holm decision):
  - SMOS against Seed-VC and kNN-VC, the draft's LT-3 and LT-4. T1 predicts both baselines to be ahead, by about 0.60–0.90 and 0.37–0.55 SMOS points pooled (§9.3).
  - MOS against Seed-VC and kNN-VC. The T1 UTMOS differences to these baselines are −0.05 to +0.19, far below the MDE, and the 95 % CI half-width at 20 raters is ±0.28 (A) or ±0.39 (B). No non-inferiority margin is set now, with T1 known, so these comparisons never support "comparable". They are reported like the SMOS comparisons against the baselines (§15).
  - Everything else (§9.3).
- The listening results never change the outcome or the wording of the plan's §7 (H1, NL, L, S, NI). §15 fixes how they are reported.

## 2. Conditions

### 2.1 In the test

| Code | System | Audio | Blocks |
|---|---|---|---|
| `nat` | Natural target speech | VCTK-2k: the target's parallel recording P (the source's sentence); otherwise an E utterance of the target (§3.2) | MOS (gold items), SMOS (upper anchor) |
| `mvc_v1` | MimiFlow-VC, the frozen mvc_v1 | T1 output at 30 s, `$MVC/test/conv/<set>/30/mvc_v1/`; rows `MR/m/t1` and `MR/m/t2`, `metrics/<set>_30.jsonl` | MOS, SMOS |
| `crtal` | CRT-AL (`cand_pm18_rq`) | Journal output at 30 s; rows `V1/m/t1` and `V1/m/t2`, `metrics/<set>_30.jsonl` | MOS, SMOS |
| `seedvc` | Seed-VC v1 small; sees only the first 25 s of R | Journal T4 output at 30 s; rows `V1/m/t4/metrics/<set>_30.jsonl` | MOS, SMOS |
| `knnvc` | kNN-VC | Journal T4 output at 30 s; same rows | MOS, SMOS |
| `src` | Natural source speech | The pair's source utterance (`source_path`) | SMOS only (lower anchor; gold items on cross-gender pairs) |

### 2.2 Why these, and what is left out

- CRT-AL is the H1 comparator, and the paper's main claim is made against it.
- Seed-VC is the strongest baseline on SIM, and on LTC-2k UTMOS (4.07). It is in families NI and S.
- kNN-VC is the classic any-to-any baseline. It is in families NI, L and S.
- The listening comparisons against Seed-VC and kNN-VC are descriptive (§1, §9.3).
- `nat` provides the gold items of the screening and the top of both scales.
- `src` is used in SMOS only. It is the bottom of the similarity scale, which the perceptual coverage of §9.3 needs (mirroring the ECAPA coverage of plan §3.6). Its cross-gender trials are the SMOS gold items of §8. It costs 16 trials per rater, about 5.5 min, and stays in the test whatever the pilot shows (Decisions 3 and 16, §6.3).
- FreeVC is out. It was trained on VCTK ("VCTK-seen", plan §5.1), and no claim against it on VCTK-2k is made. Keeping it on LTC-2k alone would unbalance the item design.
- FACodec is out (Decision 4):
  - it would add 32 trials per rater, about 10 min;
  - it has the lowest LTC-2k UTMOS of the baselines (3.58) and the largest natural-level leakage (+0.095 / +0.096);
  - of the plan's families, it is only in S.
- Also out:
  - `mvc_v1_q32`: family Q found UTMOS +0.024 / +0.014 against mvc_v1;
  - the Mimi copies: these codec ceilings are already measured;
  - X-mvc-1, which is exploratory;
  - FAISS, which belongs to the journal's own test.

## 3. Items

### 3.1 Draw

- There are 32 items: 16 from VCTK-2k and 16 from LTC-2k, 4 per gender combination (F2F, F2M, M2F, M2M) per set.
- The lists are the T1 pair lists:
  - `V1/pairs/vctk2k.jsonl`, the selection record's 5-target variant, sha256 bdbf36a32b61eff625e2acaf40163835582ca9e391079c11ca72e6bf51601c1d;
  - `V1/pairs/ltc2k.jsonl`, sha256 32fc920516f0b6e4d239ddc85994cca8f746538763e5ad71ab2e208203c56e45.
- Order: per set and gender combination, the pair_ids are sorted, then shuffled with `random.Random('20260928:mvc_v1:listening:<set>:<g>')`. This is design seed `"20260928:mvc_v1"` in the builder's `'<seed>:listening:<set>:<g>'`. In VCTK-2k, the pairs with P come first, each group kept in that order (the P preference).
- The cells are filled in a fixed order: vctk2k F2F, F2M, M2F, M2M, then ltc2k in the same order. A cell's items are the first 4 pairs of its order that are eligible (§3.2) and pass the rules of §3.3, given the items already taken in the set. They get ranks 0–3 in the order taken. The rest of each cell's order is its queue.
- The rules are written in `mvc/mos/design_mvc_v1.json`, which `build_stimuli.py draw` reads. That file is frozen with this text (§13).
- The draw runs once, after the freeze, with the builder at the tag. On the real pair lists, the builder refuses to draw unless the freeze of §13 holds.
- The draw writes `items.json`. It holds the orders, the items, the practice items, every pair passed over with its reason, and the sha256 of every input (design, pair lists, rows).
  - It names the natural clips (§11.4), so it is kept in `$D/private/` until collection closes and is committed with the results.
  - Its sha256 goes into the build manifest (§4.3, §12).
- `build` draws again from the frozen design, the pair lists and the rows, and refuses unless the result equals `items.json`.

### 3.2 Eligibility

A pair is eligible when all of the following hold. The draw reads only fields of the pair lists and whether each condition's T1 row exists without an error. No metric value and no audio enters it.

1. It is not one of the 12 pairs of §3.3 (`exclude_pairs`).
2. Its source lasts 2.0–10.0 s (`source_dur`).
3. It has a natural target clip:
   - where the pair has P: P itself, provided P lasts 2.0–10.0 s; otherwise the pair is not eligible;
   - where it has none: the target's E utterance of 2.0–10.0 s whose duration is closest to the source's, ties broken by path.
   The natural clip is never cut and is never in R at any budget.
4. It has an SMOS reference: another E utterance of the target, not in R, of 3.0–10.0 s and closest to 5.0 s (ties: path). If there is none, the shortest E utterance longer than 10.0 s is used, cut to 10.0 s (§4.2). If there is no such utterance either, the pair is not eligible.
5. Its source utterance (`src`) is neither the natural clip, nor the reference, nor in R.
6. Every converted condition of §2.1 has a T1 row at 30 s with no error and a WAV path. The row must agree with the pair list on genders, target speaker, E, P and source, or the draw stops. All rows exist, since T1 and job M had 0 failures.

Counts from the pair lists, made with the builder's own rules:

| Set | Eligible pairs | Pairs with P | Per cell | Target speakers |
|---|---|---|---|---|
| VCTK-2k | 1,875 of 2,000 | 919 (227–234 per cell, all 24 shared sentences) | 457–484 | 20 of 20 |
| LTC-2k | 1,549 of 1,800 | none | 350–424 | 32 of 36 (17 F, 15 M) |

- VCTK-2k pairs left out: 70 for source duration, 6 for P duration, 43 for no SMOS reference, 6 shown.
- LTC-2k pairs left out: 201 for no natural clip of 2–10 s, 44 for no SMOS reference, 6 shown.
- The reference fallback (a cut reference) applies to 391 of the 1,549 eligible LTC-2k pairs (25.2 %) and to no VCTK-2k pair. In the 300 throwaway draws, 4.4 of the 16 LTC-2k items had a cut reference on average (2–8).

### 3.3 Speaker and sentence rules; excluded pairs

- Within a set:
  - a source speaker appears in at most one item;
  - a target speaker appears in at most one item;
  - a source sentence (`source_text`, case and punctuation ignored) appears in at most one item.
- Consequences:
  - VCTK-2k has 16 of its 20 source speakers, 16 of its 20 target speakers and 16 different sentences. Its source and target speakers are disjoint sets.
  - LTC-2k has 16 sources and 16 targets out of 36 speakers, each of whom is both a source and a target in the pair list. There, a speaker can be the source of one item and the target of another. That happened in 7.0 of the 16 items per throwaway draw on average (3–11) and is reported.
- Why:
  - items are then nearly independent in speaker and content terms, which the item bootstrap of §9 assumes;
  - SMOS covers 16 target voices per set;
  - no rater hears one sentence in 10 or more MOS trials.
  Without the sentence rule, every one of the 300 throwaway draws had VCTK-2k items sharing a sentence: 4.4 of 16 on average, and up to 8.
- Feasibility in the 300 throwaway draws:
  - the rules filled every cell every time;
  - all 16 VCTK-2k items had P in 282 draws; in 18, one M2M item had none, and its natural clip was an E utterance chosen by rule 3 of §3.2;
  - pairs passed over per draw: 125 in VCTK-2k on average (28–322), mostly by the rules of this section; 17 in LTC-2k (5–44).
- The 12 pairs of `mvc_listen_test.html` are not eligible:
  - VCTK-2k: `vctk2k__p253_007__p284`, `vctk2k__p262_009__p261`, `vctk2k__p293_177__p299`, `vctk2k__p361_007__p257`, `vctk2k__p374_011__p281`, `vctk2k__p374_012__p299`;
  - LTC-2k: `ltc2k__1284_1180_000043_000000__5105`, `ltc2k__4077_13751_000023_000001__7127`, `ltc2k__5142_36377_000013_000003__5105`, `ltc2k__7176_92135_000085_000002__7729`, `ltc2k__8230_279154_000028_000003__4992`, `ltc2k__8463_294825_000015_000002__6829`.
  These were checked against the page's speaker pairs and sentences.

### 3.4 Replacement

- If a WAV of an item cannot be fetched or decoded at build time, the build stops and leaves nothing (§4.3).
- A transient cause is fixed and the build is run again.
- A file lost for good is a deviation (§14), decided before any rating exists:
  1. the pair goes into `exclude_pairs` of a revised design file;
  2. the revised file is frozen like the first (commit, the annotated tag `listening-addendum-mvc-v1-r1`, push; the builder's `TAG` is changed to match);
  3. the draw runs again.
  The lost item's rank then goes to the next pair of its queue that passes §3.2–§3.3. Every item that differs from the first draw is reported.
- No replacement is expected, because every T1 row has a WAV.

### 3.5 Practice, attention, and the record of the draw

- Practice is in the MOS block only, as in the journal: 3 trials, never analysed.
  - Each practice pair is the first pair of its cell's queue after the cell's last item that is eligible for the condition it is played as, and whose target speaker no item and no earlier practice item of the set has. So no practice clip is a gold clip or comes from an item's target voice.
  - The three: vctk2k F2F played as `nat`; ltc2k M2M played as `crtal`; vctk2k F2M played as `seedvc`.
  - In the 300 throwaway draws, all three were always found.
- Attention trials use the natural target clips of rank-1 items, under ids of their own and with their own padding (§4.2):
  - MOS, 2 trials: vctk2k F2M asks for 2, ltc2k M2F asks for 4;
  - SMOS, 2 trials, each with its item's reference: ltc2k F2F asks for 4, vctk2k M2M asks for 2.
  The instruction text is in §5.3.
- The design file's sha256, filled in on 30 Oct 2026 and checked at the freeze (§13 step 2): 6a1cb8fa5b095fa387a6fa32567f1258aaaa577216bc909b362d283b734a13b5. The sha256 of `items.json` goes into the build manifest (§4.3).

## 4. Stimuli

### 4.1 Audio sources

- Converted audio is the WAV named by the `wav` field of each T1 row in §2.1.
- Natural audio (P, E and source) comes from the paths in the pair lists. The journal's paths under `/datastore/nvquoc/crt/` moved to `/datastore/ceai/crt/` (DEVIATIONS_v1 item 4) and are mapped there.
- Everything is copied read-only from ceai by rsync over ssh into `$D/private/cache/`, as `mvc/analysis/dnsmos_t1.py` does. Nothing is written on ceai. The cache paths name the systems, so the cache is kept like the key (§11.5).
- The reference budget is 30 s (R[30]), the plan's primary budget.
  - Raters never hear R.
  - The SMOS reference is an E utterance that is in no system's conversion reference, which is checked at every budget.
  - FreeVC saw VCTK in training, and Seed-VC's training data have not been checked, so "no system heard it" is not claimed.

### 4.2 Processing (the journal's `benchmark/listening_stimuli.py`, plus padding)

- Loading: each clip is loaded as mono and resampled to 24 kHz with soxr HQ, librosa's default. Each system's own bandwidth is kept: no low-pass filter, no denoising and no silence trimming. The key records the native sample rate, the channel count and the sha256 of every source file.
- Cutting: only fallback SMOS references are cut, to 10.0 s with a 50 ms linear fade-out. Nothing else is cut.
- Loudness: the active-speech RMS is set to −26 dBFS, measured over the 20 ms frames within 40 dB of the loudest frame. The gain is lowered where needed so that the peak stays at −1 dBFS. This approximates the active speech level of ITU-T P.56.
- Padding:
  - Every clip gets 0–250 ms of digital silence at its end: whole milliseconds, uniform, drawn with Python `secrets`, and recorded in the key.
  - Copies of one clip (the attention trials) get different pads, so their files differ from the gold clip's.
  - The reason: on the T1 rows, output length minus source length is nearly constant per system. It is 0.000 s for mvc_v1 (exactly the source length), about −0.02 s for kNN-VC, 0 or +0.040 s for CRT-AL, −0.003 to −0.009 s for Seed-VC and +0.005 to +0.010 s for FACodec. The clips of one item share one SMOS reference id, so their lengths alone would tell the systems apart.
  - Silent frames do not count in the active level, so the padding leaves the level unchanged.
- Durations: the durations the browser receives (`lt_data.json` and the server's plan) are rounded to 0.1 s. A browser can still measure a file's exact length, but after the padding a difference in length says nothing about the system.
- Format: clips are written as FLAC (16-bit, 24 kHz, mono, no tags beyond the encoder's). Each is stored as `stimuli/<id>.flac` under a random 48-bit id (Python `secrets`) that says nothing about the clip. Trial and stimulus lists are written in random order.
- The key: for every clip and trial, it records condition, pair, role, trial kind, source path, native rate and channels, source sha256, gain, cut, pad, duration and sha256. It is written with mode 0600 to `$D/private/key.json`, never inside the package, the repository or any served folder.

### 4.3 Package and manifest

- The package is `$D/package/lt_data.json` (the journal's LT_DATA version 1, without CMOS) and `$D/package/stimuli/`. It is written through a temporary folder that is renamed on success, so a failed build leaves no partial package and no key.
- Each rater's browser receives only the trial list of that rater's half and the clips it uses.
- The package is built once. Ids are random, so a rebuild changes every id and breaks every running session. A rebuild after the first rating is a deviation (§14), with a new package id, reported with the raters it affected.
- The build writes `mvc/LISTENING_BUILD_mvc_v1.json`, which is committed (§12 step 6). It holds:
  - the sorted list of clip sha256 values;
  - the sha256 of the key, `lt_data.json`, `items.json` and the design;
  - the frozen files and the tag's commit;
  - the trial counts.
  It holds no condition mapping and no id.

## 5. Scales and wording

### 5.1 MOS (naturalness)

- The scale is the 5-point ACR of ITU-T P.800 and P.808, asked as naturalness, as on the journal page.
- EN: "How natural does the speech in this recording sound?" 5 Excellent, 4 Good, 3 Fair, 2 Poor, 1 Bad.
- VI: "Giọng nói trong đoạn ghi âm này nghe tự nhiên đến mức nào?" 5 Xuất sắc, 4 Tốt, 3 Trung bình, 2 Kém, 1 Rất kém.
- Instruction, EN: "For each recording, rate how natural the speech sounds, that is, how much it sounds like a real person speaking. Ignore what is being said."
- Instruction, VI: "Với mỗi đoạn, hãy chấm mức độ tự nhiên của giọng nói, tức là nghe giống một người thật đang nói đến mức nào. Không để ý đến nội dung được nói."
- The test follows P.808 in its scale, its gold and trapping questions and its listening-device check. It is not claimed to conform to P.808 as a whole.

### 5.2 SMOS (similarity to the target speaker)

- The scale has 5 points. It adapts the same/different judgement with a confidence level of the Voice Conversion Challenges, which, as far as is known, used 4 points with no midpoint. This test adds "Not sure" as the midpoint, as on the journal page. The paper calls it "a 5-point adaptation of the VCC same/different scale".
- EN: "How similar is the voice in the Sample to the voice in the Reference? Ignore the words and the audio quality." 5 Definitely the same speaker, 4 Probably the same, 3 Not sure, 2 Probably different, 1 Definitely different speakers.
- VI: "Giọng trong đoạn Mẫu giống giọng trong đoạn Tham chiếu đến mức nào? Không để ý đến nội dung lời nói và chất lượng âm thanh." 5 Chắc chắn cùng một người nói, 4 Có lẽ cùng một người, 3 Không chắc, 2 Có lẽ là hai người khác nhau, 1 Chắc chắn là hai người khác nhau.
- An earlier draft of the page labelled 2 as "Có lẽ khác người". In Vietnamese that can also read as "probably an odd person", so this addendum fixes the longer wording, which the page now uses.
- The Reference and the Sample must both be played to the end before the rating buttons work.

### 5.3 Attention instruction

- EN: "To show you are paying attention, please select {r} for this sample." (MOS) and "… for this pair." (SMOS).
- VI: "Để cho thấy bạn đang chú ý, vui lòng chọn {r} cho đoạn ghi âm này." and "… cho cặp này."
- {r} is the number and its label in the page language. The page builds the text from the trial's `attention` value, and `lt_data.json` carries no instruction text.

### 5.4 Language

- The page is in Vietnamese by default, with an English switch on every screen.
- The stored answer is the number, so it does not depend on the language. The language in use at each rating is stored (§11.3).
- The recordings are in English.

## 6. Session

### 6.1 Order (the same for every rater)

1. Access code, then consent. The rater types in an invite code, or, without one, starts a new session and receives an anonymous code from the site, shown once (§7.3, §11.2).
2. Sound check: the rater sets the volume, then takes the headphone check of Woods et al. (2017), as on the journal page.
   - There are 6 trials of three 200 Hz tones, and the rater chooses the quietest.
   - The check is passed at 5 or more correct, with at most 2 attempts.
   - A rater who fails twice is thanked, and the session stops.
3. MOS block: instructions, 3 practice trials, then 82 trials (80 main and 2 attention). A break screen comes after trial 41.
4. A break screen.
5. SMOS block: instructions, then 98 trials (96 main and 2 attention). A break screen comes after trial 49.
6. The background questions (§7.4).
7. A thank-you screen.

Rules within the session:

- Every clip must play to the end before the rating buttons work, and the server refuses a rating of a trial whose clips were not played. Replays are allowed and counted.
- The main trials of each block are shuffled per rater (mulberry32 seeded with FNV-1a of `<rater_id>:<block>`, as in `lt_core.js`).
- No item appears on two consecutive main trials (Decision 14).
- Attention trials never come in the first 3 positions and are never next to each other.
- MOS always comes before SMOS, so that speaker identity is not primed before naturalness is rated.

### 6.2 Halves

- Each rater rates 16 of the 32 items in every condition.
  - Half A is ranks 0 and 2 of every cell; half B is ranks 1 and 3.
  - Each half thus holds 8 VCTK-2k and 8 LTC-2k items, 2 per gender combination and set.
  - The 4 attention trials go to every rater.
- The half is set when the access code is issued and is stored with it (§7.1, §11.2). It is not derived from the code.
- Per rater: 3 practice + 82 MOS + 98 SMOS = 183 ratings.
- The halves are fixed now because a full session (32 items) is estimated at 95 min, or 106 min with `src`, against 52 or 58 min for one half. The estimate assumes:
  - the mean durations of the drawn items over the 300 throwaway draws: source 5.99 / 6.09 s, natural clip 5.55 / 5.47 s, reference 4.23 / 6.56 s (VCTK-2k / LTC-2k, cut references counted at 10 s), plus 0.125 s of padding;
  - 1.2 plays per MOS clip and 1.3 per SMOS clip;
  - 3 s to answer a MOS trial and 4 s an SMOS trial, with 1.5 s between trials;
  - 8 min for consent, sound check, practice, breaks and questions.

### 6.3 Pilot

- 3 pilot raters, who are not main raters, each do a full half-session in one sitting on the built package. The pilot comes after UIT's approval or exemption is in [ETHICS_NOTE] and before invitations go out, while open registration is off (§12 step 7).
  - The pilot raters get invite codes with the note "pilot". These 3 codes come on top of the 40 of §7.1: the admin sets the server's code cap to 43 before issuing them.
  - After the pilot, their results are exported with "completed sessions only" unticked (§11.4), and that export folder is moved from `$D/export/` to `$D/pilot/`, because deleting a rater removes their files from every folder under `$D/export/` (§11.5). The admin's raters list (§11.4), the body of `GET /api/admin/raters`, is saved in that folder as `raters.json` before the pilot raters are deleted.
  - The pilot raters are then deleted from the database (§11.4) before invitations go out, so that they enter neither the counts of §7.1 nor the main export. Their rater IDs are not given out again.
- The pilot checks the site end to end and measures session length. Only session length, the sound check and technical problems are read from it. Session length is read from the raters list, from the time of consent to the time of the last rating. No rating summary is computed.
- Fixed rule: if the median pilot session is longer than 60 min, every main rater is asked to take the session in two sittings: to stop at a break screen, preferably the one between the MOS and SMOS blocks, and to continue later with the same code (§6.1, §11.2). The request goes into the announcement, the invitations and the message that carries an invite code. The consent already allows one or two sittings (§11.5), and the break screens say that the rater can close the page there and continue later with the same code, so neither changes.
- Nothing else follows from the pilot. `src` stays in SMOS, and the package, the trials, the screening and the analysis stay as they are. Whether the two-sitting request applied is logged (§13 step 6) and reported (§15).

## 7. Raters

### 7.1 Number and stopping rule

- The target is 20 screened raters who complete both blocks, 10 per half, counted over both access routes together (§7.3). §10 gives the power argument. At least 8 per half are needed for the LT decisions (below).
- Halves:
  - Every new code, whether the admin issues it or open registration does, gets the half with fewer consented raters who can still complete (disabled raters and raters who failed the sound check not counted; each earlier code of the same admin request counted as consented). On a tie, it gets the half with fewer codes (disabled and deleted codes not counted); on a further tie, half A.
  - The server applies this rule when it issues the code; the admin cannot choose the half. The rule uses those counts only, never a rating.
  - The rule balances consented raters. A half where more raters stop or are excluded by §8 rules 2–5 can therefore lag in screened, complete raters; collection then goes on until the close. This rule is accepted as the server applies it (Decision 17).
- At most 40 codes are issued over both routes together, besides the 3 pilot codes (§6.3).
  - Every code ever issued counts, disabled and deleted ones included, so an open-route start without consent uses up a code.
  - The server enforces the cap, a setting of the admin page with the default 40. Before the pilot the admin sets it to 43 (§12 step 7).
  - At the cap, the admin page creates no invite code and the page offers no new session; codes already issued still work.
  - Any other change of the cap is an operational change (§14). The server logs it, it goes into `mvc/DEVIATIONS_mvc_v1.md`, and it is decided from code counts only, never from a rating.
- Open registration is off from the saving of the consent (§12 step 7) until invitations go out, on from the start of collection (§12 step 8) until the close, and off after the close. The server's default stays on (§11.2); the procedure turns it off where this text says. The admin may turn it off earlier only for an operational reason, such as automated sign-ups. That is logged (§14) and is never decided from any rating.
- Collection closes when both halves have at least 10 screened, complete raters, counted over both routes, or on 15 Dec 2026 at 23:59 +07:00, whichever comes first. The paper is not submitted before the close (Decision 34).
  - At the close, the admin turns open registration off, disables every code without a recorded consent (§11.4), and issues no further invite code.
  - A session counts as started when its consent was recorded before the close. Started sessions may be finished until the export, 3 days after the close (the grace period; §12 step 9, §16 item 17). A session not complete at the export counts as incomplete (§8 rule 5).
  - Every screened, complete rater of either route is analysed, including any beyond 10 in a half.
- If either half has fewer than 8 screened, complete raters at the export, LT-1 and LT-2 are reported with CIs and p-values, labelled descriptive, with no Holm decision.

### 7.2 Who can rate

- Adults (18 or older) who declare normal hearing, understand spoken English, and use headphones of any type in a quiet place, on a computer or a phone (Decision 18).
- Not eligible:
  - the authors;
  - anyone who worked on MimiFlow-VC, CRT or the baseline runs;
  - anyone who has listened to a page with labelled outputs of these systems (§0.2 item 3); the recruitment message asks about this;
  - the pilot raters;
  - anyone who has already completed this test.
- Raters of the journal's listening test may take part; question 6 of §7.4 records it.
- The rules are the same on both access routes, and neither route can verify them.
  - On the invite route, the recruitment message asks about them.
  - On the open route, the announcement and the information screen state them, and the rater confirms them by agreeing (§11.5).
  - An open-route rater could start again with a new code. The information screen asks people to take the test only once (§11.5), and question 6 of §7.4 records an earlier listening test of this group (§9.4).

### 7.3 Recruitment and access routes

- A general announcement goes to UIT students and staff and to colleagues, by e-mail, through the faculty's channels and in social media posts (Decision 19). It carries the link and says who can take part (§7.2).
- Both access routes are open during collection (decided 29 Oct 2026, Decision 17):
  - invite route: a volunteer receives a personal access code, issued by the admin (§11.2), and the link by private message;
  - open route: while open registration is on and the code cap of §7.1 is not reached, anyone who opens the link can choose "Tôi chưa có mã: bắt đầu phiên mới / I have no code: start a new session". The site then issues an anonymous code, shows it once, and asks the rater to write it down, since it is needed to resume and to withdraw (§11.5). The organiser does not know who holds it.
  The link carries no code on either route.
- The route of every rater (invite or open) is recorded with the code (§11.3). Open-route raters are handled like invite-route raters: the same session, the same screening (§8), the same stopping rule (§7.1) and the same analysis (§9). The route enters the analysis only through the descriptive breakdown of §9.3.
- No payment is made and no course credit is given (Decision 19); the consent says so.
- Students in the author's current courses may take part. Taking part or not is never linked to their marks. Invitations are not sent through course announcements, and these students get no personal invite code: they take part through the open route, after the general announcement.
- Apart from raters who write to the contact, who make their e-mail address known to the contact (§11.3), the author does not learn who rated. Only rater IDs reach the analysis. When invite codes are created, the admin page shows each code with its rater ID and half; only the code goes into the person ↔ code list of the invite route, which holds names and codes, never a rater ID, and is never matched with the raters list or the export. The admin finds a rater by code only for a withdrawal that the code's holder asks for (§11.5).

### 7.4 Background questions

These are the journal's five questions, plus one more. Each also offers "Prefer not to say / Không muốn trả lời", and there is no free text:

1. age group;
2. first language;
3. understanding of spoken English;
4. known hearing problem (health information, §11.5);
5. listening device;
6. new: "Have you taken part in a listening test of this research group before?" (Yes / No / Not sure).

## 8. Screening (fixed now, applied by script before any analysis)

The rules are the same on both access routes. A rater is excluded who:

1. did not pass the headphone check (the session stops there);
2. rated natural target speech ≤ 2 on more than 2 of their 16 gold MOS trials (kind `gold`);
3. rated natural source speech ≥ 4 on more than 2 of their 8 cross-gender `src` SMOS trials (kind `gold_src`);
4. gave any answer other than the one asked on any of the 4 attention trials;
5. did not complete both blocks. Such raters are counted but not analysed.

Consequences and limits:

- No constant rater survives. The MOS attention trials ask for 2 and 4, and the SMOS ones ask for 4 and 2, so any single repeated rating in a block fails rule 4.
- The screening script reads the key only to find the gold, `gold_src` and attention trials and their expected values.
- During collection, the script of §9.5, once its sha256 is committed (§12 step 8), is run on completed raters to apply the stopping rule (§7.1). It then prints only "kept" or "excluded" with the reasons. The counts of kept raters per half that the stopping rule uses (§7.1) include both routes. No condition mean or difference is computed before the close.
- Rules 2 and 3 remove raters who rate natural speech low or the source high. That favours the anchors, and the paper says so.

## 9. Analysis

### 9.1 Confirmatory family LT

- For a comparator c:
  - d_ri(c) = rating(mvc_v1) − rating(c), for rater r on item i, in the block of the hypothesis;
  - m_i(c) = the mean of d_ri(c) over the screened raters of item i's half;
  - Δ(c) = (1/32) Σ_i m_i(c).
  Every item weighs 1/32, however many raters each half ends with. Positive values favour mvc_v1. Values are pooled over the two sets.
  - LT-1: Δ_MOS(crtal);
  - LT-2: Δ_SMOS(crtal).
- Each test is two-sided at α 0.05, with Holm over 2 (`stats.holm`). The p-value is `stats.p_two_sided` over the bootstrap replicates of Δ, p = min(1, 2 min(1 + #{Δ* ≤ 0}, 1 + #{Δ* ≥ 0}) / (B + 1)), as in plan §5.2.
- Bootstrap: B = 10,000, seed 20260928, crossed and stratified. In each replicate:
  - raters are drawn with replacement within each half, giving counts w_r;
  - items are drawn with replacement within each (set, half) stratum of 8, giving counts w_i;
  - m_i* = Σ_r w_r d_ri / Σ_r w_r, over the drawn raters of i's half;
  - Δ* = (1/32) Σ_i w_i m_i*.
  CIs are 95 % percentile intervals. The SE is the standard deviation of the replicates. Every call starts from the seed.
- Only complete, screened raters enter, from both access routes alike. The page does not advance without a rating, so no rating is missing.
- The same estimand and bootstrap give the CIs of the descriptive comparisons of §9.3, among them SMOS against Seed-VC and kNN-VC. These get no Holm decision.

### 9.2 Secondary

- For LT-1 and LT-2, a linear mixed model of the paired differences: d_ri ~ 1 + (1 | rater) + (1 | item), fitted by REML. This is the journal's `listening_analysis.mixed(paired_diff(…))` through `stats.variance_components` (statsmodels MixedLM, variance components).
- The intercept and its Wald 95 % CI are reported beside each LT result. They decide nothing. A model that does not converge is reported as not estimable.
- The model holds the rater × condition and item × condition variance that §10 assumes, because it is fitted to the differences rather than to the ratings.

### 9.3 Descriptive analyses (CIs from the same bootstrap)

- MOS and SMOS of every condition, pooled and per set.
- Paired differences, pooled and per set, with the estimand of §9.1:
  - SMOS: mvc_v1 − seedvc and mvc_v1 − knnvc, the draft's LT-3 and LT-4, which the author chose not to confirm (§1);
  - MOS: mvc_v1 − seedvc, mvc_v1 − knnvc, mvc_v1 − nat, crtal − nat;
  - SMOS: mvc_v1 − nat, crtal − nat.
- Context for SMOS against the baselines: the draft's predictions from T1, on which no rule depends. Here the baselines are expected to win.
  - On VCTK-2k, ECAPA coverage is 0.943 (Seed-VC) and 0.834 (kNN-VC) against 0.522, which predicts 0.84–1.26 and 0.62–0.94 SMOS points.
  - If the LTC-2k differences scale with the ECAPA SIM gaps (−0.113 against −0.271, and −0.034 against −0.191), they are 0.35–0.53 and 0.11–0.17 there.
  - Pooled, that gives about 0.60–0.90 against Seed-VC and 0.37–0.55 against kNN-VC, in the baselines' favour.
  - Family S has already rejected on SIM in Seed-VC's favour (S-seedvc-3, S-seedvc-4).
  - At 20 raters, the 95 % CI half-width is ±0.28 (A) or ±0.39 (B) (§10).
- SMOS per gender stratum (same, cross) for every converted condition, and mvc_v1 minus each comparator per stratum.
- Perceptual coverage of each converted system X: C_X = (SMOS_X − SMOS_src) / (SMOS_nat − SMOS_src).
  - The ratio is formed in every replicate, pooled, per set and per stratum.
  - It is shown beside the ECAPA coverage of the same 16 VCTK-2k items (plan §3.6 formula) and of the 980 P pairs: mvc_v1 0.522, CRT-AL 0.070, kNN-VC 0.834, Seed-VC 0.943.
- Agreement with the objective scores, over the 32 × 4 (item, converted condition) cells:
  - Spearman's ρ of mean MOS against UTMOS and against DNSMOS P808;
  - Spearman's ρ of mean SMOS against WavLM-SV SIM and against ECAPA SIM;
  - Kendall's τ of the system means.
- Also reported:
  - natural speech per set (resampled VCTK P against LibriTTS-R E);
  - the items with a cut reference, the items without P, and the LTC-2k speakers who are a source in one item and a target in another;
  - session length and plays per trial;
  - the screening table and the background counts.
- By access route (invite, open):
  - per half, the codes issued and used (from the admin's raters list saved with the export as `raters.json`, §12 step 9; deleted raters, such as the pilot raters and withdrawals, are not in it, so the total of codes issued, deleted ones included, is read from the admin status saved beside it as `status.json`), and the raters who consented, passed the headphone check, completed both blocks, and were kept or excluded, with the reasons of §8;
  - the background counts and session length;
  - where a route has screened, complete raters in both halves, LT-1 and LT-2 on that route's raters alone, with the estimand and bootstrap of §9.1 (CIs only).
  The breakdown decides nothing.

### 9.4 Sensitivity analyses (each with the LT decisions that change)

- The unstratified crossed bootstrap (`listening_analysis.crossed`), and the mean over all (rater, item) ratings instead of equal item weights.
- Without raters who answered Yes or Not sure on hearing, Basic on English, or Other on the listening device.
- Without raters who answered Yes to question 6.
- With screening rules 2–4 ignored, keeping every rater who passed the headphone check and completed both blocks.
- LT-1 and LT-2 per set, on the 16 VCTK-2k and the 16 LTC-2k items separately (CIs).

### 9.5 Implementation

The analysis script lives in `mvc/mos/` and runs in Python with `.venv-stats`. It copies from `benchmark/listening_analysis.py` (sha256 faa564e727113c1731c19828d7515b9725335664f40ceacb786df26a2aa2d577) the functions it reuses: `load_results`, `screen`, `ratings`, `crossed`, `summarise`, `paired_diff`, `mixed`, `session_minutes` and `background`. These changes are needed:

1. Remove CMOS: `cmos_values`, P8, K8, the hidden-pair rule, `CODEC_PAIRS` and `cmos.json`.
2. Screen as in §8: gold on more than 2 of 16, the `gold_src` rule, the 4 attention values read from the key, and incomplete raters. A rater is complete when every trial of the session has an answer (the export's `finished` is set); the export's `completed` field, which a failed sound check also sets, is not used for this.
3. `ratings()` counts kind `gold_src` as a rating of condition `src`, besides `main` and `gold`.
4. Add the estimand and the stratified bootstrap of §9.1, and keep `crossed()` for §9.4.
5. Add family LT: LT-1 and LT-2 with `stats.holm` (Holm over 2), and the 8-per-half rule of §7.1.
6. Fit the mixed model on the paired differences, and add the Wald SE and CI of the intercept, which `stats.variance_components` does not return.
7. Add the perceptual coverage, the per-stratum tables, the agreement table and the breakdown by access route. These read the T1 rows and the DNSMOS and ECAPA rows, with their sha256, and, for the codes issued and used per route, the raters list and the status saved with the export (`raters.json`, `status.json`).
8. Limit pilot mode to session length, the sound check and technical events, read from `$D/pilot/` (for the report; the pilot rule itself uses the raters list, §6.3).
9. Read the website's export: the journal's results-file schema (version 1) plus the fields of §11.3, among them the access route, keyed by rater IDs R001, R002, …

Further:

- The script imports `benchmark/stats.py` unchanged and checks its sha256 at start (a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6).
- Its analysis runs once, after the close, and writes `mvc/LISTENING_RESULTS_mvc_v1.{md,json}` with the sha256 of every input: the export, the raters list, the status, the key, `items.json`, the manifest and the rows.
- Its pytest tests, in `mvc/mos/tests/`, cover at least:
  - every screening rule at and next to its threshold;
  - the estimand and the stratified bootstrap against a direct computation on a small design with unequal halves;
  - Holm over 2 and the 8-per-half rule;
  - the breakdown by route, including a route with raters in one half only;
  - the coverage ratio;
  - refusal of a duplicate rater, of a rater with trials from both halves, and of a trial id that is not in the key.
- The script is written, tested and checked by an independent reviewer against this addendum after the freeze (Decision 5). It is not in the freeze file (§13 step 3); the rules it applies are those of this frozen text. The script and its tests are committed, staged by name on the author's instruction, and the script's sha256 is logged in `mvc/DEVIATIONS_mvc_v1.md` before the script first reads any answer of a real rater, the pilot export in `$D/pilot/` and the screening runs of §8 included, and in any case before the close (§12 steps 8 and 9). Any later change of it is a deviation (§14).

## 10. Power and rater count

The model. For two conditions rated by the same raters on the same items, the rater and item main effects cancel, and for the estimand of §9.1

Var(Δ̂) ≈ 2σ²_rc / n_r + V_i / 32 + 2σ²_e / (16 n_r),

where:
- σ_rc is the SD of the rater × condition effects;
- V_i is the variance over items of the true per-item difference;
- σ_e is the residual SD of one rating;
- n_r is the number of screened raters (n_r / 2 per half), 32 the number of items and 16 the items per rater.

Design values:
- From T1, the per-pair SD of the UTMOS difference over the eligible pairs (VCTK-2k / LTC-2k) is 0.37 / 0.41 against CRT-AL, 0.35 / 0.33 against Seed-VC and 0.34 / 0.30 against kNN-VC.
- Scenario A uses √V_i = 0.40. Scenario B uses 0.60, because listeners may spread items more than a predictor does.
- No listening data exist for σ_rc and σ_e. A sets them to 0.25 and 0.80, B to 0.40 and 1.00. These are assumptions, not estimates.
- The test uses the stratified bootstrap of §9.1, which is conservative for a crossed design. Its SE was therefore simulated: 300 data sets per row from the model, each with 400 bootstrap replicates. The model SE and the simulated SD of the estimator are shown beside it.
- MDE = multiplier × bootstrap SE. For Holm over 2, two-sided, at 80 % power, the multiplier is z_{1−0.05/4} + z_{0.80} = 2.24 + 0.84 = 3.08. This uses the level of Holm's first step, α/2, so it is conservative for each of the two hypotheses. Under the draft's Holm over 4 it was z_{1−0.05/8} + z_{0.80} = 2.50 + 0.84 = 3.34, an MDE of 0.47 (A) or 0.66 (B) at 20 raters.
- Once one hypothesis of LT is rejected, Holm tests the other at α 0.05, where the multiplier is z_{0.975} + z_{0.80} = 1.96 + 0.84 = 2.80 and the MDE at 20 raters is 0.40 (A) or 0.55 (B).

| Screened raters (per half) | SE model, A | SD simulated, A | SE bootstrap, A | MDE, A | 95 % CI half-width, A | SE model, B | SD simulated, B | SE bootstrap, B | MDE, B | 95 % CI half-width, B |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 (8) | 0.133 | 0.130 | 0.152 | 0.47 | 0.30 | 0.198 | 0.198 | 0.215 | 0.66 | 0.42 |
| 20 (10) | 0.123 | 0.129 | 0.142 | 0.44 | 0.28 | 0.183 | 0.195 | 0.198 | 0.61 | 0.39 |
| 24 (12) | 0.116 | 0.116 | 0.133 | 0.41 | 0.26 | 0.173 | 0.175 | 0.187 | 0.58 | 0.37 |
| 30 (15) | 0.109 | 0.120 | 0.123 | 0.38 | 0.24 | 0.162 | 0.165 | 0.175 | 0.54 | 0.34 |
| 40 (20) | 0.101 | 0.103 | 0.111 | 0.34 | 0.22 | 0.150 | 0.144 | 0.159 | 0.49 | 0.31 |
| item floor (n_r → ∞) | 0.071 | – | – | – | – | 0.106 | – | – | – | – |

The simulated SDs carry a Monte Carlo error of a few per cent. The MDE columns are for Holm over 2; the other columns do not depend on the family.

- LT-1: on the eligible pairs, CRT-AL trails mvc_v1 by 1.19 / 1.30 UTMOS and by 0.51 / 0.26 DNSMOS P808, and its capped WER is 16.26 / 10.33 %. With 20 raters, LT-1 has 80 % power for a true MOS difference of 0.44 (A) or 0.61 (B). That is well below the UTMOS difference; it is above the LTC-2k DNSMOS difference (0.26) under both scenarios, and above the VCTK-2k one (0.51) under B.
- LT-2: the proportional argument of §1 gives about 0.72–1.08 points pooled (0.90–1.36 on VCTK-2k), above the MDE under both scenarios.
- SMOS against Seed-VC and kNN-VC (descriptive, §9.3): the same argument gives about 0.60–0.90 and 0.37–0.55 points pooled, in the baselines' favour. The 95 % CI half-width at 20 raters is ±0.28 (A) or ±0.39 (B). These comparisons are descriptive by the author's decision (§1), not for lack of power. No rule relies on these predictions.
- Why 20 raters: beyond 20 the item term dominates, and at 40 raters the MDE falls only to 0.34 (A) or 0.49 (B). With 20, recruitment is feasible at UIT, and every item gets 10 ratings per condition. The rater count was set under Holm over 4 and is kept under Holm over 2.
- MOS against Seed-VC and kNN-VC: a CI of ±0.28 (A) or ±0.39 (B) cannot resolve T1 UTMOS differences of −0.05 to +0.19, so those comparisons stay descriptive (§1).
- The pilot does not change the rater count. All figures here are design values, and the realised SEs are reported.

## 11. Website and data handling

The server and page are reviewed against this section before the freeze (§12 step 2).

### 11.1 Hosting

- The server is written in Bun and TypeScript in `mvc/mos/`, with SQLite through `bun:sqlite`. The rater page is in `mvc/mos/public/`, adapted from `benchmark/listening/page/`.
- It runs on the author's Mac and listens on 127.0.0.1 only. A Cloudflare tunnel publishes it at https://mos.betaschool.edu.vn, and Cloudflare terminates TLS. The domain is not a UIT domain; it is accepted, and the consent says so and names the research team as the operator (Decision 27). The server answers only requests addressed to that host name or to a loopback address, and refuses a POST from another origin.
- The server and the tunnel (cloudflared) run in Terminal tabs on the Mac; no launchd job is used. The Mac stays on and awake during collection, and a server or tunnel that stops is restarted by hand (an operational change, §14). The tunnel and DNS are set up outside this addendum.
- No third-party script, font, analytics or tracker is used. Every file comes from the Mac.
- The runtime data live in `$D`, outside the repository:
  - `mos.sqlite`;
  - `package/` (`lt_data.json` and the stimuli);
  - `private/` (`key.json`, `items.json` and the audio cache `cache/`);
  - the filled consent text `consent_v1.html`;
  - the admin token `admin_token` and the server secret `server_secret` of §11.2, both created by the server with mode 600;
  - `export/` (the final export with the admin's raters list and status in it as `raters.json` and `status.json`, §12 step 9), `pilot/` (§6.3) and `backup/`.
- The data directory is owner-only (mode 700), and the server writes its files with mode 600.

### 11.2 Access

- A rater gets a code by one of two routes, both on during collection (decided 29 Oct 2026, Decision 17):
  - invite route: the organiser creates invite codes on the admin page; the server sets each code's half by the rule of §7.1.
  - open route (open registration): a rater without a code chooses to start a new session, and the server issues a fresh anonymous code at once, with its half set by the rule of §7.1, and shows it to the rater once. The admin turns open registration on and off (§7.1). The server's default is on, also in a fresh database, and stays so (Decision 24); the procedure turns it off before the consent is saved (§12 step 7), keeps it off until invitations go out (step 8), and turns it off at the close (step 9).
  - Both routes together are held to the code cap of §7.1, which the server enforces.
- Every code, from either route, has 60 bits of randomness: 12 characters of Crockford base32 from a cryptographic generator, after an optional prefix on invite codes. Only such codes sign in. The server does not start on a database of its first version that holds codes, which had 50 bits and were stored in plain text.
- The database stores, per code:
  - an HMAC-SHA-256 of the code, keyed with a random server secret kept in `$D/server_secret` (never in `$D/private` or the package);
  - the rater ID (R001, R002, … in issue order, over both routes; an ID is never given out again);
  - the half;
  - the route (`invite` or `open`);
  - on the invite route, the admin's optional note.
- The organiser keeps the list of who received which invite code outside `$D` and outside the repository. For an open-route code no such list exists: only the rater knows it.
- The admin's optional note on an invite code (for example "pilot") never holds a name, an e-mail or other personal data.
- Codes are typed into the page and sent in a POST body, never in a URL. Logins and registrations are rate-limited, by default per 10 minutes: 100 failed logins and 1,000 logins and registrations in all per client address (an IPv4 address, or an IPv6 /64), and 30 registrations from all clients together. The address, taken from the CF-Connecting-IP header of the tunnel, is held in memory for the limit only and is never stored.
- A session cookie holds a random token (HttpOnly, Secure, SameSite=Strict) and ends with the browser session; the server keeps only the token's hash, valid for at most 24 h, and ends every sign-in of a rater at the end of the session. The language choice is kept in the browser.
- With the same code, a rater resumes at the first unanswered trial on any device, on either route.

### 11.3 What is stored

- Per rater:
  - the code's HMAC, the rater ID, the half, the access route and, on the invite route, the admin's note (§11.2);
  - the times the code was issued, first used and last used;
  - the time of consent, the consent language and the sha256 of the consent text shown;
  - the sound-check trials and their result;
  - every response: trial id, stimulus ids, rating, plays per clip, listening time, response time, time of answer and page language;
  - block times, and session start and end;
  - the six background answers;
  - the browser family and major version.
- While a rater is signed in, the hash of the sign-in token and its expiry.
- Not stored: name, e-mail, IP address, the full user-agent string, location.
- The server keeps no access log with addresses or codes. cloudflared runs at its default log level (info), at which its log records no client addresses; the operator checked this on the running tunnel's Terminal tab on 30 Oct 2026 (Decision 24).
- Exports and analysis files carry the rater ID and the route, never the code.
- The page keeps answers that have not yet reached the server in the browser and sends them again. If they still have not arrived at the end, the end screen offers the rater a download of their results file (the journal's format), asks them to write to the contact for help, and says that only answers that reach the server are used.
  - Such an e-mail is a support channel only (Decision 24). The contact helps the rater to get the waiting answers to the server. The file is never imported or analysed: only answers stored on the server count, and a session the server does not hold in full is incomplete (§8 rule 5).
  - An e-mail to the contact, for support or for a withdrawal, gives the contact the sender's address. It is used only to reply, and the e-mail and any file with it are deleted once the request has been handled (§11.5).

### 11.4 No key on the server

- The server is configured with `$D/package` only. It never opens `$D/private`, and it refuses to start if the package holds a file named `key*.json`.
- A package marked `dry_run` (synthetic audio) is shown as TEST DATA on every page.
- A stimulus is served only by exact lookup of its id in the package's stimulus table, and only to a session whose half needs it.
- No endpoint returns a condition label, a pair id, a source path, an unrounded duration or any part of the key. A test runs a full session against a built test package and checks every response body for these.
- The admin endpoints need the token kept in `$D/admin_token`. They give:
  - progress counts per half (codes issued, consented, finished, completed, failed sound check), the number of codes per route, the codes issued and left under the cap, and each rater's route, half and progress (the raters list, `GET /api/admin/raters`; the admin page shows it but has no download, so it is saved with the command given in the README);
  - turning open registration on and off, setting the code cap (§7.1), and setting the footer contact;
  - importing the package, and saving the consent text;
  - creating invite codes;
  - disabling a code, and finding a rater by code (for withdrawal);
  - the export: one file per consented rater in the journal's results-file schema, and `answers.csv`, with ids only. With "completed sessions only" ticked (the admin page's default), it leaves out every rater who did not reach the end screen, so the pilot export and the final export are made with it unticked (§6.3, §12 step 9);
  - deletion of a rater.

### 11.5 Consent, privacy, retention

- The consent text, in Vietnamese and English, is in `mvc/mos/consent_template_v1.html` with the placeholders [CONTACT_NAME], [CONTACT_EMAIL], [ETHICS_NOTE] and [RETENTION_DATE]. The template is frozen with this text (§13).
  - The template also says (Decision 29): that a code is either sent by the organiser or issued by the site; that a code issued by the site is known only to the rater, is shown once and is needed to continue and to withdraw, and that losing it means no continuing and no withdrawal; that people take the test only once; that e-mail to the contact is a support channel and an e-mailed results file is never analysed; that there is no payment and no course credit; and that the test takes about one hour, in one sitting or two (§6.3). Its example rater ID is R007, and its sentence on who knows which code covers both routes.
  - The contact is decided (29 Oct 2026, Decision 27): [CONTACT_NAME] is Quoc Nguyen (UIT, VNU-HCM), and [CONTACT_EMAIL] is nvquoc@uit.edu.vn. The admin sets the same contact for the page footer (§12 step 7). [RETENTION_DATE] is 31 Dec 2027 (Decision 25), and [ETHICS_NOTE] is UIT's text (Decision 26).
  - The repository keeps the template with its placeholders. The filled copy is `$D/consent_v1.html`; it lives only in `$MOS_DATA_DIR` and is not committed. It differs from the template only in the four placeholders, unless UIT asks for a change (a deviation, §14).
  - It is saved once, before the pilot and after UIT's approval or exemption (§12 step 7); the admin page saves it trimmed, with one final newline. The sha256 of the saved file, which the admin's status page shows and which is stored with every consent, is logged in `mvc/DEVIATIONS_mvc_v1.md` (section "Listening test") when the file is saved. It is not in the freeze file (§13 step 3).
  - The server refuses a consent text that still holds any of the four placeholders, or a script, an event handler, a frame or a form.
  - The code's HMAC, rater ID, half and route are stored when the code is issued: on the invite route by the admin, on the open route when the rater starts. Before the rater ticks "Tôi đồng ý / I agree", nothing else is stored about the rater except the times the code was first and last used and, while signed in, the hash of the sign-in token.
- The hearing question is health information. The consent says so, says that it is optional, and says what it is used for (§9.4). The request to UIT asks for the consent to be checked against the Personal Data Protection Law and the decree that implements it (Decree 13/2023/ND-CP or its successor) (Decision 26).
- Withdrawal: until publication, a rater who sends the access code to the contact is deleted from the database, the exports and the analysis. A withdrawal after the analysis removes the rater from the published data. The results are then computed again, and both versions are reported (Decision 31).
  - The admin finds the rater by the code and deletes the rater. The server overwrites the rater's rows in the database and empties its write-ahead log, removes the rater's results file and rows of `answers.csv` from every folder under `$D/export/`, and removes the rater from every daily copy in `$D/backup/` (a copy it cannot clean is deleted).
  - The server does not change a saved raters list (`raters.json`, §6.3, §12 step 9) or any copy outside these folders, such as the pilot export in `$D/pilot/` or an analysis input: the withdrawn rater's entries there are removed by hand.
  - On the open route, nothing but the code links a rater to the answers. An open-route rater who has lost the code cannot be found in the data and so cannot withdraw. The information screen says so (Decision 29).
- Retention (Decision 25):
  - the person ↔ code list of the invite route is deleted when collection closes and the export has been checked;
  - e-mails to the contact, with any file attached, are deleted once the request has been handled (§11.3);
  - the audio cache is deleted once the build has been checked against the manifest;
  - the database and its backups (with the codes' HMACs, the browser family and the background answers), the exports and the pilot export are deleted when the paper is published, at the latest on 31 Dec 2027 ([RETENTION_DATE] in the consent);
  - the anonymised ratings (rater IDs R001 …, without codes, background answers or browser family) are written from the export at the analysis (§12 step 10) and published with the paper, together with `items.json` and the key. The background answers are reported as counts only, so the sensitivity analyses of §9.4 that use them are reported in the paper but cannot be recomputed from the published ratings.
- FileVault is off on this Mac, so its disk is not encrypted. The data directory is mode 700 under the author's account, and the server writes its files with mode 600. Turning FileVault on is recommended; it is the author's action, since it is a system security setting. A daily copy of the database goes to `$D/backup/`.
- VCTK and LibriTTS-R are licensed CC BY 4.0, and the site credits both. The site is closed by stopping the tunnel at the end of the grace period, before the final export (§12 step 9).

## 12. Order of work and unblinding

1. The author approves this text. Every item of §16 is decided (30 Oct 2026, by the author's delegation), except UIT's answer on ethics, which is pending and is needed only before the pilot (step 7).
2. The `mvc/mos/` code, the design file, the consent template and the tests are final:
   - the server and page meet §5, §6 and §11;
   - an independent review has checked them against this addendum;
   - the guard test against real git repositories (`MOS_GIT_TESTS=1`) has passed once.
3. Freeze (§13): commit, tag `listening-addendum-mvc-v1`, push.
4. `build_stimuli.py draw` runs once with the tagged builder and design, with `--out $D/private/items.json`. `$D/private/` is created first with mode 700 if it does not exist, since `draw` does not create it. Its guard checks the freeze. `items.json` and `items.json.sha256` are then in `$D/private/`.
5. `build_stimuli.py build` checks the freeze again, draws again and compares, fetches the audio, and writes the package and the key into `$D`, and the manifest into `mvc/LISTENING_BUILD_mvc_v1.json`.
6. The manifest is committed, staged by name, on the author's instruction.
7. Before the pilot, in this order:
   1. UIT's approval or exemption is in hand. No rater, pilot included, takes part before its text is in [ETHICS_NOTE] (Decision 26). If UIT asks for a change of the design, the code, the consent or this procedure, that is a deviation (§14).
   2. A fresh database: the server and the tunnel are stopped, and `$D/mos.sqlite` (with its `-wal` and `-shm` files), the sample-mode copies in `$D/backup/` and any export in `$D/export/` are moved out of `$D` or deleted. So no test or sample-mode rater is left, rater IDs start at R001, and the counts of §7.1 start at zero. Deleting raters is not enough: deleted codes still count towards the cap. The server is started again without `MOS_SAMPLE`.
   3. On the admin page, reached on 127.0.0.1 while the tunnel is still stopped, the admin turns open registration off (the server's default is on, also in a fresh database), sets the code cap to 43 (§7.1), and checks both on the status page. The tunnel is then started again.
   4. The consent is filled (§11.5) and saved as `$D/consent_v1.html`. Its sha256, from the status page, is logged at once in `mvc/DEVIATIONS_mvc_v1.md` (section "Listening test"), with the time.
   5. The admin imports the package (§4.3), checks on the status page that the consent has the logged sha256 and no problem, and sets the footer contact (§11.5).
   6. The pilot runs (§6.3). Its length rule is applied, and the pilot raters are exported and deleted (§6.3).
8. Invitations go out and open registration is turned on. During collection, only screening statuses are computed (§8), and only by the screening and analysis script of §9.5. That script is written, tested and independently reviewed after the freeze, and its sha256 is committed and logged before it first reads any answer of a real rater (pilot included), and in any case before step 9.
9. Collection closes (§7.1): open registration is turned off, codes without a recorded consent are disabled, and no further invite code is issued. After the grace period of 3 days, the tunnel is stopped, which closes the site (§11.5), and the export is made with "completed sessions only" unticked (§11.4). The admin's raters list (§11.4), the body of `GET /api/admin/raters`, and the admin status, the body of `GET /api/admin/status`, are saved in the export folder as `raters.json` and `status.json`. The person ↔ code list of the invite route is then deleted.
10. The analysis runs once. The results and `items.json` are committed, and the anonymised ratings are written for publication (§11.5).

Before step 10, no MOS or SMOS value of any condition is computed, plotted or looked at. The export is opened only by the screening and analysis script. The organiser does not open the key or `items.json`: the builder writes them, and only the scripts read them.

## 13. Freeze procedure and the builder's guard

1. The author reads, edits and approves this text and the design file. Only the author freezes them.
2. The text is saved with the status line "frozen on (date, time +07:00) by Quoc Nguyen", and the design file's sha256 in §3.5 is checked against the file (a change of the design file before the freeze changes it there).
3. `mvc/LISTENING_FREEZE_mvc_v1.sha256` holds, in the format of `mvc/FREEZE_mvc_v1.sha256` (`shasum -a 256` lines, paths relative to the repository root), the sha256 of:
   - this addendum and `mvc/mos/design_mvc_v1.json`;
   - every file of `mvc/mos/` that the freeze commit holds: server, page, builder, tests, README, `package.json` and the consent template (not `__pycache__/`, which git ignores);
   - `benchmark/stats.py`;
   - the two pair lists and the six T1 row files the draw reads (Mac copies).
   It does not hold the screening and analysis script of §8–§9.5, which is written after the freeze and committed before it reads any rating (§9.5, §12 step 8), nor the filled consent `$D/consent_v1.html`, whose sha256 is logged when it is saved, before the pilot (§11.5, §12 step 7).
4. Commit, with the addendum's sha256 in the message, staging by name only this addendum, the files of `mvc/mos/` that git does not ignore (26 on 30 Oct 2026, the design file included), `.gitignore` and `mvc/LISTENING_FREEZE_mvc_v1.sha256`. The pair lists, the T1 row files and `benchmark/stats.py` are hashed in the freeze file but not staged; nothing under `paper/` or `benchmark/` is staged. Create the annotated tag `listening-addendum-mvc-v1`, then push the branch `mvc-v1-freeze` and the tag to the public GitHub repository. Claude does this only on the author's explicit instruction. Without the push, the paper calls the addendum "internally frozen", not "pre-registered".
5. The builder's guard. `build_stimuli.py draw` on the real pair lists, and `build_stimuli.py build`, refuse to run unless all of these hold:
   - the addendum file has the sha256 given on the command line;
   - the folder two levels above the builder (the repository root) is the top of its git repository (there is no `--repo` option);
   - the tag exists, and at the tag's commit the addendum, the running `build_stimuli.py` and the design file each equal the file on disk;
   - the tag on the public remote points at the same commit, checked with `git ls-remote --tags origin` (the peeled commit of an annotated tag).
   `build` also refuses unless:
   - drawing again from the design file, the pair lists and the rows named in `items.json` gives `items.json` exactly;
   - the package, the key folder and the cache are outside the repository, and neither the key folder nor the cache contains the package or lies inside it;
   - the package folder is empty or absent, and no key or manifest exists yet.
   A draw with `--throwaway_seed` (a feasibility check) runs without the freeze, and `build` refuses its output. `build --dry_run` skips the freeze and the fetch for synthetic inputs only: it refuses the real pair lists and any `/datastore` path, and marks its package `dry_run` (§11.4).
6. These go into `mvc/DEVIATIONS_mvc_v1.md`, under the heading "Listening test": the push time, draw time and build time; UIT's approval or exemption (date and reference); the fresh database; the saving of the filled consent, with its sha256; the pilot dates, the median pilot session and whether the two-sitting request of §6.3 applied; the deletion of the pilot raters; the commit of the screening and analysis script, with its sha256; the times open registration is turned on and off; every change of the code cap; first and last main rating; the close, the stopping of the tunnel and the export.
7. The freeze comes before the draw, and the draw comes before any stimulus exists.

## 14. Deviations

- Every change after the freeze goes into `mvc/DEVIATIONS_mvc_v1.md` with:
  - its date and time (+07:00) and its reason;
  - what existed when it was decided: whether the items were drawn and the package built, how many raters had started or completed, and whether any screening status had been computed;
  - the files changed, with their new sha256;
  - whether a rule is affected.
- A change to the design file or the builder after the freeze needs a new tag (`listening-addendum-mvc-v1-r1`, …), pushed before the next draw or build. §3.4 gives the one foreseen case.
- A change that UIT asks for (Decision 26) is a deviation like any other. It is decided before the pilot, and if it touches the design file or the builder it needs a new tag.
- A change of the filled consent after it is saved is logged with its new sha256. A rater who has not yet agreed is shown only the new text (one who had the old text on screen is asked to agree to the new one); each consent keeps the sha256 of the text agreed to, and raters who agreed earlier are not asked again. A change of the screening and analysis script after its sha256 is committed is a deviation.
- No rule of §8–§9 is changed after the first main rating exists. If one must be, the frozen rule's result is still reported, and every other result is labelled post hoc.
- Operational changes are logged and change no rule. Examples: the port, the tunnel, a restart, a server fix that changes neither the trials nor the stored answers, turning open registration off before the close for an operational reason, or a change of the code cap (§7.1).
- A fix that changes what raters see, or a rebuild of the package (§4.3), is logged with the raters it affected.

## 15. Reporting

- LT-1 or LT-2 rejected in favour of mvc_v1: "Listeners rated mvc_v1 as more natural (LT-1), or as more similar to the target speaker (LT-2), than CRT-AL (Δ [CI], Holm over 2)."
- SMOS against Seed-VC and kNN-VC: Δ [CI], labelled descriptive, with no p-value and no Holm decision.
  - They are reported whatever the result, in the same table as LT-1 and LT-2 and in the text beside LT-2, with the note that the author chose to confirm only against CRT-AL (§1).
  - Where a CI excludes 0, the text gives its direction with the word "descriptive", for example "listeners rated Seed-VC as more similar to the target speaker than mvc_v1 (Δ [CI], descriptive)".
  - Where a CI lies entirely below 0, any sentence of the abstract on the listening test says so too.
  - Never "comparable", "matches" or "parity".
- Not rejected: "no difference in rated naturalness (or similarity) shown", with the CI. Never "comparable", "matches" or "parity".
- LT-1 or LT-2 rejected in favour of CRT-AL: stated in the abstract, beside the H1 result.
- Fewer than 8 screened raters in a half: the test is called "a descriptive listening test".
- MOS against Seed-VC and kNN-VC: reported as SMOS against them above (Δ [CI], labelled descriptive, whatever the result; where a CI excludes 0, its direction with the word "descriptive"; where it lies entirely below 0, any sentence of the abstract on the listening test says so too). Never "comparable", "matches" or "parity".
- The values are relative to this test: non-native listeners, this rater group (both access routes), and this range of conditions, with CRT-AL as a low anchor. They are not compared with MOS values from other papers.
- Reported whatever the result:
  - screening counts with reasons, and kept raters per half;
  - the counts per access route and the breakdown by route (§9.3);
  - background counts and session length, and whether the two-sitting request of §6.3 applied;
  - MOS and SMOS of every condition per set, and the agreement table;
  - the sensitivity analyses with the decisions they change;
  - the deviations;
  - the note that the gold rules favour the natural anchors;
  - the development draw of §0.2 item 6.

## 16. Decisions of the author

On 30 Oct 2026 the author delegated every open decision of this section: "make it easy, you decide it, let go". The items decided on 29 Oct 2026 stand. Every other item is now marked "Decided 30 Oct 2026 (delegated by the author)", and the body of this text agrees with it. Only UIT's answer on ethics (item 26) is pending.

1. **Status.** Decided 29 Oct 2026: confirmatory only against CRT-AL. Family LT is LT-1 (MOS of mvc_v1 − CRT-AL) and LT-2 (SMOS of mvc_v1 − CRT-AL), both two-sided, Holm over 2, α 0.05 (MDE 0.44 / 0.61 at 20 raters, §10). Everything against Seed-VC and kNN-VC is descriptive (CIs only), including the draft's LT-3 and LT-4 (SMOS against Seed-VC and kNN-VC), which move to §9.3. §1 says plainly that the author chose to confirm only against CRT-AL.
2. **Estimand.** Decided 30 Oct 2026 (delegated by the author): the mean over the 32 items of each item's mean over its raters (equal item weights), pooled over the two sets. Per-set values are descriptive.
3. **Conditions.** Decided 29 Oct 2026: `nat`, `mvc_v1`, `crtal`, `seedvc` and `knnvc` in both blocks; `src` in SMOS only. The design file holds these conditions. Decided 30 Oct 2026 (delegated by the author): `src` stays in SMOS whatever the pilot shows (item 16).
4. **Baselines left out.** Decided 29 Oct 2026 through item 3: FreeVC and FACodec are out (FACodec would add about 10 min).
5. **Freeze order.** Decided 30 Oct 2026 (delegated by the author):
   - The text, the design file, the code of `mvc/mos/` (server, page, builder, tests, README, `package.json`) and the consent template are frozen first (commit, tag, push); then the draw runs with the tagged builder; then the build. The builder enforces this order (§13), and `items.json` stays private until the close.
   - UIT's approval or exemption comes before the pilot (§12 step 7), not before the freeze. If UIT asks for a design change, that is a deviation handled by §14.
   - The filled consent `$D/consent_v1.html` is not in the freeze file. Its sha256 is logged in `mvc/DEVIATIONS_mvc_v1.md` (section "Listening test") when it is saved, before the pilot.
   - The screening and analysis script of §8–§9.5 is written, tested and independently reviewed after the freeze. Its sha256 is committed and logged before it reads any rating, the pilot's included, and before the close (§9.5, §12 step 8). The freeze file does not hold it.
6. **Seed and the development draw.** Decided 30 Oct 2026 (delegated by the author): seed string `'20260928:mvc_v1'`. The development draw's file was deleted on 29 Oct 2026, and no person read its item list (§0.2 item 6). The draw is reported (§15). The alternative, a joint session reusing the journal's 32 items, is not taken.
7. **Eligibility (§3.2).** Decided 30 Oct 2026 (delegated by the author), as drafted:
   - Source and P of 2–10 s.
   - Natural clip never cut.
   - SMOS reference of 3–10 s closest to 5 s; otherwise the shortest E over 10 s, cut to 10 s (about 4 of the 16 LTC-2k items).
8. **Speaker rule.** Decided 30 Oct 2026 (delegated by the author), as drafted: one item per source speaker and one per target speaker within a set. In LTC-2k, a speaker may be the source of one item and the target of another (about 7 of 16 items).
9. **Sentence rule.** Decided 30 Oct 2026 (delegated by the author): on, with one item per source sentence within a set. Its cost: in about 6 % of draws, one VCTK-2k M2M item has no P.
10. **Excluded pairs.** Decided 30 Oct 2026 (delegated by the author): the 12 pairs of `mvc_listen_test.html` (§3.3).
11. **Processing.** Decided 30 Oct 2026 (delegated by the author), as drafted:
    - Active level −26 dBFS, peak at most −1 dBFS.
    - FLAC, 16-bit, 24 kHz; each system's own bandwidth kept.
    - 0–250 ms of random trailing silence per clip.
    - Durations rounded to 0.1 s in anything the browser receives.
12. **Wording (§5).** Decided 30 Oct 2026 (delegated by the author), as drafted: Vietnamese by default with an English switch; SMOS label 2 is "Có lẽ là hai người khác nhau"; the SMOS scale is described as a 5-point adaptation of the VCC same/different scale.
13. **Session layout.** Decided 30 Oct 2026 (delegated by the author), as drafted: halves fixed now; MOS before SMOS; a break screen in the middle of each block.
14. **Trial order.** Decided 30 Oct 2026 (delegated by the author), as drafted: no item on two consecutive main trials. Implemented in `mvc/mos/public/lt_core.js` (`orderMain`: rejection sampling, uniform over valid orders, with a deterministic repair as a fallback).
15. **Attention trials.** Decided 30 Oct 2026 (delegated by the author), as drafted: MOS asks for 2 and 4; SMOS asks for 4 and 2.
16. **Pilot.** Decided 30 Oct 2026 (delegated by the author):
    - As drafted: 3 raters with invite codes noted "pilot", on top of the 40 codes (cap 43 before the pilot); their export is moved to `$D/pilot/`, and they are deleted from the database before invitations go out (§6.3).
    - `src` stays in SMOS. If the median pilot session is longer than 60 min, main raters are asked to take the session in two sittings, stopping at a break screen and continuing later with their code, which the site supports. Nothing in the package changes.
    - The earlier mechanism, a second trial list without `src` (`lt_data_nosrc.json`) imported in place of the first, is dropped. The builder never wrote that file, so no code changes.
17. **Raters.**
    - Decided 29 Oct 2026 (registration): raters enter through both routes, invite codes issued by the admin and open registration, where the site issues an anonymous code to anyone who starts. Both routes are on during collection, and the route is recorded per rater. Open-route raters get the same screening; the stopping rule counts screened, complete raters of both routes; the results are broken down by route, descriptively; the admin turns open registration off at the close (§7.1, §7.3, §9.3, §11.2).
    - Decided 29 Oct 2026 (code cap): at most 40 codes over both routes together, besides the 3 pilot codes (cap set to 43 before the pilot). This is what the server enforces: every code ever issued counts, including open-route starts without consent, and at the cap neither route issues codes.
    - Decided 30 Oct 2026 (delegated by the author): 20 screened raters (10 per half) over both routes, and at least 8 per half for the LT decisions.
    - Decided 30 Oct 2026 (delegated by the author): the server's half rule is accepted. The server sets the half when a code is issued, on both routes, by the rule of §7.1 (fewer consented raters who can still complete, then fewer codes, then half A), never by a rating; the admin cannot choose the half.
    - Decided 30 Oct 2026 (delegated by the author): collection closes when both halves have 10 screened, complete raters, or on 15 Dec 2026 at 23:59 +07:00, whichever comes first. At the close, codes without a recorded consent are disabled, and started sessions may be finished during a grace period of 3 days before the export (§7.1, §12 step 9).
18. **Devices.** Decided 29 Oct 2026: computers and phones, both with headphones.
19. **Recruitment.** Decided 30 Oct 2026 (delegated by the author): UIT students and staff, and colleagues, reached by e-mail, through the faculty's channels and in social media posts. No payment and no course credit; the consent says so. Students in the author's current courses may take part, but taking part is never linked to marks, invitations are not sent through course announcements, and these students take part through the open route, without a personal invite code. Apart from raters who write to the contact, the author does not learn who rated: only rater IDs reach the analysis, and the person ↔ code list holds no rater ID (§7.3).
20. **Exclusions.** Decided 30 Oct 2026 (delegated by the author), as drafted: the authors, people who worked on the systems, anyone who has listened to a labelled output page, the pilot raters, and anyone who has already completed this test (§7.2). `mvc_listen_test.html` has never been in the public repository (item 30).
21. **Background question 6.** Decided 30 Oct 2026 (delegated by the author), as drafted: an earlier listening test of this group (Yes / No / Not sure).
22. **Screening thresholds (§8).** Decided 30 Oct 2026 (delegated by the author), as drafted: gold, more than 2 of 16; `gold_src`, rated 4 or more on more than 2 of 8; all 4 attention answers must be as asked.
23. **Analysis.** Decided 30 Oct 2026 (delegated by the author), as drafted:
    - Stratified crossed bootstrap, B 10,000, seed 20260928.
    - Secondary mixed model on the paired differences.
    - Power reported with the bootstrap SE: MDE 0.44 / 0.61 (Holm over 2, following item 1) at 20 raters.
24. **Website and data.**
    - Access routes: decided under item 17.
    - Decided 30 Oct 2026 (delegated by the author), as implemented in the server: codes of 60 bits on both routes, stored as HMAC-SHA-256 with a server secret; rater IDs R001, …; half and route stored with the code and in the exports; the code cap; no IP addresses; browser family and major version only; withdrawal also from the exports and the backups.
    - Decided by the author before 30 Oct 2026: no launchd; the server and the tunnel run in Terminal tabs (§11.1).
    - Decided 30 Oct 2026 (delegated by the author): open registration stays on by default in the server. The procedure turns it off from the saving of the consent until invitations go out, and at the close (§7.1, §11.2, §12).
    - Decided 30 Oct 2026 (delegated by the author): cloudflared runs at its default log level (info), at which its log records no client addresses; the operator checked the running tunnel (§11.3).
    - Decided 30 Oct 2026 (delegated by the author): a results file that a rater e-mails after a failed send is a support channel only. It is never analysed; only answers stored on the server count (§11.3).
    - The server and page are aligned with §5, §6 and §11 before the freeze (§12 step 2).
25. **Retention.** Decided 30 Oct 2026 (delegated by the author):
    - [RETENTION_DATE] is 31 Dec 2027. By then, or earlier when the paper is published, the database and its backups (with the codes' HMACs, the browser family and the background answers), the exports and the pilot export are deleted.
    - The anonymised ratings (rater IDs), `items.json` and the key are published with the paper.
    - The audio cache is handled like the key and deleted once the build has been checked.
    - FileVault is off on this Mac. The data directory is mode 700 under the author's account. Turning FileVault on is recommended; it is the author's action, since it is a system security setting.
26. **Ethics.** Pending UIT. Decided 29 Oct 2026: the author asks UIT for approval or exemption; no rater, pilot included, takes part before its text is in [ETHICS_NOTE], and until then the site serves only TEST DATA. Decided 30 Oct 2026 (delegated by the author): UIT's answer is needed before the pilot (§12 step 7), not before the freeze; a change that UIT asks for is a deviation (§14). The request to UIT includes a check of the consent against Vietnam's personal data rules (the Personal Data Protection Law and the decree that implements it, Decree 13/2023/ND-CP or its successor); the hearing question is health information.
27. **Contact and domain.**
    - Decided 29 Oct 2026 (contact): Quoc Nguyen (UIT, VNU-HCM), nvquoc@uit.edu.vn, for [CONTACT_NAME] and [CONTACT_EMAIL]. The repository keeps the template with its placeholders; the filled copy lives only in `$MOS_DATA_DIR` (§11.5).
    - Decided 30 Oct 2026 (delegated by the author): the domain mos.betaschool.edu.vn is accepted. The consent says it is not a UIT domain and names the research team as the operator.
28. **Builder's guard (§13 step 5).** Decided 30 Oct 2026 (delegated by the author), as drafted:
    - The tag must be on the public remote.
    - The running builder and the design file must equal their copies at the tag.
    - No `--repo` option.
    - `build` draws again and compares.
29. **Consent template.** Decided 30 Oct 2026 (delegated by the author): `mvc/mos/consent_template_v1.html` keeps its four placeholders ([CONTACT_NAME], [CONTACT_EMAIL], [ETHICS_NOTE], [RETENTION_DATE]), which are filled only in `$D`. It covers both access routes: a code sent by the organiser, or one issued by the site, known only to the rater and needed to continue and to withdraw, so that losing such a code means no withdrawal. It asks people to take the test only once, uses R007 as its example rater ID, says that the organiser's list of who received which code concerns invite codes only, and names e-mail to the contact as a support channel whose results files are never analysed. It also says that there is no course credit, and gives the length as about one hour, in one sitting or two (§6.3, §11.5).
30. **§0.2 item 3.** Decided 30 Oct 2026 (delegated by the author): `mvc_listen_test.html` has been heard only by the author, who does not rate. It is a private claude.ai artifact and has never been in the public repository: no file under `paper/mimi_vc/` has ever been committed (checked 30 Oct 2026), and `.gitignore` now excludes the labelled pages.
31. **Withdrawal after the analysis.** Decided 30 Oct 2026 (delegated by the author): compute again and report both versions (§11.5).
32. **Replacement and rebuild.** Decided 30 Oct 2026 (delegated by the author), as drafted:
    - A permanently lost file leads to `exclude_pairs`, a new tag `-r1` and a new draw (§3.4).
    - The package is built once; a rebuild after the first rating is a deviation (§4.3).
33. **Freeze file contents (§13 step 3).** Decided 30 Oct 2026 (delegated by the author): the addendum, the design file, the files of `mvc/mos/` in the freeze commit (the consent template included), `benchmark/stats.py`, the two pair lists and the six T1 row files, in the format of `mvc/FREEZE_mvc_v1.sha256`. Not the screening and analysis script, and not the filled consent (item 5). The freeze commit stages only the addendum, the files of `mvc/mos/`, `.gitignore` and the freeze file; the pair lists, the row files and `benchmark/stats.py` are hashed, not staged (§13 step 4).
34. **Timing.** Decided 30 Oct 2026 (delegated by the author): the paper is not submitted before the close. If fewer than 8 screened, complete raters per half exist at the export, LT is reported as descriptive (§7.1, §15).
35. **Reporting the descriptive MOS and SMOS comparisons against Seed-VC and kNN-VC (§15).** Decided 30 Oct 2026 (delegated by the author), as drafted, following item 1: reported whatever the result, beside LT-1 and LT-2; where a CI excludes 0, its direction is given with the word "descriptive"; where a CI lies entirely below 0, any sentence of the abstract on the listening test says so too. The alternative for MOS, CIs only with no direction, is not taken.
36. **Start and close of collection (§12 steps 7 and 9).** Decided 30 Oct 2026 (delegated by the author):
    - A fresh `$D/mos.sqlite` before the pilot, and the code cap set to 43.
    - Open registration off from the saving of the consent until invitations go out (step 8).
    - At the close, open registration off and codes without a recorded consent disabled; at the end of the grace period of 3 days, the tunnel stopped and the export made, with the admin's raters list and status saved in it as `raters.json` and `status.json` (§12 step 9).
