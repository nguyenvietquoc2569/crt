# MimiFlow-VC test results v1 (mvc/ANALYSIS_PLAN_mvc_v1.md §5–§7)

Written 2026-10-28T22:43:29+07:00 by `mvc/analysis/analyze_mvc.py` (sha256 9897f7e0e99fc6dbaf09d1ac12cf9b5dbb0b63afc3685741ee74501788a5958c; tests 0818761f5f4489651e7114be6f0dc2df3cbb055faa7ac899d0daa7ba63f0fb10; stats.py a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6; analyze_tests.py 03f2bbf76f79338a46c0ad191b59979c6358dd1f6c7aaee908508a927cf2a933; analyze_exploratory.py ac53e40bac8ca55bf6327f3dbea3bf09268994c0d46465931b585de49a10ced3). Selection record sha256 b84885feb5fa0dc9b8c4cf35ee0f772f0cffe65cf21016f74d70f64c0920b9c8 (.json), 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42 (.md). Crossed bootstrap B = 10000, seed 20260928: VCTK-2k independent, LTC-2k linked.

- FreeVC was trained on VCTK ("VCTK-seen"): its VCTK-2k rows stay in family S, but no claim of beating or matching FreeVC on VCTK-2k is made from them (§5.1, §7.3).
- Seed-VC uses only the first 25 s of the reference at 30 s and 180 s.
- MimiFlow-VC decodes in fp32; the journal's Mimi copies decode 32 codes in fp16. Hardware: H200 MIG 2g.35gb on ceai.
- LTC-2k at 180 s: the 101 pairs with a reference shortfall (targets 3729, 6930) are removed from every 180 s pair list before any comparison, for every system (§1, §5.4).
- The decisions rest on one speaker model (WavLM-SV), with ECAPA as the second; the NL margin of +0.03 was set on Dev; LTC-2k has no natural-level check.
- Wilcoxon signed-rank p and the rank-biserial r are possibly anti-conservative, since pairs share speakers (§5.2).

## Outcome (§7.1) and leakage (§7.2)

- **Outcome A**: mvc_v1 improves on CRT-AL in intelligibility, speaker similarity and naturalness on both test sets. Rejections favouring mvc_v1: H1-1, H1-2, H1-3, H1-4, H1-5, H1-6; favouring CRT-AL: none; not rejected: none (point estimate favours CRT-AL: none).
- **Natural level (NL)**: leakage above the natural level not excluded (NL1s, NL1c); no claim of a converter without source leakage. This goes in the abstract.
- **Relative to kNN-VC (L)**: leakage relative to kNN-VC not shown to be within +0.01 (L1c).

mvc_v1 on Dev-900, selected on Dev: capped WER 1.89, SIM 0.913, UTMOS 3.863, NL same / cross -0.008 / 0.020.

### Tiers of PLAN §6.1 (point-estimate benchmarks, not tests)

- Must: outcome A True, ECAPA coverage 0.522 (≥ 0.50: True), leakage reported per stratum → **True**.
- Should, vctk2k: capped WER 2.51, SIM 0.886, UTMOS 3.829, WER(3 s) − WER(30 s) -0.31 (RBN-3 CI rule True) → False; Stretch → False.
- Should, ltc2k: capped WER 2.37, SIM 0.928, UTMOS 4.027, WER(3 s) − WER(30 s) 0.03 (RBN-3 CI rule True) → False; Stretch → False.
- Should overall False; Stretch overall False.

## Family H1 (primary): mvc_v1 − CRT-AL (cand_pm18_rq)

Holm over 6, α 0.05.

| Hypothesis | Set | Endpoint | A | B | Δ = A − B [95 % CI] | p | Holm decision | n pairs | Wilcoxon p (r_rb) | mvc_v1 on Dev-900, selected on Dev |
|---|---|---|---|---|---|---|---|---|---|---|
| H1-1 | vctk2k | capped WER % | 2.51 | 16.26 | -13.75 [-17.36, -10.44] | 0.0002 | reject, favours mvc_v1 | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.893) | 1.89 |
| H1-2 | ltc2k | capped WER % | 2.37 | 10.33 | -7.96 [-9.67, -6.42] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.924) | 1.89 |
| H1-3 | vctk2k | SIM | 0.886 | 0.806 | 0.080 [0.052, 0.109] | 0.0002 | reject, favours mvc_v1 | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.749) | 0.913 |
| H1-4 | ltc2k | SIM | 0.928 | 0.911 | 0.017 [0.009, 0.026] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.475) | 0.913 |
| H1-5 | vctk2k | UTMOS | 3.829 | 2.628 | 1.201 [1.132, 1.271] | 0.0002 | reject, favours mvc_v1 | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (1.000) | 3.863 |
| H1-6 | ltc2k | UTMOS | 4.027 | 2.731 | 1.296 [1.216, 1.374] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (1.000) | 3.863 |

## Family NL: natural-level leakage on VCTK-2k, H0 NL_g ≥ +0.03

| Test | Stratum | SIM_src mvc_v1 | SIM_src target_copy | NL [95 % CI] | SE | Rejects | n (no source E; failed mvc_v1/target_copy) | Worst case | summarize.py value [95 % CI] | SIM mvc_v1 / target_copy, same pairs | ECAPA NL [95 % CI] | Flag | NL, mvc_v1 on Dev-900, selected on Dev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NL1s | same | 0.826 | 0.807 | 0.019 [-0.001, 0.040] | 0.010 | False | 491 (0; 0/0) | – | 0.009 [-0.010, 0.029] | 0.906 [0.889, 0.922] / 0.963 [0.957, 0.969] | -0.011 [-0.036, 0.015] |  | -0.008 |
| NL1c | cross | 0.609 | 0.581 | 0.028 [-0.002, 0.056] | 0.015 | False | 489 (0; 0/0) | – | 0.016 [-0.013, 0.044] | 0.891 [0.872, 0.907] / 0.963 [0.957, 0.969] | 0.029 [0.004, 0.053] |  | 0.020 |

Both reject (the claim): False; at the worst case: False.

Reference, paired NL of the comparators (descriptive): cand_pm18_rq 0.001 / -0.014; knnvc 0.049 / 0.055; freevc 0.070 / 0.100; seedvc 0.051 / 0.065; facodec 0.095 / 0.096.

## Family L: SIM_src(mvc_v1) − SIM_src(kNN-VC), H0 Δ_src ≥ +0.01

| Test | Set / stratum | SIM_src mvc_v1 | SIM_src kNN-VC | Δ_src [95 % CI] | Rejects | n (no source E; failed) | Worst case | Flag |
|---|---|---|---|---|---|---|---|---|
| L1s | vctk2k / same | 0.816 | 0.846 | -0.030 [-0.051, -0.009] | True | 1000 (0; 0/0) | – |  |
| L1c | vctk2k / cross | 0.598 | 0.626 | -0.028 [-0.069, 0.014] | False | 1000 (0; 0/0) | – |  |
| L2s | ltc2k / same | 0.824 | 0.858 | -0.034 [-0.049, -0.022] | True | 900 (0; 0/0) | – |  |
| L2c | ltc2k / cross | 0.559 | 0.612 | -0.053 [-0.070, -0.037] | True | 900 (0; 0/0) | – |  |

All four reject (the claim): False; at the worst case: False.

## Family S: mvc_v1 − each baseline

Holm over 24, α 0.05. NL (§7.2): NL1s False, NL1c False; leakage above the natural level not excluded (NL1s, NL1c); no claim of a converter without source leakage.

| Hypothesis | Set | Endpoint | A | B | Δ = A − B [95 % CI] | p | Holm decision | n pairs | Wilcoxon p (r_rb) | mvc_v1 on Dev-900, selected on Dev |
|---|---|---|---|---|---|---|---|---|---|---|
| S-knnvc-1 | vctk2k | capped WER % | 2.51 | 6.23 | -3.73 [-5.57, -2.13] | 0.0002 | reject, favours mvc_v1 | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.593) | 1.89 |
| S-knnvc-2 | ltc2k | capped WER % | 2.37 | 3.79 | -1.42 [-1.95, -0.93] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.463) | 1.89 |
| S-knnvc-3 | vctk2k | SIM | 0.886 | 0.895 | -0.009 [-0.028, 0.009] | 0.3502 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.132) | 0.913 |
| S-knnvc-4 | ltc2k | SIM | 0.928 | 0.929 | -0.001 [-0.007, 0.006] | 0.7837 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.3345 (0.026) | 0.913 |
| S-knnvc-5 | vctk2k | UTMOS | 3.829 | 3.664 | 0.165 [0.092, 0.232] | 0.0002 | reject, favours mvc_v1 | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.530) | 3.863 |
| S-knnvc-6 [not confirmed by a second naturalness predictor] | ltc2k | UTMOS | 4.027 | 3.837 | 0.190 [0.143, 0.236] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.636) | 3.863 |
| S-freevc-1 [VCTK-seen] | vctk2k | capped WER % | 2.51 | 2.03 | 0.48 [-0.37, 1.27] | 0.2542 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.0443 (0.205) | 1.89 |
| S-freevc-2 | ltc2k | capped WER % | 2.37 | 2.53 | -0.16 [-0.65, 0.30] | 0.5091 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0142 (-0.134) | 1.89 |
| S-freevc-3 [VCTK-seen] | vctk2k | SIM | 0.886 | 0.887 | -0.001 [-0.017, 0.014] | 0.8667 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.4318 (-0.020) | 0.913 |
| S-freevc-4 | ltc2k | SIM | 0.928 | 0.897 | 0.031 [0.018, 0.047] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.500) | 0.913 |
| S-freevc-5 [VCTK-seen] | vctk2k | UTMOS | 3.829 | 4.001 | -0.172 [-0.245, -0.102] | 0.0002 | reject, favours freevc | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.498) | 3.863 |
| S-freevc-6 | ltc2k | UTMOS | 4.027 | 4.019 | 0.007 [-0.052, 0.066] | 0.7819 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.6139 (0.014) | 3.863 |
| S-seedvc-1 | vctk2k | capped WER % | 2.51 | 2.59 | -0.08 [-0.90, 0.71] | 0.8609 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.5378 (-0.065) | 1.89 |
| S-seedvc-2 | ltc2k | capped WER % | 2.37 | 2.50 | -0.13 [-0.50, 0.22] | 0.4630 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0896 (-0.100) | 1.89 |
| S-seedvc-3 | vctk2k | SIM | 0.886 | 0.919 | -0.033 [-0.042, -0.025] | 0.0002 | reject, favours seedvc | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.697) | 0.913 |
| S-seedvc-4 | ltc2k | SIM | 0.928 | 0.946 | -0.018 [-0.023, -0.013] | 0.0002 | reject, favours seedvc | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.592) | 0.913 |
| S-seedvc-5 | vctk2k | UTMOS | 3.829 | 3.787 | 0.042 [-0.017, 0.105] | 0.1656 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.166) | 3.863 |
| S-seedvc-6 | ltc2k | UTMOS | 4.027 | 4.072 | -0.045 [-0.111, 0.018] | 0.1678 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.137) | 3.863 |
| S-facodec-1 | vctk2k | capped WER % | 2.51 | 2.45 | 0.06 [-0.85, 0.95] | 0.8963 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.7350 (0.032) | 1.89 |
| S-facodec-2 | ltc2k | capped WER % | 2.37 | 2.14 | 0.23 [-0.17, 0.62] | 0.2612 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0140 (0.142) | 1.89 |
| S-facodec-3 | vctk2k | SIM | 0.886 | 0.876 | 0.010 [-0.001, 0.021] | 0.0728 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.218) | 0.913 |
| S-facodec-4 | ltc2k | SIM | 0.928 | 0.910 | 0.018 [0.008, 0.029] | 0.0012 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.415) | 0.913 |
| S-facodec-5 | vctk2k | UTMOS | 3.829 | 3.628 | 0.201 [0.115, 0.288] | 0.0002 | reject, favours mvc_v1 | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.498) | 3.863 |
| S-facodec-6 | ltc2k | UTMOS | 4.027 | 3.580 | 0.447 [0.367, 0.516] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.840) | 3.863 |

## Family NI: non-inferiority against Seed-VC and kNN-VC

Holm over 12 at α 0.025 on p_NI = (1 + #{s_q Δ* ≤ −δ_q}) / (B + 1). NL (§7.2): NL1s False, NL1c False; leakage above the natural level not excluded (NL1s, NL1c); no claim of a converter without source leakage.

| Hypothesis | Set | Endpoint | mvc_v1 | B | Δ [95 % CI] | Margin | p_NI | CI rule | Holm | n pairs | mvc_v1 on Dev-900, selected on Dev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NI-seedvc-1 | vctk2k | capped WER % | 2.51 | 2.59 | -0.08 [-0.90, 0.71] | Δ < +1 | 0.0046 | True | non-inferior | 1020 (0/0 failed) | 1.89 |
| NI-seedvc-2 | ltc2k | capped WER % | 2.37 | 2.50 | -0.13 [-0.50, 0.22] | Δ < +1 | 0.0001 | True | non-inferior | 1800 (0/0 failed) | 1.89 |
| NI-seedvc-3 | vctk2k | SIM | 0.886 | 0.919 | -0.033 [-0.042, -0.025] | Δ > −0.01 | 1.0000 | False | not rejected | 2000 (0/0 failed) | 0.913 |
| NI-seedvc-4 | ltc2k | SIM | 0.928 | 0.946 | -0.018 [-0.023, -0.013] | Δ > −0.01 | 0.9986 | False | not rejected | 1800 (0/0 failed) | 0.913 |
| NI-seedvc-5 | vctk2k | UTMOS | 3.829 | 3.787 | 0.042 [-0.017, 0.105] | Δ > −0.1 | 0.0001 | True | non-inferior | 2000 (0/0 failed) | 3.863 |
| NI-seedvc-6 | ltc2k | UTMOS | 4.027 | 4.072 | -0.045 [-0.111, 0.018] | Δ > −0.1 | 0.0512 | False | not rejected | 1800 (0/0 failed) | 3.863 |
| NI-knnvc-1 | vctk2k | capped WER % | 2.51 | 6.23 | -3.73 [-5.57, -2.13] | Δ < +1 | 0.0001 | True | non-inferior | 1020 (0/0 failed) | 1.89 |
| NI-knnvc-2 | ltc2k | capped WER % | 2.37 | 3.79 | -1.42 [-1.95, -0.93] | Δ < +1 | 0.0001 | True | non-inferior | 1800 (0/0 failed) | 1.89 |
| NI-knnvc-3 | vctk2k | SIM | 0.886 | 0.895 | -0.009 [-0.028, 0.009] | Δ > −0.01 | 0.4407 | False | not rejected | 2000 (0/0 failed) | 0.913 |
| NI-knnvc-4 | ltc2k | SIM | 0.928 | 0.929 | -0.001 [-0.007, 0.006] | Δ > −0.01 | 0.0015 | True | non-inferior | 1800 (0/0 failed) | 0.913 |
| NI-knnvc-5 | vctk2k | UTMOS | 3.829 | 3.664 | 0.165 [0.092, 0.232] | Δ > −0.1 | 0.0001 | True | non-inferior | 2000 (0/0 failed) | 3.863 |
| NI-knnvc-6 | ltc2k | UTMOS | 4.027 | 3.837 | 0.190 [0.143, 0.236] | Δ > −0.1 | 0.0001 | True | non-inferior | 1800 (0/0 failed) | 3.863 |

## Family RB: mvc_v1 at 3 s and 180 s − mvc_v1 at 30 s

Holm over 12, α 0.05. LTC-2k at 180 s and its 30 s comparator on the pairs without a reference shortfall.

| Hypothesis | Set | Endpoint | A | B | Δ = A − B [95 % CI] | p | Holm decision | n pairs | Wilcoxon p (r_rb) |
|---|---|---|---|---|---|---|---|---|---|
| RB-3-1 | vctk2k | capped WER % | 2.20 | 2.51 | -0.31 [-0.90, 0.30] | 0.3034 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.1079 (-0.231) |
| RB-3-2 | ltc2k | capped WER % | 2.41 | 2.37 | 0.03 [-0.21, 0.28] | 0.7593 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.5550 (0.047) |
| RB-3-3 | vctk2k | SIM | 0.844 | 0.886 | -0.042 [-0.051, -0.033] | 0.0002 | reject, favours mvc_v1 30 s | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.827) |
| RB-3-4 | ltc2k | SIM | 0.894 | 0.928 | -0.035 [-0.046, -0.025] | 0.0002 | reject, favours mvc_v1 30 s | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.844) |
| RB-3-5 | vctk2k | UTMOS | 3.797 | 3.829 | -0.031 [-0.087, 0.022] | 0.2704 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0005 (-0.090) |
| RB-3-6 | ltc2k | UTMOS | 3.950 | 4.027 | -0.077 [-0.109, -0.045] | 0.0002 | reject, favours mvc_v1 30 s | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.342) |
| RB-180-1 | vctk2k | capped WER % | 2.46 | 2.51 | -0.04 [-0.55, 0.44] | 0.8915 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.9400 (-0.013) |
| RB-180-2 | ltc2k | capped WER % | 2.35 | 2.42 | -0.07 [-0.25, 0.10] | 0.4378 | not rejected | 1699 (0/0 failed, 0 undefined, 0 redraws) | 0.1156 (-0.149) |
| RB-180-3 | vctk2k | SIM | 0.892 | 0.886 | 0.006 [0.003, 0.009] | 0.0004 | reject, favours mvc_v1 180 s | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.314) |
| RB-180-4 | ltc2k | SIM | 0.935 | 0.928 | 0.007 [0.004, 0.011] | 0.0002 | reject, favours mvc_v1 180 s | 1699 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.506) |
| RB-180-5 | vctk2k | UTMOS | 3.819 | 3.829 | -0.010 [-0.028, 0.010] | 0.3116 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0017 (-0.081) |
| RB-180-6 | ltc2k | UTMOS | 4.044 | 4.018 | 0.026 [0.002, 0.050] | 0.0332 | not rejected | 1699 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.138) |

### RBN (CI rule, no Holm)

| Check | Set | From | Δ [95 % CI] | Rule | Passes |
|---|---|---|---|---|---|
| RBN-1 | vctk2k | RB-3-3 | -0.042 [-0.051, -0.033] | lower limit above -0.06 | True |
| RBN-2 | vctk2k | RB-180-3 | 0.006 [0.003, 0.009] | lower limit above -0.005 | True |
| RBN-3 | vctk2k | RB-3-1 | -0.31 [-0.90, 0.30] | upper limit below +1.5 | True |
| RBN-1 | ltc2k | RB-3-4 | -0.035 [-0.046, -0.025] | lower limit above -0.06 | True |
| RBN-2 | ltc2k | RB-180-4 | 0.007 [0.004, 0.011] | lower limit above -0.005 | True |
| RBN-3 | ltc2k | RB-3-2 | 0.03 [-0.21, 0.28] | upper limit below +1.5 | True |

## Family Q: mvc_v1_q32 − mvc_v1

Holm over 6, α 0.05.

| Hypothesis | Set | Endpoint | A | B | Δ = A − B [95 % CI] | p | Holm decision | n pairs | Wilcoxon p (r_rb) | mvc_v1 on Dev-900, selected on Dev |
|---|---|---|---|---|---|---|---|---|---|---|
| Q-1 | vctk2k | capped WER % | 2.32 | 2.51 | -0.18 [-0.64, 0.15] | 0.3498 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.1650 (-0.294) | 1.89 |
| Q-2 | ltc2k | capped WER % | 2.31 | 2.37 | -0.06 [-0.24, 0.11] | 0.5009 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.1855 (-0.123) | 1.89 |
| Q-3 | vctk2k | SIM | 0.886 | 0.886 | -0.000 [-0.001, 0.001] | 0.8839 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.7493 (-0.008) | 0.913 |
| Q-4 | ltc2k | SIM | 0.928 | 0.928 | -0.001 [-0.002, 0.000] | 0.0776 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.124) | 0.913 |
| Q-5 | vctk2k | UTMOS | 3.852 | 3.829 | 0.024 [0.009, 0.039] | 0.0012 | reject, favours mvc_v1_q32 | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.208) | 3.863 |
| Q-6 | ltc2k | UTMOS | 4.041 | 4.027 | 0.014 [0.003, 0.026] | 0.0134 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.142) | 3.863 |

## Wording against the baselines (§7.3)

| Hypothesis | Set | Endpoint | Wording | Flag |
|---|---|---|---|---|
| S-knnvc-1 | vctk2k | wer | better than knnvc; comparable to knnvc (non-inferiority, Δ < +1) |  |
| S-knnvc-2 | ltc2k | wer | better than knnvc; comparable to knnvc (non-inferiority, Δ < +1) |  |
| S-knnvc-3 | vctk2k | sim | no claim; the CI is given |  |
| S-knnvc-4 | ltc2k | sim | comparable to knnvc (non-inferiority, Δ > −0.01) |  |
| S-knnvc-5 | vctk2k | utmos | better than knnvc; comparable to knnvc (non-inferiority, Δ > −0.1) |  |
| S-knnvc-6 | ltc2k | utmos | better than knnvc; comparable to knnvc (non-inferiority, Δ > −0.1) | not confirmed by a second naturalness predictor |
| S-freevc-1 | vctk2k | wer | no claim; the CI is given |  |
| S-freevc-2 | ltc2k | wer | no claim; the CI is given |  |
| S-freevc-3 | vctk2k | sim | no claim; the CI is given |  |
| S-freevc-4 | ltc2k | sim | better than freevc |  |
| S-freevc-5 | vctk2k | utmos | worse than freevc |  |
| S-freevc-6 | ltc2k | utmos | no claim; the CI is given |  |
| S-seedvc-1 | vctk2k | wer | comparable to seedvc (non-inferiority, Δ < +1) |  |
| S-seedvc-2 | ltc2k | wer | comparable to seedvc (non-inferiority, Δ < +1) |  |
| S-seedvc-3 | vctk2k | sim | worse than seedvc |  |
| S-seedvc-4 | ltc2k | sim | worse than seedvc |  |
| S-seedvc-5 | vctk2k | utmos | comparable to seedvc (non-inferiority, Δ > −0.1) |  |
| S-seedvc-6 | ltc2k | utmos | no claim; the CI is given |  |
| S-facodec-1 | vctk2k | wer | no claim; the CI is given |  |
| S-facodec-2 | ltc2k | wer | no claim; the CI is given |  |
| S-facodec-3 | vctk2k | sim | no claim; the CI is given |  |
| S-facodec-4 | ltc2k | sim | better than facodec |  |
| S-facodec-5 | vctk2k | utmos | better than facodec |  |
| S-facodec-6 | ltc2k | utmos | better than facodec |  |

## Power with the realised SEs (§5.5)

Multipliers: H1 3.48, Q 3.48, S 3.92, NI 3.71, RB 3.71, NL 2.80, L 2.80.

| Family | Hypothesis | SE | MDE | 80 % power needs |
|---|---|---|---|---|
| H1 | H1-1 | 1.7562 | 6.1112 | abs(Δ) ≥ 6.1112 |
| H1 | H1-2 | 0.8204 | 2.8550 | abs(Δ) ≥ 2.8550 |
| H1 | H1-3 | 0.0145 | 0.0505 | abs(Δ) ≥ 0.0505 |
| H1 | H1-4 | 0.0042 | 0.0146 | abs(Δ) ≥ 0.0146 |
| H1 | H1-5 | 0.0354 | 0.1232 | abs(Δ) ≥ 0.1232 |
| H1 | H1-6 | 0.0401 | 0.1394 | abs(Δ) ≥ 0.1394 |
| S | S-knnvc-1 | 0.8726 | 3.4203 | abs(Δ) ≥ 3.4203 |
| S | S-knnvc-2 | 0.2622 | 1.0278 | abs(Δ) ≥ 1.0278 |
| S | S-knnvc-3 | 0.0096 | 0.0375 | abs(Δ) ≥ 0.0375 |
| S | S-knnvc-4 | 0.0033 | 0.0128 | abs(Δ) ≥ 0.0128 |
| S | S-knnvc-5 | 0.0358 | 0.1405 | abs(Δ) ≥ 0.1405 |
| S | S-knnvc-6 | 0.0239 | 0.0936 | abs(Δ) ≥ 0.0936 |
| S | S-freevc-1 | 0.4150 | 1.6268 | abs(Δ) ≥ 1.6268 |
| S | S-freevc-2 | 0.2435 | 0.9543 | abs(Δ) ≥ 0.9543 |
| S | S-freevc-3 | 0.0077 | 0.0304 | abs(Δ) ≥ 0.0304 |
| S | S-freevc-4 | 0.0076 | 0.0296 | abs(Δ) ≥ 0.0296 |
| S | S-freevc-5 | 0.0364 | 0.1426 | abs(Δ) ≥ 0.1426 |
| S | S-freevc-6 | 0.0303 | 0.1189 | abs(Δ) ≥ 0.1189 |
| S | S-seedvc-1 | 0.4066 | 1.5938 | abs(Δ) ≥ 1.5938 |
| S | S-seedvc-2 | 0.1830 | 0.7172 | abs(Δ) ≥ 0.7172 |
| S | S-seedvc-3 | 0.0045 | 0.0178 | abs(Δ) ≥ 0.0178 |
| S | S-seedvc-4 | 0.0026 | 0.0103 | abs(Δ) ≥ 0.0103 |
| S | S-seedvc-5 | 0.0308 | 0.1208 | abs(Δ) ≥ 0.1208 |
| S | S-seedvc-6 | 0.0328 | 0.1285 | abs(Δ) ≥ 0.1285 |
| S | S-facodec-1 | 0.4535 | 1.7775 | abs(Δ) ≥ 1.7775 |
| S | S-facodec-2 | 0.2004 | 0.7857 | abs(Δ) ≥ 0.7857 |
| S | S-facodec-3 | 0.0055 | 0.0216 | abs(Δ) ≥ 0.0216 |
| S | S-facodec-4 | 0.0055 | 0.0214 | abs(Δ) ≥ 0.0214 |
| S | S-facodec-5 | 0.0439 | 0.1719 | abs(Δ) ≥ 0.1719 |
| S | S-facodec-6 | 0.0378 | 0.1482 | abs(Δ) ≥ 0.1482 |
| RB | RB-3-1 | 0.3003 | 1.1130 | abs(Δ) ≥ 1.1130 |
| RB | RB-3-2 | 0.1268 | 0.4701 | abs(Δ) ≥ 0.4701 |
| RB | RB-3-3 | 0.0047 | 0.0174 | abs(Δ) ≥ 0.0174 |
| RB | RB-3-4 | 0.0055 | 0.0205 | abs(Δ) ≥ 0.0205 |
| RB | RB-3-5 | 0.0280 | 0.1037 | abs(Δ) ≥ 0.1037 |
| RB | RB-3-6 | 0.0161 | 0.0598 | abs(Δ) ≥ 0.0598 |
| RB | RB-180-1 | 0.2442 | 0.9052 | abs(Δ) ≥ 0.9052 |
| RB | RB-180-2 | 0.0873 | 0.3237 | abs(Δ) ≥ 0.3237 |
| RB | RB-180-3 | 0.0016 | 0.0060 | abs(Δ) ≥ 0.0060 |
| RB | RB-180-4 | 0.0017 | 0.0062 | abs(Δ) ≥ 0.0062 |
| RB | RB-180-5 | 0.0097 | 0.0359 | abs(Δ) ≥ 0.0359 |
| RB | RB-180-6 | 0.0122 | 0.0452 | abs(Δ) ≥ 0.0452 |
| Q | Q-1 | 0.2039 | 0.7096 | abs(Δ) ≥ 0.7096 |
| Q | Q-2 | 0.0915 | 0.3185 | abs(Δ) ≥ 0.3185 |
| Q | Q-3 | 0.0006 | 0.0022 | abs(Δ) ≥ 0.0022 |
| Q | Q-4 | 0.0004 | 0.0015 | abs(Δ) ≥ 0.0015 |
| Q | Q-5 | 0.0076 | 0.0264 | abs(Δ) ≥ 0.0264 |
| Q | Q-6 | 0.0060 | 0.0207 | abs(Δ) ≥ 0.0207 |
| NI | NI-seedvc-1 | 0.4066 | 1.5073 | Δ ≤ -0.5073 |
| NI | NI-seedvc-2 | 0.1830 | 0.6782 | Δ ≤ +0.3218 |
| NI | NI-seedvc-3 | 0.0045 | 0.0168 | Δ ≥ +0.0068 |
| NI | NI-seedvc-4 | 0.0026 | 0.0098 | Δ ≥ -0.0002 |
| NI | NI-seedvc-5 | 0.0308 | 0.1143 | Δ ≥ +0.0143 |
| NI | NI-seedvc-6 | 0.0328 | 0.1215 | Δ ≥ +0.0215 |
| NI | NI-knnvc-1 | 0.8726 | 3.2346 | Δ ≤ -2.2346 |
| NI | NI-knnvc-2 | 0.2622 | 0.9720 | Δ ≤ +0.0280 |
| NI | NI-knnvc-3 | 0.0096 | 0.0355 | Δ ≥ +0.0255 |
| NI | NI-knnvc-4 | 0.0033 | 0.0121 | Δ ≥ +0.0021 |
| NI | NI-knnvc-5 | 0.0358 | 0.1328 | Δ ≥ +0.0328 |
| NI | NI-knnvc-6 | 0.0239 | 0.0885 | Δ ≥ -0.0115 |
| NL | NL1s | 0.0104 | 0.0292 | true value ≤ +0.0008 |
| NL | NL1c | 0.0148 | 0.0415 | true value ≤ -0.0115 |
| L | L1s | 0.0107 | 0.0300 | true value ≤ -0.0200 |
| L | L1c | 0.0209 | 0.0585 | true value ≤ -0.0485 |
| L | L2s | 0.0069 | 0.0194 | true value ≤ -0.0094 |
| L | L2c | 0.0085 | 0.0239 | true value ≤ -0.0139 |

## Sensitivity analyses (§5.6)

- Uncapped corpus WER, family H1: decisions that change: none.
- Uncapped corpus WER, family S: decisions that change: none.
- Uncapped corpus WER, family NI: decisions that change: none.
- Uncapped corpus WER, family RB: decisions that change: none.
- Uncapped corpus WER, family Q: decisions that change: none.
- LTC-2k with the independent bootstrap, family H1: decisions that change: none.
- LTC-2k with the independent bootstrap, family S: decisions that change: none.
- LTC-2k with the independent bootstrap, family NI: decisions that change: NI-knnvc-4, NI-seedvc-1.
- LTC-2k with the independent bootstrap, family RB: decisions that change: none.
- LTC-2k with the independent bootstrap, family Q: decisions that change: none.
- LTC-2k with the independent bootstrap, L: L2s True, L2c True.
- Failed conversions at their worst case, family H1: computed for no hypothesis (equal failure counts); decisions that change: none.
- Failed conversions at their worst case, family S: computed for no hypothesis (equal failure counts); decisions that change: none.
- Failed conversions at their worst case, family RB: computed for no hypothesis (equal failure counts); decisions that change: none.
- Failed conversions at their worst case, family Q: computed for no hypothesis (equal failure counts); decisions that change: none.
- Failed conversions at their worst case, family NI: computed for no hypothesis (equal failure counts); decisions that change: none.
- Worst case, NL: {'NL1s': False, 'NL1c': False}; L: {'L1s': True, 'L1c': False, 'L2s': True, 'L2c': True} (all four False).
- H1 ∪ S, Holm over 30: decisions that change: none.
- Outcome with uncapped WER: A; with LTC-2k independent: A.
- NL beside the summarize.py value: in the NL table (descriptive; it decides nothing).

## Second speaker model: ECAPA (§6.1)

| Comparison | Set | mvc_v1 | B | Δ SIM [95 % CI] | p | n |
|---|---|---|---|---|---|---|
| E-knnvc-vctk2k | vctk2k | 0.456 | 0.647 | -0.191 [-0.220, -0.164] | 0.0002 | 2000 |
| E-freevc-vctk2k | vctk2k | 0.456 | 0.590 | -0.134 [-0.174, -0.097] | 0.0002 | 2000 |
| E-seedvc-vctk2k | vctk2k | 0.456 | 0.727 | -0.271 [-0.295, -0.247] | 0.0002 | 2000 |
| E-facodec-vctk2k | vctk2k | 0.456 | 0.482 | -0.025 [-0.053, 0.005] | 0.0918 | 2000 |
| E-cand_pm18_rq-vctk2k | vctk2k | 0.456 | 0.161 | 0.295 [0.259, 0.332] | 0.0002 | 2000 |
| E-knnvc-ltc2k | ltc2k | 0.631 | 0.665 | -0.034 [-0.053, -0.014] | 0.0006 | 1800 |
| E-freevc-ltc2k | ltc2k | 0.631 | 0.475 | 0.156 [0.126, 0.187] | 0.0002 | 1800 |
| E-seedvc-ltc2k | ltc2k | 0.631 | 0.743 | -0.113 [-0.130, -0.096] | 0.0002 | 1800 |
| E-facodec-ltc2k | ltc2k | 0.631 | 0.569 | 0.062 [0.045, 0.080] | 0.0002 | 1800 |
| E-cand_pm18_rq-ltc2k | ltc2k | 0.631 | 0.455 | 0.176 [0.153, 0.199] | 0.0002 | 1800 |

| Comparison | Set / stratum | Δ SIM_src [95 % CI] | p | n |
|---|---|---|---|---|
| Esrc-knnvc-vctk2k-same | vctk2k / same | -0.081 [-0.104, -0.055] | 0.0002 | 1000 |
| Esrc-knnvc-vctk2k-cross | vctk2k / cross | -0.020 [-0.046, 0.004] | 0.1100 | 1000 |
| Esrc-freevc-vctk2k-same | vctk2k / same | -0.144 [-0.172, -0.112] | 0.0002 | 1000 |
| Esrc-freevc-vctk2k-cross | vctk2k / cross | -0.053 [-0.077, -0.029] | 0.0002 | 1000 |
| Esrc-seedvc-vctk2k-same | vctk2k / same | -0.062 [-0.082, -0.041] | 0.0002 | 1000 |
| Esrc-seedvc-vctk2k-cross | vctk2k / cross | -0.007 [-0.027, 0.012] | 0.4894 | 1000 |
| Esrc-facodec-vctk2k-same | vctk2k / same | -0.180 [-0.218, -0.141] | 0.0002 | 1000 |
| Esrc-facodec-vctk2k-cross | vctk2k / cross | -0.075 [-0.094, -0.055] | 0.0002 | 1000 |
| Esrc-cand_pm18_rq-vctk2k-same | vctk2k / same | 0.037 [0.008, 0.067] | 0.0112 | 1000 |
| Esrc-cand_pm18_rq-vctk2k-cross | vctk2k / cross | 0.057 [0.039, 0.076] | 0.0002 | 1000 |
| Esrc-knnvc-ltc2k-same | ltc2k / same | -0.081 [-0.103, -0.058] | 0.0002 | 900 |
| Esrc-knnvc-ltc2k-cross | ltc2k / cross | -0.026 [-0.040, -0.012] | 0.0012 | 900 |
| Esrc-freevc-ltc2k-same | ltc2k / same | -0.079 [-0.103, -0.052] | 0.0002 | 900 |
| Esrc-freevc-ltc2k-cross | ltc2k / cross | -0.037 [-0.053, -0.019] | 0.0002 | 900 |
| Esrc-seedvc-ltc2k-same | ltc2k / same | -0.083 [-0.103, -0.063] | 0.0002 | 900 |
| Esrc-seedvc-ltc2k-cross | ltc2k / cross | -0.043 [-0.054, -0.031] | 0.0002 | 900 |
| Esrc-facodec-ltc2k-same | ltc2k / same | -0.148 [-0.168, -0.127] | 0.0002 | 900 |
| Esrc-facodec-ltc2k-cross | ltc2k / cross | -0.051 [-0.068, -0.035] | 0.0002 | 900 |
| Esrc-cand_pm18_rq-ltc2k-same | ltc2k / same | -0.001 [-0.019, 0.018] | 0.9513 | 900 |
| Esrc-cand_pm18_rq-ltc2k-cross | ltc2k / cross | -0.008 [-0.021, 0.006] | 0.2660 | 900 |

| ECAPA coverage, VCTK-2k pairs with P, 30 s | Coverage [95 % CI] | E_out | E_src | E_tgt | n |
|---|---|---|---|---|---|
| mvc_v1 | 0.522 [0.475, 0.567] | 0.473 | 0.122 | 0.795 | 980 |
| mvc_v1_q32 | 0.523 [0.476, 0.568] | 0.474 | 0.122 | 0.795 | 980 |
| mvc_copy_latent | 0.005 [-0.005, 0.014] | 0.125 | 0.122 | 0.795 | 980 |
| cand_pm18_rq | 0.070 [-0.002, 0.142] | 0.169 | 0.122 | 0.795 | 980 |
| knnvc | 0.834 [0.804, 0.861] | 0.683 | 0.122 | 0.795 | 980 |
| freevc | 0.747 [0.683, 0.807] | 0.625 | 0.122 | 0.795 | 980 |
| seedvc | 0.943 [0.923, 0.963] | 0.756 | 0.122 | 0.795 | 980 |
| facodec | 0.563 [0.526, 0.600] | 0.501 | 0.122 | 0.795 | 980 |
| faiss | 0.715 [0.675, 0.752] | 0.603 | 0.122 | 0.795 | 980 |

| Set | values | Spearman ρ | Kendall τ of system means |
|---|---|---|---|
| vctk2k | 20980 | 0.724 | 0.782 |
| ltc2k | 18000 | 0.732 | 0.867 |

## Second naturalness predictor: DNSMOS (§6.2, descriptive)

| Comparison | Set | mvc_v1 | B | Δ [95 % CI] | n |
|---|---|---|---|---|---|
| D-OVRL-knnvc-vctk2k | vctk2k | 3.284 | 3.208 | 0.076 [0.046, 0.105] | 2000 |
| D-OVRL-freevc-vctk2k | vctk2k | 3.284 | 3.180 | 0.104 [0.068, 0.142] | 2000 |
| D-OVRL-seedvc-vctk2k | vctk2k | 3.284 | 3.234 | 0.049 [0.029, 0.069] | 2000 |
| D-OVRL-facodec-vctk2k | vctk2k | 3.284 | 3.235 | 0.049 [0.018, 0.081] | 2000 |
| D-OVRL-cand_pm18_rq-vctk2k | vctk2k | 3.284 | 2.825 | 0.459 [0.428, 0.491] | 2000 |
| D-OVRL-knnvc-ltc2k | ltc2k | 3.314 | 3.353 | -0.039 [-0.056, -0.022] | 1800 |
| D-OVRL-freevc-ltc2k | ltc2k | 3.314 | 3.315 | -0.001 [-0.027, 0.025] | 1800 |
| D-OVRL-seedvc-ltc2k | ltc2k | 3.314 | 3.345 | -0.031 [-0.052, -0.011] | 1800 |
| D-OVRL-facodec-ltc2k | ltc2k | 3.314 | 3.323 | -0.009 [-0.031, 0.012] | 1800 |
| D-OVRL-cand_pm18_rq-ltc2k | ltc2k | 3.314 | 2.915 | 0.399 [0.360, 0.438] | 1800 |
| D-P808-knnvc-vctk2k | vctk2k | 3.793 | 3.695 | 0.098 [0.057, 0.133] | 2000 |
| D-P808-freevc-vctk2k | vctk2k | 3.793 | 3.775 | 0.018 [-0.017, 0.052] | 2000 |
| D-P808-seedvc-vctk2k | vctk2k | 3.793 | 3.731 | 0.062 [0.023, 0.105] | 2000 |
| D-P808-facodec-vctk2k | vctk2k | 3.793 | 3.677 | 0.116 [0.082, 0.149] | 2000 |
| D-P808-cand_pm18_rq-vctk2k | vctk2k | 3.793 | 3.244 | 0.549 [0.497, 0.600] | 2000 |
| D-P808-knnvc-ltc2k | ltc2k | 3.865 | 3.958 | -0.094 [-0.121, -0.067] | 1800 |
| D-P808-freevc-ltc2k | ltc2k | 3.865 | 3.916 | -0.051 [-0.090, -0.015] | 1800 |
| D-P808-seedvc-ltc2k | ltc2k | 3.865 | 3.973 | -0.108 [-0.134, -0.084] | 1800 |
| D-P808-facodec-ltc2k | ltc2k | 3.865 | 3.895 | -0.031 [-0.059, -0.005] | 1800 |
| D-P808-cand_pm18_rq-ltc2k | ltc2k | 3.865 | 3.608 | 0.256 [0.210, 0.300] | 1800 |

## Descriptive (CIs only)

### Levels, vctk2k, 30 s

| System | rows (failed) | capped WER % | WER % | capped WER % base | WER % base | capped CER % | SIM | SECS | UTMOS | SIM_src | SECS_src | M9 D | share D > 0 | EER | dB-MCD | log-F0 RMSE | V/UV error | F0 PCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| faiss | 2000 (0) | 89.38 [87.30, 91.52] | 132.24 [111.93, 158.26] | 93.70 [92.12, 95.13] | 168.24 [138.03, 206.68] | 72.51 [69.98, 75.06] | 0.899 [0.888, 0.910] | 0.816 [0.807, 0.825] | 1.834 [1.771, 1.900] | 0.690 [0.659, 0.720] | 0.624 [0.609, 0.637] | -0.210 [-0.240, -0.180] | 0.044 | 0.159 | 8.62 [8.49, 8.74] | 0.219 [0.203, 0.237] | 0.144 [0.125, 0.166] | 0.124 [0.097, 0.151] |
| cand_pm18_rq | 2000 (0) | 16.26 [12.70, 20.16] | 16.53 [12.77, 20.67] | 22.13 [17.89, 26.70] | 22.42 [17.99, 27.13] | 9.79 [7.45, 12.34] | 0.806 [0.778, 0.831] | 0.690 [0.672, 0.708] | 2.628 [2.553, 2.708] | 0.667 [0.632, 0.701] | 0.596 [0.583, 0.611] | -0.139 [-0.180, -0.098] | 0.245 | 0.278 | 9.02 [8.85, 9.19] | 0.194 [0.182, 0.207] | 0.128 [0.110, 0.147] | 0.348 [0.311, 0.388] |
| knnvc | 2000 (0) | 6.23 [4.28, 8.37] | 6.26 [4.29, 8.40] | 13.50 [10.60, 16.50] | 13.63 [10.68, 16.68] | 3.76 [2.43, 5.28] | 0.895 [0.872, 0.914] | 0.806 [0.790, 0.820] | 3.664 [3.600, 3.731] | 0.736 [0.698, 0.771] | 0.629 [0.614, 0.644] | -0.159 [-0.205, -0.115] | 0.144 | 0.194 | 7.43 [7.30, 7.55] | 0.181 [0.165, 0.198] | 0.109 [0.096, 0.123] | 0.342 [0.302, 0.382] |
| freevc [VCTK-seen] | 2000 (0) | 2.03 [1.16, 3.04] | 2.03 [1.16, 3.04] | 6.81 [4.77, 9.05] | 6.82 [4.77, 9.09] | 1.07 [0.56, 1.66] | 0.887 [0.864, 0.907] | 0.842 [0.831, 0.853] | 4.001 [3.946, 4.051] | 0.770 [0.732, 0.806] | 0.670 [0.653, 0.687] | -0.117 [-0.165, -0.068] | 0.228 | 0.222 | 7.04 [6.91, 7.16] | 0.184 [0.170, 0.199] | 0.112 [0.100, 0.125] | 0.422 [0.381, 0.464] |
| seedvc | 2000 (0) | 2.59 [1.56, 3.82] | 2.59 [1.56, 3.82] | 6.26 [4.57, 8.01] | 6.27 [4.58, 8.03] | 1.36 [0.81, 2.01] | 0.919 [0.909, 0.929] | 0.886 [0.880, 0.893] | 3.787 [3.720, 3.850] | 0.741 [0.708, 0.773] | 0.651 [0.634, 0.666] | -0.178 [-0.213, -0.143] | 0.091 | 0.140 | 7.15 [6.98, 7.30] | 0.173 [0.157, 0.190] | 0.106 [0.093, 0.120] | 0.444 [0.402, 0.486] |
| facodec | 2000 (0) | 2.45 [1.50, 3.56] | 2.45 [1.50, 3.56] | 7.59 [5.15, 10.38] | 7.63 [5.16, 10.47] | 1.26 [0.72, 1.93] | 0.876 [0.860, 0.892] | 0.798 [0.785, 0.811] | 3.628 [3.540, 3.714] | 0.777 [0.744, 0.809] | 0.700 [0.679, 0.720] | -0.099 [-0.136, -0.063] | 0.303 | 0.231 | 6.76 [6.64, 6.88] | 0.230 [0.211, 0.249] | 0.113 [0.102, 0.125] | 0.309 [0.259, 0.358] |
| source_copy | 2000 (0) | 0.84 [0.31, 1.49] | 0.84 [0.31, 1.49] | 4.41 [3.12, 5.75] | 4.41 [3.12, 5.75] | 0.41 [0.14, 0.75] | 0.677 [0.645, 0.710] | 0.630 [0.616, 0.643] | 4.045 [3.965, 4.115] | 0.953 [0.946, 0.959] | 0.895 [0.889, 0.902] | 0.276 [0.244, 0.305] | 0.993 | n/a | 8.87 [8.70, 9.04] | 0.192 [0.178, 0.207] | 0.121 [0.106, 0.137] | 0.386 [0.340, 0.430] |
| target_copy | 980 (0) | n/a | n/a | n/a | n/a | n/a | 0.963 [0.957, 0.969] | 0.901 [0.892, 0.909] | 4.083 [4.013, 4.149] | 0.694 [0.658, 0.728] | 0.636 [0.620, 0.652] | -0.269 [-0.307, -0.234] | 0.002 | n/a | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| mimi_copy_src | 2000 (0) | 1.12 [0.53, 1.78] | 1.12 [0.53, 1.78] | 5.11 [3.44, 6.79] | 5.11 [3.44, 6.79] | 0.52 [0.22, 0.89] | 0.668 [0.634, 0.703] | 0.638 [0.623, 0.653] | 3.800 [3.717, 3.875] | 0.947 [0.939, 0.954] | 0.882 [0.875, 0.888] | 0.279 [0.247, 0.310] | 0.992 | n/a | 8.92 [8.76, 9.08] | 0.185 [0.171, 0.200] | 0.122 [0.108, 0.136] | 0.399 [0.352, 0.444] |
| mimi_copy_P | 980 (0) | n/a | n/a | n/a | n/a | n/a | 0.962 [0.956, 0.967] | 0.894 [0.885, 0.902] | 3.829 [3.753, 3.906] | 0.693 [0.658, 0.727] | 0.643 [0.626, 0.659] | -0.268 [-0.305, -0.234] | 0.004 | n/a | 4.38 [4.27, 4.48] | 0.110 [0.097, 0.124] | 0.082 [0.072, 0.091] | 0.743 [0.698, 0.788] |
| mvc_copy_latent | 2000 (0) | 1.40 [0.75, 2.15] | 1.40 [0.75, 2.15] | 4.76 [3.14, 6.57] | 4.76 [3.14, 6.57] | 0.67 [0.32, 1.07] | 0.670 [0.636, 0.705] | 0.638 [0.623, 0.652] | 3.936 [3.849, 4.014] | 0.948 [0.940, 0.954] | 0.885 [0.879, 0.891] | 0.278 [0.246, 0.309] | 0.993 | 0.499 | 8.87 [8.72, 9.02] | 0.188 [0.175, 0.203] | 0.121 [0.106, 0.135] | 0.389 [0.346, 0.430] |
| mvc_v1 | 2000 (0) | 2.51 [1.63, 3.46] | 2.51 [1.63, 3.46] | 6.08 [4.58, 7.53] | 6.08 [4.58, 7.53] | 1.36 [0.83, 1.93] | 0.886 [0.871, 0.901] | 0.773 [0.759, 0.785] | 3.829 [3.750, 3.902] | 0.707 [0.668, 0.744] | 0.593 [0.574, 0.611] | -0.179 [-0.220, -0.140] | 0.123 | 0.190 | 7.62 [7.49, 7.75] | 0.172 [0.161, 0.185] | 0.100 [0.087, 0.112] | 0.467 [0.425, 0.511] |
| mvc_v1_q32 | 2000 (0) | 2.32 [1.43, 3.29] | 2.32 [1.43, 3.29] | 6.12 [4.50, 7.74] | 6.12 [4.50, 7.74] | 1.27 [0.74, 1.84] | 0.886 [0.871, 0.900] | 0.770 [0.756, 0.783] | 3.852 [3.773, 3.926] | 0.705 [0.666, 0.742] | 0.590 [0.571, 0.608] | -0.181 [-0.221, -0.143] | 0.115 | 0.188 | 7.70 [7.56, 7.84] | 0.171 [0.160, 0.184] | 0.099 [0.087, 0.111] | 0.471 [0.430, 0.515] |

mvc_v1 on Dev-900, selected on Dev: capped WER 1.89, SIM 0.913, UTMOS 3.863, NL same / cross -0.008 / 0.020.

### Levels, ltc2k, 30 s

| System | rows (failed) | capped WER % | WER % | capped WER % base | WER % base | capped CER % | SIM | SECS | UTMOS | SIM_src | SECS_src | M9 D | share D > 0 | EER | dB-MCD | log-F0 RMSE | V/UV error | F0 PCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| faiss | 1800 (0) | 91.27 [89.80, 92.68] | 128.72 [118.08, 141.45] | 96.38 [95.61, 97.07] | 208.94 [187.46, 231.87] | 71.42 [69.46, 73.40] | 0.920 [0.909, 0.929] | 0.864 [0.856, 0.871] | 1.610 [1.573, 1.652] | 0.696 [0.663, 0.729] | 0.691 [0.679, 0.704] | -0.224 [-0.257, -0.191] | 0.047 | 0.119 | n/a | n/a | n/a | n/a |
| cand_pm18_rq | 1800 (0) | 10.33 [8.72, 12.00] | 10.83 [8.99, 12.79] | 17.66 [15.58, 19.74] | 18.18 [15.99, 20.44] | 5.03 [4.27, 5.80] | 0.911 [0.898, 0.923] | 0.798 [0.786, 0.809] | 2.731 [2.662, 2.798] | 0.711 [0.681, 0.743] | 0.666 [0.654, 0.678] | -0.200 [-0.231, -0.168] | 0.109 | 0.190 | n/a | n/a | n/a | n/a |
| knnvc | 1800 (0) | 3.79 [3.04, 4.62] | 4.31 [3.18, 6.03] | 10.41 [8.90, 12.00] | 10.97 [9.28, 12.90] | 1.68 [1.34, 2.08] | 0.929 [0.919, 0.939] | 0.895 [0.888, 0.901] | 3.837 [3.790, 3.884] | 0.735 [0.706, 0.765] | 0.721 [0.708, 0.733] | -0.194 [-0.221, -0.166] | 0.081 | 0.129 | n/a | n/a | n/a | n/a |
| freevc | 1800 (0) | 2.53 [1.91, 3.24] | 3.05 [1.99, 4.75] | 7.28 [6.00, 8.63] | 7.83 [6.34, 9.65] | 1.03 [0.76, 1.34] | 0.897 [0.879, 0.914] | 0.860 [0.850, 0.869] | 4.019 [3.966, 4.070] | 0.758 [0.732, 0.786] | 0.726 [0.714, 0.738] | -0.139 [-0.168, -0.109] | 0.188 | 0.214 | n/a | n/a | n/a | n/a |
| seedvc | 1800 (0) | 2.50 [1.87, 3.21] | 3.02 [1.96, 4.74] | 6.61 [5.44, 7.85] | 7.17 [5.65, 9.09] | 1.06 [0.77, 1.41] | 0.946 [0.936, 0.956] | 0.918 [0.911, 0.924] | 4.072 [4.030, 4.111] | 0.748 [0.720, 0.778] | 0.715 [0.701, 0.729] | -0.198 [-0.227, -0.168] | 0.049 | 0.111 | n/a | n/a | n/a | n/a |
| facodec | 1800 (0) | 2.14 [1.58, 2.74] | 2.67 [1.67, 4.29] | 6.65 [5.52, 7.87] | 7.21 [5.81, 8.99] | 0.75 [0.50, 1.04] | 0.910 [0.893, 0.927] | 0.867 [0.859, 0.875] | 3.580 [3.515, 3.648] | 0.780 [0.755, 0.806] | 0.751 [0.739, 0.762] | -0.130 [-0.158, -0.102] | 0.204 | 0.204 | n/a | n/a | n/a | n/a |
| source_copy | 1800 (0) | 1.66 [1.07, 2.29] | 2.18 [1.15, 3.81] | 5.44 [4.36, 6.60] | 6.00 [4.61, 7.73] | 0.54 [0.32, 0.80] | 0.696 [0.661, 0.733] | 0.694 [0.680, 0.707] | 4.249 [4.206, 4.288] | 0.957 [0.941, 0.967] | 0.915 [0.904, 0.923] | 0.261 [0.220, 0.300] | 0.985 | n/a | n/a | n/a | n/a | n/a |
| mimi_copy_src | 1800 (0) | 1.94 [1.27, 2.67] | 2.46 [1.35, 4.18] | 5.78 [4.62, 7.03] | 6.40 [4.89, 8.40] | 0.71 [0.42, 1.07] | 0.692 [0.656, 0.729] | 0.694 [0.681, 0.707] | 4.156 [4.112, 4.200] | 0.953 [0.936, 0.964] | 0.908 [0.898, 0.916] | 0.261 [0.220, 0.300] | 0.984 | n/a | n/a | n/a | n/a | n/a |
| mvc_copy_latent | 1800 (0) | 1.91 [1.27, 2.62] | 2.43 [1.35, 4.13] | 5.88 [4.74, 7.14] | 6.43 [5.01, 8.30] | 0.67 [0.41, 0.98] | 0.696 [0.660, 0.733] | 0.696 [0.683, 0.710] | 4.159 [4.115, 4.203] | 0.954 [0.937, 0.965] | 0.908 [0.898, 0.917] | 0.258 [0.218, 0.298] | 0.983 | 0.503 | n/a | n/a | n/a | n/a |
| mvc_v1 | 1800 (0) | 2.37 [1.78, 3.03] | 2.89 [1.86, 4.56] | 6.27 [5.04, 7.54] | 6.83 [5.33, 8.63] | 0.92 [0.67, 1.22] | 0.928 [0.917, 0.940] | 0.806 [0.795, 0.816] | 4.027 [3.973, 4.079] | 0.691 [0.661, 0.724] | 0.617 [0.606, 0.629] | -0.237 [-0.268, -0.204] | 0.047 | 0.138 | n/a | n/a | n/a | n/a |
| mvc_v1_q32 | 1800 (0) | 2.31 [1.75, 2.96] | 2.83 [1.82, 4.51] | 6.29 [5.11, 7.52] | 6.84 [5.39, 8.64] | 0.89 [0.63, 1.22] | 0.928 [0.916, 0.939] | 0.801 [0.790, 0.812] | 4.041 [3.992, 4.089] | 0.687 [0.656, 0.720] | 0.612 [0.600, 0.623] | -0.241 [-0.272, -0.207] | 0.045 | 0.138 | n/a | n/a | n/a | n/a |

mvc_v1 on Dev-900, selected on Dev: capped WER 1.89, SIM 0.913, UTMOS 3.863, NL same / cross -0.008 / 0.020.

### SIM_src and M9 per stratum: mvc_v1 − comparator

| Set / stratum | B | Δ SIM_src [95 % CI] | Δ M9 [95 % CI] | share D > 0 mvc_v1 / B |
|---|---|---|---|---|
| vctk2k / same | knnvc | -0.030 [-0.051, -0.009] | -0.026 [-0.059, 0.008] | 0.186 / 0.249 |
| vctk2k / same | freevc | -0.053 [-0.071, -0.038] | -0.051 [-0.079, -0.027] | 0.186 / 0.344 |
| vctk2k / same | seedvc | -0.030 [-0.041, -0.019] | -0.004 [-0.015, 0.007] | 0.186 / 0.161 |
| vctk2k / same | facodec | -0.071 [-0.087, -0.056] | -0.087 [-0.108, -0.069] | 0.186 / 0.520 |
| vctk2k / same | cand_pm18_rq | 0.030 [0.001, 0.059] | -0.055 [-0.099, -0.011] | 0.186 / 0.450 |
| vctk2k / cross | knnvc | -0.028 [-0.069, 0.014] | -0.014 [-0.072, 0.046] | 0.061 / 0.039 |
| vctk2k / cross | freevc | -0.075 [-0.103, -0.046] | -0.074 [-0.115, -0.030] | 0.061 / 0.112 |
| vctk2k / cross | seedvc | -0.038 [-0.060, -0.016] | 0.002 [-0.024, 0.031] | 0.061 / 0.022 |
| vctk2k / cross | facodec | -0.070 [-0.107, -0.032] | -0.073 [-0.118, -0.026] | 0.061 / 0.087 |
| vctk2k / cross | cand_pm18_rq | 0.049 [-0.023, 0.120] | -0.026 [-0.098, 0.047] | 0.061 / 0.040 |
| ltc2k / same | knnvc | -0.034 [-0.049, -0.022] | -0.036 [-0.053, -0.022] | 0.089 / 0.157 |
| ltc2k / same | freevc | -0.058 [-0.075, -0.042] | -0.089 [-0.116, -0.067] | 0.089 / 0.353 |
| ltc2k / same | seedvc | -0.038 [-0.049, -0.027] | -0.025 [-0.038, -0.013] | 0.089 / 0.091 |
| ltc2k / same | facodec | -0.073 [-0.090, -0.057] | -0.092 [-0.117, -0.070] | 0.089 / 0.383 |
| ltc2k / same | cand_pm18_rq | -0.008 [-0.023, 0.008] | -0.030 [-0.045, -0.013] | 0.089 / 0.200 |
| ltc2k / cross | knnvc | -0.053 [-0.070, -0.037] | -0.050 [-0.065, -0.034] | 0.004 / 0.004 |
| ltc2k / cross | freevc | -0.076 [-0.101, -0.055] | -0.107 [-0.142, -0.077] | 0.004 / 0.022 |
| ltc2k / cross | seedvc | -0.076 [-0.092, -0.060] | -0.054 [-0.070, -0.037] | 0.004 / 0.008 |
| ltc2k / cross | facodec | -0.104 [-0.127, -0.083] | -0.122 [-0.152, -0.093] | 0.004 / 0.026 |
| ltc2k / cross | cand_pm18_rq | -0.032 [-0.050, -0.014] | -0.045 [-0.063, -0.026] | 0.004 / 0.018 |

### mvc_v1 − each baseline at 3 s (descriptive; Holm over 24 (guide only, descriptive))

| Hypothesis | Set | Endpoint | A | B | Δ = A − B [95 % CI] | p | Holm decision | n pairs | Wilcoxon p (r_rb) |
|---|---|---|---|---|---|---|---|---|---|
| D3-knnvc-1 | vctk2k | capped WER % | 2.20 | 60.27 | -58.07 [-64.49, -51.25] | 0.0002 | reject, favours mvc_v1 | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.994) |
| D3-knnvc-2 | ltc2k | capped WER % | 2.41 | 34.57 | -32.16 [-37.04, -27.65] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.997) |
| D3-knnvc-3 | vctk2k | SIM | 0.844 | 0.821 | 0.023 [-0.008, 0.053] | 0.1424 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.274) |
| D3-knnvc-4 | ltc2k | SIM | 0.894 | 0.820 | 0.073 [0.049, 0.097] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.702) |
| D3-knnvc-5 | vctk2k | UTMOS | 3.797 | 2.460 | 1.338 [1.160, 1.531] | 0.0002 | reject, favours mvc_v1 | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.998) |
| D3-knnvc-6 | ltc2k | UTMOS | 3.950 | 2.538 | 1.412 [1.340, 1.490] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (1.000) |
| D3-freevc-1 [VCTK-seen] | vctk2k | capped WER % | 2.20 | 2.32 | -0.13 [-1.22, 0.80] | 0.8419 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.8463 (-0.019) |
| D3-freevc-2 | ltc2k | capped WER % | 2.41 | 2.56 | -0.16 [-0.58, 0.27] | 0.4624 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0509 (-0.103) |
| D3-freevc-3 [VCTK-seen] | vctk2k | SIM | 0.844 | 0.882 | -0.038 [-0.057, -0.020] | 0.0002 | reject, favours freevc | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.516) |
| D3-freevc-4 | ltc2k | SIM | 0.894 | 0.881 | 0.013 [-0.006, 0.034] | 0.1894 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.166) |
| D3-freevc-5 [VCTK-seen] | vctk2k | UTMOS | 3.797 | 4.007 | -0.209 [-0.288, -0.135] | 0.0002 | reject, favours freevc | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.565) |
| D3-freevc-6 | ltc2k | UTMOS | 3.950 | 3.998 | -0.048 [-0.112, 0.016] | 0.1520 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.155) |
| D3-seedvc-1 | vctk2k | capped WER % | 2.20 | 2.59 | -0.39 [-1.15, 0.40] | 0.3190 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.0903 (-0.189) |
| D3-seedvc-2 | ltc2k | capped WER % | 2.41 | 2.37 | 0.04 [-0.32, 0.39] | 0.8057 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.9469 (0.004) |
| D3-seedvc-3 | vctk2k | SIM | 0.844 | 0.898 | -0.054 [-0.068, -0.041] | 0.0002 | reject, favours seedvc | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.783) |
| D3-seedvc-4 | ltc2k | SIM | 0.894 | 0.923 | -0.029 [-0.042, -0.019] | 0.0002 | reject, favours seedvc | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.575) |
| D3-seedvc-5 | vctk2k | UTMOS | 3.797 | 3.791 | 0.007 [-0.081, 0.093] | 0.8703 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0169 (0.062) |
| D3-seedvc-6 | ltc2k | UTMOS | 3.950 | 4.035 | -0.085 [-0.139, -0.035] | 0.0008 | reject, favours seedvc | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.273) |
| D3-facodec-1 | vctk2k | capped WER % | 2.20 | 2.63 | -0.43 [-1.28, 0.29] | 0.2706 | not rejected | 1020 (0/0 failed, 0 undefined, 0 redraws) | 0.3967 (-0.086) |
| D3-facodec-2 | ltc2k | capped WER % | 2.41 | 2.22 | 0.19 [-0.23, 0.59] | 0.3550 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0388 (0.116) |
| D3-facodec-3 | vctk2k | SIM | 0.844 | 0.865 | -0.021 [-0.038, -0.005] | 0.0078 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (-0.275) |
| D3-facodec-4 | ltc2k | SIM | 0.894 | 0.899 | -0.005 [-0.021, 0.010] | 0.5127 | not rejected | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.1043 (-0.044) |
| D3-facodec-5 | vctk2k | UTMOS | 3.797 | 3.767 | 0.031 [-0.053, 0.107] | 0.4334 | not rejected | 2000 (0/0 failed, 0 undefined, 0 redraws) | 0.0001 (0.105) |
| D3-facodec-6 | ltc2k | UTMOS | 3.950 | 3.754 | 0.196 [0.129, 0.257] | 0.0002 | reject, favours mvc_v1 | 1800 (0/0 failed, 0 undefined, 0 redraws) | 0.0000 (0.477) |

### mvc_v1 − kNN-VC at 180 s (CIs only)

| Id | Set | Endpoint | A | B | Δ [95 % CI] | n (failed A/B) |
|---|---|---|---|---|---|---|
| D180-knnvc-1 | vctk2k | capped WER % | 2.46 | 2.89 | -0.42 [-1.54, 0.68] | 1020 (0/0) |
| D180-knnvc-2 | ltc2k | capped WER % | 2.35 | 2.65 | -0.30 [-0.65, 0.03] | 1699 (0/0) |
| D180-knnvc-3 | vctk2k | SIM | 0.892 | 0.901 | -0.009 [-0.025, 0.006] | 2000 (0/0) |
| D180-knnvc-4 | ltc2k | SIM | 0.935 | 0.935 | 0.000 [-0.006, 0.007] | 1699 (0/0) |
| D180-knnvc-5 | vctk2k | UTMOS | 3.819 | 3.786 | 0.033 [-0.032, 0.099] | 2000 (0/0) |
| D180-knnvc-6 | ltc2k | UTMOS | 4.044 | 3.967 | 0.077 [0.032, 0.121] | 1699 (0/0) |

### mvc_v1 − CRT-AL at 3 s and 180 s (CIs only)

| Id | Set | Endpoint | A | B | Δ [95 % CI] | n (failed A/B) |
|---|---|---|---|---|---|---|
| D3-crt-1 | vctk2k | capped WER % | 2.20 | 16.13 | -13.94 [-17.94, -10.47] | 1020 (0/0) |
| D3-crt-2 | ltc2k | capped WER % | 2.41 | 10.51 | -8.10 [-9.66, -6.68] | 1800 (0/0) |
| D3-crt-3 | vctk2k | SIM | 0.844 | 0.799 | 0.045 [0.011, 0.077] | 2000 (0/0) |
| D3-crt-4 | ltc2k | SIM | 0.894 | 0.894 | -0.000 [-0.016, 0.015] | 1800 (0/0) |
| D3-crt-5 | vctk2k | UTMOS | 3.797 | 2.627 | 1.170 [1.079, 1.259] | 2000 (0/0) |
| D3-crt-6 | ltc2k | UTMOS | 3.950 | 2.703 | 1.247 [1.151, 1.344] | 1800 (0/0) |
| D180-crt-1 | vctk2k | capped WER % | 2.46 | 16.23 | -13.77 [-17.53, -10.35] | 1020 (0/0) |
| D180-crt-2 | ltc2k | capped WER % | 2.35 | 10.46 | -8.11 [-9.71, -6.63] | 1699 (0/0) |
| D180-crt-3 | vctk2k | SIM | 0.892 | 0.807 | 0.085 [0.058, 0.114] | 2000 (0/0) |
| D180-crt-4 | ltc2k | SIM | 0.935 | 0.909 | 0.026 [0.017, 0.035] | 1699 (0/0) |
| D180-crt-5 | vctk2k | UTMOS | 3.819 | 2.629 | 1.190 [1.120, 1.262] | 2000 (0/0) |
| D180-crt-6 | ltc2k | UTMOS | 4.044 | 2.726 | 1.318 [1.230, 1.403] | 1699 (0/0) |

### mvc_v1 − mvc_copy_latent, 30 s (CIs only)

| Id | Set | Endpoint | A | B | Δ [95 % CI] | n (failed A/B) |
|---|---|---|---|---|---|---|
| DC-1 | vctk2k | capped WER % | 2.51 | 1.40 | 1.11 [0.32, 1.94] | 1020 (0/0) |
| DC-2 | ltc2k | capped WER % | 2.37 | 1.91 | 0.46 [0.03, 0.89] | 1800 (0/0) |
| DC-3 | vctk2k | SIM | 0.886 | 0.670 | 0.216 [0.184, 0.250] | 2000 (0/0) |
| DC-4 | ltc2k | SIM | 0.928 | 0.696 | 0.232 [0.199, 0.266] | 1800 (0/0) |
| DC-5 | vctk2k | UTMOS | 3.829 | 3.936 | -0.107 [-0.205, -0.007] | 2000 (0/0) |
| DC-6 | ltc2k | UTMOS | 4.027 | 4.159 | -0.133 [-0.188, -0.081] | 1800 (0/0) |

### mvc_copy_latent − mimi_copy_src, 30 s (CIs only)

| Id | Set | Endpoint | A | B | Δ [95 % CI] | n (failed A/B) |
|---|---|---|---|---|---|---|
| DM-1 | vctk2k | capped WER % | 1.40 | 1.12 | 0.28 [-0.15, 0.78] | 1020 (0/0) |
| DM-2 | ltc2k | capped WER % | 1.91 | 1.94 | -0.03 [-0.26, 0.20] | 1800 (0/0) |
| DM-3 | vctk2k | SIM | 0.670 | 0.668 | 0.002 [-0.001, 0.004] | 2000 (0/0) |
| DM-4 | ltc2k | SIM | 0.696 | 0.692 | 0.004 [0.002, 0.006] | 1800 (0/0) |
| DM-5 | vctk2k | UTMOS | 3.936 | 3.800 | 0.137 [0.118, 0.156] | 2000 (0/0) |
| DM-6 | ltc2k | UTMOS | 4.159 | 4.156 | 0.003 [-0.006, 0.012] | 1800 (0/0) |

### Budget table (E16)

| Set | System | Endpoint | 3 s | 30 s | 180 s |
|---|---|---|---|---|---|
| vctk2k | mvc_v1 | capped WER % | 2.20 [1.33, 3.15] (n 1020) | 2.51 [1.63, 3.46] (n 1020) | 2.46 [1.55, 3.46] (n 1020) |
| vctk2k | mvc_v1 | WER % | 2.20 [1.33, 3.15] (n 1020) | 2.51 [1.63, 3.46] (n 1020) | 2.46 [1.55, 3.46] (n 1020) |
| vctk2k | mvc_v1 | SIM | 0.844 [0.825, 0.863] (n 2000) | 0.886 [0.871, 0.901] (n 2000) | 0.892 [0.878, 0.906] (n 2000) |
| vctk2k | mvc_v1 | UTMOS | 3.797 [3.711, 3.875] (n 2000) | 3.829 [3.750, 3.902] (n 2000) | 3.819 [3.739, 3.895] (n 2000) |
| vctk2k | mvc_v1 | dB-MCD | 8.04 [7.86, 8.24] (n 980) | 7.62 [7.49, 7.75] (n 980) | 7.59 [7.45, 7.73] (n 980) |
| vctk2k | cand_pm18_rq | capped WER % | 16.13 [12.40, 20.33] (n 1020) | 16.26 [12.70, 20.16] (n 1020) | 16.23 [12.45, 20.44] (n 1020) |
| vctk2k | cand_pm18_rq | WER % | 16.40 [12.45, 20.88] (n 1020) | 16.53 [12.77, 20.67] (n 1020) | 16.55 [12.54, 20.96] (n 1020) |
| vctk2k | cand_pm18_rq | SIM | 0.799 [0.769, 0.828] (n 2000) | 0.806 [0.778, 0.831] (n 2000) | 0.807 [0.779, 0.831] (n 2000) |
| vctk2k | cand_pm18_rq | UTMOS | 2.627 [2.550, 2.704] (n 2000) | 2.628 [2.553, 2.708] (n 2000) | 2.629 [2.554, 2.709] (n 2000) |
| vctk2k | cand_pm18_rq | dB-MCD | 9.13 [8.94, 9.32] (n 980) | 9.02 [8.85, 9.19] (n 980) | 9.02 [8.85, 9.19] (n 980) |
| vctk2k | knnvc | capped WER % | 60.27 [53.47, 66.55] (n 1020) | 6.23 [4.28, 8.37] (n 1020) | 2.89 [1.69, 4.27] (n 1020) |
| vctk2k | knnvc | WER % | 70.36 [58.78, 88.03] (n 1020) | 6.26 [4.29, 8.40] (n 1020) | 2.89 [1.69, 4.27] (n 1020) |
| vctk2k | knnvc | SIM | 0.821 [0.788, 0.852] (n 2000) | 0.895 [0.872, 0.914] (n 2000) | 0.901 [0.881, 0.916] (n 2000) |
| vctk2k | knnvc | UTMOS | 2.460 [2.277, 2.624] (n 2000) | 3.664 [3.600, 3.731] (n 2000) | 3.786 [3.725, 3.844] (n 2000) |
| vctk2k | knnvc | dB-MCD | 8.18 [7.95, 8.39] (n 980) | 7.43 [7.30, 7.55] (n 980) | 7.35 [7.23, 7.48] (n 980) |
| ltc2k | mvc_v1 | capped WER % | 2.41 [1.85, 3.04] (n 1800) | 2.37 [1.78, 3.03] (n 1800) | 2.35 [1.78, 3.04] (n 1699) |
| ltc2k | mvc_v1 | WER % | 2.93 [1.93, 4.59] (n 1800) | 2.89 [1.86, 4.56] (n 1800) | 2.85 [1.86, 4.44] (n 1699) |
| ltc2k | mvc_v1 | SIM | 0.894 [0.874, 0.913] (n 1800) | 0.928 [0.917, 0.940] (n 1800) | 0.935 [0.925, 0.945] (n 1699) |
| ltc2k | mvc_v1 | UTMOS | 3.950 [3.894, 4.002] (n 1800) | 4.027 [3.973, 4.079] (n 1800) | 4.044 [3.990, 4.097] (n 1699) |
| ltc2k | cand_pm18_rq | capped WER % | 10.51 [8.96, 12.14] (n 1800) | 10.33 [8.72, 12.00] (n 1800) | 10.46 [8.84, 12.15] (n 1699) |
| ltc2k | cand_pm18_rq | WER % | 11.31 [9.30, 14.03] (n 1800) | 10.83 [8.99, 12.79] (n 1800) | 10.94 [9.13, 12.91] (n 1699) |
| ltc2k | cand_pm18_rq | SIM | 0.894 [0.876, 0.911] (n 1800) | 0.911 [0.898, 0.923] (n 1800) | 0.909 [0.896, 0.922] (n 1699) |
| ltc2k | cand_pm18_rq | UTMOS | 2.703 [2.619, 2.782] (n 1800) | 2.731 [2.662, 2.798] (n 1800) | 2.726 [2.653, 2.800] (n 1699) |
| ltc2k | knnvc | capped WER % | 34.57 [29.89, 39.44] (n 1800) | 3.79 [3.04, 4.62] (n 1800) | 2.65 [2.04, 3.35] (n 1699) |
| ltc2k | knnvc | WER % | 35.49 [30.64, 40.79] (n 1800) | 4.31 [3.18, 6.03] (n 1800) | 3.15 [2.15, 4.68] (n 1699) |
| ltc2k | knnvc | SIM | 0.820 [0.801, 0.838] (n 1800) | 0.929 [0.919, 0.939] (n 1800) | 0.935 [0.924, 0.945] (n 1699) |
| ltc2k | knnvc | UTMOS | 2.538 [2.437, 2.634] (n 1800) | 3.837 [3.790, 3.884] (n 1800) | 3.967 [3.923, 4.011] (n 1699) |

### Gender pairs

vctk2k:

| System | Pair | capped WER % | SIM | UTMOS | SIM_src |
|---|---|---|---|---|---|
| mvc_v1 | F2F | 2.62 [1.02, 4.63] | 0.893 [0.870, 0.918] | 3.782 [3.650, 3.898] | 0.837 [0.798, 0.872] |
| mvc_v1 | F2M | 2.12 [0.94, 3.40] | 0.866 [0.844, 0.890] | 3.813 [3.705, 3.915] | 0.654 [0.575, 0.731] |
| mvc_v1 | M2F | 2.43 [1.21, 3.90] | 0.891 [0.869, 0.911] | 3.913 [3.810, 4.004] | 0.541 [0.469, 0.615] |
| mvc_v1 | M2M | 2.88 [1.27, 4.88] | 0.894 [0.873, 0.913] | 3.807 [3.683, 3.922] | 0.794 [0.768, 0.823] |
| cand_pm18_rq | F2F | 18.56 [12.61, 24.57] | 0.840 [0.815, 0.864] | 2.658 [2.548, 2.790] | 0.817 [0.786, 0.846] |
| cand_pm18_rq | F2M | 18.83 [12.78, 25.62] | 0.776 [0.732, 0.814] | 2.518 [2.431, 2.607] | 0.496 [0.450, 0.546] |
| cand_pm18_rq | M2F | 12.64 [9.07, 17.02] | 0.832 [0.809, 0.853] | 2.741 [2.642, 2.851] | 0.603 [0.539, 0.670] |
| cand_pm18_rq | M2M | 14.47 [9.71, 19.91] | 0.777 [0.736, 0.815] | 2.596 [2.512, 2.689] | 0.754 [0.706, 0.790] |
| knnvc | F2F | 5.72 [2.61, 9.94] | 0.898 [0.861, 0.926] | 3.703 [3.603, 3.806] | 0.859 [0.818, 0.893] |
| knnvc | F2M | 7.21 [3.75, 11.11] | 0.898 [0.862, 0.926] | 3.608 [3.521, 3.696] | 0.630 [0.536, 0.719] |
| knnvc | M2F | 5.34 [2.60, 8.56] | 0.885 [0.853, 0.914] | 3.721 [3.632, 3.822] | 0.621 [0.560, 0.684] |
| knnvc | M2M | 6.59 [3.93, 9.55] | 0.897 [0.864, 0.925] | 3.625 [3.556, 3.699] | 0.833 [0.803, 0.858] |
| freevc | F2F | 1.66 [0.45, 3.25] | 0.888 [0.852, 0.917] | 4.059 [3.973, 4.126] | 0.883 [0.851, 0.911] |
| freevc | F2M | 2.18 [0.62, 4.25] | 0.872 [0.831, 0.905] | 3.946 [3.853, 4.027] | 0.710 [0.618, 0.792] |
| freevc | M2F | 2.08 [0.62, 4.07] | 0.887 [0.857, 0.913] | 4.021 [3.968, 4.070] | 0.635 [0.567, 0.700] |
| freevc | M2M | 2.24 [0.86, 4.16] | 0.903 [0.873, 0.928] | 3.977 [3.912, 4.045] | 0.854 [0.826, 0.880] |
| seedvc | F2F | 2.35 [0.77, 4.48] | 0.913 [0.895, 0.931] | 3.805 [3.684, 3.909] | 0.861 [0.835, 0.887] |
| seedvc | F2M | 2.55 [1.23, 4.11] | 0.916 [0.900, 0.931] | 3.778 [3.693, 3.859] | 0.670 [0.589, 0.746] |
| seedvc | M2F | 2.73 [0.89, 5.16] | 0.922 [0.905, 0.936] | 3.786 [3.678, 3.880] | 0.603 [0.540, 0.666] |
| seedvc | M2M | 2.76 [1.08, 5.08] | 0.927 [0.914, 0.938] | 3.780 [3.705, 3.858] | 0.831 [0.807, 0.856] |
| facodec | F2F | 2.03 [0.75, 3.91] | 0.887 [0.861, 0.911] | 3.694 [3.558, 3.827] | 0.905 [0.880, 0.928] |
| facodec | F2M | 3.08 [1.28, 5.36] | 0.868 [0.842, 0.892] | 3.458 [3.345, 3.571] | 0.674 [0.589, 0.754] |
| facodec | M2F | 2.02 [0.87, 3.60] | 0.882 [0.856, 0.904] | 3.655 [3.510, 3.796] | 0.661 [0.594, 0.727] |
| facodec | M2M | 2.65 [1.07, 4.87] | 0.869 [0.844, 0.892] | 3.704 [3.589, 3.813] | 0.868 [0.851, 0.887] |

ltc2k:

| System | Pair | capped WER % | SIM | UTMOS | SIM_src |
|---|---|---|---|---|---|
| mvc_v1 | F2F | 2.82 [1.85, 3.97] | 0.924 [0.907, 0.939] | 3.935 [3.831, 4.031] | 0.815 [0.779, 0.849] |
| mvc_v1 | F2M | 2.82 [1.75, 4.08] | 0.932 [0.917, 0.946] | 4.062 [3.998, 4.123] | 0.550 [0.496, 0.610] |
| mvc_v1 | M2F | 1.99 [1.14, 2.95] | 0.916 [0.897, 0.934] | 4.072 [4.004, 4.140] | 0.567 [0.520, 0.616] |
| mvc_v1 | M2M | 1.95 [1.28, 2.69] | 0.941 [0.924, 0.955] | 4.038 [3.971, 4.099] | 0.833 [0.791, 0.867] |
| cand_pm18_rq | F2F | 12.18 [9.32, 15.21] | 0.903 [0.882, 0.922] | 2.852 [2.775, 2.925] | 0.823 [0.794, 0.851] |
| cand_pm18_rq | F2M | 12.12 [9.49, 14.88] | 0.916 [0.899, 0.932] | 2.542 [2.423, 2.652] | 0.578 [0.511, 0.647] |
| cand_pm18_rq | M2F | 9.21 [7.33, 11.21] | 0.906 [0.884, 0.927] | 2.895 [2.826, 2.960] | 0.604 [0.557, 0.652] |
| cand_pm18_rq | M2M | 8.21 [6.61, 9.88] | 0.919 [0.906, 0.932] | 2.635 [2.549, 2.719] | 0.840 [0.803, 0.872] |
| knnvc | F2F | 4.70 [3.35, 6.24] | 0.924 [0.910, 0.936] | 3.822 [3.749, 3.892] | 0.846 [0.815, 0.872] |
| knnvc | F2M | 4.85 [3.28, 6.68] | 0.934 [0.920, 0.945] | 3.811 [3.740, 3.882] | 0.588 [0.532, 0.649] |
| knnvc | M2F | 2.90 [1.96, 3.94] | 0.922 [0.905, 0.937] | 3.864 [3.801, 3.928] | 0.636 [0.590, 0.680] |
| knnvc | M2M | 2.93 [2.09, 3.75] | 0.938 [0.920, 0.951] | 3.851 [3.776, 3.929] | 0.871 [0.846, 0.894] |
| freevc | F2F | 3.20 [2.19, 4.40] | 0.894 [0.873, 0.914] | 4.007 [3.924, 4.073] | 0.874 [0.846, 0.897] |
| freevc | F2M | 3.20 [2.19, 4.35] | 0.882 [0.845, 0.909] | 4.048 [3.982, 4.112] | 0.634 [0.578, 0.690] |
| freevc | M2F | 1.98 [1.14, 2.95] | 0.905 [0.888, 0.921] | 3.941 [3.835, 4.039] | 0.636 [0.591, 0.682] |
| freevc | M2M | 1.88 [1.07, 2.79] | 0.908 [0.878, 0.933] | 4.082 [4.015, 4.145] | 0.890 [0.873, 0.906] |
| seedvc | F2F | 2.83 [1.84, 4.00] | 0.936 [0.921, 0.949] | 4.123 [4.071, 4.171] | 0.854 [0.826, 0.879] |
| seedvc | F2M | 3.24 [1.96, 4.79] | 0.949 [0.937, 0.960] | 4.029 [3.965, 4.090] | 0.637 [0.585, 0.694] |
| seedvc | M2F | 1.97 [1.14, 2.89] | 0.945 [0.932, 0.956] | 4.126 [4.075, 4.175] | 0.633 [0.591, 0.673] |
| seedvc | M2M | 2.08 [1.33, 2.91] | 0.956 [0.941, 0.968] | 4.009 [3.951, 4.071] | 0.870 [0.841, 0.896] |
| facodec | F2F | 2.31 [1.46, 3.21] | 0.903 [0.879, 0.925] | 3.712 [3.582, 3.833] | 0.885 [0.856, 0.909] |
| facodec | F2M | 2.65 [1.68, 3.81] | 0.919 [0.902, 0.934] | 3.480 [3.384, 3.574] | 0.619 [0.567, 0.677] |
| facodec | M2F | 1.81 [0.94, 2.79] | 0.895 [0.870, 0.918] | 3.448 [3.335, 3.562] | 0.707 [0.665, 0.746] |
| facodec | M2M | 1.87 [1.03, 2.85] | 0.924 [0.897, 0.945] | 3.679 [3.584, 3.775] | 0.909 [0.890, 0.927] |

### ECAPA levels, vctk2k

| System | SIM | SIM_src | M9 D | share D > 0 |
|---|---|---|---|---|
| mvc_v1 | 0.456 [0.426, 0.486] | 0.132 [0.113, 0.153] | -0.324 [-0.358, -0.290] | 0.015 |
| mvc_v1_q32 | 0.457 [0.427, 0.487] | 0.131 [0.112, 0.150] | -0.326 [-0.360, -0.293] | 0.016 |
| mvc_copy_latent | 0.122 [0.102, 0.142] | 0.736 [0.715, 0.757] | 0.614 [0.585, 0.644] | 1.000 |
| cand_pm18_rq | 0.161 [0.130, 0.194] | 0.086 [0.069, 0.102] | -0.075 [-0.113, -0.040] | 0.279 |
| knnvc | 0.647 [0.624, 0.670] | 0.183 [0.157, 0.209] | -0.464 [-0.499, -0.429] | 0.002 |
| freevc | 0.590 [0.546, 0.635] | 0.231 [0.204, 0.258] | -0.359 [-0.419, -0.299] | 0.026 |
| seedvc | 0.727 [0.712, 0.741] | 0.167 [0.147, 0.188] | -0.560 [-0.581, -0.538] | 0.000 |
| facodec | 0.482 [0.456, 0.509] | 0.260 [0.226, 0.295] | -0.222 [-0.257, -0.185] | 0.083 |
| faiss | 0.545 [0.520, 0.570] | 0.109 [0.087, 0.133] | -0.436 [-0.470, -0.402] | 0.002 |
| source_copy | 0.121 [0.101, 0.144] | 0.785 [0.765, 0.806] | 0.664 [0.633, 0.694] | 1.000 |
| target_copy | 0.795 [0.780, 0.809] | 0.124 [0.103, 0.149] | -0.671 [-0.697, -0.643] | 0.000 |

### DNSMOS levels, vctk2k

| System | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| mvc_v1 | 3.546 [3.517, 3.574] | 4.117 [4.106, 4.129] | 3.284 [3.252, 3.316] | 3.793 [3.734, 3.851] |
| mvc_v1_q32 | 3.552 [3.525, 3.579] | 4.122 [4.111, 4.133] | 3.293 [3.262, 3.324] | 3.798 [3.741, 3.855] |
| mvc_copy_latent | 3.478 [3.454, 3.501] | 4.043 [4.026, 4.060] | 3.183 [3.155, 3.212] | 3.679 [3.628, 3.728] |
| cand_pm18_rq | 3.164 [3.132, 3.196] | 3.905 [3.875, 3.934] | 2.825 [2.786, 2.863] | 3.244 [3.206, 3.283] |
| knnvc | 3.502 [3.480, 3.523] | 4.053 [4.031, 4.072] | 3.208 [3.180, 3.235] | 3.695 [3.639, 3.750] |
| freevc | 3.489 [3.469, 3.509] | 4.029 [4.003, 4.053] | 3.180 [3.150, 3.209] | 3.775 [3.721, 3.833] |
| seedvc | 3.527 [3.506, 3.548] | 4.047 [4.017, 4.071] | 3.234 [3.204, 3.263] | 3.731 [3.657, 3.803] |
| facodec | 3.526 [3.503, 3.548] | 4.058 [4.035, 4.078] | 3.235 [3.205, 3.264] | 3.677 [3.620, 3.738] |
| mimi_copy_src | 3.353 [3.324, 3.383] | 3.976 [3.958, 3.993] | 3.035 [3.002, 3.069] | 3.438 [3.384, 3.492] |
| source_copy | 3.544 [3.524, 3.565] | 4.065 [4.043, 4.083] | 3.257 [3.229, 3.285] | 3.725 [3.685, 3.766] |

### Acoustic guards, vctk2k

| System | rows (failed) | hnr | hnr_src | voiced_kept | f0_pcc | st_sd_ratio | rms_db_diff |
|---|---|---|---|---|---|---|---|
| mvc_v1 | 2000 (0) | 11.936 [11.199, 12.701] | 10.930 [10.083, 11.807] | 0.929 [0.919, 0.939] | 0.754 [0.711, 0.796] | 1.018 [0.881, 1.175] | -6.675 [-8.193, -4.994] |
| mvc_v1_q32 | 2000 (0) | 11.879 [11.143, 12.649] | 10.930 [10.083, 11.807] | 0.932 [0.922, 0.941] | 0.761 [0.717, 0.805] | 0.970 [0.844, 1.114] | -6.790 [-8.304, -5.116] |
| cand_pm18_rq | 2000 (0) | 8.274 [7.363, 9.193] | 10.930 [10.083, 11.807] | 0.837 [0.810, 0.861] | 0.465 [0.424, 0.505] | 0.949 [0.825, 1.089] | -3.015 [-4.740, -1.201] |
| knnvc | 2000 (0) | 11.630 [10.796, 12.468] | 10.930 [10.083, 11.807] | 0.910 [0.892, 0.929] | 0.474 [0.429, 0.521] | 0.913 [0.785, 1.055] | 8.900 [7.532, 10.423] |
| freevc | 2000 (0) | 11.642 [10.999, 12.343] | 10.930 [10.083, 11.807] | 0.917 [0.904, 0.930] | 0.742 [0.693, 0.790] | 0.977 [0.873, 1.087] | -0.585 [-2.054, 0.903] |
| seedvc | 2000 (0) | 10.750 [9.826, 11.697] | 10.930 [10.083, 11.807] | 0.894 [0.870, 0.915] | 0.631 [0.569, 0.688] | 1.266 [1.036, 1.547] | 1.267 [-0.616, 3.146] |
| facodec | 2000 (0) | 9.880 [9.057, 10.796] | 10.930 [10.083, 11.807] | 0.927 [0.913, 0.940] | 0.661 [0.588, 0.732] | 1.205 [1.072, 1.358] | -0.644 [-2.271, 1.023] |

### ECAPA levels, ltc2k

| System | SIM | SIM_src | M9 D | share D > 0 |
|---|---|---|---|---|
| mvc_v1 | 0.631 [0.610, 0.652] | 0.217 [0.197, 0.238] | -0.414 [-0.439, -0.386] | 0.001 |
| mvc_v1_q32 | 0.628 [0.607, 0.650] | 0.214 [0.194, 0.235] | -0.414 [-0.440, -0.386] | 0.001 |
| mvc_copy_latent | 0.223 [0.204, 0.242] | 0.719 [0.690, 0.744] | 0.496 [0.461, 0.529] | 0.996 |
| cand_pm18_rq | 0.455 [0.430, 0.480] | 0.221 [0.203, 0.240] | -0.234 [-0.258, -0.208] | 0.058 |
| knnvc | 0.665 [0.643, 0.686] | 0.270 [0.250, 0.293] | -0.395 [-0.422, -0.364] | 0.008 |
| freevc | 0.475 [0.448, 0.501] | 0.275 [0.258, 0.293] | -0.200 [-0.229, -0.172] | 0.079 |
| seedvc | 0.743 [0.722, 0.764] | 0.280 [0.259, 0.304] | -0.463 [-0.492, -0.432] | 0.002 |
| facodec | 0.569 [0.550, 0.588] | 0.316 [0.297, 0.338] | -0.252 [-0.279, -0.224] | 0.059 |
| faiss | 0.558 [0.533, 0.582] | 0.179 [0.160, 0.199] | -0.379 [-0.407, -0.350] | 0.004 |
| source_copy | 0.229 [0.211, 0.249] | 0.748 [0.721, 0.771] | 0.519 [0.483, 0.552] | 0.997 |

### DNSMOS levels, ltc2k

| System | SIG | BAK | OVRL | P808 |
|---|---|---|---|---|
| mvc_v1 | 3.573 [3.551, 3.595] | 4.112 [4.092, 4.129] | 3.314 [3.284, 3.343] | 3.865 [3.811, 3.915] |
| mvc_v1_q32 | 3.594 [3.573, 3.615] | 4.123 [4.104, 4.140] | 3.338 [3.309, 3.366] | 3.869 [3.817, 3.918] |
| mvc_copy_latent | 3.553 [3.534, 3.574] | 4.072 [4.052, 4.091] | 3.272 [3.245, 3.300] | 3.782 [3.727, 3.836] |
| cand_pm18_rq | 3.229 [3.196, 3.260] | 3.937 [3.911, 3.961] | 2.915 [2.884, 2.945] | 3.608 [3.577, 3.639] |
| knnvc | 3.607 [3.591, 3.624] | 4.123 [4.103, 4.140] | 3.353 [3.329, 3.377] | 3.958 [3.918, 3.996] |
| freevc | 3.591 [3.578, 3.603] | 4.079 [4.060, 4.097] | 3.315 [3.295, 3.335] | 3.916 [3.881, 3.948] |
| seedvc | 3.613 [3.598, 3.627] | 4.100 [4.080, 4.118] | 3.345 [3.324, 3.366] | 3.973 [3.930, 4.015] |
| facodec | 3.596 [3.579, 3.613] | 4.090 [4.071, 4.107] | 3.323 [3.299, 3.346] | 3.895 [3.853, 3.938] |
| mimi_copy_src | 3.541 [3.521, 3.561] | 4.069 [4.047, 4.088] | 3.258 [3.230, 3.286] | 3.737 [3.683, 3.790] |
| source_copy | 3.596 [3.577, 3.615] | 4.091 [4.069, 4.110] | 3.322 [3.295, 3.350] | 3.880 [3.830, 3.929] |

### Acoustic guards, ltc2k

| System | rows (failed) | hnr | hnr_src | voiced_kept | f0_pcc | st_sd_ratio | rms_db_diff |
|---|---|---|---|---|---|---|---|
| mvc_v1 | 1800 (0) | 13.438 [12.791, 14.085] | 13.358 [12.715, 14.017] | 0.940 [0.934, 0.945] | 0.878 [0.857, 0.898] | 1.002 [0.956, 1.050] | -10.820 [-11.176, -10.452] |
| mvc_v1_q32 | 1800 (0) | 13.433 [12.785, 14.082] | 13.358 [12.715, 14.017] | 0.944 [0.938, 0.949] | 0.881 [0.862, 0.899] | 0.984 [0.940, 1.030] | -10.958 [-11.311, -10.593] |
| cand_pm18_rq | 1800 (0) | 7.957 [7.331, 8.591] | 13.358 [12.715, 14.017] | 0.838 [0.818, 0.858] | 0.456 [0.423, 0.488] | 0.865 [0.805, 0.925] | -3.367 [-3.737, -3.005] |
| knnvc | 1800 (0) | 12.673 [12.170, 13.192] | 13.358 [12.715, 14.017] | 0.945 [0.938, 0.951] | 0.676 [0.638, 0.712] | 0.881 [0.824, 0.940] | 2.477 [2.048, 2.916] |
| freevc | 1800 (0) | 12.244 [11.732, 12.767] | 13.358 [12.715, 14.017] | 0.940 [0.934, 0.946] | 0.809 [0.785, 0.834] | 0.873 [0.822, 0.925] | -1.895 [-2.232, -1.568] |
| seedvc | 1800 (0) | 12.391 [11.740, 13.059] | 13.358 [12.715, 14.017] | 0.935 [0.928, 0.942] | 0.791 [0.763, 0.818] | 1.132 [1.062, 1.207] | 1.731 [1.442, 2.010] |
| facodec | 1800 (0) | 11.791 [11.109, 12.484] | 13.358 [12.715, 14.017] | 0.958 [0.952, 0.965] | 0.692 [0.656, 0.735] | 1.106 [1.053, 1.161] | -0.754 [-0.962, -0.546] |

## Inputs

- journal job M part t1: `job 4653 2026-10-20 06:45:22 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- journal job M part t2: `job 4656 2026-10-20 08:47:37 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- journal job M part t3: `job 4670 2026-10-20 12:07:10 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- journal job M part t4: `job 4672 2026-10-20 08:46:29 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- journal job X1: `job 4959 2026-10-21 11:31:45 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- journal job X2: `job 5067 2026-10-21 20:11:45 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- MimiFlow-VC T1 part t1: `job 6724 2026-10-28 12:13:43 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- MimiFlow-VC T1 part t2: `job 6761 2026-10-28 13:30:09 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- MimiFlow-VC T1 part t3: `job 6766 2026-10-28 15:33:14 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- `/Users/harry/duplex/mvc/MVC_SELECTION_RECORD_v1.json` sha256 b84885feb5fa0dc9b8c4cf35ee0f772f0cffe65cf21016f74d70f64c0920b9c8
- `/Users/harry/duplex/mvc/MVC_SELECTION_RECORD_v1.md` sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42
- `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/vctk2k.jsonl` sha256 bdbf36a32b61eff625e2acaf40163835582ca9e391079c11ca72e6bf51601c1d
- `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/ltc2k.jsonl` sha256 32fc920516f0b6e4d239ddc85994cca8f746538763e5ad71ab2e208203c56e45
- `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/vctk2k_source_E.json` sha256 ccbd53dfc507ad86b814a5e2cac2f8538f2bc7ca58a5d41f0f3c86e9627cb8ea
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t1/metrics/vctk2k_30.jsonl` sha256 af3a938254acf692408d1c9da099337ff42ab1981bb3e8bd699c3de1831519f6 (29960 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t1/metrics/vctk2k_30_src.jsonl` sha256 43be5f1d6163f60f21d2c299b1d5d92945e99c54f074051ed7f666dace6ea285 (29960 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/vctk2k_3.jsonl` sha256 d47c67dca8456b6eb9fe2c35be0af61147c1eff5f66c649011719ce3986d3fcd (8000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/vctk2k_3_src.jsonl` sha256 6443ad376da2d277c9ced72f7eedfc1fff562139668edc9cfc5dfe6e71b99f2f (8000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/vctk2k_180.jsonl` sha256 4040c83469e8a3ab893adb2d1461cf50d393a982f2ba1bb8328440c523e6dd85 (8000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/vctk2k_180_src.jsonl` sha256 f94c205a67c4f8ad9728a41a69535bdbb1c112d036ef272eee11d289bc62bbb2 (8000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/vctk2k_30.jsonl` sha256 83aa2c300fcc6a8807203651ad8f290b4d57fc258551180c5a4d8141a08189e2 (8000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/vctk2k_30_src.jsonl` sha256 4488b8990c124119100ca6cab535d0120d9328dcab1bfe672e23b2ac3c41b2e1 (8000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/vctk2k_3.jsonl` sha256 ec6a07ebe18930249c504497e3b798e0aacb36048a93ed07d8dc61d1da3452fb (2000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/vctk2k_3_src.jsonl` sha256 5403de56706c7c991c5dc444c0350c13d795e3683d8602730d2471d01329afac (2000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/vctk2k_180.jsonl` sha256 1d81ec27e959cc102dc270a2f048da89023a354ad5b48ebf3f8a28c322f94e82 (2000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/vctk2k_180_src.jsonl` sha256 f4ea2e875231bc88ba62031c344901f64ecaa8a38bde42d77bf83fc0a1029679 (2000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x1/m/metrics/vctk2k_3.jsonl` sha256 8845a1593995b47bb1f324eef501e706ebd87f90ac40c5468705b79519bdcebd (6000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x1/m/metrics/vctk2k_3_src.jsonl` sha256 5fb74c0d4fe698f162e715c549063a9018baf7da574e331cc34810d50c92126d (6000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t1/metrics/vctk2k_30.jsonl` sha256 b84fe6fea87f051c7dc239149092cea9315bc2036dc80074c13d9ef0c332e462 (6000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t1/metrics/vctk2k_30_src.jsonl` sha256 c3be713cd8b69626d08f65b13cb4eeda0c861a83bee8dc2379c41afb5351270d (6000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/vctk2k_3.jsonl` sha256 156ffd4c21f15e7479700b7f9f501c64ce7cdb02e19f1fe2a0cfbf2c5bc4d5ad (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/vctk2k_3_src.jsonl` sha256 1bb3faedd99707f0b7e1db97281b4225576fc2bbde954e99a57e76b008783625 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/vctk2k_180.jsonl` sha256 4eaa2f3c3416d54b834956dd320cb2ee694da56c7500b77640e87f548fe2f119 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/vctk2k_180_src.jsonl` sha256 53021566cc1c1b212dd51ec42ebc7ef1bf86dcaf501a3d1cb39f450e80a5d405 (2000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x2/ecapa/vctk2k_30.jsonl` sha256 49ec80df5d1c15c587548a02a5341cd79ece021c4b930be0d28c68db34954054 (16980 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x2/ecapa/vctk2k_30_src.jsonl` sha256 c05f17fb92d6530f8f434cf86904983739b2bc09877e4ed47662f77e65954ed7 (16980 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/vctk2k_30.jsonl` sha256 6ed3653e44d9e9945a502d162e37f7ee33c257f1b43b035adb0a734eaac6bb1d (6000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/vctk2k_30_src.jsonl` sha256 33cfcabed0f130c9c7245be92550710f9c2c9aeab9fc73f5645e09c035a0ab2d (6000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_cand_pm18_rq.jsonl` sha256 667cee4e2a21b04ad0b462fce411384ce6e51f42347d5b3c9b3e6e2400bee7c4 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_facodec.jsonl` sha256 b77f0d55cdd6246c59cc3da6da3afaccce699523a9103d018f4ba21a725e725c (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_freevc.jsonl` sha256 dbe135811d8edb3fa297991e0cd590c376322c3f3053b4bb8adaf7219af1b180 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_knnvc.jsonl` sha256 fb92bdd46abb41f52d692c5063560bfdf503710047f71185c2b427da2a03955b (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_mimi_copy_src.jsonl` sha256 9e4b2fcbeaf41c23d6d69f978292f089353440a3204313bc70f706d9ab04a304 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_mvc_copy_latent.jsonl` sha256 73791c00ff4e289bc780388ba653e869047c1ab27c5f1f7d7a947528ea630c8f (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_mvc_v1.jsonl` sha256 5b4829686b56e22339e24ad9b9fc12658c318af7eb137055b0f3dad9a62dae23 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_mvc_v1_q32.jsonl` sha256 7b0c974365ff5d0a5a29eb0d3d5ccac4341db8f9b3243e4394d4c78d52f61f02 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_seedvc.jsonl` sha256 571a66a833135d95188595b6afe4659eb00503578e955c85e007be6d4f31b1e4 (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_source_copy.jsonl` sha256 73ac992f682b0e1e4e112d0159b2664ae145864ff0abc0c13f46f0dd364089bb (2000 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/guard/vctk2k_30.jsonl` sha256 fe06233f25a9604cfb259f803c11d0f33d8fbe9c172e6eff08e337eb6552fa38 (14000 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t2/metrics/ltc2k_30.jsonl` sha256 cffd84042dfccb27469577e83852892286a4114c4065abcf17341ecff68fc7c0 (25200 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t2/metrics/ltc2k_30_src.jsonl` sha256 59c056a602b54cd054936446a11231914e755922a9e07a3a24eaaee146d7b206 (25200 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/ltc2k_3.jsonl` sha256 d9bf03e0e29b8e0e31be0df0976b18df6a7673520ec31e9ca8f72c3f79ee7219 (7200 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/ltc2k_3_src.jsonl` sha256 9bcd3ad75adecfaa12dd35d6f4e090c694d2614769e475710c58426608595ed6 (7200 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/ltc2k_180.jsonl` sha256 af26e7bebfebe261ce21363ac3f67a0e7502f3089b1452e04b25be02b75d19f9 (7099 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t3/metrics/ltc2k_180_src.jsonl` sha256 8c935161a8149eb4f6a6ede2e65afcb56357c935c3c3d8f2d07902c8f878b683 (6796 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/ltc2k_30.jsonl` sha256 bf3ef807d33d0e41503b04cb154da95ab77ee0b76cf64dc2731f4ee7ef393043 (7200 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/ltc2k_30_src.jsonl` sha256 9396db596d7467abaf480b5f5a34a725999870ba39fd0f4923cdabd34380d29b (7200 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/ltc2k_3.jsonl` sha256 c2fdcd342a9917a1f00859d3c666d5b186c407458515f0bdb6309fad32a129f2 (1800 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/ltc2k_3_src.jsonl` sha256 f4953fc7e2a02c29e884b7c03195f170d8d1640235b69239192c9109de6f6534 (1800 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/ltc2k_180.jsonl` sha256 3d66ecf68a3d378a08a8112abde0183d3b04f7153f5301e3d3fab3ec45d42f77 (1800 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/m/t4/metrics/ltc2k_180_src.jsonl` sha256 efe3b1b4c0462830ab4a04f2cba5985e26cfbbef302c69873036a0eb3aabb996 (1699 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x1/m/metrics/ltc2k_3.jsonl` sha256 13a82cb3ce6e686b698278c3e54b052699f3dbb1dbe73b374eb17f2b4d5a470a (5400 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x1/m/metrics/ltc2k_3_src.jsonl` sha256 1dc52bbbc3e6ddb971d688156ebd08cbc499af27dd109bf4edc9ff9c5ed231c5 (5400 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t2/metrics/ltc2k_30.jsonl` sha256 2384bf4277a023dd12581d6809a9d52b1c0d229ad0d68dad2f313802a9473aec (5400 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t2/metrics/ltc2k_30_src.jsonl` sha256 89820ec34ef16bb320d85ea160ef4af29120a79351e3e99dbefaeacb24611282 (5400 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/ltc2k_3.jsonl` sha256 6c1225be3e2f56924fb8332b684b4e98dc776a0277fb3c7fa02d7e22050b3980 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/ltc2k_3_src.jsonl` sha256 ee92a30cdae79a10c4dfe486f4ff5cdbb0588a124f765335cdee79c88562207f (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/ltc2k_180.jsonl` sha256 a664fff3d3389c8b5596896dd31bbbdb347fb5efc9db6f01a15a025a73b3a242 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/m/t3/metrics/ltc2k_180_src.jsonl` sha256 a77b5152788fa230720ebbdaabf08a93b1b186b0c9bdce876e89b11710333a73 (1699 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x2/ecapa/ltc2k_30.jsonl` sha256 0f0349de4ed93f743e5998d9886a5fcb0a83f454b02c1611e1e816ffc02155b8 (14400 rows)
- `/Users/harry/duplex/paper/journal_extension/results/v1/x2/ecapa/ltc2k_30_src.jsonl` sha256 7239f200016bd4c3ea1619968487b2e1a021db86264b53c307baf2b9ce6b39de (14400 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/ltc2k_30.jsonl` sha256 d697cab15a11fda50c5c8cb56095ecc0411eabd72a98e3b3856583ba9b2218b8 (5400 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/ltc2k_30_src.jsonl` sha256 085f882d7cc8c87812867d3960497ea029eb62bab4bb1fa349e55898439f7ed1 (5400 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_cand_pm18_rq.jsonl` sha256 1a3995cd9fb57c2c849469c53b9ce8a09d02f6480d550c89a1ba91721b87525d (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_facodec.jsonl` sha256 2b5852a55588eef07343141722af323162334d1ac2e40b77d31a74984b665697 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_freevc.jsonl` sha256 c57c72e9a4146725ec6f921b1107eadbaa21ad4a0e54a2dffc598bcfe2180d03 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_knnvc.jsonl` sha256 9da29ad7e1a9e54937595cd73cdbd1537b18822658f8b77296b95599ea47a479 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_mimi_copy_src.jsonl` sha256 cf0e18ac6e1a9a46e437d22619ce271990fa72dd6ee6bc951cd7bb53f3d9fa0b (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_mvc_copy_latent.jsonl` sha256 f6bee239529d46a45f48fac09364612a4426675ea21d380815e3138dd61e68c8 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_mvc_v1.jsonl` sha256 5af011d32d9a9015e156302568ac214fa988a772b723e2ea8a4cdab7ec4faa1e (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_mvc_v1_q32.jsonl` sha256 09575afe6948c9ce973266371d1d59509a7f47cda1e6359f6b6d55d4927e4452 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_seedvc.jsonl` sha256 a245df7d99639745511b650f63898d2e2c6bf918569420f2c6d13c7e4c4e30f3 (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_source_copy.jsonl` sha256 587f5e35b157a8ce24feb0dd7237289419655371a2b1378d051e32cfd3e6eb8d (1800 rows)
- `/Users/harry/duplex/paper/mimi_vc/results/guard/ltc2k_30.jsonl` sha256 88a9aa098bfc7f33f34a623e069fc72ac1b0a0f63dd6315dae4f2b7669a225e6 (12600 rows)
