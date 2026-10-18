# Dev selection record (ANALYSIS_PLAN_v1 §4.6)

Written 2026-10-19 00:37:40 by `benchmark/select_crt_al.py` (sha256 9405fb8bf0b73339bed21bfbfd42494a15d910c07a9cd9f7f5df0a28028bff46) with `benchmark/stats.py` (sha256 a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6). Linked crossed bootstrap of the Dev speakers, B = 10,000, seed 20260928; 95 % percentile CIs.

**Status: complete**

## Decisions

- S2: τ_dev = 0.9 (`anc_R1_r0_tau0.9`), decided by step 3 of §4.2; no τ meets SIM >= FAISS: step 3, the smallest SIM deficit
- S1u: `front_us32_w0.2_0.2_0.1` (both have SIM >= FAISS: the lower capped WER)
- E17 (`front_lmbr_k2_g1_T1`) enters the test runs: yes (capped WER 87.93 < S2 91.38 (two decimals; uncapped 127.38 vs 157.55); SIM 0.913 >= FAISS 0.908)
- CRT-AL: C2 (`cand_pm18_rq`)
- 950-pair check: C2 (same as on all pairs)
- VCTK-2k variant for T1: 5 targets
- Listening items: 32, file `/Users/harry/duplex/benchmark/DEV_SELECTION_RECORD_v1.listening_items.json` sha256 145aa331c4e8b16b9ce5b45a4aebf07108eac758cbda4c412f5f06e6900fc019

## Inputs

| file | rows read | sha256 |
|---|---|---|
| `/Users/harry/duplex/paper/journal_extension/results/v1/d1/metrics.jsonl` | 36000 | aa72f0e9e395d96318186e5a1a851b97b3f4abd7a147dcb13316aa2691425b91 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d1/metrics_src.jsonl` | 36000 | 774b3344af49f79a85d51afe572a07128d1ebf947a4c39758aed8c50c3f368f7 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d2/lab_metrics.jsonl` | 8000 | 2c0ac07df9c9125347bd4b7c594b8589a08a9930edd45cfb8b7cdcfb6c3d3167 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d2/lab_metrics_src.jsonl` | 8000 | e9e3d17794a31b34779b3e67997f971bb2c3c15c16a4ce232039cfbc3f0be550 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d2/controls_metrics.jsonl` | 3000 | dc60484e30286e0a1666f01cf55e8d646f652cc0cc59d05694e3befea4a06406 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d2/controls_metrics_src.jsonl` | 3000 | 77c17cc5679fa5171f4a93299fdbc7b0fdb0a6727fb30e1ed0f85c1e33638b23 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/sacct_nvquoc.txt` | – | c3f40ba79a347a6dd100aa98831b81ef9a1b6c8eb38a2f691a95e8e07a9c8f3b |
| `/Users/harry/duplex/paper/journal_extension/results/v1/sacct_2634.txt` | – | 34da7e566cca35137218aa1363683985704ce9bb1f0f96730dea388dcd86a0c3 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d0/check_d0.json` | – | e8c5c416bd15747641672edd453c6344df9ed8d87be6183b1a91fd51df2e2650 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d1/repro_d1_vs_d0.json` | – | 6155c075e62d9dc7e06478d2a50a6b709ebbc961a75aac783837ab1ca8aa54fd |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d2/repro_d2_vs_d0.json` | – | 55ff968638cda21611323e30dc4c3686592a5d9076a0054c71d0091e1765950a |
| `/Users/harry/duplex/paper/journal_extension/results/v1/baselines.json` | – | 229e584510a1f37a7a4e0c313ed6d28682d677ebeec4ef046b50ac6336e023ac |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d2/c1_cand_wavs.sha256` | – | 46b4a790d39ef86b575fd7599793bf663175e65701bf9430c34fa89d758aba69 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/d1/c1_anchor_wavs.sha256` | – | 46b4a790d39ef86b575fd7599793bf663175e65701bf9430c34fa89d758aba69 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/vctk2k.jsonl` | – | bdbf36a32b61eff625e2acaf40163835582ca9e391079c11ca72e6bf51601c1d |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/summary.json` | – | ea098aa81f3908664c9ce01ee72853da80de0f62b33a58e00c93bb0fa5282918 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/vctk2k_source_E.json` | – | ccbd53dfc507ad86b814a5e2cac2f8538f2bc7ca58a5d41f0f3c86e9627cb8ea |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs_vctk2k_t10/vctk2k.jsonl` | – | ab1edc9b73212fb79004682a9858c281823e2a3996ebe8c77ec04a8ee1902afb |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs_vctk2k_t10/summary.json` | – | 73d02669fe18831aadf94c09857e886e1aa955496e2aa629e40299c9e0bc008e |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs_vctk2k_t10/vctk2k_source_E.json` | – | adaa747866c46f106bff39ecebb9ff1f680c4720fe68aedfca17fd86be289a45 |
| `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/ltc2k.jsonl` | – | 32fc920516f0b6e4d239ddc85994cca8f746538763e5ad71ab2e208203c56e45 |

Pair list `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/dev.jsonl` (1000 pairs, sha256 ff5cd1147e1236652cfee022c4fdfd7a9bd63ac11bc64989cbdb82a7097ec194); screen = rows k % 20 == 0 (50 pairs). Ignored rows: none.
Common set (§4.1): 1000 pairs (500 wer_primary, 500 with P, 500 same-gender, 500 cross-gender); pairs with no row of any required system: 0. Excluded per required system (no row or an error row, among the pairs that have rows): none.
Required systems without rows: none.

## Reproduction check (§4.4, §8.1 item 9)

- `/Users/harry/duplex/paper/journal_extension/results/v1/d0/check_d0.json` (sha256 e8c5c416bd15747641672edd453c6344df9ed8d87be6183b1a91fd51df2e2650): passed = True; D0 vs the screen (50 screen pairs): S1/S3 codes OK (50/50 pairs equal); A1-A5,C1 out_codes of anc OK; A1-A5,C1 out_codes of anc_r1r3 OK; C1-C5 WAVs: C1 identical, C2 identical, C3 identical, C4 identical, C5 identical; pair lists OK
- `/Users/harry/duplex/paper/journal_extension/results/v1/d1/repro_d1_vs_d0.json` (sha256 6155c075e62d9dc7e06478d2a50a6b709ebbc961a75aac783837ab1ca8aa54fd): passed = True; D1 vs D0 (50 screen pairs): S1/S3 codes OK (50/50 pairs equal); A1-A5,C1 out_codes of anc OK
- `/Users/harry/duplex/paper/journal_extension/results/v1/d2/repro_d2_vs_d0.json` (sha256 55ff968638cda21611323e30dc4c3686592a5d9076a0054c71d0091e1765950a): passed = True; D2 vs D0 (50 screen pairs): C1-C5 WAVs: C1 identical, C2 identical, C3 identical, C4 identical, C5 identical

## Jobs (sacct)

| role | job ID | Submit | End | State |
|---|---|---|---|---|
| 2634 | 2634 | 2026-10-11T06:31:00 | 2026-10-11T07:00:40 | COMPLETED |
| D0 | 4321 | 2026-10-18T05:20:56 | 2026-10-18T06:31:11 | COMPLETED |
| D1 | 4344 | 2026-10-18T11:43:03 | 2026-10-18T12:42:08 | COMPLETED |
| D1m | 4361 | 2026-10-18T12:46:45 | 2026-10-19T00:34:59 | COMPLETED |
| D2 | 4380 | 2026-10-18T13:52:22 | 2026-10-18T17:59:58 | COMPLETED |

## S2: the v0 τ rule (§4.2)

FAISS SIM 0.908. WER over the common wer_primary pairs; the rule uses v0's uncapped corpus WER, the capped WER (M3c) is printed beside it (§0.3 row 20).

| τ | system | uncapped WER % (rule) | capped WER % | SIM | SIM − FAISS | dB-MCD | SIM ≥ FAISS |
|---|---|---|---|---|---|---|---|
| 0.1 | `anc_R1_r0_tau0.1` | 105.48 | 88.53 | 0.833 | -0.074 | 9.87 | no |
| 0.2 | `anc_R1_r0_tau0.2` | 128.47 | 89.63 | 0.866 | -0.042 | 9.40 | no |
| 0.3 | `anc_R1_r0_tau0.3` | 146.31 | 89.48 | 0.886 | -0.021 | 9.09 | no |
| 0.4 | `anc_R1_r0_tau0.4` | 140.95 | 90.46 | 0.896 | -0.011 | 8.92 | no |
| 0.5 | `anc_R1_r0_tau0.5` | 159.65 | 90.58 | 0.901 | -0.006 | 8.83 | no |
| 0.6 | `anc_R1_r0_tau0.6` | 141.44 | 90.26 | 0.904 | -0.004 | 8.77 | no |
| 0.7 | `anc_R1_r0_tau0.7` | 150.14 | 91.50 | 0.906 | -0.002 | 8.73 | no |
| 0.8 | `anc_R1_r0_tau0.8` | 152.13 | 91.33 | 0.907 | -0.000 | 8.71 | no |
| 0.9 | `anc_R1_r0_tau0.9` | 157.55 | 91.38 | 0.908 | -0.000 | 8.70 | no |

τ_dev = 0.9, step 3. no τ meets SIM >= FAISS: step 3, the smallest SIM deficit.

## S1u setting and E17 (§4.3)

On 1000 common pairs; the S1u rule uses the capped WER (M3c), the uncapped WER is printed beside it:

| system | capped WER % (rule) | uncapped WER % | SIM | UTMOS | dB-MCD |
|---|---|---|---|---|---|
| `front_us32_w0.2_0.2_0.1` | 85.62 | 104.61 | 0.930 | 2.29 | 8.41 |
| `front_us32_w0.5_0.3_0.3` | 88.65 | 109.31 | 0.938 | 2.67 | 8.52 |
| `anc_faiss` | 91.27 | 164.18 | 0.908 | 1.83 | 8.70 |

S1u: `front_us32_w0.2_0.2_0.1` — both have SIM >= FAISS: the lower capped WER.

E17 `front_lmbr_k2_g1_T1` enters: yes — capped WER 87.93 < S2 91.38 (two decimals; uncapped 127.38 vs 157.55); SIM 0.913 >= FAISS 0.908 (on 1000 pairs).

## CRT-AL on all common pairs (§4.4)

1000 pairs (500 wer_primary); S2 = `anc_R1_r0_tau0.9`. Candidates without rows: none. E4 and steps 1–2 use the capped WER; capped (M3c) and uncapped corpus WER are both printed (§0.3 row 20).

| system | capped WER % | uncapped WER % | capped CER % | uncapped CER % | SIM | SECS | UTMOS | SRC same | SRC cross | D same (share > 0) | D cross (share > 0) | dB-MCD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 `cand_pm_rq_T1` | 23.52 [17.73, 29.94] | 23.63 [17.83, 30.09] | 13.89 [10.41, 17.74] | 13.89 [10.41, 17.74] | 0.800 [0.778, 0.821] | 0.671 [0.656, 0.686] | 2.12 [2.02, 2.23] | 0.740 [0.707, 0.774] | 0.557 [0.491, 0.624] | -0.059 [-0.093, -0.031] (0.28) | -0.245 [-0.320, -0.171] (0.08) | 9.13 [8.95, 9.32] |
| C2 `cand_pm18_rq` | 22.97 [17.17, 29.11] | 23.00 [17.21, 29.16] | 13.18 [9.62, 16.95] | 13.18 [9.62, 16.95] | 0.805 [0.783, 0.825] | 0.669 [0.650, 0.688] | 2.63 [2.54, 2.73] | 0.747 [0.711, 0.783] | 0.552 [0.483, 0.621] | -0.058 [-0.093, -0.028] (0.27) | -0.253 [-0.331, -0.177] (0.08) | 9.24 [9.04, 9.45] |
| C3 `cand_gv1_pm_cont_T1` | 25.73 [18.84, 33.04] | 25.85 [18.91, 33.28] | 15.66 [11.20, 20.29] | 15.67 [11.21, 20.30] | 0.806 [0.785, 0.826] | 0.718 [0.703, 0.733] | 2.36 [2.26, 2.46] | 0.740 [0.706, 0.775] | 0.559 [0.493, 0.625] | -0.064 [-0.098, -0.035] (0.25) | -0.249 [-0.324, -0.176] (0.06) | 9.46 [9.28, 9.65] |
| C4 `cand_gv1_shrink0.85_cont` | 25.94 [18.90, 33.69] | 26.17 [19.03, 34.07] | 15.78 [11.15, 21.18] | 15.78 [11.15, 21.18] | 0.814 [0.794, 0.835] | 0.735 [0.721, 0.748] | 2.30 [2.20, 2.40] | 0.742 [0.708, 0.777] | 0.558 [0.490, 0.625] | -0.071 [-0.104, -0.042] (0.24) | -0.259 [-0.335, -0.186] (0.06) | 9.12 [8.93, 9.30] |
| C5 `cand_gv0.5_shrink0.75_cont` | 25.27 [19.03, 32.29] | 25.48 [19.12, 32.73] | 15.26 [11.16, 19.79] | 15.26 [11.16, 19.79] | 0.826 [0.808, 0.845] | 0.725 [0.711, 0.739] | 2.06 [1.96, 2.15] | 0.749 [0.714, 0.783] | 0.560 [0.490, 0.630] | -0.075 [-0.110, -0.046] (0.23) | -0.268 [-0.347, -0.194] (0.05) | 8.62 [8.46, 8.80] |
| `anc_faiss` | 91.27 [88.60, 93.70] | 164.18 [125.56, 219.96] | 74.34 [71.17, 77.07] | 114.94 [84.81, 170.83] | 0.908 [0.901, 0.914] | 0.812 [0.803, 0.822] | 1.83 [1.76, 1.90] | 0.786 [0.743, 0.825] | 0.592 [0.514, 0.666] | -0.123 [-0.164, -0.087] (0.05) | -0.314 [-0.390, -0.243] (0.01) | 8.70 [8.55, 8.86] |
| `anc_R1_r0_tau0.9` | 91.38 [88.65, 93.78] | 157.55 [118.84, 221.46] | 74.10 [70.93, 76.93] | 110.98 [80.76, 171.74] | 0.908 [0.901, 0.913] | 0.812 [0.802, 0.822] | 1.82 [1.75, 1.90] | 0.786 [0.744, 0.825] | 0.593 [0.515, 0.667] | -0.123 [-0.163, -0.088] (0.06) | -0.313 [-0.389, -0.243] (0.01) | 8.70 [8.56, 8.85] |
| `anc_src0_faiss` | 90.20 [87.14, 93.14] | 149.71 [115.58, 197.50] | 73.00 [69.56, 76.24] | 101.11 [80.14, 131.82] | 0.906 [0.900, 0.912] | 0.812 [0.802, 0.822] | 1.81 [1.74, 1.89] | 0.786 [0.744, 0.826] | 0.592 [0.514, 0.666] | -0.122 [-0.163, -0.086] (0.07) | -0.313 [-0.389, -0.242] (0.01) | 8.64 [8.50, 8.80] |

Bootstrap redraws per quantity (replicates without a pair of positive weight or reference words; the same for every system on the same pairs): wer 0, cer 0, wer_u 0, cer_u 0, sim 0, secs 0, utmos 0, mcd 0, src_same 0, src_cross 0, m9_same 0, m9_cross 0. Values missing or not finite in rows without error (left out): none.

| candidate | SIM (E1 ≥ 0.80) | UTMOS (E2 ≥ FAISS) | E3 same / cross (≤ FAISS + 0.01) | WER − S2, capped (E4) | WER − A5, capped (E4) | WER − S2, uncapped | WER − A5, uncapped (E4 with uncapped WER) | eligible |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.800 yes | 2.12 vs 1.83 yes | 0.740 vs 0.796 / 0.557 vs 0.602 yes | -67.87 [-73.61, -62.45] | -66.69 [-72.40, -61.04] yes | -133.92 [-198.06, -94.69] | -126.08 [-173.57, -93.04] yes | yes |
| C2 | 0.805 yes | 2.63 vs 1.83 yes | 0.747 vs 0.796 / 0.552 vs 0.602 yes | -68.41 [-74.12, -63.03] | -67.23 [-73.02, -61.67] yes | -134.55 [-197.93, -95.01] | -126.71 [-174.87, -92.37] yes | yes |
| C3 | 0.806 yes | 2.36 vs 1.83 yes | 0.740 vs 0.796 / 0.559 vs 0.602 yes | -65.65 [-71.86, -59.26] | -64.47 [-70.79, -58.18] yes | -131.70 [-195.64, -91.94] | -123.86 [-171.01, -90.35] yes | yes |
| C4 | 0.814 yes | 2.30 vs 1.83 yes | 0.742 vs 0.796 / 0.558 vs 0.602 yes | -65.45 [-71.73, -58.81] | -64.27 [-70.52, -57.73] yes | -131.38 [-195.93, -91.19] | -123.54 [-170.75, -89.97] yes | yes |
| C5 | 0.826 yes | 2.06 vs 1.83 yes | 0.749 vs 0.796 / 0.560 vs 0.602 yes | -66.11 [-71.72, -60.28] | -64.93 [-70.48, -59.22] yes | -132.07 [-195.62, -92.67] | -124.24 [-171.37, -90.98] yes | yes |

| candidate | WER(Ck) − WER(C_m), capped | s_k (its SE) | WER(Ck) − WER(C_m), uncapped | its SE | in T |
|---|---|---|---|---|---|
| C1 | +0.55 [-2.18, +3.11] | 1.34 | +0.63 [-2.09, +3.17] | 1.33 | yes |
| C2 | 0 (C_m) | 0.00 | 0 (C_m) | 0.00 | yes |
| C3 | +2.77 [-0.43, +6.13] | 1.64 | +2.85 [-0.36, +6.19] | 1.64 | no |
| C4 | +2.97 [-0.75, +7.16] | 2.01 | +3.17 [-0.60, +7.44] | 2.05 | no |
| C5 | +2.31 [-0.61, +5.53] | 1.55 | +2.48 [-0.45, +5.71] | 1.56 | no |

Trace:

- eligible (E1-E4): ['C1', 'C2', 'C3', 'C4', 'C5']
- step 1: C_m = C2, the lowest capped WER (22.97 %; ties at two decimals -> lower number)
- step 2: s_k = bootstrap SE of capped WER(Ck) - WER(C_m); T = {Ck: WER(Ck) - WER(C_m) <= s_k} = C1 (+0.55 vs 1.34), C2 (+0.00 vs 0.00), C3 (+2.77 vs 1.64), C4 (+2.97 vs 2.01), C5 (+2.31 vs 1.55) -> T = ['C1', 'C2']
- steps 3-4: highest SIM in T (ties at three decimals -> dB-MCD, UTMOS, number): C2 (SIM C1 0.800, C2 0.805)

**Choice: C2**

## Pre-specified check: without the screen (950-pair rule)

950 pairs (480 wer_primary); S2 = `anc_R1_r0_tau0.9`. Candidates without rows: none. E4 and steps 1–2 use the capped WER; capped (M3c) and uncapped corpus WER are both printed (§0.3 row 20).

| candidate | SIM (E1 ≥ 0.80) | UTMOS (E2 ≥ FAISS) | E3 same / cross (≤ FAISS + 0.01) | WER − S2, capped (E4) | WER − A5, capped (E4) | WER − S2, uncapped | WER − A5, uncapped (E4 with uncapped WER) | eligible |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.799 no | 2.12 vs 1.83 yes | 0.738 vs 0.794 / 0.552 vs 0.600 yes | -67.68 [-73.38, -62.28] | -66.48 [-72.26, -60.83] yes | -132.29 [-191.47, -93.76] | -127.71 [-177.68, -93.28] yes | no |
| C2 | 0.804 yes | 2.63 vs 1.83 yes | 0.745 vs 0.794 / 0.546 vs 0.600 yes | -68.21 [-73.93, -62.84] | -67.02 [-72.86, -61.45] yes | -132.92 [-191.06, -94.33] | -128.34 [-178.63, -92.61] yes | yes |
| C3 | 0.804 yes | 2.36 vs 1.83 yes | 0.737 vs 0.794 / 0.553 vs 0.600 yes | -65.19 [-71.57, -58.62] | -63.99 [-70.54, -57.39] yes | -129.81 [-188.25, -91.09] | -125.22 [-175.04, -90.39] yes | yes |
| C4 | 0.812 yes | 2.30 vs 1.83 yes | 0.739 vs 0.794 / 0.552 vs 0.600 yes | -65.28 [-71.71, -58.47] | -64.08 [-70.56, -57.32] yes | -129.78 [-188.94, -90.79] | -125.19 [-175.36, -90.06] yes | yes |
| C5 | 0.825 yes | 2.06 vs 1.83 yes | 0.746 vs 0.794 / 0.554 vs 0.600 yes | -66.00 [-71.73, -60.01] | -64.80 [-70.51, -58.87] yes | -130.53 [-188.87, -92.08] | -125.94 [-175.53, -91.21] yes | yes |

| candidate | WER(Ck) − WER(C_m), capped | s_k (its SE) | WER(Ck) − WER(C_m), uncapped | its SE | in T |
|---|---|---|---|---|---|
| C2 | 0 (C_m) | 0.00 | 0 (C_m) | 0.00 | yes |
| C3 | +3.03 [-0.40, +6.52] | 1.72 | +3.12 [-0.34, +6.60] | 1.73 | no |
| C4 | +2.94 [-0.94, +7.23] | 2.07 | +3.15 [-0.79, +7.58] | 2.12 | no |
| C5 | +2.22 [-0.73, +5.43] | 1.56 | +2.40 [-0.63, +5.68] | 1.59 | no |

Trace:

- eligible (E1-E4): ['C2', 'C3', 'C4', 'C5']
- step 1: C_m = C2, the lowest capped WER (22.86 %; ties at two decimals -> lower number)
- step 2: s_k = bootstrap SE of capped WER(Ck) - WER(C_m); T = {Ck: WER(Ck) - WER(C_m) <= s_k} = C2 (+0.00 vs 0.00), C3 (+3.03 vs 1.72), C4 (+2.94 vs 2.07), C5 (+2.22 vs 1.56) -> T = ['C2']
- steps 3-4: highest SIM in T (ties at three decimals -> dB-MCD, UTMOS, number): C2 (SIM C2 0.804)

**Choice: C2**

## Sensitivity of the choice (§4.4)

| rule in step 2 (capped WER) | T | choice |
|---|---|---|
| one standard error (the rule) | C1, C2 | C2 |
| fixed tolerance 0 points | C2 | C2 |
| fixed tolerance 1 points | C1, C2 | C2 |
| fixed tolerance 2 points | C1, C2 | C2 |
| 95 % CI of WER(Ck) − WER(C_m) contains 0 | C1, C2, C3, C4, C5 | C5 |

Uncapped: with uncapped corpus WER in E4 and steps 1–2, the §4.4 rule picks C2 (eligible ['C1', 'C2', 'C3', 'C4', 'C5'], C_m = C2, T = ['C1', 'C2']); the same as with capped WER (§0.3 row 20; reported, never used).

## C1 identity check

`cand_pm_rq_T1` vs `anc_R3_s0_pm_rq_T1` on 1000 common pairs: identical Whisper hypotheses 999/1000, max |ΔSIM| 0.0000, max |ΔUTMOS| 0.0000, max |ΔdB-MCD| 0.0000, capped WER 23.52 vs 23.49 % (uncapped 23.63 vs 23.60 %); @src pairs 1000, max |ΔSIM_src| 0.0000. Within the §8.1 item 9 tolerance: yes; identical: yes. WAVs (sha256): 1000/1000 byte-identical; the check passes by identity (§8.1 item 9), and a differing Whisper hypothesis on identical audio comes from batched decoding.

## By gender pair (§4.4; point estimates)

| system | pair | n | capped WER % (n primary) | uncapped WER % | SIM | UTMOS | SIM_src | mean D | D > 0 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | F2F | 250 | 91.14 (124) | 152.61 | 0.912 | 1.75 | 0.799 | -0.112 | 0.06 |
| S1 | F2M | 250 | 91.22 (126) | 179.00 | 0.910 | 1.91 | 0.592 | -0.318 | 0.00 |
| S1 | M2F | 250 | 90.98 (126) | 166.55 | 0.902 | 1.73 | 0.592 | -0.310 | 0.02 |
| S1 | M2M | 250 | 91.74 (124) | 158.09 | 0.907 | 1.92 | 0.773 | -0.133 | 0.04 |
| C1 | F2F | 250 | 25.11 (124) | 25.23 | 0.813 | 2.20 | 0.754 | -0.059 | 0.26 |
| C1 | F2M | 250 | 23.89 (126) | 24.22 | 0.817 | 2.00 | 0.484 | -0.333 | 0.00 |
| C1 | M2F | 250 | 24.44 (126) | 24.44 | 0.787 | 2.20 | 0.630 | -0.157 | 0.16 |
| C1 | M2M | 250 | 20.54 (124) | 20.54 | 0.785 | 2.09 | 0.725 | -0.059 | 0.29 |
| C2 | F2F | 250 | 24.89 (124) | 24.89 | 0.820 | 2.66 | 0.763 | -0.057 | 0.26 |
| C2 | F2M | 250 | 24.78 (126) | 24.89 | 0.811 | 2.57 | 0.480 | -0.331 | 0.00 |
| C2 | M2F | 250 | 23.84 (126) | 23.84 | 0.798 | 2.66 | 0.623 | -0.176 | 0.15 |
| C2 | M2M | 250 | 18.18 (124) | 18.18 | 0.790 | 2.63 | 0.730 | -0.060 | 0.29 |
| C3 | F2F | 250 | 26.93 (124) | 27.16 | 0.823 | 2.40 | 0.760 | -0.063 | 0.21 |
| C3 | F2M | 250 | 27.56 (126) | 27.67 | 0.815 | 2.32 | 0.492 | -0.323 | 0.00 |
| C3 | M2F | 250 | 26.57 (126) | 26.69 | 0.799 | 2.37 | 0.625 | -0.174 | 0.12 |
| C3 | M2M | 250 | 21.72 (124) | 21.72 | 0.785 | 2.34 | 0.720 | -0.065 | 0.29 |
| C4 | F2F | 250 | 28.41 (124) | 28.86 | 0.834 | 2.34 | 0.765 | -0.069 | 0.22 |
| C4 | F2M | 250 | 27.00 (126) | 27.22 | 0.820 | 2.29 | 0.483 | -0.337 | 0.00 |
| C4 | M2F | 250 | 26.81 (126) | 27.05 | 0.813 | 2.26 | 0.632 | -0.180 | 0.12 |
| C4 | M2M | 250 | 21.37 (124) | 21.37 | 0.791 | 2.29 | 0.719 | -0.072 | 0.26 |
| C5 | F2F | 250 | 25.34 (124) | 25.80 | 0.841 | 2.07 | 0.767 | -0.074 | 0.20 |
| C5 | F2M | 250 | 25.00 (126) | 25.11 | 0.834 | 2.04 | 0.484 | -0.349 | 0.00 |
| C5 | M2F | 250 | 27.40 (126) | 27.52 | 0.823 | 2.04 | 0.636 | -0.187 | 0.11 |
| C5 | M2M | 250 | 23.38 (124) | 23.49 | 0.808 | 2.07 | 0.731 | -0.077 | 0.26 |

## Controls (descriptive)

| system | n | capped WER % | uncapped WER % | SIM | UTMOS | dB-MCD | SIM_src |
|---|---|---|---|---|---|---|---|
| `source_copy` | 1000 | 1.59 | 1.59 | 0.697 | 4.04 | 8.96 | 0.961 |
| `target_copy` | 500 | – | – | 0.970 | 4.07 | 0.00 | 0.705 |
| `mimi_copy_src` | 1000 | 2.45 | 2.45 | 0.698 | 3.82 | 9.04 | 0.957 |
| `mimi_copy_P` | 500 | – | – | 0.968 | 3.87 | 4.39 | 0.705 |

## Size check (§5.3)

Variance components (REML, MixedLM, one group; the optimiser with the highest REML log-likelihood is kept) of the Dev per-pair differences; WER linearised as 100 (E_A − E_B) / N̄ with the capped error counts E = min(S + D + I, N) (M3c) for P and K, and with v0's uncapped counts for V (F3 only), the corpus-WER difference of the other definition printed beside its Dev mean.

| endpoint | A − B | Dev n | Dev mean (WER: as fitted) | Dev mean, other WER definition | σa² | σb² | σc² | σe² | converged, optimiser |
|---|---|---|---|---|---|---|---|---|---|
| P1 / P2 | `cand_pm18_rq` − `anc_faiss` | 500 | -68.3 (capped) | -141.2 (uncapped) | 186 | 21.2 | 130 | 899 | yes, powell |
| P3 / P4 | `cand_pm18_rq` − `anc_faiss` | 1000 | -0.1029 | – | 0.000401 | 0.00267 | 9.43e-05 | 0.00777 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| P5 / P6 | `cand_pm18_rq` − `anc_faiss` | 1000 | +0.8055 | – | 0.0141 | 0.0251 | 1.8e-11 | 0.18 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| P7 | `cand_pm18_rq` − `anc_faiss` | 500 | +0.5437 | – | 3.54e-11 | 0.288 | 9.2e-11 | 0.353 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| K1 / K2 | `cand_pm18_rq` − `anc_R1_r0_tau0.4` | 500 | -67.49 (capped) | -118 (uncapped) | 191 | 35.4 | 1.1e-07 | 1.07e+03 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| K3 / K4 | `cand_pm18_rq` − `anc_R1_r0_tau0.4` | 1000 | -0.09155 | – | 0.00032 | 0.00266 | 5.28e-05 | 0.00841 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| K5 / K6 | `cand_pm18_rq` − `anc_R1_r0_tau0.4` | 1000 | +0.9277 | – | 0.0157 | 0.03 | 2e-11 | 0.185 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| K7 | `cand_pm18_rq` − `anc_R1_r0_tau0.4` | 500 | +0.3208 | – | 0.00651 | 0.241 | 3.59e-11 | 0.344 | yes, powell (The MLE may be on the boundary of the parameter space.) |
| L1s / L2s | `cand_pm18_rq@src` − `anc_faiss@src` | 500 | -0.03971 | – | 0.00313 | 0.00239 | 0.000285 | 0.0102 | yes, nm (The MLE may be on the boundary of the parameter space.) |
| L1c / L2c | `cand_pm18_rq@src` − `anc_faiss@src` | 500 | -0.04088 | – | 0.00162 | 0.00499 | 2.27e-12 | 0.0135 | yes, nm (The MLE may be on the boundary of the parameter space.) |

Predicted for the test lists (SE² = σa²/n_s + σb²/n_t + σc²/n_c + σe²/n; MDE = 3.53 SE; gate L: 2.80 SE_g ≤ 0.01 − Δ_dev,g). Approximate: cell imbalance is ignored and the components are treated as known.

| endpoint | design | n | n_s | n_t | n_c | SE | MDE or 2.80 SE | threshold | pass |
|---|---|---|---|---|---|---|---|---|---|
| P1 | vctk2k_5 | 1020 | 20 | 20 | 400 | 3.4025 | 12.0108 | 2.000 | no |
| P1 | vctk2k_10 | 2810 | 20 | 20 | 400 | 3.3190 | 11.7160 | 2.000 | no |
| P2 | ltc2k | 1800 | 36 | 36 | 1260 | 2.5227 | 8.9050 | 2.000 | no |
| P3 | vctk2k_5 | 2000 | 20 | 20 | 400 | 0.0125 | 0.0443 | 0.010 | no |
| P3 | vctk2k_10 | 4000 | 20 | 20 | 400 | 0.0125 | 0.0440 | 0.010 | no |
| P4 | ltc2k | 1800 | 36 | 36 | 1260 | 0.0095 | 0.0334 | 0.010 | no |
| P5 | vctk2k_5 | 2000 | 20 | 20 | 400 | 0.0453 | 0.1599 | 0.050 | no |
| P5 | vctk2k_10 | 4000 | 20 | 20 | 400 | 0.0448 | 0.1581 | 0.050 | no |
| P6 | ltc2k | 1800 | 36 | 36 | 1260 | 0.0345 | 0.1217 | 0.050 | no |
| P7 | vctk2k_5 | 980 | 20 | 20 | 400 | 0.1215 | 0.4287 | 0.100 | no |
| P7 | vctk2k_10 | 1190 | 20 | 20 | 390 | 0.1212 | 0.4278 | 0.100 | no |
| K1 | vctk2k_5 | 1020 | 20 | 20 | 400 | 3.5185 | 12.4204 | 2.000 | no |
| K1 | vctk2k_10 | 2810 | 20 | 20 | 400 | 3.4225 | 12.0815 | 2.000 | no |
| K2 | ltc2k | 1800 | 36 | 36 | 1260 | 2.6248 | 9.2654 | 2.000 | no |
| K3 | vctk2k_5 | 2000 | 20 | 20 | 400 | 0.0124 | 0.0437 | 0.010 | no |
| K3 | vctk2k_10 | 4000 | 20 | 20 | 400 | 0.0123 | 0.0434 | 0.010 | no |
| K4 | ltc2k | 1800 | 36 | 36 | 1260 | 0.0094 | 0.0330 | 0.010 | no |
| K5 | vctk2k_5 | 2000 | 20 | 20 | 400 | 0.0488 | 0.1722 | 0.050 | no |
| K5 | vctk2k_10 | 4000 | 20 | 20 | 400 | 0.0483 | 0.1705 | 0.050 | no |
| K6 | ltc2k | 1800 | 36 | 36 | 1260 | 0.0371 | 0.1308 | 0.050 | no |
| K7 | vctk2k_5 | 980 | 20 | 20 | 400 | 0.1128 | 0.3982 | 0.100 | no |
| K7 | vctk2k_10 | 1190 | 20 | 20 | 390 | 0.1125 | 0.3973 | 0.100 | no |
| L1s | vctk2k_5 | 1000 | 20 | 20 | 200 | 0.0170 | 0.0475 | 0.01 − (-0.0397) = 0.0497 | yes |
| L1s | vctk2k_10 | 2000 | 20 | 20 | 200 | 0.0168 | 0.0471 | 0.01 − (-0.0397) = 0.0497 | yes |
| L2s | ltc2k | 900 | 36 | 36 | 612 | 0.0129 | 0.0360 | 0.01 − (-0.0397) = 0.0497 | yes |
| L1c | vctk2k_5 | 1000 | 20 | 20 | 200 | 0.0186 | 0.0519 | 0.01 − (-0.0409) = 0.0509 | no |
| L1c | vctk2k_10 | 2000 | 20 | 20 | 200 | 0.0184 | 0.0514 | 0.01 − (-0.0409) = 0.0509 | no |
| L2c | ltc2k | 900 | 36 | 36 | 648 | 0.0141 | 0.0395 | 0.01 − (-0.0409) = 0.0509 | yes |

Rebuild rule: conditions failing with 5 targets and passing with 10: none -> T1 uses the 5-target VCTK-2k list. Underpowered on vctk2k_5: ['P1', 'P3', 'P5', 'P7', 'K1', 'K3', 'K5', 'K7', 'L1c'].
wer_primary pairs (summary.json / counted in the list): VCTK-2k 5 targets 1020 / 1020, VCTK-2k 10 targets 2810 / 2810.

## Baselines that will run (§2.4)

| ID | system | repository commit | checkpoint sha256 | sample rate | command |
|---|---|---|---|---|---|
| S5 | knnvc | https://github.com/bshall/knn-vc@c616845c4e309e24d5927f15adbdf277a3d65358 | {'/datastore/nvquoc/crt/mimi_vc/data_storage/torch_cache/hub/checkpoints/WavLM-Large.pt': '6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f', '/datastore/nvquoc/crt/mimi_vc/data_storage/torch_cache/hub/checkpoints/prematch_g_02500000.pt': 'f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5'} | 16000 | `/datastore/nvquoc/crt/envs/mimi_vc/bin/python /datastore/nvquoc/crt/mimi_vc/benchmark/journal/run_baselines.py --pairs /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke/pairs_smoke.jsonl --set dev --budget 30 --out_dir /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke --device cuda --record --systems knnvc` |
| S6 | freevc | https://github.com/OlaWod/FreeVC@60032a7b1b030831ae17a4b6959656d88df6cc9c | {'/datastore/nvquoc/crt/mimi_vc/benchmark/FreeVC/checkpoints/freevc.pth': 'e2cc2d047f63b80d1d6780e37611cec11a01d597560393b1fe6118158b3bd47f', '/datastore/nvquoc/crt/mimi_vc/benchmark/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt': 'bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca', '/datastore/nvquoc/crt/mimi_vc/benchmark/FreeVC/wavlm/WavLM-Large.pt': '6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f'} | 16000 | `/datastore/nvquoc/crt/envs/mimi_vc/bin/python /datastore/nvquoc/crt/mimi_vc/benchmark/journal/run_baselines.py --pairs /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke/pairs_smoke.jsonl --set dev --budget 30 --out_dir /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke --device cuda --record --systems freevc` |
| S7 | seedvc | https://github.com/Plachtaa/seed-vc@51383efd921027683c89e5348211d93ff12ac2a8 | {'/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--Plachta--Seed-VC/snapshots/257283f9f41585055e8f858fba4fd044e5caed6e/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth': '8ec8841b20bb46df9f7e8e570a6946a4b87b940133c7f0e778487ff33841f720', '/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--funasr--campplus/snapshots/e4b6ede7ce16997aff4ae69fbca1f0175e2afede/campplus_cn_common.bin': '3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8', '/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513/bigvgan_generator.pt': 'e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced', '/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d/model.safetensors': '1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b'} | 22050 | `python -u /datastore/nvquoc/crt/mimi_vc/benchmark/journal/run_baselines.py --pairs /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke/pairs_smoke.jsonl --set dev --budget 30 --systems seedvc --out_dir /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke --device cuda --record` |
| S8 | facodec | https://github.com/open-mmlab/Amphion@26f6883110181f1dbfe95c70a7c7dbaf4de5f42a | {'/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--amphion--naturalspeech3_facodec/snapshots/314afc3ea1455ba881a0e484ef9408b6cb996736/ns3_facodec_encoder_v2.bin': '26636b05867f02f8da3690efb8c36f82909f0a8801ccc4bfdc73cdecf5f9c470', '/datastore/nvquoc/crt/mimi_vc/data_storage/hf_cache/hub/models--amphion--naturalspeech3_facodec/snapshots/314afc3ea1455ba881a0e484ef9408b6cb996736/ns3_facodec_decoder_v2.bin': 'e6a38d81916affae40a72f5517f39ebadeec4fefea67b074f21d4ec3a0156e3a'} | 16000 | `/datastore/nvquoc/crt/envs/mimi_vc/bin/python /datastore/nvquoc/crt/mimi_vc/benchmark/journal/run_baselines.py --pairs /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke/pairs_smoke.jsonl --set dev --budget 30 --out_dir /datastore/nvquoc/crt/mimi_vc/benchmark/journal/v1/baselines_smoke --device cuda --record --systems facodec` |

## Listening-test items (§5.4)

32 items from VCTK-2k (5-target list) and LTC-2k, 4 per gender combination and set, seed 20260928. File `/Users/harry/duplex/benchmark/DEV_SELECTION_RECORD_v1.listening_items.json`, sha256 145aa331c4e8b16b9ce5b45a4aebf07108eac758cbda4c412f5f06e6900fc019; it also holds each cell's full seeded order (the replacement order).

| set | pair | rank | pair_id |
|---|---|---|---|
| vctk2k | F2F | 0 | `vctk2k__p361_115__p308` |
| vctk2k | F2F | 1 | `vctk2k__p294_004__p250` |
| vctk2k | F2F | 2 | `vctk2k__p253_010__p261` |
| vctk2k | F2F | 3 | `vctk2k__p253_110__p261` |
| vctk2k | F2M | 0 | `vctk2k__p297_122__p363` |
| vctk2k | F2M | 1 | `vctk2k__p294_129__p275` |
| vctk2k | F2M | 2 | `vctk2k__p361_032__p363` |
| vctk2k | F2M | 3 | `vctk2k__p297_049__p284` |
| vctk2k | M2F | 0 | `vctk2k__p286_022__p299` |
| vctk2k | M2F | 1 | `vctk2k__p254_019__p250` |
| vctk2k | M2F | 2 | `vctk2k__p345_105__p283` |
| vctk2k | M2F | 3 | `vctk2k__p374_022__p362` |
| vctk2k | M2M | 0 | `vctk2k__p304_350__p298` |
| vctk2k | M2M | 1 | `vctk2k__p345_006__p259` |
| vctk2k | M2M | 2 | `vctk2k__p376_286__p363` |
| vctk2k | M2M | 3 | `vctk2k__p260_116__p275` |
| ltc2k | F2F | 0 | `ltc2k__5683_32865_000042_000000__8463` |
| ltc2k | F2F | 1 | `ltc2k__8463_294825_000015_000002__6829` |
| ltc2k | F2F | 2 | `ltc2k__121_127105_000007_000003__8555` |
| ltc2k | F2F | 3 | `ltc2k__4992_41806_000020_000002__1284` |
| ltc2k | F2M | 0 | `ltc2k__1580_141083_000108_000000__2300` |
| ltc2k | F2M | 1 | `ltc2k__8463_294825_000015_000002__6930` |
| ltc2k | F2M | 2 | `ltc2k__2961_961_000004_000017__8230` |
| ltc2k | F2M | 3 | `ltc2k__4446_2275_000038_000004__8224` |
| ltc2k | M2F | 0 | `ltc2k__1089_134691_000025_000003__1284` |
| ltc2k | M2F | 1 | `ltc2k__7729_102255_000016_000008__2961` |
| ltc2k | M2F | 2 | `ltc2k__7729_102255_000027_000001__5142` |
| ltc2k | M2F | 3 | `ltc2k__6930_76324_000002_000004__4507` |
| ltc2k | M2M | 0 | `ltc2k__8224_274384_000032_000001__908` |
| ltc2k | M2M | 1 | `ltc2k__6930_76324_000002_000004__5105` |
| ltc2k | M2M | 2 | `ltc2k__2830_3980_000024_000003__5105` |
| ltc2k | M2M | 3 | `ltc2k__2830_3979_000018_000009__6930` |

