# Exploratory addition X-mvc-1: R1x at 300k on the test sets (mvc/ANALYSIS_PLAN_mvc_v1.md §4.10)

Written 2026-10-29T07:59:15+07:00 by `mvc/analysis/analyze_xmvc1.py` (sha256 d6243ad735dad2f3fd7b233908a9a98f2d2b8488e69194b260f0636f24eb73aa; analyze_mvc.py 9897f7e0e99fc6dbaf09d1ac12cf9b5dbb0b63afc3685741ee74501788a5958c; stats.py a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6; analyze_tests.py 03f2bbf76f79338a46c0ad191b59979c6358dd1f6c7aaee908508a927cf2a933; analyze_exploratory.py ac53e40bac8ca55bf6327f3dbea3bf09268994c0d46465931b585de49a10ced3). Crossed bootstrap B = 10000, seed 20260928: VCTK-2k independent, LTC-2k linked.

- **Exploratory (§4.10).** X-mvc-1 was decided on 29 Oct 2026 from R1x's 300k Dev-S100 numbers, after T1 of mvc_v1 was complete and recorded (DEVIATIONS_mvc_v1.md, operational note 4). It is in no hypothesis family of §5.
- **Descriptive only.** Estimates with 95 % CIs; nothing here is a test or a decision.
- **It never replaces or relabels a mvc_v1 result.** The mvc_v1 numbers below are computed from the T1 rows that TEST_RESULTS_mvc_v1.md analysed (same sha256), with the same functions and seed; the T1 results stand as reported there.
- **mvc_r1x300** is R1x's EMA snapshot at step 300k, `/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/ema_0300000.pt` (sha256 e668c87e664f1cdde6c3c26a04cf2f51551a666e26363edd2c904e67a06cb453), converted with mvc_v1's frozen inference settings: the selection record's mvc_v1.json with only run_dir and ckpt changed, and R1's normalisation statistics (`/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/stats.npz`, sha256 f6aae232b3e493e848d2f030776c7551a175184e6f73717b26ed9049a02935ef, as frozen for R1). 30 s reference budget only, VCTK-2k and LTC-2k.
- R1x continued R1 from step 200k with a restarted learning-rate schedule in its own run directory (§4.10); it never becomes mvc_v1 (§0.3 row 9).
- **R1x's 250k snapshot was unusable**: a transient loss spike at step ~239.4k corrupted the EMA (DEVIATIONS_mvc_v1.md, operational note 3); the EMA had recovered by 300k (operational note 4). No other R1x snapshot runs on the test sets.
- Δ = mvc_r1x300 − mvc_v1 per pair, over the endpoint's pairs where neither conversion failed; failed conversions are counted and, where the counts differ, shown at their worst case (§5.4).

## vctk2k, 30 s

### Levels (95 % CIs)

| System | rows (failed) | capped WER % | SIM | UTMOS | SIM_src | ECAPA SIM | DNSMOS OVRL | DNSMOS P808 |
|---|---|---|---|---|---|---|---|---|
| mvc_r1x300 | 2000 (0) | 2.31 [1.47, 3.19] | 0.881 [0.866, 0.896] | 3.900 [3.821, 3.971] | 0.713 [0.672, 0.751] | 0.444 [0.414, 0.475] | 3.279 [3.248, 3.311] | 3.783 [3.725, 3.841] |
| mvc_v1 | 2000 (0) | 2.51 [1.63, 3.46] | 0.886 [0.871, 0.901] | 3.829 [3.750, 3.902] | 0.707 [0.668, 0.744] | 0.456 [0.426, 0.486] | 3.284 [3.252, 3.316] | 3.793 [3.734, 3.851] |

### mvc_r1x300 − mvc_v1, paired (95 % CIs)

| Quantity | mvc_r1x300 | mvc_v1 | Δ [95 % CI] | SE | n pairs (failed mvc_r1x300/mvc_v1) | failed conversions at their worst case: Δ [95 % CI] |
|---|---|---|---|---|---|---|
| capped WER % | 2.31 | 2.51 | -0.20 [-0.74, 0.28] | 0.2544 | 1020 (0/0) | – |
| SIM | 0.881 | 0.886 | -0.005 [-0.009, -0.002] | 0.0018 | 2000 (0/0) | – |
| UTMOS | 3.900 | 3.829 | 0.071 [0.044, 0.100] | 0.0141 | 2000 (0/0) | – |
| ECAPA SIM | 0.444 | 0.456 | -0.012 [-0.018, -0.006] | 0.0032 | 2000 (0/0) | – |
| DNSMOS OVRL | 3.279 | 3.284 | -0.005 [-0.013, 0.005] | 0.0046 | 2000 (0/0) | – |

### Acoustic guards (95 % CIs)

| System | rows (failed) | hnr | hnr_src | voiced_kept | f0_pcc | st_sd_ratio | rms_db_diff |
|---|---|---|---|---|---|---|---|
| mvc_r1x300 | 2000 (0) | 12.343 [11.641, 13.060] | 10.930 [10.083, 11.807] | 0.929 [0.920, 0.938] | 0.757 [0.713, 0.801] | 0.999 [0.866, 1.148] | -5.909 [-7.381, -4.271] |
| mvc_v1 | 2000 (0) | 11.936 [11.199, 12.701] | 10.930 [10.083, 11.807] | 0.929 [0.919, 0.939] | 0.754 [0.711, 0.796] | 1.018 [0.881, 1.175] | -6.675 [-8.193, -4.994] |

## ltc2k, 30 s

### Levels (95 % CIs)

| System | rows (failed) | capped WER % | SIM | UTMOS | SIM_src | ECAPA SIM | DNSMOS OVRL | DNSMOS P808 |
|---|---|---|---|---|---|---|---|---|
| mvc_r1x300 | 1800 (0) | 2.35 [1.78, 3.00] | 0.927 [0.915, 0.939] | 4.113 [4.067, 4.156] | 0.694 [0.664, 0.727] | 0.626 [0.604, 0.649] | 3.321 [3.293, 3.349] | 3.872 [3.816, 3.923] |
| mvc_v1 | 1800 (0) | 2.37 [1.78, 3.03] | 0.928 [0.917, 0.940] | 4.027 [3.973, 4.079] | 0.691 [0.661, 0.724] | 0.631 [0.610, 0.652] | 3.314 [3.284, 3.343] | 3.865 [3.811, 3.915] |

### mvc_r1x300 − mvc_v1, paired (95 % CIs)

| Quantity | mvc_r1x300 | mvc_v1 | Δ [95 % CI] | SE | n pairs (failed mvc_r1x300/mvc_v1) | failed conversions at their worst case: Δ [95 % CI] |
|---|---|---|---|---|---|---|
| capped WER % | 2.35 | 2.37 | -0.02 [-0.21, 0.16] | 0.0948 | 1800 (0/0) | – |
| SIM | 0.927 | 0.928 | -0.001 [-0.003, 0.000] | 0.0008 | 1800 (0/0) | – |
| UTMOS | 4.113 | 4.027 | 0.086 [0.063, 0.112] | 0.0123 | 1800 (0/0) | – |
| ECAPA SIM | 0.626 | 0.631 | -0.004 [-0.008, -0.001] | 0.0017 | 1800 (0/0) | – |
| DNSMOS OVRL | 3.321 | 3.314 | 0.007 [0.000, 0.015] | 0.0036 | 1800 (0/0) | – |

### Acoustic guards (95 % CIs)

| System | rows (failed) | hnr | hnr_src | voiced_kept | f0_pcc | st_sd_ratio | rms_db_diff |
|---|---|---|---|---|---|---|---|
| mvc_r1x300 | 1800 (0) | 13.804 [13.164, 14.445] | 13.358 [12.715, 14.017] | 0.942 [0.936, 0.947] | 0.882 [0.862, 0.900] | 1.000 [0.953, 1.050] | -10.596 [-10.956, -10.225] |
| mvc_v1 | 1800 (0) | 13.438 [12.791, 14.085] | 13.358 [12.715, 14.017] | 0.940 [0.934, 0.945] | 0.878 [0.857, 0.898] | 1.002 [0.956, 1.050] | -10.820 [-11.176, -10.452] |

## Natural-level leakage NL on VCTK-2k, 30 s (§3.3, computed by analyze_mvc.nl_family)

NL_g = mean over the stratum's pairs with a parallel recording P of SIM_src(system) − SIM_src(target_copy), paired per pair. Δ NL is the paired SIM_src difference mvc_r1x300 − mvc_v1 over the same pairs; target_copy's term cancels pair by pair, so it equals NL(mvc_r1x300) − NL(mvc_v1) when neither system failed on the stratum's pairs.

| Stratum | pairs with P (no source E) | NL mvc_r1x300 [95 % CI] | NL mvc_v1 [95 % CI] | Δ NL [95 % CI] | n (failed mvc_r1x300/mvc_v1) |
|---|---|---|---|---|---|
| same | 491 (0) | 0.022 [0.001, 0.043] | 0.019 [-0.001, 0.040] | 0.003 [-0.001, 0.008] | 491 (0/0) |
| cross | 489 (0) | 0.041 [0.006, 0.075] | 0.028 [-0.002, 0.056] | 0.013 [0.005, 0.024] | 489 (0/0) |

## Inputs

- snapshot: `/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/ema_0300000.pt` sha256 e668c87e664f1cdde6c3c26a04cf2f51551a666e26363edd2c904e67a06cb453
- normalisation statistics: `/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x/stats.npz` sha256 f6aae232b3e493e848d2f030776c7551a175184e6f73717b26ed9049a02935ef (R1's, as frozen)
- changed against mvc_v1.json: {'ckpt': ['ema_0200000.pt', 'ema_0300000.pt'], 'run_dir': ['/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1', '/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1x']}
- X-mvc-1 vctk2k: `job 6996 2026-10-29 06:27:35 snapshot sha256 e668c87e664f1cdde6c3c26a04cf2f51551a666e26363edd2c904e67a06cb453 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- X-mvc-1 ltc2k: `job 7002 2026-10-29 07:30:13 snapshot sha256 e668c87e664f1cdde6c3c26a04cf2f51551a666e26363edd2c904e67a06cb453 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- MimiFlow-VC T1 part t1: `job 6724 2026-10-28 12:13:43 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- MimiFlow-VC T1 part t2: `job 6761 2026-10-28 13:30:09 selection sha256 9f939ae6fb138010b46d4a2060d4c1158fd6cb2f2df7e7c7f4522b188d746a42`
- journal job M part t1: `job 4653 2026-10-20 06:45:22 selection sha256 9c7275f56ac5779235e5838304dbd4edfc319e53f6d4142c77fe537b938bc105`
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/cfg/mvc_r1x300.json` sha256 b6dafc52550eb227d0caebf3c56a39ee72898d3d8bd65f6b08698ef977e02f23
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/snapshot.sha256` sha256 693971b7b07a1fd15e9393d064f890db4418268a8905f98fb7f0954c56f6779e
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/m/t1/metrics/vctk2k_30.jsonl` sha256 4cab8f9579c38c2becd3043d845595885ae68faad213ccad8419177b1368ecd6 (2000 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/m/t1/metrics/vctk2k_30_src.jsonl` sha256 01e86f0a6cb0e36a9181c5438bc02896c4a790374884692cca8df3f83a23e9f1 (2000 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/x2/ecapa/vctk2k_30.jsonl` sha256 25c7ba6f843f382d24b5860b6ff47642f35e5c94ea03fe5802011e9f5911f0c0 (2000 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/x2/ecapa/vctk2k_30_src.jsonl` sha256 2b6ccdd11e818c4a56d6e13381c34a7723ed56d28482bc5f310eebd1c616b074 (2000 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/dnsmos/vctk2k_30_mvc_r1x300.jsonl` sha256 ad3a10d6052cf4c3c11dd4e0cef64e5517afc20f465c83ffeecf3fef601da703 (2000 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/guard/vctk2k_30.jsonl` sha256 cd867d275bc52c973e6abcf31aa543dcca8869f0cae439c76315a68723529170 (2000 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/m/t2/metrics/ltc2k_30.jsonl` sha256 cd5a903002e3c3c9b59bd5714563f2746cf802db37069d889d6cb89a4021d7d6 (1800 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/m/t2/metrics/ltc2k_30_src.jsonl` sha256 e17e0520b25867138b5abce18f6b2e2e85b5026da78331d2bbb2f8aa7d3bf5c1 (1800 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/x2/ecapa/ltc2k_30.jsonl` sha256 22486d6cecaf3f246590b2b942770430799e6b7d8c9be9d821e6bab5916a332b (1800 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/x2/ecapa/ltc2k_30_src.jsonl` sha256 422e70fdff1ff8a6454274d0397e3dfd48a9ccd56768a0d87d3f30bb48816f9e (1800 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/dnsmos/ltc2k_30_mvc_r1x300.jsonl` sha256 ef592de57b2e87ca5321ca73c5f57f578019c89b131c164bc8b02b4f59a9c726 (1800 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/guard/ltc2k_30.jsonl` sha256 62d72c0ec7ba75d5b75045caad6115218d66da3f8c370e0fc54593360913f256 (1800 rows)
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/vctk2k.sha256` sha256 624a6d2394266d542e42a1f39ae58df79d0100c6b11688ac93009f6fca08ed6f
- X-mvc-1: `/Users/harry/duplex/paper/mimi_vc/results/xmvc1/ltc2k.sha256` sha256 628195a40266f78c0c0c6e97dca5498f9919440129979d77806398ce7244937e
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/vctk2k.jsonl` sha256 bdbf36a32b61eff625e2acaf40163835582ca9e391079c11ca72e6bf51601c1d
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/ltc2k.jsonl` sha256 32fc920516f0b6e4d239ddc85994cca8f746538763e5ad71ab2e208203c56e45
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/journal_extension/results/v1/pairs/vctk2k_source_E.json` sha256 ccbd53dfc507ad86b814a5e2cac2f8538f2bc7ca58a5d41f0f3c86e9627cb8ea
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/m/t1/metrics/vctk2k_30.jsonl` sha256 b84fe6fea87f051c7dc239149092cea9315bc2036dc80074c13d9ef0c332e462 (6000 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/m/t1/metrics/vctk2k_30_src.jsonl` sha256 c3be713cd8b69626d08f65b13cb4eeda0c861a83bee8dc2379c41afb5351270d (6000 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/journal_extension/results/v1/m/t1/metrics/vctk2k_30.jsonl` sha256 af3a938254acf692408d1c9da099337ff42ab1981bb3e8bd699c3de1831519f6 (29960 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/journal_extension/results/v1/m/t1/metrics/vctk2k_30_src.jsonl` sha256 43be5f1d6163f60f21d2c299b1d5d92945e99c54f074051ed7f666dace6ea285 (29960 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/vctk2k_30.jsonl` sha256 6ed3653e44d9e9945a502d162e37f7ee33c257f1b43b035adb0a734eaac6bb1d (6000 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/vctk2k_30_src.jsonl` sha256 33cfcabed0f130c9c7245be92550710f9c2c9aeab9fc73f5645e09c035a0ab2d (6000 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/vctk2k_30_mvc_v1.jsonl` sha256 5b4829686b56e22339e24ad9b9fc12658c318af7eb137055b0f3dad9a62dae23 (2000 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/guard/vctk2k_30.jsonl` sha256 fe06233f25a9604cfb259f803c11d0f33d8fbe9c172e6eff08e337eb6552fa38 (14000 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/m/t2/metrics/ltc2k_30.jsonl` sha256 2384bf4277a023dd12581d6809a9d52b1c0d229ad0d68dad2f313802a9473aec (5400 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/m/t2/metrics/ltc2k_30_src.jsonl` sha256 89820ec34ef16bb320d85ea160ef4af29120a79351e3e99dbefaeacb24611282 (5400 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/ltc2k_30.jsonl` sha256 d697cab15a11fda50c5c8cb56095ecc0411eabd72a98e3b3856583ba9b2218b8 (5400 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/x2/ecapa/ltc2k_30_src.jsonl` sha256 085f882d7cc8c87812867d3960497ea029eb62bab4bb1fa349e55898439f7ed1 (5400 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/dnsmos/ltc2k_30_mvc_v1.jsonl` sha256 5af011d32d9a9015e156302568ac214fa988a772b723e2ea8a4cdab7ec4faa1e (1800 rows)
- T1 and journal (inputs of TEST_RESULTS_mvc_v1): `/Users/harry/duplex/paper/mimi_vc/results/guard/ltc2k_30.jsonl` sha256 88a9aa098bfc7f33f34a623e069fc72ac1b0a0f63dd6315dae4f2b7669a225e6 (12600 rows)
- selection record cfg and T1 results: `/Users/harry/duplex/mvc/MVC_SELECTION_RECORD_v1_cfg/mvc_v1.json` sha256 00184073f7c3caae6551e76f38f2e4d508bbdc844488fd9cb63968b04db0fbbd
- selection record cfg and T1 results: `/Users/harry/duplex/mvc/TEST_RESULTS_mvc_v1.json` sha256 7674852b93814402a2e66e7aaf136ec8c61b7c4866953bdc46504f8154f46c40
