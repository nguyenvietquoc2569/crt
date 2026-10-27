# MimiFlow-VC selection record v1 (ANALYSIS_PLAN_mvc_v1.md §4)

Status: **complete**. Written 2026-10-28 06:20:32 by `mvc/analysis/select_mvc.py` (sha256 7bbb93e8369fc76358c03a002029985fbffbc98dfe8891b8ed925090611a8b0a; tests 2897ecc418710295e3f76e960da8836558dd744d8c864d6325f96574361eb8bd) with `benchmark/stats.py` (sha256 a7129260b3c1af19f11cff49d1c4fb0fda9c8da80b70584912e272d2032835d6); plan sha256 46f2b586640fb5a7fee6654343a5403a4f93e96d5e7b1558219f701755bfcd86. Linked crossed bootstrap of the Dev speakers, B = 10000, seed 20260928.

## mvc_v1

- Configuration: **cfg1_w1** (F), round r1.
- Lineage: R1 (/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1: the 121M DiT of its args.json, 200k steps on 501 h of LibriTTS-R), EMA snapshot ema_0200000.pt (sha256 f82d70cc9a6a66e842afe52bcb76cf10840c2bb0df77d159a49b27d026f1920b) chosen by R2.0; B1 not triggered.
- Snapshot: `/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1/ema_0200000.pt`, sha256 f82d70cc9a6a66e842afe52bcb76cf10840c2bb0df77d159a49b27d026f1920b.
- describe() commit: ba09a8efc7678e38 (frozen code: ba09a8efc7678e38; the Mac mvc/*.py at writing: ba09a8efc7678e38).
- mvc/*.py changed since the freeze (new sha256; deviations, §10): none.

| File | sha256 | JSON |
|---|---|---|
| `mvc_v1.json` | 00184073f7c3caae6551e76f38f2e4d508bbdc844488fd9cb63968b04db0fbbd | `{"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "ema_0200000.pt", "steps": 16, "cfg": 1.0, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "latent", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler", "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std", "neg_src": 1.0, "neg_src_until": 1.0}` |
| `mvc_v1_q32.json` | 0238459a3bfefc7c99866d71b49bc68e1c4269f68f03478ae893b43112583088 | `{"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "ema_0200000.pt", "steps": 16, "cfg": 1.0, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "codes", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler", "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std", "neg_src": 1.0, "neg_src_until": 1.0}` |
| `mvc_copy_latent.json` | e896775b66466f7b4b71f2b4ffd99739a784ee770ba1965069da84dcd9e6acb9 | `{"run_dir": null, "ckpt": "ema_last.pt", "steps": 16, "cfg": 0.7, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "latent", "seed_from_pair": true, "mode": "copy_latent", "whisper_src": "clean", "solver": "euler", "temp": 1.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.0, "relax_k": 2, "cfg_branch": "std", "neg_src": 0.0, "neg_src_until": 1.0}` |

## R2.0: snapshot (§4.2)

| Snapshot | run | WER % | SIM | UTMOS | recon WER % | sha256 |
|---|---|---|---|---|---|---|
| ema_0150000.pt | `r2_0/r2_0_k150_ref` | 2.63 | 0.904 | 3.843 | 3.34 | 85d48dc7ba46839da8f058e365cbd32e3cb88feda5104fcb1afc14a623dacc42 |
| ema_0200000.pt | `r2_0/r2_0_k200_ref` | 2.34 | 0.903 | 3.874 | 3.20 | f82d70cc9a6a66e842afe52bcb76cf10840c2bb0df77d159a49b27d026f1920b |

Conditions for 150k: {'recon_wer': False, 'sim': False, 'utmos': False, 'wer': False}. Snapshot: **ema_0200000.pt**. R1x (§4.10): runs ({'sim_rose': False, 'utmos_rose': True, 'runs': True}).

## Round r1: R2.1 Dev-S100 screen (§4.3)

| Config | WER % [CI] | SIM [CI] | UTMOS [CI] | NL same (SE) | NL cross (SE) | summarize.py same / cross | cross ΔSIM_src WavLM / ECAPA | 1 | 2 | 3 | 4 | 5 | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cfg0_w0 | 1.75 [0.00, 5.74] | 0.879 [0.838, 0.911] | 3.739 [3.638, 3.839] | +0.083 (0.027) | +0.112 (0.045) | +0.057 / +0.110 | – | no | yes | yes | yes | yes | w 0: condition 1 fails |
| cfg0_w0.5 | 1.46 [0.00, 4.67] | 0.902 [0.867, 0.930] | 3.825 [3.719, 3.924] | +0.038 (0.028) | +0.061 (0.048) | +0.011 / +0.065 | -0.045 / -0.026 | no | yes | yes | yes | yes | fails 1 |
| cfg0_w0.75 | 2.05 [0.00, 5.96] | 0.905 [0.871, 0.932] | 3.853 [3.738, 3.955] | +0.020 (0.028) | +0.040 (0.049) | -0.008 / +0.048 | -0.062 / -0.033 | no | yes | yes | yes | yes | fails 1 |
| cfg0_w1 | 2.34 [0.00, 6.36] | 0.903 [0.871, 0.931] | 3.874 [3.747, 3.986] | +0.005 (0.027) | +0.022 (0.050) | -0.024 / +0.034 | -0.076 / -0.039 | yes | yes | yes | yes | yes | conditions 1-5 met |
| cfg1_w0 | 1.46 [0.00, 5.46] | 0.904 [0.865, 0.932] | 3.810 [3.702, 3.907] | +0.054 (0.025) | +0.075 (0.045) | +0.032 / +0.079 | – | no | yes | yes | yes | yes | w 0: condition 1 fails |
| cfg1_w0.5 | 2.05 [0.00, 5.24] | 0.910 [0.877, 0.937] | 3.872 [3.750, 3.981] | +0.026 (0.022) | +0.043 (0.045) | +0.001 / +0.050 | -0.029 / -0.018 | no | yes | yes | yes | yes | fails 1 |
| cfg1_w0.75 | 1.75 [0.00, 5.00] | 0.911 [0.879, 0.937] | 3.892 [3.770, 4.000] | +0.011 (0.023) | +0.028 (0.045) | -0.017 / +0.037 | -0.042 / -0.021 | yes | yes | yes | yes | no | fails 5 |
| cfg1_w1 | 1.46 [0.00, 3.99] | 0.909 [0.879, 0.936] | 3.887 [3.760, 3.997] | -0.003 (0.024) | +0.019 (0.045) | -0.029 / +0.029 | -0.050 / -0.027 | yes | yes | yes | yes | yes | conditions 1-5 met |
| cfg0_w0.5_u0.5 | 1.17 [0.00, 4.41] | 0.900 [0.864, 0.928] | 3.825 [3.720, 3.922] | +0.042 (0.028) | +0.064 (0.048) | +0.015 / +0.068 | -0.042 / -0.024 | no | yes | yes | yes | yes | not eligible: the plain cell does not fail condition 3 alone; fails 1 |
| cfg0_w0.75_u0.5 | 2.05 [0.00, 5.96] | 0.903 [0.869, 0.931] | 3.857 [3.746, 3.949] | +0.026 (0.028) | +0.043 (0.048) | -0.003 / +0.050 | -0.060 / -0.030 | no | yes | yes | yes | no | not eligible: the plain cell does not fail condition 3 alone; fails 1, 5 |
| cfg0_w1_u0.5 | 2.05 [0.00, 5.96] | 0.901 [0.869, 0.929] | 3.887 [3.759, 4.000] | +0.012 (0.027) | +0.028 (0.049) | -0.018 / +0.038 | -0.072 / -0.036 | yes | yes | yes | yes | no | not eligible: the plain cell does not fail condition 3 alone; fails 5 |

D-mvc-2 pick on this snapshot: **cfg0_w1**; B1 trigger i: False.

## Round r1: R2.2 finalists (§4.4)

the D-mvc-2 pick cfg0_w1, then the eligible configurations within SIM >= best - 0.010 and UTMOS >= best - 0.05 by R2.3's order: cfg0_w1, cfg1_w1.

## Round r1: R2.3 on full Dev (§4.5)

| Config | WER % [CI] | uncapped WER % | SIM [CI] | UTMOS [CI] | NL same ± SE (summarize.py) | NL cross ± SE (summarize.py) | ECAPA SIM | ECAPA SIM_src same / cross | M9 same / cross (share D > 0) | L-pass | WER check | E3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cfg0_w0 (nosng) | 1.70 [0.75, 2.79] | 1.70 | 0.884 [0.856, 0.910] | 3.762 [3.683, 3.837] | +0.072 ± 0.011 (+0.061) | +0.108 ± 0.022 (+0.099) | 0.353 [0.323, 0.382] | 0.243 [0.209, 0.279] / 0.131 [0.113, 0.149] | -0.031 (0.35) / -0.169 (0.09) | – | – | – |
| cfg0_w1 | 1.90 [0.97, 2.99] | 1.90 | 0.904 [0.881, 0.923] | 3.858 [3.784, 3.933] | -0.002 ± 0.013 (-0.008) | +0.027 ± 0.023 (+0.020) | 0.439 [0.417, 0.460] | 0.141 [0.117, 0.169] / 0.088 [0.070, 0.108] | -0.112 (0.12) / -0.275 (0.02) | False | True | False |
| cfg1_w1 | 1.84 [1.07, 2.71] | 1.84 | 0.912 [0.890, 0.930] | 3.865 [3.791, 3.942] | -0.007 ± 0.012 (-0.011) | +0.020 ± 0.022 (+0.013) | 0.493 [0.471, 0.514] | 0.145 [0.120, 0.174] / 0.092 [0.074, 0.111] | -0.122 (0.11) / -0.292 (0.01) | False | True | False |

Best SIM –, best UTMOS –; E3 order: empty; fallback F: cfg1_w1.

| Config | Pair | WER % | SIM | UTMOS | SIM_src |
|---|---|---|---|---|---|
| cfg0_w0 | F2F | 1.93 [0.00, 5.38] | 0.919 [0.895, 0.943] | 3.843 [3.770, 3.918] | 0.882 [0.831, 0.921] |
| cfg0_w0 | F2M | 2.00 [0.70, 3.60] | 0.876 [0.848, 0.901] | 3.680 [3.567, 3.783] | 0.705 [0.620, 0.781] |
| cfg0_w0 | M2F | 1.30 [0.00, 3.66] | 0.867 [0.810, 0.912] | 3.757 [3.634, 3.882] | 0.700 [0.621, 0.763] |
| cfg0_w0 | M2M | 1.53 [0.30, 3.28] | 0.876 [0.841, 0.907] | 3.766 [3.637, 3.883] | 0.850 [0.799, 0.887] |
| cfg0_w1 | F2F | 1.82 [0.00, 4.45] | 0.933 [0.912, 0.953] | 3.817 [3.705, 3.914] | 0.819 [0.746, 0.879] |
| cfg0_w1 | F2M | 2.78 [1.11, 4.60] | 0.899 [0.878, 0.918] | 3.880 [3.791, 3.965] | 0.645 [0.564, 0.724] |
| cfg0_w1 | M2F | 1.30 [0.13, 3.31] | 0.899 [0.867, 0.923] | 3.903 [3.807, 4.008] | 0.603 [0.516, 0.682] |
| cfg0_w1 | M2M | 1.65 [0.41, 3.51] | 0.883 [0.851, 0.910] | 3.830 [3.695, 3.959] | 0.773 [0.696, 0.834] |
| cfg1_w1 | F2F | 1.93 [0.00, 4.20] | 0.938 [0.919, 0.958] | 3.798 [3.698, 3.898] | 0.813 [0.736, 0.875] |
| cfg1_w1 | F2M | 2.33 [0.79, 4.23] | 0.906 [0.886, 0.924] | 3.901 [3.816, 3.986] | 0.636 [0.556, 0.717] |
| cfg1_w1 | M2F | 1.54 [0.24, 3.62] | 0.911 [0.882, 0.933] | 3.905 [3.808, 4.007] | 0.598 [0.507, 0.681] |
| cfg1_w1 | M2M | 1.53 [0.37, 3.10] | 0.894 [0.859, 0.920] | 3.856 [3.728, 3.980] | 0.773 [0.694, 0.835] |

## Round r1: R2.3 on Dev-900 (selected on Dev) (§4.5)

| Config | WER % [CI] | uncapped WER % | SIM [CI] | UTMOS [CI] | NL same ± SE (summarize.py) | NL cross ± SE (summarize.py) | ECAPA SIM | ECAPA SIM_src same / cross | M9 same / cross (share D > 0) | L-pass | WER check | E3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cfg0_w0 (nosng) | 1.69 [0.75, 2.84] | 1.69 | 0.885 [0.858, 0.911] | 3.764 [3.679, 3.844] | +0.070 ± 0.012 (+0.062) | +0.107 ± 0.021 (+0.098) | 0.354 [0.323, 0.383] | 0.243 [0.210, 0.279] / 0.130 [0.112, 0.149] | -0.030 (0.34) / -0.167 (0.09) | – | – | – |
| cfg0_w1 | 1.85 [0.92, 2.96] | 1.85 | 0.904 [0.882, 0.923] | 3.856 [3.780, 3.932] | -0.003 ± 0.013 (-0.007) | +0.028 ± 0.021 (+0.018) | 0.439 [0.416, 0.461] | 0.142 [0.118, 0.169] / 0.087 [0.069, 0.107] | -0.109 (0.14) / -0.274 (0.02) | False | True | False |
| cfg1_w1 | 1.89 [1.05, 2.83] | 1.89 | 0.913 [0.891, 0.930] | 3.863 [3.788, 3.940] | -0.008 ± 0.013 (-0.009) | +0.020 ± 0.021 (+0.011) | 0.494 [0.471, 0.515] | 0.146 [0.121, 0.174] / 0.091 [0.072, 0.109] | -0.119 (0.12) / -0.291 (0.02) | False | True | False |

Best SIM –, best UTMOS –; E3 order: empty; fallback F: cfg1_w1.

| Config | Pair | WER % | SIM | UTMOS | SIM_src |
|---|---|---|---|---|---|
| cfg0_w0 | F2F | 2.19 [0.00, 5.91] | 0.919 [0.894, 0.944] | 3.860 [3.787, 3.934] | 0.882 [0.832, 0.921] |
| cfg0_w0 | F2M | 2.20 [0.79, 3.91] | 0.877 [0.847, 0.903] | 3.669 [3.547, 3.780] | 0.709 [0.624, 0.785] |
| cfg0_w0 | M2F | 1.17 [0.00, 3.22] | 0.868 [0.812, 0.911] | 3.765 [3.635, 3.893] | 0.701 [0.624, 0.763] |
| cfg0_w0 | M2M | 1.18 [0.15, 2.46] | 0.876 [0.840, 0.909] | 3.761 [3.628, 3.882] | 0.852 [0.800, 0.888] |
| cfg0_w1 | F2F | 2.06 [0.00, 5.13] | 0.933 [0.912, 0.954] | 3.832 [3.713, 3.929] | 0.819 [0.745, 0.880] |
| cfg0_w1 | F2M | 2.69 [0.99, 4.60] | 0.900 [0.878, 0.918] | 3.870 [3.779, 3.959] | 0.648 [0.566, 0.728] |
| cfg0_w1 | M2F | 1.17 [0.00, 3.22] | 0.900 [0.872, 0.922] | 3.899 [3.803, 4.005] | 0.604 [0.518, 0.683] |
| cfg0_w1 | M2M | 1.44 [0.33, 2.75] | 0.882 [0.849, 0.910] | 3.821 [3.682, 3.953] | 0.778 [0.702, 0.837] |
| cfg1_w1 | F2F | 2.19 [0.00, 4.69] | 0.938 [0.918, 0.958] | 3.807 [3.704, 3.908] | 0.813 [0.735, 0.877] |
| cfg1_w1 | F2M | 2.57 [0.90, 4.57] | 0.907 [0.885, 0.924] | 3.892 [3.808, 3.980] | 0.639 [0.556, 0.720] |
| cfg1_w1 | M2F | 1.30 [0.00, 3.43] | 0.913 [0.889, 0.932] | 3.902 [3.804, 4.005] | 0.599 [0.508, 0.682] |
| cfg1_w1 | M2M | 1.44 [0.30, 2.87] | 0.893 [0.857, 0.920] | 3.849 [3.716, 3.981] | 0.779 [0.701, 0.838] |

Dev-900 check: first on 1,000 pairs cfg1_w1, on Dev-900 cfg1_w1 (agree).

## Round r1: R2.4 reference budgets (§4.6)

- cfg1_w1: SIM 3 s 0.879, 30 s 0.912, 180 s 0.918 -> passes
- Choice: **cfg1_w1** (F); R2.4 failure recorded: False.

Triggers: B1 (i) False, (ii) False -> B1 not triggered; R5 (reported only) False.

## Dev counterparts of family RB (mvc_v1 on full Dev)

| Budget | WER % | SIM | UTMOS |
|---|---|---|---|
| 3 s | 1.90 [0.87, 3.06] | 0.879 [0.852, 0.904] | 3.801 [3.706, 3.881] |
| 30 s | 1.84 [1.07, 2.71] | 0.912 [0.890, 0.930] | 3.865 [3.791, 3.942] |
| 180 s | 2.05 [1.05, 3.29] | 0.918 [0.895, 0.936] | 3.861 [3.788, 3.935] |

## Dev configurations ever scored (§4.8)

Before R2 (summary.json files under the TAGs of the jobs before the freeze):
- Dev-S100: 7 at 3 s, 101 at 30 s, 9 at 180 s
- screen 0/20: 6 at 30 s
- plus the R2 runs (summary.json files under the R2 and B1 TAGs, used or not): 18, of which 18 used by the selection and 0 not used (refused or repeated); by pair set: Dev-S100 13; full Dev 5. The dev-clean reconstructions of R2.0 are no Dev configuration and are not counted.
- census file sha256 01e571d4beb4a20ff3894203455f9e00cda57a9904413e450f121291c48bd018.

## Other items (§4.8)

- analyze_mvc.py and its tests: `analyze_mvc.py` 9897f7e0e99fc6dbaf09d1ac12cf9b5dbb0b63afc3685741ee74501788a5958c; `test_analyze_mvc.py` 0818761f5f4489651e7114be6f0dc2df3cbb055faa7ac899d0daa7ba63f0fb10
- Contamination test (PLAN §3.1): **passed**; manifest `paper/mimi_vc/results/r1_meta/r1_train_ids.txt` sha256 9b07d52949fdac12d754655bfde5f24eea9021703347ca0f598397202eaede9f; `/Users/harry/duplex/.venv-stats/bin/python mvc/analysis/contamination_r1.py --shards_root paper/mimi_vc/results/r1_meta/mimi --exclude paper/mimi_vc/results/r1_meta/exclude.txt --pairs paper/mimi_vc/results/r1_meta/pairs/dev.jsonl,paper/mimi_vc/results/r1_meta/pairs/vctk2k.jsonl,paper/mimi_vc/results/r1_meta/pairs/ltc2k.jsonl --out_manifest paper/mimi_vc/results/r1_meta/r1_train_ids.txt --out_json paper/mimi_vc/results/r1_meta/contamination.json`; evidence file sha256 48a66eea363b0b7e4c266a8b329fd063add00d70b334c86a8a1c09b2ed82df21
- R1 run files (§4.10): `args.json` 2bb570f3a5a00dbc0ec3c15e149351aa6f07085eb00406d97f023d4225e673e6; `ema_0010000.pt` 3595e23b6f5d3c9e856697008fa3dd748cc0b550b78a4142e5fbe219795e122d; `ema_0020000.pt` db6e4b49a133418e42ec073203df46668a902fbbbd8f1d35f4eedc8f36146954; `ema_0030000.pt` b1e52cfd62a7062eb1b353b942c64b313a893247d4c0fd3c63d7d950c25d5f31; `ema_0040000.pt` f336973c9311c660953ea878ef3c42b97f3ad41affe788a5b29a5d3b5f0a23ea; `ema_0050000.pt` edb963b09ff64ec81f5317a3c403b6b32ed6249691953c1846a111eacfbb7dfa; `ema_0060000.pt` 1c667764483c44e70deedce3aa49ee0b66a5427fab86e38f7e662d31e27474e4; `ema_0070000.pt` ba6f6c46ef54fde3c0a48120ee9319f962b7bcea3db711b2ccdb3b033e00c8a9; `ema_0080000.pt` a8524e6359876d33122aaaef97725996458cdb76b12a8a4d1d673043cbce1077; `ema_0090000.pt` 39dd65902eec57a243356c066e408c2fe49c83ed12fc5435b3ea2c69c44a5f16; `ema_0100000.pt` 2aa3000328f6c60d0c01f85e4ef80cb32559187b5ca185ab18b88f57d8a45a7f; `ema_0110000.pt` 94eb159b1acde2e084f969fdf11dac26ae5d15fa1aa3e6ec95b36064817097d3; `ema_0120000.pt` bf706aec77a9f9f56a6bf1388df523867ed1a432988c916011d300bc9095b7d3; `ema_0130000.pt` 07531faf45dd1d1effe67e6d6449f5f396815b1184273ff5b5129aa211df198f; `ema_0140000.pt` 1c44574e2a0482dd93cd95bfa2d8728fa715a3f40c46ab5d9a1b623a6eb38322; `ema_0150000.pt` 85d48dc7ba46839da8f058e365cbd32e3cb88feda5104fcb1afc14a623dacc42; `ema_0160000.pt` 6bc07825ffccc1175e4a5d284507bb978e7155cfd4b61eeb81f0986a7eaf4baa; `ema_0170000.pt` 7b58e8703f9d3e5f23976fae73546ad70661be195beb8f2774b91c3960004585; `ema_0180000.pt` 37fa4cc8ddfedcb25114b08062e2bbd0b9811aa4491d3857dcb0a2d827f33dff; `ema_0190000.pt` f6f2a36fc1b981cb752e4adf358222be48a191a32aa0498b455d33550011c09a; `ema_0200000.pt` f82d70cc9a6a66e842afe52bcb76cf10840c2bb0df77d159a49b27d026f1920b; `ema_last.pt` b971ac622516365758683b7ec1ab81f7425f3babdae5d8afbd71c58ec637e48b; `last.pt` c714d185b772ae2952a665f2d1ddd88c5e204628ba75f85bc11f268ad7d48fc7; `stats.npz` f6aae232b3e493e848d2f030776c7551a175184e6f73717b26ed9049a02935ef
- DNSMOS: DNSMOS runs on the Mac (§3.7): the 30 s WAVs named by the metric rows are copied read-only from ceai to MR/wav/ with their absolute path kept below that prefix, and the rows are rewritten to point there; no audio is converted or edited. Models: $DS/models/dnsmos/ and paper/mimi_vc/models/dnsmos/.
- ceai sha256 of the pair lists and the source-E file: `/datastore/ceai/crt/mimi_vc/benchmark/journal/pairs/dev.jsonl` abedac9696ec5c486d2b79f2256c955ec566f669df81524bc605fa25188133b9; `/datastore/ceai/crt/mimi_vc/benchmark/journal/pairs/vctk2k.jsonl` 242727e0570365f8a030d27cccacdf1b4a35aa38f21cfe0e99c523d3bb0ea309; `/datastore/ceai/crt/mimi_vc/benchmark/journal/pairs/ltc2k.jsonl` 882824e969221ae06966eb57e6de5f98e391940da14e66e16ee6296512c825bf; `/datastore/ceai/crt/mimi_vc/benchmark/journal/pairs/vctk2k_source_E.json` da16fdd176e0eda257f1f6e250e7eda9034d991506cc84a825c308ae2dffcc2f; `/datastore/ceai/crt/mimi_vc/mvc/pairs/dev_s100.txt` fbd7b632b2fbef37e44f2e1e8458da5b8374226da4b071fb7ffae3ba0a314a8d

| Job (TAG) | ID | Submit | End | State |
|---|---|---|---|---|
| r2_0 | 6625 | 2026-10-28T01:51:42 | 2026-10-28T02:05:41 | COMPLETED |
| r2_0rec | 6625 | 2026-10-28T01:51:42 | 2026-10-28T02:05:41 | COMPLETED |
| r2_1 | 6633 | 2026-10-28T02:54:50 | 2026-10-28T03:26:17 | COMPLETED |
| r2_3 | 6641 | 2026-10-28T04:11:50 | 2026-10-28T04:53:11 | COMPLETED |
| r2_4 | 6644 | 2026-10-28T05:13:48 | 2026-10-28T05:57:51 | COMPLETED |

## Inputs

- Dev pairs: `/Users/harry/duplex/paper/mimi_vc/results/r2/dev.jsonl` sha256 abedac9696ec5c486d2b79f2256c955ec566f669df81524bc605fa25188133b9
- Dev-S100: sha256 fbd7b632b2fbef37e44f2e1e8458da5b8374226da4b071fb7ffae3ba0a314a8d
- target_copy@src: ceai $J/v1/d2/controls/metrics_src.jsonl, sha256 4f11b82e480630697b683ee38c5e76bf30e9a9a231587a9e2d08c78f95262289
- Runs used: 18 Dev runs; reconstructions ['r2_0rec/r2_0_k150_rec', 'r2_0rec/r2_0_k200_rec'].

| File (under $MVC/eval) | sha256 |
|---|---|
| `r2_0/r2_0_k150_ref/conv/dev/30/r2_0_k150_ref/runs.jsonl` | 98a6425987739fe5eeb81a0609ad1518808d34868f3c3a8764835cbc179b5fc9 |
| `r2_0/r2_0_k150_ref/metrics/dev_30.jsonl` | 722e774e6360d2d2b8bd6bb0e23b971c20108ce997ba7641113595f5713f1496 |
| `r2_0/r2_0_k150_ref/metrics/dev_30_src.jsonl` | 36159a86aaabc4c504f760da91c79b8c0a5b1dfa8335fcb528a9d46eed671ede |
| `r2_0/r2_0_k200_ref/conv/dev/30/r2_0_k200_ref/runs.jsonl` | 8cc6a50cc3cb82f307f3d9dba11ca883ecbe412ef5636fa359a6a4667a03be1b |
| `r2_0/r2_0_k200_ref/metrics/dev_30.jsonl` | 72cf249360ea145dfff74d94546e5826ebcec8dd319b60ba7ed89b9c46cb61c6 |
| `r2_0/r2_0_k200_ref/metrics/dev_30_src.jsonl` | 600567f08556b5cbc83ac440d0162c3465b37f48f48a5c5d2bdb52d440e15188 |
| `r2_0rec/r2_0_k150_rec/metrics.jsonl` | c0c8fcc785d117fd884cb54417726ee3cfbc179fe6fa8157824e7fb29902092d |
| `r2_0rec/r2_0_k200_rec/metrics.jsonl` | 1cdfea8043d279fbdf4fdbb3d9a09b131b929531fa7a20faa91257736de63d71 |
| `r2_1/r2_1_c0_w0.5/conv/dev/30/r2_1_c0_w0.5/runs.jsonl` | b817380d058d069c3b1e447876b395239f31be024bba55bf0289a3ba99253447 |
| `r2_1/r2_1_c0_w0.5/ecapa/dev_30.jsonl` | f77dd14219e2e2fa6d1a235ba151d7c948adff3cf2ddfe91302f2d7aef121f5b |
| `r2_1/r2_1_c0_w0.5/ecapa/dev_30_src.jsonl` | fca5505b27b1542a5189edbd0117bc90bca1b9711f2c15574d019fddb662f4e2 |
| `r2_1/r2_1_c0_w0.5/metrics/dev_30.jsonl` | 9ff4d7f04e3cbfb489d276403b15eda2240023cf29b3404a02f48e27c4a86c0e |
| `r2_1/r2_1_c0_w0.5/metrics/dev_30_src.jsonl` | 962246dbbb69ab3a5a6b7339da86bf966479bf27b0b51917e8d2bb01a1415d0d |
| `r2_1/r2_1_c0_w0.5_u0.5/conv/dev/30/r2_1_c0_w0.5_u0.5/runs.jsonl` | a774270f6dd55b9c4d6b23f78ff4d43e2ec2ced9ed31ff371a98149b7ac2bd48 |
| `r2_1/r2_1_c0_w0.5_u0.5/ecapa/dev_30.jsonl` | 357bd86f1e635063f0fbb84706282128e6045ada8b55ef749f0b7b8e3052d770 |
| `r2_1/r2_1_c0_w0.5_u0.5/ecapa/dev_30_src.jsonl` | 54fc1d785b016e41768bd09d938d1108425c3f5d796e777ad2fd0370b5ac0e4a |
| `r2_1/r2_1_c0_w0.5_u0.5/metrics/dev_30.jsonl` | 5e46e3ac7a12dca6f621c3f92091cf785a86deff97a991c2eb8b3de9d3b9b215 |
| `r2_1/r2_1_c0_w0.5_u0.5/metrics/dev_30_src.jsonl` | c1bad4ad05ba1a4851e53fb058eb445187f82bc24fda2206daa39d0b4d72ccca |
| `r2_1/r2_1_c0_w0.75/conv/dev/30/r2_1_c0_w0.75/runs.jsonl` | 67422b9fcfa2bd5fc22a6f655c461f4b6676f9da3d7b23e3a0d0a7b73253b441 |
| `r2_1/r2_1_c0_w0.75/ecapa/dev_30.jsonl` | 35f0b26882ecdd8aa9671fa5acdcfaed28d935bbc9bd7d07e343edc84b2cb0fe |
| `r2_1/r2_1_c0_w0.75/ecapa/dev_30_src.jsonl` | eaf0c2db54c31def98fb2af9b6dbde2f975d8fe7b6e51bad22cdbdc930eeb9c0 |
| `r2_1/r2_1_c0_w0.75/metrics/dev_30.jsonl` | 8ded8b411bfd9b2cb9f30e15aa4519ad43270e71e7d0f6c2a8c2ac3d1aaea2d4 |
| `r2_1/r2_1_c0_w0.75/metrics/dev_30_src.jsonl` | e361f9475f2202a2f2e95ab3d7d0843ebec2c012b7318f80e402d156002d78ae |
| `r2_1/r2_1_c0_w0.75_u0.5/conv/dev/30/r2_1_c0_w0.75_u0.5/runs.jsonl` | e47f30c0252bf1945f56913ce11908c4615e4801674aa68a999060dac7a31cde |
| `r2_1/r2_1_c0_w0.75_u0.5/ecapa/dev_30.jsonl` | ea63a98e126652b2c9ea2f6f1b2424f2c272ed082089aac0a75bd63f37cde75b |
| `r2_1/r2_1_c0_w0.75_u0.5/ecapa/dev_30_src.jsonl` | 6c10c6ba6ba8406ae9385c7026f3f870252e6e16c570aea9dae9c12f4acdd5d4 |
| `r2_1/r2_1_c0_w0.75_u0.5/metrics/dev_30.jsonl` | e4a7e25068f46389e72d159029dac0aae0e494907cfc1ad5dd274a206bb1da7d |
| `r2_1/r2_1_c0_w0.75_u0.5/metrics/dev_30_src.jsonl` | d633290b5f3ed78249de6397991594115d287a9a32119deefa868757cd7e18d5 |
| `r2_1/r2_1_c0_w0/conv/dev/30/r2_1_c0_w0/runs.jsonl` | 5692d481c25e1d45fe259eca78bcc4e7b2e9bdf553a3b1db412821161802b0fe |
| `r2_1/r2_1_c0_w0/ecapa/dev_30.jsonl` | 783c33b3c07e0cd02699cad3e5848a92e4e21fc30c22b0a8f54b5ba717eff6da |
| `r2_1/r2_1_c0_w0/ecapa/dev_30_src.jsonl` | 2d7337e9bb2791605b4c64cb0615c9ce9bc112c20ceb60072b5924766be0ec12 |
| `r2_1/r2_1_c0_w0/metrics/dev_30.jsonl` | c55d804059a554b5708e3cecfe14d7a98f6ad8d59f69b336490b36702c4d10a6 |
| `r2_1/r2_1_c0_w0/metrics/dev_30_src.jsonl` | 0c46b96ba93c2876623dee938775655ee1a0003699cab5e39578580f2a439fa7 |
| `r2_1/r2_1_c0_w1/conv/dev/30/r2_1_c0_w1/runs.jsonl` | 0e2a7c59603ce5a8c84a589f6734e046e610e07fb9437b7bf6698519dab2a003 |
| `r2_1/r2_1_c0_w1/ecapa/dev_30.jsonl` | 74afade8d2140a65bb224182287ece1adff0c76839f77f85e60d9f99927346b5 |
| `r2_1/r2_1_c0_w1/ecapa/dev_30_src.jsonl` | 2578bf60954c060a668b385acc82d803cb28adcad446b5f60d373dbce4cd9f98 |
| `r2_1/r2_1_c0_w1/metrics/dev_30.jsonl` | 8e4ec67e84e7c659924973d41de3ec95b1f773009ffb0d321c6ff1b5b0fa377b |
| `r2_1/r2_1_c0_w1/metrics/dev_30_src.jsonl` | e6f51cceb963f9e43697351d949d5289d1d83b548c993f3c04084914fa51bccb |
| `r2_1/r2_1_c0_w1_u0.5/conv/dev/30/r2_1_c0_w1_u0.5/runs.jsonl` | 4e7b2385f651cdbebd8c1d434d91f3117db16d2e492e38af364c31883d65576a |
| `r2_1/r2_1_c0_w1_u0.5/ecapa/dev_30.jsonl` | bd847db41267ad3ba42fbcb4b3b570a100438d669eb2037cd799d4ae482fbd9c |
| `r2_1/r2_1_c0_w1_u0.5/ecapa/dev_30_src.jsonl` | 13ab75e56ac9a87a2b6b8fa40c28cb06fa1349ecaccba362dbe2b27d6a21991f |
| `r2_1/r2_1_c0_w1_u0.5/metrics/dev_30.jsonl` | 23059c11e2d12968736e783497bd0e071a6dc5a9fc5deeac1bc5041dacec8e4a |
| `r2_1/r2_1_c0_w1_u0.5/metrics/dev_30_src.jsonl` | a7176129f1e1dd44476a649790ceabce68203aef7f661c04fcd16bfa13741b2b |
| `r2_1/r2_1_c1_w0.5/conv/dev/30/r2_1_c1_w0.5/runs.jsonl` | 3269970080633b6c35e658749ba7416e68e62ebb652d0ed197f99efc5cd39674 |
| `r2_1/r2_1_c1_w0.5/ecapa/dev_30.jsonl` | d83d2dd6bf88f8805ef49fc87411e82d4f570924306429fc948e6e6715306316 |
| `r2_1/r2_1_c1_w0.5/ecapa/dev_30_src.jsonl` | 7d0efaf0d9b661ab27d0782091dc0e8cd7643a3a26d7a20c0c112cd0d88f191b |
| `r2_1/r2_1_c1_w0.5/metrics/dev_30.jsonl` | 4f9a03339668e55b32f51cea1f40c9462f4b06282d341a82e14d0d8f30ccaec6 |
| `r2_1/r2_1_c1_w0.5/metrics/dev_30_src.jsonl` | ed6b6e6a2cb4cea55234c290e25173b08782e19b78c21a56ab379dd1a53a1749 |
| `r2_1/r2_1_c1_w0.75/conv/dev/30/r2_1_c1_w0.75/runs.jsonl` | 4ac4800a08b86fd3d81d583968059082ab3e16876d8275c2b3c545c5a38666b9 |
| `r2_1/r2_1_c1_w0.75/ecapa/dev_30.jsonl` | 7ce14d1bbd420fa1f5eeb6bc20b077710e883afb6714777c3c19fb4614c4f46c |
| `r2_1/r2_1_c1_w0.75/ecapa/dev_30_src.jsonl` | c2d697c9cee7a5fe2c5f274fb415028ac912f7550b5ae6b5c1f07d039729cb0f |
| `r2_1/r2_1_c1_w0.75/metrics/dev_30.jsonl` | fecc312160c3ae5939f141f371fe93634fc97d3490c3f24c71f98cae16d3df6e |
| `r2_1/r2_1_c1_w0.75/metrics/dev_30_src.jsonl` | 9df685bcf996fbe72f574555869a9e90d676aaf8440502906d8eae040874704c |
| `r2_1/r2_1_c1_w0/conv/dev/30/r2_1_c1_w0/runs.jsonl` | 917a5a11a8ab8369597ddbac105746503495f56fa825abda105a7eeed9180fbc |
| `r2_1/r2_1_c1_w0/ecapa/dev_30.jsonl` | 12a5b27ec453ef08ab8c1f66dcab06c90a9b268f88a40ca79ea2014c0a673f2c |
| `r2_1/r2_1_c1_w0/ecapa/dev_30_src.jsonl` | fbdacd4ee7654a8597a860f047c4b1e4f5799fcbc9f495b2436b103c49fc0e9b |
| `r2_1/r2_1_c1_w0/metrics/dev_30.jsonl` | e44e6eb6f344c8a09d09ea744cf658ce37317c38b4226c03aa9b52bb7c5bdd72 |
| `r2_1/r2_1_c1_w0/metrics/dev_30_src.jsonl` | 4a57fea3b0aa7bfda447b358fd0e51d60b9bcba92044e491d043c448e26c30c1 |
| `r2_1/r2_1_c1_w1/conv/dev/30/r2_1_c1_w1/runs.jsonl` | a1ecdd1d2a05eb8c75cdebebb8e965a4ea5c0788e8bc0a61eb5a6f381a51c7a5 |
| `r2_1/r2_1_c1_w1/ecapa/dev_30.jsonl` | 85b63bd545b889e632c6f0e0784a9a17c5904aa10884692f3040209131638472 |
| `r2_1/r2_1_c1_w1/ecapa/dev_30_src.jsonl` | 2e4e5a472d97aa0f6f73c52fc30a76054fb8b2da58155e5ab7a9bae82601e2a3 |
| `r2_1/r2_1_c1_w1/metrics/dev_30.jsonl` | 1ec7151df10ee6483e027ba09d23cf96d3b907dea0e67168ae28ffc93b3318eb |
| `r2_1/r2_1_c1_w1/metrics/dev_30_src.jsonl` | a56f1f1783b93d872c13f48a835f5f479b5cc00256eca227bae43546b728aff3 |
| `r2_3/r2_full_cfg0_w0/conv/dev/30/r2_full_cfg0_w0/runs.jsonl` | 50980f2894f29910d8d8d617470e64db16991c69f817b08c252442e3506a0927 |
| `r2_3/r2_full_cfg0_w0/ecapa/dev_30.jsonl` | 45390907c8f1cdade1431ca90f405fe2dce0a7dcef5ae5173c7598a0ceee8985 |
| `r2_3/r2_full_cfg0_w0/ecapa/dev_30_src.jsonl` | 74625459e0a187daba4eaab9d0f1206be00024d124846d41390db04d59af73d1 |
| `r2_3/r2_full_cfg0_w0/metrics/dev_30.jsonl` | 0e81ef5aab3ee1ab76fefc98ee556cdc46129c765cb53791be0a70c8c43850e5 |
| `r2_3/r2_full_cfg0_w0/metrics/dev_30_src.jsonl` | 347c8d785816fa1382881aac0f787e767b402e5d3aeb7d9a54947a2b5bb5e96a |
| `r2_3/r2_full_cfg0_w1/conv/dev/30/r2_full_cfg0_w1/runs.jsonl` | 9e7b82e2166c37aee75fc546b7f606fb3b9f520d1a9b5a20eb14001ef4938331 |
| `r2_3/r2_full_cfg0_w1/ecapa/dev_30.jsonl` | 8c8beffb592ebcb1ace8d3be6bbf00bfdbab99b2f97c61ccdb8cdf4cf78f47a7 |
| `r2_3/r2_full_cfg0_w1/ecapa/dev_30_src.jsonl` | b4f4dfeadc2c0dd2ed712572a69d0ddbf429cade6955311c755cbd9b5c6d9835 |
| `r2_3/r2_full_cfg0_w1/metrics/dev_30.jsonl` | 9b890b161d7c7a3d03b09f3563be1789ace753c4105c68fbd5d91214297f827a |
| `r2_3/r2_full_cfg0_w1/metrics/dev_30_src.jsonl` | 72c767e7d3595a5a894ec824f3eec246299380a68900914743eddb30118bda75 |
| `r2_3/r2_full_cfg1_w1/conv/dev/30/r2_full_cfg1_w1/runs.jsonl` | 5b0158666927e57787dfacd264985739418ca5c289a2176b5bb15f7ea3088be4 |
| `r2_3/r2_full_cfg1_w1/ecapa/dev_30.jsonl` | 3991adb0e5dddbe14a644b4d0b3e9d1ade0573cc5d05a3f53163b0432e2a7383 |
| `r2_3/r2_full_cfg1_w1/ecapa/dev_30_src.jsonl` | 06e4644e725b69676a0e660dc5098acf8fcd9a4e4cf76575b4c6bc38e14789d2 |
| `r2_3/r2_full_cfg1_w1/metrics/dev_30.jsonl` | 842088a174977e290d54030c667d51c6c30baf6eb3d4f05b020adca0ec428cb2 |
| `r2_3/r2_full_cfg1_w1/metrics/dev_30_src.jsonl` | a93f39db455d42cbdca7dc8026eaaf06ae4b6db92257da283c3e08435046a5d2 |
| `r2_4/r2_4_cfg1_w1_b180/conv/dev/180/r2_4_cfg1_w1/runs.jsonl` | e74cb4ffb980de69e141108295c16b2315805e4704dba748fe4cd7e5e8cceaf2 |
| `r2_4/r2_4_cfg1_w1_b180/metrics/dev_180.jsonl` | 1ddc2790108a88817a84323fce5ca912cc8beb031fdc5c933301e77c2787da0f |
| `r2_4/r2_4_cfg1_w1_b180/metrics/dev_180_src.jsonl` | 50024635d5571f75bcda95765ef2f9f29ff0de6502c52fdb6bedd3c6e2cddf91 |
| `r2_4/r2_4_cfg1_w1_b3/conv/dev/3/r2_4_cfg1_w1/runs.jsonl` | cd84c323dcfd73dc12761edd180a917a76a8e322b48d6e082afbeb3185c994d5 |
| `r2_4/r2_4_cfg1_w1_b3/metrics/dev_3.jsonl` | d72d98302d80263167e791c376801bbe5b1f7c03a5bb18d9a928624065d2eb0d |
| `r2_4/r2_4_cfg1_w1_b3/metrics/dev_3_src.jsonl` | e5507e53db59d6a05e9d4624dea8873a663f5ff289f90c4e0d991c613923ef64 |

## Choices where the plan is silent

- Ties (§4.1): orderings compare WER at two decimals, SIM and UTMOS at three; thresholds use unrounded point estimates (tolerance 1e-9).
- R2.2 orders by R2.3's full order; without a D-mvc-2 pick at most three other eligible configurations.
- Exact NL ties (R2.2 under trigger i, fallback F) are broken by smaller w, smaller CFG, plain before interval.
- The D-mvc-2 pick includes w 0.
- In the B1 round, mvc_r1_nosng is B1's snapshot at CFG 0, SNG 0.
- Conditions 2-4 and the R2.3 WER check compare levels; the B1 keep rule is paired.
- Condition 5 as job e3's rule5.py: the WavLM mean over the cross-gender pairs where both WavLM @src rows are usable and the ECAPA mean over those where both ECAPA @src rows are usable, separately; a row is usable without error and with a finite value (e3 took every row carrying the value).
- The contamination test's result is recorded; a failure does not stop the record.
- The Dev-900 repeat compares the first of the order (or F); R2.4 is not repeated on Dev-900.

## Traces (every step, as printed)

### r1/r2_0

- §4.2: ema_0150000.pt against ema_0200000.pt at the reference setting (temp 0, relax 0.15 x 2, CFG 0, SNG 1.0) on Dev-S100, and dev-clean reconstruction
- ema_0150000.pt (r2_0/r2_0_k150_ref; sha256 85d48dc7ba46839da8f058e365cbd32e3cb88feda5104fcb1afc14a623dacc42): WER 2.63 %, SIM 0.904, UTMOS 3.843, failed 0; reconstruction WER 3.34 % (r2_0rec/r2_0_k150_rec, 0 errors)
- ema_0200000.pt (r2_0/r2_0_k200_ref; sha256 f82d70cc9a6a66e842afe52bcb76cf10840c2bb0df77d159a49b27d026f1920b): WER 2.34 %, SIM 0.903, UTMOS 3.874, failed 0; reconstruction WER 3.20 % (r2_0rec/r2_0_k200_rec, 0 errors)
- conditions for 150k: recon WER(200k) > recon WER(150k) + 1.0: False; SIM(200k) < SIM(150k) - 0.010: False; UTMOS(200k) < UTMOS(150k) - 0.05: False; WER(200k) > WER(150k) + 2.0: False
- snapshot: ema_0200000.pt (sha256 f82d70cc9a6a66e842afe52bcb76cf10840c2bb0df77d159a49b27d026f1920b)
- R1x (§4.10): SIM rose False, UTMOS rose True -> R1x runs

### r1/r2_1

- §4.3: Dev-S100 screen on /datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1/ema_0200000.pt: eight plain cells and three interval variants
- cfg0_w0 (r2_1/r2_1_c0_w0): WER 1.75, SIM 0.879, UTMOS 3.739; NL same +0.083 (SE 0.027), cross +0.112 (SE 0.045); summarize.py value +0.057 / +0.110; conditions 1:0 2:1 3:1 4:1 5:1; eligible False (w 0: condition 1 fails)
- cfg0_w0.5 (r2_1/r2_1_c0_w0.5): WER 1.46, SIM 0.902, UTMOS 3.825; NL same +0.038 (SE 0.028), cross +0.061 (SE 0.048); summarize.py value +0.011 / +0.065; cross ΔSIM_src WavLM -0.045, ECAPA -0.026; conditions 1:0 2:1 3:1 4:1 5:1; eligible False (fails 1)
- cfg0_w0.75 (r2_1/r2_1_c0_w0.75): WER 2.05, SIM 0.905, UTMOS 3.853; NL same +0.020 (SE 0.028), cross +0.040 (SE 0.049); summarize.py value -0.008 / +0.048; cross ΔSIM_src WavLM -0.062, ECAPA -0.033; conditions 1:0 2:1 3:1 4:1 5:1; eligible False (fails 1)
- cfg0_w1 (r2_1/r2_1_c0_w1): WER 2.34, SIM 0.903, UTMOS 3.874; NL same +0.005 (SE 0.027), cross +0.022 (SE 0.050); summarize.py value -0.024 / +0.034; cross ΔSIM_src WavLM -0.076, ECAPA -0.039; conditions 1:1 2:1 3:1 4:1 5:1; eligible True (conditions 1-5 met)
- cfg1_w0 (r2_1/r2_1_c1_w0): WER 1.46, SIM 0.904, UTMOS 3.810; NL same +0.054 (SE 0.025), cross +0.075 (SE 0.045); summarize.py value +0.032 / +0.079; conditions 1:0 2:1 3:1 4:1 5:1; eligible False (w 0: condition 1 fails)
- cfg1_w0.5 (r2_1/r2_1_c1_w0.5): WER 2.05, SIM 0.910, UTMOS 3.872; NL same +0.026 (SE 0.022), cross +0.043 (SE 0.045); summarize.py value +0.001 / +0.050; cross ΔSIM_src WavLM -0.029, ECAPA -0.018; conditions 1:0 2:1 3:1 4:1 5:1; eligible False (fails 1)
- cfg1_w0.75 (r2_1/r2_1_c1_w0.75): WER 1.75, SIM 0.911, UTMOS 3.892; NL same +0.011 (SE 0.023), cross +0.028 (SE 0.045); summarize.py value -0.017 / +0.037; cross ΔSIM_src WavLM -0.042, ECAPA -0.021; conditions 1:1 2:1 3:1 4:1 5:0; eligible False (fails 5)
- cfg1_w1 (r2_1/r2_1_c1_w1): WER 1.46, SIM 0.909, UTMOS 3.887; NL same -0.003 (SE 0.024), cross +0.019 (SE 0.045); summarize.py value -0.029 / +0.029; cross ΔSIM_src WavLM -0.050, ECAPA -0.027; conditions 1:1 2:1 3:1 4:1 5:1; eligible True (conditions 1-5 met)
- cfg0_w0.5_u0.5 (r2_1/r2_1_c0_w0.5_u0.5): WER 1.17, SIM 0.900, UTMOS 3.825; NL same +0.042 (SE 0.028), cross +0.064 (SE 0.048); summarize.py value +0.015 / +0.068; cross ΔSIM_src WavLM -0.042, ECAPA -0.024; conditions 1:0 2:1 3:1 4:1 5:1; eligible False (not eligible: the plain cell does not fail condition 3 alone; fails 1)
- cfg0_w0.75_u0.5 (r2_1/r2_1_c0_w0.75_u0.5): WER 2.05, SIM 0.903, UTMOS 3.857; NL same +0.026 (SE 0.028), cross +0.043 (SE 0.048); summarize.py value -0.003 / +0.050; cross ΔSIM_src WavLM -0.060, ECAPA -0.030; conditions 1:0 2:1 3:1 4:1 5:0; eligible False (not eligible: the plain cell does not fail condition 3 alone; fails 1, 5)
- cfg0_w1_u0.5 (r2_1/r2_1_c0_w1_u0.5): WER 2.05, SIM 0.901, UTMOS 3.887; NL same +0.012 (SE 0.027), cross +0.028 (SE 0.049); summarize.py value -0.018 / +0.038; cross ΔSIM_src WavLM -0.072, ECAPA -0.036; conditions 1:1 2:1 3:1 4:1 5:0; eligible False (not eligible: the plain cell does not fail condition 3 alone; fails 5)
- eligible: ['cfg0_w1', 'cfg1_w1']; D-mvc-2 pick: cfg0_w1; B1 trigger i: False

### r1/r2_2

- §4.4: the D-mvc-2 pick cfg0_w1, then the eligible configurations within SIM >= best - 0.010 and UTMOS >= best - 0.05 by R2.3's order
- bests over the eligible configurations: SIM 0.909, UTMOS 3.887; within the bounds: ['cfg1_w1']
- finalist cfg0_w1: WER 2.34, SIM 0.903, UTMOS 3.874, max_g NL +0.022
- finalist cfg1_w1: WER 1.46, SIM 0.909, UTMOS 3.887, max_g NL +0.019
- full-Dev runs to make (1,000 pairs, 30 s, ECAPA), finalists and mvc_r1_nosng: r2_full_cfg0_w1={"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "ema_0200000.pt", "steps": 16, "cfg": 0.0, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "latent", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler", "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std", "neg_src": 1.0, "neg_src_until": 1.0}; r2_full_cfg1_w1={"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "ema_0200000.pt", "steps": 16, "cfg": 1.0, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "latent", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler", "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std", "neg_src": 1.0, "neg_src_until": 1.0}; r2_full_cfg0_w0={"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "ema_0200000.pt", "steps": 16, "cfg": 0.0, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "latent", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler", "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std", "neg_src": 0.0, "neg_src_until": 1.0}

### r1/r2_3

- §4.5 on full Dev (1,000 pairs); mvc_r1_nosng (r2_3/r2_full_cfg0_w0) WER 1.70
- cfg0_w1: WER 1.90 (+0.20 vs nosng, check True), SIM 0.904, UTMOS 3.858; NL same -0.002 (SE 0.013), cross +0.027 (SE 0.023); L-pass False, SIM ok False, UTMOS ok False, E3 False
- cfg1_w1: WER 1.84 (+0.14 vs nosng, check True), SIM 0.912, UTMOS 3.865; NL same -0.007 (SE 0.012), cross +0.020 (SE 0.022); L-pass False, SIM ok False, UTMOS ok False, E3 False
- best SIM –, best UTMOS – (over the L-passing finalists); E3 order: empty; fallback F: cfg1_w1
- §4.5 on Dev-900 (outside Dev-S100); mvc_r1_nosng (r2_3/r2_full_cfg0_w0) WER 1.69
- cfg0_w1: WER 1.85 (+0.16 vs nosng, check True), SIM 0.904, UTMOS 3.856; NL same -0.003 (SE 0.013), cross +0.028 (SE 0.021); L-pass False, SIM ok False, UTMOS ok False, E3 False
- cfg1_w1: WER 1.89 (+0.19 vs nosng, check True), SIM 0.913, UTMOS 3.863; NL same -0.008 (SE 0.013), cross +0.020 (SE 0.021); L-pass False, SIM ok False, UTMOS ok False, E3 False
- best SIM –, best UTMOS – (over the L-passing finalists); E3 order: empty; fallback F: cfg1_w1
- Dev-900 check: first on 1,000 pairs cfg1_w1, on Dev-900 cfg1_w1
- R2.4 runs to make (full Dev at 3 s and 180 s): r2_4_cfg1_w1={"run_dir": "/datastore/ceai/crt/mimi_vc/data_storage/mvc/runs/r1", "ckpt": "ema_0200000.pt", "steps": 16, "cfg": 1.0, "sway": -1.0, "prompt_max": null, "prompt_chunk": 750, "perturb_src": false, "decode": "latent", "seed_from_pair": true, "mode": "convert", "whisper_src": "clean", "solver": "euler", "temp": 0.0, "cfg_rescale": 0.0, "sharpen": 0.0, "gv": 0.0, "stats_path": null, "decoder_ckpt": null, "init_t": 0.0, "match_rms": false, "relax": 0.15, "relax_k": 2, "cfg_branch": "std", "neg_src": 1.0, "neg_src_until": 1.0} (BUDGETS=3,180)

### r1/r2_4

- cfg1_w1: SIM 3 s 0.879 >= 0.912 - 0.06: True; SIM 180 s 0.918 >= 0.912 - 0.005: True -> passes
- E3 is empty: fallback F cfg1_w1; R2.4 for the record: passes
- B1 trigger i (no eligible configuration in R2.1): False; trigger ii (NL_g > +0.03 on full Dev: same -0.007, cross +0.020): False -> B1 not triggered
- R5 trigger (reported only; R5 is not run): full-Dev UTMOS 3.865 < 3.52: False

