# CRT journal extension — analysis plan v0 (lean)

Frozen on 28 Sep 2026, **before any new evaluation number exists**. The only numbers seen so far are the published SPIE results and the 2-pair smoke test of `rescore_conference.py`, which was a harness check.

Changes after this date go into v1. v1 is dated and frozen at the end of dev selection, before any test-set conversion is scored. Anything changed after v1 is reported as a deviation.

Source of the design: `paper/journal_extension/extension_plan.md` §4, trimmed for the lean plan. The lean plan drops the Seed-TTS-eval set, the CRT-Viterbi and conformal candidates, and the mixed-model requirement.

## 1. Sets

All pair lists are built by `benchmark/make_pairs.py`: fixed seed 20260928, mic1 only, genders from `speaker-info.txt`. Each list stores pair_id, texts, paths and durations.

| Set | Construction | Use |
|---|---|---|
| **Dev** | The 20 SPIE conference speakers (10 F / 10 M), 10 source utterances each, 5 targets per source, gender-balanced. | τ and every other hyper-parameter; nothing else |
| **VCTK-2k** (primary test) | 20 source + 20 target speakers (10 F / 10 M each), disjoint from each other and from Dev. p315 (no text) and s5 are excluded. 20 source utterances per source speaker, 5 targets each = 2,000 pairs. Up to 10 utterances per speaker come from the shared-text block (for MCD and F0 against the target's parallel recording); the rest are speaker-specific (for WER). | Main tables |
| **LTC-2k** (second test) | LibriTTS-R test-clean. About 5 utterances of 3–10 s per speaker, each sent to 10 gender-balanced targets, about 2,000 pairs. No parallel target, so no MCD. | Main tables |
| Conference-100 | The 100 SPIE pairs (source `_001` for all, "Please call Stella."). | Reproduction and transparency only (E0b), never a headline |

**Reference rule.** Each target speaker's reference pool R, evaluation-parallel recordings P and SIM-reference set E are mutually disjoint. A target recording of the same sentence as the source is never in R.

**Reference budget.** Every system gets the same R truncated to 30 s in the main tables. The sweep over {3 s, 30 s, ~180 s} (E16) covers FAISS, CRT and kNN-VC.

## 2. Systems

| System | Role |
|---|---|
| S1 FAISS | Stage 1 only |
| S2 CRT(τ_dev) | Primary |
| S3 CRT τ = 0.4 | The SPIE setting; secondary |
| S4 CRT argmax | Descriptive |
| S5 kNN-VC | `prematched=True`, k = 4 |
| S6 FreeVC | Flagged "VCTK-seen" |
| S7 Seed-VC | |
| S8 FACodec or Vevo | Chosen at setup by licence and cost; recorded in v1 |

**Controls.** Natural target speech; natural source (the WER floor); Mimi copy-synthesis at Q = 32 (the codec ceiling).

## 3. Metrics (`benchmark/metrics.py`; every signal resampled to 16 kHz)

| ID | Metric | Definition |
|---|---|---|
| M1 | dB-MCD | pyworld → pysptk mel-cepstrum, order 24, α = mcepalpha(16000), c0 excluded, DTW, constant 10√2/ln10. Against the target's parallel recording, VCTK shared-text subset only. |
| M2 | F0 | Speaker-normalised log-F0 RMSE and V/UV error against the parallel recording after DTW |
| M3 | WER / CER | Whisper-large-v3 (primary) and Whisper-base (secondary). Ground-truth text, Whisper EnglishTextNormalizer, corpus level. **VCTK primary WER uses speaker-specific sentences only.** |
| M4 | SIM | WavLM-base-plus-SV cosine to the mean embedding of E (primary); Resemblyzer SECS secondary, disclosed as the encoder CRT was trained with |
| M5 | EER | On M4 scores |
| M6 | Quality | UTMOS22-strong |
| M7 | Coverage | Fraction of CB1–18 tokens overridden, per codebook |

## 4. τ rule (dev only)

1. Take the global τ ∈ {0.1, 0.2, …, 0.9} with the lowest Dev corpus WER (Whisper-large-v3), subject to Dev SIM ≥ FAISS SIM.
2. Break ties by the lower dB-MCD.
3. If no τ meets the SIM constraint, take the τ with the smallest SIM deficit and report that.

**Optional frame-level re-ranking candidate (E17).** It enters the paper as a secondary system only if, on Dev, it beats CRT(τ_dev) on WER with SIM ≥ FAISS.

## 5. Endpoints and statistics

**Primary family.** Holm correction over 7 hypotheses, or 8 with P8. Each is Δ = CRT(τ_dev) − FAISS:
- P1 WER on VCTK-2k (speaker-specific sentences);
- P2 WER on LTC-2k;
- P3 SIM on VCTK-2k;
- P4 SIM on LTC-2k;
- P5 UTMOS on VCTK-2k;
- P6 UTMOS on LTC-2k;
- P7 dB-MCD on the VCTK parallel subset;
- P8 CMOS, CRT(τ_dev) vs FAISS. P8 is included only if at least 15 screened raters complete it; otherwise the family is P1–P7.

**Secondary family.** CRT(τ_dev) vs each baseline on WER, SIM and UTMOS, with Holm correction within the family.

**Descriptive.** τ = 0.4, argmax, controls, gate probes, lineage ablations: confidence intervals only.

**Inference.**
- **Continuous metrics:** paired per-conversion differences with a crossed bootstrap (source and target speakers resampled independently, 10,000 replicates). Report the 95% CI and a two-sided bootstrap p-value.
- **Corpus WER:** the same crossed bootstrap on the difference in corpus WER.
- **Secondary statistics:** Wilcoxon signed-rank on pair means and the matched-pairs rank-biserial effect size, labelled as possibly anti-conservative.
- **Listening test:** bootstrap over raters × items.
- **Size check:** after dev, estimate the variance and the minimum detectable effect. If it is too large, raise targets per source from 5 to 10 **before** scoring test.

## 6. Outcome framing, fixed in advance

| Outcome | Condition | Framing |
|---|---|---|
| A | At least one primary endpoint significantly favours CRT and none favours FAISS | Method paper |
| B | Significant effects in both directions | Trade-off / operating-point paper built on the risk–coverage curves |
| C | No significant effect, or only effects favouring FAISS | Analysis paper: "when does post-retrieval refinement help?" |

## 7. Pending for v1 (decisions made while building the harness, 28 Sep 2026)

These choices were made before any Dev or test conversion was scored. They become binding when v1 is frozen.

- **τ-rule WER.** Computed on Dev pairs flagged `wer_primary` only (speaker-specific sentences; 500 Dev pairs).
- **LTC-2k size.** 1,800 pairs, not about 2,000.
- **References.**
  - Each budget is cut exactly to 3, 30 or 180 s, and the budgets are nested.
  - 5 LTC targets have less than 180 s available (minimum 57.6 s).
  - The SIM reference set E is 10 utterances per target.
- **dB-MCD silence rule.** Frames at or below −20 dB of the utterance's mean WORLD power are removed before DTW (the sprocket / s3prl-vc rule). This removes a median 48% of VCTK frames. MCD is stable from −30 to −15 dB.
- **F0.** Harvest extractor. log-F0 is normalised by each utterance's own voiced mean.
- **EER trial list.** Each output is scored against every target centroid in the set: its own target is the genuine trial, the others are impostor trials.
- **Whisper decoding.** Greedy, language en, no timestamps, batched. Whisper-large-v3 WER never falls back to base.
- **WavLM-SV weights.** A safetensors copy of microsoft/wavlm-base-plus-sv with identical weights.
- **τ grid.** Must include low values (0.1–0.3). The published model's confidence is low: at τ = 0.4 the gate opens on 0–6.8% of tokens on conference pairs and 9–23% on two longer Dev pairs.
- **Speaker embedding for retrieval and CRT.** Resemblyzer on the first 15 s of the reference, as published. With a 180 s budget, retrieval uses the whole budget but the embedding still uses 15 s.
- **Temperature-scaled gates** (if used) need full logits, so the calibration sets run with `--dump_logits`.

## 8. Rules against rescuing results

- No metric, τ, subset or reference budget is chosen after test scores are seen.
- Test sets are scored once per frozen system.
- Null and negative rows stay in the tables.
- Numbers from the SPIE paper are never reused unless re-measured by this harness. Conference-100 results appear only in the corrections appendix.
