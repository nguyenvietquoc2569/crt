# MimiFlow-VC: final research plan v1 (paper 3)

Date: 25 Oct 2026. Code base: `/Users/harry/duplex/mvc`. Cluster: ceai, `/datastore/ceai/crt/mimi_vc/mvc`, data under `$DATA_STORAGE/mvc`.

Execution note. No ssh was run in this planning step, so nothing below has been run yet. Every probe and run below is a ceai job submitted through `submit_ceai.sh`: normal partition, `gpu:2g.35gb:1`, at most 72 h, a 1 h gap between submissions, no submissions in the 03:00-04:10 blackout. The user's mandate of 25 Oct ("use ceai for all needed probes", recorded in memory `project_mimi_vc.md`) means these sbatch calls need no per-job approval. Each submission is still announced to the interpreter session first. As of the memory note (25 Oct, about 02:20, not re-checked here), three things were in flight: the patched encode job (train-clean-100 + dev-clean, K=2 perturbed copies, CAM++, probe P0); the A/B pilot chained behind it; and the big encode chained behind the downloads.

---

## 1. Decision and rationale

### 1.1 Decision

We build MimiFlow-VC. A DiT of about 121M parameters, trained by conditional flow matching, generates Mimi's continuous 512-d pre-RVQ latent at 12.5 Hz. It is conditioned on two things:
- the CB0 tokens of an information-perturbed copy of the source (content);
- an in-context prompt of the target reference's own Mimi latents, plus a CAM++ global speaker embedding (voice).

Frozen Mimi decodes the result, either on its unquantised path or after re-quantisation to 32 genuine Mimi codes.

This plan merges the two top designs. Their mean judge scores were within 0.08 of each other (Design 2 7.45, Design 1 7.37, Design 3 6.37), and two of the three judges named Design 1 the winner.
- Where the two differ on risk, we take Design 2's conservative choices: generate the standardised latent rather than the decoder-space input; hard perturbed CB0 by default; the existing `[prompt | target]` layout; Q32 as a decode route; a chunk ensemble for 180 s references; a parallel-text leakage probe.
- From Design 1 we take the CAM++ prior (on from day one), the soft-CB0 knob, latent-space self-transformation, and the constant-LR schedule that keeps ablations comparable.
- From Design 3 we take neighbour-token corruption, the decoder-input-metric loss (as an ablation), the contamination test, early stopping on reconstruction WER, a Mimi-native AAM speaker vector (as an ablation), and a causal Mimi-CTC content adapter as the first content fallback.

### 1.2 Why this design (evidence)

1. Token-domain VC on Mimi cannot work, and the reason is structural.
   - Mimi's semantic VQ (CB0) runs in parallel with the acoustic RVQ, and the acoustic levels are not a residual of CB0 ([Moshi/Mimi, arXiv 2410.00037](https://arxiv.org/abs/2410.00037)). CB1-31 therefore carry content.
   - Swapping them fails every time. Mimi token-swap WER is 102-120% in Kanade ([2602.00594](https://arxiv.org/abs/2602.00594)) and 110% in FocalCodec ([2502.04465](https://arxiv.org/abs/2502.04465)). On our own protocol, FAISS gives 126% and CRT 89%.
   - Decoding CB0 alone gives WER 45.5% and SIM 0.19 ([LM-SPT, 2506.16738](https://arxiv.org/abs/2506.16738)).
   - So the converter must generate the acoustic part of the decoder input, and all of it. Speaker similarity of Mimi copies is 0.672 with 8 codebooks, 0.829 with 16 and 0.902 with 32 ([Voxtral, 2603.25551](https://arxiv.org/abs/2603.25551)).

2. A continuous latent is in distribution for the decoder. Mimi was trained with quantisation skipped on 50% of sequences (2410.00037), and our journal work already decoded continuous latents. Generation in codec latent space works elsewhere:
   - X-VC, a one-step converter in a codec latent space, beats its Seed-VC-small teacher on SIM, 0.63 vs 0.56 ([2604.12456](https://arxiv.org/abs/2604.12456));
   - Kyutai's own CALM generates continuous Mimi-derived latents ([2509.06926](https://arxiv.org/abs/2509.06926));
   - CALM's 32-d VAE caps SIM at 0.52, which argues against compressing the space. We keep all 512 dimensions.

3. The generator family is proven at our data scale.
   - In-context conditional flow matching reaches SECS 0.930 and WER 3.51 at 2 NFE, scored with our own SV model, WavLM-base-plus-sv ([R-VC, 2506.01014](https://arxiv.org/abs/2506.01014)).
   - Trained on LibriTTS only, a 123M flow-matching model reaches WER 1.69 and UTMOS 4.16 ([ZipVoice, 2506.13053](https://arxiv.org/abs/2506.13053)).
   - 16 NFE costs almost nothing against 32: WER 2.53 vs 2.42 ([F5-TTS, 2410.06885](https://arxiv.org/abs/2410.06885)).
   - Other LibriTTS-only VC systems reach VCTK WER 2.97 ([VoicePrompter, 2501.17612](https://arxiv.org/abs/2501.17612)) and 2.99 with UTMOS 4.19 ([HierSpeech++, 2311.12454](https://arxiv.org/abs/2311.12454)).

4. Disentanglement comes from perturbed, discrete content.
   - Continuous content copies the source voice: Vevo's SIM to the source is 0.505 with continuous HuBERT features against 0.236 with K=4096 tokens ([2502.07243](https://arxiv.org/abs/2502.07243)).
   - Strong systems train on content from perturbed audio: [Seed-VC 2411.09943](https://arxiv.org/abs/2411.09943), [NANSY 2110.14513](https://arxiv.org/abs/2110.14513), [FreeVC 2210.15418](https://arxiv.org/abs/2210.15418), R-VC.
   - Signal perturbation alone leaves residual source similarity of 0.704; a VC-model shifter brings it to 0.411 ([Zero-VC, 2606.20218](https://arxiv.org/abs/2606.20218)). That is why our leakage levers are ordered as they are.
   - A sized bottleneck beats adversarial training ([AutoVC 1905.05879](https://arxiv.org/abs/1905.05879); [FACodec 2403.03100](https://arxiv.org/abs/2403.03100): SIM 0.86 drops to 0.73 without its bottleneck).
   - Soft units recover the content that hard units lose (Soft-VC WER 2.6 vs 5.4, [2111.02392](https://arxiv.org/abs/2111.02392)). This is the basis of our soft-CB0 knob and its link to the journal's posterior-mean result.

5. Voice comes from an in-context prompt plus a global embedding.
   - X-VC loses 0.11 SIM without its utterance-level speaker condition.
   - Seed-VC's full-reference prompting raises SECS from 0.795 to 0.868.
   - CAM++ is not one of our evaluators. We use no SV loss, because SV losses inflate scores: [CTEFM-VC 2411.02026](https://arxiv.org/abs/2411.02026) gains 0.15 SECS from one.

6. The work is novel.
   - No published zero-shot VC operates on Mimi latents. [StarVC 2506.02414](https://arxiv.org/abs/2506.02414) generates 8 Mimi codebooks autoregressively and reaches SECS-WavLM 0.472. Kyutai withheld its voice encoder ([DSM-TTS 2509.08753](https://arxiv.org/abs/2509.08753)).
   - No CB0-only speaker probe exists. [2608.15799](https://arxiv.org/abs/2608.15799) probes phonetics only (PNMI 0.66, dialect 32.6%); [SpInv 2607.16870](https://arxiv.org/abs/2607.16870) attacks the full stream only.

### 1.3 What we drop, and why

| Dropped | Reason |
|---|---|
| Decoder-space target x = M(lat) with one global scale (D1) | M is a sum of two rank-256 projections, so the token route would need an ill-conditioned inverse, and low-variance dimensions get low SNR. We generate the per-dimension standardised latent; the decoder-input metric survives as loss ablation K6. |
| Segment-reset RoPE without offsets (D1) | Target frame t aligns with frame t of every prompt segment, which invites copying, and 180 s means 18 segments against at most 6 in training. Replaced by random per-segment offsets (§2.4) and the chunk ensemble. |
| Split-posterior decoding (D3) | Its premise, that the semantic branch carries the content, is contradicted by the split RVQ and by our own C15 (the source CB0 with foreign acoustic codes gives 148% WER). |
| Streaming modes co-trained into the main run (D3) | 60% causal training dilutes the offline model that carries the paper's main claim. Streaming becomes an optional post-hoc fine-tune. |
| "Bit-exact" streaming claim | GPU kernels differ by shape, as job 5625 showed for Mimi itself. We claim equivalence within a tolerance only. |
| Any SV loss; WavLM-SV, ECAPA or Resemblyzer anywhere in training | Metric gaming. |

The literature URLs above come from the three sweeps, whose agents marked them verified. They were not re-opened in this step.

---

## 2. Final architecture v1

### 2.1 Inputs, outputs, target space

- Mimi `kyutai/mimi`, commit 89091b3e466eb6a9d11e537bf26b144f194978f7. Fully frozen. Encoding runs in fp32 through `encode_mimi.MimiEncoder`, both for training data and in the adapter.
- Target: `lat` [T,512], Mimi's encoder output after the transformer and downsampling, before the RVQ. It is standardised per dimension with corpus statistics (mean and std over 2M frames, `train.corpus_stats`). The model generates z = (lat − μ)/σ.
- The output length T equals the source length (ceil(N/1920) frames). Source timing is kept frame for frame.
- Decode routes:
  - continuous: `mimi_io.decode_latent`, i.e. `sem.out(sem.in(x)) + ac.out(ac.in(x))` → upsample → decoder transformer → SEANet, in fp32;
  - Q32: `mimi_io.quantize` → 32 codes → `MimiModel.decode` in fp16, the same route as the published Mimi copy controls. It emits genuine Moshi-compatible tokens.

### 2.2 Content stream

- Training content for target frames: hard CB0 of one of K=2 NANSY-perturbed copies, chosen per example.
  - Perturbation follows `perturb.PRESET`: formant U(1,1.4)^±1, median F0 U(1,2)^±1, F0 range U(1,1.5)^±1 via Praat "Change gender", then a random PEQ of ±12 dB with 8 peaks.
  - Near-identity draws are skipped. A draw counts as near-identity if |ln formant| < ln 1.1 and |ln pitch| < ln 1.15. Draws are seeded by (seed, utt id, k), so their parameters can be recomputed by replaying the RNG without re-encoding. About 2.5% of draws are affected; the other draw is used instead.
- Regularisers: 10% neighbour-token corruption, where a token is replaced by one of its 8 nearest CB0 codewords in E0 (a static 2048×8 table), and 5% content dropout to a null index.
- Prompt-frame content: clean CB0 of the prompt, in training and at inference, which removes the mismatch Design 1 had.
- Embedding: `Emb(2050×256)`, initialised from Mimi's CB0 codewords E0 (`--content_init mimi`).
- Soft variant (knob K1): s_t = Σ_{k≤8} softmax(−d²_tk/(T·σ0²))·Emb[idx_tk], with T from P2. The top-8 indices and squared distances of the perturbed copies are already stored by `encode_perturbed.py` (`.psoft<k>.idx/.d2`). For clean prompt and source frames they are computed on the GPU from `lat` via `sem.input_proj` (about 3e10 FLOP per batch, negligible).
- Inference content: clean source CB0 by default. Dev knob: K=2 perturbed source draws with the velocities averaged at every ODE step.

### 2.3 Speaker path

- In-context prompt: clean standardised reference latents in the `cond` channel on prompt frames. This is the existing `model.MimiFlow` layout.
- Global vector: CAM++ 192-d (`spk_embed.CamPlus`, Seed-VC's `campplus_cn_common.bin`). It goes through MLP 192→768→768 and is added to the τ embedding that drives adaLN.
  - Training: the mean of the L2-normalised per-utterance embeddings (`.spk.npy`) of the prompt utterances, renormalised.
  - Inference: the mean of the L2-normalised embeddings of the reference's 10 s windows (one window at 3 s).
- Ablation alternative (K3): a Mimi-native pooled vector. Two transformer layers (d=512) over the prompt latents, attentive-statistics pooling to 256-d, and an AAM-softmax auxiliary loss (λ=0.1, margin 0.2, scale 30) over the training speakers.

### 2.4 Backbone

| Part | Size | Params |
|---|---|---|
| Input projection | Linear(512 x_τ + 512 cond + 256 content → 768) + region embedding (prompt/target) + segment embedding | 0.99M |
| DiT blocks | 16 × [pre-LN, MHSA 12 heads (head dim 64), MLP 3072, GELU-tanh] | 113.2M |
| adaLN-single | one Linear(768 → 6·768) from c = τ-MLP + speaker MLP, plus learned per-block offsets (16×6×768) | 3.6M |
| τ-MLP / speaker MLP | 256→768→768 / 192→768→768 | 0.79M / 0.74M |
| Output | adaLN(2·768), LN, Linear(768→512), zero-initialised | 1.57M |
| Content embedding | 2050×256 | 0.52M |
| Total | | ≈121M |

- Attention is bidirectional with a key-padding mask (SDPA mem-efficient kernel), under bf16 autocast.
- Positions:
  - the prompt is cut into segments of at most 125 frames (10 s), the same in training and at inference;
  - each segment and the target get their own RoPE index 0..n−1 plus an offset;
  - in training the offset is drawn independently per segment from U{0..4095}, so the relative phase between target and prompt carries no information and cross-region matching must use content;
  - at inference the offsets are fixed: target 0, prompt segment k at 512·k.
- Pilot model (P5, P6): d=512, 8 layers, 8 heads, per-block adaLN, about 40M (the scaffold labels it "30M"; the count is about 40M).

### 2.5 Training examples

- Target: a random crop of at most 125 frames (10 s) of utterance u of speaker s.
- Prompt: a concatenation of 1-n other utterances of s, never u. If s has at least 2 chapters, the prompt utterances come from a different chapter with p=0.5.
- Prompt length mixture: 20% 3-5 s (38-62 frames), 60% 5-30 s (63-375), 20% 30-60 s (376-750).
  - This replaces the scaffold's single-utterance prompt. LibriTTS-R utterances average about 5.6 s, so `data.PairSampler` currently gives mostly 3-10 s prompts.
- Batching: token-budgeted at 32k tokens (prompt + target), length-bucketed. That is about 78 examples and about 8.6k loss frames per step.

### 2.6 Losses and optimisation

- Rectified flow: x_τ = (1−τ)ε + τz₁, target velocity v = z₁ − ε, τ ~ U(0,1) (F5 recipe), MSE on target frames only (`model.flow_loss`).
- CFG dropout:
  - prompt and CAM++ dropped together, p=0.15;
  - CAM++ alone, p=0.10 (so the prompt-only path is trained and the CAM++ effect can be measured by switching it off at inference);
  - content, p=0.05.
- AdamW, betas (0.9, 0.99), weight decay 0.01, grad clip 1.0, bf16, EMA 0.9999.
- Learning rate: 1.5e-4 after a 5k-step warm-up, constant until 80% of the run, then cosine to 1.5e-5. Ablations stop at 100k on the same constant phase, so they compare directly with the main run's 100k EMA checkpoint.
- Main run: 200k steps, about 80 epochs of target frames. Snapshots every 20k; resume checkpoint every 2k (`last.pt`). Extension to 300k only if Dev-S100 still improves between 150k and 200k.
- Early stop: same-speaker reconstruction WER on 300 dev-clean utterances, evaluated per snapshot. Never select on training loss (training tip C5).

### 2.7 Sampling

- Euler, 16 NFE, sway s=−1. CFG: v = v_c + w(v_c − v_u), w=0.7, where the unconditional pass drops prompt and CAM++. That is 32 forward passes per conversion.
- Noise seed: from the harness's per-pair seed (`seed_all(sha256(pair_id))`).
- Budgets:
  - 3 s = 37 frames and 30 s = 375 frames, both inside the training range;
  - 180 s = 2,250 frames. Default is a chunk ensemble: three 60 s chunks, velocities averaged per step. Dev alternative: the full prompt with logit scaling log(P)/log(375) on prompt keys.
- Cost: about 3.9 TFLOP per 5 s source with a 30 s reference, about 0.05 s on a 2g slice (3× for 180 s). Including the Mimi codec, RTF is about 0.02-0.05.

### 2.8 Mimi changes

None in v1. Two conditional changes are kept outside the core:
- R5, a prematched decoder fine-tune of decoder_transformer + SEANet decoder with the [T-Mimi](https://arxiv.org/abs/2601.20094) losses (mel L1 ×2, MS-STFT LSGAN + feature matching ×4, never L1 alone). The stock decoder is kept for token output.
- E2, a last-resort invariance fine-tune of a copy of Mimi's last 2 encoder layers plus `sem.input_proj`, used only as the content extractor.

### 2.9 Ablation knobs

| Knob | Values (default first) | Where decided |
|---|---|---|
| K1 content | perturbed hard CB0 / perturbed soft CB0 (T from P2) / clean CB0 (A1) / + second stream (F1 Mimi-CTC, F2 WavLM-L units) | P2, P6, A1 |
| K2 perturbation | PRESET + near-identity skip / milder (fs 1.25, ps 1.5) / + kNN-VC shifter (E1) | P0, P5 |
| K3 global speaker | CAM++ / Mimi-AAM vector / none | P4, P6, A2 |
| K4 positions | random segment offsets / scaffold `restart` (A3) / cross-attention memory (fallback) | P6, A3 |
| K5 prompt mixture | 3-60 s multi-utterance / single utterance 3-20 s (A3) | A3 |
| K6 loss metric | standardised-lat MSE / decoder-input-weighted MSE ‖P·diag(σ)(v̂−v)‖² | P1, P6 (optional arm) |
| K7 inference | NFE {1,2,4,8,16,32}; w {0, 0.5, 0.7, 1, 2}; decode {continuous, Q32}; source perturbation {off, K=2 avg}; 180 s {ensemble, full+scaling} | R2 (Dev only) |
| K8 post-training | self-transformation (R4), decoder prematch (R5), streaming fine-tune (R6), few-step shortcut distillation (R6b) | gated |

### 2.10 Code changes against the scaffold

Each change gets pytest tests in `mvc/tests`, run on the Mac (`uvx --with torch ... pytest -q tests`) before submission.

1. `data.PairSampler`: multi-utterance prompts with the length mixture and the cross-chapter rule; segment ids; near-identity skip; clean prompt content; length-bucketed token-budget `batch()`.
2. `model.py`:
   - `positions(..., mode='segment_random')`;
   - adaLN-single;
   - an independent CAM++ dropout in `flow_loss`;
   - soft-CB0 weighted embedding;
   - neighbour corruption.
3. `train.py`: constant+tail LR schedule, EMA 0.9999, token budget, snapshot evaluation hook.
4. `vc_adapter.py`:
   - remove the reference truncation. `ref_features` currently keeps `lat[:prompt_max]` with `prompt_max=250` (20 s), so the 30 s and 180 s budgets are cut;
   - 10 s segmentation;
   - the 180 s ensemble;
   - windowed CAM++;
   - `perturb_src` with K-draw velocity averaging;
   - soft-CB0 at inference.
5. New probe scripts: `probe_decoder.py` (P1), `probe_content_ctc.py` (P2), `probe_latent_floor.py` (P3), `probe_parallel_text.py` (P0b). Plus `mvc/pairs/dev_s100.txt`, a contamination test, and `analyze_mvc.py` (Mac).

---

## 3. Data pipeline

### 3.1 Corpus and exclusions

| Use | Subset | Hours / speakers |
|---|---|---|
| Train | LibriTTS-R train-clean-100, train-clean-360, train-other-500 (minus 20 speakers kept for in-distribution validation loss) | ≈555 h, ≈2,311 speakers, ≈355k utterances |
| Probes and monitoring (never trained on) | dev-clean (40 speakers); dev-other (33 speakers, if downloaded) | 9 h + 9 h |
| Never read | test-clean (= LTC-2k), test-other; all of VCTK for training, statistics or k-means | |
| Selection only | VCTK Dev (20 SPIE speakers): Dev-S100, full Dev, parallel-text probe P0b | |
| Untouched until T1 | VCTK-2k and LTC-2k | |

- `processed_merged_v3` is never used, because it may contain VCTK.
- Fallback: if train-other-500 is not encoded when R1 is ready to start, R1 runs on train-clean-100+360 (245 h, 1,151 speakers) and the paper reports that.
- Contamination test (pytest on the Mac). It fails if any training speaker or utterance appears in dev-clean, dev-other, test-clean or test-other, or in the speaker sets of `results/v1/pairs/{dev,vctk2k,ltc2k}.jsonl`. The training ID manifest is saved with its sha256.

### 3.2 Precompute (all on ceai)

| Item | Script | Per frame | 555 h (≈25M frames) |
|---|---|---|---|
| Clean latent [T,512] fp16 (from an fp32 encode), last 2 frames trimmed (TRIM=2), batch-vs-single check with fallback | `encode_mimi.py` | 1,024 B | 25.6 GB |
| Clean codes [T,32] int16 | same | 64 B | 1.6 GB |
| Perturbed codes, K=2 draws (`.pcodes<k>`) | `encode_perturbed.py` | 2×64 B | 3.2 GB |
| Perturbed soft profile, top-8 idx int16 + d² fp16, K=2 (`.psoft<k>`) | same | 2×32 B | 1.6 GB |
| CAM++ per utterance (`.spk.npy`) | `spk_embed.py` | – | 0.27 GB |
| Latent mean/std; CB0 neighbour table (2048×8) | `train.corpus_stats`; one-off | – | <1 MB |
| Optional F2: WavLM-L L* k-means-500 top-8, 4:1 pooled | gated by P2/P6 | 32 B/draw | 1.6 GB |
| Optional E1: CB0 of kNN-VC-shifted content audio | gated by P0/P5 | 2 B | 0.05 GB |

- Total core storage is about 32 GB of the 1.2 TB quota.
- Cost estimate for the big encode (train-clean-360 + other-500): Praat on 8 CPUs at about 240× real time is about 2 h; Mimi fp32 encode of clean + 2 perturbed copies (about 1,500 h of audio) is about 3-4 h; CAM++ is 1 h. Budget 10-14 slice-h.
- The big encode must not perturb 360/500 before P0 has been read. If its clean encode has started, let it finish, and re-run only `encode_perturbed.py` with the chosen preset under a new suffix.
- I/O: one memory map per array. The trainer's prefetch thread (threads, not forked workers, so no copy-on-write) reads from node-local disk if the P1 job's `df` shows at least 40 GB free there; otherwise from NFS with `--mem=60G`. Gate: data wait under 20% of step time (measured in P6).

---

## 4. Experiment ladder

Gate results are recorded in `docs/training_tips.md` (standing rule) and in each job's `summary.json`. Every result is shared with the interpreter session.

The probes P0b-P4 each cost at most 3 slice-h. They are packed into one job, J-probe (about 6 h, 10 h limit), to respect the 1 h submission gap.

### P0. Speaker information in Mimi streams (inside the queued encode job)
- Purpose: decide whether NANSY perturbation is enough before the big encode.
- Cost: about 0.2 slice-h after the encode of train-clean-100 + dev-clean (6-8 slice-h, already queued).
- Metric:
  - `probe_speaker_info.py` on dev-clean, chapter-disjoint enrolment and test (speakers with one chapter are dropped), nearest-centroid speaker-ID accuracy and utterance-vs-centroid EER, for `cb0`, `pcb0_0`, `pcb0_1`, `cb1`, `lat`, and the transcript word-bag control `text`;
  - excess(x) = acc(x) − acc(text);
  - CB0 agreement between clean and perturbed audio (logged by `encode_perturbed.py`).
- Go/no-go:
  - agreement ≥ 0.30 → keep PRESET. Below 0.30, perturbation scrambles content: use the milder preset (fs 1.25, ps 1.5) for the big encode.
  - excess(pcb0) ≤ 0.5·excess(cb0) → NANSY is enough.
  - excess(pcb0) ≥ 0.8·excess(cb0) → also generate E1 kNN-VC-shifted CB0 in the big encode (+15 slice-h).
  - In between → proceed, and recheck at P5.
- On failure: probe crash → rerun inside J-probe (it is CPU minutes).

### P0b. Content-controlled leakage probe on VCTK Dev shared text (J-probe)
- Purpose: a speaker-ID measurement in which the text is identical across speakers, so anything above chance is voice or accent. This is the analysis-section number.
- Cost: 0.3 slice-h. It encodes the shared-text sentences of the 20 Dev speakers (identified by `shared_sentence_id` in the Dev pair list), clean plus 2 perturbed draws.
- Metric: 20-way speaker-ID (chance 0.05) and balanced sex accuracy, leave-one-sentence-out, with nearest centroid over bag-of-CB0 and over mean CB0-codeword embeddings. Streams: cb0, pcb0, cb1, lat.
- Gate: none. It feeds the P0 decision. If perturbed-CB0 speaker-ID is still at least 5× chance or sex accuracy is at least 0.80, E1 is enabled even if P0 passed.

### P1. Decoder tolerance (J-probe)
- Purpose: the decode route, the precision, and how accurate the generator must be.
- Cost: 1.0 slice-h. Inputs: 300 dev-clean utterances plus the 50 Dev-screen sources. The basic copy_latent vs copy_codes check is also in P5.
- Conditions:
  - 32-code fp16 decode (reference);
  - continuous decode in fp32 and fp16;
  - Q8 and Q16;
  - z + N(0, ε²) for ε ∈ {0.05, 0.1, 0.2, 0.4}, in standardised-lat units, and the same MSE applied in decoder-input metric;
  - 3-frame moving average.
- Metric: UTMOS22, capped WER with Whisper-large-v3, WavLM-SV SIM to the speaker's centroid (metrics.py functions).
- Go/no-go:
  - continuous route allowed if |ΔUTMOS| ≤ 0.05, ΔWER ≤ 0.5 points and ΔSIM ≥ −0.01 against the 32-code decode; otherwise Q32 is the only output;
  - record ε₀.₂, the noise at which UTMOS drops by 0.2;
  - if ε₀.₂ < 0.1, schedule R5 early and make NFE 32 the default;
  - if decoder-metric noise is at least 2× more harmful per unit MSE than standardised-lat noise, the K6 decoder-metric loss becomes a P6 arm;
  - if fp16 and fp32 continuous decodes differ by more than 0.02 UTMOS, decode continuous in fp32 (the default) and log it.
- Also: `df -h` of node-local disk and the free RAM under the cgroup, for the I/O decision.

### P2. Content ceiling at 12.5 Hz (J-probe)
- Purpose: can perturbed CB0 carry enough content, and is soft-CB0 worth having?
- Cost: 3.0 slice-h.
- Probe model (about 5M): input projection → 4-layer transformer (d=256, 4 heads) → ConvTranspose1d stride 4 to 50 Hz → character CTC over 28 symbols plus blank.
  - The 4× upsampling is required: read speech runs at about 15 characters/s, above 12.5 frames/s.
  - Training: 15k steps, batch 64, on 50 h of train-clean-100 (LibriTTS-R `.normalized.txt` lowercased).
- Streams:
  - (a) clean hard CB0;
  - (b) perturbed hard CB0 (draw 0);
  - (c) perturbed soft CB0, T=0.5;
  - (d) perturbed soft CB0, T=1;
  - (e) the full standardised latent (upper bound).
- Metric: greedy-CTC CER on 300 dev-clean utterances and on 100 VCTK Dev wer_primary sources (perturbed on the fly), plus the P0-style chapter-disjoint speaker-ID on (c) and (d).
- Go/no-go:
  - CB0-only content is GO if CER(b) ≤ 1.5·CER(e) and CER(b) ≤ 15% on dev-clean;
  - soft-CB0 becomes a P6 arm if CER(soft) ≤ 0.85·CER(b) and its speaker-ID excess is at most excess(b) + 2 points;
  - if CER(b) > 1.5·CER(e), start implementing F1 (Mimi-CTC content adapter) on the Mac in parallel with P6; P6's WER gate decides whether it is needed.

### P3. Training-free Mimi floors and baselines on Dev-S100 (J-probe)
- Purpose: calibrate the iteration subset and provide Mimi-native floor rows for the paper.
- Cost: 1.5 slice-h.
- Systems:
  - kNN in standardised latent (k=4, cosine), continuous decode;
  - Gaussian OT of the source latent to the reference's mean and covariance (`crt_decode_lab.gaussian_ot` logic);
  - `copy_latent` and `copy_codes`;
  - kNN-VC, FreeVC, Seed-VC and FACodec through `run_baselines.py --pair_ids` (Dev baselines exist only as a 20-pair smoke test).
- Metric: the Dev-S100 protocol of §5.2.
- Gate: none. These are the reference levels the P6 and R1 gates are read against.

### P4. CAM++ sanity (J-probe)
- Purpose: CAM++ was trained on Chinese data; check its English coverage.
- Cost: 0.3 slice-h.
- Metric: EER of CAM++ on dev-clean (10 s segments) and on VCTK Dev (3 s segments).
- Go/no-go: keep CAM++ if EER ≤ 5% on dev-clean and ≤ 8% on VCTK Dev. Otherwise the Mimi-AAM vector becomes the default in P6. A different external SV model would need a download, which is the user's decision.

### P5. A/B pilot (already chained, `jobs/job_pilot.slurm`, unchanged)
- Purpose: a first end-to-end signal on whether perturbed CB0 lowers leakage at acceptable WER.
- Setup: 2 × 40M, 60k steps on train-clean-100. Arm A: clean CB0. Arm B: perturbed CB0. Dev screen shard 0/20 (50 pairs, female targets only, the 30 s budget of which the adapter uses only the first 20 s). Includes copy_latent and copy_codes.
- Cost: 6-9 slice-h.
- Metric: `summarize.py` output against CRT-AL (cand_pm18_rq), anc_faiss, target_copy and mimi_copy_src on the same pairs.
- Go/no-go for P6:
  - arm B capped WER ≤ 12%, SIM ≥ 0.84, UTMOS ≥ 3.0;
  - SIM_src(B) ≤ SIM_src(A) − 0.01 on M→F pairs.
- On failure:
  - both arms WER > 20% → check P1 (decoder) and P2 (content) first;
  - WER(B) − WER(A) > 5 points → perturbation too strong: milder preset;
  - SIM < 0.80 → expected from the short-prompt mismatch; P6's multi-utterance prompts and uncut reference address it;
  - SIM_src(B) ≥ SIM_src(A) → enable E1.

### Gate record, 25 Oct 2026 (pilot job 5644, rescore job 5718; all numbers on Dev pairs only)
| Probe | Result | Decision |
|---|---|---|
| P0 | CB0 agreement 0.519-0.541 (7 shards); speaker ID: text 0.150, CB0 0.198, pCB0 0.156/0.162, CB1 0.433, latent 0.674 | keep PRESET; NANSY enough (excess ratio about 0.2) |
| P0b | VCTK shared text: speaker ID CB0 0.47, pCB0 0.32/0.36 (chance 0.05), CB1 0.76, latent 0.94; sex accuracy for pCB0 0.72/0.76 | the E1 trigger (pCB0 ≥ 5× chance) is met. It was defined for CB0-only content, so it is deferred to P6's leakage gate, where E1 remains the second remedy (deviation D-mvc-1) |
| P1 | continuous decode = Q32 (ΔUTMOS −0.007); Q16 −0.06, Q8 −0.29; ε₀.₂ = 0.22 (std-lat) / 0.24 (decoder metric); 3-frame smoothing −2.8 UTMOS | continuous route allowed; no early R5 and no NFE 32 default; no K6 arm. Over-smoothing is what the decoder cannot forgive |
| P2 | CTC CER on dev-clean (VCTK): CB0 16.8 % (30.8), pCB0 33.8 % (48.5), soft T=1 27.4 % (42.7), latent 9.1 % (21.8) | CB0-only content is NO-GO; start F1 (second content stream). The soft-CB0 arm qualifies but is dominated by Whisper (below) |
| P3 | Dev-S100, 30 s, capped WER / SIM / UTMOS / leakage same, cross: kNN-VC 8.77 / 0.927 / 3.61 / +0.034, +0.064; FreeVC 4.39 / 0.919 / 3.91 / +0.037, +0.078; Seed-VC 3.80 / 0.926 / 3.74 / +0.018, +0.058; FACodec 2.63 / 0.895 / 3.60 / +0.070, +0.104; CRT-AL 28.07 / 0.809 / 2.63; kNN-latent 73.98 / 0.883 / 1.45; OT-latent 54.39 / 0.851 / 1.72; Mimi copy 2.05 / – / 3.94 | reference levels for P6 and R1. All four baselines exceed the +0.03 leakage margin cross-gender |
| P4 | CAM++ EER 4.1 % (dev-clean, 10 s), 2.4 % (VCTK Dev, 3 s); Mimi-latent mean 11.2 % / 25.3 % | keep CAM++ |
| P5 | screen shard 0/20 (50 pairs, female targets), 30 s: A (CB0) 38.6 / 0.742 / 2.15; B (pCB0) 48.5 / 0.725 / 2.10; **C (pCB0 + Whisper-small of the clean source) 2.27 / 0.805 / 2.52, leakage same −0.034, cross +0.017**; C with perturbed-source Whisper 8.33 / 0.824 / 2.49 / −0.025, +0.028; CRT-AL 25.8 / 0.818 / 2.59; dev-clean reconstruction of B: WER 25.5, SIM 0.864, UTMOS 2.01 | B fails the P5 gate on WER, SIM and UTMOS. The Whisper stream (arm C) becomes the default content route for P6 (F1 by Whisper features, not a Mimi-CTC adapter). The open problem is naturalness: UTMOS 2.5 against a 3.9 decode ceiling |

| P6 | job 5771, Dev-S100, 30 s, capped WER / SIM / UTMOS / leakage same, cross. **V** (v1 + Whisper, 29.7M, 60k steps, train-clean-100): 4.39 / 0.837 / 2.49 / −0.027, +0.020. **Vf** (V + audit exclusions, 2,545 training utterances out): 4.97 / 0.842 / 2.52 / −0.029, +0.028. V at 3 s: 4.97 / 0.797; at 180 s (chunk ensemble): 5.26 / 0.836. Arm C on Dev-S100: 4.97 / 0.808 / 2.66. Same-speaker reconstruction (V): WER 4.35, SIM 0.903, UTMOS 2.36. Reference levels on the same pairs: CRT-AL 28.07 / 0.809 / 2.63; Mimi copy 3.22 / – / 3.83 | Passes WER ≤ 10, leakage ≤ +0.03 in both strata, the 3 s and 180 s SIM conditions, and reconstruction WER ≤ 5. **Fails SIM ≥ 0.86 (0.837) and UTMOS ≥ 3.2 (2.49).** A0 rule: WER(Vf) ≤ WER(V) + 1.0 and SIM(Vf) ≥ SIM(V) − 0.005 both hold, so R1 uses the audit filter. Fix cycle (≤ 6 slice-h, as planned): an inference sweep on Dev-S100 (NFE, CFG w, sway, Q32 decode), then the AAM + CAM++ arm if SIM stays below 0.86 |

Deviation D-mvc-1 (25 Oct): P0b met the E1 trigger, but the trigger assumed CB0-only content. With the Whisper stream, leakage is measured directly at P6 (≤ +0.03 per stratum against target_copy). If it fails there, the remedies are, in order: source perturbation at inference, then E1.

Deviation D-mvc-2 (27 Oct 2026, written before job e3 reads R1's 100k snapshot):
- Finding:
  - R1 at 40k and 60k (temp 0 + Mimi relax 0.15×2, the inference default since E1) raises SIM to 0.88–0.91.
  - It also fails the leakage gate: cross-gender +0.085 / +0.090, same-gender +0.044.
  - The leak runs through the Whisper-small content stream (content-only output SIM_src 0.81, against 0.68 for p6_V).
  - D-mvc-1's first remedy, source perturbation at inference, makes it worse in e2: whisper_src perturbed +0.119, perturb_src +0.129.
  - CAM++ sees no R1 − p6_V leakage difference, so part of the WavLM regression is a global similarity offset.
- New first remedy: source-negative guidance (SNG). Each ODE step also runs the model with the clean source's own latent as the prompt and the source's CAM++ vector, and adds w (v_cond − v_source) (vc_adapter neg_src, neg_src_until).
- Rule, applied on the snapshot that will be used (never carried over from another):
  - pick the smallest w in {0.5, 0.75, 1.0} that meets all of the conditions below;
  - if only condition 3 fails, use the t < 0.5 interval variant at that w.

  Conditions:
  1. Leakage ≤ +0.03 in both strata (the existing gate). The version paired on target_copy is reported with its SE.
  2. SIM ≥ the same snapshot's SIM without SNG − 0.005.
  3. UTMOS ≥ the same snapshot's UTMOS without SNG − 0.08.
  4. Capped WER change ≤ +2.0 (a sanity check; the binding WER check is full Dev at R2).
  5. ECAPA ΔSIM_src, cross-gender, has the same sign as WavLM's and at least half its size.
  6. SIM(3 s) ≥ SIM(30 s) − 0.06 and SIM(180 s) ≥ SIM(30 s) − 0.005.
- Diagnostic only, not a gate: the cohort split (global, gender, source-specific terms); flag a global term below target_copy − 0.03.
- If no w passes:
  - the fallback is B1, a shifted-content fine-tune of the final R1: Whisper and CB0 taken from p6_V conversions of the training utterances to other speakers, p_shift 0.5, 20k steps;
  - its keep rule against the same R1 parent: ΔSIM_src ≤ −0.02 in both strata, ΔSIM ≥ −0.005, ΔUTMOS ≥ −0.05, WER change ≤ +0.5 on full Dev.
- R1 is always also reported without SNG, as an ablation.
- R2 addendum (the R2 rule is not frozen yet):
  - "best SIM" in the R2 rule is computed over leakage-passing configurations only;
  - the grid is SNG {0, 0.5, 0.75, 1} × CFG {0, 1}, with no CFG ≥ 2;
  - a UTMOS constraint is added: UTMOS ≥ best − 0.05 among those.

D-mvc-2 applied (27 Oct, job e3 = 6108, R1 ema_0100000, Dev-S100, temp 0 + relax 0.15×2):
- Results:

  | w | SIM | UTMOS | WER | leakage same / cross |
  |---|---|---|---|---|
  | 0 | 0.888 | 3.686 | 2.34 | +0.050 / +0.096 |
  | 0.5 | 0.901 | 3.673 | 3.22 | +0.000 / +0.047 |
  | 0.75 | 0.901 | 3.693 | 3.22 | −0.022 / +0.031 |
  | 1.0 | 0.898 | 3.714 | 1.75 | −0.042 / +0.016 |
  | 1.0, t < 0.5 | 0.899 | 3.727 | 1.75 | −0.033 / +0.022 |

- 0.5 and 0.75 fail condition 1.
- w 1.0 passes all six conditions:
  - 3. ΔUTMOS +0.028;
  - 5. ECAPA cross ΔSIM_src −0.048±0.006 against WavLM −0.081±0.010 (0.60×), and ECAPA SIM to the target +0.073±0.009;
  - 6. SIM at 3 s 0.868, at 180 s 0.902.
- Guards: HNR 11.6 dB, voiced frames kept 0.936.
- Selection on this snapshot: w 1.0. The rule is re-applied on R1's final snapshot.
- 27 Oct 2026, 16:28 (+07:00), at the freeze of `mvc/ANALYSIS_PLAN_mvc_v1.md`, before R1's final snapshot is scored:
  - from here on, condition 1 uses the paired natural-level leakage NL, and conditions 1–5 become R2's Dev-S100 screen with the choice made on full Dev (analysis plan §0.3 rows 4–6, §3.3, §4.3–4.6);
  - the 300k extension is replaced by the exploratory R1x (analysis plan §0.3 row 9, §4.10).

### A0. Per-utterance data audit (added 25 Oct 2026, `audit_data.py`, `jobs/job_audit.slurm`)
- Purpose: find the training utterances that would teach the model something wrong, before the main run.
- Pass 1 (GPU job, about 2-4 h for 555 h of audio):
  - audio: readability, finite samples, clipping fraction, speech fraction, leading and trailing silence;
  - stored frames against the audio length;
  - latent RMS;
  - UTMOS22-strong.
- Pass 2: CAM++ cosine of each utterance to the leave-one-out centroid of its speaker, CB0 agreement per perturbed draw, and robust z-scores.
- The rules are fixed in the script before any filtered run (`RULES`, `EXCLUDING`). They do not depend on training results. A rule that flags more than 5% of the audited hours is reviewed once, on the audit distribution alone, before the first filtered run, and any change is logged here.
- Outputs: `results/audit_v1/exclude.txt` (with its sha256), `flags.jsonl`, `audit_summary.json`.
- Use: P6 gets an arm V-filt (V plus `--exclude_utts results/audit_v1/exclude.txt`; train.py refuses a shard the audit does not list as audited).
- Decision rule, fixed before P6: R1 uses the filter if capped WER(V-filt) ≤ WER(V) + 1.0 point and SIM(V-filt) ≥ SIM(V) − 0.005 on Dev-S100, on the same 100 pairs. Otherwise R1 trains on everything, and the audit is reported as a data description. The per-speaker and per-duration breakdown of `audit_summary.json` is logged beside the decision.
- Besides a rule over 5% of hours, two more things trigger the one-time review: a speaker losing more than 25% of its hours, or a duration bin excluded at more than 3× the corpus rate.
- 25 Oct review (adversarial, 2 lenses) changed the rules before any run:
  - an absolute −70 dBFS floor in the activity test;
  - speech fraction with up to 1 s of edge silence discounted;
  - the fixed CAM++ floor of 0.5 replaced by "closer to another speaker's centroid than to its own";
  - speakers that would lose more than 25% to the speaker rule are reviewed, not cut;
  - coverage is checked by train.py;
  - an existing list is never silently replaced.
- **Review 1 (25 Oct, 15:10), on the distribution of audit job 5650 alone, before any filtered run.** At that time speaker and perturbation data covered train-clean-100 + dev-clean, 62.4 h.
  - Triggers:
    - perturbation_extreme flagged 11.8 % of the covered hours, mostly one bad draw of two on short utterances (57 % under 3 s);
    - speaker_outlier flagged 43 % of utterances under 2 s, 3.6× its overall rate. Half of those were within 0.02 cosine of the nearest other speaker.
  - Changes:
    - perturbation_extreme requires every draw outside 0.15-0.9: 0.7 % of hours.
    - speaker_outlier requires spk_cos < spk_other − 0.05, or spk_z < −4: 1.6 % of hours.
    - Shares are computed over the trainable (≥ 2 s) hours each rule covers. Job 5650 had divided by all 471 h, which hid the rates.
  - Known limit: one bad perturbation draw of two stays in training as noise.

### P6. v1 pilot plus throughput (v1 code)
- Purpose: validate the v1 recipe at small scale before R1.
- Setup: 40M, 60k steps, 16k-token batches, on train-clean-100 (+360 if encoded).
  - Arm V: the v1 recipe with hard perturbed CB0 and CAM++.
  - Arm V-soft: only if P2 said so; otherwise V-AAM (Mimi-AAM vector instead of CAM++).
  - Optional arm V-K6 (decoder-metric loss) if P1 asked for it.
  - P5 arm B serves as the "scaffold recipe" control.
  - The job ends with 1,000 steps of the 121M config to measure throughput.
- Cost: 8-10 slice-h.
- Metrics:
  - Dev-S100 at 30 s (NFE 16, w 0.7, continuous);
  - SIM at 3 s and 180 s (both 180 s modes);
  - dev-clean reconstruction WER;
  - for the 121M config: s/step, peak memory, data-wait share.
- Go to R1 if:
  - capped WER ≤ 10%, SIM ≥ 0.86, UTMOS ≥ 3.2;
  - SIM_src − SIM_src(target_copy) ≤ +0.03 in both the same-gender and cross-gender strata;
  - SIM(3 s) ≥ SIM(30 s) − 0.06, and SIM(180 s) ≥ SIM(30 s) − 0.005;
  - reconstruction WER ≤ 5%.
- Throughput:
  - ≤ 0.6 s/step → 200k steps as planned;
  - 0.6-0.9 s/step → d=640 with 16 layers (about 85M), or 150k steps;
  - data wait > 20% of the step → stage the shards to node-local disk.
- On failure (one fix cycle of at most 6 slice-h, then a re-think):
  - WER fails → content (soft, then F1, then F2);
  - leakage fails → inference source perturbation, then E1;
  - 180 s degrades → keep the ensemble, or test the cross-attention memory;
  - SIM fails → w ∈ {1, 2} and the AAM+CAM++ combination.

### R1. Main run
- Setup: 121M, about 555 h, 200k steps, 32k-token batches. One 48 h job with resume, rather than 72 h, so it starts sooner behind reservations.
- Cost: 30-40 slice-h, plus 1.5 slice-h for snapshot evaluations (`job_eval.slurm` on Dev-S100 at 50k, 100k, 150k and 200k).
- Gate at 100k: Dev-S100 WER ≤ 8% and SIM ≥ 0.87. Otherwise stop and apply the matching P6 remedy rather than training on.
- Checkpoint choice: Dev-S100 plus dev-clean reconstruction WER, never training loss.

### R2. Dev selection (rule frozen in `ANALYSIS_PLAN_mvc_v1.md` before R2 starts)
- Cost: about 6 slice-h.
- Steps:
  1. One-factor grid on Dev-S100 around the default: w ∈ {0.5, 1, 2}; decode Q32; source perturbation K=2; NFE 32; 180 s full+scaling. About 8 configs × 0.15 h.
  2. Up to 3 finalists plus the default on full Dev (1,000 pairs, 30 s, @src rows, ECAPA).
  3. The chosen configuration at 3 s and 180 s on full Dev.
- Rule: lowest capped WER subject to SIM ≥ best − 0.01 and SIM_src − SIM_src(target_copy) ≤ +0.03 in every gender stratum. If no configuration passes the leakage condition, take the least-leaking one within 0.01 SIM of the best.
- Triggers:
  - R4 if any stratum's leakage exceeds +0.03, or the gate-L margin is worse than kNN-VC's;
  - R5 if Dev UTMOS < mimi_copy_src − 0.3.

### R4. Latent-space self-transformation (conditional, before the freeze)
- Steps:
  1. Convert all ≈355k training utterances to random other training speakers (prompt 10-30 s, NFE 8). Take CB0 by quantising the predicted latent, with no audio decode. 3-6 slice-h.
  2. Validity check: CB0 agreement between `quantize(pred)` and `encode(decode(pred))` on 200 utterances must be at least 0.8; otherwise re-encode decoded audio (+6 slice-h).
  3. Fine-tune 60k steps, with content 50% self-transformed and 50% perturbed. 7.5 slice-h.
  4. Dev evaluation. 1 slice-h.
- Keep only if Dev SIM_src falls by at least 0.01 with capped WER up by at most 0.5 points and SIM down by at most 0.005. Otherwise report it as an ablation and keep R1.

### T1. One-shot test (after the freeze; §5.4)
- Cost: 12-14 slice-h.
- No decisions are made from it.

### R3. Paper ablations (Dev only; can run after T1)
- Each is 100k steps, compared with R1's 100k EMA checkpoint. About 13.5 slice-h each including evaluation; about 41 slice-h for all three.
- Priority order:
  - A1, clean-CB0 content: the leakage headline (prediction: cross-gender SIM_src +0.02 to +0.05);
  - A2, no CAM++ (prompt only), or AAM in place of CAM++;
  - A3, scaffold prompts: single utterance, `restart` positions.
- Inference-only ablations at no training cost: NFE 1-32, CFG w, decode route, 3/30/180 s, 180 s mode, CAM++ switched off.

### R5 / R6 (optional)
- R5, prematched decoder fine-tune: 30 slice-h, if triggered. Kept as the variant "MimiFlow-VC+D" only if Dev UTMOS rises by at least 0.15 with no WER or SIM loss.
- R6, block-causal streaming fine-tune: 2-frame lookahead content encoder, KV-cached prompt, tolerance-based equivalence test between incremental and masked generation. 10-15 slice-h.
- R6b, shortcut few-step distillation: about 8 slice-h.
- Both are secondary systems and need their own pre-registration before any test run.

### Compute summary (2g.35gb slice-hours)

| Stage | Slice-h |
|---|---|
| S0 encode train-clean-100 + dev-clean, P0 (queued) | 6-8 |
| J-probe: P0b, P1, P2, P3, P4 | 6 |
| P5 pilot (queued) | 6-9 |
| S2 encode train-clean-360 + other-500 | 10-14 |
| P6 v1 pilot + throughput | 8-10 |
| R1 main + snapshot evaluations | 32-42 |
| R2 Dev selection | 6 |
| T1 test | 12-14 |
| R3 ablations | 41 |
| Core total | ≈127-150 |
| Conditional: R4 12-18, E1 15, F1 20-25, R5 30, R6 10-15, R6b 8 | up to +116 |

Calendar, assuming 1-2 concurrent jobs:
- 25-27 Oct: S0, P5 and J-probe; v1 code and tests on the Mac.
- 27-30 Oct: S2 and P6.
- 30 Oct-2 Nov: R1.
- 3-5 Nov: R2 (and R4 if triggered), then freeze.
- 6-8 Nov: T1.
- 8-14 Nov: R3 and Mac analysis.

---

## 5. Evaluation

### 5.1 Plumbing (same rows, same metrics, same statistics as CRT-AL and the baselines)

- Conversion: `mvc/run_mvc.py` reuses `run_baselines.convert_pair`, `get_reference`, `seed_all` and `describe`, so the frozen `baselines.SYSTEMS` and every FREEZE_v1 file stay unchanged. The reference is always `make_pairs.load_reference(R[budget])`.
  - `python -u run_mvc.py --journal $J --pairs $J/pairs/<set>.jsonl --set <set> --budget <b> --name mvc_v1 --cfg '{"run_dir":..., "ckpt":"ema_200000.pt", "steps":16, "cfg":0.7, "decode":"latent"}' --out_dir $MVC/test/conv`
  - Also run `mvc_v1_q32` (decode "codes") and `mvc_copy_latent` (mode copy_latent, the codec ceiling of the deployed path) at 30 s.
- Rows: `python -u $J/lab_rows.py --pairs ... [--source_e $J/pairs/vctk2k_source_E.json] --manifest <each manifest> --out $MVC/test/m/<part>/rows/<set>_<b>.jsonl`. This also writes the `_src.jsonl` rows.
- Metrics:
  - targets: `$J/metrics.py --metrics mcd,wer,sim,secs,utmos --whisper_models large-v3,base --wavlm_model $J/models/wavlm-base-plus-sv --device cuda`;
  - @src rows: `--metrics sim,secs`;
  - Dev uses large-v3 only.
  - After any code or checkpoint change, write to a new `--out_jsonl`, because metrics.py reuses rows with unchanged input fields (RUNBOOK §0).
- ECAPA: `PYTHONPATH=$DATA_STORAGE/site/x2_ecapa python -u $J/score_ecapa.py --model_dir $DATA_STORAGE/models/spkrec-ecapa-voxceleb --systems mvc_v1,mvc_v1_q32 --rows ... --out ...`, on both target and `_src` rows, with `--smoke` first.
- Statistics (Mac, `.venv-stats`): `analyze_mvc.py` follows the `analyze_exploratory.py` template.
  - `analyze_tests.load_set` on the job-M dirs (`results/v1/m/t1..t4`, `x1`) merged with the mvc dirs; `TestSet`; `hypothesis` for q ∈ {wer (capped), sim, utmos, mcd, sim_src, m9};
  - crossed speaker bootstrap, B=10,000, seed 20260928, INDEPENDENT for VCTK-2k and LINKED for LTC-2k;
  - Holm within each family; `gate_l_strata` with margin +0.01; `sv_trials`/`eer`; `levels`.
  - Comparators: cand_pm18_rq (CRT-AL), faiss, knnvc, freevc, seedvc, facodec; mimi_copy_src and target_copy as ceilings.
  - Dev analyses call `stats.level`/`compare` with LINKED mode, as `select_crt_al.Dev` does.
- Precision deviation: fp32 Mimi encoding and fp32 continuous decoding, against the fp16 used for published controls. Recorded in `describe()` and in the paper-3 plan, and controlled by `mvc_copy_latent` and the `copy_codes` run.

### 5.2 Quick Dev protocol for iteration

- Dev-S100 (`mvc/pairs/dev_s100.txt`), built once on the Mac from the Dev pair list with seed 20260928:
  - 100 pairs, 25 per gender combination (F2F, F2M, M2F, M2M);
  - at least 50 wer_primary and at least 50 with P;
  - every Dev target used at least 3 times.
  - Run via `run_mvc.py --pair_ids`.
- Cost per configuration: about 0.15 slice-h (conversion 2 min; large-v3 WER, SIM and UTMOS 6-8 min; @src SIM).
- Reported: capped WER (wer_primary pairs), SIM, UTMOS, and SIM_src per stratum; ΔSIM_src against target_copy; M9. Shown beside CRT-AL and anc_faiss (v1/d1-d2) and the four baselines (P3) on the same pairs.
  - `summarize.py` must be extended to stratify by `genders` if it does not already.
- Precision at n=100: SE about 1-2 WER points and about 0.005-0.01 SIM. Smaller differences are noise; full Dev decides.
- Kept for continuity: the legacy screen shard 0/20 (50 pairs, female targets only), used by P5.
- In-domain proxy: same-speaker reconstruction on 300 dev-clean utterances (capped WER, SIM to the own-speaker centroid).

### 5.3 Pre-registration (`mvc/ANALYSIS_PLAN_mvc_v1.md`, frozen before R2)

- H1 (primary): mvc_v1 beats CRT-AL on capped WER, SIM and UTMOS on VCTK-2k and LTC-2k at 30 s. Six tests, Holm.
- H2: mvc_v1 against each of the four baselines, two-sided, same endpoints plus SIM_src. Holm is a guide here.
- H3, leakage:
  - SIM_src − SIM_src(target_copy) per stratum;
  - gate L against kNN-VC;
  - M9;
  - ECAPA coverage of the source-to-target distance.
- H4: the budget sweep at 3 s and 180 s.
- S: the Q32 route and the ECAPA agreement.
- Rule: one test run per frozen system, no rescue runs (as in ANALYSIS_PLAN_v1 §9). Optional systems (R5, R6) need their own addendum.

### 5.4 Test run T1

- VCTK-2k (2,000 pairs) and LTC-2k (1,800 pairs) at 30 s (primary) plus 3 s and 180 s. `mvc_v1_q32` and `mvc_copy_latent` at 30 s only.
- Cost: conversions about 2 h; metrics on about 19k target rows about 8 h; @src rows about 1.5 h; ECAPA about 1 h.
- Outputs go under `$MVC/test/`, never into the frozen `$J/v1/` dirs.

---

## 6. Paper targets, angle, titles

### 6.1 Numbers to beat (30 s reference; VCTK-2k / LTC-2k)

| System | Capped WER % | SIM (WavLM-SV) | UTMOS |
|---|---|---|---|
| CRT-AL (published, re-measured) | 16.3 / 10.3 | 0.806 / 0.911 | 2.63 / 2.73 |
| kNN-VC | 6.2 / 3.8 | 0.895 / 0.929 | 3.66 / 3.84 |
| FreeVC (VCTK-seen) | 2.0 / 2.5 | 0.887 / 0.897 | ≈4.0 |
| FACodec | 2.5 / 2.1 | 0.876 / 0.910 | 3.63 / 3.58 |
| Seed-VC v1 small | 2.6 / 2.5 | 0.919 / 0.946 | 3.79 / 4.07 |
| Mimi copy, 32 codes (ceiling) | 1.1-1.9 | ≈0.98 | 3.80 / 4.16 |
| Natural target | – | 0.963 | – |

- 3 s reference: kNN-VC WER 60.3 / 34.6 (X1); Seed-VC SIM 0.898 / 0.923.
- ECAPA on VCTK-2k: CRT-AL 0.161, target_copy 0.795.

Tiers:
- Must (the paper is viable): beat CRT-AL on all three endpoints on both sets, Holm-significant; ECAPA coverage of the source-to-target distance ≥ 50% on VCTK-2k (CRT-AL: 6%); leakage reported per stratum.
- Should: capped WER ≤ 5.0 / 3.5 (beats kNN-VC), SIM ≥ 0.89 / 0.93, UTMOS ≥ 3.6 / 3.9, and at 3 s WER within 1.5 points of 30 s.
- Stretch (Seed-VC parity): WER ≤ 3.0 / 2.5, SIM ≥ 0.91 / 0.945.

Prediction (subjective, priors from §1.2):
- VCTK-2k: WER 4-7% with CB0 alone (3-5% with a second content stream), SIM 0.86-0.90, UTMOS 3.4-3.7.
- LTC-2k: WER 3-5%, SIM 0.91-0.94, UTMOS 3.8-4.0.
- Probabilities: P(must) ≈ 0.85, P(should) ≈ 0.45, P(stretch) ≈ 0.15.
- A gate-L pass is not promised. Source rhythm, F0 contour and accent survive frame-aligned CB0 by design, and both SV models are sensitive to them.

### 6.2 Angle

1. Analysis: where speaker identity and content live in Mimi.
   - P0, P0b and P2: CB0, perturbed CB0, CB1 and the latent.
   - The split-RVQ explanation of why token swaps fail (FAISS 126%, CRT 89%, CRT-AL 16%, against the external results above 100%).
2. Method: the first zero-shot VC native to Mimi's continuous latent, with a frozen codec and output either continuous or as 32 genuine Mimi tokens for Moshi, Hibiki, PersonaPlex or CSM pipelines. RTF about 0.02-0.05.
3. Theory link to the journal paper (decide on the decoder input, prefer the posterior mean), tested as ablations rather than claimed:
   - continuous vs Q32 decoding;
   - soft-CB0 (posterior-mean content) vs hard;
   - decoder-input-metric loss.
4. Leakage-aware, pre-registered evaluation: SIM_src per stratum, gate L, a second SV model, a crossed speaker bootstrap, three reference budgets. Ablation A1 quantifies how much source voice flows through CB0.
5. A small listening test (MOS/SMOS), using the existing paper-2 listening tooling.
6. Venue: chosen by the same advisor rule as paper 2 (SCIE + JCR Q2 + Scopus Q2); see `paper/journal_extension/journal_target.md`.

### 6.3 Title ideas

- "MimiFlow-VC: Zero-Shot Voice Conversion by Flow Matching in the Continuous Latent Space of the Mimi Speech Codec"
- "Where Is the Speaker in Mimi? Probing and Converting Voice Identity in a Split-RVQ Streaming Codec"
- "Beyond Token Swapping: Leakage-Aware Zero-Shot Voice Conversion on Neural-Codec Latents"

---

## 7. Risks and fallbacks

| # | Risk | Trigger (measured) | Fallback, in order | Extra cost |
|---|---|---|---|---|
| 1 | CB0 content ceiling | P2 CER(b) > 1.5·CER(e), or P6/R1 Dev-S100 WER > 8-10% | Soft-CB0 → F1 Mimi-CTC adapter (a copy of Mimi's encoder transformer, char CTC on LibriTTS-R text, causal, posteriorgram as a second stream) → F2 WavLM-L L12/L18 k-means units, offline (drops the streaming-ready claim) | 0 / 20-25 / 8-10 |
| 2 | Source leakage via CB0 | SIM_src − target_copy > +0.03 in any stratum; P0 excess ratio ≥ 0.8 | Inference source perturbation (K=2) → R4 self-transformation → E1 kNN-VC-shifted content → E2 CB0-extractor invariance fine-tune (breaks the "re-voice Moshi CB0" use case) | 0 / 12-18 / 15 / 3-5 |
| 3 | Perturbation draws near identity teach a voice shortcut | A1 vs main shows no leakage gain | Already mitigated by the near-identity skip; widen the skip band to ln 1.15 | 0 |
| 4 | Decoder off-manifold | P1 ε₀.₂ < 0.1; Dev UTMOS < copy − 0.3 | Q32 route, NFE 32 → R5 prematched decoder fine-tune (adversarial, never L1-only) | 0 / 30 |
| 5 | Weak speaker generalisation to accented VCTK | Dev SIM < 0.87 while dev-clean reconstruction SIM ≥ 0.93 | w ∈ {1, 2}; CAM++ + AAM; 180 s ensemble → add an accented, VCTK-disjoint corpus (GLOBE; the user must accept its terms) | 0 / ≈25 |
| 6 | Long-reference failure | SIM(180 s) < SIM(30 s) − 0.005 | Chunk ensemble (default) → full prompt with logit scaling → retrain with cross-attention memory (D3 style, position-free keys) | 0 / 0 / ≈35 |
| 7 | Throughput below estimate | P6 > 0.6 s/step | d=640×16 (≈85M) or 150k steps; node-local staging | 0 |
| 8 | Data not ready | other-500 not encoded at R1 start | Train on train-clean-100+360 (245 h) and report | 0 |
| 9 | Overfitting (~80 epochs) | dev-clean reconstruction WER rises between snapshots | Early stop at the best snapshot; more perturbation draws (K=3) only for a revision | 0 |
| 10 | Precision mismatch with the published fp16 controls | – | `mvc_copy_latent` and `copy_codes` controls on every set; dtype in `describe()`; noted in the paper-3 plan | 0 |
| 11 | Protocol integrity | – | Pre-registration before R2; new output dirs; no edits to frozen v1 files; one-shot T1; contamination pytest | 0 |
| 12 | Metric gaming | – | No WavLM-SV, ECAPA or Resemblyzer in training; no SV loss; ECAPA reported alongside WavLM-SV | 0 |
| 13 | Queue contention (one CRT slot is typical) | – | One script per stage, 48 h requests with resume, probes packed into J-probe, interpreter informed before each sbatch | – |
| 14 | Probe confounds | – | P0 already uses chapter-disjoint splits and a text control (`probe_speaker_info.py`); P0b controls text exactly; only excess over the control is interpreted | 0 |

Fixed process rules:
- pytest tests in `mvc/tests` for every change, run before submission;
- training tips and measured effects go to `docs/training_tips.md`;
- every cluster action and every good result is shared with the interpreter session;
- no frozen benchmark file is modified. A new system name lives only in the mvc harness and in `analyze_mvc.py`.