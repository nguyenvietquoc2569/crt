# Mimi-VC: Voice Conversion via Mimi Neural Audio Codec

**Authors:** Quoc V. Nguyen (UIT, VNU-HCM), Harry Nguyen (Kyndryl US), Lung Duc Vu (UIT, VNU-HCM)

## Overview

Mimi-VC exploits the disentangled codebook structure of Kyutai's [Mimi neural audio codec](https://huggingface.co/kyutai/mimi) to perform voice conversion. Mimi encodes audio into 32 codebooks at 12.5 Hz: CB0 carries semantic/content information (distilled from WavLM), while CB1-CB31 carry acoustic/speaker information. By preserving the source speaker's CB0 tokens and replacing the acoustic tokens (CB1+), we convert the voice while keeping the content intact.

### Mimi Codec Key Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 24,000 Hz |
| Frame rate | 12.5 Hz |
| Codebooks | 32 (CB0 = semantic, CB1-31 = acoustic) |
| Codebook size | 2,048 tokens each |
| Codebook dimension | 256 |
| Quantizer structure | `MimiSplitResidualVectorQuantizer` (semantic + acoustic RVQ) |

---

## Version History

### V0 -- FAISS+Smooth Retrieval Baseline

**Notebooks:** `mimi_vc_demo.ipynb`, `mimi_vc_lab.ipynb`

#### Motivation

Establish a training-free baseline for voice conversion using Mimi's codebook structure.

#### Approach

Three retrieval methods were explored, culminating in FAISS+Smooth:

1. **Naive swap**: Replace CB1-7 of source with arbitrary target tokens. Broke the RVQ residual chain -- each codebook encodes the residual error after the previous one, so misaligned replacements produced artifacts.

2. **Lookup table**: Find exact CB0 token matches between source and target. Limited by coverage -- many source CB0 tokens had no exact match in the target data.

3. **FAISS soft matching**: Build a FAISS index of target CB0 embeddings (256-dim), query each source frame by cosine similarity, take the nearest neighbor. Better coverage than exact matching, but frame-level independence caused temporal jitter.

4. **FAISS+Smooth** (best): Added temporal smoothing on top of FAISS. For each source frame, retrieve k=8 candidates, then score them by: `combined = alpha * semantic_similarity + (1-alpha) * temporal_continuity`, where temporal continuity penalizes jumping to distant positions in the target. This produced the smoothest and most natural-sounding conversions.

#### Architecture

No neural model. Pipeline:
- Encode source and target audio with Mimi
- Build FAISS `IndexFlatIP` on target CB0 embeddings (L2-normalized for cosine similarity)
- For each source frame: query k=8 neighbors, smooth-select based on combined semantic + temporal score
- Replace ALL source codebooks (CB0-CB31) with selected target frame's tokens
- Decode with Mimi

#### Results

- Best training-free method with SECS around 0.83-0.89 depending on source/target pairing
- Fast inference (real-time capable)
- Content preservation limited to phonetic contexts present in target data
- CB0 is also replaced, which theoretically loses some content fidelity

#### Problems

- Cannot generate novel acoustic patterns not present in target data
- Quality degrades when target speaker has limited data
- Replacing CB0 means content tokens come from the target, not source
- Frame-level retrieval has inherent granularity limitations

---

### V1 -- First Transformer Acoustic Translator

**Notebooks:** `mimi_vc_transformer_demo.ipynb`, `mimi_vc_lab.ipynb`

#### Motivation

Move beyond retrieval by learning a neural mapping from CB0 (content) to CB1-7 (acoustics), conditioned on target speaker data. This allows generating novel acoustic patterns and preserves source CB0 exactly.

#### Architecture: `AcousticTranslator`

| Component | Detail |
|-----------|--------|
| Input | CB0 tokens only |
| Embedding | `nn.Embedding(2048, 256)` + learned absolute positional embeddings (max_len=1024) |
| Encoder | `nn.TransformerEncoder`, 4 layers, 4 heads, d_model=256, ff_dim=1024 |
| Attention | Causal mask (autoregressive across time) |
| Output | 7 independent `nn.Linear(256, 2048)` heads, one per codebook |
| Parameters | ~5M |
| Loss | Equal weighting across all 7 codebooks |

#### Training

- Single target speaker only (no multi-speaker generalization)
- Self-supervised: the model learns target speaker's own CB0 -> CB1-7 mapping
- Adaptive steps: `min(500 * max(1, T_total / 500), 3000)`
- batch_size=8, seq_len=64, LR=3e-4, AdamW + CosineAnnealingLR

#### Results

- Successfully preserved source CB0 at 100% (vs retrieval which replaced it)
- Temperature sweep showed t=0.5-0.8 was the sweet spot
- Quality was variable and heavily dependent on target data quantity (30s = poor, 3+ min = best)

#### Problems Identified

1. **Independent CB prediction**: Each codebook was predicted independently, ignoring the RVQ residual structure. CB2 is the residual after CB1, so it should be conditioned on CB1's prediction.
2. **No source acoustic input**: Only CB0 was used. Source prosody, energy, and rhythm (partially encoded in source CB1-7) were completely ignored.
3. **Fixed positional embeddings**: Learned absolute positions don't generalize to sequence lengths unseen during training.
4. **Equal loss weighting**: CB1 is far more important than CB7 for perceptual quality, but all were weighted equally.
5. **Causal attention**: The model predicts all time steps at once (non-autoregressive across time), so masking future positions is unnecessary and reduces context.

---

### V2 -- Autoregressive Across Codebooks + Source Conditioning

**Notebooks:** `mimi_vc_transformer_v2_demo.ipynb`, `mimi_vc_analysis.ipynb`

#### Motivation

Address all five V1 problems: add codebook dependencies, source acoustic conditioning, RoPE, codebook-aware decoding, and exponential loss weighting.

#### Architecture: `AcousticTranslatorV2`

| Component | Detail |
|-----------|--------|
| Semantic Encoder | 4 layers, RoPE, causal self-attention |
| Source Acoustic Encoder | 2 layers, RoPE, **bidirectional** self-attention (can see full source) |
| Decoder | 4 layers, RoPE, causal self-attention + cross-attention to source acoustics |
| CB input | Source CB1-7 via per-CB embeddings (`nn.ModuleList` of 7 `nn.Embedding(2048, 256)`) |
| CB index | `nn.Embedding(n_cb+1, d_model)` -- decoder knows which CB it's predicting |
| Output | Single shared `nn.Linear(256, 2048)` across all codebooks |
| d_model | 256, n_heads=4, ff_dim=1024, dropout=0.1 |
| Loss weighting | Exponential: `cb_weights = [2^(-i/2)]` for i=0..6, CB1 weighted ~4x more than CB7 |
| Decoding | Autoregressive across codebooks: CB1 -> CB2 -> ... -> CB7 sequentially |
| Training | Teacher forcing (uses ground truth previous CB to predict next CB) |

#### Key Changes from V1

| Issue | V1 | V2 |
|-------|-----|-----|
| CB dependencies | Independent (7 parallel heads) | **Autoregressive** (CB1->CB2->...->CB7) |
| Source acoustics | CB0 only | **Cross-attention to source CB1-7** |
| Positional encoding | Learned absolute (max_len=1024) | **RoPE** (length-generalizable) |
| CB awareness | None | **Codebook index embedding** |
| Loss weighting | Equal | **Exponential** (CB1 >> CB7) |

#### Results

- Hybrid mode (V2 for CB1-3, retrieval for CB4-7) showed promise
- Source acoustic conditioning preserved some prosodic characteristics
- Per-codebook accuracy tracked during training

#### Problems Identified

1. **Mode collapse**: Analysis notebook revealed the model collapsed to producing a narrow set of tokens. Predicted tokens had ~38 unique values and entropy ~4.85, vs ground truth ~160 unique tokens and entropy ~7.1.

2. **Error cascade (exposure bias)**: During training, teacher forcing used ground truth CB(i-1) to predict CB(i). At inference, the model used its own (potentially wrong) CB(i-1) predictions, causing errors to compound. CB1 accuracy might be 70%, but by CB7 errors had cascaded to >93% mismatch.

3. **Still single-speaker**: Each model was trained from scratch on one target speaker, with no pre-training or multi-speaker generalization.

4. **Causal attention across time**: The model predicted all time steps in parallel (non-AR across time), making the causal mask across time unnecessary and limiting context.

#### Diagnosis (from `mimi_vc_analysis.ipynb`)

The analysis notebook implemented a comprehensive diagnostic framework:
- **Exposure bias detection**: Compared teacher-forced accuracy vs free-generation accuracy. Large gap = error cascade.
- **Mode collapse detection**: If free-gen entropy << ground truth entropy, the model is collapsing.
- **Confidence analysis**: Very high prediction confidence = overfitting to narrow patterns.
- **Decision tree**: Low entropy -> MODE COLLAPSE; High teacher-forced but low free-gen -> EXPOSURE BIAS; Self-recon SECS < 0.7 -> FUNDAMENTAL LEARNING FAILURE.

---

### V3a -- Multi-Speaker Pre-training (Cross-Speaker)

**Notebook:** `mimi_vc_multispeaker.ipynb` (first iteration)

#### Motivation

Single-speaker training severely limits the model. Pre-training on many speakers should teach a general "content -> acoustics" mapping, which can then be fine-tuned on a specific target.

The V2 AR-across-codebooks approach was abandoned because:
- Error cascade was fundamentally hard to solve without scheduled sampling
- CB1-6 can be predicted independently if the model is powerful enough (non-AR across codebooks)
- Parallel prediction is faster

#### Architecture: `AcousticTranslatorV3a`

| Component | Detail |
|-----------|--------|
| Source acoustic encoder | 2 layers (encodes source CB1-6) |
| Semantic encoder | 4 layers with RoPE |
| Decoder | 4 layers with RoPE, causal attention |
| Speaker conditioning | CB0 average embedding (weak signal) |
| Output | 6 independent heads (CB1-CB6) |
| d_model | 256, ~15.9M params |

#### Training: Cross-Speaker Paradigm

- 40 speakers from LibriSpeech train-clean-100
- **Cross-speaker pairs**: Take CB0 from speaker A, target CB1-6 from speaker B
- Rationale: since CB0 is speaker-independent, the model should learn to map any CB0 to any speaker's acoustic tokens, conditioned on speaker embedding
- 5,000 steps, batch_size=8, seq_len=64

#### Results

- **Mode collapse persisted**: 38 unique tokens, entropy ~4.85 -- identical to V2
- SECS: V3a ~0.77-0.80 vs FAISS ~0.83

#### Root Cause Analysis

The cross-speaker training paradigm was fundamentally broken:
- ~70% of training batches paired CB0 from utterance A (speaker A) with CB1-6 from utterance B (speaker B)
- These utterances contained **completely different content** (different words, phonemes, durations)
- The model could not learn a content->acoustics mapping from mismatched content pairs -- it was pure noise
- Only ~30% of batches were self-reconstruction (same utterance), providing too little useful signal
- The model defaulted to predicting a "safe" average distribution -> mode collapse

#### Lesson Learned

CB0 being "speaker-independent" does NOT mean any CB0 can be randomly paired with any speaker's CB1-6. The CB0 and CB1-6 must come from the **same utterance** because CB1-6 encode the acoustic details **of that specific content**.

---

### V3b -- Self-Reconstruction Across Speakers

**Notebook:** `mimi_vc_multispeaker.ipynb` (second iteration)

#### Motivation

Fix the cross-speaker training failure by switching to 100% self-reconstruction. Each training sample uses the same utterance's CB0 and CB1-6, but from different speakers. The speaker embedding tells the model WHOSE acoustics to produce.

At inference, we simply swap the speaker embedding to the target speaker -> the model generates target-speaker-style acoustic tokens for the source content.

#### Architecture: `AcousticTranslatorV3b`

Major simplification and scale-up from V3a:

| Component | V3a | V3b |
|-----------|-----|-----|
| Source acoustic encoder | 2-layer encoder | **Removed** (unnecessary -- only CB0 + speaker_emb needed) |
| Model size | d_model=256, ~15.9M params | **d_model=512, ~50M params** |
| Encoder | 4 layers | **6 layers** |
| Decoder | 4 layers | **6 layers** |
| Heads | 4 | **8** |
| Speaker embeddings | CB0 average (weak) | **Resemblyzer d-vectors** (256-dim, proven speaker encoder) |
| Speaker conditioning | Additive | **FiLM** (Feature-wise Linear Modulation: scale + shift) |
| Training steps | 5,000 | **30,000** (with warmup + cosine decay) |
| Sequence length | 64 | **128** |
| Batch size | 8 | **24** |

#### FiLM Conditioning

Instead of simply adding the speaker embedding to the representation, FiLM applies an affine transformation:
```
gamma, beta = project(speaker_embedding)
output = input * (1 + gamma) + beta
```
This is more expressive: the speaker embedding can both scale and shift the features, allowing it to modulate different dimensions independently.

#### Performance Optimizations

Three levels of parallelization were added for Colab training speed:

1. **Parallel CB forward pass**: Instead of running the decoder 6 times (once per codebook), batch all 6 codebooks together by expanding `[B, T, D]` to `[B*6, T, D]` and running a single decoder pass. ~5x speedup.

2. **AMP (Automatic Mixed Precision)**: float16 training with `torch.amp.GradScaler`. ~2x throughput.

3. **torch.compile**: Fused CUDA kernels for additional ~1.3x speedup.

#### Google Drive Checkpointing

- All checkpoints saved to `/content/drive/MyDrive/mimi_vc_checkpoints/`
- Periodic saves every 5,000 steps + best model tracking
- Auto-resume from `v3b_pretrain_latest.pt` on Colab reconnection
- Handles `torch.compile` `_orig_mod` prefix in state dict keys

#### Results

- **Pre-training achieved 99% accuracy** on self-reconstruction (CB1-6 all 98-99%)
- **But inference STILL showed mode collapse**: 39 unique tokens, entropy ~4.91
- SECS: V3b ~0.77-0.80 vs FAISS ~0.84

#### Root Cause Analysis

99% accuracy was a **red flag**, not a success:

- The model **memorized** speaker-specific CB0 -> CB1-6 mappings
- For each speaker, given their CB0 tokens + their exact speaker embedding, the model learned to look up the exact CB1-6 tokens from training data
- This is essentially a learned lookup table per speaker, not a generalizable mapping
- At inference, when source CB0 comes from a **different speaker** (not seen during training with the target's speaker embedding), the CB0 tokens are **out-of-distribution** for that speaker embedding
- The model has never seen "these CB0 tokens + this speaker embedding" together -> falls back to a collapsed average distribution

The fundamental issue: without any noise or regularization, the model found a shortcut (memorization) instead of learning the intended generalizable function.

---

### V3c -- Bidirectional + Regularized

**Notebook:** `mimi_vc_multispeaker.ipynb` (previous iteration)

#### Motivation

Force the model to learn a generalizable mapping by making memorization impossible. If the exact CB0 tokens and exact speaker embedding are never reliable (due to noise), the model must learn higher-level patterns.

Also fix the unnecessary causal attention mask -- since the model predicts all time steps in parallel, every position should see the full sequence context.

#### Architecture: `AcousticTranslatorV3c`

| Component | Detail |
|-----------|--------|
| Semantic Encoder | 6 layers, RoPE, **bidirectional** self-attention |
| Decoder | 6 layers, RoPE, **bidirectional** self-attention + FiLM speaker conditioning |
| CB0 noise | 15% random token replacement during training |
| Speaker embedding noise | Gaussian noise, std=0.2, during training |
| Label smoothing | 0.1 in cross-entropy loss |
| Dropout | 0.15 (up from 0.1) |
| d_model=512, n_heads=8, n_cb=6 | Same as V3b |
| Parallel CB | All 6 codebooks batched in one decoder pass (same as V3b) |
| AMP + torch.compile | Same as V3b |

#### Five Specific Fixes

**1. Bidirectional Attention**

V3b used a causal mask (each position only attends to past positions). But the model is non-autoregressive across time -- it predicts all T positions simultaneously. Causal masking was pointless and actively harmful: position t=0 could only see itself, position t=1 could see t=0 and t=1, etc. With bidirectional attention, every position sees the full CB0 sequence, giving the model maximum context for each prediction.

**2. CB0 Token Noise (15% random replacement)**

During training, 15% of CB0 input tokens are randomly replaced with random tokens from the vocabulary. This prevents the model from memorizing exact "CB0 token X at position t for speaker S -> CB1 token Y" mappings. The model must rely on surrounding context and higher-level patterns rather than individual token identity.

Implementation:
```python
def _augment_cb0(self, sem_tokens):
    if not self.training: return sem_tokens
    mask = torch.rand_like(sem_tokens.float()) < 0.15
    noise = torch.randint_like(sem_tokens, 0, 2048)
    return torch.where(mask, noise, sem_tokens)
```

**3. Speaker Embedding Noise (Gaussian, std=0.2)**

During training, Gaussian noise is added to the speaker embedding. This prevents the model from using the exact speaker embedding as a lookup key for memorized mappings. The model must learn to handle a range of similar speaker embeddings, forcing it to learn the underlying speaker-to-acoustics relationship rather than a per-speaker lookup table.

Implementation:
```python
def _augment_spk(self, spk_emb):
    if not self.training: return spk_emb
    return spk_emb + torch.randn_like(spk_emb) * 0.2
```

**4. Label Smoothing (0.1)**

Standard cross-entropy pushes the model toward 100% confidence on one token and 0% on all others. Label smoothing distributes 10% of the probability mass uniformly across the vocabulary. This prevents the overconfident, peaked distributions that characterize mode collapse -- even the "correct" output should not have 99.9% confidence.

**5. Higher Dropout (0.15)**

Increased from 0.1 to 0.15 for additional regularization, making it harder for the model to rely on exact input patterns.

#### Training Setup

**Pre-training (30,000 steps):**
- 40 speakers from LibriSpeech train-clean-100, 20 utterances each
- Self-reconstruction: CB0 + speaker_emb -> CB1-6 from the SAME utterance
- batch_size=24, seq_len=128, LR=2e-4
- 1,000-step warmup + cosine decay
- Exponential loss weighting: `cb_weights = [2^(-i/2)]` (CB1 weighted most)
- Checkpoints every 5,000 steps to Google Drive, with best-model tracking
- Resume from `v3c_pretrain_latest.pt`

**Fine-tuning (3,000 steps):**
- On uploaded target speaker audio
- LR=5e-5, batch_size=16, CosineAnnealingLR
- **Augmentation kept ON**: CB0 noise + speaker embedding noise + label smoothing active during fine-tuning to prevent overfitting to target speaker's specific CB0 patterns

#### Expected Outcomes

- Training accuracy should be **70-85%** (NOT 99%)
  - 99% was memorization; lower accuracy with noise = the model is actually learning generalizable patterns
- Unique tokens at inference should be **closer to 160** (vs 38-39 in all prior versions)
- Entropy should be **closer to 7.1** (vs ~4.9 in prior versions)
- The key indicator: does the model produce **diverse, content-dependent** acoustic tokens, or does it collapse to a small fixed set?

#### Actual Results

- Pre-training accuracy: CB1=97.6%, CB2=97.0%, CB3=94.9% -- **still too high** despite regularization
- Inference: **40 unique tokens, entropy ~4.89** -- identical to V3b. Mode collapse unchanged.
- SECS: V3c t=0.5 = 0.8356 (matched FAISS 0.835), but token diversity was still collapsed
- The 15% CB0 noise was too weak: bidirectional attention reconstructed corrupted tokens from neighbors
- Speaker noise std=0.2 was too small relative to inter-speaker similarity (mean=0.646)
- **Conclusion:** Label smoothing and noise on discrete cross-entropy cannot fix mode collapse. The problem is structural: cross-entropy on 2048 classes inherently produces peaked distributions.

---

### V3d -- Continuous Embedding Prediction

**Notebook:** `mimi_vc_multispeaker.ipynb` (previous iteration)

#### Motivation

V3c proved that regularization (noise, label smoothing) cannot fix mode collapse caused by discrete token classification. The cross-entropy loss on 2048 classes inherently pushes toward peaked distributions -- the model converges to predicting a small set of "safe" tokens regardless of noise.

The key insight: V3c achieved SECS=0.836 matching FAISS=0.835. Speaker identity IS learned. Only token diversity is collapsed. The fix must target the output representation, not the regularization.

#### Architecture: `AcousticTranslatorV3d`

Same encoder/decoder backbone as V3c, but fundamentally different output and loss:

| Component | V3c | V3d |
|-----------|-----|-----|
| Output heads | `nn.Linear(512, 2048)` -- 2048-class logits | `nn.Linear(512, CB_DIM)` -- embedding vectors |
| Loss | Cross-entropy on token IDs | Cosine similarity to codebook vectors |
| Inference | Sample from softmax logits | Nearest neighbor in codebook |
| Diversity | Label smoothing (ineffective) | Entropy regularization on codebook usage |
| CB0 augmentation | 15% random token noise | 30% span masking (contiguous chunks) |
| Speaker noise | std=0.2 | std=0.3 |

#### Key Innovation: Embedding Prediction

Instead of classifying among 2048 discrete tokens, the model predicts a continuous vector in CB_DIM-dimensional embedding space:

1. **Training**: For each time step, the model predicts a CB_DIM-dim vector. The loss is `1 - cosine_similarity(predicted, target_codebook_vector)`.

2. **Inference**: The predicted vector is compared to all codebook entries via cosine similarity. Temperature scaling and top-k filtering are applied to the similarity scores, then a token is sampled.

3. **Why it prevents mode collapse**: In continuous space, small variations in the predicted vector naturally select different codebook entries. There's no "peaked distribution" problem because the output isn't a probability over classes.

#### Diversity Regularization

An entropy term maximizes the diversity of codebook usage across the batch:
- Compute soft assignments: distance from each predicted vector to all codebook entries
- Average the soft assignments across the batch
- Maximize the entropy of this average distribution (penalize concentration on few entries)

#### Stronger CB0 Augmentation: Span Masking

V3c's 15% random token replacement was too easy to reconstruct with bidirectional attention. V3d uses **span masking**: corrupt ~30% of CB0 tokens in contiguous chunks of 3-10 tokens. This is much harder to fill in from context, forcing the model to rely on broader patterns rather than exact token-level mappings.

#### Training Setup

Same as V3c (30k pre-training steps, 3k fine-tuning steps, same optimizer/scheduler), but with:
- Cosine embedding loss instead of cross-entropy
- Diversity regularization weight = 0.1
- CB0 span masking rate = 30%
- Speaker noise std = 0.3
- Checkpoints saved as `v3d_pretrain_*.pt` / `v3d_finetuned.pt`

#### Actual Results

- Pre-training NN-match accuracy: CB1=23%, CB2=66%, CB3=56%, CB4=42%, CB5=29%, CB6=18%
- Inference: **47 unique tokens, entropy ~5.5** -- slightly better than V3c (40, 4.89) but far from GT (160, 7.1)
- **SECS dropped**: V3d t=0.5 = 0.779 vs V3c = 0.836 vs FAISS = 0.819
- **MCD increased**: V3d = 154 vs FAISS = 113 -- much more distortion
- 100% change rate on CB1-6 -- model changes every token but most are **wrong**

#### Root Cause Analysis

Cosine embedding loss was **too weak** for discrete token prediction:
- Cross-entropy gave 95% accuracy (accurate but collapsed). Cosine gave 23% accuracy (inaccurate AND somewhat collapsed).
- The embedding space is too smooth -- the gradient signal doesn't strongly penalize being "close but wrong" to a codebook entry.
- Slight diversity improvement (47 vs 40 tokens) came at a massive quality cost.
- **Verdict:** Changing the loss function doesn't fix the core problem.

#### Deeper Insight: The Real Root Cause

Looking across V3a-V3d, the persistent failure has a deeper cause than loss function choice:

**The model is ONLY trained on self-reconstruction** (same speaker in = same speaker out). At inference with a DIFFERENT target speaker, the model encounters a distribution shift it has never seen. All regularization attempts (noise, masking, label smoothing, embedding prediction) are weak proxies for actual cross-speaker training.

FAISS works because it directly maps into the target speaker's **real** token space. The model has never seen what cross-speaker conversion actually looks like during training.

---

### V3e -- Cross-Speaker Training with FAISS Pseudo-Labels (Current Version)

**Notebook:** `mimi_vc_multispeaker.ipynb` (current)

#### Motivation

All V3a-V3d versions shared a fundamental flaw: the model only saw self-reconstruction examples during training. At inference, cross-speaker conversion (different speaker's CB0 + target speaker embedding) is an entirely unseen scenario. No amount of regularization can fix this distribution shift.

The solution: give the model actual cross-speaker training examples by using FAISS retrieval as a teacher.

#### Architecture: `AcousticTranslatorV3e`

Same backbone as V3c/V3d. The architectural change is back to cross-entropy:

| Component | V3d | V3e |
|-----------|-----|-----|
| Output heads | `nn.Linear(512, CB_DIM)` embedding vectors | `nn.Linear(512, 2048)` logits -- **back to cross-entropy** |
| Loss | Cosine similarity (weak gradients) | **Cross-entropy + explicit diversity regularization** |
| Training data | Self-reconstruction only | **50% self-recon + 50% cross-speaker (FAISS pseudo-labels)** |
| Diversity | Entropy on soft codebook assignment | **Entropy on batch output distribution** |
| CB0 augmentation | 30% span masking | 30% span masking (kept) |
| Speaker noise | std=0.3 | std=0.3 (kept) |
| Label smoothing | None (V3d used cosine loss) | **None** (diversity loss replaces it) |

#### Key Innovation: Cross-Speaker FAISS Distillation

During training, 50% of each batch consists of cross-speaker examples:

1. Sample source utterance from speaker A (get CB0 tokens)
2. Pick a random different speaker B
3. Use FAISS+Smooth retrieval to map speaker A's CB0 into speaker B's codebook space
4. The retrieved CB1-6 tokens from speaker B become the pseudo ground truth
5. Train: CB0_A + embedding_B -> pseudo_CB1-6_B (cross-entropy)

This teaches the model what cross-speaker conversion looks like. The FAISS pseudo-labels are not perfect, but they provide a reasonable approximation of the true cross-speaker mapping.

#### Per-Speaker FAISS Indices

FAISS indices are built for all 40 training speakers during data preparation:
- Each speaker's CB0 tokens are embedded using the codebook embedding table
- A `faiss.IndexFlatIP` index is built per speaker on L2-normalized CB0 embeddings
- At training time, `faiss_retrieve_tokens(src_cb0, tgt_spk_id)` performs K=8 FAISS search with temporal smoothing (alpha=0.5) into any speaker's index

#### Explicit Diversity Regularization

Instead of label smoothing (V3c) or embedding entropy (V3d), V3e uses direct batch-level entropy:

```
avg_probs = softmax(logits, dim=-1).mean(dim=batch)  # [vocab_size]
entropy = -sum(avg_probs * log(avg_probs))
div_loss = 1.0 - (entropy / max_entropy)  # 0=diverse, 1=collapsed
```

This directly measures whether the model is using diverse tokens across the batch. Unlike label smoothing, this is a proper objective that the optimizer can actively push toward.

Weight: `diversity_weight=0.3` (much stronger than V3d's 0.1).

#### Training Setup

**Pre-training (30,000 steps):**
- 50% self-reconstruction + 50% cross-speaker (FAISS pseudo-labels)
- batch_size=24, seq_len=128, LR=2e-4
- Cross-entropy + diversity loss (weight=0.3)
- CB0 span masking 30%, speaker noise std=0.3
- Checkpoints: `v3e_pretrain_*.pt`

**Fine-tuning (3,000 steps):**
- Self-reconstruction on target speaker only (model already knows cross-speaker conversion)
- LR=5e-5, batch_size=16
- Same loss (cross-entropy + diversity)
- Checkpoints: `v3e_finetuned.pt`

#### Expected Outcomes

- Training accuracy: **60-80%** (lower than V3c's 95% due to harder cross-speaker examples)
- Unique tokens: **80-160** (model has seen diverse cross-speaker outputs)
- Entropy: **6-7** (closer to GT 7.1)
- SECS: **>0.80** (should match or beat FAISS since model learned from FAISS)
- MCD: **<130** (should improve over V3d's 154)
- The key test: does the model generalize cross-speaker conversion to unseen target speakers?

---

## Summary of Evolution

```
V0 (Retrieval)     -> Training-free FAISS+Smooth baseline
                      SECS ~0.83-0.89, but limited to existing target phonemes

V1 (Transformer)   -> First neural approach, 7 independent output heads
                      Problem: independent CBs, no source acoustics, fixed positions

V2 (AR + Source)    -> Autoregressive across CBs, source cross-attention, RoPE
                      Problem: error cascade (exposure bias), mode collapse (38 unique tokens)

V3a (Multi-speaker) -> Cross-speaker training on 40 speakers
                      Problem: mismatched content = noise, mode collapse persisted

V3b (Self-recon)    -> Self-reconstruction, bigger model (50M), FiLM, d-vectors
                      Problem: 99% accuracy = memorization, mode collapse at inference

V3c (Regularized)   -> Bidirectional attention, CB0 noise, speaker noise, label smoothing
                      Result: 95-98% accuracy, STILL 40 unique tokens. Regularization insufficient.

V3d (Embedding)     -> Predict codebook embeddings (not token IDs), cosine loss, diversity reg
                      Result: 23% accuracy, 47 tokens. Weaker loss = worse quality. Wrong approach.

V3e (Cross-speaker) -> Cross-entropy + diversity loss + 50% cross-speaker FAISS distillation
                      Fix: model finally sees cross-speaker examples during training.
```

## File Index

| File | Description |
|------|-------------|
| `mimi_vc_demo.ipynb` | V0 -- FAISS+Smooth retrieval baseline demo |
| `mimi_vc_lab.ipynb` | V0 -- Research lab notebook (all 3 approaches) |
| `mimi_vc_transformer_demo.ipynb` | V1 -- Transformer acoustic translator demo |
| `mimi_vc_transformer_v2_demo.ipynb` | V2 -- AR across codebooks + source conditioning |
| `mimi_vc_analysis.ipynb` | Diagnostic tool (V2-based) -- mode collapse/exposure bias detection |
| `mimi_vc_experiments.ipynb` | Systematic evaluation for paper (SECS, CER, MCD, PESQ, RTF) |
| `mimi_vc_multispeaker.ipynb` | V3e -- Multi-speaker with cross-speaker FAISS training (latest) |
| `generate_paper.js` | Paper generation script |
| `mimi_vc_paper.docx` | Paper draft |

## Metrics

| Metric | Description | Tool |
|--------|-------------|------|
| SECS | Speaker Embedding Cosine Similarity (higher = more like target) | Resemblyzer |
| CER | Character Error Rate (lower = better content preservation) | Whisper ASR |
| MCD | Mel Cepstral Distortion (lower = less distortion) | librosa MFCC |
| PESQ | Perceptual Evaluation of Speech Quality | pesq library |
| RTF | Real-Time Factor (lower = faster) | wall-clock timing |
| Unique tokens | Number of distinct tokens predicted per codebook | Counter |
| Entropy | Shannon entropy of predicted token distribution (bits) | numpy |
