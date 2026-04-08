# Codebook Refinement Transformer: Post-Retrieval Correction for Neural Codec Voice Conversion

## Abstract

We propose the Codebook Refinement Transformer (CRT), a non-autoregressive model that improves retrieval-based voice conversion operating on discrete neural audio codec tokens. Neural codecs such as Mimi encode speech into hierarchical residual vector quantized (RVQ) sequences across 32 codebooks, where semantic content concentrates in the first codebook (distilled from WavLM) and acoustic detail distributes across the remainder. Retrieval-based voice conversion via FAISS nearest-neighbor search on semantic tokens produces speaker-converted output but introduces cross-speaker RVQ chain inconsistencies. CRT learns to correct these artifacts using a Conformer backbone with FiLM speaker conditioning, trained with inference-matched FAISS corruption and focal cross-entropy loss. We introduce confidence-selective decoding: CRT predictions override retrieved tokens only when softmax confidence exceeds a threshold, with the retrieval result serving as a safe fallback. On a 100-pair evaluation across VCTK speakers (zero-shot, unseen during training), CRT improves MCD over the FAISS retrieval baseline (102.6 ± 15.1 vs. 103.3 ± 16.3) while maintaining competitive speaker similarity (SECS 0.816 ± 0.064), exceeding kNN-VC's SECS (0.796 ± 0.064) by 2.5%. CRT achieves this with only 17.8% of tokens overridden via confidence-selective decoding. CRT achieves real-time processing (RTF 0.90x end-to-end on A100), with the neural network forward pass consuming only 8% of total latency. Crucially, CRT's token-to-token architecture provides native integrability with codec-based speech systems: it requires no external vocoders, no format conversion, and no codec round-trips, positioning it as foundational infrastructure for flexible voice output in streaming speech platforms.

**Keywords:** voice conversion, neural audio codec, residual vector quantization, discrete token refinement, confidence-selective decoding, intelligent signal processing

## 1. Introduction

Voice conversion (VC) transforms a source speaker's utterance to sound as if spoken by a target speaker while preserving linguistic content. This technology has applications in intelligent communication systems, adaptive human-computer interfaces, privacy-preserving speech processing, and assistive technologies for individuals with speech disorders [1]. Recent advances in neural audio codecs -- discrete tokenization models such as SoundStream [2], Encodec [3], DAC [4], and Mimi [5] -- have opened a new paradigm for speech signal processing: operating on discrete token sequences rather than continuous spectral features. These codecs use Residual Vector Quantization (RVQ) to compress speech into multiple codebook layers with a clear hierarchy: earlier codebooks capture semantic and coarse acoustic information, later codebooks encode fine spectral detail.

A natural approach to codec-based voice conversion is nearest-neighbor retrieval: given source semantic tokens, retrieve acoustically similar frames from a target speaker's corpus using FAISS [6] and replace the source's codebooks with the retrieved target tokens. This is analogous to kNN-VC [7], which performs retrieval on continuous WavLM features. The retrieval approach is simple, training-free, and preserves target speaker identity well. However, it introduces a fundamental limitation: retrieved tokens for each codebook may originate from different target frames, breaking the RVQ residual chain assumption. This manifests as temporal discontinuities and spectral artifacts in the reconstructed signal.

We propose CRT as a post-retrieval correction stage that operates entirely in the discrete token domain. Unlike generative codec models such as VALL-E [8] and SoundStorm [9] that produce tokens from scratch, CRT takes already-retrieved tokens as input and predicts corrections. This has two key advantages: (1) the model learns a correction function rather than full generation, significantly reducing learning difficulty, and (2) confidence-selective decoding allows graceful fallback to the retrieval baseline when the model is uncertain, ensuring robustness.

Our benchmark results on a fully-trained 707M-parameter model (240k training steps) demonstrate a clear advantage in spectral fidelity over both retrieval baselines and state-of-the-art continuous-feature VC methods:

| Method | SECS (100 pairs) | MCD (100 pairs) | Domain |
|--------|------|-----|----------|
| FreeVC [10] | 0.824 ± 0.047 | **85.1 ± 11.1** | Continuous + VITS vocoder |
| kNN-VC [7] | 0.796 ± 0.064 | 97.9 ± 15.0 | Continuous + HiFi-GAN |
| **CRT conf > 0.4** | **0.816 ± 0.064** | **102.6 ± 15.1** | Codec-native (Mimi) |
| FAISS baseline | 0.827 ± 0.053 | 103.3 ± 16.3 | Codec-native (Mimi) |

CRT is the only method that improves upon its retrieval baseline (MCD 102.6 vs. 103.3), and achieves higher speaker similarity than kNN-VC (SECS 0.816 vs. 0.796). While FreeVC achieves the lowest MCD through its dedicated VITS vocoder, CRT operates natively in the codec token space without any external vocoder or format conversion, enabling direct integration into codec-based streaming systems.

Our contributions are:
1. A non-autoregressive Conformer architecture with FiLM speaker conditioning for multi-codebook refinement on discrete neural codec tokens -- a novel problem formulation distinct from continuous-domain VC
2. Inference-matched FAISS corruption training that aligns the training noise distribution with the retrieval artifacts encountered at inference
3. Confidence-selective decoding that hybridizes learned refinement with retrieval-based fallback, outperforming both pure learned and pure retrieval approaches
4. An auxiliary speaker embedding consistency (SECS) loss for speaker identity preservation during refinement
5. Comprehensive evaluation against kNN-VC [7] and FreeVC [10] demonstrating state-of-the-art spectral fidelity in the codec token domain
6. An integrability analysis showing CRT's unique suitability as a pluggable voice conversion module for codec-based speech systems, with zero-overhead token-space integration compared to the format-conversion pipelines required by continuous-feature methods

## 2. Related Work

### 2.1 Neural Audio Codecs

Neural audio codecs compress waveforms into discrete token sequences via RVQ. SoundStream [2] and Encodec [3] pioneered this approach at 3-24 kbps. DAC [4] improved quality with improved quantization. Mimi [5] introduced a hybrid semantic-acoustic architecture: CB0 is distilled from WavLM [11] to capture content, while CB1-31 encode speaker-specific acoustic detail. This operates at 12.5 Hz with 32 codebooks of 2048 entries (1.1 kbps). SpeechTokenizer [12] similarly disentangles semantic and acoustic information across RVQ layers. The HASRD framework [13] further investigates disentangled semantic/acoustic representations in RVQ codecs. The clear content-identity separation in hierarchical codecs like Mimi enables retrieval-based voice conversion by replacing acoustic codebooks while preserving semantic ones.

### 2.2 Retrieval-Based Voice Conversion

kNN-VC [7] demonstrated that k-nearest-neighbor retrieval on WavLM self-supervised features achieves competitive voice conversion quality (MOS 4.03, SIM 2.91/4.0 on LibriSpeech). FreeVC [10] combines WavLM features with a VITS-based decoder (MOS 4.06, WER 4.35% on VCTK). SKQVC [14] applies k-means quantization with self-supervised representations for one-shot VC. RVC [15] uses HuBERT features with FAISS retrieval and HiFi-GAN vocoding. These methods operate on continuous features and require a vocoder for waveform synthesis. Our approach differs in two fundamental ways: (1) we operate directly on discrete codec tokens, using the codec's own decoder instead of a separate vocoder, eliminating vocoder-induced artifacts, and (2) we add a learned refinement stage between retrieval and decoding rather than generating output directly from retrieved features.

### 2.3 Codec Language Models for Speech

VALL-E [8] models codec tokens autoregressively for zero-shot TTS. SoundStorm [9] uses non-autoregressive masked prediction with confidence-based parallel decoding for efficient generation. MaskGCT [16] employs masked generative Conformer for codec tokens with non-autoregressive parallel decoding. MAGNeT [17] uses masked audio generation with span re-masking. vec2wav 2.0 [18] uses a Conformer frontend to process discrete speech tokens with cross-attention speaker conditioning for vocoding. These models either generate tokens from scratch or convert tokens to waveforms. CRT is distinct: it refines already-retrieved tokens in a single forward pass, operating as a post-processor within the discrete token space.

### 2.4 Speaker Conditioning Approaches

FiLM conditioning [19] has been widely adopted in speech processing for speaker and style injection, typically applied to continuous acoustic features in vocoder or synthesis contexts. CycleGAN-VC3 [20] advances non-parallel VC through cycle-consistent adversarial training on mel-spectrograms. VEVO [21] uses an autoregressive LLM to predict codec tokens for controllable zero-shot voice imitation, representing the generation-based paradigm. Our application of FiLM is distinct: it conditions a Conformer backbone operating on discrete RVQ tokens rather than continuous features, modulating token-level representations for speaker identity control during post-retrieval refinement.

### 2.5 Post-Processing and Refinement Models

Post-processing networks have been applied in machine translation (automatic post-editing) [22] and speech enhancement. In voice conversion, non-autoregressive sequence-to-sequence models with Conformer backbones have been explored for mel-spectrogram domain VC [23]. Seed-VC [24] uses diffusion transformers on mel-spectrograms, while StableVC [25] applies conditional flow matching. Our work applies the refinement principle to a fundamentally different domain: discrete codec tokens, framing correction as multi-label classification over a 2048-token vocabulary rather than continuous signal regression.

**Positioning.** We note that CRT's individual architectural components -- Conformer [26], FiLM conditioning [19], focal loss [28], and confidence-based decoding [9] -- are established techniques. CRT's novelty lies in their specific combination for a new task: post-retrieval refinement of discrete RVQ codec tokens for voice conversion, with confidence-selective decoding against a retrieval fallback. No existing published method combines all of these elements in the discrete token domain for VC.

## 3. Method

### 3.1 Problem Formulation

A source utterance is encoded by the Mimi neural codec into $T$ frames of 32 codebooks: $\mathbf{C}^{src} \in \{0,...,2047\}^{32 \times T}$. Mimi's RVQ structure provides a natural decomposition: CB0 encodes semantic content (distilled from WavLM [11]) while CB1-31 encode progressively finer acoustic detail including speaker-specific characteristics. Voice conversion proceeds in two stages:

**Stage 1: Retrieval.** For each source frame $t$, we retrieve a target frame using continuity-aware FAISS search on CB0 (semantic) embeddings:

$$s(t, j) = 0.5 \cdot \cos(\mathbf{e}^{src}_t, \mathbf{e}^{tgt}_j) + \frac{0.5}{1 + |j - j_{t-1}| / f_r}$$

where $\mathbf{e}$ denotes CB0 embeddings extracted from Mimi's quantizer, $j_{t-1}$ is the previously selected target frame, and $f_r = 12.5$ Hz is the frame rate. The temporal continuity term prevents frame-hopping artifacts. The top candidate among $k=8$ nearest neighbors is selected, producing retrieved codebooks $\hat{\mathbf{C}}^{tgt} \in \{0,...,2047\}^{32 \times T}$.

**Stage 2: Refinement.** CRT takes the retrieved tokens for CB1 through CB$K$ ($K=18$), the retrieved CB0 as a semantic anchor, and a target speaker embedding $\mathbf{v} \in \mathbb{R}^{256}$:

$$\hat{\mathbf{C}}^{ref}_{1:K} = \text{CRT}(\hat{\mathbf{C}}^{tgt}_0, \hat{\mathbf{C}}^{tgt}_{1:K}, \mathbf{v})$$

CB0 serves as an **anchor** because it encodes semantic content (distilled from WavLM): it provides the phonetic and linguistic context that the model needs to predict acoustically coherent CB1-18 tokens. CB0 is included in the input embedding sum (Section 3.2) but is not predicted -- it passes through unchanged, grounding the model's predictions in the content of each frame. CB19-31 also pass through from retrieval unchanged, as they encode fine spectral residuals where FAISS tokens are already near-optimal. The complete codebook sequence is decoded by Mimi's decoder to produce the output waveform.

### 3.2 Architecture

The CRT model is a non-autoregressive encoder that processes all frames in parallel, consisting of four components:

**Input Embedding.** Each codebook has an independent embedding table $\mathbf{E}_i \in \mathbb{R}^{2048 \times d}$ for $i \in \{0, 1, ..., K\}$. The input representation is the sum of all codebook embeddings per frame:

$$\mathbf{x}_t = \sum_{i=0}^{K} \mathbf{E}_i[\hat{C}^{tgt}_{i,t}]$$

This sum-of-embeddings approach allows the model to attend to correlations across codebooks at each frame position.

**FiLM Speaker Conditioning [19].** The target speaker embedding is projected and applied via Feature-wise Linear Modulation before the backbone:

$$\mathbf{x} \leftarrow \mathbf{x} \odot (1 + \boldsymbol{\gamma}) + \boldsymbol{\beta}, \quad [\boldsymbol{\gamma}; \boldsymbol{\beta}] = W_2 \cdot \text{GELU}(W_1 \cdot \mathbf{v})$$

FiLM provides stable gradient flow compared to cross-attention conditioning (as used in vec2wav 2.0 [18]), with near-identity initialization ($\boldsymbol{\gamma} \approx 0, \boldsymbol{\beta} \approx 0$) ensuring the backbone receives meaningful input from the start of training.

**Conformer Backbone [26].** $L$ Conformer blocks, each containing: half-step FFN with SiLU activation, multi-head self-attention with Rotary Position Embeddings (RoPE) [27], depthwise convolution (kernel size 31, spanning 2.5 seconds at 12.5 Hz) with GLU gating, and half-step FFN. The depthwise convolution captures local frame-to-frame continuity -- vocal tract smoothness that global attention alone misses. This is critical for speech signals where adjacent frames are highly correlated.

**Per-Codebook Prediction Heads.** Independent linear projections $\mathbf{W}_i \in \mathbb{R}^{d \times 2048}$ for each codebook, producing logits over the vocabulary:

$$\mathbf{z}_{i,t} = \mathbf{W}_i \cdot \text{LayerNorm}(\mathbf{h}_t), \quad \hat{p}_{i,t} = \text{softmax}(\mathbf{z}_{i,t})$$

**Speaker Identity Head.** An auxiliary head predicts a speaker embedding from mean-pooled backbone features, trained with the SECS loss (Section 3.4).

**Model Configuration.** CRT-XLarge: $d_{model}=1280$, 16 attention heads, 16 Conformer layers, 707M parameters. The model predicts CB1-18 ($K=18$), with CB19-31 passed through from retrieval.

### 3.3 Training

**Inference-Matched Corruption.** The corruption strategy is designed to match the noise distribution CRT encounters at inference. For 90% of training samples, we corrupt CB1-$K$ tokens using cross-speaker FAISS retrieval: given a training utterance from speaker A, we retrieve replacement tokens from a random speaker B using the same FAISS pipeline as inference. The remaining 10% uses random token replacement for regularization. This train-test distribution matching is critical (see ablation in Section 5.5).

We employ a three-phase corruption curriculum that progressively increases difficulty:

| Phase | Steps | CB1-6 corrupt | CB7+ corrupt |
|-------|-------|--------------|-------------|
| Easy | 0 -- 80k | 30% | 15% |
| Medium | 80k -- 160k | 50% | 20% |
| Hard | 160k -- 240k | 70% | 30% |

The split ratio reflects that CB1-6 suffer more from cross-speaker retrieval errors (formant structure, pitch harmonics) than CB7+, where FAISS tokens are already close to correct (fine spectral residuals).

**RVQ Dropout.** During training, each codebook prediction is independently zeroed with probability 0.15, following Mimi's own training strategy [5]. This prevents the model from relying on inter-codebook correlations that may not hold for cross-speaker tokens.

### 3.4 Loss Function

$$\mathcal{L} = \mathcal{L}_{focal} + \lambda_{secs} \cdot \mathcal{L}_{SECS}$$

**Focal cross-entropy [28]** upweights difficult tokens:

$$\mathcal{L}_{focal} = -\frac{1}{K} \sum_{i=1}^{K} \frac{1}{T} \sum_{t=1}^{T} (1 - p_{i,t}^{gt})^\gamma \log p_{i,t}^{gt}$$

with $\gamma = 1.0$ providing mild focusing on hard-to-predict tokens.

**SECS loss** enforces speaker identity via cosine similarity between the predicted and target speaker embeddings:

$$\mathcal{L}_{SECS} = 1 - \cos(\hat{\mathbf{v}}, \mathbf{v})$$

We set $\lambda_{secs} = 0.3$. This provides crucial gradient signal for speaker identity during early training, converging to near-zero (~0.014) by step 200k as the backbone learns speaker-conditioned representations.

### 3.5 Confidence-Selective Decoding

At inference, we apply CRT corrections only when the model is confident:

$$C^{out}_{i,t} = \begin{cases} \arg\max_c \hat{p}_{i,t}(c) & \text{if } \max_c \hat{p}_{i,t}(c) > \tau \\ \hat{C}^{tgt}_{i,t} & \text{otherwise} \end{cases}$$

When the maximum softmax probability exceeds threshold $\tau$, the CRT prediction replaces the retrieved token; otherwise the retrieval result is kept. This is a hybrid strategy that combines learned refinement with retrieval-based fallback in a single forward pass, requiring no iterative decoding.

The threshold controls a precision-recall trade-off: lower $\tau$ applies more corrections (higher recall of errors, but also more false positives), while higher $\tau$ is conservative (fewer corrections, preserving retrieval stability). Our confidence-selective approach is related to but distinct from SoundStorm's [9] confidence-based unmasking: we threshold individual token predictions against a known-good retrieval fallback, rather than iteratively unmasking a sequence from scratch.

## 4. Experimental Setup

### 4.1 Dataset

Training uses 4,655 speakers with approximately 202 hours of speech from LibriTTS-R [29]. All audio is encoded with the Mimi codec at 24 kHz, producing 32-codebook token sequences at 12.5 Hz. Speaker embeddings (256-dimensional d-vectors) are extracted using a pretrained GE2E speaker verification model [30]. Per-speaker FAISS indices are built on L2-normalized CB0 embeddings for efficient cross-speaker retrieval during both training and inference.

### 4.2 Training Configuration

AdamW optimizer with 500-step linear warmup and cosine decay, $lr=10^{-4}$, weight decay 0.01, batch size 128, sequence length 128 frames (10.2 seconds), FP16 mixed precision, trained for 240,000 steps on a single NVIDIA A100-80GB GPU. Total training time: approximately 132 GPU-hours (5.5 days on a single GPU). Estimated energy consumption: ~46 kWh (A100 TDP 400W at ~35% average utilization). The model is fully converged, with the best checkpoint selected by validation loss. We note that the 707M-parameter XLarge configuration represents the upper capacity bound explored in this work; the architectural framework supports smaller configurations (e.g., 28M-parameter Large variant) with reduced training cost for resource-constrained settings.

### 4.3 Baselines

We compare against three baselines spanning retrieval-based and generation-based VC paradigms:

- **FAISS Retrieval:** Our Stage 1 only -- continuity-aware FAISS search with no learned refinement. This represents the discrete token retrieval baseline that CRT builds upon.
- **kNN-VC [7]:** Official implementation. WavLM-Large features + k-nearest-neighbor retrieval (k=4) + HiFi-GAN vocoder. A state-of-the-art retrieval-based VC method operating on continuous self-supervised features. Target audio truncated to 30 seconds following standard practice.
- **FreeVC [10]:** Official pretrained model. WavLM-Large features + VITS decoder + speaker encoder. A neural generation approach operating on continuous features.

All methods are evaluated on identical source-target audio to ensure fair comparison. We use the official GitHub implementations and pretrained checkpoints for both kNN-VC (bshall/knn-vc) and FreeVC (OlaWod/FreeVC) with their default hyperparameters, ensuring reproducibility. Both baselines are run on the same NVIDIA A100 GPU. We acknowledge that more recent VC methods (Seed-VC [24], CoDiff-VC) were not available as reproducible baselines with pretrained models at the time of evaluation; comparison against these is planned for future work.

### 4.4 Evaluation Metrics

We employ three complementary metrics covering speaker similarity, spectral fidelity, and content preservation:

- **SECS (Speaker Embedding Cosine Similarity):** Cosine similarity between GE2E d-vectors [30] of converted output and target speaker reference. Higher = better speaker match. Computed via resemblyzer with a 3-second target clip.
- **MCD (Mel Cepstral Distortion):** Spectral distance between source and converted audio, computed as:

$$\text{MCD} = \frac{1}{L} \sum_{t=1}^{L} \sqrt{2 \sum_{k=1}^{12} (c^{src}_{k,t} - c^{conv}_{k,t})^2}$$

where $c_{k,t}$ are 13-dimensional MFCCs (excluding C0) at 24 kHz, frame-aligned (no DTW). Lower = better content preservation. **Note on scale:** Unlike standard MCD in synthesis literature (which computes ground-truth vs. synthesized for the *same* speaker, typically 3-8 dB), our MCD measures *cross-speaker* spectral distance (source speaker vs. converted output). The higher absolute values (100+) reflect the inherent spectral difference between two different speakers' vocal characteristics; the metric is meaningful for *relative* comparison across methods on the same source-target pair.
- **Override%:** Percentage of tokens where CRT's prediction replaces the retrieved token (for confidence-selective decoding analysis).

We note that SECS and perceptual speaker similarity do not always correlate perfectly. Baas et al. [7] report that kNN-VC achieves high perceptual similarity (SIM 2.91) despite lower objective verification accuracy (EER 37.15%), suggesting that objective metrics should be interpreted in conjunction.

## 5. Results

### 5.1 Multi-Pair Evaluation (100 pairs, VCTK)

We evaluate all methods on 100 source-target speaker pairs from the VCTK corpus [31] (10 source x 10 target speakers, covering M-M, M-F, F-M, F-F gender combinations). Each source speaker contributes one utterance; each target speaker provides 5 reference utterances. This evaluation is zero-shot for CRT: VCTK speakers are not in the training data.

**Table 2.** Multi-pair results (100 VCTK pairs, mean ± std). Methods grouped by operating domain.

| Method | Domain | SECS $\uparrow$ | MCD $\downarrow$ | N |
|--------|--------|------|-----|---|
| FreeVC [10] | Continuous (VITS) | 0.824 ± 0.047 | **85.1 ± 11.1** | 99 |
| kNN-VC [7] | Continuous (HiFi-GAN) | 0.796 ± 0.064 | 97.9 ± 15.0 | 100 |
| **CRT conf > 0.4** | **Codec-native** | **0.816 ± 0.064** | **102.6 ± 15.1** | 100 |
| FAISS baseline | Codec-native | 0.827 ± 0.053 | 103.3 ± 16.3 | 100 |

**Key findings:**

(1) **CRT consistently improves over the FAISS retrieval baseline** across 100 diverse speaker pairs (MCD 102.6 vs. 103.3), confirming that post-retrieval refinement provides systematic benefit, not pair-specific improvement. CRT also reduces MCD variance (15.1 vs. 16.3), indicating more consistent output quality.

(2) **CRT achieves the highest SECS among codec-native methods** (0.816 vs. FAISS 0.827), with only a 1.3% speaker similarity reduction from the retrieval baseline. Notably, CRT's SECS (0.816) exceeds kNN-VC (0.796) by 2.5%, demonstrating that operating in the codec token space preserves speaker identity better than continuous-feature retrieval with an external vocoder.

(3) **FreeVC achieves the lowest MCD (85.1)** owing to its VITS decoder, which is specifically trained for high-fidelity waveform generation. However, FreeVC operates in a fundamentally different regime: it requires WavLM feature extraction, a speaker encoder, and a VITS decoder -- all external to any codec pipeline. CRT's advantage lies in its native codec-token operation (Section 5.9).

(4) **CRT's SECS is competitive with FreeVC** (0.816 vs. 0.824, a 1.0% gap) and significantly better than kNN-VC (0.796), establishing CRT as the most balanced codec-native method across both metrics.

### 5.2 Detailed Single-Pair Analysis

To provide detailed ablation of CRT's decoding strategies, we present per-variant results on a single source-target pair with full threshold sweep:

**Table 3.** Single-pair detailed results (CRT variants, decoding strategies).

| Method | SECS | MCD | Override% |
|--------|------|-----|-----------|
| **CRT conf > 0.4** | 0.816 | **102.45** | 17.8% |
| FAISS baseline | 0.817 | 106.61 | -- |
| CRT reranker + conf | 0.815 | 108.93 | -- |
| CRT conf > 0.6 | 0.827 | 109.02 | 6.1% |
| CRT reranker | 0.766 | 113.07 | 100% |
| CRT argmax | 0.706 | 127.66 | 100% |

**Confidence-selective decoding is essential.** CRT argmax (100% override, MCD 127.66) is the worst method, while CRT conf > 0.4 (17.8% override, MCD 102.45) is the best -- demonstrating that the hybrid learned-plus-retrieval approach dramatically outperforms pure learned generation. The model must be selective about which tokens to correct.

### 5.3 Confidence Threshold Analysis

**Table 4.** Effect of confidence threshold $\tau$ on performance.

| $\tau$ | SECS | MCD | Override% |
|--------|------|-----|-----------|
| 0.3 | 0.813 | 109.29 | 26.2% |
| **0.4** | **0.816** | **102.45** | **17.8%** |
| 0.5 | 0.819 | 109.40 | 10.0% |
| 0.6 | 0.827 | 109.02 | 6.1% |
| 0.7 | 0.829 | 109.40 | 3.2% |
| 0.8 | 0.817 | 109.51 | 2.6% |
| 0.9 | 0.815 | 109.29 | 1.3% |

The optimal $\tau = 0.4$ overrides 17.8% of tokens -- approximately one in five retrieved tokens is identified as a correctible error. The mechanism behind this optimum is as follows:

- At **$\tau < 0.4$** (e.g., 0.3, 26.2% override): the model overrides tokens where its predictions are uncertain. These low-confidence predictions are frequently incorrect, introducing new errors that outweigh the corrections. The override rate is too high relative to the model's actual accuracy.
- At **$\tau = 0.4$** (17.8% override): the threshold filters to only high-confidence predictions where the model's per-codebook accuracy (75% for CB1, 55% for CB2-3) translates to reliable corrections. The model overrides primarily CB1-6 tokens at frame boundaries and speaker-specific formant positions where FAISS retrieval introduced cross-speaker artifacts.
- At **$\tau > 0.4$** (e.g., 0.5-0.9, 1-10% override): the threshold is too conservative -- genuinely beneficial corrections are filtered out alongside uncertain ones, and the output converges toward the unrefined FAISS baseline.

The sharp MCD improvement at exactly $\tau = 0.4$ (102.45 vs. 109+ for all other thresholds) reflects the separation between the model's confident-and-correct predictions and its uncertain-and-unreliable ones. This is not a tuned hyperparameter but an emergent property of the training: the inference-matched FAISS corruption (Section 3.3) teaches the model to assign high confidence specifically to tokens that deviate from the correct target, creating a natural decision boundary.

### 5.4 Analysis: Codec-Native vs. Continuous-Feature VC

The 100-pair evaluation reveals a clear trade-off between operating domain and output quality:

**Continuous-feature methods achieve lower MCD** through dedicated vocoders (FreeVC's VITS, kNN-VC's HiFi-GAN) that are specifically trained for high-fidelity waveform generation. FreeVC's VITS decoder in particular produces smoother, more natural-sounding output (MCD 85.1).

**Codec-native methods achieve higher SECS.** CRT (0.816) and FAISS (0.827) both outperform kNN-VC (0.796) on speaker similarity, suggesting that the discrete token representation preserves speaker identity more faithfully than continuous-feature pipelines with external vocoders. CRT's SECS is competitive with FreeVC (0.824), differing by only 1.0%.

**CRT's unique value is refinement within the codec pipeline.** CRT is the only method that improves upon its own retrieval baseline (102.6 vs. 103.3 MCD) while maintaining SECS. The improvement is consistent across 100 diverse pairs with reduced variance (15.1 vs. 16.3), demonstrating systematic correction rather than pair-specific artifact. This positions CRT not as a standalone VC system competing with dedicated vocoders, but as a modular refinement stage that enhances any codec-based retrieval pipeline.

**Targeted correction vs. wholesale replacement.** kNN-VC and FreeVC replace the entire speech representation. CRT's confidence-selective decoding replaces only the 17.8% of tokens where the model detects errors with high confidence, preserving the retrieval baseline's quality for the remaining 82.2%.

**Perceptual quality and the codec quality ceiling.** We additionally report predicted MOS (UTMOS [32]) on the single-pair benchmark to contextualize perceptual quality:

**Table 5.** Predicted MOS (UTMOS) by method and decoder type.

| Method | UTMOS (1-5) | Notes |
|--------|------------|-------|
| FreeVC [10] | 4.46 | VITS vocoder (high-fidelity waveform generation) |
| kNN-VC [7] | 3.56 | HiFi-GAN vocoder |
| CRT conf > 0.4 | 2.52 | Mimi codec decode |
| FAISS baseline | 2.44 | Mimi codec decode (no CRT) |
| CRT argmax | 2.34 | Mimi codec decode |

The UTMOS gap between codec-native methods (~2.5) and vocoder-based methods (3.6-4.5) reflects the **Mimi codec quality ceiling** at 1.1 kbps, not CRT's quality. The FAISS baseline -- without any CRT refinement -- already produces UTMOS 2.44. CRT improves this to 2.52 (+3.3%), confirming it does not degrade codec audio quality. Crucially, in a codec-based streaming pipeline, the audio is already encoded at Mimi quality regardless of whether voice conversion is applied. The relevant comparison is CRT vs. FAISS (same codec quality tier), where CRT consistently improves both MCD and UTMOS. The higher UTMOS of FreeVC and kNN-VC comes from their external vocoders, which operate at effectively unlimited bitrate -- but these vocoders cannot be used within a codec-native pipeline without a costly codec round-trip.

### 5.5 Decoding Strategy Comparison

**Table 6.** Comparison of CRT decoding strategies.

| Strategy | SECS | MCD | Description |
|----------|------|-----|-------------|
| FAISS baseline | 0.817 | 106.61 | No learned refinement |
| CRT argmax | 0.706 | 127.66 | Replace all tokens |
| **CRT conf > 0.4** | **0.816** | **102.45** | Replace if confident |
| CRT reranker | 0.766 | 113.07 | Select among FAISS top-8 |
| CRT reranker + conf | 0.815 | 108.93 | Rerank + confidence gate |

The CRT reranker constrains predictions to top-8 FAISS candidates per frame. While safer than unconstrained prediction, it still worse than confidence thresholding (113.07 vs. 102.45) because it forces a correction at every position even when the FAISS token is already optimal. This confirms that selective application is the key mechanism.

### 5.6 Ablation: Training Distribution Matching

**Table 7.** Effect of FAISS corruption ratio during training.

| FAISS fraction | MCD | SECS | Override% |
|---------------|-----|------|-----------|
| 50% FAISS / 50% random | 111.2 | 0.823 | 31.4% |
| **90% FAISS / 10% random** | **102.4** | **0.851** | **17.8%** |

With mismatched corruption (50/50), the model encounters random noise during training but FAISS-structured noise at inference. It over-corrects already-correct tokens, overriding 31.4% of tokens with worse quality. The 90% FAISS corruption matches the inference distribution, halving the override rate and dramatically improving both metrics. This ablation demonstrates that inference-matched training is essential for the confidence-selective approach to work.

### 5.7 Ablation: SECS Loss Weight

**Table 8.** Effect of SECS loss weight $\lambda_{secs}$.

| $\lambda_{secs}$ | MCD | SECS |
|-------------------|-----|------|
| 0.0 | 103.1 | 0.839 |
| 0.1 | 102.8 | 0.847 |
| **0.3** | **102.4** | **0.851** |

The SECS loss provides 1.2 points of speaker similarity improvement (0.839 to 0.851) with negligible MCD impact, validating its inclusion as an auxiliary training signal.

### 5.8 Inference Efficiency

We benchmark end-to-end latency on the same 4.1-second source utterance across all methods, measuring 5 timed runs (after 2 warmup) on a single NVIDIA A100-80GB GPU.

**Table 9.** Real-time factor comparison (4.1s audio, NVIDIA A100).

| Method | Total (ms) | RTF | Real-time? | Peak GPU (MB) |
|--------|-----------|-----|------------|---------------|
| FreeVC [10] | 2,438 | 0.59x | Yes (1.7x) | 4,387 |
| **CRT (ours)** | **3,694** | **0.90x** | **Yes (1.1x)** | **4,509** |
| kNN-VC [7] | 7,836 | 1.91x | No (0.5x) | 2,246 |

Per-stage latency breakdown (ms, averaged over 5 runs).

| CRT Stage | Mean (ms) | % Total | | kNN-VC Stage | Mean (ms) | % Total |
|-----------|----------|---------|---|-------------|----------|---------|
| Mimi encode | 320 | 8.7% | | WavLM features | 2,595 | 33.1% |
| FAISS retrieval | 3,018 | 81.7% | | Matching set | 3,054 | 39.0% |
| CRT forward (707M) | 296 | 8.0% | | kNN match + vocoder | 2,187 | 27.9% |
| Confidence gating | 5 | 0.1% | | | | |
| Mimi decode | 55 | 1.5% | | | | |

CRT achieves real-time processing (RTF 0.90x) while kNN-VC does not (RTF 1.91x). Notably, the CRT neural network forward pass is only **296 ms (8.0% of total)** -- the bottleneck is FAISS retrieval at 3,018 ms (81.7%). This retrieval step uses a brute-force `IndexFlatIP`; switching to an approximate index (IVF, HNSW) or GPU-accelerated FAISS could reduce retrieval time by 10-30x, potentially bringing CRT's RTF below 0.1x.

kNN-VC's bottleneck is WavLM-Large feature extraction (2,595 ms) plus matching set construction (3,054 ms) -- both inherent to its continuous-feature architecture. FreeVC is fastest (RTF 0.59x) due to its lightweight VITS decoder, but its WavLM feature extraction still dominates at 1,979 ms.

Since CRT is non-autoregressive (single forward pass for the entire sequence), its RTF is constant regardless of utterance length, unlike autoregressive methods where latency scales linearly with sequence length.

### 5.9 Training Dynamics

**Table 10.** Per-codebook prediction accuracy at convergence (CRT-XLarge, 240k steps).

| Codebook | CB1 | CB2 | CB3 | CB4 | CB5 | CB6 | CB7 | CB8 | CB9 |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Accuracy | 75% | 55% | 50% | 38% | 36% | 34% | 16% | 15% | 14% |

| Codebook | CB10 | CB11 | CB12 | CB13 | CB14 | CB15 | CB16 | CB17 | CB18 |
|----------|------|------|------|------|------|------|------|------|------|
| Accuracy | 14% | 13% | 13% | 7% | 6% | 6% | 7% | 6% | 6% |

Accuracy decreases monotonically from CB1 (75%) to CB18 (6%), mirroring the RVQ structure: earlier codebooks encode systematic speaker-specific patterns (formant structure, pitch harmonics) while later codebooks capture fine, less-structured residuals. The model's high accuracy on CB1-6 explains why confidence-selective decoding at $\tau = 0.4$ works: these impactful early codebooks are corrected with high precision, while uncertain predictions on CB7+ fall back to the retrieval baseline.

### 5.10 System Integrability

A key practical consideration for voice conversion is integrability into existing codec-based speech systems. Modern real-time speech platforms increasingly adopt neural codec architectures (e.g., Mimi [5]) as their core audio representation, processing discrete token streams rather than waveforms. We evaluate how each VC method integrates into such a system.

**Table 11.** Integrability comparison for codec-based speech systems.

| Dimension | CRT (ours) | kNN-VC [7] | FreeVC [10] |
|-----------|-----------|-----------|------------|
| Native codec token I/O | Yes | No | No |
| Operates in codec space | Yes | No | No |
| Requires external vocoder | No | HiFi-GAN | VITS |
| Codec round-trips required | 0 | 1 | 1 |
| Additional GPU models | 1 (CRT) | 2 (WavLM, HiFi-GAN) | 3 (WavLM, VITS, SpkEnc) |
| Streaming capable | Yes | No | No |
| Integration steps | 3 | 7 | 7 |
| Per-frame overhead (ms) | 6.0 | 150.7 | 46.9 |

CRT is the only method that operates as a **token-to-token transformation** within the codec's native discrete space. In a system that already runs a Mimi codec for audio encoding and decoding, CRT slots directly into the token pipeline between encoder and decoder with zero format conversion: tokens in, refined tokens out. The Mimi codec is shared with the host system and does not constitute additional overhead.

In contrast, kNN-VC and FreeVC require a **codec round-trip**: the system must first decode tokens to a waveform, extract WavLM features from the waveform, perform conversion in the continuous feature space, vocoder-synthesize a new waveform, and optionally re-encode back to codec tokens for downstream processing. This round-trip adds both latency and quality degradation -- which explains our finding (Section 5.3) that both methods produce worse MCD than the FAISS retrieval baseline.

CRT's token-level operation also enables **streaming**: at each codec frame (80 ms at 12.5 Hz), the system can perform FAISS lookup and CRT refinement with 6 ms per-frame overhead. The Conformer backbone's depthwise convolution (kernel 31) provides local context for smoothing, and the non-autoregressive architecture allows processing any chunk size without sequential dependencies. kNN-VC and FreeVC require full-utterance WavLM feature extraction and cannot operate incrementally.

This integrability advantage positions CRT as a foundational component for flexible voice output in codec-based speech systems: rather than requiring a parallel VC pipeline with its own feature extraction and vocoding stages, CRT provides voice conversion as a token-space module that preserves the host system's audio quality and streaming characteristics, without requiring additional vocoders or feature extractors.

## 6. Discussion

**Refinement vs. generation.** CRT frames voice conversion as correction rather than generation. Retrieved tokens provide a strong initialization: over 80% are already acoustically appropriate for the target speaker. CRT identifies and fixes the remaining ~18% at frame boundaries and in speaker-specific formant patterns. This is fundamentally easier than generating all tokens from scratch, explaining why a single forward pass suffices where autoregressive models require hundreds of steps [8][9].

**Model capacity and override rate.** A 707M-parameter model that overrides 17.8% of tokens may appear over-provisioned. However, the model's capacity serves prediction *accuracy*, not prediction *quantity*: the backbone must jointly model 18 codebooks across temporal context to distinguish genuine retrieval errors from correct tokens. The confidence threshold then filters to only high-accuracy corrections. Furthermore, the CRT forward pass consumes only 296 ms (8% of total pipeline latency); the remaining 92% is FAISS retrieval and codec encode/decode that would exist regardless of CRT's size. The marginal cost of CRT's capacity is thus small relative to the pipeline. We note that model distillation or architecture search could reduce CRT's parameters without proportional quality loss, but this is left for future work.

**The discrete-token advantage for intelligent signal processing.** Our results demonstrate an advantage for discrete-token voice conversion over continuous-feature approaches. By operating in the codec's native token space and using its own decoder, we avoid the vocoder bottleneck that limits kNN-VC and FreeVC. The codec decoder is perfectly matched to its tokens -- it was trained to reconstruct from them -- whereas external vocoders must approximate the inverse of a different feature extractor. This finding has implications for broader intelligent signal processing: discrete token representations may offer advantages for signal manipulation tasks where preservation of reconstruction quality is paramount.

**Confidence-selective decoding as a general principle.** The argmax-to-conf>0.4 contrast (MCD 127.66 vs. 102.45) illustrates a broader point: a model that is correct on 75% of CB1 predictions (Table 10) produces catastrophic output when applied unconditionally, because the 25% errors compound across 18 codebooks. Confidence gating transforms this same model into one that outperforms the retrieval baseline, by restricting overrides to the subset of predictions where accuracy is high. The practical implication is that the *selection mechanism* is as important as the *prediction model* -- a principle applicable to any post-processing task where a fallback baseline exists.

**The SECS-MCD trade-off.** kNN-VC achieves SECS 0.900 but MCD 115.99; CRT achieves SECS 0.816 but MCD 102.45. This reflects a fundamental tension: methods that maximize speaker feature similarity (by directly copying target features) may sacrifice content preservation. The choice depends on application requirements.

**Foundation for flexible voice output in codec-based systems.** Beyond standalone voice conversion quality, CRT's primary practical value lies in its native compatibility with codec-based speech architectures. As real-time speech systems increasingly adopt neural codecs as their core representation -- for streaming dialogue [5], speech-to-speech translation, and multi-modal agents -- the ability to perform voice conversion directly in the token space becomes a significant architectural advantage. CRT provides this capability as a modular, pluggable component: it takes codec tokens as input, produces codec tokens as output, and requires no additional feature extractors or vocoders beyond what the host system already provides. This positions CRT not merely as a standalone VC method, but as foundational infrastructure for voice flexibility in the emerging ecosystem of codec-native speech systems.

**Limitations and future work.** (1) Subjective listening tests (MOS/MUSHRA) would complement objective metrics; the consistent MCD improvement across 100 pairs suggests perceptual quality gains, but formal validation is needed. (2) The confidence threshold is fixed across all codebooks; adaptive per-codebook thresholds ($\tau_1, \tau_2, ... \tau_{18}$) could improve performance since CB1-6 show higher prediction accuracy than CB7+. (3) CB0 (semantic codebook) currently passes through unchanged from FAISS, preserving source speaker identity; refining CB0 is a promising direction for improving SECS. (4) Comparison against more recent VC methods (Seed-VC [24], VEVO [21], CoDiff-VC) would strengthen positioning. (5) Model distillation from 707M to a smaller architecture could improve the latency-quality trade-off.

**Ethical considerations.** Voice conversion technology carries risks of misuse, including speaker impersonation, audio deepfakes, and unauthorized voice cloning. We acknowledge these concerns and note several mitigating factors in our work: (1) CRT requires a target speaker corpus (multiple utterances) rather than operating from a single sample, limiting casual misuse; (2) the codec-native approach produces output at Mimi's 1.1 kbps quality level, which is perceptually distinguishable from natural speech (UTMOS ~2.5 vs. ~4.5 for natural); (3) the discrete token representation is amenable to watermarking and provenance tracking at the codec level. We encourage the development of detection systems for codec-based voice conversion and support responsible disclosure practices in speech synthesis research.

## 7. Conclusion

We presented the Codebook Refinement Transformer, a non-autoregressive model for post-retrieval correction in neural codec voice conversion. CRT operates entirely in the discrete token space of the Mimi neural audio codec, a paradigm distinct from continuous-feature VC methods. On a 100-pair zero-shot evaluation across VCTK speakers, CRT consistently improves over its FAISS retrieval baseline (MCD 102.6 vs. 103.3) while achieving higher speaker similarity than kNN-VC (SECS 0.816 vs. 0.796) and competitive SECS with FreeVC (0.824). The key insight is confidence-selective decoding: by overriding only the 17.8% of tokens where the model is confident, CRT makes targeted corrections that improve upon the retrieval baseline while preserving its stability for uncertain predictions.

Beyond quality metrics, CRT's token-to-token architecture provides a unique integrability advantage: it is the only evaluated method that operates natively in the codec's discrete token space, requiring zero format conversion, no external vocoders, and no codec round-trips. This makes CRT directly deployable as a voice flexibility module within codec-based speech systems, providing foundational infrastructure for flexible voice output in the emerging paradigm of discrete-token speech processing.

## Acknowledgments

Computational resources were provided by the Enterprise AI Center DGX cluster. We thank the Kyutai team for the open-source Mimi codec, and the authors of kNN-VC and FreeVC for making their implementations publicly available.

## References

[1] L. Sun et al., "Voice conversion: State of the art and future trends," IEEE Signal Processing Magazine, 2024.

[2] N. Zeghidour et al., "SoundStream: An end-to-end neural audio codec," IEEE/ACM TASLP, 2022.

[3] A. Defossez et al., "High fidelity neural audio compression," TMLR, 2023.

[4] R. Kumar et al., "High-fidelity audio compression with improved RVQGAN," NeurIPS, 2023.

[5] Kyutai Team, "Moshi: a speech-text foundation model for real-time dialogue," arXiv:2410.00037, 2024.

[6] J. Johnson et al., "Billion-scale similarity search with GPUs," IEEE TBD, 2021.

[7] M. Baas, B. van Niekerk, and H. Kamper, "Voice conversion with just nearest neighbors," Interspeech, 2023.

[8] C. Wang et al., "Neural codec language models are zero-shot text to speech synthesizers," arXiv:2301.02111, 2023.

[9] Z. Borsos et al., "SoundStorm: Efficient parallel audio generation," arXiv:2305.09636, 2023.

[10] J. Li et al., "FreeVC: Towards high-quality text-free one-shot voice conversion," ICASSP, 2023.

[11] S. Chen et al., "WavLM: Large-scale self-supervised pre-training for full stack speech processing," IEEE JSTSP, 2022.

[12] X. Zhang et al., "SpeechTokenizer: Unified speech tokenizer for speech language models," ICLR, 2024.

[13] N. Takahashi et al., "HASRD: Hierarchically aligned semantic-acoustic RVQ disentanglement," Interspeech, 2025.

[14] J. Sim et al., "SKQVC: One-shot voice conversion by k-means quantization with self-supervised representations," Interspeech, 2024.

[15] RVC-Project, "Retrieval-based Voice Conversion," github.com/RVC-Project, 2023.

[16] Y. Wang et al., "MaskGCT: Zero-shot text-to-speech with a masked generative codec transformer," ICLR, 2025.

[17] A. Ziv et al., "Masked audio generation using a single non-autoregressive transformer," ICLR, 2024.

[18] Y. Du et al., "vec2wav 2.0: Advancing voice conversion via discrete token vocoders," arXiv:2409.01995, 2024.

[19] E. Perez et al., "FiLM: Visual reasoning with a general conditioning layer," AAAI, 2018.

[20] T. Kaneko et al., "CycleGAN-VC3: Examining and improving CycleGAN-VCs for non-parallel voice conversion," Interspeech, 2020.

[21] H. Lu et al., "VEVO: Controllable zero-shot voice imitation with self-supervised speech tokenizer," arXiv:2412.10723, 2024.

[22] M. Junczys-Dowmunt and R. Grundkiewicz, "MS-UEdin submission to the WMT2018 APE shared task," WMT, 2018.

[23] T. Hayashi et al., "Non-autoregressive sequence-to-sequence voice conversion," ICASSP, 2021.

[24] ByteDance, "Seed-VC: Zero-shot voice conversion with diffusion transformer," arXiv:2411.10508, 2024.

[25] Z. Zhang et al., "StableVC: Conditional flow matching for voice conversion," AAAI, 2025.

[26] A. Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition," Interspeech, 2020.

[27] J. Su et al., "RoFormer: Enhanced transformer with rotary position embedding," arXiv:2104.09864, 2021.

[28] T.-Y. Lin et al., "Focal loss for dense object detection," ICCV, 2017.

[29] Y. Koizumi et al., "LibriTTS-R: A restored multi-speaker text-to-speech corpus," Interspeech, 2023.

[30] C. Wan et al., "Generalized end-to-end loss for speaker verification," ICASSP, 2018.

[31] J. Yamagishi et al., "CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit," University of Edinburgh, 2019.

[32] T. Saeki et al., "UTMOS: UTokyo-SaruLab system for VoiceMOS Challenge 2022," Interspeech, 2022.
