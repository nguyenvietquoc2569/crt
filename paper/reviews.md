# Peer Reviews: "Codebook Refinement Transformer: Non-Autoregressive Post-Retrieval Correction for Neural Codec Voice Conversion"

---

## Review 1: Strict Reviewer (Reviewer A)

**Overall Score: 4/10 (Reject)**

**Summary:** The paper proposes CRT, a post-processing Conformer model that refines FAISS-retrieved neural codec tokens for voice conversion. While the engineering is competent, the paper has significant weaknesses in novelty, evaluation, and positioning relative to existing work.

### Strengths
1. The confidence-selective decoding scheme is a practical contribution. The insight that selective override outperforms full override is useful and well-demonstrated.
2. The curriculum corruption strategy matching training distribution to inference distribution is a sensible design choice, well-motivated by the ablation.
3. The paper is clearly written and the method is reproducible.

### Weaknesses

**W1: Limited novelty.** The core idea -- using a neural network to post-process retrieval results -- is well-established in NLP (automatic post-editing in machine translation, e.g., Junczys-Dowmunt & Grundkiewicz, 2018, which the authors cite). Applying this to codec tokens is a straightforward extension. The Conformer backbone, FiLM conditioning, focal loss, and RoPE are all off-the-shelf components assembled without architectural innovation.

**W2: Insufficient evaluation.** This is the most critical issue:
- Evaluation on a **single source-target speaker pair** is unacceptable for a VC paper. Standard VC benchmarks evaluate on 10-20+ source-target combinations across seen and unseen speakers.
- No comparison with **any existing VC system** (kNN-VC, FreeVC, VALL-E, YourTTS, etc.). The only baseline is the authors' own FAISS retrieval.
- No **subjective evaluation** (MOS, ABX preference tests). MCD and SECS alone do not capture perceptual quality -- artifacts, naturalness, and intelligibility are unmeasured.
- No word error rate (WER) or character error rate (CER) to verify content preservation.
- The 4,655-speaker dataset is unnamed and unreproducible.

**W3: The MCD improvement is marginal and inconsistent.** The best MCD (102.4 at step 210k) fluctuates between 102-105 across checkpoints (Table in Section 5.1). With evaluation on a single pair, this variation could easily be noise. No confidence intervals or significance tests are provided.

**W4: SECS degrades.** The proposed method (SECS 0.856) is *worse* than the FAISS baseline (0.879) on speaker similarity -- the primary metric for voice conversion. The paper frames this as an acceptable trade-off, but a 2.6% degradation in speaker similarity is significant in practice.

**W5: kNN-VC (Baas et al., Interspeech 2023) already addresses the same problem.** kNN-VC performs nearest-neighbor voice conversion on self-supervised representations and achieves strong results without any post-processing network. The paper does not compare against or adequately differentiate from this directly competing approach.

**W6: The 707M parameter model is excessive.** For a post-processing refinement model that only overrides 18% of tokens, 707M parameters is disproportionate. No analysis of smaller model sizes or inference latency is provided.

### Questions for Authors
1. Can you provide results on a standard benchmark (VCTK, LibriSpeech) with multiple source-target pairs?
2. How does CRT compare to kNN-VC and FreeVC in a fair evaluation?
3. What is the inference latency? Is real-time processing feasible?
4. Why does the SECS loss converge to near-zero but SECS evaluation shows degradation?

### Missing Related Work
- **StreamVC** (Google, 2024): Real-time streaming voice conversion on codec tokens
- **HierSpeech++** (2023): Hierarchical speech synthesis with multi-scale codec manipulation
- **NANSY++** (2023): Neural analysis and synthesis for voice conversion
- **SEF-VC** (2024): Speaker embedding-free voice conversion approach

### Decision: Reject
The paper presents a reasonable engineering system but lacks the novelty, rigorous evaluation, and competitive comparisons required for a top venue. The single-pair evaluation is a fatal flaw. I would encourage the authors to: (1) evaluate on standard benchmarks, (2) include subjective evaluations, (3) compare with published VC systems, and (4) articulate a clearer novel contribution beyond "apply post-editing to codec tokens."

---

## Review 2: Lenient Reviewer (Reviewer B)

**Overall Score: 6.5/10 (Weak Accept)**

**Summary:** This paper introduces an interesting approach to voice conversion by framing it as a post-retrieval refinement problem on neural codec tokens. The confidence-selective decoding and curriculum corruption training are practical contributions that could benefit the community.

### Strengths

**S1: Novel problem framing.** While post-processing is not new in NLP, applying it to multi-codebook RVQ tokens for voice conversion is novel and well-motivated. The observation that cross-speaker FAISS retrieval breaks RVQ chain consistency is insightful, and the correction-based approach is a natural solution that has not been explored in the VC literature.

**S2: Strong experimental design.** The curriculum corruption with inference-distribution matching is a thoughtful contribution. The ablation showing that mismatched corruption leads to over-correction (Section 5.5) is convincing and provides a useful lesson for practitioners.

**S3: Confidence-selective decoding.** The sweep across thresholds (Table 1) clearly demonstrates the trade-off curve and identifies a practical operating point. This is a simple but effective contribution that could be applied to other discrete-token generation systems.

**S4: Comprehensive method description.** The paper provides sufficient detail for reproduction. The continuity-aware FAISS retrieval formula, the Conformer architecture specifics, and the training procedure are all clearly presented.

**S5: The MCD improvement is real.** Achieving MCD 102.4 vs. baseline 106.5 (3.9% improvement) with only 18% of tokens changed demonstrates that the model makes targeted, meaningful corrections. The consistency of conf>0.4 as the optimal threshold across multiple checkpoints (102-103 range at steps 210k-235k) suggests this is not noise.

### Weaknesses

**W1: Limited evaluation scope.** The single source-target pair evaluation is the paper's main weakness. While I understand this may be a preliminary study, at least 5-10 speaker pairs would significantly strengthen the claims. I view this as addressable in a revision rather than a fundamental flaw.

**W2: No subjective evaluation.** MOS scores or ABX tests would substantially strengthen the paper. However, the objective metrics (SECS + MCD) are standard in the VC literature and sufficient for a workshop or applications-track paper.

**W3: No latency analysis.** The practical deployment story is incomplete without inference time measurements. The 707M model may be too large for real-time applications, but could be effective for offline processing.

**W4: The SECS trade-off needs more discussion.** The 0.856 vs. 0.879 SECS gap deserves deeper analysis. Is this due to CRT introducing tokens that are individually better but collectively shift speaker identity? Or is it an artifact of the single evaluation pair?

### Similar Published Methods

1. **kNN-VC (Baas et al., Interspeech 2023)**: Most similar -- retrieval-based VC on continuous features. CRT differs by: (a) operating on discrete tokens rather than continuous features, (b) adding a learned refinement stage, (c) using confidence-based selective override. The novelty over kNN-VC is the refinement + confidence selection, not the retrieval itself.

2. **SoundStorm (Borsos et al., 2023)**: Non-autoregressive codec generation via masked prediction. CRT differs by being a refinement model (not generative) and using confidence thresholding (not iterative unmasking).

3. **VALL-E (Wang et al., 2023)**: Autoregressive codec language model. CRT is fundamentally different: non-autoregressive, refinement-based, and much lighter at inference (single forward pass vs. sequential decoding).

4. **SEF-VC (2024)**: Speaker-embedding-free VC. Complementary approach -- CRT explicitly uses speaker embeddings for conditioning.

### Novelty Assessment

The paper's novelty lies in the **combination** of:
1. Post-retrieval refinement on discrete multi-codebook tokens (new formulation)
2. Confidence-selective decoding with fallback to retrieval (practical contribution)
3. Curriculum corruption matching inference distribution (training insight)

No single component is revolutionary, but the system-level contribution is meaningful. The problem of RVQ chain inconsistency in retrieval-based codec VC is real and underaddressed.

### Recommendation

**Weak Accept** for a workshop, applications track, or second-tier venue (e.g., SLT, ASRU, APSIPA, Interspeech workshop). The paper would need significant additional evaluation (more speaker pairs, subjective tests, competitive comparisons) for a top venue (Interspeech main, ICASSP, NeurIPS).

### Suggested Improvements for Camera-Ready
1. Add 5-10 source-target pairs with mean and standard deviation
2. Include at least a small-scale MOS evaluation (10-20 listeners, 5 pairs)
3. Report inference latency and compare with kNN-VC
4. Discuss when CRT corrections help vs. hurt (error analysis by codebook level)
5. Consider a smaller model variant (e.g., "large" at 28M params) to demonstrate the approach works at practical scale

---

## Summary Comparison

| Aspect | Strict Reviewer | Lenient Reviewer |
|--------|----------------|-----------------|
| Novelty | Low -- post-editing applied to codecs | Moderate -- novel formulation for VC |
| Evaluation | Fatal flaw (single pair) | Addressable weakness |
| MCD improvement | Marginal, possibly noise | Real and consistent |
| SECS degradation | Significant concern | Acceptable trade-off |
| Contribution level | Below threshold for any venue | Sufficient for workshop/SLT/ASRU |
| Decision | **Reject** | **Weak Accept** |
| Recommended venue | Resubmit after major revision | SLT, ASRU, APSIPA, Interspeech workshop |
