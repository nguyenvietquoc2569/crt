# v5.6.2 — Confidence-Based Selective Refinement + FAISS Re-ranker

## Problem

v5.6.1 CRT replaces ALL CB1-18 tokens via argmax, even when FAISS was correct.
With ~50% CB1 accuracy, CRT replaces correct FAISS tokens with wrong predictions
half the time, losing speaker identity (SECS 0.783 vs FAISS 0.879).

## Solution (inference-time only, no retraining)

Two techniques:

### 1. Confidence-based selective refinement
Only override FAISS when CRT is confident (softmax prob > threshold):
- High confidence: CRT is likely right -> use CRT token
- Low confidence: CRT is uncertain -> keep FAISS token (preserves speaker ID)

### 2. FAISS re-ranker
Instead of picking from full 2048 vocabulary, constrain CRT to choose
among the 8 FAISS candidate tokens. This guarantees the output token
exists in the target speaker's data (no hallucinated tokens).

## Output variants

The script generates and compares all variants:
- faiss_baseline: raw FAISS (all 32 CB)
- crt_argmax: original CRT (replace all)
- crt_conf_0.3 through crt_conf_0.9: confidence sweep
- crt_reranker: CRT picks best among FAISS top-8
- crt_reranker_conf: re-ranker + confidence threshold

## No retraining needed

Uses checkpoint from v5.6.1 directly. Only the inference path changed.
