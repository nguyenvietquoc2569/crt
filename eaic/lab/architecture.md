# Mimi-VC V3e: Architecture

## 1. Mimi Codec Structure (Foundation)

```
 Audio Waveform (24kHz)
        |
        v
+------------------------------------------------------+
|                  Mimi Encoder                         |
|              (Convolutional + Transformer)            |
+----------------------+-------------------------------+
                       |
                       v
+------------------------------------------------------+
|          Split Residual Vector Quantizer              |
|                                                       |
|  +---------+  +---------------------------------+    |
|  |Semantic  |  |     Acoustic RVQ (CB1-31)       |    |
|  |RVQ (CB0) |  |                                 |    |
|  |          |  |  CB1  CB2  CB3 ... CB31         |    |
|  | WavLM    |  |  +-+  +-+  +-+      +-+         |    |
|  |distilled |  |  +-+  +-+  +-+      +-+         |    |
|  |          |  |  Speaker identity, prosody,     |    |
|  | Content  |  |  timbre, acoustic details       |    |
|  | (what)   |  |  (how it sounds)                |    |
|  +---------+  +---------------------------------+    |
|   2048 codes    2048 codes each x 31 codebooks       |
|   12.5 Hz       12.5 Hz                              |
+------------------------------------------------------+
```

## 2. Voice Conversion Idea

```
  Source Speaker A                    Target Speaker B
  "Hello world"                       (voice identity)
       |                                    |
       v                                    |
  +---------+                               |
  |  Mimi   |                               |
  | Encode  |                               |
  +----+----+                               |
       |                                    |
       v                                    v
  +---------+ +-----------+         +------------+
  |  CB0    | |  CB1-CB6  |         |  Speaker   |
  |(content)| | (A's voice|  --X--  |  Embedding |
  | KEEP    | |  DISCARD) |         |  (d-vector)|
  +----+----+ +-----------+         +-----+------+
       |                                  |
       |         +--------------+         |
       +-------->| Acoustic     |<--------+
                 | Translator   |
                 |    V3e       |
                 +------+-------+
                        |
                        v
                 +--------------+
                 |  New CB1-CB6 |
                 | (B's voice,  |
                 |  A's content)|
                 +------+-------+
                        |
         CB0 (kept) ----+
                        v
                 +--------------+
                 | Mimi Decode  |---> Audio: "Hello world"
                 |              |     in Speaker B's voice
                 +--------------+
```

## 3. AcousticTranslatorV3e Model Architecture

```
Inputs:
  CB0 tokens [B, T]          Speaker Embedding [B, 256]
       |                            |
       v                            |
+-------------+                     |
|CB0 Embedding|                     |
|Table (2048) |                     |
+------+------+                     |
       | [B, T, 512]                |
       v                            |
+-----------------------------+     |
|      ENCODER (6 layers)     |     |
|  +-----------------------+  |     |
|  |  Self-Attention + RoPE|  |     |
|  |  (bidirectional)      |  |     |
|  +-----------------------+  |     |
|  |  FiLM Conditioning <--+--+-----+
|  |  scale = W_s*spk + b  |  |  Speaker identity injected
|  |  shift = W_h*spk + b  |  |  at every encoder layer
|  |  out = scale*x + shift |  |
|  +-----------------------+  |
|  |  Feed-Forward (x4)    |  |
|  +-----------------------+  |
|         x 6 layers          |
+----------+------------------+
           |
           | [B, T, 512]
           |
           |  CB Index Embedding [6, 512]
           |  (which codebook am I predicting?)
           |         |
           v         v
+-----------------------------+
|      DECODER (6 layers)     |
|  +-----------------------+  |
|  |  Cross-Attention      |  |  Decoder queries attend
|  |  Q=cb_query K,V=enc   |  |  to encoder output
|  +-----------------------+  |
|  |  Self-Attention + RoPE|  |
|  +-----------------------+  |
|  |  Feed-Forward (x4)    |  |
|  +-----------------------+  |
|         x 6 layers          |
|                             |
|  Input: [B*6, T, 512]      |  All 6 CBs processed
|  (6 CB queries in parallel) |  in one forward pass
+----------+------------------+
           |
           v
+-----------------------------+
|  Output Heads (per CB)      |
|                             |
|  Linear(512 -> 2048) x 6   |  One head per codebook
|                             |
|  -> logits [B, 6, T, 2048] |  2048-class classification
+----------+------------------+
           |
           v
      argmax -> predicted tokens [B, 6, T]
```

## 4. V3e Training: Mixed Batch (The Key Innovation)

```
+----------------------------------------------------------+
|                    TRAINING BATCH                         |
|                                                           |
|   50% Self-Reconstruction          50% Cross-Speaker      |
|   ---------------------          ------------------       |
|                                                           |
|   Speaker A -> Speaker A         Speaker A -> Speaker B   |
|                                                           |
|   +-------+    +-------+        +-------+    +-------+   |
|   |A: CB0 |    |A:CB1-6|        |A: CB0 |    |B: spk |   |
|   |tokens |    |tokens |        |tokens |    |embed  |   |
|   +---+---+    +---+---+        +---+---+    +---+---+   |
|       |            |                |            |        |
|       |     Ground Truth            |      +-----+        |
|       |     (real tokens)           |      |              |
|       v            |                v      v              |
|   +------------+   |           +------------+             |
|   |  Model     |   |           |  Model     |             |
|   |  forward   |   |           |  forward   |             |
|   +-----+------+   |           +-----+------+             |
|         |          |                 |                     |
|         v          v                 v                     |
|   +------------------+        +------------------+        |
|   | CE Loss:         |        | CE Loss:         |        |
|   | predict A's own  |        | predict FAISS    |        |
|   | acoustic tokens  |        | pseudo-labels    |<--+    |
|   +------------------+        +------------------+   |    |
|                                                      |    |
|                                       +----------+   |    |
|                                       |  FAISS   |---+    |
|                                       | Retrieval|        |
|                                       +----------+        |
+----------------------------------------------------------+
```

## 5. FAISS Cross-Speaker Pseudo-Label Generation

```
Source: Speaker A says "hello"
Target: Speaker B's voice

Speaker A's CB0 tokens: [42, 108, 108, 77, 203]
                          h    e    l    l    o

For each source CB0 token (e.g., token 108 = "e"):

    +--------------------------------------+
    |  CB0 embedding of token 108          |
    |  (content = "e" sound)               |
    +--------------+-----------------------+
                   |
                   v
    +--------------------------------------+
    |  Search Speaker B's FAISS Index      |
    |  (all of B's CB0 embeddings)         |
    |                                      |
    |  Find K=8 nearest neighbors          |
    |  by cosine similarity                |
    +--------------+-----------------------+
                   |
                   v
    +--------------------------------------+
    |  Matches in B's data:                |
    |  frame 45 (sim=0.97) -> CB1-6 tokens |
    |  frame 112(sim=0.95) -> CB1-6 tokens |
    |  frame 203(sim=0.94) -> CB1-6 tokens |  Weighted
    |  ...                                 |  average
    |  frame 89 (sim=0.88) -> CB1-6 tokens |  (top-K)
    +--------------+-----------------------+
                   |
                   v
    +--------------------------------------+
    |  + Temporal Smoothing (alpha=0.5)    |
    |  Blend with previous frame's result  |
    |  to avoid jittery token sequences    |
    +--------------+-----------------------+
                   |
                   v
    Pseudo-label: B's CB1-6 tokens for "e" sound
    = "What speaker B's acoustic tokens look like
       when saying the same content as speaker A"
```

## 6. Loss Function (with Bug Fix)

```
For each codebook i in {1..6}:

  logits [B*T, 2048]     targets [B*T]
       |                      |
       +----------+           |
       |          |           |
       v          |           v
  +---------+    |    +-------------+
  |Cross-   |    |    |             |
  |Entropy  |    |    |  (integer   |
  |Loss     |<---+----+   labels)   |
  +----+----+    |    +-------------+
       |         |
       |         v
       |    +----------------------------------------------+
       |    |  Diversity Loss (BUG WAS HERE)                |
       |    |                                               |
       |    |  avg_probs = softmax(logits).mean(0)          |
       |    |              ---------------------            |
       |    |              NO .detach() !!                  |
       |    |              gradients flow                   |
       |    |                                               |
       |    |  entropy = -sum( p * log(p) )                 |
       |    |  max_ent = log(2048) = 7.62                   |
       |    |  div_loss = 1 - entropy / max_ent             |
       |    +----------+-----------------------------------+
       |               |
       v               v
  +----------------------------+
  |  total = CE + 1.0 * div    |   weight = 1.0 (pre-train)
  |                            |   weight = 0.5 (fine-tune)
  +------------+---------------+
               |
               v
          Backprop

  +----------------------------------------------+
  |  What diversity gradient does:                |
  |                                               |
  |  Token 42 used 80% of time -> gradient says:  |
  |  "REDUCE probability of token 42"             |
  |                                               |
  |  Token 1837 used 0.01% -> gradient says:      |
  |  "INCREASE probability of token 1837"         |
  |                                               |
  |  -> Pushes toward UNIFORM usage of all 2048   |
  |  -> Fights mode collapse directly             |
  +----------------------------------------------+

  THE BUG: .detach() cut this entire gradient path.
  Model saw diversity loss in logs but received
  ZERO gradient signal from it. Training was
  pure cross-entropy = identical to V3c.
```

## 7. Inference Pipeline

```
+------------------+    +------------------+
|  Source Audio     |    |  Target Audio    |
|  (Speaker A)     |    |  (Speaker B)     |
+--------+---------+    +--------+---------+
         |                       |
         v                       v
+-----------------+    +------------------+
|  Mimi Encode    |    |  Resemblyzer     |
|  -> 32 codebooks|    |  -> d-vector     |
+--------+--------+    |    [256-dim]     |
         |             +--------+---------+
    +----+----+                 |
    |         |                 |
    v         v                 |
 +-----+ +------+              |
 | CB0 | |CB1-31|              |
 |     | |(drop)|              |
 +--+--+ +------+              |
    |                          |
    v                          v
+--------------------------------------+
|     AcousticTranslatorV3e            |
|                                      |
|  CB0 tokens --> Encoder <-- spk_emb  |
|                    |                 |
|                    v                 |
|               Decoder                |
|                    |                 |
|                    v                 |
|  logits [1, 6, T, 2048]             |
|  argmax -> new CB1-6 tokens          |
|  CB7-31 = zeros (not learned)        |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|  Reassemble: CB0 (original)          |
|            + CB1-6 (predicted)       |
|            + CB7-31 (zeros)          |
|  -> full 32 codebooks                |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          Mimi Decode                  |
|  32 codebooks -> Audio waveform      |
|  "Hello world" in Speaker B's voice  |
+--------------------------------------+
```

## 8. Key Configuration

| Parameter | Pre-training | Fine-tuning |
|-----------|-------------|-------------|
| Batch size (A100) | 48 | 32 |
| Learning rate | 3e-4 | 5e-5 |
| Diversity weight | 1.0 | 0.5 |
| Cross-speaker ratio | 50% | 0% (target only) |
| Steps | 30,000 | 3,000 |
| CB0 masking | 30% span | 30% span |
| Speaker noise | std=0.3 | std=0.1 |

## 9. Version Evolution

```
V3a: Cross-entropy only          -> 40 tokens (mode collapse)
V3b: + label smoothing (0.1)     -> 40 tokens (no change)
V3c: + entropy regularization    -> 40 tokens (.detach = no-op)
V3d: Cosine embedding loss       -> 47 tokens (quality worse)
V3e: Cross-speaker FAISS         -> 39 tokens (.detach bug!)
V3e (fixed): Remove .detach()    -> ??? (pending results)
     + diversity_weight = 1.0
```
