# How Fisher English Corpus Is Used & Vietnamese Equivalents

## 1. Fisher English Corpus: What It Is

The Fisher Corpus is a **telephone conversation** dataset created by the Linguistic Data
Consortium (LDC) specifically for training speech recognition systems.

| Attribute | Fisher Part 1 (LDC2004S13) | Fisher Part 2 (LDC2005S13) |
|---|---|---|
| Total audio files | 5,850 | ~5,000 |
| Hours of audio | 984 hours | ~800 hours |
| Duration per call | Up to 10 minutes | Up to 10 minutes |
| Sample rate | **8 kHz** (telephone quality) | **8 kHz** |
| Channels | **2 channels** (one per speaker) | **2 channels** |
| Encoding | mu-law, NIST SPHERE format | mu-law, NIST SPHERE format |
| Speakers | ~11,900 (6,813 female, 5,104 male) | Similar |
| Collection method | Automated phone system | Automated phone system |
| Content | Casual conversations on assigned topics | Casual conversations |
| Transcripts | Separate files (LDC2004T19, LDC2005T19) | Separate files |

**How conversations were collected:**
1. Participants called an automated telephone system
2. System paired two strangers randomly
3. System assigned a topic (e.g., "pets", "movies", "politics")
4. Participants talked for up to 10 minutes
5. Audio recorded as **two separate channels** — one per speaker

---

## 2. How PersonaPlex Uses Fisher Corpus

PersonaPlex uses **7,303 conversations (~1,217 hours)** from Fisher (both parts combined).
Here is the exact processing pipeline:

### Step 1: Audio Preprocessing

```
Fisher Raw Audio (8 kHz, mu-law, SPHERE format)
    ↓
Convert from SPHERE to WAV
    ↓
Upsample from 8 kHz → 24 kHz   ← Required by Mimi codec
    ↓
Split two channels into separate files:
    ├── speaker_A.wav (24 kHz, mono)
    └── speaker_B.wav (24 kHz, mono)
```

**Why upsample?** Mimi codec requires 24 kHz input. Upsampling from 8 kHz introduces
interpolation artifacts, but the Fisher data's value is in its **natural conversation
patterns**, not audio fidelity.

### Step 2: Retrospective Prompt Annotation (Back-Annotation)

This is the critical step that makes Fisher data usable for PersonaPlex.
Fisher was recorded WITHOUT any persona/role information — just two strangers talking.
PersonaPlex needs text prompts describing each speaker's role.

**Solution: Use a large LLM to generate prompts after the fact.**

```
For each Fisher conversation:
    1. Feed the transcript to GPT-OSS-120B (NVIDIA's internal LLM)
    2. LLM reads the conversation and generates:
       a. Text prompt for Speaker A:
          "You are a person having a casual phone conversation about pets.
           You are enthusiastic about your dogs and enjoy sharing stories."
       b. Text prompt for Speaker B:
          "You are a person listening to someone talk about pets.
           You are polite and ask follow-up questions."
    3. These prompts capture:
       - The apparent role of each speaker
       - Their conversational style
       - The topic and their engagement level
       - Personality traits observable from the conversation
```

**Example of back-annotation:**

```
Original Fisher transcript:
  A: So, uh, I have two golden retrievers. They're just the best dogs.
  B: Oh really? How old are they?
  A: One is three and the other just turned seven. The older one, Max...
  B: Mm-hmm, yeah...

Generated text prompt for A:
  "You are having a casual phone conversation about your pets.
   You own two golden retrievers and love talking about them.
   You are enthusiastic and enjoy sharing personal stories."

Generated text prompt for B:
  "You are having a casual phone conversation. Someone is telling
   you about their pets. You are a good listener and ask follow-up
   questions. You occasionally use backchannels to show engagement."
```

### Step 3: Voice Prompt Extraction

```
For each speaker in each conversation:
    1. Select a representative 3-10 second audio segment
       (clear speech, no overlap, no background noise)
    2. Encode with Mimi encoder → voice embedding tensor
    3. Save as .pt file → this is the voice prompt
```

### Step 4: Mimi Tokenization

```
For each speaker's full conversation audio (upsampled to 24 kHz):
    1. Run through Mimi encoder
    2. Output: 8 codebook token streams at 12.5 Hz
       codebook_1[T], codebook_2[T], ..., codebook_8[T]
       where T = duration_seconds * 12.5
```

### Step 5: Text Token Alignment

```
For each conversation:
    1. Tokenize the transcript using Helium tokenizer
    2. Align text tokens to 12.5 Hz audio frames using:
       - Forced alignment (if word-level timestamps exist), OR
       - ASR with timestamps, OR
       - Simple proportional distribution
    3. Most frames get PAD token (text is much sparser than audio)
```

### Step 6: Assembly

```
voice_prompt:  [embed_dim] tensor
text_prompt:   tokenized string, max 200 tokens
codes:         [K=9, T] tensor
    codes[0,:] = text tokens (aligned to audio frames)
    codes[1,:] = audio codebook 1 (model's speech)
    codes[2,:] = audio codebook 2
    ...
    codes[8,:] = audio codebook 8
user_codes:    [8, T] tensor (other speaker's audio codebooks)
```

### What Fisher Provides That Synthetic Data Cannot

| Feature | Fisher (Real) | Synthetic (TTS) |
|---|---|---|
| Backchanneling | Natural "mm-hmm", "yeah", "uh-huh" | Absent or scripted |
| Overlapping speech | Organic overlaps | Turn-based only |
| Disfluencies | Real "um", "uh", false starts | Unnatural if added |
| Emotional expression | Genuine laughter, surprise, hesitation | Limited expressivity |
| Turn-taking rhythm | Variable, context-dependent | Rigid timing |
| Silence patterns | Meaningful pauses (thinking, listening) | Fixed-length gaps |
| Speech rate variation | Natural acceleration/deceleration | Mostly constant |

**This is why PersonaPlex achieves 95% interruption handling and natural backchanneling —
it learned these patterns from real human conversations.**

---

## 3. Vietnamese Equivalents: What Exists

### 3.1 Open/Free Datasets

| Dataset | Hours | Type | Speakers | Channels | Sample Rate | Availability |
|---|---|---|---|---|---|---|
| **VIVOS** | 15h | Read speech | ~65 | Mono | 16 kHz | Free, open-source |
| **VinBigdata-VLSP2020-100h** | 100h | 20h read + 80h spontaneous | Unknown | Mono | 16 kHz | Free (via VLSP) |
| **Mozilla Common Voice (vi)** | ~50h (validated) | Read sentences | Crowdsourced | Mono | 48 kHz | Free, CC-0 |
| **DataOcean Vietnamese Conv.** | 213h | Conversational (mobile) | 253 | Unknown | 16 kHz | Free (until 2027) |

### 3.2 Commercial Datasets

| Dataset | Hours | Type | Speakers | Channels | Sample Rate | Notes |
|---|---|---|---|---|---|---|
| **FutureBeeAI General Conv.** | 30-50h | Two-speaker conversations | 70 | **Stereo (dual-channel)** | 16 kHz | Commercial license |
| **FutureBeeAI Telecom Call Center** | Unknown | Call center conversations | Unknown | Dual-channel, speaker-separated | 16 kHz | Commercial license |
| **FutureBeeAI Travel Call Center** | Unknown | Travel industry calls | Unknown | Dual-channel | 16 kHz | Commercial license |
| **Speechocean Vietnamese (Mobile)** | 76.6h | Read speech | 144 | Unknown | Unknown | Commercial |
| **Speechocean Vietnamese (In-Car)** | 305h | Read speech | 300 | Unknown | Unknown | Commercial |
| **GlobalPhone Vietnamese** | 22.5h | Read newspapers | 15 | Mono | 16 kHz | Commercial (ELRA) |

### 3.3 Analysis: Gap Between Fisher and Vietnamese Options

```
Fisher English:
  ✅ 1,217 hours of CONVERSATIONAL speech
  ✅ Two-channel (one per speaker) — CRITICAL for dual-stream training
  ✅ Natural telephone conversations with backchannels, overlaps
  ✅ Assigned topics → easy to generate text prompts
  ✅ Large speaker diversity (~12,000 speakers)

Best Vietnamese equivalent (FutureBeeAI General Conv.):
  ⚠️  Only 30-50 hours (vs 1,217 hours — 24x smaller)
  ✅ Two-channel (dual-channel, speaker-separated) — good!
  ✅ Natural conversations with various topics
  ⚠️  Only 70 speakers (vs ~12,000)
  ❌ Commercial license required
  ⚠️  16 kHz (needs upsampling to 24 kHz, but better than Fisher's 8 kHz)

DataOcean Vietnamese Conv.:
  ✅ 213 hours — decent volume
  ⚠️  Channel separation unclear
  ✅ Free until 2027
  ⚠️  253 speakers (decent but not huge)
  ⚠️  16 kHz sample rate
```

**The fundamental gap:**
There is NO Vietnamese dataset equivalent to Fisher Corpus in both **scale** (1000+ hours)
and **format** (two-channel conversational telephone speech). The largest conversational
Vietnamese dataset available is ~213 hours, roughly 6x smaller than Fisher.

---

## 4. Strategy for Building Vietnamese Training Data

Given the gap, here is a practical research strategy:

### Phase 1: Bootstrap with Available Data (~200-300 hours)

```
Data Source                          Hours   Type            Channel
─────────────────────────────────────────────────────────────────────
DataOcean Vietnamese Conv.           213h    Conversational  TBD
FutureBeeAI General Conv.            50h     Two-speaker     Stereo ✓
VinBigdata-VLSP2020 (spontaneous)    80h     Spontaneous     Mono
─────────────────────────────────────────────────────────────────────
Subtotal (real data)                 ~343h
```

Apply the same back-annotation pipeline as PersonaPlex uses for Fisher:
1. Get transcripts (ASR if not available)
2. Use Vietnamese-capable LLM (Vistral, Qwen3) to generate text prompts
3. Extract voice embeddings with Mimi
4. Assemble into training format

### Phase 2: Create Vietnamese Synthetic Data (~1,000-2,000 hours)

This is where the tools we built come in:

```
Scenario Generation (generate_scenarios.py)
    ↓ Vietnamese scenarios: customer service, assistant, casual, etc.
Transcript Generation (generate_transcript.py)
    ↓ Vietnamese LLM generates conversation transcripts
Audio Synthesis
    ↓ Vietnamese TTS converts transcripts to audio
    ↓ Options: ViXTTS, VITS-Vietnamese, Edge TTS, Resemble AI
Tokenization
    ↓ Mimi encoder → token streams
Assembly
    ↓ codes[B, 9, T] training tensors
```

**Target: 100K+ synthetic conversations across diverse Vietnamese scenarios**

### Phase 3: Collect Original Vietnamese Conversations

For a PhD project, collecting original data adds significant research value:

**Option A: Vietnamese Telephone Conversations (Fisher-style)**
- Set up a recording system (web-based is easier than phone)
- Recruit participants (university students, online communities)
- Assign conversation topics relevant to Vietnamese culture
- Record as two separate audio channels
- Target: 100-500 hours
- Advantage: Most direct equivalent to Fisher

**Option B: Vietnamese Podcast/YouTube Mining**
- Scrape Vietnamese podcasts with two hosts/guests
- Use speaker diarization to separate speakers
- Transcribe with Vietnamese ASR (e.g., Whisper fine-tuned for Vietnamese)
- Back-annotate with LLM
- Target: 500-1,000 hours (large volume, but noisy)

**Option C: Call Center Partnerships**
- Partner with Vietnamese companies (telecom, e-commerce)
- Obtain anonymized call recordings
- Already two-channel (agent + customer)
- Already task-oriented (perfect for text prompt generation)
- Target: varies by partnership

### Phase 4: Combined Training (~2,000-3,000 hours total)

```
Real conversational data:    ~300-500 hours  (natural patterns, backchannels)
Synthetic conversations:     ~1,000-2,000 hours (task-following, role control)
Original collected data:     ~200-500 hours  (Vietnamese-specific patterns)
────────────────────────────────────────────────────────────────────────────
Total:                       ~1,500-3,000 hours
```

For comparison, PersonaPlex used ~3,467 hours total. Reaching 2,000+ hours for Vietnamese
would be a strong foundation.

---

## 5. Key Technical Considerations

### 5.1 Audio Format Requirements

All audio must be converted to:
- **24 kHz sample rate** (Mimi requirement)
- **Mono** per speaker (two files per conversation)
- **WAV format** (lossless)

Most Vietnamese datasets are at 16 kHz — upsampling from 16 kHz to 24 kHz is
much better quality than Fisher's 8 kHz → 24 kHz upsampling.

### 5.2 Vietnamese Tones in Mimi

Vietnamese has 6 tones that carry lexical meaning:
```
Tone          Mark    Example   Meaning
──────────────────────────────────────
Level (ngang)  none   ma        ghost
Falling (huyền)  `    mà        but
Rising (sắc)    ´    má        mother/cheek
Dipping (hỏi)   ̉    mả        grave/tomb
Broken (ngã)    ~    mã        horse (code)
Heavy (nặng)    .    mạ        rice seedling
```

Mimi's first codebook captures **semantic** information (via WavLM distillation).
Since WavLM was primarily trained on English, it may not encode Vietnamese tonal
distinctions well. This is a critical research question:

**Experiment needed:** Encode Vietnamese minimal tone pairs through Mimi and
check if the first codebook tokens differ. If they don't, Mimi needs fine-tuning.

### 5.3 Vietnamese Backchanneling Patterns

Vietnamese backchannels differ from English and must be present in training data:

| Vietnamese | Context | English Equivalent |
|---|---|---|
| Dạ / Vâng | Polite acknowledgment (to elder/superior) | "Yes" / "I see" |
| Ừ / Ừm | Casual acknowledgment (to peer/younger) | "Mm-hmm" / "Yeah" |
| À | Realization or mild surprise | "Oh" / "Ah" |
| Ờ | Casual agreement | "Uh-huh" |
| Đúng rồi | Agreement/confirmation | "That's right" |
| Vậy à / Thật hả | Surprise/interest | "Really?" / "Is that so?" |
| Rồi | Completion acknowledgment | "OK" / "Got it" |

These MUST appear naturally in training data — both in real conversations
and in synthetic data generation prompts.

### 5.4 Vietnamese Formality System

Vietnamese has a complex pronoun/address system based on age and social status.
Training data must cover:

| Register | Speaker→Listener | Typical Scenario |
|---|---|---|
| Formal | Em→Anh/Chị | Customer service, junior→senior |
| Formal | Con→Cô/Chú | To uncle/aunt age |
| Semi-formal | Mình→Bạn | Between peers |
| Informal | Tao→Mày | Close friends |
| Professional | Tôi→Anh/Chị | Business contexts |

Text prompts should specify the appropriate formality level.

---

## 6. Research Contribution Opportunities

For your PhD, the Vietnamese adaptation presents several publishable contributions:

1. **Vietnamese Conversational Speech Dataset**: Creating a Fisher-equivalent dataset
   for Vietnamese would be a significant resource contribution

2. **Mimi Codec Adaptation for Tonal Languages**: Studying how well Mimi represents
   Vietnamese tones, and fine-tuning if needed — applicable to Chinese, Thai, etc.

3. **Cross-lingual Full-Duplex Speech**: First Vietnamese full-duplex conversational
   AI system

4. **Vietnamese Backchannel Modeling**: Analyzing and modeling Vietnamese-specific
   conversational patterns in neural speech systems

5. **Synthetic Data Generation for Low-Resource Languages**: Methodology for creating
   training data when no Fisher-equivalent exists
