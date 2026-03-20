# Training Data Generation Guide

## 1. End-to-End Data Pipeline

This document describes how to create training data for Vietnamese PersonaPlex,
from raw materials to the final tensor format.

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA GENERATION PIPELINE                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Scenario    │───▶│  Transcript  │───▶│  Audio         │   │
│  │  Templates   │    │  Generation  │    │  Synthesis     │   │
│  │  (JSON)      │    │  (LLM)       │    │  (TTS)         │   │
│  └─────────────┘    └──────────────┘    └────────────────┘   │
│         │                  │                     │             │
│         ▼                  ▼                     ▼             │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │  Text Prompt │    │  Text Token  │    │  Mimi Audio    │   │
│  │  Generation  │    │  Alignment   │    │  Tokenization  │   │
│  └─────────────┘    └──────────────┘    └────────────────┘   │
│         │                  │                     │             │
│         └──────────────────┼─────────────────────┘             │
│                            ▼                                   │
│                   ┌────────────────┐                           │
│                   │  Training      │                           │
│                   │  Sample        │                           │
│                   │  Assembly      │                           │
│                   │  codes[B,9,T]  │                           │
│                   └────────────────┘                           │
│                            │                                   │
│                            ▼                                   │
│                   ┌────────────────┐                           │
│                   │  Validation &  │                           │
│                   │  Quality Check │                           │
│                   └────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Step 1: Scenario Templates

### 2.1 Scenario Template Schema

Each scenario defines the context for generating a conversation:

```json
{
  "scenario_id": "cs_techstore_001",
  "category": "customer_service",
  "subcategory": "product_complaint",
  "language": "vi",

  "agent": {
    "role": "Nhân viên chăm sóc khách hàng",
    "company": "TechStore Việt Nam",
    "personality": ["kiên nhẫn", "đồng cảm", "chuyên nghiệp"],
    "expertise": ["điện tử tiêu dùng", "chính sách bảo hành"],
    "constraints": ["không được giảm giá quá 15%", "phải xác minh mã đơn hàng"]
  },

  "customer": {
    "issue": "Nhận được tai nghe sai màu",
    "emotion": "thất vọng nhưng lịch sự",
    "goal": "đổi sản phẩm đúng màu hoặc hoàn tiền",
    "context": "Đã đặt tai nghe Bluetooth SoundMax Pro màu đen, nhận được màu trắng"
  },

  "product": {
    "name": "Tai nghe Bluetooth SoundMax Pro",
    "category": "Phụ kiện công nghệ",
    "price": "1,200,000 VNĐ"
  },

  "expected_turns": [8, 15],
  "expected_duration_seconds": [60, 300],

  "resolution_options": [
    "Đổi sản phẩm miễn phí",
    "Hoàn tiền đầy đủ",
    "Giảm giá cho lần mua tiếp theo"
  ]
}
```

### 2.2 Scenario Categories for Vietnamese

| Category | Subcategories | Example Scenarios |
|---|---|---|
| **Dịch vụ khách hàng** | Khiếu nại, hỗ trợ kỹ thuật, đặt hàng, hoàn tiền | Gọi điện hotline VinMart, FPT Shop |
| **Trợ lý AI** | Hỏi đáp, tra cứu thông tin, dịch thuật | Chatbot ngân hàng, trợ lý sức khỏe |
| **Gọi điện thoại** | Gọi bạn bè, gia đình, đồng nghiệp | Hẹn đi ăn, bàn công việc |
| **Phỏng vấn** | Tuyển dụng, báo chí, nghiên cứu | Phỏng vấn xin việc IT |
| **Tư vấn** | Giáo dục, sức khỏe, tài chính | Tư vấn chọn trường đại học |

---

## 3. Step 2: Transcript Generation

### 3.1 LLM Prompt for Vietnamese Transcript

```
Hệ thống: Bạn là một trình tạo hội thoại thực tế bằng tiếng Việt.

Ngữ cảnh:
- Công ty: {company}
- Sản phẩm: {product_name}
- Vấn đề: {issue}
- Tính cách nhân viên: {personality}
- Mục tiêu khách hàng: {customer_goal}

Tạo một cuộc hội thoại tự nhiên giữa nhân viên chăm sóc khách hàng
và khách hàng. Bao gồm:
- Lời chào và xác định vấn đề
- Thể hiện sự đồng cảm
- Xác minh thông tin đơn hàng
- Đề xuất giải pháp
- Kết thúc cuộc gọi

Yêu cầu:
- Sử dụng tiếng Việt tự nhiên, có thể pha trộn ít tiếng Anh cho thuật ngữ công nghệ
- Bao gồm các từ đệm tự nhiên: "ạ", "dạ", "vâng", "ừm"
- Thể hiện cảm xúc phù hợp
- Tối thiểu {min_turns} lượt, tối đa {max_turns} lượt

Định dạng mỗi lượt:
NHÂN VIÊN: <nội dung>
KHÁCH HÀNG: <nội dung>
```

### 3.2 Output Transcript Format

```json
{
  "scenario_id": "cs_techstore_001",
  "transcript": [
    {
      "turn": 1,
      "speaker": "agent",
      "text": "Dạ xin chào anh/chị, em là Linh từ TechStore Việt Nam ạ. Em có thể giúp gì cho anh/chị ạ?",
      "emotion": "friendly",
      "duration_estimate_ms": 4500
    },
    {
      "turn": 2,
      "speaker": "customer",
      "text": "Ừm, chào em. Anh gọi vì anh đặt tai nghe SoundMax Pro màu đen nhưng nhận được màu trắng. Anh hơi thất vọng vì đã chờ mấy ngày rồi.",
      "emotion": "disappointed",
      "duration_estimate_ms": 6000
    },
    {
      "turn": 3,
      "speaker": "agent",
      "text": "Dạ, em rất xin lỗi về sự bất tiện này ạ. Anh có thể cho em xin mã đơn hàng để em kiểm tra không ạ?",
      "emotion": "empathetic",
      "duration_estimate_ms": 4000
    }
  ],

  "text_prompt_agent": "Bạn là nhân viên chăm sóc khách hàng tại TechStore Việt Nam. Bạn kiên nhẫn, đồng cảm và chuyên nghiệp. Bạn hỗ trợ khách hàng về các vấn đề đơn hàng và bảo hành sản phẩm điện tử.",
  "text_prompt_customer": "Bạn là khách hàng đã đặt tai nghe Bluetooth SoundMax Pro màu đen nhưng nhận được màu trắng. Bạn thất vọng nhưng lịch sự. Bạn muốn đổi sản phẩm đúng màu.",

  "metadata": {
    "generated_by": "qwen3-32b",
    "generation_timestamp": "2026-03-02T10:00:00Z",
    "quality_score": 0.92,
    "total_turns": 12,
    "total_duration_estimate_ms": 58000
  }
}
```

---

## 4. Step 3: Audio Synthesis

### 4.1 TTS Options for Vietnamese

| TTS System | Vietnamese Support | Quality | Open Source | Notes |
|---|---|---|---|---|
| **ViXTTS** | Native | Good | Yes | Based on XTTS, Vietnamese-focused |
| **VITS** (Vietnamese) | Fine-tuned | Good | Yes | Fast, lightweight |
| **Resemble AI** | Yes (API) | High | No | Commercial, has Vietnamese |
| **Edge TTS** | Yes | Medium | Free API | Microsoft Azure backend |
| **Underthesea TTS** | Native | Medium | Yes | Vietnamese NLP library |

### 4.2 Audio Synthesis Pipeline

```python
# Conceptual pipeline for audio synthesis

for each conversation transcript:
    # 1. Select voices for each speaker
    agent_voice = select_voice(gender="female", style="professional")
    customer_voice = select_voice(gender="male", style="casual")

    # 2. Synthesize each turn
    audio_segments = []
    for turn in transcript.turns:
        voice = agent_voice if turn.speaker == "agent" else customer_voice
        audio = tts.synthesize(
            text=turn.text,
            voice=voice,
            language="vi",
            sample_rate=24000,  # MUST be 24kHz for Mimi
            emotion=turn.emotion
        )
        audio_segments.append({
            "speaker": turn.speaker,
            "audio": audio,
            "duration_ms": len(audio) / 24  # samples / sample_rate_khz
        })

    # 3. Mix into two-channel audio with timing
    agent_channel, customer_channel = mix_channels(
        segments=audio_segments,
        pause_between_turns_ms=random.uniform(200, 800),
        allow_overlap=True,
        overlap_probability=0.1
    )

    # 4. Save as two WAV files (one per speaker)
    save_wav("agent.wav", agent_channel, sr=24000)
    save_wav("customer.wav", customer_channel, sr=24000)
```

### 4.3 Audio Requirements

| Parameter | Requirement | Notes |
|---|---|---|
| Sample rate | **24,000 Hz** | Mimi codec requirement |
| Channels | Mono per speaker | Two separate files or channels |
| Bit depth | 16-bit or float32 | Standard WAV format |
| Format | WAV | Lossless required |
| Duration | 30s - 10min per conversation | Matching Fisher Corpus range |

---

## 5. Step 4: Prompt Annotation

### 5.1 For Synthetic Data

Text prompts are generated during scenario design (already available).

### 5.2 For Real Vietnamese Conversations (Back-Annotation)

For existing Vietnamese conversation recordings, we need to retrospectively
generate prompts using an LLM:

```
Input to LLM:
- Transcript of the conversation (ASR output or manual)
- Speaker information (if available)
- Recording context (if available)

Prompt for LLM:
"Given this Vietnamese conversation transcript, generate:
1. A text prompt (max 200 tokens) describing Speaker A's role and behavior
2. A text prompt (max 200 tokens) describing Speaker B's role and behavior
The prompts should capture the speaker's apparent role, personality,
topic expertise, and conversational style."

Output:
- text_prompt_speaker_a: "Bạn là một người bạn đang kể về chuyến du lịch..."
- text_prompt_speaker_b: "Bạn đang nghe bạn kể chuyện du lịch..."
```

---

## 6. Step 5: Tokenization

### 6.1 Audio Tokenization with Mimi

```python
import torch
from mimi import MimiEncoder

encoder = MimiEncoder.load("mimi_checkpoint")

# Load audio at 24kHz
audio = load_wav("speaker_audio.wav", sr=24000)  # [1, num_samples]

# Encode to tokens
# Output: codes of shape [1, 8, T] where T = num_samples / 1920
audio_codes = encoder.encode(audio)

# audio_codes[0, 0, :] = codebook 1 tokens (semantic)
# audio_codes[0, 1, :] = codebook 2 tokens
# ...
# audio_codes[0, 7, :] = codebook 8 tokens (finest detail)
```

### 6.2 Text Tokenization

```python
from transformers import AutoTokenizer

# Helium tokenizer (or a multilingual alternative for Vietnamese)
tokenizer = AutoTokenizer.from_pretrained("helium_tokenizer")

text_prompt = "Bạn là nhân viên chăm sóc khách hàng tại TechStore Việt Nam."
text_tokens = tokenizer.encode(text_prompt)
# [token_id_1, token_id_2, ..., token_id_N]  (max 200 tokens)
```

### 6.3 Inner Monologue Text Tokens

For the text stream (stream 0) during conversation, text tokens are aligned
to the 12.5 Hz frame rate. Since text is much sparser than audio:

```
Frame:  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | ...
Text:   | "Xin"| PAD | PAD |"chào"| PAD |"anh"| PAD | PAD | PAD |"ạ" | ...
Audio:  | a0  | a1  | a2  | a3  | a4  | a5  | a6  | a7  | a8  | a9  | ...
```

Most frames have a PAD/empty text token since speech is much denser than text.

---

## 7. Step 6: Training Sample Assembly

### 7.1 Final Assembly

```python
def assemble_training_sample(
    voice_prompt: torch.Tensor,      # [embed_dim] or [8, T_voice]
    text_prompt_tokens: List[int],   # max 200 token IDs
    user_audio_codes: torch.Tensor,  # [8, T_conv] from Mimi
    model_audio_codes: torch.Tensor, # [8, T_conv] from Mimi
    text_tokens: torch.Tensor,       # [T_conv] aligned text tokens
) -> dict:
    """
    Assemble a complete training sample in PersonaPlex format.
    """
    T = user_audio_codes.shape[1]

    # Build the K=9 code tensor
    codes = torch.zeros(9, T, dtype=torch.long)
    codes[0, :] = text_tokens          # Stream 0: text (inner monologue)
    codes[1:9, :] = model_audio_codes  # Stream 1-8: model's audio codebooks

    # User audio is provided as separate input
    user_codes = user_audio_codes  # [8, T]

    return {
        "voice_prompt": voice_prompt,
        "text_prompt_tokens": text_prompt_tokens,
        "codes": codes,           # [9, T]
        "user_codes": user_codes, # [8, T]
    }
```

### 7.2 Batch Collation

```python
def collate_batch(samples: List[dict]) -> dict:
    """Collate samples into a batch with padding."""
    max_T = max(s["codes"].shape[1] for s in samples)
    B = len(samples)

    batch_codes = torch.zeros(B, 9, max_T, dtype=torch.long)
    batch_user_codes = torch.zeros(B, 8, max_T, dtype=torch.long)
    batch_mask = torch.zeros(B, max_T, dtype=torch.bool)

    for i, s in enumerate(samples):
        T = s["codes"].shape[1]
        batch_codes[i, :, :T] = s["codes"]
        batch_user_codes[i, :, :T] = s["user_codes"]
        batch_mask[i, :T] = True

    return {
        "codes": batch_codes,           # [B, 9, max_T]
        "user_codes": batch_user_codes, # [B, 8, max_T]
        "mask": batch_mask,             # [B, max_T]
    }
```

---

## 8. Quality Validation Checklist

Before using generated data for training:

- [ ] Audio sample rate is exactly 24,000 Hz
- [ ] Audio duration is between 30 seconds and 10 minutes
- [ ] Text prompt is under 200 tokens
- [ ] Vietnamese text is properly encoded (UTF-8, correct diacritics)
- [ ] Vietnamese tones are properly represented in audio
- [ ] Backchannel words ("ừ", "vâng", "dạ") are present and natural
- [ ] Turn-taking pauses are realistic (200-1000ms)
- [ ] Speaker voices are distinct and consistent
- [ ] Emotional expression matches transcript annotation
- [ ] No audio artifacts (clipping, silence gaps, noise)
- [ ] Mimi tokenization produces valid token ranges (0-2047)
- [ ] Text tokens align correctly with audio frames
