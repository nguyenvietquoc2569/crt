"""
Training Data Format Visualizer for Vietnamese PersonaPlex.

This tool generates visual examples of how training data is structured
at the token level. Useful for understanding and debugging the data pipeline.

Usage:
    python tools/visualize_data_format.py --transcript data/processed/transcripts/transcript_cs_xxx.json
    python tools/visualize_data_format.py --example
"""

import argparse
import json
import random
from pathlib import Path


def simulate_mimi_tokens(text: str, duration_ms: int, num_codebooks: int = 8) -> dict:
    """
    Simulate Mimi codec tokens for a segment of audio.
    In production, these come from the actual Mimi encoder.

    Returns simulated token IDs for visualization purposes.
    Codebook values are in range [0, 2047].
    Frame rate: 12.5 Hz (80ms per frame).
    """
    num_frames = max(1, duration_ms // 80)
    tokens = {}
    for cb in range(num_codebooks):
        # Simulate token IDs (in reality, from Mimi RVQ)
        tokens[f"codebook_{cb+1}"] = [random.randint(0, 2047) for _ in range(num_frames)]
    return {
        "num_frames": num_frames,
        "duration_ms": duration_ms,
        "tokens": tokens,
    }


def simulate_text_tokens(text: str, num_frames: int, vocab_size: int = 32000) -> list:
    """
    Simulate text token alignment with audio frames.
    Text tokens are sparse — most frames get a PAD token.

    In reality, text is tokenized by Helium tokenizer and aligned
    to audio frames using forced alignment or ASR output.
    """
    words = text.split()
    # Estimate: each word occupies ~1-2 tokens
    num_text_tokens = min(len(words) * 2, num_frames)

    text_stream = [0] * num_frames  # 0 = PAD token

    if num_text_tokens > 0:
        # Spread text tokens roughly evenly across frames
        step = max(1, num_frames // num_text_tokens)
        for i in range(num_text_tokens):
            frame_idx = min(i * step, num_frames - 1)
            text_stream[frame_idx] = random.randint(1, vocab_size - 1)

    return text_stream


def visualize_single_sample(transcript: dict):
    """
    Print a detailed visualization of a single training sample's token structure.
    """
    print("=" * 80)
    print("PERSONAPLEX TRAINING SAMPLE VISUALIZATION")
    print("=" * 80)

    # Header info
    print(f"\nScenario ID: {transcript.get('scenario_id', 'unknown')}")
    print(f"Category: {transcript.get('category', 'unknown')}")
    print(f"Language: {transcript.get('language', 'vi')}")
    print(f"Total turns: {transcript['metadata']['total_turns']}")
    print(f"Est. duration: {transcript['metadata']['total_duration_estimate_ms'] / 1000:.1f}s")

    # Text prompts
    print(f"\n{'─' * 80}")
    print("CONDITIONING: TEXT PROMPTS (max 200 tokens each)")
    print(f"{'─' * 80}")
    print(f"\nSpeaker A prompt:")
    print(f"  \"{transcript.get('text_prompt_speaker_a', 'N/A')}\"")
    print(f"\nSpeaker B prompt:")
    print(f"  \"{transcript.get('text_prompt_speaker_b', 'N/A')}\"")

    # Voice prompt
    print(f"\n{'─' * 80}")
    print("CONDITIONING: VOICE PROMPT")
    print(f"{'─' * 80}")
    print("  Voice prompt is a PyTorch tensor (.pt file)")
    print("  Shape: [embed_dim] extracted from reference audio by Mimi encoder")
    print("  Example: voice_prompt = torch.load('assets/voices/NATF0.pt')")

    # Token-level view
    print(f"\n{'─' * 80}")
    print("CONVERSATION: TOKEN-LEVEL VIEW")
    print(f"{'─' * 80}")
    print(f"\nTensor shape: codes[K=9, T]")
    print(f"  K=9 streams: 1 text + 8 audio codebooks")
    print(f"  T = total frames at 12.5 Hz\n")

    total_frames = 0
    all_turn_tokens = []

    for turn in transcript["transcript"]:
        duration = turn["duration_estimate_ms"]
        mimi = simulate_mimi_tokens(turn["text"], duration)
        text_tokens = simulate_text_tokens(turn["text"], mimi["num_frames"])

        all_turn_tokens.append({
            "turn": turn["turn"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "emotion": turn["emotion"],
            "num_frames": mimi["num_frames"],
            "duration_ms": duration,
            "audio_tokens": mimi["tokens"],
            "text_tokens": text_tokens,
        })
        total_frames += mimi["num_frames"]
        # Add pause frames between turns (200-600ms = 3-8 frames)
        pause_frames = random.randint(3, 8)
        total_frames += pause_frames

    print(f"Total conversation frames: {total_frames} ({total_frames * 80 / 1000:.1f}s)")
    print()

    # Show first few turns in detail
    for tt in all_turn_tokens[:3]:
        print(f"  Turn {tt['turn']} - Speaker {tt['speaker']} ({tt['emotion']})")
        print(f"  Text: \"{tt['text'][:70]}{'...' if len(tt['text']) > 70 else ''}\"")
        print(f"  Frames: {tt['num_frames']} ({tt['duration_ms']}ms)")
        print()

        # Show first 8 frames of this turn
        show_frames = min(8, tt["num_frames"])
        print(f"  First {show_frames} frames (of {tt['num_frames']}):")
        print(f"  {'Frame':>8} | {'Text':>6} | {'CB1':>5} | {'CB2':>5} | {'CB3':>5} | {'...':>5} | {'CB8':>5}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")

        for f in range(show_frames):
            txt = tt["text_tokens"][f]
            cb1 = tt["audio_tokens"]["codebook_1"][f]
            cb2 = tt["audio_tokens"]["codebook_2"][f]
            cb3 = tt["audio_tokens"]["codebook_3"][f]
            cb8 = tt["audio_tokens"]["codebook_8"][f]
            txt_display = f"{txt:>5}" if txt > 0 else "  PAD"
            print(f"  {f:>8} | {txt_display} | {cb1:>5} | {cb2:>5} | {cb3:>5} | {'...':>5} | {cb8:>5}")

        print()

    if len(all_turn_tokens) > 3:
        print(f"  ... ({len(all_turn_tokens) - 3} more turns)")

    # Full tensor summary
    print(f"\n{'─' * 80}")
    print("FINAL TENSOR FORMAT")
    print(f"{'─' * 80}")
    print(f"""
  codes = torch.zeros(9, {total_frames}, dtype=torch.long)

  codes[0, :]   = text_tokens          # {total_frames} text token IDs (mostly PAD=0)
  codes[1, :]   = audio_codebook_1     # {total_frames} tokens in [0, 2047]
  codes[2, :]   = audio_codebook_2     # {total_frames} tokens in [0, 2047]
  codes[3, :]   = audio_codebook_3     # {total_frames} tokens in [0, 2047]
  codes[4, :]   = audio_codebook_4     # {total_frames} tokens in [0, 2047]
  codes[5, :]   = audio_codebook_5     # {total_frames} tokens in [0, 2047]
  codes[6, :]   = audio_codebook_6     # {total_frames} tokens in [0, 2047]
  codes[7, :]   = audio_codebook_7     # {total_frames} tokens in [0, 2047]
  codes[8, :]   = audio_codebook_8     # {total_frames} tokens in [0, 2047]

  # Additional inputs:
  voice_prompt  = torch.load("voice.pt")  # [embed_dim] tensor
  text_prompt   = tokenizer.encode("Bạn là...")  # list of token IDs, max 200
  user_codes    = torch.zeros(8, {total_frames})  # other speaker's audio tokens

  # For training:
  #   model input  = voice_prompt + text_prompt + user_codes
  #   model target = codes (predict text + audio tokens autoregressively)
""")

    # Training configuration reference
    print(f"{'─' * 80}")
    print("TRAINING CONFIGURATION REFERENCE")
    print(f"{'─' * 80}")
    print(f"""
  Model:           PersonaPlex-7B (fine-tuned Moshi)
  Base weights:    nvidia/personaplex-7b-v1 (or kyutai/moshiko-pytorch-bf16)
  Audio codec:     Mimi (8 codebooks, 2048 entries each, 12.5 Hz)
  Text vocab:      32,000 tokens (Helium tokenizer)
  Max text prompt: 200 tokens
  Audio sample rate: 24,000 Hz
  Frame size:      1,920 samples (80ms)
  Temporal Transformer: 32 layers, 4096 dim, 32 heads
  Depth Transformer:    6 layers, 1024 dim
""")


def generate_example_and_visualize():
    """Generate and visualize a hardcoded example."""
    example_transcript = {
        "scenario_id": "example_cs_demo",
        "category": "customer_service",
        "language": "vi",
        "transcript": [
            {
                "turn": 1,
                "speaker": "A",
                "text": "Dạ xin chào anh/chị, em là Linh từ TechStore Việt Nam ạ. Em có thể giúp gì cho anh/chị ạ?",
                "emotion": "friendly",
                "duration_estimate_ms": 4500,
                "has_backchannel": True,
                "backchannel_word": "dạ",
            },
            {
                "turn": 2,
                "speaker": "B",
                "text": "Ừm, chào em. Anh gọi vì anh đặt tai nghe SoundMax Pro màu đen nhưng nhận được màu trắng.",
                "emotion": "disappointed",
                "duration_estimate_ms": 5000,
                "has_backchannel": True,
                "backchannel_word": "ừm",
            },
            {
                "turn": 3,
                "speaker": "A",
                "text": "Dạ, em rất xin lỗi về sự bất tiện này ạ. Anh có thể cho em xin mã đơn hàng không ạ?",
                "emotion": "empathetic",
                "duration_estimate_ms": 4000,
                "has_backchannel": True,
                "backchannel_word": "dạ",
            },
            {
                "turn": 4,
                "speaker": "B",
                "text": "Ờ, để anh tìm... mã đơn là DH-2026-03-12345.",
                "emotion": "neutral",
                "duration_estimate_ms": 3000,
                "has_backchannel": True,
                "backchannel_word": "ờ",
            },
            {
                "turn": 5,
                "speaker": "A",
                "text": "Dạ vâng, em tìm thấy đơn rồi ạ. Đúng là giao nhầm màu. Em có hai phương án: đổi hàng miễn phí hoặc hoàn tiền đầy đủ ạ.",
                "emotion": "helpful",
                "duration_estimate_ms": 5500,
                "has_backchannel": True,
                "backchannel_word": "dạ",
            },
            {
                "turn": 6,
                "speaker": "B",
                "text": "À, vậy anh chọn đổi hàng đi em. Cảm ơn em nhiều nha.",
                "emotion": "relieved",
                "duration_estimate_ms": 3000,
                "has_backchannel": True,
                "backchannel_word": "à",
            },
        ],
        "text_prompt_speaker_a": "Bạn là nhân viên chăm sóc khách hàng tại TechStore Việt Nam. Bạn kiên nhẫn, đồng cảm và chuyên nghiệp. Bạn hỗ trợ khách hàng về sản phẩm điện tử.",
        "text_prompt_speaker_b": "Bạn là khách hàng đã đặt tai nghe Bluetooth SoundMax Pro màu đen nhưng nhận được màu trắng. Bạn thất vọng nhưng lịch sự.",
        "metadata": {
            "total_turns": 6,
            "total_duration_estimate_ms": 25000,
            "speaker_a_turns": 3,
            "speaker_b_turns": 3,
            "generation_timestamp": "2026-03-02T10:00:00",
        },
    }
    visualize_single_sample(example_transcript)


def main():
    parser = argparse.ArgumentParser(description="Visualize PersonaPlex training data format")
    parser.add_argument("--transcript", type=str, help="Path to transcript JSON file")
    parser.add_argument("--example", action="store_true", help="Show built-in example")

    args = parser.parse_args()

    if args.example or not args.transcript:
        generate_example_and_visualize()
    else:
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        visualize_single_sample(transcript)


if __name__ == "__main__":
    main()
