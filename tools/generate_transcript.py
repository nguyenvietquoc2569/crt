"""
Transcript Generator for Vietnamese PersonaPlex Training Data.

This tool takes scenario templates and generates conversation transcripts
using an LLM API (OpenAI-compatible). The output follows the exact format
needed for the downstream audio synthesis and tokenization pipeline.

Usage:
    # With OpenAI-compatible API (e.g., local Ollama, vLLM, or OpenAI)
    python tools/generate_transcript.py \
        --scenario data/examples/individual/cs_abc12345.json \
        --api-base http://localhost:11434/v1 \
        --model qwen3-32b \
        --output data/processed/transcripts/

    # Batch mode
    python tools/generate_transcript.py \
        --scenario-dir data/examples/individual/ \
        --api-base http://localhost:11434/v1 \
        --model qwen3-32b \
        --output data/processed/transcripts/

    # Dry run (just print the LLM prompt, don't call API)
    python tools/generate_transcript.py \
        --scenario data/examples/individual/cs_abc12345.json \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str, scenario: dict) -> dict | None:
    """
    Parse the LLM's response into a structured transcript.
    Handles both JSON output and plain text with speaker labels.
    """
    # Try JSON parsing first
    try:
        # Find JSON block in response
        json_match = re.search(r'\{[\s\S]*"turns"[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            if "turns" in data and len(data["turns"]) > 0:
                return _build_transcript(data["turns"], scenario)
    except json.JSONDecodeError:
        pass

    # Fallback: parse speaker-labeled text
    turns = []
    # Match patterns like "A:", "NHÂN VIÊN:", "KHÁCH HÀNG:", etc.
    pattern = r'(?:^|\n)\s*(?:(?:NHÂN VIÊN|KHÁCH HÀNG|TRỢ LÝ|NGƯỜI DÙNG|NGƯỜI A|NGƯỜI B|TƯ VẤN VIÊN|NGƯỜI PHỎNG VẤN|ỨNG VIÊN|Speaker\s*[AB]|[AB])\s*[:\-])\s*(.+?)(?=\n\s*(?:NHÂN VIÊN|KHÁCH HÀNG|TRỢ LÝ|NGƯỜI DÙNG|NGƯỜI A|NGƯỜI B|TƯ VẤN VIÊN|NGƯỜI PHỎNG VẤN|ỨNG VIÊN|Speaker\s*[AB]|[AB])\s*[:\-]|\Z)'
    matches = re.findall(
        r'(NHÂN VIÊN|KHÁCH HÀNG|TRỢ LÝ|NGƯỜI DÙNG|NGƯỜI A|NGƯỜI B|TƯ VẤN VIÊN|NGƯỜI PHỎNG VẤN|ỨNG VIÊN|Speaker\s*[AB]|[AB])\s*[:\-]\s*(.+?)(?=\n\s*(?:NHÂN VIÊN|KHÁCH HÀNG|TRỢ LÝ|NGƯỜI DÙNG|NGƯỜI A|NGƯỜI B|TƯ VẤN VIÊN|NGƯỜI PHỎNG VẤN|ỨNG VIÊN|Speaker\s*[AB]|[AB])\s*[:\-]|\Z)',
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    if not matches:
        return None

    speaker_map_first = None
    for label, text in matches:
        label = label.strip().upper()
        # Map to A/B
        if label in ("NHÂN VIÊN", "TRỢ LÝ", "TƯ VẤN VIÊN", "NGƯỜI PHỎNG VẤN", "NGƯỜI A", "SPEAKER A", "A"):
            speaker = "A"
        else:
            speaker = "B"

        turns.append({
            "speaker": speaker,
            "text": text.strip(),
            "emotion": "neutral",
            "has_backchannel": False,
            "backchannel_word": None,
        })

    if turns:
        return _build_transcript(turns, scenario)

    return None


def _build_transcript(turns: list, scenario: dict) -> dict:
    """Build the complete transcript structure from parsed turns."""
    prompts = scenario.get("text_prompts", {})

    # Estimate duration for each turn (rough: ~150ms per Vietnamese syllable)
    for turn in turns:
        words = turn["text"].split()
        # Vietnamese words are mostly monosyllabic
        estimated_syllables = len(words) * 1.2
        turn["duration_estimate_ms"] = int(estimated_syllables * 150)

    total_duration = sum(t["duration_estimate_ms"] for t in turns)
    # Add pauses between turns (~400ms average)
    total_duration += len(turns) * 400

    transcript = {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "language": "vi",
        "transcript": [
            {
                "turn": i + 1,
                "speaker": t["speaker"],
                "text": t["text"],
                "emotion": t.get("emotion", "neutral"),
                "duration_estimate_ms": t["duration_estimate_ms"],
                "has_backchannel": t.get("has_backchannel", False),
                "backchannel_word": t.get("backchannel_word"),
            }
            for i, t in enumerate(turns)
        ],
        "text_prompt_speaker_a": prompts.get("text_prompt_speaker_a", ""),
        "text_prompt_speaker_b": prompts.get("text_prompt_speaker_b", ""),
        "metadata": {
            "total_turns": len(turns),
            "total_duration_estimate_ms": total_duration,
            "speaker_a_turns": sum(1 for t in turns if t["speaker"] == "A"),
            "speaker_b_turns": sum(1 for t in turns if t["speaker"] == "B"),
            "generation_timestamp": datetime.now().isoformat(),
        },
    }
    return transcript


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def generate_with_llm(
    scenario: dict,
    api_base: str,
    model: str,
    api_key: str = "not-needed",
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """Call LLM API to generate a conversation transcript."""
    if not HAS_OPENAI:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        )

    client = OpenAI(base_url=api_base, api_key=api_key)

    llm_prompt = scenario.get("llm_generation_prompt", "")
    if not llm_prompt:
        raise ValueError("Scenario missing 'llm_generation_prompt'. "
                         "Regenerate with generate_scenarios.py --include-llm-prompt")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý tạo dữ liệu huấn luyện hội thoại tiếng Việt. Luôn trả lời bằng JSON format."},
            {"role": "user", "content": llm_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Example transcript generator (no LLM needed)
# ---------------------------------------------------------------------------

def generate_example_transcript(scenario: dict) -> dict:
    """
    Generate a hardcoded example transcript for demonstration.
    Use this to understand the data format without needing an LLM.
    """
    cat = scenario["category"]
    prompts = scenario.get("text_prompts", {})

    if cat == "customer_service":
        turns = [
            {"speaker": "A", "text": "Dạ xin chào anh/chị, em là Linh từ bộ phận chăm sóc khách hàng ạ. Em có thể giúp gì cho anh/chị ạ?", "emotion": "friendly"},
            {"speaker": "B", "text": "Ừm, chào em. Anh gọi vì anh đặt hàng tuần trước nhưng nhận được sản phẩm sai màu. Anh hơi thất vọng vì đã chờ mấy ngày rồi.", "emotion": "disappointed"},
            {"speaker": "A", "text": "Dạ, em rất xin lỗi về sự bất tiện này ạ. Anh có thể cho em xin mã đơn hàng để em kiểm tra không ạ?", "emotion": "empathetic"},
            {"speaker": "B", "text": "Ờ, để anh tìm... mã đơn là DH-2026-03-12345 nhé.", "emotion": "neutral"},
            {"speaker": "A", "text": "Dạ vâng, em đã tìm thấy đơn hàng của anh rồi ạ. Đúng là hệ thống ghi nhận anh đặt màu đen nhưng giao nhầm màu trắng. Em xin lỗi anh rất nhiều ạ.", "emotion": "apologetic"},
            {"speaker": "B", "text": "Ừ, vậy giờ giải quyết sao em?", "emotion": "expectant"},
            {"speaker": "A", "text": "Dạ, em có hai phương án cho anh ạ. Một là em gửi đổi sản phẩm đúng màu cho anh, miễn phí vận chuyển, giao trong hai ngày. Hai là em hoàn tiền đầy đủ cho anh ạ. Anh muốn chọn phương án nào ạ?", "emotion": "helpful"},
            {"speaker": "B", "text": "À, vậy anh chọn đổi sản phẩm đi em. Nhưng lần này phải giao đúng màu đen nha.", "emotion": "relieved"},
            {"speaker": "A", "text": "Dạ chắc chắn rồi ạ. Em đã tạo đơn đổi hàng cho anh. Mã đổi hàng là EX-2026-67890. Anh sẽ nhận được hàng trong hai ngày làm việc ạ.", "emotion": "confident"},
            {"speaker": "B", "text": "Ok, cảm ơn em nhiều nha. Hy vọng lần này ổn.", "emotion": "satisfied"},
            {"speaker": "A", "text": "Dạ vâng, em cảm ơn anh đã liên hệ ạ. Chúc anh một ngày tốt lành! Nếu cần hỗ trợ thêm, anh cứ gọi lại bất cứ lúc nào ạ.", "emotion": "friendly"},
            {"speaker": "B", "text": "Ừ, bye em.", "emotion": "neutral"},
        ]
    elif cat == "casual_conversation":
        turns = [
            {"speaker": "A", "text": "Ê, lâu quá không nói chuyện! Dạo này sao rồi?", "emotion": "excited"},
            {"speaker": "B", "text": "Ừ, bận quá trời luôn á. Mà nè, tao mới đi Đà Lạt về nè, đẹp cực!", "emotion": "enthusiastic"},
            {"speaker": "A", "text": "Thật hả? Trời, tao cũng đang muốn đi Đà Lạt. Đi mấy ngày? Chỗ nào hay?", "emotion": "curious"},
            {"speaker": "B", "text": "Ừm, tao đi bốn ngày ba đêm. Tao ở homestay trên đồi, sáng ra thấy sương mù phủ khắp, đẹp lắm. Rồi tao đi chợ đêm, ăn bánh tráng nướng, uống sữa đậu nành nóng...", "emotion": "nostalgic"},
            {"speaker": "A", "text": "Nghe thèm quá! Mà đi mấy người?", "emotion": "interested"},
            {"speaker": "B", "text": "Đi nhóm bốn người, hai cặp. Thuê xe máy chạy vòng vòng. À mà có chỗ mới nè, cánh đồng cỏ hồng, chụp hình đẹp lắm.", "emotion": "happy"},
            {"speaker": "A", "text": "Ờ, tao thấy trên mạng rồi. Ok, chắc tháng sau tao plan đi. Mày gửi tao lịch trình với địa chỉ homestay đi.", "emotion": "planning"},
            {"speaker": "B", "text": "Ok ok, để tao gửi. Mà mày nhớ book sớm nha, cuối tuần đông lắm.", "emotion": "advising"},
        ]
    elif cat == "assistant":
        turns = [
            {"speaker": "A", "text": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?", "emotion": "friendly"},
            {"speaker": "B", "text": "Ừm, tôi muốn hỏi về... kiểu là tôi muốn bắt đầu tập thể dục nhưng không biết bắt đầu từ đâu.", "emotion": "confused"},
            {"speaker": "A", "text": "Dạ, câu hỏi rất hay ạ! Trước tiên, bạn có thể cho tôi biết tình trạng sức khỏe hiện tại không? Ví dụ bạn có bệnh nền gì không, và mục tiêu của bạn là gì?", "emotion": "helpful"},
            {"speaker": "B", "text": "À, tôi khỏe bình thường, không có bệnh gì. Mục tiêu là giảm cân, tôi nặng khoảng 75 ký, cao 1m65.", "emotion": "honest"},
            {"speaker": "A", "text": "Vâng, với chỉ số BMI của bạn khoảng 27.5, tôi khuyên bạn nên kết hợp cardio và tập tạ nhẹ. Bắt đầu với đi bộ nhanh 30 phút mỗi ngày, rồi tăng dần lên. Bạn muốn tôi gợi ý lịch tập cụ thể không?", "emotion": "professional"},
            {"speaker": "B", "text": "Ồ, được luôn! Gợi ý cho tôi đi.", "emotion": "eager"},
        ]
    else:
        turns = [
            {"speaker": "A", "text": "Xin chào, cảm ơn bạn đã đến. Bắt đầu nhé?", "emotion": "friendly"},
            {"speaker": "B", "text": "Dạ vâng, em sẵn sàng rồi ạ.", "emotion": "nervous"},
            {"speaker": "A", "text": "Bạn có thể giới thiệu về bản thân và kinh nghiệm liên quan không?", "emotion": "professional"},
            {"speaker": "B", "text": "Dạ, em tên là Minh, em đã có ba năm kinh nghiệm trong lĩnh vực này...", "emotion": "confident"},
        ]

    # Add metadata
    for i, t in enumerate(turns):
        words = t["text"].split()
        t["duration_estimate_ms"] = int(len(words) * 1.2 * 150)
        t["has_backchannel"] = any(bc in t["text"].lower() for bc in ["ừ", "ừm", "vâng", "dạ", "à", "ờ"])
        t["backchannel_word"] = None
        if t["has_backchannel"]:
            for bc in ["ừm", "ừ", "vâng", "dạ", "à", "ờ"]:
                if bc in t["text"].lower():
                    t["backchannel_word"] = bc
                    break

    return _build_transcript(turns, scenario)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese conversation transcripts from scenario templates"
    )
    parser.add_argument("--scenario", type=str, help="Path to a single scenario JSON file")
    parser.add_argument("--scenario-dir", type=str, help="Directory of scenario JSON files (batch mode)")
    parser.add_argument("--output", type=str, default="data/processed/transcripts/", help="Output directory")
    parser.add_argument("--api-base", type=str, default="http://localhost:11434/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", type=str, default="not-needed", help="API key")
    parser.add_argument("--model", type=str, default="qwen3-32b", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--dry-run", action="store_true", help="Print LLM prompt without calling API")
    parser.add_argument("--example-mode", action="store_true", help="Generate hardcoded examples (no LLM needed)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect scenario files
    scenario_files = []
    if args.scenario:
        scenario_files = [Path(args.scenario)]
    elif args.scenario_dir:
        scenario_files = sorted(Path(args.scenario_dir).glob("*.json"))
    else:
        parser.error("Provide --scenario or --scenario-dir")

    print(f"Processing {len(scenario_files)} scenario(s)...")

    for sf in scenario_files:
        with open(sf, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        scenario_id = scenario.get("scenario_id", sf.stem)
        print(f"\n{'='*50}")
        print(f"Processing: {scenario_id} ({scenario['category']})")

        if args.dry_run:
            print(f"\n--- LLM Prompt ---")
            print(scenario.get("llm_generation_prompt", "No prompt found"))
            print(f"--- End Prompt ---\n")
            continue

        if args.example_mode:
            transcript = generate_example_transcript(scenario)
        else:
            try:
                raw_response = generate_with_llm(
                    scenario=scenario,
                    api_base=args.api_base,
                    model=args.model,
                    api_key=args.api_key,
                    temperature=args.temperature,
                )
                transcript = parse_llm_response(raw_response, scenario)
                if transcript is None:
                    print(f"  WARNING: Could not parse LLM response for {scenario_id}")
                    print(f"  Raw response: {raw_response[:200]}...")
                    continue
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        # Save transcript
        output_file = output_dir / f"transcript_{scenario_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {output_file}")
        print(f"  Turns: {transcript['metadata']['total_turns']}")
        print(f"  Duration: ~{transcript['metadata']['total_duration_estimate_ms'] / 1000:.1f}s")

        if not args.example_mode and args.delay > 0:
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print("Done!")


if __name__ == "__main__":
    main()
