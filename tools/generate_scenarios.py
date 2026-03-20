"""
Scenario Template Generator for Vietnamese PersonaPlex Training Data.

This tool generates structured scenario templates (JSON) that define the
context for Vietnamese conversation generation. These templates are then
fed to an LLM to generate conversation transcripts.

Usage:
    python tools/generate_scenarios.py --category customer_service --count 10 --output data/examples/
    python tools/generate_scenarios.py --category all --count 5 --output data/examples/
    python tools/generate_scenarios.py --list-categories
"""

import argparse
import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Vietnamese scenario building blocks
# ---------------------------------------------------------------------------

COMPANIES = [
    {"name": "TechStore Việt Nam", "industry": "Bán lẻ điện tử", "products": ["laptop", "điện thoại", "tai nghe", "máy tính bảng"]},
    {"name": "VinMart Online", "industry": "Thương mại điện tử", "products": ["thực phẩm", "đồ gia dụng", "mỹ phẩm"]},
    {"name": "FPT Telecom", "industry": "Viễn thông", "products": ["internet cáp quang", "truyền hình", "camera an ninh"]},
    {"name": "Tiki", "industry": "Thương mại điện tử", "products": ["sách", "thời trang", "điện tử", "gia dụng"]},
    {"name": "Viettel Money", "industry": "Tài chính số", "products": ["ví điện tử", "chuyển tiền", "thanh toán hóa đơn"]},
    {"name": "Be Group", "industry": "Giao thông", "products": ["xe máy", "ô tô", "giao hàng"]},
    {"name": "Grab Việt Nam", "industry": "Giao thông & Giao hàng", "products": ["xe ôm", "giao đồ ăn", "giao hàng"]},
    {"name": "VNPT", "industry": "Viễn thông", "products": ["internet", "điện thoại cố định", "dịch vụ số"]},
    {"name": "VinFast", "industry": "Ô tô điện", "products": ["xe ô tô điện", "xe máy điện", "pin"]},
    {"name": "Thế Giới Di Động", "industry": "Bán lẻ điện tử", "products": ["điện thoại", "laptop", "phụ kiện"]},
]

AGENT_PERSONALITIES = [
    ["kiên nhẫn", "đồng cảm", "chuyên nghiệp"],
    ["nhiệt tình", "nhanh nhẹn", "chu đáo"],
    ["thân thiện", "tận tâm", "giải quyết vấn đề nhanh"],
    ["lịch sự", "chi tiết", "am hiểu sản phẩm"],
    ["vui vẻ", "hỗ trợ tận tình", "kiên nhẫn"],
]

CUSTOMER_EMOTIONS = [
    "thất vọng nhưng lịch sự",
    "tức giận",
    "bối rối và cần hướng dẫn",
    "vui vẻ nhưng có thắc mắc",
    "lo lắng về đơn hàng",
    "nóng vội cần giải quyết nhanh",
    "thân thiện và kiên nhẫn",
]

CUSTOMER_SERVICE_ISSUES = [
    {"issue": "Nhận được sản phẩm sai", "subcategory": "wrong_product"},
    {"issue": "Sản phẩm bị hư hỏng khi giao", "subcategory": "damaged_product"},
    {"issue": "Giao hàng trễ hơn dự kiến", "subcategory": "late_delivery"},
    {"issue": "Muốn đổi trả sản phẩm", "subcategory": "return_exchange"},
    {"issue": "Tính sai giá hoặc phí", "subcategory": "billing_error"},
    {"issue": "Không thể kích hoạt bảo hành", "subcategory": "warranty_activation"},
    {"issue": "Cần hỗ trợ kỹ thuật sử dụng sản phẩm", "subcategory": "tech_support"},
    {"issue": "Hỏi về chương trình khuyến mãi", "subcategory": "promotion_inquiry"},
    {"issue": "Mất mật khẩu tài khoản", "subcategory": "account_recovery"},
    {"issue": "Hoàn tiền chưa nhận được", "subcategory": "refund_pending"},
]

CASUAL_TOPICS = [
    {"topic": "Du lịch", "context": "Kể về chuyến du lịch gần đây hoặc lên kế hoạch đi chơi"},
    {"topic": "Ẩm thực", "context": "Bàn về món ăn yêu thích, quán ăn mới, công thức nấu ăn"},
    {"topic": "Công nghệ", "context": "Nói về điện thoại mới, ứng dụng, xu hướng công nghệ"},
    {"topic": "Phim ảnh", "context": "Thảo luận về phim mới xem, phim sắp chiếu"},
    {"topic": "Thể thao", "context": "Bàn về giải bóng đá, tập gym, chạy bộ"},
    {"topic": "Công việc", "context": "Chia sẻ về dự án, đồng nghiệp, thay đổi công việc"},
    {"topic": "Gia đình", "context": "Nói về con cái, bố mẹ, sự kiện gia đình"},
    {"topic": "Học tập", "context": "Thảo luận về khóa học, chứng chỉ, kế hoạch học tập"},
]

ASSISTANT_DOMAINS = [
    {"domain": "Sức khỏe", "expertise": "tư vấn sức khỏe cơ bản, lịch khám, triệu chứng thường gặp"},
    {"domain": "Giáo dục", "expertise": "tư vấn chọn trường, học bổng, phương pháp học tập"},
    {"domain": "Tài chính cá nhân", "expertise": "quản lý chi tiêu, tiết kiệm, đầu tư cơ bản"},
    {"domain": "Nấu ăn", "expertise": "công thức nấu ăn, nguyên liệu, kỹ thuật nấu"},
    {"domain": "Lập trình", "expertise": "hỗ trợ code, debug, giải thích khái niệm IT"},
    {"domain": "Pháp luật cơ bản", "expertise": "quyền người tiêu dùng, hợp đồng, thủ tục hành chính"},
]

CONSULTATION_SCENARIOS = [
    {"type": "Tư vấn chọn trường đại học", "consultant_role": "Chuyên viên tư vấn giáo dục"},
    {"type": "Tư vấn mua nhà", "consultant_role": "Nhân viên tư vấn bất động sản"},
    {"type": "Tư vấn bảo hiểm", "consultant_role": "Chuyên viên bảo hiểm"},
    {"type": "Tư vấn sức khỏe", "consultant_role": "Tổng đài tư vấn sức khỏe"},
    {"type": "Tư vấn du học", "consultant_role": "Chuyên viên tư vấn du học"},
]

INTERVIEW_SCENARIOS = [
    {"type": "Phỏng vấn tuyển dụng IT", "position": "Lập trình viên Python", "company_type": "Startup công nghệ"},
    {"type": "Phỏng vấn tuyển dụng kinh doanh", "position": "Nhân viên kinh doanh", "company_type": "Công ty FMCG"},
    {"type": "Phỏng vấn báo chí", "position": "Khách mời chuyên gia", "company_type": "Đài truyền hình"},
]

# Vietnamese backchannel words and fillers for reference
VIETNAMESE_BACKCHANNELS = ["ừ", "ừm", "vâng", "dạ", "à", "ờ", "uh huh", "đúng rồi", "vậy à", "thật hả"]
VIETNAMESE_FILLERS = ["ừm", "à", "thì", "kiểu", "nói chung là", "thực ra là", "cơ bản là"]


# ---------------------------------------------------------------------------
# Scenario generators per category
# ---------------------------------------------------------------------------

def generate_customer_service_scenario() -> dict:
    company = random.choice(COMPANIES)
    issue_info = random.choice(CUSTOMER_SERVICE_ISSUES)
    product = random.choice(company["products"])
    personality = random.choice(AGENT_PERSONALITIES)
    emotion = random.choice(CUSTOMER_EMOTIONS)

    price = random.choice([
        f"{random.randint(1, 50) * 100_000:,} VNĐ".replace(",", "."),
        f"{random.randint(1, 20)},{random.randint(0, 9)}00,000 VNĐ",
    ])

    scenario = {
        "scenario_id": f"cs_{uuid.uuid4().hex[:8]}",
        "category": "customer_service",
        "subcategory": issue_info["subcategory"],
        "language": "vi",
        "agent": {
            "role": "Nhân viên chăm sóc khách hàng",
            "company": company["name"],
            "industry": company["industry"],
            "personality": personality,
            "expertise": [company["industry"], "chính sách bảo hành", "xử lý khiếu nại"],
            "constraints": [
                "Phải xác minh thông tin khách hàng trước khi xử lý",
                "Luôn xin lỗi về sự bất tiện",
                "Đề xuất ít nhất 2 giải pháp",
            ],
        },
        "customer": {
            "issue": issue_info["issue"],
            "product": product,
            "emotion": emotion,
            "goal": f"Giải quyết vấn đề: {issue_info['issue'].lower()}",
            "context": f"Khách hàng đã mua {product} từ {company['name']} với giá {price}",
        },
        "product": {
            "name": product,
            "company": company["name"],
            "price": price,
        },
        "expected_turns": [random.randint(6, 10), random.randint(12, 20)],
        "expected_duration_seconds": [random.randint(45, 90), random.randint(120, 360)],
        "backchannel_examples": random.sample(VIETNAMESE_BACKCHANNELS, 4),
        "filler_examples": random.sample(VIETNAMESE_FILLERS, 3),
    }
    return scenario


def generate_assistant_scenario() -> dict:
    domain_info = random.choice(ASSISTANT_DOMAINS)

    scenario = {
        "scenario_id": f"ast_{uuid.uuid4().hex[:8]}",
        "category": "assistant",
        "subcategory": domain_info["domain"].lower().replace(" ", "_"),
        "language": "vi",
        "agent": {
            "role": f"Trợ lý AI chuyên về {domain_info['domain']}",
            "personality": random.choice(AGENT_PERSONALITIES),
            "expertise": [domain_info["expertise"]],
            "constraints": [
                "Trả lời ngắn gọn, rõ ràng",
                "Hỏi lại khi chưa rõ yêu cầu",
                "Khuyến khích người dùng tìm chuyên gia nếu vấn đề phức tạp",
            ],
        },
        "user": {
            "need": f"Cần hỗ trợ về {domain_info['domain'].lower()}",
            "knowledge_level": random.choice(["mới bắt đầu", "trung bình", "khá hiểu biết"]),
            "emotion": random.choice(["tò mò", "lo lắng", "hào hứng", "bối rối"]),
        },
        "domain": domain_info,
        "expected_turns": [random.randint(5, 8), random.randint(10, 16)],
        "expected_duration_seconds": [random.randint(30, 60), random.randint(90, 240)],
        "backchannel_examples": random.sample(VIETNAMESE_BACKCHANNELS, 3),
    }
    return scenario


def generate_casual_scenario() -> dict:
    topic_info = random.choice(CASUAL_TOPICS)

    scenario = {
        "scenario_id": f"cas_{uuid.uuid4().hex[:8]}",
        "category": "casual_conversation",
        "subcategory": topic_info["topic"].lower().replace(" ", "_"),
        "language": "vi",
        "speaker_a": {
            "role": random.choice(["bạn thân", "đồng nghiệp", "người quen"]),
            "personality": random.choice([
                ["hoạt bát", "vui vẻ"],
                ["trầm tính", "chu đáo"],
                ["nhiệt tình", "hay nói"],
            ]),
            "relationship_to_b": "bạn",
        },
        "speaker_b": {
            "role": random.choice(["bạn thân", "đồng nghiệp", "người quen"]),
            "personality": random.choice([
                ["hoạt bát", "vui vẻ"],
                ["trầm tính", "chu đáo"],
                ["nhiệt tình", "hay nói"],
            ]),
            "relationship_to_a": "bạn",
        },
        "topic": topic_info,
        "expected_turns": [random.randint(8, 12), random.randint(15, 25)],
        "expected_duration_seconds": [random.randint(60, 120), random.randint(180, 600)],
        "backchannel_examples": random.sample(VIETNAMESE_BACKCHANNELS, 5),
        "filler_examples": random.sample(VIETNAMESE_FILLERS, 4),
        "notes": "Cuộc trò chuyện tự nhiên, không theo kịch bản. Có thể chuyển chủ đề.",
    }
    return scenario


def generate_consultation_scenario() -> dict:
    consult = random.choice(CONSULTATION_SCENARIOS)

    scenario = {
        "scenario_id": f"con_{uuid.uuid4().hex[:8]}",
        "category": "consultation",
        "subcategory": consult["type"].lower().replace(" ", "_"),
        "language": "vi",
        "consultant": {
            "role": consult["consultant_role"],
            "personality": random.choice(AGENT_PERSONALITIES),
            "expertise": [consult["type"]],
            "constraints": [
                "Tư vấn dựa trên thông tin khách hàng cung cấp",
                "Không đưa ra cam kết chắc chắn",
                "Đề xuất bước tiếp theo rõ ràng",
            ],
        },
        "client": {
            "need": consult["type"],
            "knowledge_level": random.choice(["ít hiểu biết", "biết cơ bản", "đã tìm hiểu"]),
            "emotion": random.choice(["lo lắng", "hào hứng", "thận trọng", "cởi mở"]),
        },
        "consultation_type": consult,
        "expected_turns": [random.randint(8, 12), random.randint(15, 25)],
        "expected_duration_seconds": [random.randint(120, 240), random.randint(300, 600)],
        "backchannel_examples": random.sample(VIETNAMESE_BACKCHANNELS, 4),
    }
    return scenario


def generate_interview_scenario() -> dict:
    interview = random.choice(INTERVIEW_SCENARIOS)

    scenario = {
        "scenario_id": f"int_{uuid.uuid4().hex[:8]}",
        "category": "interview",
        "subcategory": interview["type"].lower().replace(" ", "_"),
        "language": "vi",
        "interviewer": {
            "role": "Người phỏng vấn",
            "company_type": interview["company_type"],
            "personality": ["chuyên nghiệp", "thân thiện"],
            "style": random.choice(["formal", "semi_formal", "casual"]),
        },
        "interviewee": {
            "position_applied": interview["position"],
            "experience_years": random.randint(1, 10),
            "emotion": random.choice(["hồi hộp", "tự tin", "bình tĩnh"]),
        },
        "interview_info": interview,
        "expected_turns": [random.randint(10, 15), random.randint(20, 30)],
        "expected_duration_seconds": [random.randint(300, 600), random.randint(600, 1200)],
        "backchannel_examples": random.sample(VIETNAMESE_BACKCHANNELS, 3),
    }
    return scenario


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CATEGORY_GENERATORS = {
    "customer_service": generate_customer_service_scenario,
    "assistant": generate_assistant_scenario,
    "casual_conversation": generate_casual_scenario,
    "consultation": generate_consultation_scenario,
    "interview": generate_interview_scenario,
}


def generate_text_prompts(scenario: dict) -> dict:
    """
    Generate PersonaPlex-style text prompts from a scenario.
    Returns a dict with text_prompt_speaker_a and text_prompt_speaker_b.
    """
    cat = scenario["category"]

    if cat == "customer_service":
        agent = scenario["agent"]
        customer = scenario["customer"]
        prompt_a = (
            f"Bạn là {agent['role']} tại {agent['company']}. "
            f"Bạn {', '.join(agent['personality'])}. "
            f"Bạn hỗ trợ khách hàng về {agent['industry']}. "
            f"{' '.join(agent['constraints'])}."
        )
        prompt_b = (
            f"Bạn là khách hàng gọi đến {agent['company']}. "
            f"Vấn đề của bạn: {customer['issue']}. "
            f"Bạn {customer['emotion']}. "
            f"Mục tiêu: {customer['goal']}."
        )
    elif cat == "assistant":
        agent = scenario["agent"]
        user = scenario["user"]
        prompt_a = (
            f"Bạn là {agent['role']}. "
            f"Bạn {', '.join(agent['personality'])}. "
            f"Chuyên môn: {', '.join(agent['expertise'])}. "
            f"{' '.join(agent['constraints'])}."
        )
        prompt_b = (
            f"Bạn là người dùng cần hỗ trợ về {scenario['domain']['domain'].lower()}. "
            f"Trình độ hiểu biết: {user['knowledge_level']}. "
            f"Bạn đang cảm thấy {user['emotion']}."
        )
    elif cat == "casual_conversation":
        sa = scenario["speaker_a"]
        sb = scenario["speaker_b"]
        topic = scenario["topic"]
        prompt_a = (
            f"Bạn đang nói chuyện điện thoại với {sa['relationship_to_b']}. "
            f"Tính cách: {', '.join(sa['personality'])}. "
            f"Chủ đề: {topic['topic']} - {topic['context']}. "
            f"Nói chuyện tự nhiên, thoải mái."
        )
        prompt_b = (
            f"Bạn đang nói chuyện điện thoại với {sb['relationship_to_a']}. "
            f"Tính cách: {', '.join(sb['personality'])}. "
            f"Chủ đề: {topic['topic']} - {topic['context']}. "
            f"Nói chuyện tự nhiên, thoải mái."
        )
    elif cat == "consultation":
        cons = scenario["consultant"]
        client = scenario["client"]
        prompt_a = (
            f"Bạn là {cons['role']}. "
            f"Bạn {', '.join(cons['personality'])}. "
            f"Chuyên tư vấn: {', '.join(cons['expertise'])}. "
            f"{' '.join(cons['constraints'])}."
        )
        prompt_b = (
            f"Bạn là khách hàng cần {client['need'].lower()}. "
            f"Trình độ hiểu biết: {client['knowledge_level']}. "
            f"Bạn đang {client['emotion']}."
        )
    elif cat == "interview":
        interviewer = scenario["interviewer"]
        interviewee = scenario["interviewee"]
        prompt_a = (
            f"Bạn là người phỏng vấn tại {interviewer['company_type']}. "
            f"Phong cách: {interviewer['style']}. "
            f"Đang phỏng vấn ứng viên cho vị trí {interviewee['position_applied']}."
        )
        prompt_b = (
            f"Bạn là ứng viên phỏng vấn cho vị trí {interviewee['position_applied']}. "
            f"Kinh nghiệm: {interviewee['experience_years']} năm. "
            f"Bạn đang cảm thấy {interviewee['emotion']}."
        )
    else:
        prompt_a = "Bạn là người tham gia cuộc trò chuyện."
        prompt_b = "Bạn là người tham gia cuộc trò chuyện."

    return {
        "text_prompt_speaker_a": prompt_a,
        "text_prompt_speaker_b": prompt_b,
    }


def generate_llm_prompt(scenario: dict) -> str:
    """
    Generate the LLM prompt that will be used to create the conversation transcript.
    This prompt is sent to Qwen, GPT, or any capable Vietnamese LLM.
    """
    prompts = generate_text_prompts(scenario)
    min_turns, max_turns = scenario["expected_turns"]

    cat = scenario["category"]

    if cat == "customer_service":
        context_block = f"""Ngữ cảnh:
- Công ty: {scenario['agent']['company']} ({scenario['agent']['industry']})
- Sản phẩm: {scenario['customer'].get('product', 'N/A')}
- Vấn đề: {scenario['customer']['issue']}
- Cảm xúc khách hàng: {scenario['customer']['emotion']}
- Mục tiêu: {scenario['customer']['goal']}"""
        speakers = "NHÂN VIÊN và KHÁCH HÀNG"
    elif cat == "assistant":
        context_block = f"""Ngữ cảnh:
- Lĩnh vực: {scenario['domain']['domain']}
- Chuyên môn: {scenario['domain']['expertise']}
- Trình độ người dùng: {scenario['user']['knowledge_level']}"""
        speakers = "TRỢ LÝ và NGƯỜI DÙNG"
    elif cat == "casual_conversation":
        context_block = f"""Ngữ cảnh:
- Chủ đề: {scenario['topic']['topic']}
- Bối cảnh: {scenario['topic']['context']}
- Mối quan hệ: Bạn bè/người quen"""
        speakers = "NGƯỜI A và NGƯỜI B"
    elif cat == "consultation":
        context_block = f"""Ngữ cảnh:
- Loại tư vấn: {scenario['consultation_type']['type']}
- Vai trò tư vấn viên: {scenario['consultation_type']['consultant_role']}
- Trình độ khách hàng: {scenario['client']['knowledge_level']}"""
        speakers = "TƯ VẤN VIÊN và KHÁCH HÀNG"
    elif cat == "interview":
        context_block = f"""Ngữ cảnh:
- Loại phỏng vấn: {scenario['interview_info']['type']}
- Vị trí: {scenario['interview_info']['position']}
- Kinh nghiệm ứng viên: {scenario['interviewee']['experience_years']} năm"""
        speakers = "NGƯỜI PHỎNG VẤN và ỨNG VIÊN"
    else:
        context_block = "Ngữ cảnh: Cuộc trò chuyện chung"
        speakers = "NGƯỜI A và NGƯỜI B"

    backchannels = scenario.get("backchannel_examples", VIETNAMESE_BACKCHANNELS[:4])
    fillers = scenario.get("filler_examples", VIETNAMESE_FILLERS[:3])

    prompt = f"""Bạn là trình tạo hội thoại tiếng Việt tự nhiên cho huấn luyện mô hình AI hội thoại.

{context_block}

Vai trò người nói A:
{prompts['text_prompt_speaker_a']}

Vai trò người nói B:
{prompts['text_prompt_speaker_b']}

Yêu cầu:
1. Tạo cuộc hội thoại tự nhiên bằng tiếng Việt, {min_turns}-{max_turns} lượt nói
2. Sử dụng ngôn ngữ giao tiếp đời thường, KHÔNG quá trang trọng
3. BẮT BUỘC sử dụng các từ đệm tự nhiên: {', '.join(backchannels)}
4. Thêm các từ lấp đầy: {', '.join(fillers)}
5. Đôi khi có câu bỏ lửng, nói lắp nhẹ, hoặc tự sửa lời
6. Thể hiện cảm xúc phù hợp với ngữ cảnh
7. Mỗi lượt nói 1-4 câu (có thể dài hơn nếu cần giải thích)

Định dạng OUTPUT (JSON):
{{
  "turns": [
    {{
      "speaker": "A",
      "text": "<nội dung lượt nói>",
      "emotion": "<cảm xúc>",
      "has_backchannel": false,
      "backchannel_word": null
    }},
    ...
  ]
}}

Lưu ý: Cuộc hội thoại giữa {speakers}. Bắt đầu ngay."""

    return prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese scenario templates for PersonaPlex training data"
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_GENERATORS.keys()) + ["all"],
        default="all",
        help="Scenario category to generate",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of scenarios per category",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/examples/",
        help="Output directory for generated scenarios",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available scenario categories and exit",
    )
    parser.add_argument(
        "--include-llm-prompt",
        action="store_true",
        default=True,
        help="Include the LLM prompt for transcript generation in output",
    )

    args = parser.parse_args()

    if args.list_categories:
        print("Available scenario categories:")
        for name in CATEGORY_GENERATORS:
            print(f"  - {name}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = list(CATEGORY_GENERATORS.keys()) if args.category == "all" else [args.category]

    all_scenarios = []

    for cat in categories:
        generator = CATEGORY_GENERATORS[cat]
        print(f"\nGenerating {args.count} '{cat}' scenarios...")

        for i in range(args.count):
            scenario = generator()
            prompts = generate_text_prompts(scenario)
            scenario["text_prompts"] = prompts

            if args.include_llm_prompt:
                scenario["llm_generation_prompt"] = generate_llm_prompt(scenario)

            all_scenarios.append(scenario)
            print(f"  [{i+1}/{args.count}] {scenario['scenario_id']}")

    # Save all scenarios to a single file
    output_file = output_dir / f"scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_scenarios)} scenarios to: {output_file}")

    # Also save individual scenario files for easy inspection
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    for scenario in all_scenarios:
        fname = individual_dir / f"{scenario['scenario_id']}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(scenario, f, ensure_ascii=False, indent=2)

    print(f"Individual scenario files saved to: {individual_dir}/")

    # Print a summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cat in categories:
        count = sum(1 for s in all_scenarios if s["category"] == cat)
        print(f"  {cat}: {count} scenarios")
    print(f"  Total: {len(all_scenarios)} scenarios")
    print(f"\nNext step: Feed the 'llm_generation_prompt' from each scenario")
    print(f"to a Vietnamese-capable LLM (Qwen3, GPT-4, Vistral, etc.)")
    print(f"to generate conversation transcripts.")


if __name__ == "__main__":
    main()
