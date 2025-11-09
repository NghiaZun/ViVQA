"""
Prompt utilities for teacher reasoning (Day 1)
Author: Nghĩa Duong (refined)
"""

# ===========================
# System Prompt
# ===========================
SYSTEM_PROMPT = (
    "Bạn là mô hình Visual Question Answering tiếng Việt. "
    "Trả lời NGẮN, RÕ và luôn theo đúng chuẩn định dạng XML sau:\n\n"
    "<answer>Câu trả lời ngắn</answer>\n"
    "<reasoning>[LOẠI_REASONING] 1-2 câu giải thích</reasoning>\n\n"
    "Trong đó LOẠI_REASONING ∈ {DESCRIPTIVE, CAUSAL, SPATIAL, COUNTING, "
    "OBJECT, COMMONSENSE, INTENT}.\n"
)

# ===========================
# Few-shot Examples (đã sửa về XML format)
# ===========================
FEW_SHOTS = [
    {
        "question": "Màu của chiếc bình là gì?",
        "answer": (
            "<answer>xanh lá</answer>\n"
            "<reasoning>[DESCRIPTIVE] Chiếc bình trong ảnh có màu xanh lá đặc trưng.</reasoning>"
        )
    },
    {
        "question": "Cô gái đang ngồi ở đâu?",
        "answer": (
            "<answer>trên giường</answer>\n"
            "<reasoning>[SPATIAL] Cô gái ngồi trên bề mặt có chăn gối, đặc trưng của giường.</reasoning>"
        )
    },
    {
        "question": "Có bao nhiêu con chim đang đậu trên cành cây?",
        "answer": (
            "<answer>hai</answer>\n"
            "<reasoning>[COUNTING] Có hai con chim đứng cạnh nhau và không có con nào khác.</reasoning>"
        )
    },
    {
        "question": "Tại sao người đàn ông đội mũ bảo hiểm?",
        "answer": (
            "<answer>để bảo vệ đầu</answer>\n"
            "<reasoning>[COMMONSENSE] Khi lái xe máy, người ta đội mũ bảo hiểm để tránh chấn thương đầu.</reasoning>"
        )
    },
    {
        "question": "Tại sao mặt đất lại ướt?",
        "answer": (
            "<answer>vì trời mưa</answer>\n"
            "<reasoning>[CAUSAL] Có vệt nước và giọt mưa trong ảnh cho thấy trời đang mưa.</reasoning>"
        )
    },
    {
        "question": "Trong hai chiếc xe, chiếc nào lớn hơn?",
        "answer": (
            "<answer>chiếc xe tải</answer>\n"
            "<reasoning>[OBJECT] Xe tải có kích thước lớn hơn rõ rệt so với xe còn lại.</reasoning>"
        )
    },
]

# ===========================
# Build few-shot prompt
# ===========================
def build_fewshot_prompt(question: str) -> str:
    """Tạo few-shot prompt đúng định dạng XML của mô hình."""
    examples = []
    for ex in FEW_SHOTS:
        examples.append(
            f"Q: {ex['question']}\n{ex['answer']}"
        )

    fewshot_text = "\n\n".join(examples)

    prompt = (
        "Dưới đây là các ví dụ minh họa cách mô hình trả lời câu hỏi VQA bằng tiếng Việt.\n"
        "Luôn dùng đúng format XML:\n"
        "<answer>...</answer>\n"
        "<reasoning>[TYPE] ...</reasoning>\n\n"
        f"{fewshot_text}\n\n"
        f"Bây giờ, hãy trả lời câu hỏi sau theo đúng định dạng:\n\n"
        f"Q: {question}"
    )

    return prompt
