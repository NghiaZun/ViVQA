"""
Prompt utilities for teacher reasoning (Day 1)
Author: Nghĩa Duong
"""

# ===========================
# System Prompt
# ===========================
SYSTEM_PROMPT = (
    "Bạn là mô hình Visual Question Answering thông minh, "
    "trả lời bằng tiếng Việt ngắn gọn và có phần giải thích (Reasoning). "
    "Luôn trả lời theo định dạng:\n\n"
    "Answer: <câu trả lời ngắn>\n"
    "Reasoning: <giải thích ngắn, thuộc một trong các loại: "
    "Visual Recognition, Spatial, Counting, Commonsense, Causal, hoặc Comparative>.\n"
)

# ===========================
# Few-shot Examples
# ===========================
FEW_SHOTS = [
    # Visual Recognition
    {
        "question": "Màu của chiếc bình là gì?",
        "answer": (
            "Answer: màu xanh lá\n"
            "Reasoning (Visual Recognition): Chiếc bình trong ảnh có sắc xanh lá, "
            "nên màu của nó là xanh lá."
        )
    },
    # Spatial
    {
        "question": "Cô gái ngồi ở đâu?",
        "answer": (
            "Answer: trên giường\n"
            "Reasoning (Spatial): Cô gái ngồi trên bề mặt có chăn gối, đặc trưng của giường."
        )
    },
    # Counting
    {
        "question": "Có bao nhiêu con chim đậu trên cành cây bên cạnh nhau?",
        "answer": (
            "Answer: hai\n"
            "Reasoning (Counting): Có hai con chim đậu cạnh nhau, không có con nào khác."
        )
    },
    # Commonsense
    {
        "question": "Tại sao người đàn ông đội mũ bảo hiểm?",
        "answer": (
            "Answer: để bảo vệ đầu khi lái xe máy\n"
            "Reasoning (Commonsense): Người lái xe đội mũ bảo hiểm để tránh chấn thương đầu."
        )
    },
    # Causal
    {
        "question": "Tại sao chiếc ô bị ướt?",
        "answer": (
            "Answer: vì trời đang mưa\n"
            "Reasoning (Causal): Có giọt nước rơi và nền đường ướt cho thấy trời mưa."
        )
    },
    # Comparative
    {
        "question": "Chiếc xe nào lớn hơn giữa hai chiếc?",
        "answer": (
            "Answer: chiếc xe tải lớn hơn\n"
            "Reasoning (Comparative): Xe tải có thân và bánh lớn hơn xe con."
        )
    },
]

# ===========================
# Build few-shot prompt
# ===========================
def build_fewshot_prompt(question: str) -> str:
    """Tạo prompt có ví dụ few-shot"""
    examples = [f"Q: {ex['question']}\n{ex['answer']}" for ex in FEW_SHOTS]
    fewshot_text = "\n\n".join(examples)
    prompt = (
        "Dưới đây là các ví dụ về cách mô hình trả lời câu hỏi thị giác bằng tiếng Việt. "
        "Mỗi câu trả lời có hai phần: Answer và Reasoning.\n\n"
        f"{fewshot_text}\n\n"
        f"Bây giờ, hãy trả lời câu hỏi sau theo cùng định dạng:\n\n"
        f"Q: {question}"
    )
    return prompt
