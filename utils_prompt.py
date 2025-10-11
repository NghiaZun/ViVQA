# utils_prompt.py — few-shot & reasoning taxonomy

FEW_SHOTS = [
    # 1️⃣ Visual Recognition
    {
        "question": "Màu của chiếc bình là gì?",
        "answer": (
            "Answer: màu xanh lá\n"
            "Reasoning (Visual Recognition): Chiếc bình trong ảnh có thân và cổ đều mang sắc xanh lá cây, "
            "nên màu của nó là xanh lá."
        )
    },

    # 2️⃣ Spatial Reasoning
    {
        "question": "Cô gái ngồi ở đâu?",
        "answer": (
            "Answer: trên giường\n"
            "Reasoning (Spatial): Cô gái được nhìn thấy ngồi trên một bề mặt phẳng có chăn và gối, "
            "đặc trưng của một chiếc giường, nên vị trí là trên giường."
        )
    },

    # 3️⃣ Counting
    {
        "question": "Có bao nhiêu con chim đậu trên cành cây bên cạnh nhau?",
        "answer": (
            "Answer: hai\n"
            "Reasoning (Counting): Trên nhánh cây có hai con chim đứng gần nhau, "
            "không có con nào khác trên cùng nhánh, nên số lượng là hai."
        )
    },

    # 4️⃣ Commonsense
    {
        "question": "Tại sao người đàn ông đội mũ bảo hiểm?",
        "answer": (
            "Answer: để bảo vệ đầu khi lái xe máy\n"
            "Reasoning (Commonsense): Theo hiểu biết thông thường, người lái xe máy đội mũ bảo hiểm "
            "để tránh chấn thương đầu khi xảy ra tai nạn, nên ông ấy đội mũ để bảo vệ đầu."
        )
    },

    # 5️⃣ Causal
    {
        "question": "Tại sao chiếc ô bị ướt?",
        "answer": (
            "Answer: vì trời đang mưa\n"
            "Reasoning (Causal): Hình ảnh cho thấy có các giọt nước rơi và nền đường ướt, "
            "cho thấy nguyên nhân là trời đang mưa khiến chiếc ô bị ướt."
        )
    },

    # 6️⃣ Comparative
    {
        "question": "Chiếc xe nào lớn hơn giữa hai chiếc?",
        "answer": (
            "Answer: chiếc xe tải lớn hơn\n"
            "Reasoning (Comparative): Xe tải có thân và bánh lớn hơn rõ rệt so với xe con, "
            "nên xe tải là chiếc lớn hơn."
        )
    }
]


SYSTEM_PROMPT = (
    "Bạn là mô hình Visual Question Answering thông minh, "
    "trả lời bằng tiếng Việt ngắn gọn và có phần giải thích (Reasoning). "
    "Luôn trả lời theo định dạng:\n\n"
    "Answer: <câu trả lời ngắn>\n"
    "Reasoning: <giải thích ngắn, thuộc một trong các loại: "
    "Visual Recognition, Spatial, Counting, Commonsense, Causal, hoặc Comparative>.\n"
)

def build_fewshot_prompt(question: str) -> str:
    examples = "\n\n".join(
        [f"Question: {ex['question']}\nAnswer: {ex['answer']}\nReasoning: {ex['reasoning']}" for ex in FEW_SHOTS]
    )
    return f"{examples}\n\nQuestion: {question}\nAnswer và Reasoning của bạn là:"
