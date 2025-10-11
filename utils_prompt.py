# utils_prompt.py — few-shot & reasoning taxonomy

FEW_SHOTS = [
    {
        "question": "Con vật trong hình là gì?",
        "answer": "Một con chó.",
        "reasoning": "Tôi nhận ra vì hình có con vật bốn chân, tai cụp và lông ngắn màu nâu."
    },
    {
        "question": "Người phụ nữ đang làm gì?",
        "answer": "Cô ấy đang đọc sách.",
        "reasoning": "Cô ấy đang nhìn vào một cuốn sách đang mở, nên có thể đang đọc."
    },
    {
        "question": "Có bao nhiêu quả chuối trên bàn?",
        "answer": "Ba quả chuối.",
        "reasoning": "Tôi đếm được ba quả màu vàng nằm gần nhau trên bàn."
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
