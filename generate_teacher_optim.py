"""
teacher_generate.py – STABLE version với tối ưu đơn giản
Author: Nghia-Duong (stable + faster)
"""

import os
import json
import re
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"  # GT-guided dataset
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
OUT_JSONL = "/kaggle/working/teacher_outputs_gt_guided.jsonl"

# Reasoning type keywords for auto-classification
REASONING_KEYWORDS = {
    "COUNTING": ["bao nhiêu", "mấy", "số lượng", "đếm"],
    "SPATIAL": ["ở đâu", "vị trí", "phía", "trên", "dưới", "trong", "ngoài"],
    "CAUSAL": ["tại sao", "vì sao", "lý do", "nguyên nhân"],
    "OBJECT": ["cái gì", "con gì", "là gì", "vật gì"],
    "INTENT": ["mục đích", "ý định", "dùng để"],
    "COMMONSENSE": ["nên", "thường", "có thể", "phải"],
    "DESCRIPTIVE": []
}

REASONING_WEIGHTS = {
    "CAUSAL": 5.0,
    "DESCRIPTIVE": 4.0,
    "INTENT": 4.0,
    "OBJECT": 2.0,
    "COUNTING": 2.0,
    "SPATIAL": 1.5,
    "COMMONSENSE": 1.0
}

def infer_reasoning_type(question: str) -> str:
    """Auto-classify reasoning type from question"""
    q_lower = question.lower().strip()
    for rtype, keywords in REASONING_KEYWORDS.items():
        if rtype == "DESCRIPTIVE":
            continue
        for kw in keywords:
            if kw in q_lower:
                return rtype
    return "DESCRIPTIVE"

# ===========================
# LOAD MODEL - ĐƠN GIẢN HÓA
# ===========================
device = "cuda:0"  # Chỉ dùng GPU đầu tiên cho ổn định
print(f"[INFO] Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",  # Để tự động chọn, nhưng sẽ ưu tiên GPU 0
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()

# ===========================
# PARSE OUTPUT - SIMPLE FORMAT
# ===========================
def parse_structured_output(text: str, question: str = ""):
    """Parse simple format: Answer: X / Type: Y / Reasoning: Z"""
    answer, reasoning, reasoning_type = "", "", ""
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Answer:'):
            answer = line.split(':', 1)[1].strip()
        elif line.startswith('Type:'):
            reasoning_type = line.split(':', 1)[1].strip().upper()
        elif line.startswith('Reasoning:'):
            reasoning = line.split(':', 1)[1].strip()
    
    # Fallback to heuristic if no type found
    if not reasoning_type and question:
        reasoning_type = infer_reasoning_type(question)
    
    return answer, reasoning, reasoning_type

# ===========================
# TEACHER GENERATION - GT-GUIDED + OPTIMIZED
# ===========================
@torch.no_grad()
def call_teacher_qwen(image_path: str, question: str, ground_truth: str):
    """GT-guided: Teacher explains WHY answer is ground_truth"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None

    # Simple format prompt
    user_prompt = f"""Câu hỏi: {question}
Đáp án: {ground_truth}

Giải thích ngắn gọn TẠI SAO đáp án là "{ground_truth}" dựa vào hình ảnh.

Format:
Answer: {ground_truth}
Type: [COUNTING/SPATIAL/CAUSAL/OBJECT/INTENT/COMMONSENSE/DESCRIPTIVE]
Reasoning: (1 câu)

Type:
- COUNTING: đếm
- SPATIAL: vị trí
- CAUSAL: nguyên nhân
- OBJECT: nhận diện
- DESCRIPTIVE: mô tả
- COMMONSENSE: kiến thức chung
- INTENT: mục đích
"""

    enhanced_system_prompt = "Bạn là VQA model. Trả lời ngắn gọn theo format."

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    try:
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Mixed precision + optimized generation
        with torch.amp.autocast('cuda'):
            output = model.generate(
                **inputs,
                max_new_tokens=80,        # Reduced for speed
                do_sample=False,          # Greedy = faster
                temperature=1.0,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        gen = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        answer, reasoning, reasoning_type = parse_structured_output(gen, question)

        return {
            "answer": answer,
            "reasoning": reasoning,
            "reasoning_type": reasoning_type,
            "raw": gen,
            "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0)
        }

    except Exception as e:
        print(f"[ERROR] Generation failed for {image_path}: {e}")
        return None

# ===========================
# MAIN LOOP - CẢI THIỆN
# ===========================
df = pd.read_csv(CSV_PATH)
results = []

# Periodic save để tránh mất dữ liệu
SAVE_INTERVAL = 200

print(f"[INFO] Total samples to process: {len(df)}")

for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="GT-Guided Teacher")):
    image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    
    if not os.path.exists(image_path):
        continue

    q = str(row["question"]).strip()
    gt_answer = str(row["answer"]).strip()  # Ground truth

    res = call_teacher_qwen(image_path, q, gt_answer)

    if res and res["answer"]:  # Kiểm tra res không None
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": q,
            "reasoning_type": res["reasoning_type"],
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"],
            "reasoning_weight": res["reasoning_weight"]
        })
    
    # Save định kỳ
    if len(results) % SAVE_INTERVAL == 0 and len(results) > 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[INFO] 💾 Checkpoint: Saved {len(results)} samples")
    
    # Memory management mỗi 100 samples
    if idx % 100 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()  # Python garbage collection

# Final save
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n[INFO] ✅ Completed! Saved {len(results)}/{len(df)} teacher samples → {OUT_JSONL}")
print(f"[INFO] Success rate: {len(results)/len(df)*100:.1f}%")
