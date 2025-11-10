"""
teacher_generate_resume.py – Stable RESUME version
Author: Nghia-Duong (resume + safe checkpoint)
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
CSV_PATH = "/kaggle/input/train-balanced-syn/train_balanced_synthetic.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

OLD_JSONL = "/kaggle/input/teacher-checkpoint-old/teacher_outputs.jsonl"   # <--- CHECKPOINT CŨ
OUT_JSONL = "/kaggle/working/teacher_outputs.jsonl"                       # <--- OUTPUT MỚI

REASONING_WEIGHTS = {
    "CAUSAL": 5.0,
    "DESCRIPTIVE": 4.0,
    "INTENT": 4.0,
    "OBJECT": 2.0,
    "COUNTING": 2.0,
    "SPATIAL": 1.5,
    "COMMONSENSE": 1.0
}

SAVE_INTERVAL = 200  # checkpoint định kỳ

# ===========================
# LOAD MODEL
# ===========================
device = "cuda:0"
print(f"[INFO] Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()


# ===========================
# PARSE OUTPUT
# ===========================
def parse_structured_output(text: str):
    answer, reasoning, reasoning_type = "", "", ""

    m1 = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m1:
        answer = m1.group(1).strip()

    m2 = re.search(r"<reasoning>\s*\[(.*?)\]\s*(.*?)</reasoning>", text, re.S)
    if m2:
        reasoning_type = m2.group(1).strip().upper()
        reasoning = m2.group(2).strip()

    return answer, reasoning, reasoning_type


# ===========================
# LOAD OLD CHECKPOINT
# ===========================
done = set()
results = []

if os.path.exists(OLD_JSONL):
    print(f"[INFO] Loading old checkpoint from {OLD_JSONL}")
    with open(OLD_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                img_id = str(obj["img_id"])
                done.add(img_id)
                results.append(obj)
            except:
                pass

print(f"[INFO] Resume loaded: {len(done)} items already completed")


# ===========================
# TEACHER CALL
# ===========================
@torch.no_grad()
def call_teacher_qwen(image_path: str, question: str, expected_type: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return None

    from utils_prompt import SYSTEM_PROMPT, build_fewshot_prompt
    user_prompt = build_fewshot_prompt(question)

    enhanced_system_prompt = f"""{SYSTEM_PROMPT}

BẠN PHẢI TRẢ LỜI THEO FORMAT:

<answer>Câu trả lời ngắn</answer>
<reasoning>[{expected_type}] 1-2 câu giải thích</reasoning>
"""

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.amp.autocast("cuda"):
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.6,
                top_p=0.85,
                top_k=40,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        gen = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        answer, reasoning, reasoning_type = parse_structured_output(gen)
        if not reasoning_type:
            reasoning_type = expected_type

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
# MAIN LOOP (RESUME)
# ===========================
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Total rows: {len(df)}")
print(f"[INFO] Remaining after resume: {len(df) - len(done)}")

processed = len(results)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Teacher Generating"):
    image_id = str(row.get("img_id", row.get("image_id", ""))).strip()

    if image_id in done:
        continue

    img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        continue

    q = str(row["question"]).strip()
    expected_type = row["reasoning_type"]

    res = call_teacher_qwen(img_path, q, expected_type)

    if res and res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": img_path,
            "question": q,
            "reasoning_type": res["reasoning_type"],
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"],
            "reasoning_weight": res["reasoning_weight"]
        })
        done.add(image_id)
        processed += 1

    # checkpoint (APPEND, không overwrite)
    if processed % SAVE_INTERVAL == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[INFO] ✅ Saved checkpoint: {processed} samples")

    if idx % 100 == 0:
        torch.cuda.empty_cache()

# Final save
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n[INFO] ✅ COMPLETED")
print(f"[INFO] Total output: {len(results)}")
print(f"[INFO] Saved to: {OUT_JSONL}")
