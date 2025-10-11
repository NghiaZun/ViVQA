"""
Day 1 – Offline Teacher using Qwen2-VL-7B-Instruct
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from utils_prompt import SYSTEM_PROMPT, build_fewshot_prompt

# ===========================
# Config
# ===========================
CSV_PATH  = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
OUT_JSONL = "/kaggle/working/teacher_outputs_offline.jsonl"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
NUM_SAMPLES = 10  # test subset

# ===========================
# Load model
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ===========================
# Dataset
# ===========================
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
subset = df.head(NUM_SAMPLES)
print(f"[INFO] Loaded {len(subset)} samples from ViVQA")

# ===========================
# Teacher Function
# ===========================
def call_teacher_qwen(image_path: str, question: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return {"answer": "", "reasoning": "", "raw": ""}

    user_prompt = build_fewshot_prompt(question)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    try:
        inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=200)
        text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        print("\n[DEBUG] Raw output sample:\n", text, "\n")

        answer, reasoning = "", ""
        for line in text.splitlines():
            if line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
            elif line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        return {"answer": answer, "reasoning": reasoning, "raw": text}

    except Exception as e:
        print(f"[WARN] Error: {e}")
        return {"answer": "", "reasoning": "", "raw": ""}

# ===========================
# Main loop
# ===========================
results = []
for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating teacher answers"):
    image_id = str(row["img_id"])
    question = str(row["question"])
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        continue

    res = call_teacher_qwen(image_path, question)
    if res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": question,
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"]
        })

    if len(results) % 10 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Save cuối
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] ✅ Saved {len(results)} reasoning samples to {OUT_JSONL}")
