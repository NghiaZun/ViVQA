"""
prepare_dataset.py – Pre-classify reasoning types + balance dataset
Author: Nghia-Duong (refined)
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===========================
# Config
# ===========================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
OUT_CSV = "/kaggle/working/train_balanced.csv"

REASONING_TYPES = [
    "DESCRIPTIVE","CAUSAL","SPATIAL","COUNTING",
    "OBJECT","COMMONSENSE","INTENT"
]

MIN_SAMPLES_PER_TYPE = 100

# ===========================
# Load model
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()

# ===========================
# LLM classifier
# ===========================
def classify_reasoning(question: str) -> str:
    prompt = f"""Phân loại câu hỏi VQA vào 1 loại: 
DESCRIPTIVE, CAUSAL, SPATIAL, COUNTING, OBJECT, COMMONSENSE, INTENT.

Câu hỏi: "{question}"
Trả lời đúng 1 từ."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].upper()
        for t in REASONING_TYPES:
            if t in gen:
                return t
        return "COMMONSENSE"
    except:
        return "COMMONSENSE"

# ===========================
# Main
# ===========================
df = pd.read_csv(CSV_PATH)
df["question"] = df["question"].astype(str)

print("[INFO] 🔍 Classifying reasoning types...")
reasoning_cache = {}
rtype_list = []

for q in tqdm(df["question"], desc="Classifying"):
    q = q.strip()
    if q not in reasoning_cache:
        reasoning_cache[q] = classify_reasoning(q)
    rtype_list.append(reasoning_cache[q])

df["reasoning_type"] = rtype_list

print("[INFO] ✅ Distribution:")
print(df["reasoning_type"].value_counts())

# ===========================
# Oversample
# ===========================
balanced = []
for r in REASONING_TYPES:
    group = df[df["reasoning_type"] == r]
    if len(group) == 0:
        continue
    n = max(MIN_SAMPLES_PER_TYPE, len(group))
    repeated = pd.concat([group] * (n // len(group) + 1))
    balanced.append(repeated.head(n))

df_balanced = pd.concat(balanced).sample(frac=1.0).reset_index(drop=True)

df_balanced.to_csv(OUT_CSV, index=False)
print(f"[INFO] ✅ Saved balanced dataset: {OUT_CSV} ({len(df_balanced)} samples)")
