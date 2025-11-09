"""
prepare_dataset_synthetic.py – Pre-classify reasoning types + balance dataset via synthetic generation
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
CSV_PATH  = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
OUT_CSV = "/kaggle/working/train_balanced_synthetic.csv"

REASONING_TYPES = [
    "DESCRIPTIVE",
    "CAUSAL", 
    "SPATIAL",
    "COUNTING",
    "OBJECT",
    "COMMONSENSE",
    "INTENT"
]

MIN_SAMPLES_PER_TYPE = 100
SYNTHETIC_TYPES = ["CAUSAL", "INTENT"]  # types to generate synthetic questions
SYNTHETIC_PER_SAMPLE = 5  # số câu synthetic mỗi câu gốc

# ===========================
# Load model
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ===========================
# LLM-based Reasoning Type Classifier
# ===========================
def classify_reasoning_type_llm(question: str):
    classification_prompt = f"""Phân loại câu hỏi VQA vào 1 trong 7 loại reasoning:
1. DESCRIPTIVE
2. CAUSAL
3. SPATIAL
4. COUNTING
5. OBJECT
6. COMMONSENSE
7. INTENT

Câu hỏi: "{question}"
Trả lời đúng 1 từ trong danh sách."""

    messages = [{"role": "user", "content": [{"type": "text", "text": classification_prompt}]}]

    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()

        for rtype in REASONING_TYPES:
            if rtype in result:
                return rtype
        return "COMMONSENSE"

    except:
        return "COMMONSENSE"

# ===========================
# Synthetic question generation
# ===========================
def generate_synthetic_questions(question: str, rtype: str, n: int = SYNTHETIC_PER_SAMPLE):
    """
    Generate n synthetic paraphrase questions using LLM
    """
    generation_prompt = f"""Tạo {n} câu hỏi VQA tiếng Việt có cùng reasoning type {rtype} dựa trên câu sau, giữ ý nghĩa nhưng đổi cách diễn đạt:
"{question}"
Trả lời dạng danh sách JSON: ["...", "...", ...]"""

    messages = [{"role": "user", "content": [{"type": "text", "text": generation_prompt}]}]

    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7
            )

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        result_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # parse JSON list
        try:
            synthetic_questions = json.loads(result_text)
            if isinstance(synthetic_questions, list):
                return [q.strip() for q in synthetic_questions if isinstance(q, str)]
        except:
            return []

    except:
        return []

# ===========================
# Main
# ===========================
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

classification_cache = {}
reasoning_types = []

print("[INFO] 🔍 Pre-classifying all questions...")
for q in tqdm(df["question"].astype(str).tolist(), desc="Classifying"):
    q_strip = q.strip()
    if q_strip not in classification_cache:
        classification_cache[q_strip] = classify_reasoning_type_llm(q_strip)
    reasoning_types.append(classification_cache[q_strip])

df["reasoning_type"] = reasoning_types

print("[INFO] Initial reasoning type distribution:")
print(df["reasoning_type"].value_counts())

print("[INFO] 🔄 Generating synthetic questions for underrepresented types...")
synthetic_data = []

for rtype in SYNTHETIC_TYPES:
    type_samples = df[df["reasoning_type"] == rtype]
    for _, row in tqdm(type_samples.iterrows(), total=len(type_samples), desc=f"Generating {rtype}"):
        synthetic_questions = generate_synthetic_questions(row["question"], rtype)
        for sq in synthetic_questions:
            synthetic_data.append({"question": sq, "answer": row["answer"], "reasoning_type": rtype})

df_synthetic = pd.DataFrame(synthetic_data)

# Combine original + synthetic
df_combined = pd.concat([df, df_synthetic], ignore_index=True)

print("[INFO] 🔄 Balancing dataset via oversample...")
balanced_samples = []

for rtype in REASONING_TYPES:
    type_samples = df_combined[df_combined["reasoning_type"] == rtype]
    if len(type_samples) == 0:
        continue
    n = max(MIN_SAMPLES_PER_TYPE, len(type_samples))
    sampled = pd.concat([type_samples] * ((n // max(1, len(type_samples))) + 1))
    balanced_samples.append(sampled.head(n))

df_balanced = pd.concat(balanced_samples).sample(frac=1.0, random_state=42).reset_index(drop=True)

df_balanced.to_csv(OUT_CSV, index=False)
print(f"[INFO] ✅ Saved balanced dataset: {OUT_CSV}, size={len(df_balanced)} samples")
