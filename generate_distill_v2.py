"""
Day 1 – Offline Teacher using Qwen2-VL-7B-Instruct with LLM-based Reasoning Type + Oversample & Weighted Loss
Author: Nghia-Duong (refined)
"""

import os
import json
import re
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
NUM_SAMPLES = None  # None = full dataset

# Reasoning type tokens
REASONING_TYPES = [
    "DESCRIPTIVE",
    "CAUSAL", 
    "SPATIAL",
    "COUNTING",
    "OBJECT",
    "COMMONSENSE",
    "INTENT"
]

# Weighted loss map (for student)
REASONING_WEIGHTS = {
    "CAUSAL": 5.0,
    "DESCRIPTIVE": 4.0,
    "INTENT": 4.0,
    "OBJECT": 2.0,
    "COUNTING": 2.0,
    "SPATIAL": 1.5,
    "COMMONSENSE": 1.0
}

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

if NUM_SAMPLES is None or NUM_SAMPLES > len(df):
    subset = df
else:
    subset = df.head(NUM_SAMPLES)

print(f"[INFO] Loaded {len(subset)} samples from ViVQA (total={len(df)})")

# ===========================
# LLM-based Reasoning Type Classifier
# ===========================
def classify_reasoning_type_llm(question: str, processor, model, device) -> str:
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
                do_sample=False,
                temperature=0.3
            )

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()

        for rtype in REASONING_TYPES:
            if rtype in result:
                return rtype
        return "COMMONSENSE"

    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
        return "COMMONSENSE"

# ===========================
# Helper: Parse model output
# ===========================
def parse_structured_output(text: str):
    answer, reasoning, reasoning_type = "", "", ""
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    reasoning_match = re.search(r'<reasoning>\s*\[(\w+)\]\s*(.*?)</reasoning>', text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_type = reasoning_match.group(1).strip().upper()
        reasoning = reasoning_match.group(2).strip()
    if not answer:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            answer = lines[0]
    if not reasoning:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1:
            reasoning = " ".join(lines[1:])
    return answer, reasoning, reasoning_type

# ===========================
# Teacher generation
# ===========================
def call_teacher_qwen(image_path: str, question: str, expected_type: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open image {image_path}: {e}")
        return {"answer": "", "reasoning": "", "reasoning_type": "", "raw": ""}

    user_prompt = build_fewshot_prompt(question)
    enhanced_system_prompt = f"""{SYSTEM_PROMPT}

BẠN PHẢI TRẢ LỜI THEO FORMAT:

<answer>Câu trả lời ngắn</answer>
<reasoning>[{expected_type}] 1-2 câu giải thích</reasoning>
"""

    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": user_prompt}]}
    ]

    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                top_p=0.85,
                top_k=40
            )

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        answer, reasoning, reasoning_type = parse_structured_output(text)
        if not reasoning_type:
            reasoning_type = expected_type
        return {
            "answer": answer,
            "reasoning": reasoning,
            "reasoning_type": reasoning_type,
            "raw": text,
            "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0)
        }

    except Exception as e:
        print(f"[WARN] Generation failed for {image_path}: {e}")
        return {"answer": "", "reasoning": "", "reasoning_type": "", "raw": "", "reasoning_weight": 1.0}

# ===========================
# Pre-classify reasoning types
# ===========================
PRE_CLASSIFY = True
MIN_SAMPLES_PER_TYPE = 100
BALANCE_DATASET = True

classification_cache = {}

if PRE_CLASSIFY:
    print("[INFO] 🔍 Pre-classifying all questions...")
    for question in tqdm(subset["question"].unique(), desc="Classifying"):
        question = str(question).strip()
        if question not in classification_cache:
            rtype = classify_reasoning_type_llm(question, processor, model, device)
            classification_cache[question] = rtype

    # Calculate type distribution
    type_counts = {r: 0 for r in REASONING_TYPES}
    for q in subset["question"]:
        rtype = classification_cache.get(str(q).strip(), "COMMONSENSE")
        type_counts[rtype] += 1
    print("[INFO] Initial reasoning type distribution:")
    for r in REASONING_TYPES:
        print(f"  {r}: {type_counts[r]}")

    # Oversample / balance
    if BALANCE_DATASET:
        print("[INFO] 🔄 Balancing dataset via oversample...")
        balanced_samples = []
        for rtype in REASONING_TYPES:
            type_samples = subset[subset["question"].apply(lambda q: classification_cache.get(str(q).strip()) == rtype)]
            n = max(MIN_SAMPLES_PER_TYPE, len(type_samples))
            # Oversample by repetition if needed
            sampled = pd.concat([type_samples] * ((n // max(1, len(type_samples)) + 1))) if len(type_samples) > 0 else type_samples
            balanced_samples.append(sampled)
        subset = pd.concat(balanced_samples).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[INFO] ✅ Balanced dataset size: {len(subset)}")

# ===========================
# Generate teacher outputs
# ===========================
results, skipped = [], 0
for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating teacher answers"):
    image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    question = str(row["question"]).strip()
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        skipped += 1
        continue

    # LLM-based classification cache
    reasoning_type = classification_cache.get(question, "COMMONSENSE")
    res = call_teacher_qwen(image_path, question, reasoning_type)
    if res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": question,
            "reasoning_type": res["reasoning_type"],
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"],
            "reasoning_weight": res["reasoning_weight"]
        })
    else:
        skipped += 1

    if len(results) % 10 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ===========================
# Final save + stats
# ===========================
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] ✅ Saved {len(results)} samples to {OUT_JSONL}, skipped {skipped}")
