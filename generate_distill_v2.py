"""
Day 1 – Offline Teacher using Qwen2-VL-7B-Instruct with LLM-based Reasoning Type
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
    """
    Sử dụng LLM để tự động phân loại reasoning type
    """
    classification_prompt = f"""Phân loại câu hỏi Visual Question Answering sau vào ĐÚNG MỘT trong 7 loại reasoning:

1. DESCRIPTIVE: Mô tả thuộc tính, đặc điểm (màu sắc, hình dạng, kích thước, chất liệu...)
2. CAUSAL: Hỏi về nguyên nhân, lý do, giải thích tại sao
3. SPATIAL: Hỏi về vị trí, phương hướng, quan hệ không gian
4. COUNTING: Đếm số lượng vật thể, người, sự vật
5. OBJECT: Nhận dạng, xác định vật thể, người, sự vật là gì
6. COMMONSENSE: Hiểu biết thông thường, kiến thức đời sống, logic thường ngày
7. INTENT: Mục đích, ý định, dự định của hành động hoặc sự vật

Câu hỏi: "{question}"

Trả lời ĐÚNG MỘT TỪ trong danh sách: DESCRIPTIVE, CAUSAL, SPATIAL, COUNTING, OBJECT, COMMONSENSE, INTENT

Phân loại:"""

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": classification_prompt}]
        }
    ]

    try:
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,  # Deterministic cho classification
                temperature=0.3
            )

        generated_ids = output[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()

        # Extract first valid reasoning type
        for rtype in REASONING_TYPES:
            if rtype in result:
                return rtype
        
        # Fallback
        return "COMMONSENSE"

    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
        return "COMMONSENSE"

# ===========================
# Helper: Parse model output
# ===========================
def parse_structured_output(text: str):
    """
    Parse output format:
    <answer>...</answer>
    <reasoning>[TYPE] ...</reasoning>
    """
    answer = ""
    reasoning = ""
    reasoning_type = ""
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    # Extract reasoning with type
    reasoning_match = re.search(r'<reasoning>\s*\[(\w+)\]\s*(.*?)</reasoning>', text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_type = reasoning_match.group(1).strip().upper()
        reasoning = reasoning_match.group(2).strip()
    
    # Fallback parsing
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
# Teacher Function
# ===========================
def call_teacher_qwen(image_path: str, question: str, expected_type: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open image {image_path}: {e}")
        return {"answer": "", "reasoning": "", "reasoning_type": "", "raw": ""}

    user_prompt = build_fewshot_prompt(question)

    # Updated system prompt with new format
    enhanced_system_prompt = f"""{SYSTEM_PROMPT}

BẠN PHẢI TRẢ LỜI THEO ĐÚNG FORMAT SAU (NGẮN GỌN):

<answer>Câu trả lời ngắn gọn</answer>
<reasoning>[{expected_type}] Lý do suy luận ngắn gọn (1-2 câu)</reasoning>

Reasoning types: DESCRIPTIVE, CAUSAL, SPATIAL, COUNTING, OBJECT, COMMONSENSE, INTENT

Lưu ý:
- Answer phải NGẮN GỌN (2-5 từ)
- Reasoning chỉ 1-2 câu giải thích
- BẮT BUỘC dùng đúng tag XML và reasoning type token [{expected_type}]
"""

    messages = [
        {
            "role": "system",
            "content": enhanced_system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"{user_prompt}\n\nTrả lời theo format:\n<answer>...</answer>\n<reasoning>[{expected_type}] ...</reasoning>"
                }
            ]
        }
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

        # --- DEBUG LOG ---
        print("=" * 80)
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Expected Type: {expected_type}")
        print(f"[DEBUG] Raw output:\n{text}\n")

        # --- Parse ---
        answer, reasoning, reasoning_type = parse_structured_output(text)
        
        # Fallback to expected type if not found
        if not reasoning_type:
            reasoning_type = expected_type
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "reasoning_type": reasoning_type,
            "raw": text
        }

    except Exception as e:
        print(f"[WARN] Generation failed for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": "", "reasoning": "", "reasoning_type": "", "raw": ""}

# ===========================
# Pre-classification Analysis (Optional)
# ===========================
PRE_CLASSIFY = True  # Set False to skip pre-analysis
MIN_SAMPLES_PER_TYPE = 100  # Minimum samples per type for balanced training

if PRE_CLASSIFY:
    print("\n[INFO] 🔍 Pre-classifying all questions for distribution analysis...")
    classification_cache = {}
    
    for question in tqdm(subset["question"].unique(), desc="Pre-classifying"):
        question = str(question).strip()
        if question not in classification_cache:
            rtype = classify_reasoning_type_llm(question, processor, model, device)
            classification_cache[question] = rtype
    
    # Analyze distribution
    type_dist = {}
    for question in subset["question"]:
        rtype = classification_cache.get(str(question).strip(), "COMMONSENSE")
        type_dist[rtype] = type_dist.get(rtype, 0) + 1
    
    print("\n" + "="*80)
    print("[INFO] 📊 Initial Reasoning Type Distribution:")
    print("="*80)
    total = sum(type_dist.values())
    for rtype in sorted(REASONING_TYPES):
        count = type_dist.get(rtype, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {rtype:15s}: {count:5d} samples ({pct:5.1f}%)")
    
    # Check for severe imbalance
    print("\n[INFO] 🎯 Class Balance Analysis:")
    max_count = max(type_dist.values())
    min_count = min(type_dist.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"  Max class size: {max_count}")
    print(f"  Min class size: {min_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("  ⚠️  WARNING: Severe class imbalance detected!")
        print("  💡 Consider using stratified sampling or weighted loss")
    elif imbalance_ratio > 5:
        print("  ⚠️  Moderate class imbalance detected")
        print("  💡 Consider rebalancing strategies")
    else:
        print("  ✅ Class distribution is reasonably balanced")
    
    # Optional: Stratified sampling for balanced subset
    BALANCE_DATASET = False  # Set True to create balanced subset
    if BALANCE_DATASET and imbalance_ratio > 5:
        print(f"\n[INFO] 🔄 Creating balanced subset (min {MIN_SAMPLES_PER_TYPE} per type)...")
        balanced_samples = []
        
        for rtype in REASONING_TYPES:
            # Get all samples of this type
            type_samples = subset[subset["question"].apply(
                lambda q: classification_cache.get(str(q).strip()) == rtype
            )]
            
            # Sample or take all
            if len(type_samples) > MIN_SAMPLES_PER_TYPE:
                sampled = type_samples.sample(n=MIN_SAMPLES_PER_TYPE, random_state=42)
            else:
                sampled = type_samples
            
            balanced_samples.append(sampled)
        
        subset = pd.concat(balanced_samples).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[INFO] ✅ Balanced dataset size: {len(subset)}")
else:
    classification_cache = {}

# ===========================
# Main loop
# ===========================
results, skipped = [], 0

for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating teacher answers"):
    image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    question = str(row["question"]).strip()
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

    if not os.path.exists(image_path):
        print(f"[WARN] Missing image: {image_path}")
        skipped += 1
        continue

    # LLM-based classification with caching
    if question not in classification_cache:
        reasoning_type = classify_reasoning_type_llm(question, processor, model, device)
        classification_cache[question] = reasoning_type
    else:
        reasoning_type = classification_cache[question]
    
    res = call_teacher_qwen(image_path, question, reasoning_type)
    if res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": question,
            "reasoning_type": res["reasoning_type"],
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"]
        })
    else:
        skipped += 1

    # Autosave every 10 samples
    if len(results) % 10 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ===========================
# Final save + Statistics
# ===========================
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Print statistics
print("=" * 80)
print(f"[INFO] ✅ Saved {len(results)} reasoning samples to {OUT_JSONL}")
print(f"[INFO] ⚠️ Skipped {skipped} samples (no output or missing image)")
print(f"[INFO] Dataset size: {len(df)}, Processed subset: {len(subset)}")
print(f"[INFO] 🎯 Unique questions classified: {len(classification_cache)}")

# Reasoning type distribution
if results:
    type_counts = {}
    for r in results:
        rt = r["reasoning_type"]
        type_counts[rt] = type_counts.get(rt, 0) + 1
    
    print("\n[INFO] Reasoning Type Distribution:")
    for rt in sorted(type_counts.keys()):
        print(f"  {rt}: {type_counts[rt]} ({type_counts[rt]/len(results)*100:.1f}%)")
