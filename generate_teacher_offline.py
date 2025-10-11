"""
Day 1 – Offline Teacher using Qwen2-VL-7B-Instruct (Debug version)
Author: Nghia-Duong (refined by ChatGPT)
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
NUM_SAMPLES = None  # None = full dataset, hoặc đặt số cụ thể để test

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
# Helper: Parse model output
# ===========================
def parse_answer_reasoning(text: str):
    """Try to extract Answer: and Reasoning: flexibly."""
    answer, reasoning = "", ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
    if not answer and len(lines) > 0:
        answer = lines[0]
    if not reasoning and len(lines) > 1:
        reasoning = " ".join(lines[1:])
    return answer, reasoning

# ===========================
# Teacher Function
# ===========================
def call_teacher_qwen(image_path: str, question: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open image {image_path}: {e}")
        return {"answer": "", "reasoning": "", "raw": ""}

    user_prompt = build_fewshot_prompt(question)

    # ✅ Qwen2-VL conversation format
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Hãy quan sát ảnh và trả lời câu hỏi theo đúng định dạng:\n"
                        f"{user_prompt}\n\n"
                        "Bắt đầu bằng:\nAnswer: ...\nReasoning: ..."
                    )
                }
            ]
        }
    ]

    try:
        # ✅ Apply chat template for multimodal encoding
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # ✅ Process multimodal inputs
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # ✅ Generate text output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )

        # ✅ Decode only generated tokens
        generated_ids = output[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # --- DEBUG LOG ---
        print("=" * 80)
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Raw output:\n{text}\n")

        # --- Parse ---
        answer, reasoning = parse_answer_reasoning(text)
        return {"answer": answer, "reasoning": reasoning, "raw": text}

    except Exception as e:
        print(f"[WARN] Generation failed for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": "", "reasoning": "", "raw": ""}

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
    else:
        skipped += 1

    # Autosave every 10 samples
    if len(results) % 10 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ===========================
# Final save
# ===========================
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("=" * 80)
print(f"[INFO] ✅ Saved {len(results)} reasoning samples to {OUT_JSONL}")
print(f"[INFO] ⚠️ Skipped {skipped} samples (no output or missing image)")
print(f"[INFO] Dataset size: {len(df)}, Processed subset: {len(subset)}")
