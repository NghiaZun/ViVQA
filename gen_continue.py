"""
teacher_generate_resume.py – Resume từ checkpoint
"""

import os
import json
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

# CHECKPOINT PATH (update với dataset của bạn)
CHECKPOINT_JSONL = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"
OUT_JSONL = "/kaggle/working/teacher_outputs_continued.jsonl"

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
# LOAD CHECKPOINT
# ===========================
def load_checkpoint(checkpoint_path):
    """Load processed image IDs from checkpoint"""
    processed_ids = set()
    existing_results = []
    
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data["img_id"])
                    existing_results.append(data)
                except:
                    continue
        print(f"[INFO] ✅ Loaded {len(processed_ids)} existing samples")
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")
    
    return processed_ids, existing_results

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
# PARSE OUTPUT (same as before)
# ===========================
def parse_structured_output(text: str):
    import re
    answer, reasoning, reasoning_type = "", "", ""

    m1 = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m1:
        answer = m1.group(1).strip()

    m2 = re.search(r"<reasoning>\s*\[(\w+)\]\s*(.*?)</reasoning>", text, re.S)
    if m2:
        reasoning_type = m2.group(1).upper()
        reasoning = m2.group(2).strip()

    return answer, reasoning, reasoning_type

# ===========================
# TEACHER GENERATION (same as before)
# ===========================
@torch.no_grad()
def call_teacher_qwen(image_path: str, question: str, expected_type: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
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

        with torch.amp.autocast('cuda'):
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.6,
                top_p=0.85,
                top_k=40,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
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
        print(f"[ERROR] Generation failed: {e}")
        return None

# ===========================
# MAIN LOOP - RESUME VERSION
# ===========================
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Total samples in CSV: {len(df)}")

# Load checkpoint
processed_ids, existing_results = load_checkpoint(CHECKPOINT_JSONL)

# Filter remaining samples
df_remaining = df[~df["img_id"].isin(processed_ids)].reset_index(drop=True)
print(f"[INFO] 🎯 Remaining to process: {len(df_remaining)} samples")
print(f"[INFO] Progress: {len(processed_ids)}/{len(df)} ({len(processed_ids)/len(df)*100:.1f}%)")

if len(df_remaining) == 0:
    print("[INFO] ✅ All samples already processed!")
    exit(0)

# Generate remaining
new_results = []
SAVE_INTERVAL = 50  # Save mỗi 50 samples

for idx, row in tqdm(df_remaining.iterrows(), total=len(df_remaining), desc="Resuming Generation"):
    image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    
    if not os.path.exists(image_path):
        continue

    q = str(row["question"]).strip()
    exp_type = row["reasoning_type"]

    res = call_teacher_qwen(image_path, q, exp_type)

    if res and res["answer"]:
        new_results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": q,
            "reasoning_type": res["reasoning_type"],
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"],
            "teacher_raw": res["raw"],
            "reasoning_weight": res["reasoning_weight"]
        })
    
    # Periodic save (chỉ save phần mới)
    if len(new_results) % SAVE_INTERVAL == 0 and len(new_results) > 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in new_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[INFO] 💾 Checkpoint: {len(new_results)} new samples saved")
    
    # Clear cache
    if idx % 100 == 0:
        torch.cuda.empty_cache()

# Final save (chỉ phần mới)
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in new_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n[INFO] ✅ Generated {len(new_results)} new samples → {OUT_JSONL}")

# ===========================
# MERGE WITH CHECKPOINT
# ===========================
FINAL_JSONL = "/kaggle/working/teacher_outputs_full.jsonl"

print(f"[INFO] Merging checkpoint + new results...")
with open(FINAL_JSONL, "w", encoding="utf-8") as f_out:
    # Write existing results
    for r in existing_results:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Write new results
    for r in new_results:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

total_count = len(existing_results) + len(new_results)
print(f"[INFO] ✅ MERGED! Total: {total_count}/{len(df)} samples")
print(f"[INFO] Final file: {FINAL_JSONL}")
print(f"[INFO] Completion rate: {total_count/len(df)*100:.1f}%")

# Verify
print("\n[INFO] Verification:")
print(f"  - Checkpoint: {len(existing_results)}")
print(f"  - New: {len(new_results)}")
print(f"  - Total: {total_count}")
print(f"  - Missing: {len(df) - total_count}")
