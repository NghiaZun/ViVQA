"""
teacher_generate.py – Generate teacher answers (OPTIMIZED for 2xT4)
Author: Nghia-Duong (optimized)
"""

import os
import json
import re
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from torch.utils.data import Dataset, DataLoader
from utils_prompt import SYSTEM_PROMPT, build_fewshot_prompt

# ===========================
# CONFIG
# ===========================
CSV_PATH = "/kaggle/input/train-balanced-syn/train_balanced_synthetic.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
OUT_JSONL = "/kaggle/working/teacher_outputs.jsonl"

BATCH_SIZE = 4  # Tăng lên 4-8 tùy VRAM
NUM_WORKERS = 2  # Để load ảnh song song

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
# DATASET CLASS
# ===========================
class VQADataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row.get("img_id", row.get("image_id", ""))).strip()
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = None
            
        return {
            "image": image,
            "image_id": image_id,
            "image_path": image_path,
            "question": str(row["question"]).strip(),
            "reasoning_type": row["reasoning_type"]
        }

# ===========================
# LOAD MODEL
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
print(f"[INFO] Number of GPUs: {torch.cuda.device_count()}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="balanced",  # Tự động phân bổ qua 2 GPU
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()

# Enable compilation cho PyTorch 2.0+
if hasattr(torch, 'compile'):
    print("[INFO] Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")

# ===========================
# PARSE OUTPUT
# ===========================
def parse_structured_output(text: str):
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
# BATCH GENERATION
# ===========================
def generate_batch(batch):
    # Filter out failed images
    valid_indices = [i for i, img in enumerate(batch["image"]) if img is not None]
    if not valid_indices:
        return []
    
    valid_images = [batch["image"][i] for i in valid_indices]
    valid_data = [{k: batch[k][i] for k in batch.keys()} for i in valid_indices]
    
    # Prepare messages for all samples
    messages_list = []
    for data in valid_data:
        user_prompt = build_fewshot_prompt(data["question"])
        expected_type = data["reasoning_type"]
        
        enhanced_system_prompt = f"""{SYSTEM_PROMPT}

BẠN PHẢI TRẢ LỜI THEO FORMAT:

<answer>Câu trả lời ngắn</answer>
<reasoning>[{expected_type}] 1-2 câu giải thích</reasoning>
"""
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": valid_images[0]},  # Placeholder
                {"type": "text", "text": user_prompt}
            ]}
        ]
        messages_list.append(messages)
    
    # Process all at once
    try:
        text_prompts = [processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 
                       for msgs in messages_list]
        
        inputs = processor(
            text=text_prompts,
            images=valid_images,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):  # Mixed precision
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                top_p=0.85,
                top_k=40,
                num_beams=1,  # Greedy cho nhanh hơn
                use_cache=True  # Cache KV
            )

        generated_texts = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        results = []
        for i, gen_text in enumerate(generated_texts):
            data = valid_data[i]
            answer, reasoning, reasoning_type = parse_structured_output(gen_text)
            
            if not reasoning_type:
                reasoning_type = data["reasoning_type"]
            
            if answer:  # Only save if we got an answer
                results.append({
                    "img_id": data["image_id"],
                    "image_path": data["image_path"],
                    "question": data["question"],
                    "reasoning_type": reasoning_type,
                    "teacher_answer": answer,
                    "teacher_reasoning": reasoning,
                    "teacher_raw": gen_text.strip(),
                    "reasoning_weight": REASONING_WEIGHTS.get(reasoning_type, 1.0)
                })
        
        return results
        
    except Exception as e:
        print(f"[WARN] Batch generation failed: {e}")
        return []

# ===========================
# COLLATE FUNCTION
# ===========================
def collate_fn(batch):
    return {
        "image": [item["image"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "question": [item["question"] for item in batch],
        "reasoning_type": [item["reasoning_type"] for item in batch]
    }

# ===========================
# MAIN LOOP
# ===========================
df = pd.read_csv(CSV_PATH)
dataset = VQADataset(df, IMAGE_DIR)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True  # Faster GPU transfer
)

results = []

for batch in tqdm(dataloader, desc="Teacher Generating"):
    batch_results = generate_batch(batch)
    results.extend(batch_results)
    
    # Save periodically to avoid losing progress
    if len(results) % 100 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Final save
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] ✅ Saved {len(results)} teacher samples → {OUT_JSONL}")
