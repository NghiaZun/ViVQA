"""
Day 2 – Prepare Distillation Dataset from Teacher Outputs
Author: Hân & ChatGPT
"""

import os
import json
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================
# Config
# ==============================
IN_JSONL = "/kaggle/input/qwen2-vl-7b-instruct/Qwen2-VL-7B-Instruct.jsonl"  # đổi đúng đường dẫn
OUT_CLEAN = "/kaggle/working/cleaned_teacher.jsonl"
OUT_STATS = "/kaggle/working/teacher_reasoning_stats.csv"
OUT_TRAIN = "/kaggle/working/train_student.jsonl"
OUT_PLOT = "/kaggle/working/reasoning_distribution.png"

# ==============================
# 1️⃣ Load & Clean
# ==============================
print(f"[INFO] Loading {IN_JSONL}")
df = pd.read_json(IN_JSONL, lines=True)

# Giữ các cột cần thiết
cols = ["img_id", "image_path", "question", "teacher_answer", "teacher_reasoning"]
df = df[[c for c in cols if c in df.columns]].copy()

# Lọc mẫu hợp lệ
df = df[df["teacher_answer"].notna() & (df["teacher_answer"].str.strip() != "")]
df = df[df["teacher_reasoning"].notna() & (df["teacher_reasoning"].str.strip() != "")]
print(f"[INFO] ✅ {len(df)} valid samples remaining after cleaning")

# ==============================
# 2️⃣ Extract Reasoning Type
# ==============================
def extract_reasoning_type(text: str) -> str:
    text = text.strip()
    m = re.search(r'\((.*?)\)', text)
    if m:
        label = m.group(1).strip()
        if label.lower() in ["visual recognition", "spatial", "counting", "commonsense", "causal", "comparative"]:
            return label.title()
    # fallback rule-based
    if any(w in text.lower() for w in ["đếm", "số lượng", "bao nhiêu"]):
        return "Counting"
    if any(w in text.lower() for w in ["vì", "do", "nên", "nguyên nhân"]):
        return "Causal"
    if any(w in text.lower() for w in ["so với", "hơn", "ít hơn"]):
        return "Comparative"
    if any(w in text.lower() for w in ["trên", "dưới", "bên cạnh"]):
        return "Spatial"
    if any(w in text.lower() for w in ["màu", "áo", "vật", "xe", "hoa"]):
        return "Visual Recognition"
    return "Commonsense"

df["reasoning_type"] = df["teacher_reasoning"].apply(extract_reasoning_type)

# ==============================
# 3️⃣ Stats & Visualization
# ==============================
stats = df["reasoning_type"].value_counts()
stats.to_csv(OUT_STATS)
print(f"[INFO] 📊 Saved reasoning stats to {OUT_STATS}")

# Plot chart
plt.figure(figsize=(8, 4))
stats.plot(kind="bar", color="skyblue")
plt.title("Distribution of Reasoning Types")
plt.xlabel("Reasoning Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_PLOT)
print(f"[INFO] 🖼️ Saved chart to {OUT_PLOT}")

# ==============================
# 4️⃣ Build Student Training Data
# ==============================
train_samples = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building student dataset"):
    train_samples.append({
        "instruction": "Quan sát ảnh và trả lời câu hỏi ngắn gọn bằng tiếng Việt, kèm reasoning.",
        "input": f"Câu hỏi: {row['question']}\nẢnh: {row['image_path']}",
        "output": {
            "answer": row["teacher_answer"].strip(),
            "reasoning": row["teacher_reasoning"].strip(),
            "type": row["reasoning_type"]
        }
    })

# Lưu ra file JSONL
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for sample in train_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

df.to_json(OUT_CLEAN, orient="records", lines=True, force_ascii=False)

print("=" * 70)
print(f"[INFO] ✅ Cleaned teacher data: {OUT_CLEAN}")
print(f"[INFO] ✅ Student train file: {OUT_TRAIN}")
print(f"[INFO] ✅ Stats: {OUT_STATS}")
print(f"[INFO] ✅ Visualization: {OUT_PLOT}")
print(f"[INFO] Final sample count: {len(df)}")


