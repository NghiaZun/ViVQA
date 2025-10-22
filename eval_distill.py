"""
Day 6.4 — Evaluation Final: Teacher vs Student vs Ground Truth
Author: Hân
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BlipProcessor
from model import VQAGenModel
from evaluate import load

# ===============================
# CONFIG
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

DATA_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
SAVE_PATH = "/kaggle/working/vqa_eval_teacher_student.csv"

# Model paths
TEACHER_ID = "Qwen/Qwen2-VL-7B-Instruct"
STUDENT_CKPT = "/kaggle/input/vqa-student/vqa_student_best.pt"
PHOBERT_DIR = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/phobert_tokenizer"
VIT5_DIR = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/vit5_tokenizer"
VISION_MODEL = "Salesforce/blip-vqa-base"

# Load metrics
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

# ===============================
# 1️⃣ Load Teacher
# ===============================
print("[INFO] Loading Teacher (Qwen2-VL-7B-Instruct)...")
teacher_processor = AutoProcessor.from_pretrained(TEACHER_ID, trust_remote_code=True)
teacher_model = AutoModelForVision2Seq.from_pretrained(
    TEACHER_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
teacher_model.eval()

def infer_teacher(image, question):
    messages = [
        {"role": "system", "content": "Bạn là mô hình VQA tiếng Việt, trả lời có reasoning."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Trả lời dạng:\nAnswer: ...\nReasoning: ...\nCâu hỏi: {question}"}
        ]}
    ]
    text_prompt = teacher_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = teacher_processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        output = teacher_model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
    return teacher_processor.batch_decode(output, skip_special_tokens=True)[0].strip()

# ===============================
# 2️⃣ Load Student
# ===============================
print("[INFO] Loading Student...")
vision_processor = BlipProcessor.from_pretrained(VISION_MODEL)
student = VQAGenModel(
    vision_model_name=VISION_MODEL,
    phobert_dir=PHOBERT_DIR,
    vit5_dir=VIT5_DIR
)
student.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))
student.to(device)
student.eval()

def infer_student(image, question):
    pixel_values = vision_processor(image, return_tensors="pt").pixel_values.to(device)
    q_enc = student.text_tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output_ids = student.generate(
            pixel_values=pixel_values,
            input_ids=q_enc.input_ids,
            attention_mask=q_enc.attention_mask,
            max_length=64,
            num_beams=4
        )
    return student.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

# ===============================
# 3️⃣ Inference on Test Set
# ===============================
df = pd.read_csv(DATA_PATH)
df = df.head(100)  # test subset for Kaggle runtime
records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    q = str(row["question"])
    gt = str(row["answer"]) if "answer" in row else ""
    img_id = str(row["img_id"])
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")

    try:
        image = Image.open(img_path).convert("RGB")
    except:
        continue

    t_out = infer_teacher(image, q)
    s_out = infer_student(image, q)

    records.append({
        "img_id": img_id,
        "question": q,
        "ground_truth": gt,
        "teacher_output": t_out,
        "student_output": s_out
    })

df_out = pd.DataFrame(records)
df_out.to_csv(SAVE_PATH, index=False)
print(f"[INFO] ✅ Saved predictions → {SAVE_PATH}")

# ===============================
# 4️⃣ Compute Metrics (per model)
# ===============================
print("[INFO] Computing metrics...")

ground_truths = df_out["ground_truth"].tolist()
teacher_preds = df_out["teacher_output"].tolist()
student_preds = df_out["student_output"].tolist()

# Teacher vs GT
teacher_bleu = bleu.compute(predictions=teacher_preds, references=ground_truths)["bleu"]
teacher_rouge = rouge.compute(predictions=teacher_preds, references=ground_truths)["rougeL"]
teacher_bert = sum(load("bertscore").compute(predictions=teacher_preds, references=ground_truths, lang="vi")["f1"]) / len(ground_truths)

# Student vs GT
student_bleu = bleu.compute(predictions=student_preds, references=ground_truths)["bleu"]
student_rouge = rouge.compute(predictions=student_preds, references=ground_truths)["rougeL"]
student_bert = sum(load("bertscore").compute(predictions=student_preds, references=ground_truths, lang="vi")["f1"]) / len(ground_truths)

# ===============================
# 5️⃣ Print Summary
# ===============================
print("\n=========== FINAL EVALUATION REPORT ===========")
print(f"{'Metric':<15} | {'Teacher':>10} | {'Student':>10}")
print("-" * 42)
print(f"{'BLEU':<15} | {teacher_bleu:>10.4f} | {student_bleu:>10.4f}")
print(f"{'ROUGE-L':<15} | {teacher_rouge:>10.4f} | {student_rouge:>10.4f}")
print(f"{'BERTScore (F1)':<15} | {teacher_bert:>10.4f} | {student_bert:>10.4f}")
print("===============================================\n")

summary = {
    "Teacher": {"BLEU": teacher_bleu, "ROUGE-L": teacher_rouge, "BERTScore": teacher_bert},
    "Student": {"BLEU": student_bleu, "ROUGE-L": student_rouge, "BERTScore": student_bert}
}
json.dump(summary, open("/kaggle/working/metrics_summary.json", "w"), indent=2, ensure_ascii=False)
print("[INFO] 📊 Saved metrics_summary.json for report.")


