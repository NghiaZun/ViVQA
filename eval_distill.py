"""
Day 6.4 — Evaluation Final: Teacher vs Student vs Ground Truth
Author: Hân (revised and fixed by ChatGPT)
"""

import os
import re
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
SAVE_PATH = "/kaggle/working/vqa_eval_teacher_student_fixed.csv"

TEACHER_ID = "Qwen/Qwen2-VL-7B-Instruct"
STUDENT_CKPT = "/kaggle/input/checkpoint/pytorch/default/1/vqa_student_best.pt"
PHOBERT_DIR = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/phobert_tokenizer"
VIT5_DIR = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/vit5_tokenizer"
VISION_MODEL = "Salesforce/blip-vqa-base"

bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

# ===============================
# Helper functions
# ===============================
def extract_answer(text):
    if not isinstance(text, str):
        return ""
    m = re.search(r"Answer\s*[:：]\s*(.*?)(?:Reasoning|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

def split_answer_reasoning(text):
    if not isinstance(text, str):
        return "", ""
    ans = extract_answer(text)
    rea = ""
    m = re.search(r"Reasoning\s*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        rea = m.group(1).strip()
    return ans, rea

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

# ===============================
# Load Teacher
# ===============================
print("[INFO] Loading Teacher Model...")
teacher_processor = AutoProcessor.from_pretrained(TEACHER_ID, trust_remote_code=True)
teacher_model = AutoModelForVision2Seq.from_pretrained(
    TEACHER_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
teacher_model.eval()

def infer_teacher(image, question):
    prompt = f"Trả lời dạng:\nAnswer: ...\nReasoning: ...\nCâu hỏi: {question}\nAnswer:"
    inputs = teacher_processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = teacher_model.generate(
            **inputs, max_new_tokens=80, temperature=0.2, top_p=0.95
        )
    return teacher_processor.batch_decode(out, skip_special_tokens=True)[0].strip()

# ===============================
# Load Student
# ===============================
print("[INFO] Loading Student Model...")
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
    enc = student.text_tokenizer(question, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = student.generate(
            pixel_values=pixel_values,
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_length=80,
            num_beams=4
        )
    return student.decoder_tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

# ===============================
#  Inference
# ===============================
df = pd.read_csv(DATA_PATH).head(100)
records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    q = str(row["question"])
    gt = str(row["answer"]) if "answer" in row else ""
    img_path = os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg")

    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGB")

    t_out = infer_teacher(image, q)
    s_out = infer_student(image, q)

    s_ans, s_rea = split_answer_reasoning(s_out)
    t_ans = extract_answer(t_out)

    records.append({
        "img_id": row["img_id"],
        "question": q,
        "ground_truth": gt,
        "teacher_answer": t_ans,
        "student_answer": s_ans,
        "student_reasoning": s_rea,
        "teacher_output_raw": t_out,
        "student_output_raw": s_out
    })

df_out = pd.DataFrame(records)
df_out.to_csv(SAVE_PATH, index=False)
print(f"[INFO] ✅ Predictions saved → {SAVE_PATH}")

# ===============================
#  Compute Metrics
# ===============================
df_valid = df_out[
    (df_out["ground_truth"].str.len() > 0)
    & (df_out["teacher_answer"].str.len() > 0)
    & (df_out["student_answer"].str.len() > 0)
]

gt = df_valid["ground_truth"].apply(normalize_text).tolist()
teacher_preds = df_valid["teacher_answer"].apply(normalize_text).tolist()
student_preds = df_valid["student_answer"].apply(normalize_text).tolist()

teacher_bleu = bleu.compute(predictions=teacher_preds, references=gt, smooth=True)["bleu"]
student_bleu = bleu.compute(predictions=student_preds, references=gt, smooth=True)["bleu"]

teacher_rouge = rouge.compute(predictions=teacher_preds, references=gt)["rougeL"]
student_rouge = rouge.compute(predictions=student_preds, references=gt)["rougeL"]

teacher_bert = sum(bertscore.compute(predictions=teacher_preds, references=gt, lang="vi")["f1"]) / len(gt)
student_bert = sum(bertscore.compute(predictions=student_preds, references=gt, lang="vi")["f1"]) / len(gt)

print("\n=========== FINAL EVAL ===========")
print(f"BLEU:     Teacher={teacher_bleu:.4f} | Student={student_bleu:.4f}")
print(f"ROUGE-L:  Teacher={teacher_rouge:.4f} | Student={student_rouge:.4f}")
print(f"BERT-F1:  Teacher={teacher_bert:.4f} | Student={student_bert:.4f}")
print("=================================\n")

json.dump(
    {
        "Teacher": {"BLEU": teacher_bleu, "ROUGE-L": teacher_rouge, "BERTScore": teacher_bert},
        "Student": {"BLEU": student_bleu, "ROUGE-L": student_rouge, "BERTScore": student_bert},
    },
    open("/kaggle/working/metrics_summary.json", "w"),
    indent=2,
    ensure_ascii=False
)
print("[INFO] 📊 metrics_summary.json saved.")
