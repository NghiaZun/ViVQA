"""
Day 6.4 — Evaluation Final: Teacher vs Student vs Ground Truth (Fixed Parsing)
Author: Hân (revised by ChatGPT)
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

DATA_PATH  = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_DIR  = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
SAVE_PATH  = "/kaggle/working/vqa_eval_teacher_student_fixed.csv"

TEACHER_ID    = "Qwen/Qwen2-VL-7B-Instruct"
STUDENT_CKPT  = "/kaggle/input/checkpoint/pytorch/default/1/vqa_student_best.pt"
PHOBERT_DIR   = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/phobert_tokenizer"
VIT5_DIR      = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/vit5_tokenizer"
VISION_MODEL  = "Salesforce/blip-vqa-base"

# Metrics
bleu      = load("bleu")
rouge     = load("rouge")
bertscore = load("bertscore")

# ===============================
# Helpers: Normalization & Parsing
# ===============================

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s

def strip_prompt_scaffold(text: str) -> str:
    """Loại bỏ khung 'Trả lời dạng: Answer: ... Reasoning: ...' nếu có."""
    if not isinstance(text, str):
        return ""
    # cắt phần scaffold nếu có
    text = re.sub(
        r"trả lời dạng\s*:\s*answer\s*:\s*\.\.\.\s*reasoning\s*:\s*\.\.\.\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    return text.strip()

def extract_last_answer(text: str) -> str:
    """
    Tìm TẤT CẢ 'Answer:' và lấy cái CUỐI CÙNG (tránh dính placeholder '...').
    Hỗ trợ 'Đáp án:' như synonym.
    """
    if not isinstance(text, str):
        return ""
    txt = strip_prompt_scaffold(text)
    # Gom theo dòng để tránh nuốt Reasoning
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    ans_list = []
    for line in lines:
        m = re.match(r"^(answer|đáp án)\s*[:：]\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            ans_list.append(m.group(2).strip())
    if ans_list:
        return ans_list[-1]

    # Fallback: pattern đa dòng (đến trước "Reasoning"/"Giải thích")
    m = re.findall(
        r"(?:^|\n)(?:answer|đáp án)\s*[:：]\s*(.*?)(?=\n(?:reasoning|giải thích|lý do)\s*[:：]|$)",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m[-1].strip()

    # Fallback cuối: nếu không có nhãn nào, trả toàn bộ text
    return txt.strip()

def extract_reasoning(text: str) -> str:
    """
    Lấy Reasoning sau nhãn 'Reasoning:' hoặc 'Giải thích:' hoặc 'Lý do:'.
    Lấy lần CUỐI cùng.
    """
    if not isinstance(text, str):
        return ""
    txt = strip_prompt_scaffold(text)

    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    rea_list = []
    for line in lines:
        m = re.match(r"^(reasoning|giải thích|lý do)\s*[:：]\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            rea_list.append(m.group(2).strip())
    if rea_list:
        return rea_list[-1]

    # Fallback đa dòng
    m = re.findall(
        r"(?:^|\n)(?:reasoning|giải thích|lý do)\s*[:：]\s*(.*)$",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m[-1].strip()

    return ""

def split_answer_reasoning(text: str):
    """
    Parser thống nhất cho cả teacher/student.
    - Có nhãn => lấy Answer/Reasoning (lần xuất hiện cuối).
    - Không nhãn => Answer = toàn văn, Reasoning = "".
    """
    if not isinstance(text, str):
        return "", ""
    ans = extract_last_answer(text)
    rea = extract_reasoning(text)
    # Trường hợp student chỉ trả lời ngắn không có nhãn
    if not ans and text:
        ans = text.strip()
    return ans.strip(), rea.strip()

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
    # Prompt ngắn gọn, tránh scaffold gây nhiễu
    prompt = (
        "Trả lời VQA bằng tiếng Việt, đúng format:\n"
        "Answer: <ngắn gọn>\n"
        "Reasoning: <ngắn gọn>\n\n"
        f"Câu hỏi: {question}\n"
        "Answer:"
    )
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
    q  = str(row["question"])
    gt = str(row["answer"]) if "answer" in row and pd.notna(row["answer"]) else ""
    img_path = os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg")

    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGB")

    t_out = infer_teacher(image, q)
    s_out = infer_student(image, q)

    # Robust parsing
    t_ans, t_rea = split_answer_reasoning(t_out)
    s_ans, s_rea = split_answer_reasoning(s_out)

    records.append({
        "img_id": row["img_id"],
        "question": q,
        "ground_truth": gt,
        "teacher_answer": t_ans,
        "teacher_reasoning": t_rea,
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
# Chỉ giữ hàng đủ dữ liệu
df_valid = df_out.fillna("").query("ground_truth.str.len() > 0 and teacher_answer.str.len() > 0 and student_answer.str.len() > 0")

gt            = [normalize_text(x) for x in df_valid["ground_truth"].tolist()]
teacher_preds = [normalize_text(x) for x in df_valid["teacher_answer"].tolist()]
student_preds = [normalize_text(x) for x in df_valid["student_answer"].tolist()]

# BLEU: references phải là List[List[str]]
refs_bleu = [[r] for r in gt]

teacher_bleu = bleu.compute(predictions=teacher_preds, references=refs_bleu, smooth=True)["bleu"]
student_bleu = bleu.compute(predictions=student_preds, references=refs_bleu, smooth=True)["bleu"]

teacher_rouge = rouge.compute(predictions=teacher_preds, references=gt)["rougeL"]
student_rouge = rouge.compute(predictions=student_preds, references=gt)["rougeL"]

bert_teacher = bertscore.compute(predictions=teacher_preds, references=gt, lang="vi")["f1"]
bert_student = bertscore.compute(predictions=student_preds, references=gt, lang="vi")["f1"]
teacher_bert = sum(bert_teacher) / max(1, len(bert_teacher))
student_bert = sum(bert_student) / max(1, len(bert_student))

print("\n=========== FINAL EVALUATION REPORT (FIXED) ===========")
print(f"BLEU            | Teacher: {teacher_bleu:.4f} | Student: {student_bleu:.4f}")
print(f"ROUGE-L         | Teacher: {teacher_rouge:.4f} | Student: {student_rouge:.4f}")
print(f"BERTScore (F1)  | Teacher: {teacher_bert:.4f} | Student: {student_bert:.4f}")
print("======================================================\n")

with open("/kaggle/working/metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "Teacher": {"BLEU": teacher_bleu, "ROUGE-L": teacher_rouge, "BERTScore_F1": teacher_bert},
            "Student": {"BLEU": student_bleu, "ROUGE-L": student_rouge, "BERTScore_F1": student_bert},
            "n_samples": len(df_valid)
        },
        f,
        indent=2,
        ensure_ascii=False
    )
print("[INFO] 📊 metrics_summary.json saved.")



