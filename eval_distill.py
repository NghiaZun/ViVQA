# eval_distill_fixed.py
"""
Evaluate distilled student model vs teacher (Qwen2-7b-Instruction).
- Student: VQAGenModel (vision encoder + PhoBERT + VietT5 decoder)
- Teacher: Qwen2-7b-Instruction (text-only). We provide image caption (BLIP) to teacher prompt.
- Both outputs must be in XLM:
    <answer>...</answer>
    <reasoning>[TYPE] ...</reasoning>

Outputs:
 - CSV with student and teacher extracted answer/reasoning/type + metrics
"""

import os
import re
import json
import torch
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from rouge_score import rouge_scorer, scoring

from model import VQAGenModel  # student model

# -------------------------
# Config
# -------------------------
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_BASE = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
STUDENT_CHECKPOINT = "/kaggle/input/checkpoints_2/transformers/default/1/checkpoints/vqa_student_best_multiKD.pt"
OUTPUT_CSV = "/kaggle/working/eval_distill_results.csv"
TEACHER_MODEL = "Qwen2-7b-Instruction"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STUDENT_BATCH = 8
TEACHER_BATCH = 2
MAX_TEACHER_TOKENS = 256
MAX_STUDENT_GEN = 64

# -------------------------
# Utils
# -------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gt_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

RE_ANSWER = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
RE_REASONING = re.compile(r"<reasoning>\s*(.+?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
RE_REASON_TYPE = re.compile(r"\[([A-Z]+)\]")

def extract_xlm(text: str) -> Dict[str, str]:
    text = text or ""
    ans_m = RE_ANSWER.search(text)
    ans = ans_m.group(1).strip() if ans_m else ""
    reason_m = RE_REASONING.search(text)
    reason = reason_m.group(1).strip() if reason_m else ""
    rtype = ""
    rt_m = RE_REASON_TYPE.search(reason)
    if rt_m:
        rtype = rt_m.group(1).strip()
        reason = re.sub(r"^\s*\[[A-Z]+\]\s*", "", reason)
    return {"answer": ans, "reasoning": reason, "type": rtype, "raw": text}

def make_empty_rouge():
    return {
        "rouge1": scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0),
        "rougeLsum": scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0)
    }

# -------------------------
# Load student
# -------------------------
print("[INFO] Loading student model...")
student = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)

if os.path.exists(STUDENT_CHECKPOINT):
    ck = torch.load(STUDENT_CHECKPOINT, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        student.load_state_dict(ck["model"])
    else:
        student.load_state_dict(ck)
else:
    print(f"[WARN] Student checkpoint not found. Using uninitialized weights.")

student.to(DEVICE)
student.eval()
student_decoder_tokenizer = student.decoder_tokenizer

# -------------------------
# BLIP captioner
# -------------------------
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
blip_model.eval()

def make_caption(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return ""
    inputs = blip_processor(images=img, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs, max_new_tokens=32)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# -------------------------
# Load teacher
# -------------------------
teacher_tokenizer, teacher_model, teacher_is_loaded = None, None, False
try:
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=False)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    teacher_is_loaded = True
except Exception as e:
    print(f"[WARN] Teacher load failed: {e}")
    try:
        teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=False)
        teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL)
        teacher_model.to(DEVICE)
        teacher_is_loaded = True
    except Exception as e2:
        print(f"[WARN] Teacher fully skipped: {e2}")
        teacher_is_loaded = False

def teacher_generate_from_caption_and_q(caption: str, question: str, max_new_tokens=MAX_TEACHER_TOKENS) -> str:
    prompt = (
        "You are a helpful VQA model. Use the image caption and the question to produce an answer.\n\n"
        f"Image caption: {caption}\n"
        f"Question: {question}\n\n"
        "Produce output exactly in this XLM format:\n<answer>...</answer>\n<reasoning>[DESCRIPTIVE|CAUSAL|NEUTRAL|OBJECT-BASED] ...</reasoning>\n"
    )
    inputs = teacher_tokenizer(prompt, return_tensors="pt", truncation=True).to(teacher_model.device)
    gen = teacher_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=getattr(teacher_tokenizer, "eos_token_id", None)
    )
    out_text = teacher_tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if out_text.strip() == "":
        out_text = teacher_tokenizer.decode(gen[0], skip_special_tokens=True)
    return out_text

# -------------------------
# Load test CSV
# -------------------------
df = pd.read_csv(TEST_CSV)
if "image_path" not in df.columns:
    df["image_path"] = df["img_id"].apply(lambda x: os.path.join(IMAGE_BASE, f"{x}.jpg"))
df["ground_truth"] = df["answer"].astype(str) if "answer" in df.columns else ""

# -------------------------
# Metrics
# -------------------------
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)
records = []

# -------------------------
# Main loop
# -------------------------
print("[INFO] Starting inference...")
for idx in tqdm(range(len(df))):
    row = df.iloc[idx]
    img_id, q, img_path, gt = row["img_id"], str(row["question"]), row["image_path"], str(row["ground_truth"])

    # 1) Caption
    caption = make_caption(img_path)

    # 2) Student
    q_enc = student.text_tokenizer(q, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception:
        img_pil = Image.new("RGB", (224, 224), (255, 255, 255))
    pix = blip_processor(images=img_pil, return_tensors="pt").pixel_values.to(DEVICE)

    try:
        out_ids = student.generate(pixel_values=pix, input_ids=q_enc.input_ids, attention_mask=q_enc.attention_mask, max_length=MAX_STUDENT_GEN)
        if isinstance(out_ids, torch.Tensor):
            out_ids = [out_ids.cpu().numpy()] if out_ids.ndim == 1 else [row for row in out_ids.cpu().numpy()]
        student_raw = student_decoder_tokenizer.decode(out_ids[0], skip_special_tokens=True) if out_ids else ""
    except Exception as e:
        student_raw = ""
        print(f"[WARN] Student generation failed idx {idx}: {e}")

    # 3) Teacher
    teacher_raw = ""
    if teacher_is_loaded:
        try:
            teacher_raw = teacher_generate_from_caption_and_q(caption, q, max_new_tokens=128)
        except Exception as e:
            teacher_raw = ""
            print(f"[WARN] Teacher generation failed idx {idx}: {e}")

    # 4) Extract XLM
    stu_ex, tea_ex = extract_xlm(student_raw), extract_xlm(teacher_raw)
    if stu_ex["answer"] == "" and student_raw.strip(): stu_ex["answer"] = student_raw.strip()
    if tea_ex["answer"] == "" and teacher_raw.strip(): tea_ex["answer"] = teacher_raw.strip()

    # 5) Metrics
    em_student = int(normalize_text(stu_ex["answer"]) == normalize_text(gt))
    em_teacher = int(normalize_text(tea_ex["answer"]) == normalize_text(gt))

    rouge_student = rouge.score(normalize_text(gt), normalize_text(stu_ex["answer"])) if stu_ex["answer"] else make_empty_rouge()
    rouge_teacher = rouge.score(normalize_text(gt), normalize_text(tea_ex["answer"])) if tea_ex["answer"] else make_empty_rouge()

    tokenf_student = token_f1(stu_ex["answer"], gt)
    tokenf_teacher = token_f1(tea_ex["answer"], gt)

    records.append({
        "img_id": img_id, "question": q, "ground_truth": gt, "caption": caption,
        "student_raw": student_raw, "student_answer": stu_ex["answer"], "student_reasoning": stu_ex["reasoning"], "student_type": stu_ex["type"],
        "student_em_answer": em_student, "student_tokenF1": tokenf_student,
        "student_rouge1": rouge_student["rouge1"].fmeasure, "student_rougeL": rouge_student["rougeLsum"].fmeasure,
        "teacher_raw": teacher_raw, "teacher_answer": tea_ex["answer"], "teacher_reasoning": tea_ex["reasoning"], "teacher_type": tea_ex["type"],
        "teacher_em_answer": em_teacher, "teacher_tokenF1": tokenf_teacher,
        "teacher_rouge1": rouge_teacher["rouge1"].fmeasure, "teacher_rougeL": rouge_teacher["rougeLsum"].fmeasure,
    })

# -------------------------
# Save CSV
# -------------------------
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"[INFO] Saved eval results to {OUTPUT_CSV}")

# -------------------------
# Aggregated metrics
# -------------------------
print("=== Summary ===")
print(f"Student EM (answer): {df_out['student_em_answer'].mean():.4f} | Teacher EM: {df_out['teacher_em_answer'].mean():.4f}")
print(f"Student ROUGE-1: {df_out['student_rouge1'].mean():.4f} | Teacher ROUGE-1: {df_out['teacher_rouge1'].mean():.4f}")
print(f"Student TokenF1: {df_out['student_tokenF1'].mean():.4f} | Teacher TokenF1: {df_out['teacher_tokenF1'].mean():.4f}")

# -------------------------
# Sample print
# -------------------------
print("\nSample predictions:")
for i in range(min(10, len(df_out))):
    r = df_out.iloc[i]
    print(f"{r.img_id} Q: {r.question}")
    print(f"  GT: {r.ground_truth}")
    print(f"  STUDENT: {r.student_answer}  | type={r.student_type}")
    print(f"    reasoning: {r.student_reasoning[:200]}")
    print(f"  TEACHER: {r.teacher_answer}  | type={r.teacher_type}")
    print(f"    reasoning: {r.teacher_reasoning[:200]}")
    print("-----")
