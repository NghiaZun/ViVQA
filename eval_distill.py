# eval_distill.py
"""
Evaluate distilled student model vs teacher (Qwen2-7b-Instruction).
- Student: VQAGenModel (uses vision encoder + PhoBERT + VietT5 decoder)
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
from typing import Tuple, Dict, List

# transform & model libs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration
)

# metrics
from rouge_score import rouge_scorer

# your local imports (adjust path if needed)
from model import VQAGenModel  # <-- ensure this is importable in same folder

# -------------------------
# Config - change paths as needed
# -------------------------
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"  # test.csv with columns including img_id, image_path, question, answer
IMAGE_BASE = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
STUDENT_CHECKPOINT = "/kaggle/input/checkpoints_2/transformers/default/1/checkpoints/vqa_student_best_multiKD.pt"  # change if needed
OUTPUT_CSV = "/kaggle/working/eval_distill_results.csv"

# Teacher model identifier / local path
TEACHER_MODEL = "Qwen2-7b-Instruction"  # or local path to weights
# If Qwen model is in HF hub, replace with exact repo id. If local path, set that path.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference parameters
STUDENT_BATCH = 8    # student generation batch (small if GPU mem low)
TEACHER_BATCH = 2    # teacher generation batch (Qwen large -> small)
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
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# Regex extraction (robust)
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
        # optionally remove the [TYPE] prefix from reasoning text
        reason = re.sub(r"^\s*\[[A-Z]+\]\s*", "", reason)
    return {"answer": ans, "reasoning": reason, "type": rtype, "raw": text}

# -------------------------
# Load student model
# -------------------------
print("[INFO] Loading student model...")
student = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)
# load weights if checkpoint provided (state_dict)
if os.path.exists(STUDENT_CHECKPOINT):
    ck = torch.load(STUDENT_CHECKPOINT, map_location="cpu")
    # ck may be either state_dict or dict
    if isinstance(ck, dict) and "model" in ck:
        student.load_state_dict(ck["model"])
    else:
        student.load_state_dict(ck)
else:
    print(f"[WARN] Student checkpoint not found at {STUDENT_CHECKPOINT}. Using uninitialized weights.")

student.to(DEVICE)
student.eval()
student_decoder_tokenizer = student.decoder_tokenizer  # decoder tokenizer for decoding outputs

# -------------------------
# Prepare BLIP captioner (to make short image caption for teacher)
# -------------------------
print("[INFO] Loading BLIP image captioner (for teacher prompts)...")
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
# Load teacher model (Qwen2-7b-Instruction)
# -------------------------
print(f"[INFO] Loading teacher model: {TEACHER_MODEL} ...")
teacher_tokenizer = None
teacher_model = None
teacher_is_loaded = False
try:
    # Try efficient loading on GPU (may require HF token and model availability)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=False)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    teacher_is_loaded = True
    print("[INFO] Teacher loaded on device_map='auto' (float16).")
except Exception as e:
    print(f"[WARN] Could not load teacher with device_map auto: {e}")
    try:
        teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=False)
        teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL)
        teacher_model.to(DEVICE)
        teacher_is_loaded = True
        print("[INFO] Loaded teacher model to device.")
    except Exception as e2:
        print(f"[WARN] Could not load teacher model at all: {e2}. Teacher inference will be skipped.")
        teacher_is_loaded = False

# teacher generate helper (text-only)
def teacher_generate_from_caption_and_q(caption: str, question: str, max_new_tokens=MAX_TEACHER_TOKENS) -> str:
    """
    Build a prompt combining caption and question, ask teacher to respond in XLM format.
    Returns the raw teacher text.
    """
    prompt = (
        "You are a helpful VQA model. Use the image caption and the question to produce an answer.\n\n"
        f"Image caption: {caption}\n"
        f"Question: {question}\n\n"
        "Produce output exactly in this XLM format (no extra commentary):\n"
        "<answer>...</answer>\n"
        "<reasoning>[DESCRIPTIVE|CAUSAL|NEUTRAL|OBJECT-BASED] ...</reasoning>\n\n"
        "Only output those two tags and content. Keep answer concise.\n"
    )
    inputs = teacher_tokenizer(prompt, return_tensors="pt", truncation=True).to(teacher_model.device)
    # generation
    gen = teacher_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=teacher_tokenizer.eos_token_id if hasattr(teacher_tokenizer, "eos_token_id") else None
    )
    out_text = teacher_tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # If decode returns empty, fallback to full decode
    if out_text.strip() == "":
        out_text = teacher_tokenizer.decode(gen[0], skip_special_tokens=True)
    return out_text

# -------------------------
# Load test CSV
# -------------------------
print("[INFO] Loading test CSV...")
df = pd.read_csv(TEST_CSV)
# Expect columns: img_id, question, answer (ground truth)
required_cols = ["img_id", "question"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"test csv must contain column '{c}'")

# If image path column not present, construct using image id and IMAGE_BASE
if "image_path" not in df.columns:
    def _ip(row):
        # try to build path via img_id + .jpg
        img_id = str(row["img_id"])
        cand = os.path.join(IMAGE_BASE, f"{img_id}.jpg")
        return cand
    df["image_path"] = df.apply(_ip, axis=1)

# ground truth: if 'answer' column exists use it; else empty
if "answer" in df.columns:
    df["ground_truth"] = df["answer"].astype(str)
else:
    df["ground_truth"] = ""

# -------------------------
# Inference loops
# -------------------------
records = []
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)

print("[INFO] Starting inference on test set...")
for idx in tqdm(range(len(df))):
    row = df.iloc[idx]
    img_id = row["img_id"]
    q = str(row["question"])
    img_path = row["image_path"]
    gt = str(row["ground_truth"]) if not pd.isna(row["ground_truth"]) else ""

    # 1) Caption image (BLIP)
    caption = ""
    try:
        caption = make_caption(img_path)
    except Exception as e:
        caption = ""
        # keep going

    # 2) Student inference
    # prepare inputs for student. VQAGenModel expects vision tensor and tokenized question
    # We will call student.generate(...) which returns token ids in original class
    # Build tokenized question for student's text encoder (PhoBERT). Use student.text_tokenizer
    q_enc = student.text_tokenizer(q, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    # pixel
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), (255, 255, 255))
    pix = student.vision_encoder  # but we need processed pixel_values; use BLIP processor? Student used BLIP processor in training; here we can reuse blip_processor
    proc = blip_processor(images=img, return_tensors="pt")
    pixel_values = proc.pixel_values.to(DEVICE)

    # call student's generate (the class provides .generate)
    try:
        # student.generate uses inputs_embeds generation inside model; returns list of ids
        out_ids = student.generate(pixel_values=pixel_values, input_ids=q_enc.input_ids.to(DEVICE), attention_mask=q_enc.attention_mask.to(DEVICE), max_length=MAX_STUDENT_GEN)
        # decode - may be batch of sequences; ensure list
        if isinstance(out_ids, torch.Tensor):
            out_ids = out_ids.cpu().numpy()
            # if 2D -> list of arrays
            if out_ids.ndim == 1:
                out_ids = [out_ids]
            else:
                out_ids = [row for row in out_ids]
        # decode first sequence
        student_raw = student_decoder_tokenizer.decode(out_ids[0], skip_special_tokens=True) if len(out_ids) > 0 else ""
    except Exception as e:
        student_raw = ""
        print(f"[WARN] Student generation failed at idx {idx}: {e}")

    # 3) Teacher inference (text-only) using caption + question
    teacher_raw = ""
    if teacher_is_loaded:
        try:
            teacher_raw = teacher_generate_from_caption_and_q(caption, q, max_new_tokens=128)
        except Exception as e:
            teacher_raw = ""
            print(f"[WARN] Teacher gen failed at idx {idx}: {e}")
    else:
        teacher_raw = ""  # skip

    # 4) Extract XLM fields
    stu_ex = extract_xlm(student_raw)
    tea_ex = extract_xlm(teacher_raw)

    # If student didn't produce XLM, try to coerce: if student output looks like plain answer, wrap into tags
    if stu_ex["answer"] == "" and student_raw.strip() != "":
        stu_ex["answer"] = student_raw.strip()
    if tea_ex["answer"] == "" and teacher_raw.strip() != "":
        tea_ex["answer"] = teacher_raw.strip()

    # 5) Metrics for this example (compare answer only to ground truth)
    em_student = (normalize_text(stu_ex["answer"]) == normalize_text(gt))
    em_teacher = (normalize_text(tea_ex["answer"]) == normalize_text(gt))

    rouge_student = rouge.score(normalize_text(gt), normalize_text(stu_ex["answer"])) if stu_ex["answer"] else {"rouge1":0.0, "rougeLsum":0.0}
    rouge_teacher = rouge.score(normalize_text(gt), normalize_text(tea_ex["answer"])) if tea_ex["answer"] else {"rouge1":0.0, "rougeLsum":0.0}

    tokenf_student = token_f1(stu_ex["answer"], gt)
    tokenf_teacher = token_f1(tea_ex["answer"], gt)

    records.append({
        "img_id": img_id,
        "question": q,
        "ground_truth": gt,
        "caption": caption,
        # student raw + parsed
        "student_raw": student_raw,
        "student_answer": stu_ex["answer"],
        "student_reasoning": stu_ex["reasoning"],
        "student_type": stu_ex["type"],
        "student_em_answer": int(em_student),
        "student_tokenF1": tokenf_student,
        "student_rouge1": rouge_student["rouge1"].fmeasure if "rouge1" in rouge_student else 0.0,
        "student_rougeL": rouge_student["rougeLsum"].fmeasure if "rougeLsum" in rouge_student else 0.0,
        # teacher raw + parsed
        "teacher_raw": teacher_raw,
        "teacher_answer": tea_ex["answer"],
        "teacher_reasoning": tea_ex["reasoning"],
        "teacher_type": tea_ex["type"],
        "teacher_em_answer": int(em_teacher),
        "teacher_tokenF1": tokenf_teacher,
        "teacher_rouge1": rouge_teacher["rouge1"].fmeasure if "rouge1" in rouge_teacher else 0.0,
        "teacher_rougeL": rouge_teacher["rougeLsum"].fmeasure if "rougeLsum" in rouge_teacher else 0.0,
    })

# Save CSV
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"[INFO] Saved eval results to {OUTPUT_CSV}")

# Compute aggregated metrics and print
def safe_mean(arr):
    return float(np.mean([a for a in arr if a is not None])) if len(arr) > 0 else 0.0

student_em = df_out["student_em_answer"].mean()
teacher_em = df_out["teacher_em_answer"].mean()
print("=== Summary ===")
print(f"Student EM (answer): {student_em:.4f}  | Teacher EM (answer): {teacher_em:.4f}")
print(f"Student avg ROUGE-1: {df_out['student_rouge1'].mean():.4f} | Teacher: {df_out['teacher_rouge1'].mean():.4f}")
print(f"Student avg TokenF1: {df_out['student_tokenF1'].mean():.4f} | Teacher: {df_out['teacher_tokenF1'].mean():.4f}")

# Print some samples
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
