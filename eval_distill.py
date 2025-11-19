# eval_distill_student_fixed.py
"""
Batch evaluation for VQA student model (ViVQA test set)
Supports GPU batch processing for faster inference
"""
import os
import re
import unicodedata
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer, scoring

from model import VQAGenModel

# -------------------------
# Config
# -------------------------
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_BASE = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
STUDENT_CHECKPOINT = "/kaggle/input/final/transformers/default/1/vqa_student_best_multiKD.pt"
OUTPUT_CSV = "/kaggle/working/eval_student_batch_fixed.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
MAX_SEQ_LEN = 128
MAX_GEN_LEN = 64

# -------------------------
# Utils: normalization / token-f1
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
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gt_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def make_empty_rouge():
    return {
        "rouge1": scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0),
        "rougeLsum": scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0)
    }

# -------------------------
# XML parser
# -------------------------
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
RE_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
RE_REASON_TYPE = re.compile(r"^\s*\[([A-Za-z0-9_\-]+)\]\s*(.*)", re.DOTALL)

def parse_vqa_xml(output_text: str):
    text = (output_text or "").strip()
    ans_m = RE_ANSWER.search(text)
    ans = ans_m.group(1).strip() if ans_m else ""
    reason_m = RE_REASONING.search(text)
    reason_raw = reason_m.group(1).strip() if reason_m else ""
    rtype = ""
    reasoning = reason_raw
    if reason_raw:
        t_m = RE_REASON_TYPE.match(reason_raw)
        if t_m:
            rtype = t_m.group(1).strip()
            reasoning = t_m.group(2).strip()
    if ans == "" and text:
        cleaned = RE_REASONING.sub("", text).strip()
        if cleaned:
            ans = cleaned
    return {"answer": ans, "reasoning": reasoning, "type": rtype, "raw": text}

# -------------------------
# Dataset
# -------------------------
class VQABatchDataset(Dataset):
    def __init__(self, csv_file, image_base, processor, tokenizer, max_len=MAX_SEQ_LEN):
        self.df = pd.read_csv(csv_file)
        if "image_path" not in self.df.columns:
            self.df["image_path"] = self.df["img_id"].apply(lambda x: os.path.join(image_base, f"{x}.jpg"))
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df["ground_truth"] = self.df["answer"].astype(str) if "answer" in self.df.columns else ""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_enc = self.tokenizer(
            row["question"], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        try:
            img_pil = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img_pil = Image.new("RGB", (224,224), (255,255,255))
        pix = self.processor(images=img_pil, return_tensors="pt").pixel_values.squeeze(0)  # FIXED
        return {
            "pixel_values": pix,
            "input_ids": q_enc["input_ids"].squeeze(0),
            "attention_mask": q_enc["attention_mask"].squeeze(0),
            "ground_truth": row["ground_truth"],
            "img_id": row["img_id"],
            "question": row["question"]
        }

# -------------------------
# Load student model
# -------------------------
print("[INFO] Loading student model...")
student = VQAGenModel()
ck = torch.load(STUDENT_CHECKPOINT, map_location="cpu")
if isinstance(ck, dict) and "model" in ck:
    student.load_state_dict(ck["model"])
else:
    student.load_state_dict(ck)
student.to(DEVICE)
student.eval()

q_tokenizer = student.text_tokenizer
decoder_tokenizer = student.decoder_tokenizer
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# -------------------------
# DataLoader
# -------------------------
dataset = VQABatchDataset(TEST_CSV, IMAGE_BASE, processor, q_tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Rouge scorer
# -------------------------
rouge = rouge_scorer.RougeScorer(["rouge1","rougeLsum"], use_stemmer=True)
records = []

# -------------------------
# Evaluation loop
# -------------------------
print("[INFO] Starting batch evaluation...")
with torch.no_grad():
    for batch in tqdm(loader):
        pix = batch["pixel_values"].to(DEVICE)  # [B,3,H,W]
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        gts = batch["ground_truth"]
        img_ids = batch["img_id"]
        questions = batch["question"]

        # Generate student outputs
        generated_ids = student.generate(
            pixel_values=pix,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_GEN_LEN,
            num_beams=4
        )

        # Decode & compute metrics
        for i, ids in enumerate(generated_ids):
            stu_text = decoder_tokenizer.decode(ids, skip_special_tokens=True) if ids is not None else ""
            stu_ex = parse_vqa_xml(stu_text)
            if stu_ex["answer"] == "" and stu_text.strip():
                stu_ex["answer"] = stu_text.strip()
            em = int(normalize_text(stu_ex["answer"]) == normalize_text(gts[i]))
            tf1 = token_f1(stu_ex["answer"], gts[i])
            rouge_scores = rouge.score(normalize_text(gts[i]), normalize_text(stu_ex["answer"])) if stu_ex["answer"] else make_empty_rouge()
            records.append({
                "img_id": img_ids[i],
                "question": questions[i],
                "ground_truth": gts[i],
                "student_raw": stu_ex["raw"],
                "student_answer": stu_ex["answer"],
                "student_reasoning": stu_ex["reasoning"],
                "student_type": stu_ex["type"],
                "student_em_answer": em,
                "student_tokenF1": tf1,
                "student_rouge1": rouge_scores["rouge1"].fmeasure,
                "student_rougeL": rouge_scores["rougeLsum"].fmeasure
            })

# -------------------------
# Save CSV & summary
# -------------------------
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"[INFO] Saved batch eval results to {OUTPUT_CSV}")

print("=== Summary ===")
print(f"Student EM (answer): {df_out['student_em_answer'].mean():.4f}")
print(f"Student ROUGE-1: {df_out['student_rouge1'].mean():.4f}")
print(f"Student TokenF1: {df_out['student_tokenF1'].mean():.4f}")

# Sample predictions
for i in range(min(5,len(df_out))):
    r = df_out.iloc[i]
    print(f"{r.img_id} Q:{r.question}")
    print(f"  GT: {r.ground_truth}")
    print(f"  STUDENT: {r.student_answer} | type={r.student_type}")
    print(f"    reasoning: {r.student_reasoning[:200]}")
    print("-----")
