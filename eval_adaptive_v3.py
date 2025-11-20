"""
Enhanced Evaluation Script for Adaptive VQA Model
Robust XML parsing + Post-processing + Better metrics
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
STUDENT_CHECKPOINT = "/kaggle/working/vqa_student_best_adaptive_v3.pt"
OUTPUT_CSV = "/kaggle/working/eval_adaptive_v3_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimized for Kaggle GPU
BATCH_SIZE = 4  # Reduced for memory during inference
MAX_SEQ_LEN = 64
MAX_GEN_LEN = 128
NUM_BEAMS = 4  # Reduced from 5 for memory
TEMPERATURE = 0.9
USE_CACHE = True  # Enable KV cache for faster inference

# -------------------------
# Enhanced Text Normalization
# -------------------------
def normalize_text(s: str) -> str:
    if s is None or not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    # Remove special chars but keep Vietnamese characters
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
# Enhanced XML Parser with Fallback
# -------------------------
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
RE_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
RE_REASON_TYPE = re.compile(r"^\s*\[([A-Za-z0-9_\-]+)\]\s*(.*)", re.DOTALL)

def parse_vqa_output(output_text: str):
    """
    Enhanced parser with multiple fallback strategies
    """
    text = (output_text or "").strip()
    
    # Try to extract answer
    ans_m = RE_ANSWER.search(text)
    answer = ans_m.group(1).strip() if ans_m else ""
    
    # Try to extract reasoning
    reason_m = RE_REASONING.search(text)
    reasoning_raw = reason_m.group(1).strip() if reason_m else ""
    
    # Extract reasoning type
    rtype = ""
    reasoning = reasoning_raw
    if reasoning_raw:
        t_m = RE_REASON_TYPE.match(reasoning_raw)
        if t_m:
            rtype = t_m.group(1).strip()
            reasoning = t_m.group(2).strip()
    
    # Fallback 1: If no answer but has reasoning, extract from reasoning
    if not answer and reasoning:
        # Try to find answer-like patterns in reasoning
        sentences = reasoning.split('.')
        if sentences:
            # First sentence often contains the answer
            potential_answer = sentences[0].strip()
            if len(potential_answer) < 50:  # Reasonable answer length
                answer = potential_answer
    
    # Fallback 2: If still no answer, extract from raw text
    if not answer:
        # Remove reasoning tags and extract remaining text
        cleaned = RE_REASONING.sub("", text).strip()
        cleaned = RE_ANSWER.sub("", cleaned).strip()
        if cleaned and len(cleaned) < 100:
            answer = cleaned
    
    # Fallback 3: If ONLY reasoning exists, try semantic extraction
    if not answer and text:
        # Look for key Vietnamese answer patterns
        patterns = [
            r"(?:là|có)\s+([^\.]+?)(?:\.|$)",  # "là X" or "có X"
            r"^([^\.]+?)\s+(?:trong|ở|tại)",    # "X trong/ở/tại..."
            r"(?:màu|số|loại)\s+([^\.]+?)(?:\.|$)"  # "màu X", "số X", "loại X"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
    
    # Last resort: use first 50 chars
    if not answer and text:
        answer = text[:50].split('.')[0].strip()
    
    return {
        "answer": answer,
        "reasoning": reasoning,
        "type": rtype,
        "raw": text,
        "has_format": bool(ans_m or reason_m)
    }

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
        pix = self.processor(images=img_pil, return_tensors="pt").pixel_values.squeeze(0)
        return {
            "pixel_values": pix,
            "input_ids": q_enc["input_ids"].squeeze(0),
            "attention_mask": q_enc["attention_mask"].squeeze(0),
            "ground_truth": row["ground_truth"],
            "img_id": row["img_id"],
            "question": row["question"]
        }

# -------------------------
# Load Model
# -------------------------
import gc

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("[INFO] Loading adaptive student model...")
print(f"[INFO] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

clear_memory()

student = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)

print("[INFO] Loading checkpoint...")
ckpt = torch.load(STUDENT_CHECKPOINT, map_location="cpu")  # Load to CPU first
if isinstance(ckpt, dict) and "model" in ckpt:
    student.load_state_dict(ckpt["model"])
else:
    student.load_state_dict(ckpt)
del ckpt
clear_memory()

student.to(DEVICE)
student.eval()

if torch.cuda.is_available():
    print(f"[INFO] GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

q_tokenizer = student.text_tokenizer
decoder_tokenizer = student.decoder_tokenizer
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# -------------------------
# DataLoader
# -------------------------
dataset = VQABatchDataset(TEST_CSV, IMAGE_BASE, processor, q_tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# Rouge Scorer
# -------------------------
rouge = rouge_scorer.RougeScorer(["rouge1","rougeLsum"], use_stemmer=True)
records = []

# -------------------------
# Evaluation Loop
# -------------------------
print(f"[INFO] Starting evaluation on {len(dataset)} samples...")
print(f"[INFO] Generation config: max_len={MAX_GEN_LEN}, beams={NUM_BEAMS}")

format_correct_count = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        pix = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        gts = batch["ground_truth"]
        img_ids = batch["img_id"]
        questions = batch["question"]

        # Generate with adaptive parameters
        generated_ids = student.generate(
            pixel_values=pix,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_GEN_LEN,
            num_beams=NUM_BEAMS,
            temperature=TEMPERATURE,
            do_sample=False,
            early_stopping=True,
            use_cache=USE_CACHE
        )
        
        # Clear cache every 20 batches
        if (batch_idx + 1) % 20 == 0:
            clear_memory()

        # Process each output
        for i, ids in enumerate(generated_ids):
            raw_text = decoder_tokenizer.decode(ids, skip_special_tokens=True) if ids is not None else ""
            parsed = parse_vqa_output(raw_text)
            
            # Track format correctness
            if parsed["has_format"]:
                format_correct_count += 1
            
            # Compute metrics
            pred_answer = parsed["answer"]
            gt_answer = gts[i]
            
            em = int(normalize_text(pred_answer) == normalize_text(gt_answer))
            tf1 = token_f1(pred_answer, gt_answer)
            
            # Rouge scores
            if pred_answer:
                rouge_scores = rouge.score(normalize_text(gt_answer), normalize_text(pred_answer))
            else:
                rouge_scores = make_empty_rouge()
            
            records.append({
                "img_id": img_ids[i],
                "question": questions[i],
                "ground_truth": gt_answer,
                "predicted_raw": parsed["raw"],
                "predicted_answer": pred_answer,
                "predicted_reasoning": parsed["reasoning"],
                "predicted_type": parsed["type"],
                "has_correct_format": int(parsed["has_format"]),
                "exact_match": em,
                "token_f1": tf1,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rougeL": rouge_scores["rougeLsum"].fmeasure
            })

# -------------------------
# Save Results
# -------------------------
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[INFO] ✅ Results saved to {OUTPUT_CSV}")

# -------------------------
# Summary Statistics
# -------------------------
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"Total samples:           {len(df_out)}")
print(f"Format correctness:      {format_correct_count}/{len(df_out)} ({100*format_correct_count/len(df_out):.2f}%)")
print(f"Exact Match (EM):        {df_out['exact_match'].mean():.4f}")
print(f"Token F1:                {df_out['token_f1'].mean():.4f}")
print(f"ROUGE-1:                 {df_out['rouge1'].mean():.4f}")
print(f"ROUGE-L:                 {df_out['rougeL'].mean():.4f}")
print("="*60)

# -------------------------
# Show Sample Predictions
# -------------------------
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

# Show 5 correct predictions
correct_samples = df_out[df_out['exact_match'] == 1].head(5)
if len(correct_samples) > 0:
    print("\n✅ CORRECT PREDICTIONS:")
    for idx, row in correct_samples.iterrows():
        print(f"\nQ: {row['question']}")
        print(f"GT: {row['ground_truth']}")
        print(f"PRED: {row['predicted_answer']} [Format OK: {bool(row['has_correct_format'])}]")
        if row['predicted_reasoning']:
            print(f"REASONING: {row['predicted_reasoning'][:100]}...")

# Show 5 incorrect predictions for analysis
incorrect_samples = df_out[df_out['exact_match'] == 0].head(5)
if len(incorrect_samples) > 0:
    print("\n❌ INCORRECT PREDICTIONS (for analysis):")
    for idx, row in incorrect_samples.iterrows():
        print(f"\nQ: {row['question']}")
        print(f"GT: {row['ground_truth']}")
        print(f"PRED: {row['predicted_answer']} [Format OK: {bool(row['has_correct_format'])}]")
        print(f"RAW: {row['predicted_raw'][:150]}...")

print("\n" + "="*60)
