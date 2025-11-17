import os
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import unicodedata

from transformers import AutoTokenizer, BlipImageProcessor
from rouge_score import rouge_scorer

from dataset import VQAGenDataset
from model import VQAGenModel


# =============================
# NORMALIZATION + METRICS
# =============================
def normalize_text(s):
    if s is None:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def token_f1(pred, gt):
    pred_t = normalize_text(pred).split()
    gt_t = normalize_text(gt).split()
    if len(pred_t) == 0 or len(gt_t) == 0:
        return 0
    common = set(pred_t) & set(gt_t)
    if len(common) == 0:
        return 0
    p = len(common) / len(pred_t)
    r = len(common) / len(gt_t)
    return 2 * p * r / (p + r)


# =============================
# PARSE STUDENT XLM
# =============================
def parse_student_output(text):
    """
    Student output format expected:
      <answer type="xxx"> ... </answer>
      <reasoning> ... </reasoning>
    """
    if text is None:
        return "", "", ""

    text = text.strip()

    # answer
    ans = re.search(r"<answer(?: type=\"(.*?)\")?>(.*?)</answer>", text, flags=re.DOTALL)
    if ans:
        answer_type = ans.group(1) if ans.group(1) else ""
        answer = ans.group(2).strip()
    else:
        answer = text.strip()
        answer_type = ""

    # reasoning
    rs = re.search(r"<reasoning>(.*?)</reasoning>", text, flags=re.DOTALL)
    reasoning = rs.group(1).strip() if rs else ""

    return answer, reasoning, answer_type


# =============================
# PARSE TEACHER RAW
# =============================
def parse_teacher_raw(obj):
    """Input: teacher JSONL object"""
    answer = obj.get("teacher_answer", "")
    reasoning = obj.get("teacher_reasoning", "")
    r_type = obj.get("reasoning_type", "")
    weight = obj.get("reasoning_weight", 1)
    return answer, reasoning, r_type, weight


# =============================
# PATH CONFIG
# =============================
TEST_CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_FOLDER = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"

STUDENT_CKPT = "/kaggle/input/final/transformers/default/1/vqa_student_best_multiKD.pt"
TEACHER_JSONL = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4


# =============================
# LOAD STUDENT MODEL
# =============================
print("[INFO] Loading student model...")
student = VQAGenModel().to(DEVICE)
student.load_state_dict(torch.load(STUDENT_CKPT, map_location=DEVICE))
student.eval()

q_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer")
a_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer")

vision_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")


# =============================
# LOAD DATA
# =============================
test_dataset = VQAGenDataset(TEST_CSV_PATH, IMAGE_FOLDER, vision_processor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

df_test = pd.read_csv(TEST_CSV_PATH)


# =============================
# LOAD TEACHER JSONL
# =============================
print("[INFO] Loading teacher JSONL...")
teacher_list = []
with open(TEACHER_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        teacher_list.append(json.loads(line))

assert len(teacher_list) == len(df_test), "Teacher JSONL length != test.csv length"


# =============================
# EVALUATION LOOP
# =============================
records = []

refs = []
student_predictions = []
teacher_predictions = []

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)

idx = 0

print("[INFO] Running student model inference...")

with torch.no_grad():
    for pixel_values, input_ids, attention_mask, labels in tqdm(test_loader):

        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        pred_ids = student.generate(pixel_values, input_ids, attention_mask)
        student_outs = [
            a_tokenizer.decode(p, skip_special_tokens=True)
            for p in pred_ids
        ]

        # Decode GT
        gt_batch = [
            a_tokenizer.decode([x for x in lab.tolist() if x != -100], skip_special_tokens=True)
            for lab in labels
        ]

        batch_size = len(gt_batch)

        for b in range(batch_size):
            gt = gt_batch[b]
            st_raw = student_outs[b]

            teacher_obj = teacher_list[idx]
            idx += 1

            # Parse student output
            st_answer, st_reason, st_type = parse_student_output(st_raw)

            # Parse teacher output format v2
            tc_answer, tc_reason, tc_rtype, tc_weight = parse_teacher_raw(teacher_obj)

            refs.append(gt)
            student_predictions.append(st_answer)
            teacher_predictions.append(tc_answer)

            records.append({
                "img_id": teacher_obj["img_id"],
                "question": teacher_obj["question"],
                "ground_truth": gt,

                "student_answer": st_answer,
                "student_reasoning": st_reason,
                "student_answer_type": st_type,

                "teacher_answer": tc_answer,
                "teacher_reasoning": tc_reason,
                "teacher_reasoning_type": tc_rtype,
                "reasoning_weight": tc_weight,
            })


# =============================
# METRICS (Student + Teacher)
# =============================
def compute_metrics(preds, refs):
    r1 = []
    rl = []
    f1 = []
    acc = []

    for p, r in zip(preds, refs):
        p_n = normalize_text(p)
        r_n = normalize_text(r)

        sc = scorer.score(r_n, p_n)
        r1.append(sc["rouge1"].fmeasure)
        rl.append(sc["rougeLsum"].fmeasure)
        f1.append(token_f1(p, r))
        acc.append(p_n == r_n)

    return {
        "EM": np.mean(acc),
        "ROUGE1": np.mean(r1),
        "ROUGEL": np.mean(rl),
        "F1": np.mean(f1)
    }


student_metrics = compute_metrics(student_predictions, refs)
teacher_metrics = compute_metrics(teacher_predictions, refs)


print("\n====== STUDENT METRICS ======")
for k, v in student_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n====== TEACHER METRICS ======")
for k, v in teacher_metrics.items():
    print(f"{k}: {v:.4f}")


# =============================
# SAVE CSV
# =============================
out_csv = "/kaggle/working/eval_student_teacher_v2.csv"
pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"\n[INFO] Saved CSV → {out_csv}")
print("DONE.")
