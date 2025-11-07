import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BlipImageProcessor

from rouge_score import rouge_scorer
import re
import unicodedata

from dataset import VQAGenDataset
from model import VQAGenModel


# ======================
# TEXT NORMALIZATION
# ======================
def normalize_text(s):
    s = s.lower().strip()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


# ======================
# TOKEN LEVEL F1
# ======================
def token_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# === CONFIG ===
TEST_CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_FOLDER = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
SAVE_DIR = "/kaggle/input/checkpoints_2/transformers/default/1/checkpoints"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
print("[INFO] Loading model...")
model = VQAGenModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE))
model.eval()

# === TOKENIZERS ===
q_tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_DIR, "phobert_tokenizer"))
a_tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_DIR, "vit5_tokenizer"))

# === DATASET ===
vision_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
test_dataset = VQAGenDataset(TEST_CSV_PATH, IMAGE_FOLDER, vision_processor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === EVAL LOOP ===
print("[INFO] Running evaluation...")
refs, hyps = [], []
records = []

with torch.no_grad():
    for pixel_values, input_ids, attention_mask, labels in tqdm(test_loader):
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        # Generate
        output_ids = model.generate(pixel_values, input_ids, attention_mask)
        preds = [a_tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        # Ground truth
        truths = [
            a_tokenizer.decode([i for i in lab.tolist() if i != -100], skip_special_tokens=True)
            for lab in labels
        ]

        refs.extend(truths)
        hyps.extend(preds)

        for gt, pr in zip(truths, preds):
            records.append({"ground_truth": gt, "prediction": pr})


# === METRICS: ONLY THE ESSENTIAL ONES ===
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)

rouge1_list, rougel_list, f1_list = [], [], []

for ref, hyp in zip(refs, hyps):
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)

    scores = scorer.score(ref_n, hyp_n)
    rouge1_list.append(scores["rouge1"].fmeasure)
    rougel_list.append(scores["rougeLsum"].fmeasure)
    f1_list.append(token_f1(hyp, ref))

avg_rouge1 = np.mean(rouge1_list)
avg_rougel = np.mean(rougel_list)
avg_f1 = np.mean(f1_list)

# Exact-match
acc = np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, hyps)])


# === PRINT RESULTS ===
print("========== Test Results ==========")
print(f"Accuracy (EM): {acc*100:.2f}%")
print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
print(f"ROUGE-Lsum F1: {avg_rougel:.4f}")
print(f"Token F1: {avg_f1:.4f}")


# === SAVE CSV ===
out_csv = os.path.join("/kaggle/working", "test_predictions.csv")
pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"[INFO] Saved predictions to {out_csv}")

# === PRINT SAMPLE ===
print("\n===== Sample Predictions =====")
for i in range(min(10, len(records))):
    print(f"Q{i+1} | GT: {records[i]['ground_truth']} || Pred: {records[i]['prediction']}")
