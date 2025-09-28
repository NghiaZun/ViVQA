import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

from transformers import AutoTokenizer, BlipImageProcessor
from dataset import VQAGenDataset
from model import VQAGenModel

# === CONFIG ===
TEST_CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_FOLDER = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"  # đổi nếu folder khác
SAVE_DIR = "/kaggle/input/checkpoints_2/transformers/default/1/checkpoints"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
print("[INFO] Loading model...")
model = VQAGenModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE))
model.eval()

# === LOAD TOKENIZERS ===
q_tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_DIR, "phobert_tokenizer"))
a_tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_DIR, "vit5_tokenizer"))

# === DATASET ===
vision_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
test_dataset = VQAGenDataset(TEST_CSV_PATH, IMAGE_FOLDER, vision_processor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === EVALUATION LOOP ===
print("[INFO] Running evaluation...")
refs, hyps = [], []
losses = []
records = []

with torch.no_grad():
    for pixel_values, input_ids, attention_mask, labels in tqdm(test_loader):
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        # Loss để tham khảo
        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
        losses.append(loss.item())

        # Sinh câu trả lời
        output_ids = model.generate(pixel_values, input_ids, attention_mask)
        preds = [a_tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        # Ground truth
        truths = [a_tokenizer.decode(lab, skip_special_tokens=True) for lab in labels]

        refs.extend(truths)
        hyps.extend(preds)

        for gt, pr in zip(truths, preds):
            records.append({"ground_truth": gt, "prediction": pr})

# === METRICS ===
avg_loss = np.mean(losses)
perplexity = np.exp(avg_loss)

bleu_scores = [sentence_bleu([ref.split()], hyp.split()) for ref, hyp in zip(refs, hyps)]
avg_bleu = np.mean(bleu_scores)

acc = np.mean([ref.strip().lower() == hyp.strip().lower() for ref, hyp in zip(refs, hyps)])

print("========== Test Results ==========")
print(f"Test Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"BLEU: {avg_bleu:.4f}")
print(f"Accuracy: {acc*100:.2f}%")

# === SAVE CSV ===
out_csv = os.path.join(SAVE_DIR, "test_predictions.csv")
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"[INFO] Saved predictions to {out_csv}")
