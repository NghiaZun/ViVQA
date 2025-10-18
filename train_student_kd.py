"""
Day 4 – Full Student Fine-tuning (Knowledge Distillation)
Author: Hân
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import (
    BlipProcessor,
    get_cosine_schedule_with_warmup,
    AutoTokenizer
)
from model import VQAGenModel

# =====================
# CONFIG
# =====================
DATA_PATH = "/kaggle/input/teacher/Qwen2-VL-7B-Instruct.jsonl"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
SAVE_DIR = "/kaggle/working"
SAVE_PATH = os.path.join(SAVE_DIR, "vqa_student_final.pt")

EPOCHS = 60
LR = 1e-5
ALPHA = 0.8
BATCH_SIZE = 4
WARMUP_RATIO = 0.06
EARLY_STOP_PATIENCE = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# DATASET
# =====================
class DistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=64, max_a_len=64):
        self.samples = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")]
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
    
        # Cập nhật theo định dạng Day1 output
        q = s["question"]
        img_path = s["image_path"]
        answer = s.get("teacher_answer", "")
        reasoning = s.get("teacher_reasoning", "")
    
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
    
        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]
        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")
        label_text = f"{answer}. Giải thích: {reasoning}"
        labels = self.decoder_tokenizer(label_text, truncation=True, padding="max_length",
                                        max_length=self.max_a_len, return_tensors="pt").input_ids[0]
    
        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "labels": labels
        }

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading VQAGenModel...")
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/vit5_tokenizer"
)

# Nếu bạn có checkpoint từ KD giai đoạn 3
KD_CHECKPOINT = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/best_model.pth"
if os.path.exists(KD_CHECKPOINT):
    print(f"[INFO] Loading KD checkpoint from {KD_CHECKPOINT}")
    model.load_state_dict(torch.load(KD_CHECKPOINT, map_location=device))

# Nếu có tokenizer tùy chỉnh (local)
phobert_tok_path = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/phobert_tokenizer"
vit5_tok_path = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/vit5_tokenizer"
if os.path.exists(phobert_tok_path):
    model.text_tokenizer = AutoTokenizer.from_pretrained(phobert_tok_path)
if os.path.exists(vit5_tok_path):
    model.decoder_tokenizer = AutoTokenizer.from_pretrained(vit5_tok_path)

model.to(device)
vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# =====================
# DATA LOADER
# =====================
dataset = DistillDataset(
    jsonl_path=DATA_PATH,
    vision_processor=vision_processor,
    text_tokenizer=model.text_tokenizer,
    decoder_tokenizer=model.decoder_tokenizer
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# =====================
# TRAIN SETUP
# =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
num_training_steps = len(loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
    num_training_steps=num_training_steps
)

scaler = torch.cuda.amp.GradScaler()
best_loss = float("inf")
early_stop_counter = 0
log_path = os.path.join(SAVE_DIR, "train_log.csv")

if not os.path.exists(log_path):
    pd.DataFrame(columns=["epoch", "train_loss"]).to_csv(log_path, index=False)

# =====================
# TRAIN LOOP
# =====================
print(f"[INFO] Start training for {EPOCHS} epochs on {device}...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            ce_loss, logits = model(**batch)
            loss = ALPHA * ce_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    print(f"[INFO] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # Save log
    df = pd.read_csv(log_path)
    df.loc[len(df)] = [epoch + 1, avg_loss]
    df.to_csv(log_path, index=False)

    # Early stopping logic
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vqa_student_best.pt"))
        print(f"[INFO] ✅ New best model saved at epoch {epoch+1} (loss={best_loss:.4f})")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

    # Checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"vqa_student_epoch{epoch+1}.pt"))

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print("[INFO] Early stopping triggered.")
        break

# =====================
# FINAL SAVE
# =====================
torch.save(model.state_dict(), SAVE_PATH)
print(f"[INFO] ✅ Final fine-tuned model saved → {SAVE_PATH}")



