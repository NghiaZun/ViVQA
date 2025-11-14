"""
Full Student Fine-tuning (Multi-Objective KD) with RESUME + VALIDATION
Author: Updated by ChatGPT
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, AutoTokenizer
from model import VQAGenModel

# =====================
# CONFIG
# =====================
DATA_PATH = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs.jsonl"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_multiKD.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_multiKD.pt")
RESUME_PATH = os.path.join(SAVE_DIR, "resume_state.pth")

EPOCHS = 60
LR = 1e-5
BATCH_SIZE = 4
WARMUP_RATIO = 0.06
EARLY_STOP_PATIENCE = 6

# KD Loss weights
W_FORMAT = 0.25
W_REASON = 0.45
W_ANSWER = 0.30

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# DATASET
# =====================
class DistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=64, max_a_len=128, val_split=0.1):
        self.samples = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")]
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

        # Split train/val
        n_val = int(len(self.samples) * val_split)
        self.train_samples = self.samples[n_val:]
        self.val_samples = self.samples[:n_val]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s["question"]
        img_path = s["image_path"]
        teacher_raw = s.get("teacher_raw", "")
        teacher_answer = s.get("teacher_answer", "")
        teacher_reasoning = s.get("teacher_reasoning", "")
        reasoning_weight = float(s.get("reasoning_weight", 1.0))

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]

        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")
        teacher_enc = self.decoder_tokenizer(
            teacher_raw, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )
        answer_enc = self.decoder_tokenizer(
            f"<answer>{teacher_answer}</answer>", truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )
        reason_enc = self.decoder_tokenizer(
            f"<reasoning>{teacher_reasoning}</reasoning>", truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "teacher_ids": teacher_enc.input_ids[0],
            "answer_ids": answer_enc.input_ids[0],
            "reason_ids": reason_enc.input_ids[0],
            "reasoning_weight": torch.tensor(reasoning_weight, dtype=torch.float)
        }

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
).to(device)

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# =====================
# DATA LOADERS
# =====================
dataset = DistillDataset(DATA_PATH, vision_processor, model.text_tokenizer, model.decoder_tokenizer)
train_loader = DataLoader(dataset.train_samples, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset.val_samples, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =====================
# TRAIN SETUP
# =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.decoder_tokenizer.pad_token_id)
scaler = torch.cuda.amp.GradScaler()
best_loss = float("inf")
early_stop_counter = 0
start_epoch = 0

# Scheduler: Reduce LR on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

# ---------------------
# RESUME CHECKPOINT
# ---------------------
if os.path.exists(RESUME_PATH):
    print("[INFO] Resume training from previous checkpoint...")
    ckpt = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_loss = ckpt["best_loss"]
    early_stop_counter = ckpt["early_stop_counter"]
    print(f"[INFO] Resumed at epoch {start_epoch}")

# =====================
# TRAIN LOOP
# =====================
print("[INFO] Start training...")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = total_f = total_r = total_a = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        with torch.cuda.amp.autocast():
            ce_loss_full, _ = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["teacher_ids"]
            )
            ce_loss_reason, _ = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["reason_ids"]
            )
            ce_loss_reason *= batch_t["reasoning_weight"].mean()
            ce_loss_answer, _ = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["answer_ids"]
            )
            loss_total = W_FORMAT * ce_loss_full + W_REASON * ce_loss_reason + W_ANSWER * ce_loss_answer

        scaler.scale(loss_total).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss_total.item()
        total_f += ce_loss_full.item()
        total_r += ce_loss_reason.item()
        total_a += ce_loss_answer.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_f = total_f / len(train_loader)
    avg_r = total_r / len(train_loader)
    avg_a = total_a / len(train_loader)

    # ----------- VALIDATION -----------
    model.eval()
    val_loss = val_f = val_r = val_a = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            ce_loss_full, _ = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["teacher_ids"]
            )
            ce_loss_reason, _ = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["reason_ids"]
            )
            ce_loss_reason *= batch_t["reasoning_weight"].mean()
            ce_loss_answer, _ = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["answer_ids"]
            )
            loss_val = W_FORMAT * ce_loss_full + W_REASON * ce_loss_reason + W_ANSWER * ce_loss_answer
            val_loss += loss_val.item()
            val_f += ce_loss_full.item()
            val_r += ce_loss_reason.item()
            val_a += ce_loss_answer.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_f = val_f / len(val_loader)
    avg_val_r = val_r / len(val_loader)
    avg_val_a = val_a / len(val_loader)

    print(f"[INFO] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
          f"| F={avg_val_f:.4f}, R={avg_val_r:.4f}, A={avg_val_a:.4f}")

    # Scheduler step
    scheduler.step(avg_val_loss)

    # Save resume state
    resume_state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_loss": best_loss,
        "early_stop_counter": early_stop_counter
    }
    torch.save(resume_state, RESUME_PATH)

    # Save best model
    if avg_val_loss < best_loss - 1e-4:
        best_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[INFO] ⭐ New BEST model saved at epoch {epoch+1}")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print("[INFO] EARLY STOPPING TRIGGERED.")
        break

# =====================
# SAVE FINAL
# =====================
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("[INFO] Training finished. Final model saved.")
