"""
Day 4 – Full Student Fine-tuning (Multi-Objective KD)
Author: Hân (updated by ChatGPT)
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
DATA_PATH = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
SAVE_DIR = "/kaggle/working"
SAVE_PATH = os.path.join(SAVE_DIR, "vqa_student_final_multiKD.pt")

EPOCHS = 60
LR = 1e-5
BATCH_SIZE = 4
WARMUP_RATIO = 0.06
EARLY_STOP_PATIENCE = 6

# KD Loss weights (Format + Reason + Answer)
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
                 max_q_len=64, max_a_len=128):
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

        # =====================
        # INPUT & TARGET
        # =====================
        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")

        # Full teacher output (XML structured)
        teacher_enc = self.decoder_tokenizer(
            teacher_raw,
            truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

        # Only answer part (ground truth supervision)
        answer_enc = self.decoder_tokenizer(
            f"<answer>{teacher_answer}</answer>",
            truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

        # Reasoning-only text (for reasoning KD)
        reasoning_enc = self.decoder_tokenizer(
            f"<reasoning>{teacher_reasoning}</reasoning>",
            truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "teacher_ids": teacher_enc.input_ids[0],
            "answer_ids": answer_enc.input_ids[0],
            "reason_ids": reasoning_enc.input_ids[0],
            "reasoning_weight": torch.tensor(reasoning_weight, dtype=torch.float)
        }

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)

KD_CHECKPOINT = "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/best_model.pth"
if os.path.exists(KD_CHECKPOINT):
    print(f"[INFO] Loading KD checkpoint from {KD_CHECKPOINT}")
    model.load_state_dict(torch.load(KD_CHECKPOINT, map_location=device))

phobert_tok_path = "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer"
vit5_tok_path = "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
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
log_path = os.path.join(SAVE_DIR, "train_log_multiKD.csv")

if not os.path.exists(log_path):
    pd.DataFrame(columns=["epoch", "total_loss", "format", "reason", "answer"]).to_csv(log_path, index=False)

ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# =====================
# TRAIN LOOP
# =====================
print(f"[INFO] Start training for {EPOCHS} epochs on {device}...")
model.train()
for epoch in range(EPOCHS):
    total_loss, total_f, total_r, total_a = 0, 0, 0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.cuda.amp.autocast():
            # === Student forward (generate logits)
            logits = model.forward_generate(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # logits shape: (B, L, V)
            # Align shapes for CE
            logits_flat = logits.view(-1, logits.size(-1))

            # === Compute 3 types of loss ===
            loss_format = ce_loss_fn(logits_flat, batch["teacher_ids"].view(-1))
            loss_reason = ce_loss_fn(logits_flat, batch["reason_ids"].view(-1))
            loss_answer = ce_loss_fn(logits_flat, batch["answer_ids"].view(-1))

            # Apply reasoning weight (difficulty scaling)
            loss_reason = loss_reason * batch["reasoning_weight"].mean()

            # Weighted sum (7:3 ratio)
            loss_total = W_FORMAT * loss_format + W_REASON * loss_reason + W_ANSWER * loss_answer

        scaler.scale(loss_total).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss_total.item()
        total_f += loss_format.item()
        total_r += loss_reason.item()
        total_a += loss_answer.item()
        progress.set_postfix({
            "total": f"{loss_total.item():.4f}",
            "format": f"{loss_format.item():.4f}",
            "reason": f"{loss_reason.item():.4f}",
            "answer": f"{loss_answer.item():.4f}",
        })

    avg_loss = total_loss / len(loader)
    avg_f = total_f / len(loader)
    avg_r = total_r / len(loader)
    avg_a = total_a / len(loader)

    print(f"[INFO] Epoch {epoch+1} | Total: {avg_loss:.4f} | F={avg_f:.4f}, R={avg_r:.4f}, A={avg_a:.4f}")

    df = pd.read_csv(log_path)
    df.loc[len(df)] = [epoch + 1, avg_loss, avg_f, avg_r, avg_a]
    df.to_csv(log_path, index=False)

    # Early stopping logic
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vqa_student_best_multiKD.pt"))
        print(f"[INFO] ✅ New best model saved at epoch {epoch+1} (loss={best_loss:.4f})")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

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
