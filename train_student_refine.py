"""
Day 5 – Student Refinement (Freeze Encoders, Fine-tune Decoder + Fusion)
Author: Hân
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, get_cosine_schedule_with_warmup, AutoTokenizer

from model import VQAGenModel

# =========================
# CONFIG
# =========================
DATA_PATH = "/kaggle/input/teacher/Qwen2-VL-7B-Instruct.jsonl"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
MODEL_CKPT = "/kaggle/input/checkpoint/pytorch/default/1/vqa_student_best.pt"
SAVE_DIR = "/kaggle/working"
SAVE_PATH = os.path.join(SAVE_DIR, "vqa_student_refined.pt")

EPOCHS = 10
LR = 1e-6
ALPHA = 0.8
BATCH_SIZE = 4
WARMUP_RATIO = 0.06
EARLY_STOP_PATIENCE = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[INFO] Using device: {device}")

# =========================
# DATASET
# =========================
class DistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=48, max_a_len=32):
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
        answer = s.get("teacher_answer", "")
        reasoning = s.get("teacher_reasoning", "")

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]
        q_enc = self.text_tokenizer(
            q, truncation=True, padding="max_length",
            max_length=self.max_q_len, return_tensors="pt"
        )
        label_text = f"{answer}. Giải thích: {reasoning}"
        labels = self.decoder_tokenizer(
            label_text, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "labels": labels
        }

# =========================
# LOAD MODEL
# =========================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/vqagen-checkpoint/phobert",
    vit5_dir="/kaggle/input/vqagen-checkpoint/vit5"
)
model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
model.to(device)

# Freeze vision & PhoBERT encoders
for p in model.vision_encoder.parameters():
    p.requires_grad = False
for p in model.text_encoder.parameters():
    p.requires_grad = False

# Optimizer (only train fusion + decoder)
train_params = list(model.fusion.parameters()) + list(model.decoder.parameters())
optimizer = torch.optim.AdamW(train_params, lr=LR, weight_decay=0.01)

# Scheduler
vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
dataset = DistillDataset(
    jsonl_path=DATA_PATH,
    vision_processor=vision_processor,
    text_tokenizer=model.text_tokenizer,
    decoder_tokenizer=model.decoder_tokenizer
)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
)

num_training_steps = len(loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
    num_training_steps=num_training_steps
)

scaler = torch.cuda.amp.GradScaler(enabled=True)
best_loss = float("inf")
early_stop_counter = 0
log_path = os.path.join(SAVE_DIR, "refine_log.csv")

if not os.path.exists(log_path):
    pd.DataFrame(columns=["epoch", "train_loss"]).to_csv(log_path, index=False)

# =========================
# TRAIN LOOP
# =========================
print(f"[INFO] Start refining for {EPOCHS} epochs (fusion + decoder only)...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
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

    # Log
    df = pd.read_csv(log_path)
    df.loc[len(df)] = [epoch + 1, avg_loss]
    df.to_csv(log_path, index=False)

    # Save best
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vqa_student_refined_best.pt"))
        print(f"[INFO] ✅ New best model @ epoch {epoch+1} (loss={best_loss:.4f})")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print("[INFO] Early stopping triggered.")
        break

torch.save(model.state_dict(), SAVE_PATH)
print(f"[INFO] ✅ Final refined model saved → {SAVE_PATH}")


