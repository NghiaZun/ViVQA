"""
Student Fine-Tuning (Multi-Objective KD) - CLEAN VERSION
Parameters are KEPT EXACTLY the SAME
"""

import os
import json
import random
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BlipProcessor
from model import VQAGenModel
from contextlib import nullcontext

# =====================
# CONFIG (unchanged)
# =====================
DATA_PATH = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_multiKD.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_multiKD.pt")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_multiKD.csv")

EPOCHS = 60
LR = 5e-6
BATCH_SIZE = 2
VAL_RATIO = 0.1
MAX_A_LEN = 128
EARLY_STOP_PATIENCE = 10
accum_steps = 4

# KD weights
W_FORMAT = 0.23
W_REASON = 0.50
W_ANSWER = 0.27

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# DATASET
# =====================
class DistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=64, max_a_len=MAX_A_LEN):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.samples)

    def _safe_load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (255, 255, 255))

    def encode_text(self, txt):
        return self.decoder_tokenizer(
            txt, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s.get("question", "")
        img = self._safe_load_image(s.get("image_path", ""))

        pixel_values = self.vision_processor(img, return_tensors="pt").pixel_values[0]
        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")

        teacher_ids = self.encode_text(s.get("teacher_raw", "")).input_ids[0]
        answer_ids = self.encode_text(f"<answer>{s.get('teacher_answer', '')}</answer>").input_ids[0]
        reason_ids = self.encode_text(f"<reasoning>{s.get('teacher_reasoning', '')}</reasoning>").input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "teacher_ids": teacher_ids,
            "answer_ids": answer_ids,
            "reason_ids": reason_ids,
            "reasoning_weight": torch.tensor(float(s.get("reasoning_weight", 1.0)))
        }

# =====================
# LOAD MODEL + PROCESSORS
# =====================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
).to(device)

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# =====================
# DATASET & SPLIT
# =====================
dataset = DistillDataset(DATA_PATH, vision_processor, model.text_tokenizer, model.decoder_tokenizer)
n_val = max(1, int(len(dataset) * VAL_RATIO))
indices = list(range(len(dataset)))
random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2, pin_memory=True)

print(f"[INFO] Dataset -> Total {len(dataset)}, Train {len(train_indices)}, Val {len(val_indices)}")

# =====================
# OPTIMIZER + SCHEDULER
# =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=True
)
scaler = torch.cuda.amp.GradScaler()

# =====================
# CSV INIT
# =====================
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["epoch","train_loss","train_F","train_R","train_A",
                          "val_loss","val_F","val_R","val_A"]).to_csv(LOG_CSV, index=False)

# =====================
# LOAD BEST CHECKPOINT
# =====================
print("[INFO] Loading BEST model...")
best_ckpt = torch.load("/kaggle/input/v2/transformers/default/1/vqa_student_best_multiKD.pt", map_location=device)
model.load_state_dict(best_ckpt)
print("[INFO] BEST model loaded.")

# =====================
# LOSS HELPERS
# =====================
def compute_multi_loss(model, batch, weights):
    out_f = model(pixel_values=batch["pixel_values"],
                  input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  labels=batch["teacher_ids"])
    out_r = model(pixel_values=batch["pixel_values"],
                  input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  labels=batch["reason_ids"])
    out_a = model(pixel_values=batch["pixel_values"],
                  input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  labels=batch["answer_ids"])

    ce_f = out_f[0] if isinstance(out_f, tuple) else out_f.loss
    ce_r = (out_r[0] if isinstance(out_r, tuple) else out_r.loss) * batch["reasoning_weight"].mean()
    ce_a = out_a[0] if isinstance(out_a, tuple) else out_a.loss

    total = W_FORMAT * ce_f + W_REASON * ce_r + W_ANSWER * ce_a
    return total, ce_f, ce_r, ce_a

# =====================
# TRAINING LOOP
# =====================
best_loss = float("inf")
early_stop = 0
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    sum_loss = sum_f = sum_r = sum_a = 0
    loop = tqdm(train_loader, leave=False)

    for step, batch in enumerate(loop):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx():
            loss, f, r, a = compute_multi_loss(model, batch, (W_FORMAT,W_REASON,W_ANSWER))
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        sum_loss += loss.item() * accum_steps
        sum_f += f.item()
        sum_r += r.item()
        sum_a += a.item()

        loop.set_postfix(loss=sum_loss/(step+1))

    train_loss = sum_loss / len(train_loader)
    train_f = sum_f / len(train_loader)
    train_r = sum_r / len(train_loader)
    train_a = sum_a / len(train_loader)

    # ===================== VALID =====================
    model.eval()
    val_loss = val_f = val_r = val_a = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total, f, r, a = compute_multi_loss(model, batch, (W_FORMAT,W_REASON,W_ANSWER))
            val_loss += total.item()
            val_f += f.item()
            val_r += r.item()
            val_a += a.item()

    val_loss /= len(val_loader)
    val_f /= len(val_loader)
    val_r /= len(val_loader)
    val_a /= len(val_loader)

    print(f"[E{epoch+1}] Train={train_loss:.4f} | Val={val_loss:.4f}")

    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [epoch+1, train_loss, train_f, train_r, train_a,
                       val_loss, val_f, val_r, val_a]
    df.to_csv(LOG_CSV, index=False)

    scheduler_plateau.step(val_loss)

    # BEST SAVE
    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        early_stop = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[INFO] ⭐ BEST UPDATED at epoch {epoch+1}")
    else:
        early_stop += 1
        print(f"[INFO] No improvement ({early_stop}/{EARLY_STOP_PATIENCE})")

    if early_stop >= EARLY_STOP_PATIENCE:
        print("[INFO] EARLY STOPPING TRIGGERED.")
        break

# =====================
# SAVE FINAL
# =====================
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("[INFO] Final model saved:", FINAL_MODEL_PATH)
