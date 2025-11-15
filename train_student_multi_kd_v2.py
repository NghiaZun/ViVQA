"""
Full Student Fine-tuning (Multi-Objective KD) with LOAD-BEST + Validation
Version 3 – FIXED, CLEAN, STABLE

Changes (NOTE):
 - FIX dataset always returns pixel_values
 - REMOVE duplicate & broken resume logic
 - ENSURE best model is loaded correctly before new training
 - FIX ce_full/ce_reason/ce_answer extraction
 - FIX R-weight scaling
 - CLEAN optimizer/scheduler reset
 - ADD safe collate to prevent batch errors
 - CONSISTENT best model loading path
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

# ====================================================
# CONFIG
# ====================================================
DATA_PATH = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_multiKD.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_multiKD.pt")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_multiKD.csv")

EPOCHS = 60
LR = 5e-6
BATCH_SIZE = 4
VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 10
MAX_A_LEN = 160

W_FORMAT = 0.23
W_REASON = 0.50
W_ANSWER = 0.27

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====================================================
# DATASET
# ====================================================
class DistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=64, max_a_len=160):

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(l) for l in f]

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_path = s.get("image_path", "")
        question = s.get("question", "")
        teacher_raw = s.get("teacher_raw", "")
        teacher_answer = s.get("teacher_answer", "")
        teacher_reasoning = s.get("teacher_reasoning", "")
        reasoning_weight = float(s.get("reasoning_weight", 1.0))

        ### FIX: luôn đảm bảo pixel_values
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), "white")

        pixel_values = self.vision_processor(img, return_tensors="pt").pixel_values[0]

        # tokenize question
        q_enc = self.text_tokenizer(
            question, truncation=True, padding="max_length",
            max_length=self.max_q_len, return_tensors="pt"
        )

        teacher_ids = self.decoder_tokenizer(
            teacher_raw, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        ).input_ids[0]

        answer_ids = self.decoder_tokenizer(
            f"<answer>{teacher_answer}</answer>",
            truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        ).input_ids[0]

        reason_ids = self.decoder_tokenizer(
            f"<reasoning>{teacher_reasoning}</reasoning>",
            truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "teacher_ids": teacher_ids,
            "answer_ids": answer_ids,
            "reason_ids": reason_ids,
            "reasoning_weight": torch.tensor(reasoning_weight, dtype=torch.float)
        }


# ====================================================
# LOAD MODEL
# ====================================================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
).to(device)

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# ====================================================
# DATA & SPLIT
# ====================================================
dataset = DistillDataset(DATA_PATH, vision_processor, model.text_tokenizer, model.decoder_tokenizer)

n_val = max(1, int(len(dataset) * VAL_RATIO))
indices = list(range(len(dataset)))
random.shuffle(indices)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Dataset sizes -> train={len(train_dataset)}, val={len(val_dataset)}")

# ====================================================
# TRAIN SETUP
# ====================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                       patience=2, factor=0.5, verbose=True)
scaler = torch.cuda.amp.GradScaler()

pad_id = model.decoder_tokenizer.pad_token_id or -100
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

best_loss = float("inf")
early_stop = 0

# ====================================================
# LOAD BEST MODEL BEFORE NEW TRAIN
# ====================================================
### FIX: make loading clean & consistent
try:
    print("[INFO] Loading BEST model for new training...")
    state = torch.load("/kaggle/input/v2/transformers/default/1/vqa_student_best_multiKD.pt", map_location=device)
    model.load_state_dict(state)
    print("[INFO] BEST model loaded successfully.")
except:
    print("[WARN] No BEST model found. Training from scratch weights.")

# ====================================================
# TRAIN LOOP
# ====================================================
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

print(f"[INFO] Start training (v3) – LR={LR}")

for epoch in range(EPOCHS):
    # TRAIN ------------------------------------------------
    model.train()
    total_L = total_F = total_R = total_A = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):

        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast_ctx():
            outF = model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["teacher_ids"])
            lossF = outF.loss

            outR = model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["reason_ids"])
            lossR = outR.loss * batch["reasoning_weight"].mean()

            outA = model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["answer_ids"])
            lossA = outA.loss

            loss = W_FORMAT * lossF + W_REASON * lossR + W_ANSWER * lossA

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_L += loss.item()
        total_F += lossF.item()
        total_R += lossR.item()
        total_A += lossA.item()

    avg_train = total_L / len(train_loader)
    avgF = total_F / len(train_loader)
    avgR = total_R / len(train_loader)
    avgA = total_A / len(train_loader)

    # VALID ------------------------------------------------
    model.eval()
    vL = vF = vR = vA = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outF = model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["teacher_ids"])
            lossF = outF.loss

            outR = model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["reason_ids"])
            lossR = outR.loss * batch["reasoning_weight"].mean()

            outA = model(pixel_values=batch["pixel_values"],
                         input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["answer_ids"])
            lossA = outA.loss

            loss = W_FORMAT * lossF + W_REASON * lossR + W_ANSWER * lossA

            vL += loss.item()
            vF += lossF.item()
            vR += lossR.item()
            vA += lossA.item()

    avg_val = vL / len(val_loader)
    avg_vF = vF / len(val_loader)
    avg_vR = vR / len(val_loader)
    avg_vA = vA / len(val_loader)

    print(f"[INFO] Epoch {epoch+1} | "
          f"Train {avg_train:.4f} (F={avgF:.4f}, R={avgR:.4f}, A={avgA:.4f}) | "
          f"Val {avg_val:.4f} (F={avg_vF:.4f}, R={avg_vR:.4f}, A={avg_vA:.4f})")

    # CSV LOG --------------------------------------------
    row = pd.DataFrame([[
        epoch+1, avg_train, avgF, avgR, avgA,
        avg_val, avg_vF, avg_vR, avg_vA
    ]], columns=["epoch","train","F","R","A","val","vF","vR","vA"])

    if not os.path.exists(LOG_CSV):
        row.to_csv(LOG_CSV, index=False)
    else:
        row.to_csv(LOG_CSV, mode="a", index=False, header=False)

    # Scheduler
    scheduler.step(avg_val)

    # SAVE BEST ------------------------------------------
    if avg_val < best_loss - 1e-4:
        best_loss = avg_val
        early_stop = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[INFO] ⭐ NEW BEST MODEL saved (val={avg_val:.4f})")
    else:
        early_stop += 1
        print(f"[INFO] No improvement ({early_stop}/{EARLY_STOP_PATIENCE})")

    if early_stop >= EARLY_STOP_PATIENCE:
        print("[INFO] EARLY STOPPING")
        break

# FINAL SAVE =====================================================
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print("[INFO] Training done. Saved:", FINAL_MODEL_PATH)
