"""
Full Student Fine-tuning (Multi-Objective KD) with RESUME + Validation (version 2)
Author: Updated by ChatGPT
Notes:
 - LR lowered to 5e-6
 - KD weights adjusted: F=0.23, R=0.50, A=0.27
 - Use ReduceLROnPlateau (scheduler) on val loss
 - Early stopping patience increased to 10
 - max_a_len increased to 160
 - CSV logging for train/val losses and components
"""

import os
import json
import random
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, AutoTokenizer
from model import VQAGenModel

# =====================
# CONFIG (version 2)
# =====================
DATA_PATH = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_multiKD.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_multiKD.pt")
RESUME_PATH = os.path.join(SAVE_DIR, "resume_state.pth")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_multiKD.csv")

EPOCHS = 60
LR = 5e-6                # reduced LR for fine-tuning
BATCH_SIZE = 4
WARMUP_RATIO = 0.02     # not used with ReduceLROnPlateau but kept for record
EARLY_STOP_PATIENCE = 10
VAL_RATIO = 0.1         # 10% validation
MAX_A_LEN = 160         # allow longer reasoning

# KD Loss weights (updated)
W_FORMAT = 0.23
W_REASON = 0.50
W_ANSWER = 0.27

# reproducibility
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

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s.get("question", "")
        img_path = s.get("image_path", "")
        teacher_raw = s.get("teacher_raw", "")
        teacher_answer = s.get("teacher_answer", "")
        teacher_reasoning = s.get("teacher_reasoning", "")
        reasoning_weight = float(s.get("reasoning_weight", 1.0))

        # load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]

        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")

        teacher_enc = self.decoder_tokenizer(
            teacher_raw, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )
        answer_enc = self.decoder_tokenizer(
            f"<answer>{teacher_answer}</answer>",
            truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )
        reason_enc = self.decoder_tokenizer(
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
            "reason_ids": reason_enc.input_ids[0],
            "reasoning_weight": torch.tensor(reasoning_weight, dtype=torch.float)
        }

# =====================
# LOAD MODEL + PROCESSOR
# =====================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
).to(device)

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# =====================
# DATASET & SPLIT (Subset to keep collate behavior)
# =====================
dataset = DistillDataset(DATA_PATH, vision_processor, model.text_tokenizer, model.decoder_tokenizer)
n_val = max(1, int(len(dataset) * VAL_RATIO))
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
val_indices = all_indices[:n_val]
train_indices = all_indices[n_val:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"[INFO] Dataset sizes -> total: {len(dataset)}, train: {len(train_dataset)}, val: {len(val_dataset)}")

# =====================
# TRAIN SETUP
# =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Use ReduceLROnPlateau driven by validation loss (better for fine-tuning)
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               factor=0.5, patience=2, verbose=True)

# keep scaler for mixed precision
scaler = torch.cuda.amp.GradScaler()

pad_id = getattr(model.decoder_tokenizer, "pad_token_id", None)
if pad_id is None:
    pad_id = -100
ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

best_loss = float("inf")
early_stop_counter = 0
start_epoch = 0

# Prepare CSV log
if not os.path.exists(LOG_CSV):
    df_init = pd.DataFrame(columns=["epoch",
                                    "train_loss", "train_F", "train_R", "train_A",
                                    "val_loss", "val_F", "val_R", "val_A"])
    df_init.to_csv(LOG_CSV, index=False)

# ---------------------
# RESUME CHECKPOINT
# ---------------------
if os.path.exists(RESUME_PATH):
    print("[INFO] Resuming training from previous checkpoint...")
    ckpt = torch.load(RESUME_PATH, map_location=device)
    # safe load: if keys mismatch, try partial
    try:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # scheduler state loading is optional (ReduceLROnPlateau expects dict)
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            try:
                scheduler_plateau.load_state_dict(ckpt["scheduler"])
            except Exception:
                print("[WARN] Could not fully restore scheduler state (continuing).")
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        print(f"[INFO] Resumed at epoch {start_epoch} (best_loss={best_loss:.4f})")
    except Exception as e:
        print(f"[WARN] Resume loading partial states: {e}")
        # fallback load model only
        model.load_state_dict(ckpt.get("model", ckpt))
        start_epoch = ckpt.get("epoch", 0) + 1

# =====================
# TRAIN LOOP (version 2)
# =====================
print("[INFO] Start training (version 2)... LR=%.2e, W=(F=%.2f,R=%.2f,A=%.2f), max_a_len=%d"
      % (LR, W_FORMAT, W_REASON, W_ANSWER, MAX_A_LEN))

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = total_f = total_r = total_a = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for batch in progress:
        # move tensors to device
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        # ensure reasoning weight exists
        if "reasoning_weight" not in batch_t:
            batch_t["reasoning_weight"] = torch.ones(len(batch_t["input_ids"]), device=device)

        with torch.amp.autocast(device_type='cuda') if device.startswith("cuda") else torch.cpu.amp.autocast():
            # full teacher output
            out_full = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["teacher_ids"]
            )
            # model expected to return (loss, logits) or ModelOutput with .loss
            if isinstance(out_full, tuple):
                ce_full = out_full[0]
            else:
                ce_full = out_full.loss if hasattr(out_full, "loss") else out_full[0]

            # reasoning-only
            out_reason = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["reason_ids"]
            )
            if isinstance(out_reason, tuple):
                ce_reason = out_reason[0]
            else:
                ce_reason = out_reason.loss if hasattr(out_reason, "loss") else out_reason[0]

            # scale reason by per-sample weight (mean across batch)
            rw_scalar = batch_t["reasoning_weight"].mean()
            ce_reason = ce_reason * rw_scalar

            # answer-only
            out_answer = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["answer_ids"]
            )
            if isinstance(out_answer, tuple):
                ce_answer = out_answer[0]
            else:
                ce_answer = out_answer.loss if hasattr(out_answer, "loss") else out_answer[0]

            # final weighted loss
            loss_total = W_FORMAT * ce_full + W_REASON * ce_reason + W_ANSWER * ce_answer

        scaler.scale(loss_total).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # accumulate numbers for logging
        total_loss += float(loss_total.item())
        total_f += float(ce_full.item())
        total_r += float(ce_reason.item())
        total_a += float(ce_answer.item())

        progress.set_postfix({
            "loss": f"{loss_total.item():.4f}",
            "F": f"{ce_full.item():.4f}",
            "R": f"{ce_reason.item():.4f}",
            "A": f"{ce_answer.item():.4f}"
        })

    # compute train averages
    avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float("inf")
    avg_train_f = total_f / len(train_loader) if len(train_loader) > 0 else float("inf")
    avg_train_r = total_r / len(train_loader) if len(train_loader) > 0 else float("inf")
    avg_train_a = total_a / len(train_loader) if len(train_loader) > 0 else float("inf")

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_total = val_f = val_r = val_a = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            # forward same way
            out_full = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["teacher_ids"]
            )
            ce_full = out_full[0] if isinstance(out_full, tuple) else (out_full.loss if hasattr(out_full, "loss") else out_full[0])

            out_reason = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["reason_ids"]
            )
            ce_reason = out_reason[0] if isinstance(out_reason, tuple) else (out_reason.loss if hasattr(out_reason, "loss") else out_reason[0])
            ce_reason = ce_reason * batch_t.get("reasoning_weight", torch.ones(1, device=device)).mean()

            out_answer = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                labels=batch_t["answer_ids"]
            )
            ce_answer = out_answer[0] if isinstance(out_answer, tuple) else (out_answer.loss if hasattr(out_answer, "loss") else out_answer[0])

            val_loss = W_FORMAT * ce_full + W_REASON * ce_reason + W_ANSWER * ce_answer

            val_total += float(val_loss.item())
            val_f += float(ce_full.item())
            val_r += float(ce_reason.item())
            val_a += float(ce_answer.item())

    avg_val_loss = val_total / len(val_loader) if len(val_loader) > 0 else float("inf")
    avg_val_f = val_f / len(val_loader) if len(val_loader) > 0 else float("inf")
    avg_val_r = val_r / len(val_loader) if len(val_loader) > 0 else float("inf")
    avg_val_a = val_a / len(val_loader) if len(val_loader) > 0 else float("inf")

    # print summary
    print(f"[INFO] Epoch {epoch+1} | Train: {avg_train_loss:.4f} | F={avg_train_f:.4f}, R={avg_train_r:.4f}, A={avg_train_a:.4f} "
          f"| Val: {avg_val_loss:.4f} | F={avg_val_f:.4f}, R={avg_val_r:.4f}, A={avg_val_a:.4f}")

    # log to CSV
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [epoch+1,
                       avg_train_loss, avg_train_f, avg_train_r, avg_train_a,
                       avg_val_loss, avg_val_f, avg_val_r, avg_val_a]
    df.to_csv(LOG_CSV, index=False)

    # -------------------------
    # Save RESUME checkpoint
    # -------------------------
    resume_state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_plateau.state_dict() if scheduler_plateau is not None else None,
        "scaler": scaler.state_dict(),
        "best_loss": best_loss,
        "early_stop_counter": early_stop_counter
    }
    torch.save(resume_state, RESUME_PATH)

    # --------------------------
    # Scheduler step (ReduceLROnPlateau)
    # --------------------------
    scheduler_plateau.step(avg_val_loss)

    # --------------------------
    # Save BEST MODEL (by val loss)
    # --------------------------
    if avg_val_loss < best_loss - 1e-4:
        best_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[INFO] ⭐ New BEST model saved at epoch {epoch+1} (val_loss={best_loss:.4f})")
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
print("[INFO] Training finished. Final model saved at:", FINAL_MODEL_PATH)
