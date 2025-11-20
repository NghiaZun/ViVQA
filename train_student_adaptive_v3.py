"""
Student Fine-Tuning with ADAPTIVE Multi-Objective KD
Dynamic loss weighting + Format enforcement + Answer-priority training
Author: Enhanced for thesis
"""

import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BlipProcessor
from model import VQAGenModel
from contextlib import nullcontext

# =====================
# CONFIG
# =====================
DATA_PATH = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_adaptive_v3.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_adaptive_v3.pt")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_adaptive_v3.csv")

# Optimized for Kaggle T4 GPU (16GB)
EPOCHS = 80
LR = 1e-5
BATCH_SIZE = 3  # Reduced for memory safety
VAL_RATIO = 0.1
MAX_A_LEN = 160
EARLY_STOP_PATIENCE = 12
accum_steps = 3  # Effective batch = 9

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True
EMPTY_CACHE_EVERY_N_STEPS = 50
PIN_MEMORY = False

# Initial KD weights (will be adjusted dynamically)
INITIAL_WEIGHTS = {
    "format": 0.50,    # High initially to learn structure
    "answer": 0.35,    # Strong emphasis on answer
    "reason": 0.15     # Lower initially
}

FINAL_WEIGHTS = {
    "format": 0.20,    # Lower later
    "answer": 0.30,    # Maintain
    "reason": 0.50     # Increase for reasoning quality
}

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

import gc

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")

# =====================
# ADAPTIVE WEIGHT SCHEDULER
# =====================
def get_adaptive_weights(epoch, total_epochs):
    """
    Smoothly transition from initial to final weights
    Early epochs: emphasize format + answer
    Later epochs: emphasize reasoning quality
    """
    progress = epoch / total_epochs
    
    # Cosine annealing for smooth transition
    alpha = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    w_format = INITIAL_WEIGHTS["format"] * alpha + FINAL_WEIGHTS["format"] * (1 - alpha)
    w_answer = INITIAL_WEIGHTS["answer"] * alpha + FINAL_WEIGHTS["answer"] * (1 - alpha)
    w_reason = INITIAL_WEIGHTS["reason"] * alpha + FINAL_WEIGHTS["reason"] * (1 - alpha)
    
    # Extra answer boost in first 20% of training
    if progress < 0.2:
        w_answer *= 1.5
        total = w_format + w_answer + w_reason
        w_format /= total
        w_answer /= total
        w_reason /= total
    
    return w_format.item(), w_answer.item(), w_reason.item()

# =====================
# DATASET
# =====================
class AdaptiveDistillDataset(Dataset):
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

        # Get teacher outputs
        teacher_answer = s.get("teacher_answer", "")
        teacher_reasoning = s.get("teacher_reasoning", "")
        reasoning_type = s.get("reasoning_type", "NEUTRAL")
        reasoning_weight = float(s.get("reasoning_weight", 1.0))
        
        # Full format (exactly as teacher)
        teacher_raw = s.get("teacher_raw", "")
        if not teacher_raw:
            teacher_raw = f"<answer>{teacher_answer}</answer>\n<reasoning>[{reasoning_type}] {teacher_reasoning}</reasoning>"
        
        # Encode different targets
        format_ids = self.encode_text(teacher_raw).input_ids[0]
        
        # Answer only (with tags)
        answer_only = f"<answer>{teacher_answer}</answer>"
        answer_ids = self.encode_text(answer_only).input_ids[0]
        
        # Full reasoning with type
        full_reasoning = f"<reasoning>[{reasoning_type}] {teacher_reasoning}</reasoning>"
        reason_ids = self.encode_text(full_reasoning).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "format_ids": format_ids,
            "answer_ids": answer_ids,
            "reason_ids": reason_ids,
            "reasoning_weight": torch.tensor(reasoning_weight),
            "teacher_answer": teacher_answer,  # For debugging
            "question": q
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
# LOAD CHECKPOINT (if exists)
# =====================
CHECKPOINT_PATH = "/kaggle/input/v2/transformers/default/1/vqa_student_best_multiKD.pt"
if os.path.exists(CHECKPOINT_PATH):
    print(f"[INFO] Loading checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    print("[INFO] Checkpoint loaded successfully.")
else:
    print("[INFO] No checkpoint found. Training from scratch.")

# =====================
# DATASET & SPLIT
# =====================
dataset = AdaptiveDistillDataset(DATA_PATH, vision_processor, model.text_tokenizer, model.decoder_tokenizer)
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
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-7
)
scaler = torch.cuda.amp.GradScaler()

# =====================
# CSV INIT
# =====================
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["epoch","train_loss","train_F","train_R","train_A",
                          "val_loss","val_F","val_R","val_A",
                          "w_format","w_reason","w_answer","lr"]).to_csv(LOG_CSV, index=False)

# =====================
# LOSS COMPUTATION
# =====================
def compute_adaptive_loss(model, batch, weights):
    """
    Multi-objective loss with adaptive weighting
    """
    w_format, w_answer, w_reason = weights
    
    # Format loss (complete output)
    out_format = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["format_ids"]
    )
    
    # Answer loss (answer only)
    out_answer = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["answer_ids"]
    )
    
    # Reasoning loss (reasoning only)
    out_reason = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["reason_ids"]
    )
    
    # Extract losses
    loss_format = out_format[0] if isinstance(out_format, tuple) else out_format.loss
    loss_answer = out_answer[0] if isinstance(out_answer, tuple) else out_answer.loss
    loss_reason = (out_reason[0] if isinstance(out_reason, tuple) else out_reason.loss) * batch["reasoning_weight"].mean()
    
    # Combined weighted loss
    total_loss = w_format * loss_format + w_answer * loss_answer + w_reason * loss_reason
    
    return total_loss, loss_format.item(), loss_reason.item(), loss_answer.item()

# =====================
# TRAINING LOOP
# =====================
best_val_loss = float("inf")
early_stop_counter = 0
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

print(f"[INFO] Starting adaptive training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    # Get adaptive weights for this epoch
    w_f, w_a, w_r = get_adaptive_weights(epoch, EPOCHS)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n[Epoch {epoch+1}/{EPOCHS}] Weights: Format={w_f:.3f}, Answer={w_a:.3f}, Reason={w_r:.3f}, LR={current_lr:.2e}")
    
    # =================== TRAINING ===================
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    sum_loss = sum_f = sum_r = sum_a = 0
    loop = tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast_ctx():
            loss, f, r, a = compute_adaptive_loss(model, batch, (w_f, w_a, w_r))
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        sum_loss += loss.item() * accum_steps
        sum_f += f
        sum_r += r
        sum_a += a
        
        loop.set_postfix({"loss": f"{sum_loss/(step+1):.4f}"})
    
    train_loss = sum_loss / len(train_loader)
    train_f = sum_f / len(train_loader)
    train_r = sum_r / len(train_loader)
    train_a = sum_a / len(train_loader)
    
    # =================== VALIDATION ===================
    model.eval()
    val_loss = val_f = val_r = val_a = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val E{epoch+1}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            total, f, r, a = compute_adaptive_loss(model, batch, (w_f, w_a, w_r))
            val_loss += total.item()
            val_f += f
            val_r += r
            val_a += a
    
    val_loss /= len(val_loader)
    val_f /= len(val_loader)
    val_r /= len(val_loader)
    val_a /= len(val_loader)
    
    print(f"[E{epoch+1}] Train Loss: {train_loss:.4f} (F:{train_f:.4f} A:{train_a:.4f} R:{train_r:.4f})")
    print(f"[E{epoch+1}] Val Loss:   {val_loss:.4f} (F:{val_f:.4f} A:{val_a:.4f} R:{val_r:.4f})")
    
    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [
        epoch+1, train_loss, train_f, train_r, train_a,
        val_loss, val_f, val_r, val_a,
        w_f, w_r, w_a, current_lr
    ]
    df.to_csv(LOG_CSV, index=False)
    
    # LR scheduler step
    scheduler_cosine.step()
    
    # =================== CHECKPOINT SAVING ===================
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[INFO] ⭐ NEW BEST MODEL saved! Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
    
    # Save periodic checkpoints
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print("[INFO] ⚠️ Early stopping triggered.")
        break

# =====================
# SAVE FINAL MODEL
# =====================
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"\n[INFO] ✅ Training complete! Final model saved: {FINAL_MODEL_PATH}")
print(f"[INFO] ✅ Best model: {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
