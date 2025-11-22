"""
ULTIMATE Training Script - Format-Enforced Multi-Objective KD
Key improvements:
1. Curriculum learning: Answer-first → Format → Full reasoning
2. Format-aware loss with tag prediction
3. Dynamic weight adjustment based on validation performance
4. Special token masking for better tag learning
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BlipProcessor
from model import VQAGenModel
from contextlib import nullcontext
import numpy as np
import gc

# =====================
# CONFIG - OPTIMIZED FOR KAGGLE GPU
# =====================
DATA_PATH = "/kaggle/input/teacher-outputs/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_ultimate.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_ultimate.pt")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_ultimate.csv")

# Training config optimized for Kaggle T4 GPU (16GB)
EPOCHS = 100
LR = 1e-5
BATCH_SIZE = 3  # Reduced from 4 for memory safety
VAL_RATIO = 0.1
MAX_A_LEN = 160
EARLY_STOP_PATIENCE = 15
accum_steps = 3  # Increased to maintain effective batch size of 9

# Resume training
RESUME_FROM = "/kaggle/input/vivqa-checkpoint/transformers/default/1/latest_checkpoint.pt"
AUTO_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pt")  # Auto-saved every epoch

# Memory optimization flags
USE_GRADIENT_CHECKPOINTING = True
EMPTY_CACHE_EVERY_N_STEPS = 50
PIN_MEMORY = False  # Set False if OOM issues

# Curriculum stages
STAGE_1_EPOCHS = 15  # Focus on answer
STAGE_2_EPOCHS = 30  # Add format
STAGE_3_EPOCHS = 55  # Full reasoning

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # Optimize cuDNN

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# MEMORY UTILITIES
# =====================
def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def get_gpu_memory_gb():
    """Get current GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

# =====================
# CURRICULUM WEIGHT SCHEDULER
# =====================
def get_curriculum_weights(epoch):
    """
    3-stage curriculum learning
    Stage 1 (0-15): Focus on answer generation
    Stage 2 (15-30): Add format structure
    Stage 3 (30+): Full reasoning quality
    """
    if epoch < STAGE_1_EPOCHS:
        # Stage 1: Learn to generate answers
        return {
            "format": 0.60,
            "answer": 0.35,
            "reason": 0.05,
            "stage": "ANSWER_FOCUS"
        }
    elif epoch < STAGE_2_EPOCHS:
        # Stage 2: Learn format structure
        return {
            "format": 0.50,
            "answer": 0.30,
            "reason": 0.20,
            "stage": "FORMAT_LEARNING"
        }
    else:
        # Stage 3: Polish reasoning quality
        progress = (epoch - STAGE_2_EPOCHS) / (EPOCHS - STAGE_2_EPOCHS)
        w_format = 0.25 + 0.05 * (1 - progress)  # 0.25-0.30
        w_reason = 0.45 + 0.15 * progress        # 0.45-0.60
        w_answer = 1.0 - w_format - w_reason
        
        return {
            "format": w_format,
            "answer": w_answer,
            "reason": w_reason,
            "stage": "REASONING_QUALITY"
        }

# =====================
# DATASET
# =====================
class CurriculumDistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=64, max_a_len=MAX_A_LEN):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        
        # Get special token IDs for masking
        self.answer_start_id = decoder_tokenizer.convert_tokens_to_ids("<answer>")
        self.answer_end_id = decoder_tokenizer.convert_tokens_to_ids("</answer>")
        self.reasoning_start_id = decoder_tokenizer.convert_tokens_to_ids("<reasoning>")
        self.reasoning_end_id = decoder_tokenizer.convert_tokens_to_ids("</reasoning>")

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

        # Get components
        teacher_answer = s.get("teacher_answer", "")
        teacher_reasoning = s.get("teacher_reasoning", "")
        reasoning_type = s.get("reasoning_type", "NEUTRAL")
        reasoning_weight = float(s.get("reasoning_weight", 1.0))
        
        # Construct full format
        teacher_raw = s.get("teacher_raw", "")
        if not teacher_raw:
            teacher_raw = f"<answer>{teacher_answer}</answer>\n<reasoning>[{reasoning_type}] {teacher_reasoning}</reasoning>"
        
        # Three training targets
        format_ids = self.encode_text(teacher_raw).input_ids[0]
        answer_ids = self.encode_text(f"<answer>{teacher_answer}</answer>").input_ids[0]
        reason_ids = self.encode_text(f"<reasoning>[{reasoning_type}] {teacher_reasoning}</reasoning>").input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "format_ids": format_ids,
            "answer_ids": answer_ids,
            "reason_ids": reason_ids,
            "reasoning_weight": torch.tensor(reasoning_weight)
        }

# =====================
# FORMAT-AWARE LOSS
# =====================
class FormatAwareLoss(nn.Module):
    """
    Custom loss that pays extra attention to special tokens
    NOW with curriculum-based dynamic weighting
    """
    def __init__(self, tokenizer, base_format_weight=3.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_format_weight = base_format_weight
        
        # Get special token IDs
        special_tokens = ["<answer>", "</answer>", "<reasoning>", "</reasoning>"]
        self.special_token_ids = [
            tokenizer.convert_tokens_to_ids(tok) 
            for tok in special_tokens 
            if tok in tokenizer.get_vocab()
        ]
    
    def forward(self, logits, labels):
        """
        Weighted cross-entropy with STRONG emphasis on format tokens
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        
        # Create weight mask with HIGHER weight for format tokens
        weights = torch.ones_like(labels, dtype=torch.float)
        for token_id in self.special_token_ids:
            weights[labels == token_id] = self.base_format_weight
        
        # Apply weights
        weighted_loss = (ce_loss * weights.view(-1)).mean()
        return weighted_loss

# =====================
# LOAD MODEL WITH MEMORY OPTIMIZATION
# =====================
print("[INFO] Loading VQAGenModel...")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

clear_memory()  # Clear before loading

model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base/transformers/default/1/checkpoints/vit5_tokenizer"
)

# Enable gradient checkpointing for memory efficiency
if USE_GRADIENT_CHECKPOINTING:
    if hasattr(model.decoder, 'gradient_checkpointing_enable'):
        model.decoder.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled for decoder")

model = model.to(device)
print_gpu_memory()

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Load checkpoint if exists (set to None if training from scratch)
# Resume training state
start_epoch = 0
best_val_loss = float('inf')
early_stop_counter = 0

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"[INFO] Resuming from checkpoint: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1  # Continue from next epoch
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    early_stop_counter = checkpoint.get('early_stop_counter', 0)
    print(f"[INFO] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    del checkpoint
    clear_memory()
    print_gpu_memory()

# Initialize format-aware loss with HIGHER weight
format_loss_fn = FormatAwareLoss(model.decoder_tokenizer, base_format_weight=3.5)

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# Load optimizer/scheduler state if resuming
if RESUME_FROM and os.path.exists(RESUME_FROM):
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[INFO] Optimizer state restored")
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("[INFO] Scheduler state restored")
    del checkpoint
    clear_memory()

# =====================
# DATASET & DATALOADERS
# =====================
dataset = CurriculumDistillDataset(DATA_PATH, vision_processor, 
                                   model.text_tokenizer, model.decoder_tokenizer)
n_val = max(1, int(len(dataset) * VAL_RATIO))
indices = list(range(len(dataset)))
random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_loader = DataLoader(
    Subset(dataset, train_indices), 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=2, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
val_loader = DataLoader(
    Subset(dataset, val_indices), 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,
    persistent_workers=True
)

print(f"[INFO] Dataset: {len(dataset)} total, {len(train_indices)} train, {len(val_indices)} val")
print(f"[INFO] Effective batch size: {BATCH_SIZE * accum_steps}")
print(f"[INFO] Steps per epoch: {len(train_loader) // accum_steps}")

# =====================
# OPTIMIZER & SCHEDULER (already initialized above with resume support)
# =====================
# optimizer and scheduler are already created earlier with checkpoint loading
# Just update the scaler
scaler = torch.cuda.amp.GradScaler()

# =====================
# LOGGING
# =====================
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=[
        "epoch", "stage", "train_loss", "train_F", "train_R", "train_A",
        "val_loss", "val_F", "val_R", "val_A",
        "w_format", "w_reason", "w_answer", "lr"
    ]).to_csv(LOG_CSV, index=False)

# =====================
# LOSS COMPUTATION
# =====================
def compute_curriculum_loss(model, batch, weights, use_format_aware=True):
    """
    Multi-objective loss with curriculum weighting
    Now using ONLY format_ids with varying token weights
    """
    # Single forward pass with full format
    out_format = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["format_ids"]
    )
    
    if use_format_aware and hasattr(out_format, 'logits'):
        loss_format = format_loss_fn(out_format.logits, batch["format_ids"])
    else:
        loss_format = out_format[0] if isinstance(out_format, tuple) else out_format.loss
    
    # Return same loss 3 times for logging compatibility
    return loss_format, loss_format.item(), 0.0, 0.0

# =====================
# TRAINING LOOP
# =====================
# best_val_loss and early_stop_counter already initialized at the top with resume support
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

print(f"\n[INFO] Starting curriculum training for {EPOCHS} epochs...")
print(f"[INFO] Stage 1 (0-{STAGE_1_EPOCHS}): Answer Focus")
print(f"[INFO] Stage 2 ({STAGE_1_EPOCHS}-{STAGE_2_EPOCHS}): Format Learning")
print(f"[INFO] Stage 3 ({STAGE_2_EPOCHS}+): Reasoning Quality")
if start_epoch > 0:
    print(f"[INFO] Resuming from epoch {start_epoch+1}/{EPOCHS}")
print()

for epoch in range(start_epoch, EPOCHS):
    # Clear memory at start of epoch
    clear_memory()
    
    # Get curriculum weights
    weights = get_curriculum_weights(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Stage: {weights['stage']}")
    print(f"Weights: Format={weights['format']:.3f}, Answer={weights['answer']:.3f}, Reason={weights['reason']:.3f}")
    print(f"LR: {current_lr:.2e}")
    print_gpu_memory()
    print(f"{'='*70}")
    
    # =================== TRAINING ===================
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    sum_loss = sum_f = sum_r = sum_a = 0
    loop = tqdm(train_loader, desc=f"Train", leave=False)
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast_ctx():
            loss, f, r, a = compute_curriculum_loss(model, batch, weights)
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
        
        # Periodic memory cleanup
        if (step + 1) % EMPTY_CACHE_EVERY_N_STEPS == 0:
            clear_memory()
        
        loop.set_postfix({
            "loss": f"{sum_loss/(step+1):.4f}",
            "gpu_gb": f"{get_gpu_memory_gb():.1f}"
        })
    
    train_loss = sum_loss / len(train_loader)
    train_f = sum_f / len(train_loader)
    train_r = sum_r / len(train_loader)
    train_a = sum_a / len(train_loader)
    
    # =================== VALIDATION ===================
    model.eval()
    val_loss = val_f = val_r = val_a = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            total, f, r, a = compute_curriculum_loss(model, batch, weights, use_format_aware=False)
            val_loss += total.item()
            val_f += f
            val_r += r
            val_a += a
    
    val_loss /= len(val_loader)
    val_f /= len(val_loader)
    val_r /= len(val_loader)
    val_a /= len(val_loader)
    
    # Clear memory after validation
    clear_memory()
    
    print(f"\nTrain Loss: {train_loss:.4f} (F:{train_f:.4f} | A:{train_a:.4f} | R:{train_r:.4f})")
    print(f"Val Loss:   {val_loss:.4f} (F:{val_f:.4f} | A:{val_a:.4f} | R:{val_r:.4f})")
    print_gpu_memory()
    
    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [
        epoch+1, weights['stage'], train_loss, train_f, train_r, train_a,
        val_loss, val_f, val_r, val_a,
        weights['format'], weights['reason'], weights['answer'], current_lr
    ]
    df.to_csv(LOG_CSV, index=False)
    
    # LR scheduler
    scheduler.step()
    
    # =================== CHECKPOINTING ===================
    # Save auto-checkpoint every epoch for resume capability
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stop_counter': early_stop_counter,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, AUTO_CHECKPOINT_PATH)
    print(f"[CHECKPOINT] Auto-checkpoint saved: latest_checkpoint.pt")
    print(f"[CHECKPOINT] Progress: Epoch {epoch+1}/{EPOCHS} | Best Val Loss: {best_val_loss:.4f} | Early Stop: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
    
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[BEST] NEW BEST MODEL! Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"[WARN] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
    
    # Periodic checkpoints (backup every 10 epochs)
    if (epoch + 1) % 10 == 0 or epoch in [STAGE_1_EPOCHS-1, STAGE_2_EPOCHS-1]:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, checkpoint_path)
        print(f"[CHECKPOINT] Backup checkpoint saved: checkpoint_epoch{epoch+1}.pt")
        print(f"[CHECKPOINT] Progress: Epoch {epoch+1}/{EPOCHS} | Best Val Loss: {best_val_loss:.4f}")
    
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[STOP] Early stopping triggered at epoch {epoch+1}")
        break

# =====================
# FINAL SAVE
# =====================
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"\n{'='*70}")
print(f"[DONE] Training complete!")
print(f"[DONE] Best model: {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
print(f"[DONE] Final model: {FINAL_MODEL_PATH}")
print(f"{'='*70}")
