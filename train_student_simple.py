"""
SIMPLE FORMAT Training Script - Natural Text Format
Thay vì XML phức tạp, train với format tự nhiên:
Answer: ...
Reasoning: ...

Key improvements:
1. Đơn giản hơn, model dễ học hơn
2. Không cần multi-task learning phức tạp
3. Dễ parse với regex đơn giản
4. Format đã proven từ generate_teacher_offline.py
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
from torchvision import transforms
from model import VQAGenModel
from contextlib import nullcontext
import numpy as np
import gc
import re

# =====================
# CONFIG
# =====================
DATA_PATH = "/kaggle/input/teacher-outputs/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_simple.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_simple.pt")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_simple.csv")

# Training config
EPOCHS = 100
LR = 3e-6
BATCH_SIZE = 4  # Tăng batch size vì chỉ 1 forward pass
VAL_RATIO = 0.2
MAX_A_LEN = 160
EARLY_STOP_PATIENCE = 20
accum_steps = 2  # Effective batch = 8
WARMUP_EPOCHS = 5

# Resume training
RESUME_FROM = None
AUTO_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint_simple.pt")

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True
EMPTY_CACHE_EVERY_N_STEPS = 50
PIN_MEMORY = False
GRADIENT_CLIP_VALUE = 0.5
USE_MIXED_PRECISION = True

# EMA
USE_EMA = True
EMA_DECAY = 0.9995

# Regularization
LABEL_SMOOTHING = 0.15
WEIGHT_DECAY = 3e-4

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# MEMORY UTILITIES
# =====================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# =====================
# EMA
# =====================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# =====================
# PARSER FOR EVALUATION
# =====================
def parse_output(text: str):
    """
    Parse output format:
    Answer: ...
    Reasoning: ...
    """
    answer = ""
    reasoning = ""
    
    # Try regex first
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Fallback: line-based parsing
    if not answer or not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip()
            elif line.lower().startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
    
    # Final fallback: first line = answer, rest = reasoning
    if not answer and not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 0:
            answer = lines[0]
        if len(lines) > 1:
            reasoning = ' '.join(lines[1:])
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid': bool(answer and reasoning)
    }

# =====================
# DATASET
# =====================
class SimpleFormatDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=64, max_a_len=MAX_A_LEN, augment=False):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.augment = augment
        
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.samples)

    def _safe_load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (255, 255, 255))

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s.get("question", "")
        img = self._safe_load_image(s.get("image_path", ""))
        
        if self.augment and self.augment_transform is not None:
            img = self.augment_transform(img)

        pixel_values = self.vision_processor(img, return_tensors="pt").pixel_values[0]
        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")

        # Create simple format target
        answer = s.get("teacher_answer", "")
        reasoning = s.get("teacher_reasoning", "")
        
        # Simple format: Answer: X\nReasoning: Y
        target_text = f"Answer: {answer}\nReasoning: {reasoning}"
        
        target_enc = self.decoder_tokenizer(
            target_text, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "labels": target_enc.input_ids[0]
        }

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading VQAGenModel...")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

clear_memory()

model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base/transformers/default/1/checkpoints/vit5_tokenizer"
)

if USE_GRADIENT_CHECKPOINTING:
    if hasattr(model.decoder, 'gradient_checkpointing_enable'):
        model.decoder.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled")

model = model.to(device)
print_gpu_memory()

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Resume if checkpoint exists
start_epoch = 0
best_val_loss = float('inf')
early_stop_counter = 0

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"[INFO] Resuming from: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    early_stop_counter = checkpoint.get('early_stop_counter', 0)
    print(f"[INFO] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    del checkpoint
    clear_memory()

# Loss function with label smoothing
def compute_loss(logits, labels):
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
        reduction='mean'
    )
    return loss

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def get_lr_multiplier(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
)

if RESUME_FROM and os.path.exists(RESUME_FROM):
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    del checkpoint
    clear_memory()

scaler = torch.cuda.amp.GradScaler()

# EMA
ema = None
if USE_EMA:
    ema = EMA(model, decay=EMA_DECAY)
    print(f"[INFO] EMA enabled with decay={EMA_DECAY}")

# =====================
# DATASET & DATALOADERS
# =====================
train_dataset = SimpleFormatDataset(
    DATA_PATH, vision_processor, 
    model.text_tokenizer, model.decoder_tokenizer,
    augment=True
)
val_dataset = SimpleFormatDataset(
    DATA_PATH, vision_processor,
    model.text_tokenizer, model.decoder_tokenizer,
    augment=False
)

n_val = max(1, int(len(train_dataset) * VAL_RATIO))
indices = list(range(len(train_dataset)))
random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_loader = DataLoader(
    Subset(train_dataset, train_indices), 
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=PIN_MEMORY,
    prefetch_factor=2, persistent_workers=True
)
val_loader = DataLoader(
    Subset(val_dataset, val_indices), 
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=PIN_MEMORY,
    prefetch_factor=2, persistent_workers=True
)

print(f"[INFO] Dataset: {len(train_dataset)} total, {len(train_indices)} train, {len(val_indices)} val")
print(f"[INFO] Effective batch size: {BATCH_SIZE * accum_steps}")

# =====================
# LOGGING
# =====================
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=[
        "epoch", "train_loss", "val_loss", "lr"
    ]).to_csv(LOG_CSV, index=False)

# =====================
# TRAINING LOOP
# =====================
print(f"\n{'='*70}")
print(f"SIMPLE FORMAT TRAINING (Answer: ... / Reasoning: ...)")
print(f"{'='*70}")
print(f"[STRATEGY] Single-task learning with natural format")
print(f"[FORMAT] Answer: <text>\\nReasoning: <text>")
print(f"\n[ANTI-OVERFITTING]")
print(f"  • Label Smoothing: {LABEL_SMOOTHING}")
print(f"  • Image Augmentation: Enabled")
print(f"  • EMA: {'Enabled' if USE_EMA else 'Disabled'} (decay={EMA_DECAY if USE_EMA else 'N/A'})")
print(f"  • Gradient Clipping: {GRADIENT_CLIP_VALUE}")
print(f"  • Weight Decay: {WEIGHT_DECAY}")
print(f"  • Validation Ratio: {VAL_RATIO*100:.0f}%")
print(f"{'='*70}\n")

if device == "cuda":
    autocast_ctx = lambda: torch.amp.autocast('cuda')
else:
    autocast_ctx = nullcontext

for epoch in range(start_epoch, EPOCHS):
    clear_memory()
    
    if epoch < WARMUP_EPOCHS:
        lr_mult = get_lr_multiplier(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR * lr_mult
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"LR: {current_lr:.2e}" + (f" [Warmup {epoch+1}/{WARMUP_EPOCHS}]" if epoch < WARMUP_EPOCHS else ""))
    print_gpu_memory()
    print(f"{'='*70}")
    
    # TRAINING
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    sum_loss = 0
    loop = tqdm(train_loader, desc=f"Train", leave=False)
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast_ctx():
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            if hasattr(outputs, 'logits'):
                loss = compute_loss(outputs.logits, batch["labels"])
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if ema is not None:
                ema.update()
        
        sum_loss += loss.item() * accum_steps
        
        if (step + 1) % EMPTY_CACHE_EVERY_N_STEPS == 0:
            clear_memory()
        
        loop.set_postfix({"loss": f"{sum_loss/(step+1):.4f}"})
    
    train_loss = sum_loss / len(train_loader)
    
    # VALIDATION
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            if hasattr(outputs, 'logits'):
                loss = compute_loss(outputs.logits, batch["labels"])
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    clear_memory()
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print_gpu_memory()
    
    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [epoch+1, train_loss, val_loss, current_lr]
    df.to_csv(LOG_CSV, index=False)
    
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()
    
    # CHECKPOINTING
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
    print(f"[CHECKPOINT] Auto-checkpoint saved")
    
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        early_stop_counter = 0
        
        if ema is not None:
            ema.apply_shadow()
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            ema.restore()
            print(f"[BEST] NEW BEST MODEL (EMA)! Val Loss: {best_val_loss:.4f}")
        else:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[BEST] NEW BEST MODEL! Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"[WARN] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
    
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_simple_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, checkpoint_path)
        print(f"[CHECKPOINT] Backup saved: checkpoint_simple_epoch{epoch+1}.pt")
    
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[STOP] Early stopping at epoch {epoch+1}")
        break

# FINAL SAVE
if ema is not None:
    ema.apply_shadow()
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    ema.restore()
else:
    torch.save(model.state_dict(), FINAL_MODEL_PATH)

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Total Epochs: {epoch+1}/{EPOCHS}")
print(f"\n[SAVED MODELS]")
print(f"  • Best: {BEST_MODEL_PATH}")
print(f"  • Final: {FINAL_MODEL_PATH}")
print(f"  • Latest: {AUTO_CHECKPOINT_PATH}")
print(f"{'='*70}\n")
