"""
FAST TRAINING VERSION - Reduced epochs for quick testing
Optimized for Kaggle GPU with auto-checkpoint and resume capability
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BlipProcessor
from model import VQAGenModel
import json
import random
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
from contextlib import nullcontext
import numpy as np
import gc

# =====================
# CONFIG - FAST TRAINING MODE
# =====================
DATA_PATH = "/kaggle/input/teacher-outputs/teacher_outputs.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_best_fast.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_student_final_fast.pt")
LOG_CSV = os.path.join(SAVE_DIR, "train_val_log_fast.csv")

# FAST TRAINING CONFIG - Reduced epochs, larger batch
EPOCHS = 30  # Reduced from 100
LR = 2e-5  # Slightly higher LR for faster convergence
BATCH_SIZE = 4  # Try larger batch first
VAL_RATIO = 0.1
MAX_A_LEN = 128  # Reduced from 160 for speed
EARLY_STOP_PATIENCE = 8  # Reduced from 15
accum_steps = 2  # Effective batch = 8

# Resume training
RESUME_FROM = None  # e.g., "/kaggle/input/my-checkpoint/latest_checkpoint.pt"
AUTO_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pt")

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True
CLEAR_CACHE_EVERY_N = 50

# Curriculum stages (compressed timeline)
STAGE_1_EPOCHS = 5   # Answer focus (0-5)
STAGE_2_EPOCHS = 12  # Format learning (5-12)
# STAGE_3: Reasoning quality (12-30)

# =====================
# CURRICULUM WEIGHTS
# =====================
def get_curriculum_weights(epoch):
    """
    Compressed curriculum for fast training
    """
    if epoch < STAGE_1_EPOCHS:
        return {
            "stage": "ANSWER_FOCUS",
            "format": 0.60,
            "answer": 0.35,
            "reason": 0.05
        }
    elif epoch < STAGE_2_EPOCHS:
        return {
            "stage": "FORMAT_LEARNING",
            "format": 0.70,
            "answer": 0.25,
            "reason": 0.05
        }
    else:
        return {
            "stage": "REASONING_QUALITY",
            "format": 0.45,
            "answer": 0.25,
            "reason": 0.30
        }

# =====================
# FORMAT-AWARE LOSS
# =====================
class FormatAwareLoss:
    def __init__(self, tokenizer, format_token_weight=2.5):
        self.tokenizer = tokenizer
        self.format_token_weight = format_token_weight
        special_tokens = [
            "<answer>", "</answer>", 
            "<reasoning>", "</reasoning>"
        ]
        self.special_ids = [
            tokenizer.convert_tokens_to_ids(t) 
            for t in special_tokens 
            if t in tokenizer.get_vocab()
        ]
    
    def __call__(self, logits, labels):
        """
        Weighted cross-entropy emphasizing format tokens
        """
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        
        weights = torch.ones_like(labels, dtype=torch.float)
        for sid in self.special_ids:
            weights[labels == sid] = self.format_token_weight
        
        weighted_loss = (loss * weights.view(-1)).mean()
        return weighted_loss

# =====================
# DATASET
# =====================
class FastDistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Image
        img = Image.open(item['image']).convert('RGB')
        pixel_values = self.vision_processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
        
        # Question
        question = item['question']
        text_enc = self.text_tokenizer(question, return_tensors='pt', 
                                       padding='max_length', max_length=50, truncation=True)
        
        # Targets
        answer_target = f"<answer>{item['answer']}</answer>"
        reason_target = f"<reasoning>{item['reasoning']}</reasoning>"
        full_target = f"{answer_target} {reason_target}"
        
        # Tokenize
        answer_ids = self.decoder_tokenizer(
            answer_target, return_tensors='pt', padding='max_length',
            max_length=MAX_A_LEN, truncation=True
        )['input_ids'].squeeze(0)
        
        reason_ids = self.decoder_tokenizer(
            reason_target, return_tensors='pt', padding='max_length',
            max_length=MAX_A_LEN, truncation=True
        )['input_ids'].squeeze(0)
        
        format_ids = self.decoder_tokenizer(
            full_target, return_tensors='pt', padding='max_length',
            max_length=MAX_A_LEN, truncation=True
        )['input_ids'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'answer_ids': answer_ids,
            'reason_ids': reason_ids,
            'format_ids': format_ids,
            'answer_text': item['answer'],
            'reason_text': item['reasoning']
        }

# =====================
# MEMORY MANAGEMENT
# =====================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

# =====================
# MAIN TRAINING
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
clear_memory()

# Load model
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base/transformers/default/1/checkpoints/vit5_tokenizer"
)

if USE_GRADIENT_CHECKPOINTING:
    model.decoder.gradient_checkpointing_enable()
    print("[INFO] Gradient checkpointing enabled for decoder")

model = model.to(device)
print_gpu_memory()

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Resume training state
start_epoch = 0
best_val_loss = float('inf')
early_stop_counter = 0

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"[INFO] 🔄 Resuming from checkpoint: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    early_stop_counter = checkpoint.get('early_stop_counter', 0)
    print(f"[INFO] ✅ Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    del checkpoint
    clear_memory()
    print_gpu_memory()

# Initialize format-aware loss
format_loss_fn = FormatAwareLoss(model.decoder_tokenizer, format_token_weight=2.5)

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# Load optimizer/scheduler state if resuming
if RESUME_FROM and os.path.exists(RESUME_FROM):
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[INFO] ✅ Optimizer state restored")
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("[INFO] ✅ Scheduler state restored")
    del checkpoint
    clear_memory()

# Dataset
dataset = FastDistillDataset(DATA_PATH, vision_processor, 
                             model.text_tokenizer, model.decoder_tokenizer)
n_val = max(1, int(len(dataset) * VAL_RATIO))
indices = list(range(len(dataset)))
random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=2, pin_memory=True)

print(f"[INFO] Dataset: {len(dataset)} total, {len(train_indices)} train, {len(val_indices)} val")
print(f"[INFO] Effective batch size: {BATCH_SIZE * accum_steps}")
print(f"[INFO] Steps per epoch: {len(train_loader) // accum_steps}")

scaler = torch.cuda.amp.GradScaler()

# Logging
pd.DataFrame(columns=[
    "epoch", "train_loss", "val_loss", "val_F", "val_R", "val_A",
    "w_format", "w_reason", "w_answer", "lr"
]).to_csv(LOG_CSV, index=False)

# =====================
# LOSS COMPUTATION
# =====================
def compute_curriculum_loss(model, batch, weights, use_format_aware=True):
    w_f, w_a, w_r = weights["format"], weights["answer"], weights["reason"]
    
    # Format loss
    out_format = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["format_ids"]
    )
    
    if use_format_aware:
        loss_format = format_loss_fn(out_format.logits, batch["format_ids"])
    else:
        loss_format = out_format.loss
    
    # Answer loss
    out_answer = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["answer_ids"]
    )
    loss_answer = out_answer.loss
    
    # Reasoning loss
    out_reason = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["reason_ids"]
    )
    loss_reason = out_reason.loss
    
    total_loss = w_f * loss_format + w_a * loss_answer + w_r * loss_reason
    return total_loss, loss_format.item(), loss_answer.item(), loss_reason.item()

# =====================
# VALIDATION
# =====================
def validate_epoch(model, val_loader, weights):
    model.eval()
    total_loss = 0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores_f, scores_r, scores_a = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            loss, _, _, _ = compute_curriculum_loss(model, batch, weights, use_format_aware=True)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader), 0.0, 0.0, 0.0  # Simplified metrics

# =====================
# TRAINING LOOP
# =====================
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

print(f"\n[INFO] ⚡ FAST TRAINING MODE - {EPOCHS} epochs")
print(f"[INFO] Stage 1 (0-{STAGE_1_EPOCHS}): Answer Focus")
print(f"[INFO] Stage 2 ({STAGE_1_EPOCHS}-{STAGE_2_EPOCHS}): Format Learning")
print(f"[INFO] Stage 3 ({STAGE_2_EPOCHS}+): Reasoning Quality")
if start_epoch > 0:
    print(f"[INFO] 🔄 Resuming from epoch {start_epoch+1}/{EPOCHS}")
print()

for epoch in range(start_epoch, EPOCHS):
    clear_memory()
    
    weights = get_curriculum_weights(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Stage: {weights['stage']}")
    print(f"Weights: Format={weights['format']:.3f}, Answer={weights['answer']:.3f}, Reason={weights['reason']:.3f}")
    print(f"LR: {current_lr:.2e}")
    print('='*70)
    
    # Training
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with autocast_ctx():
            loss, l_f, l_a, l_r = compute_curriculum_loss(model, batch, weights)
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accum_steps
        pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})
        
        if (step + 1) % CLEAR_CACHE_EVERY_N == 0:
            clear_memory()
    
    train_loss /= len(train_loader)
    
    # Validation
    val_loss, val_f, val_r, val_a = validate_epoch(model, val_loader, weights)
    
    print(f"\n📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [
        epoch+1, train_loss, val_loss, val_f, val_r, val_a,
        weights['format'], weights['reason'], weights['answer'], current_lr
    ]
    df.to_csv(LOG_CSV, index=False)
    
    # LR scheduler
    scheduler.step()
    
    # =================== CHECKPOINTING ===================
    # Auto-checkpoint every epoch
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
    
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"⭐ NEW BEST MODEL! Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"⚠️  No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
    
    # Periodic backups
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: checkpoint_epoch{epoch+1}.pt")
    
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
        break

# =====================
# FINAL SAVE
# =====================
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"\n{'='*70}")
print(f"✅ Training complete!")
print(f"✅ Best model: {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
print(f"✅ Final model: {FINAL_MODEL_PATH}")
print(f"✅ Auto-checkpoint: {AUTO_CHECKPOINT_PATH}")
print('='*70)
