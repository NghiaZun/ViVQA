"""
ULTIMATE Training Script - Multi-Task Learning with Format Enforcement
Key improvements:
1. Multi-task learning: 3 separate objectives (Answer, Reasoning, Format)
2. Label smoothing (0.1) for better generalization
3. Curriculum learning with dynamic weights across 3 stages
4. LR warmup + gradient clipping for training stability
5. Format validation during inference to ensure parseable XML output
6. Mixed precision training with gradient accumulation
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
import copy

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
LR = 5e-6  # Reduced LR for multi-task learning stability
BATCH_SIZE = 2  # Reduced to 2 for 3x forward passes per batch
VAL_RATIO = 0.1
MAX_A_LEN = 160
EARLY_STOP_PATIENCE = 15
accum_steps = 4  # Increased to maintain effective batch size of 8
WARMUP_EPOCHS = 3  # Warmup for stable multi-task training

# Resume training
RESUME_FROM = "/kaggle/input/vivqa-checkpoint/transformers/default/1/latest_checkpoint.pt"
AUTO_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pt")  # Auto-saved every epoch

# Memory optimization flags
USE_GRADIENT_CHECKPOINTING = True
EMPTY_CACHE_EVERY_N_STEPS = 50
PIN_MEMORY = False  # Set False if OOM issues
GRADIENT_CLIP_VALUE = 1.0  # Clip gradients for stability
USE_MIXED_PRECISION = True  # Enable AMP for memory/speed

# EMA (Exponential Moving Average) for model stability
USE_EMA = True
EMA_DECAY = 0.999  # 0.999 for smooth averaging

# Curriculum stages - Adjusted for multi-task learning
STAGE_1_EPOCHS = 20  # Focus on answer accuracy
STAGE_2_EPOCHS = 40  # Balance all components
STAGE_3_EPOCHS = 60  # Emphasize reasoning quality

# Format validation during training
VALIDATE_FORMAT_EVERY_N_EPOCHS = 5  # Check format quality every N epochs

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
# EMA (EXPONENTIAL MOVING AVERAGE)
# =====================
class EMA:
    """
    Exponential Moving Average for model parameters
    Helps stabilize training and improve generalization
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        """Register model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# =====================
# FORMAT VALIDATION UTILITIES
# =====================
def validate_format(text):
    """
    Validate if generated text has correct XML-like format:
    <answer>...</answer>
    <reasoning>[TYPE] ...</reasoning>
    """
    import re
    
    # Check for answer tags
    answer_pattern = r'<answer>(.*?)</answer>'
    reasoning_pattern = r'<reasoning>\[(.*?)\](.*?)</reasoning>'
    
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    
    has_answer = answer_match is not None
    has_reasoning = reasoning_match is not None
    
    return {
        'valid': has_answer and has_reasoning,
        'has_answer': has_answer,
        'has_reasoning': has_reasoning,
        'answer': answer_match.group(1).strip() if answer_match else None,
        'reasoning_type': reasoning_match.group(1).strip() if reasoning_match else None,
        'reasoning_text': reasoning_match.group(2).strip() if reasoning_match else None
    }

def fix_format(text):
    """
    Attempt to fix malformed output to ensure parseable format
    """
    import re
    
    # Extract answer (look for any content before reasoning or end)
    answer = ""
    reasoning_type = "NEUTRAL"
    reasoning_text = ""
    
    # Try to find answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Try to extract first meaningful text before reasoning
        before_reasoning = re.split(r'<reasoning>', text)[0]
        answer = re.sub(r'<[^>]+>', '', before_reasoning).strip()
        if len(answer) > 100:  # Too long, truncate
            answer = answer[:100]
    
    # Try to find reasoning
    reasoning_match = re.search(r'<reasoning>\[(.*?)\](.*?)</reasoning>', text, re.DOTALL)
    if reasoning_match:
        reasoning_type = reasoning_match.group(1).strip()
        reasoning_text = reasoning_match.group(2).strip()
    else:
        # Try to find reasoning without type
        reasoning_match2 = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
        if reasoning_match2:
            reasoning_text = reasoning_match2.group(1).strip()
        else:
            # Extract any text after answer
            after_answer = text.split('</answer>')[-1] if '</answer>' in text else text
            reasoning_text = re.sub(r'<[^>]+>', '', after_answer).strip()
    
    # Reconstruct valid format
    fixed = f"<answer>{answer}</answer>\n<reasoning>[{reasoning_type}] {reasoning_text}</reasoning>"
    return fixed

def validate_model_format(model, val_loader, device, tokenizer, vision_processor, num_samples=5):
    """
    Validate that model generates parseable format on validation samples
    Returns: (valid_count, total_count, sample_outputs)
    """
    model.eval()
    valid_count = 0
    outputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            # Get first sample from batch
            pixel_values = batch["pixel_values"][0:1].to(device)
            input_ids = batch["input_ids"][0:1].to(device)
            attention_mask = batch["attention_mask"][0:1].to(device)
            
            # Generate
            generated = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=160,
                num_beams=1,
                do_sample=False
            )
            
            # Decode
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Validate format
            validation = validate_format(text)
            if validation['valid']:
                valid_count += 1
            
            outputs.append({
                'text': text,
                'valid': validation['valid'],
                'has_answer': validation['has_answer'],
                'has_reasoning': validation['has_reasoning']
            })
    
    return valid_count, num_samples, outputs

# =====================
# CURRICULUM WEIGHT SCHEDULER
# =====================
def get_curriculum_weights(epoch):
    """
    3-stage curriculum learning optimized for multi-task:
    Stage 1 (0-20): Master answer generation first
    Stage 2 (20-40): Balance all components equally
    Stage 3 (40+): Emphasize reasoning quality
    """
    if epoch < STAGE_1_EPOCHS:
        # Stage 1: Answer accuracy is critical foundation
        return {
            "answer": 0.50,  # High weight on getting answers right
            "format": 0.35,  # Medium weight on structure
            "reason": 0.15,  # Low weight on reasoning details
            "stage": "ANSWER_MASTERY"
        }
    elif epoch < STAGE_2_EPOCHS:
        # Stage 2: Balanced multi-task learning
        return {
            "answer": 0.35,  # Still important
            "format": 0.35,  # Equally important
            "reason": 0.30,  # Growing importance
            "stage": "BALANCED_LEARNING"
        }
    else:
        # Stage 3: Reasoning quality refinement
        progress = (epoch - STAGE_2_EPOCHS) / (EPOCHS - STAGE_2_EPOCHS)
        w_answer = 0.25 - 0.05 * progress      # 0.25 → 0.20 (maintain but reduce)
        w_format = 0.25 - 0.05 * progress      # 0.25 → 0.20 (maintain structure)
        w_reason = 0.50 + 0.10 * progress      # 0.50 → 0.60 (increase quality)
        
        return {
            "answer": w_answer,
            "format": w_format,
            "reason": w_reason,
            "stage": "REASONING_REFINEMENT"
        }

# =====================
# DATASET
# =====================
class CurriculumDistillDataset(Dataset):
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
        
        # Image augmentation for training (reduce overfitting)
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
            ])
        else:
            self.augment_transform = None
        
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
        
        # Apply augmentation if enabled (training only)
        if self.augment and self.augment_transform is not None:
            img = self.augment_transform(img)

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
# MULTI-TASK LOSS WITH LABEL SMOOTHING
# =====================
class MultiTaskLoss(nn.Module):
    """
    Multi-task learning loss for answer, reasoning, and format
    Features:
    1. Label smoothing for better generalization
    2. Separate objectives for each component
    3. Special token emphasis for format learning
    """
    def __init__(self, tokenizer, label_smoothing=0.1, format_weight=2.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_smoothing = label_smoothing
        self.format_weight = format_weight
        
        # Get special token IDs for format emphasis
        special_tokens = ["<answer>", "</answer>", "<reasoning>", "</reasoning>"]
        self.special_token_ids = [
            tokenizer.convert_tokens_to_ids(tok) 
            for tok in special_tokens 
            if tok in tokenizer.get_vocab()
        ]
    
    def forward(self, logits, labels, use_smoothing=True):
        """
        Compute cross-entropy with optional label smoothing and format emphasis
        """
        vocab_size = logits.size(-1)
        
        # Label smoothing
        if use_smoothing and self.label_smoothing > 0:
            # Smooth labels: true_label gets (1-smoothing), others get smoothing/(vocab_size-1)
            with torch.no_grad():
                smooth_labels = torch.zeros_like(logits)
                smooth_labels.fill_(self.label_smoothing / (vocab_size - 1))
                
                # Set true label probabilities
                labels_expanded = labels.unsqueeze(-1)
                mask = labels_expanded != -100
                smooth_labels.scatter_(-1, labels_expanded.clamp(min=0), 
                                      1.0 - self.label_smoothing)
                smooth_labels = smooth_labels * mask.float()
            
            # Cross-entropy with smooth labels
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smooth_labels * log_probs).sum(dim=-1)
            
            # Mask padding tokens
            loss = loss.masked_fill(labels == -100, 0.0)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            )
        
        # Emphasize format tokens
        weights = torch.ones_like(labels, dtype=torch.float)
        for token_id in self.special_token_ids:
            weights[labels == token_id] = self.format_weight
        
        # Apply weights and reduce
        weighted_loss = (loss * weights.view(-1)).sum() / weights.view(-1).sum().clamp(min=1.0)
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

# Initialize multi-task loss with label smoothing
multi_task_loss_fn = MultiTaskLoss(
    model.decoder_tokenizer, 
    label_smoothing=0.1,  # Smooth labels to prevent overfitting
    format_weight=2.5     # Emphasize format tokens
)

# Optimizer & Scheduler with warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Warmup + Cosine Annealing for stable multi-task training
def get_lr_multiplier(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS - WARMUP_EPOCHS, 
    eta_min=1e-7
)

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
# Create separate datasets for train (with augmentation) and val (no augmentation)
train_dataset = CurriculumDistillDataset(
    DATA_PATH, vision_processor, 
    model.text_tokenizer, model.decoder_tokenizer,
    augment=True  # Enable augmentation for training
)
val_dataset = CurriculumDistillDataset(
    DATA_PATH, vision_processor,
    model.text_tokenizer, model.decoder_tokenizer,
    augment=False  # No augmentation for validation
)

n_val = max(1, int(len(train_dataset) * VAL_RATIO))
indices = list(range(len(train_dataset)))
random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_loader = DataLoader(
    Subset(train_dataset, train_indices), 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=2, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
val_loader = DataLoader(
    Subset(val_dataset, val_indices), 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=2,
    persistent_workers=True
)

print(f"[INFO] Dataset: {len(train_dataset)} total, {len(train_indices)} train, {len(val_indices)} val")
print(f"[INFO] Training augmentation: Horizontal flip, Color jitter, Rotation")
print(f"[INFO] Effective batch size: {BATCH_SIZE * accum_steps}")
print(f"[INFO] Steps per epoch: {len(train_loader) // accum_steps}")

# =====================
# OPTIMIZER & SCHEDULER (already initialized above with resume support)
# =====================
# optimizer and scheduler are already created earlier with checkpoint loading
# Just update the scaler
scaler = torch.cuda.amp.GradScaler()

# =====================
# EMA INITIALIZATION
# =====================
ema = None
if USE_EMA:
    ema = EMA(model, decay=EMA_DECAY)
    print(f"[INFO] EMA enabled with decay={EMA_DECAY}")

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
# MULTI-TASK LOSS COMPUTATION
# =====================
def compute_curriculum_loss(model, batch, weights, use_smoothing=True):
    """
    Multi-task learning with separate objectives:
    1. Answer generation (short, precise)
    2. Reasoning generation (long, detailed) 
    3. Full format (structural coherence)
    """
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    # Forward pass for answer (short generation)
    out_answer = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=batch["answer_ids"]
    )
    
    # Forward pass for reasoning (long generation)
    out_reasoning = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=batch["reason_ids"]
    )
    
    # Forward pass for full format (structural learning)
    out_format = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=batch["format_ids"]
    )
    
    # Compute individual losses with label smoothing
    if hasattr(out_answer, 'logits'):
        loss_answer = multi_task_loss_fn(out_answer.logits, batch["answer_ids"], use_smoothing)
        loss_reasoning = multi_task_loss_fn(out_reasoning.logits, batch["reason_ids"], use_smoothing)
        loss_format = multi_task_loss_fn(out_format.logits, batch["format_ids"], use_smoothing)
    else:
        # Fallback to standard loss
        loss_answer = out_answer[0] if isinstance(out_answer, tuple) else out_answer.loss
        loss_reasoning = out_reasoning[0] if isinstance(out_reasoning, tuple) else out_reasoning.loss
        loss_format = out_format[0] if isinstance(out_format, tuple) else out_format.loss
    
    # Apply reasoning quality weighting (if available)
    if "reasoning_weight" in batch:
        loss_reasoning = loss_reasoning * batch["reasoning_weight"].mean()
    
    # Weighted combination based on curriculum stage
    total_loss = (
        weights["answer"] * loss_answer +
        weights["reason"] * loss_reasoning +
        weights["format"] * loss_format
    )
    
    return total_loss, loss_format.item(), loss_reasoning.item(), loss_answer.item()

# =====================
# TRAINING LOOP
# =====================
# best_val_loss and early_stop_counter already initialized at the top with resume support
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

print(f"\n{'='*70}")
print(f"MULTI-TASK CURRICULUM TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"[STRATEGY] 3 separate objectives: Answer + Reasoning + Format")
print(f"[CURRICULUM] Stage 1 (0-{STAGE_1_EPOCHS}): Answer Mastery (50%/35%/15%)")
print(f"[CURRICULUM] Stage 2 ({STAGE_1_EPOCHS}-{STAGE_2_EPOCHS}): Balanced Learning (35%/35%/30%)")
print(f"[CURRICULUM] Stage 3 ({STAGE_2_EPOCHS}+): Reasoning Refinement (20%/20%/60%)")
print(f"\n[ANTI-OVERFITTING]")
print(f"  • Label Smoothing: 0.1")
print(f"  • Image Augmentation: Flip + ColorJitter + Rotation")
print(f"  • EMA: {'Enabled' if USE_EMA else 'Disabled'} (decay={EMA_DECAY if USE_EMA else 'N/A'})")
print(f"  • Gradient Clipping: {GRADIENT_CLIP_VALUE}")
print(f"  • Weight Decay: 1e-4")
print(f"\n[OPTIMIZATION]")
print(f"  • Base LR: {LR:.2e}")
print(f"  • LR Warmup: {WARMUP_EPOCHS} epochs")
print(f"  • Scheduler: Cosine Annealing")
print(f"  • Mixed Precision: {'Enabled' if USE_MIXED_PRECISION else 'Disabled'}")
print(f"  • Gradient Accumulation: {accum_steps} steps")
print(f"\n[FORMAT VALIDATION]")
print(f"  • Check every {VALIDATE_FORMAT_EVERY_N_EPOCHS} epochs")
print(f"  • Ensures parseable XML: <answer>...</answer> <reasoning>[TYPE]...</reasoning>")
print(f"{'='*70}")
if start_epoch > 0:
    print(f"[RESUME] Continuing from epoch {start_epoch+1}/{EPOCHS}")
print()

for epoch in range(start_epoch, EPOCHS):
    # Clear memory at start of epoch
    clear_memory()
    
    # Get curriculum weights
    weights = get_curriculum_weights(epoch)
    
    # Apply warmup to learning rate
    if epoch < WARMUP_EPOCHS:
        lr_mult = get_lr_multiplier(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR * lr_mult
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Stage: {weights['stage']}")
    print(f"Task Weights: Answer={weights['answer']:.3f}, Format={weights['format']:.3f}, Reason={weights['reason']:.3f}")
    print(f"LR: {current_lr:.2e}" + (f" [Warmup {epoch+1}/{WARMUP_EPOCHS}]" if epoch < WARMUP_EPOCHS else ""))
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA after optimizer step
            if ema is not None:
                ema.update()
        
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
            # Disable label smoothing for validation (use true labels)
            total, f, r, a = compute_curriculum_loss(model, batch, weights, use_smoothing=False)
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
    
    # Format validation check (periodic)
    if (epoch + 1) % VALIDATE_FORMAT_EVERY_N_EPOCHS == 0:
        print(f"[FORMAT CHECK] Validating output format...")
        valid_count, total_count, sample_outputs = validate_model_format(
            model, val_loader, device, model.decoder_tokenizer, vision_processor, num_samples=3
        )
        format_accuracy = valid_count / total_count * 100
        print(f"[FORMAT CHECK] Valid: {valid_count}/{total_count} ({format_accuracy:.1f}%)")
        if valid_count < total_count:
            print(f"[FORMAT WARN] Some outputs have invalid format!")
            for i, out in enumerate(sample_outputs[:2]):  # Show first 2 samples
                if not out['valid']:
                    print(f"  Sample {i+1}: has_answer={out['has_answer']}, has_reasoning={out['has_reasoning']}")
                    print(f"  Text: {out['text'][:100]}...")
    
    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [
        epoch+1, weights['stage'], train_loss, train_f, train_r, train_a,
        val_loss, val_f, val_r, val_a,
        weights['format'], weights['reason'], weights['answer'], current_lr
    ]
    df.to_csv(LOG_CSV, index=False)
    
    # LR scheduler (skip during warmup)
    if epoch >= WARMUP_EPOCHS:
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
        
        # Save best model with EMA weights if available
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
# FINAL SAVE WITH EMA
# =====================
# Save final model with EMA weights
if ema is not None:
    ema.apply_shadow()
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    ema.restore()
    print(f"\n[INFO] Final model saved with EMA weights")
else:
    torch.save(model.state_dict(), FINAL_MODEL_PATH)

# Run final format validation
print(f"\n[FINAL FORMAT CHECK] Validating output format on best model...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()
valid_count, total_count, sample_outputs = validate_model_format(
    model, val_loader, device, model.decoder_tokenizer, vision_processor, num_samples=10
)
format_accuracy = valid_count / total_count * 100

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"[SUMMARY]")
print(f"  • Best Val Loss: {best_val_loss:.4f}")
print(f"  • Total Epochs: {epoch+1}/{EPOCHS}")
print(f"  • Early Stop Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
print(f"\n[FORMAT VALIDATION] {valid_count}/{total_count} samples ({format_accuracy:.1f}%) have valid format")
if format_accuracy < 100:
    print(f"  ⚠ WARNING: Some outputs may need post-processing")
    print(f"  Use fix_format() utility to repair malformed outputs")
else:
    print(f"  ✓ All outputs are correctly formatted and parseable")
print(f"\n[SAVED MODELS]")
print(f"  • Best: {BEST_MODEL_PATH}")
print(f"  • Final: {FINAL_MODEL_PATH}")
print(f"  • Latest: {AUTO_CHECKPOINT_PATH}")
print(f"\n[LOGS]")
print(f"  • Training log: {LOG_CSV}")
print(f"{'='*70}")
print(f"\n[NEXT STEPS]")
print(f"1. Load best model: model.load_state_dict(torch.load('{BEST_MODEL_PATH}'))")
print(f"2. Generate predictions with format validation")
print(f"3. Use validate_format() and fix_format() for quality assurance")
print(f"{'='*70}")
