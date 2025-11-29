"""
Hybrid Distillation: Strategy 2 (Soft Target) + Strategy 3 (Vision-Text Alignment)

Key ideas:
1. Distill teacher's probability distribution (soft targets)
2. Align vision and text embeddings with teacher
3. Focus on answer only (no reasoning)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import (
    BlipProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup
)
from model import VQAGenModel

# =====================
# CONFIG
# =====================
TRAIN_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
TEACHER_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
SAVE_DIR = "/kaggle/working"

# Hyperparameters
BATCH_SIZE = 4
EPOCHS = 15
LR = 3e-5
WARMUP_RATIO = 0.1
VAL_RATIO = 0.1

# Loss weights
ALPHA_KL = 0.3        # KL divergence (soft target)
ALPHA_VISION = 0.2    # Vision embedding alignment
ALPHA_TEXT = 0.1      # Text embedding alignment
ALPHA_CE = 0.4        # Cross-entropy (answer)

TEMPERATURE = 3.0     # Temperature for soft targets
MAX_ANSWER_LEN = 32   # Only answer, no reasoning

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[CONFIG] Device: {device}")
print(f"[CONFIG] Loss weights: KL={ALPHA_KL}, Vision={ALPHA_VISION}, Text={ALPHA_TEXT}, CE={ALPHA_CE}")

# =====================
# LOAD MODELS
# =====================
print("\n[INFO] Loading Student model...")
student = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)
student = student.to(device)

print("[INFO] Loading Teacher model (Qwen2-VL)...")
teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
teacher = AutoModelForVision2Seq.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

student_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

print("[INFO] Models loaded successfully!")

# =====================
# DATASET
# =====================
class HybridDistillDataset(Dataset):
    def __init__(self, csv_path, image_dir, student_processor, teacher_processor, 
                 student_text_tokenizer, student_decoder_tokenizer, max_len=MAX_ANSWER_LEN):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor
        self.student_text_tokenizer = student_text_tokenizer
        self.student_decoder_tokenizer = student_decoder_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['img_id']}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        question = str(row["question"])
        answer = str(row["answer"])  # Ground truth answer only
        
        # Student inputs
        student_pixel_values = self.student_processor(image, return_tensors="pt").pixel_values[0]
        student_q_enc = self.student_text_tokenizer(
            question, truncation=True, padding="max_length",
            max_length=64, return_tensors="pt"
        )
        student_a_enc = self.student_decoder_tokenizer(
            answer, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        
        # Teacher inputs (for runtime processing)
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "student_pixel_values": student_pixel_values,
            "student_input_ids": student_q_enc.input_ids[0],
            "student_attention_mask": student_q_enc.attention_mask[0],
            "student_labels": student_a_enc.input_ids[0]
        }

# =====================
# TEACHER FORWARD
# =====================
@torch.no_grad()
def get_teacher_outputs(images, questions):
    """
    Get teacher's soft targets and embeddings
    """
    # Prepare teacher inputs
    messages_batch = []
    for img, q in zip(images, questions):
        messages_batch.append([
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q}
            ]}
        ])
    
    # Process batch
    texts = [teacher_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
             for msg in messages_batch]
    
    inputs = teacher_processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(teacher.device)
    
    # Forward pass
    outputs = teacher(**inputs, output_hidden_states=True, return_dict=True)
    
    # Extract embeddings (last hidden state)
    teacher_embeddings = outputs.hidden_states[-1].mean(dim=1)  # [batch, hidden_dim]
    
    # Get logits for soft targets
    teacher_logits = outputs.logits  # [batch, seq, vocab]
    
    return teacher_logits, teacher_embeddings

# =====================
# HYBRID LOSS FUNCTION
# =====================
def compute_hybrid_loss(student, batch, teacher_logits=None, teacher_embeddings=None):
    """
    Combined loss:
    1. KL divergence (soft targets)
    2. Vision embedding alignment
    3. Text embedding alignment  
    4. Cross-entropy (ground truth)
    """
    pixel_values = batch["student_pixel_values"].to(device)
    input_ids = batch["student_input_ids"].to(device)
    attention_mask = batch["student_attention_mask"].to(device)
    labels = batch["student_labels"].to(device)
    
    # Student forward pass
    outputs = student(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    student_logits = outputs.logits  # [batch, seq, vocab]
    
    # Loss 1: Cross-entropy (ground truth)
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Loss 2: KL divergence (soft targets from teacher)
    loss_kl = torch.tensor(0.0, device=device)
    if teacher_logits is not None:
        # Align sequence lengths
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits_aligned = student_logits[:, :min_seq_len, :]
        teacher_logits_aligned = teacher_logits[:, :min_seq_len, :].to(device)
        
        # Apply temperature
        student_log_probs = F.log_softmax(student_logits_aligned / TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(teacher_logits_aligned / TEMPERATURE, dim=-1)
        
        # KL divergence
        loss_kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (TEMPERATURE ** 2)
    
    # Loss 3 & 4: Vision and Text embedding alignment
    loss_vision = torch.tensor(0.0, device=device)
    loss_text = torch.tensor(0.0, device=device)
    
    if teacher_embeddings is not None:
        # Get student embeddings
        with torch.no_grad():
            v_out = student.vision_encoder(pixel_values=pixel_values).last_hidden_state
            v_feat = v_out.mean(dim=1)  # [batch, hidden_dim]
            
            t_out = student.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            t_feat = t_out[:, 0, :]  # CLS token
        
        # Normalize embeddings
        student_vision_emb = F.normalize(v_feat, dim=-1)
        student_text_emb = F.normalize(t_feat, dim=-1)
        teacher_emb_norm = F.normalize(teacher_embeddings.to(device), dim=-1)
        
        # MSE loss (or cosine similarity)
        loss_vision = F.mse_loss(student_vision_emb, teacher_emb_norm)
        loss_text = F.mse_loss(student_text_emb, teacher_emb_norm)
    
    # Combined loss
    total_loss = (
        ALPHA_CE * loss_ce +
        ALPHA_KL * loss_kl +
        ALPHA_VISION * loss_vision +
        ALPHA_TEXT * loss_text
    )
    
    return total_loss, loss_ce.item(), loss_kl.item(), loss_vision.item(), loss_text.item()

# =====================
# TRAINING LOOP
# =====================
def train():
    # Dataset
    dataset = HybridDistillDataset(
        TRAIN_CSV, IMAGE_DIR,
        student_processor, teacher_processor,
        student.text_tokenizer, student.decoder_tokenizer
    )
    
    # Train/Val split
    n_val = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    
    print(f"\n{'='*70}")
    print("HYBRID DISTILLATION TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*70}")
        
        # TRAINING
        student.train()
        total_loss = 0
        total_ce = 0
        total_kl = 0
        total_vision = 0
        total_text = 0
        
        loop = tqdm(train_loader, desc="Train")
        for batch in loop:
            optimizer.zero_grad()
            
            # Get teacher outputs
            images = batch["image"]
            questions = batch["question"]
            
            try:
                teacher_logits, teacher_embeddings = get_teacher_outputs(images, questions)
            except Exception as e:
                print(f"[WARN] Teacher forward failed: {e}")
                teacher_logits, teacher_embeddings = None, None
            
            # Compute loss
            with autocast():
                loss, ce, kl, vision, text = compute_hybrid_loss(
                    student, batch, teacher_logits, teacher_embeddings
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_ce += ce
            total_kl += kl
            total_vision += vision
            total_text += text
            
            loop.set_postfix({
                "loss": f"{total_loss/(loop.n+1):.4f}",
                "ce": f"{ce:.4f}",
                "kl": f"{kl:.4f}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        
        # VALIDATION
        student.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                loss, _, _, _, _ = compute_hybrid_loss(student, batch, None, None)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nTrain Loss: {avg_train_loss:.4f} (CE={total_ce/len(train_loader):.4f}, KL={total_kl/len(train_loader):.4f}, Vision={total_vision/len(train_loader):.4f}, Text={total_text/len(train_loader):.4f})")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, "vqa_hybrid_best.pt"))
            print(f"[SAVE] Best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Periodic save
        if (epoch + 1) % 5 == 0:
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, f"vqa_hybrid_epoch{epoch+1}.pt"))
    
    # Final save
    torch.save(student.state_dict(), os.path.join(SAVE_DIR, "vqa_hybrid_final.pt"))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Models saved to: {SAVE_DIR}")

if __name__ == "__main__":
    train()
