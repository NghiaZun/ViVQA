import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import BlipImageProcessor, get_cosine_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

from dataset import VQAGenDataset
from model import VQAGenModel

# ========================
# Knowledge Distillation Loss
# ========================
def distillation_loss(student_logits, teacher_logits, T=2.0):
    """
    Tính KL divergence giữa soft targets của teacher và student
    
    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        T: temperature để làm mềm distribution
    
    Returns:
        KD loss (scalar)
    """
    # Reshape để tính toán
    student_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    
    # Soft targets với temperature
    soft_teacher = F.softmax(teacher_flat / T, dim=-1)
    soft_student = F.log_softmax(student_flat / T, dim=-1)
    
    # KL divergence (scale với T^2 để cân bằng gradient)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    
    return kd_loss


# ========================
# Training Loop với KD
# ========================
def train_kd_epoch(student_model, teacher_model, train_loader, optimizer, scaler, 
                   device, cfg, scheduler=None, alpha=0.5, T=2.0):
    """
    Train 1 epoch với Knowledge Distillation
    
    Args:
        student_model: model học sinh cần train
        teacher_model: model giáo viên (frozen)
        alpha: trọng số CE loss (1-alpha cho KD loss)
        T: temperature
    """
    student_model.train()
    teacher_model.eval()  # Teacher luôn ở chế độ eval
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    steps = 0
    
    pbar = tqdm(train_loader, desc="Training with KD", leave=False)
    
    for step, batch in enumerate(pbar):
        pixel_values, input_ids, attention_mask, labels = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if cfg.use_amp:
            with autocast(dtype=torch.float16):
                # Forward student
                student_loss, student_logits = student_model(
                    pixel_values, input_ids, attention_mask, labels=labels
                )
                
                # Forward teacher (no grad)
                with torch.no_grad():
                    _, teacher_logits = teacher_model(
                        pixel_values, input_ids, attention_mask, labels=labels
                    )
                
                # CE loss (đã có từ student_loss)
                loss_ce = student_loss
                
                # KD loss
                loss_kd = distillation_loss(student_logits, teacher_logits, T=T)
                
                # Combined loss
                loss = alpha * loss_ce + (1 - alpha) * loss_kd
                loss = loss / cfg.accum_steps
            
            scaler.scale(loss).backward()
        else:
            # Forward student
            student_loss, student_logits = student_model(
                pixel_values, input_ids, attention_mask, labels=labels
            )
            
            # Forward teacher (no grad)
            with torch.no_grad():
                _, teacher_logits = teacher_model(
                    pixel_values, input_ids, attention_mask, labels=labels
                )
            
            # CE loss
            loss_ce = student_loss
            
            # KD loss
            loss_kd = distillation_loss(student_logits, teacher_logits, T=T)
            
            # Combined loss
            loss = alpha * loss_ce + (1 - alpha) * loss_kd
            loss = loss / cfg.accum_steps
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % cfg.accum_steps == 0:
            if cfg.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        # Logging
        running_loss += loss.item() * cfg.accum_steps
        running_ce += loss_ce.item()
        running_kd += loss_kd.item()
        steps += 1
        
        pbar.set_description(
            f"Loss: {running_loss/steps:.4f} | "
            f"CE: {running_ce/steps:.4f} | "
            f"KD: {running_kd/steps:.4f}"
        )
    
    return {
        "total_loss": running_loss / max(steps, 1),
        "ce_loss": running_ce / max(steps, 1),
        "kd_loss": running_kd / max(steps, 1)
    }


# ========================
# Validation Loop
# ========================
def validate_epoch(model, val_loader, device, cfg):
    """Validation loop chuẩn"""
    model.eval()
    running_loss = 0.0
    steps = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            pixel_values, input_ids, attention_mask, labels = batch
            pixel_values = pixel_values.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if cfg.use_amp:
                with autocast(dtype=torch.float16):
                    loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
            else:
                loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
            
            running_loss += loss.item()
            steps += 1
            pbar.set_description(f"Val loss: {running_loss/steps:.4f}")
    
    return running_loss / max(steps, 1)


# ========================
# Main Training Function với KD
# ========================
def train_with_kd(cfg, teacher_checkpoint_path, alpha=0.5, T=2.0):
    """
    Main training function với Knowledge Distillation
    
    Args:
        cfg: TrainConfig object
        teacher_checkpoint_path: đường dẫn đến checkpoint của teacher model
        alpha: trọng số cho CE loss (0.5 = cân bằng CE và KD)
        T: temperature (2.0-4.0 thường tốt)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] KD Config: alpha={alpha}, T={T}")
    
    # Load datasets
    vision_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
    full_dataset = VQAGenDataset(cfg.csv_path, cfg.image_folder, vision_processor)
    
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    
    # Load teacher model (frozen)
    print(f"[INFO] Loading teacher model from {teacher_checkpoint_path}")
    teacher_model = VQAGenModel().to(device)
    teacher_model.load_state_dict(torch.load(teacher_checkpoint_path, map_location=device))
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    # Initialize student model
    print("[INFO] Initializing student model")
    student_model = VQAGenModel().to(device)
    
    # Setup optimizer và scheduler
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay
    )
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    train_steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
    total_train_steps = train_steps_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(total_train_steps * cfg.warmup_ratio))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps
    )
    
    # Setup logging
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, "kd_train_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "epoch", "lr", "train_loss", "train_ce", "train_kd", 
            "val_loss", "best_val", "es_counter"
        ]).to_csv(log_path, index=False)
    
    best_val = float("inf")
    es_counter = 0
    
    # Training loop
    for epoch in range(cfg.num_epochs):
        print(f"\n[EPOCH {epoch+1}/{cfg.num_epochs}]")
        
        # Train
        train_metrics = train_kd_epoch(
            student_model, teacher_model, train_loader,
            optimizer, scaler, device, cfg, scheduler,
            alpha=alpha, T=T
        )
        
        # Validate
        val_loss = validate_epoch(student_model, val_loader, device, cfg)
        
        # Early stopping check
        improved = (best_val - val_loss) > cfg.es_min_delta
        if improved:
            best_val = val_loss
            es_counter = 0
            torch.save(student_model.state_dict(), 
                      os.path.join(cfg.save_dir, "best_kd_model.pth"))
            print(f"[INFO] New best model saved! Val loss: {val_loss:.4f}")
        else:
            es_counter += 1
        
        # Save last checkpoint
        torch.save(student_model.state_dict(), 
                  os.path.join(cfg.save_dir, "last_kd_model.pth"))
        
        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_metrics["total_loss"],
            "train_ce": train_metrics["ce_loss"],
            "train_kd": train_metrics["kd_loss"],
            "val_loss": val_loss,
            "best_val": best_val,
            "es_counter": es_counter
        }
        
        df = pd.read_csv(log_path)
        df.loc[len(df)] = row
        df.to_csv(log_path, index=False)
        
        print(f"LR: {current_lr:.2e} | "
              f"Train: {train_metrics['total_loss']:.4f} "
              f"(CE: {train_metrics['ce_loss']:.4f}, KD: {train_metrics['kd_loss']:.4f}) | "
              f"Val: {val_loss:.4f} | Best: {best_val:.4f} | ES: {es_counter}/{cfg.es_patience}")
        
        # Early stopping
        if es_counter >= cfg.es_patience:
            print("[INFO] Early stopping triggered!")
            break
    
    print("\n[INFO] Training completed!")
    print(f"[INFO] Best model saved at: {os.path.join(cfg.save_dir, 'best_kd_model.pth')}")
    
    return student_model


# ========================
# Example Usage
# ========================
if __name__ == "__main__":
    from train import TrainConfig, set_seed
    
    set_seed(42)
    
    # Config
    cfg = TrainConfig()
    cfg.num_epochs = 30
    cfg.batch_size = 4
    cfg.accum_steps = 8
    cfg.base_lr = 1e-4  # có thể thấp hơn vì học từ teacher
    cfg.save_dir = "/kaggle/working/kd_checkpoints"
    
    # Teacher checkpoint (model đã train tốt)
    teacher_checkpoint = "/kaggle/working/checkpoints/best_model.pth"
    
    # Train với KD
    student_model = train_with_kd(
        cfg=cfg,
        teacher_checkpoint_path=teacher_checkpoint,
        alpha=0.5,  # 50% CE, 50% KD
        T=2.0       # temperature
    )
