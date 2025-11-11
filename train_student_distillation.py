"""
Full Knowledge Distillation Training for VQA Student (XML Output)
- Teacher OFFLOADED TO CPU (giảm VRAM từ 15GB → 4GB)
- Memory-efficient for 16GB GPU
- Fixed fusion module issue
- Mixed precision (AMP) using new API
Author: Optimized for CPU Teacher
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BlipProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup
)
from model import VQAGenModel  # Your custom student model

# =========================
# CONFIG
# =========================
class Config:
    TEACHER_DATA = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"
    IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    STUDENT_CKPT = "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/best_model.pth"
    TEACHER_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

    SAVE_DIR = "/kaggle/working"
    SAVE_NAME = "student_distilled_xml"

    EPOCHS = 15
    BATCH_SIZE = 2
    LR = 5e-6
    WARMUP_RATIO = 0.1
    GRADIENT_ACCUM_STEPS = 4

    LOSS_WEIGHTS = {
        'ce': 0.4,
        'feature_vision': 0.2,
        'feature_text': 0.15,
        'feature_fusion': 0.15,
        'contrastive': 0.1
    }

    MAX_Q_LEN = 48
    MAX_OUTPUT_LEN = 96

    EARLY_STOP_PATIENCE = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

# =========================
# DATASET
# =========================
class XMLDistillationDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=48, max_output_len=96):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("teacher_answer") and data.get("teacher_reasoning"):
                        self.samples.append(data)
                except:
                    continue
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_output_len = max_output_len
        print(f"[INFO] Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def format_xml_output(self, answer, reasoning, reasoning_type):
        return f"<answer>{answer}</answer><reasoning>[{reasoning_type}] {reasoning}</reasoning>"

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s["question"]
        img_path = s["image_path"]

        answer = s.get("teacher_answer", "").strip()
        reasoning = s.get("teacher_reasoning", "").strip()
        reasoning_type = s.get("reasoning_type", "DESCRIPTIVE").upper()

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]

        q_enc = self.text_tokenizer(
            q, truncation=True, padding="max_length",
            max_length=self.max_q_len, return_tensors="pt"
        )

        xml_target = self.format_xml_output(answer, reasoning, reasoning_type)

        labels = self.decoder_tokenizer(
            xml_target, truncation=True, padding="max_length",
            max_length=self.max_output_len, return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "labels": labels,
            "image_path": img_path,
            "question": q,
            "xml_target": xml_target
        }

# =========================
# FUSION MODULE FIX
# =========================
class FusionModule(nn.Module):
    """Concatenate vision+text and project"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, v_feat, t_feat):
        x = torch.cat([v_feat, t_feat], dim=-1)
        return self.mlp(x)

# =========================
# DISTILLATION WRAPPER (TEACHER ON CPU)
# =========================
class DistillationWrapper(nn.Module):
    def __init__(self, student_model, teacher_model, cfg):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model  # Already on CPU
        self.cfg = cfg

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Dimensions
        student_vision_dim = 768
        teacher_vision_dim = 1536
        student_text_dim = 768
        teacher_text_dim = 1536
        student_fusion_dim = 768
        teacher_fusion_dim = 1536

        self.vision_projector = nn.Linear(student_vision_dim, teacher_vision_dim)
        self.text_projector = nn.Linear(student_text_dim, teacher_text_dim)
        self.fusion_projector = nn.Linear(student_fusion_dim, teacher_fusion_dim)

    def extract_teacher_features(self, pixel_values, input_ids, attention_mask):
        """
        Teacher trên CPU - chuyển inputs sang CPU, extract features, rồi chuyển về GPU
        """
        with torch.no_grad():
            try:
                # Chuyển inputs sang CPU với dtype phù hợp
                pixel_values_cpu = pixel_values.to("cpu", dtype=torch.float16)
                input_ids_cpu = input_ids.to("cpu")
                attention_mask_cpu = attention_mask.to("cpu")
                
                # Extract features trên CPU
                vision_outputs = self.teacher.visual(pixel_values_cpu)
                v_teacher = getattr(vision_outputs, 'pooler_output', 
                                   vision_outputs.last_hidden_state.mean(1))
                
                text_outputs = self.teacher.language_model.get_input_embeddings()(input_ids_cpu)
                t_teacher = text_outputs.mean(1)
                
                fusion_teacher = torch.cat([v_teacher, t_teacher], dim=-1)
                
                # Chuyển features về GPU để tính loss
                v_teacher = v_teacher.to(self.cfg.device, dtype=torch.float32)
                t_teacher = t_teacher.to(self.cfg.device, dtype=torch.float32)
                fusion_teacher = fusion_teacher.to(self.cfg.device, dtype=torch.float32)
                
                return v_teacher, t_teacher, fusion_teacher
            except Exception as e:
                print(f"[WARNING] Teacher extraction failed: {e}")
                return None, None, None

    def forward(self, pixel_values, input_ids, attention_mask, labels, **kwargs):
        # Student features (trên GPU)
        v_student = self.student.vision_encoder(pixel_values).last_hidden_state.mean(1)
        t_student = self.student.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state.mean(1)
        fusion_student = self.student.fusion(v_student, t_student)

        # CE loss từ student
        ce_loss, student_logits = self.student(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        losses = {'ce': ce_loss}

        # Teacher features (extract trên CPU)
        v_teacher, t_teacher, fusion_teacher = self.extract_teacher_features(
            pixel_values, input_ids, attention_mask
        )
        
        if v_teacher is None:
            return losses, student_logits

        # Feature distillation losses
        losses['feature_vision'] = F.mse_loss(
            self.vision_projector(v_student), v_teacher
        )
        losses['feature_text'] = F.mse_loss(
            self.text_projector(t_student), t_teacher
        )
        losses['feature_fusion'] = F.mse_loss(
            self.fusion_projector(fusion_student), fusion_teacher
        )

        # Contrastive loss
        fusion_student_norm = F.normalize(fusion_student, dim=-1)
        fusion_teacher_norm = F.normalize(fusion_teacher, dim=-1)
        logits_contrast = torch.matmul(fusion_student_norm, fusion_teacher_norm.T) / 0.5
        labels_contrast = torch.arange(pixel_values.size(0), device=pixel_values.device)
        losses['contrastive'] = F.cross_entropy(logits_contrast, labels_contrast)

        return losses, student_logits

    def compute_total_loss(self, losses):
        total = 0.0
        for key, loss in losses.items():
            if key in self.cfg.LOSS_WEIGHTS:
                total += self.cfg.LOSS_WEIGHTS[key] * loss
        return total

# =========================
# TRAINING LOOP
# =========================
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, cfg):
    model.train()
    model.teacher.eval()  # Teacher luôn ở eval mode
    total_losses = {k:0.0 for k in cfg.LOSS_WEIGHTS}
    total_sum = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
    optimizer.zero_grad()

    for step, batch in enumerate(progress):
        batch = {k:v.to(cfg.device) if torch.is_tensor(v) else v 
                 for k,v in batch.items()}

        with torch.amp.autocast(device_type="cuda", enabled=True):
            losses, _ = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            total_loss = model.compute_total_loss(losses) / cfg.GRADIENT_ACCUM_STEPS

        scaler.scale(total_loss).backward()

        if (step + 1) % cfg.GRADIENT_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        for k, v in losses.items():
            total_losses[k] += v.item()
        total_sum += total_loss.item() * cfg.GRADIENT_ACCUM_STEPS
        progress.set_postfix({'loss': total_loss.item()*cfg.GRADIENT_ACCUM_STEPS})

    num_batches = len(loader)
    avg_losses = {k:v/num_batches for k,v in total_losses.items()}
    avg_total = total_sum / num_batches
    return avg_total, avg_losses

# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Device: {cfg.device}")

    # ✅ STUDENT: Load lên GPU
    student = VQAGenModel(vision_model_name="Salesforce/blip-vqa-base")
    student.vision_encoder.gradient_checkpointing_enable()
    if os.path.exists(cfg.STUDENT_CKPT):
        student.load_state_dict(torch.load(cfg.STUDENT_CKPT, map_location='cpu'))
    student.fusion = FusionModule(input_dim=768*2, hidden_dim=768)
    student.to(cfg.device)
    print(f"[INFO] Student loaded on {cfg.device}")

    # ✅ TEACHER: OFFLOAD TO CPU HOÀN TOÀN
    print("[INFO] Loading teacher model on CPU (this may take a moment)...")
    teacher = AutoModelForVision2Seq.from_pretrained(
        cfg.TEACHER_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",  # ✅ CRITICAL: Offload to CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    teacher.eval()
    print("[INFO] Teacher loaded on CPU successfully")

    # Vision processor
    vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vision_processor.size = 224

    # Dataset
    dataset = XMLDistillationDataset(
        jsonl_path=cfg.TEACHER_DATA,
        vision_processor=vision_processor,
        text_tokenizer=student.text_tokenizer,
        decoder_tokenizer=student.decoder_tokenizer,
        max_q_len=cfg.MAX_Q_LEN,
        max_output_len=cfg.MAX_OUTPUT_LEN
    )

    loader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # Wrapper model (student on GPU, teacher on CPU)
    model = DistillationWrapper(student, teacher, cfg)
    model.student.to(cfg.device)  # Ensure student on GPU
    # Teacher stays on CPU (already set via device_map="cpu")
    
    # Move projectors to GPU
    model.vision_projector.to(cfg.device)
    model.text_projector.to(cfg.device)
    model.fusion_projector.to(cfg.device)
    
    print("[INFO] Model setup complete: Student on GPU, Teacher on CPU")

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.student.decoder.parameters(), 'lr': cfg.LR},
        {'params': model.student.fusion.parameters(), 'lr': cfg.LR*0.5},
        {'params': model.vision_projector.parameters(), 'lr': cfg.LR},
        {'params': model.text_projector.parameters(), 'lr': cfg.LR},
        {'params': model.fusion_projector.parameters(), 'lr': cfg.LR}
    ], weight_decay=0.01)

    num_steps = (len(loader)//cfg.GRADIENT_ACCUM_STEPS)*cfg.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(cfg.WARMUP_RATIO*num_steps),
        num_training_steps=num_steps
    )

    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    early_stop_counter = 0

    # Training loop
    for epoch in range(cfg.EPOCHS):
        torch.cuda.empty_cache()
        avg_total, avg_losses = train_epoch(
            model, loader, optimizer, scheduler, scaler, epoch, cfg
        )
        print(f"\n[INFO] Epoch {epoch+1} Total Loss: {avg_total:.4f}")
        for k,v in avg_losses.items():
            print(f"  {k}: {v:.4f}")

        # Save best model
        if avg_total < best_loss - 1e-4:
            best_loss = avg_total
            early_stop_counter = 0
            save_path = os.path.join(cfg.SAVE_DIR, f"{cfg.SAVE_NAME}_best.pt")
            torch.save(model.student.state_dict(), save_path)
            print(f"[INFO] Best model saved: {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    print("[INFO] Training completed!")

if __name__ == "__main__":
    main()
