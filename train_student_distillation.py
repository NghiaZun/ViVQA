"""
Full Knowledge Distillation Training for VQA Student (XML Format Output)
Student generates: <answer>...</answer><reasoning>[TYPE] ...</reasoning>
Author: Research-grade implementation
"""

import os
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BlipProcessor, 
    get_cosine_schedule_with_warmup,
    AutoProcessor,
    AutoModelForVision2Seq
)

from model import VQAGenModel

# =========================
# CONFIG
# =========================
class Config:
    # Paths
    TEACHER_DATA = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"
    IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    STUDENT_CKPT = "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/best_model.pth"
    TEACHER_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Output
    SAVE_DIR = "/kaggle/working"
    SAVE_NAME = "student_distilled_xml"
    
    # Training
    EPOCHS = 15
    BATCH_SIZE = 2
    LR = 5e-6
    WARMUP_RATIO = 0.1
    GRADIENT_ACCUM_STEPS = 4
    
    # Distillation loss weights
    LOSS_WEIGHTS = {
        'ce': 0.4,                # Cross-entropy (XML sequence)
        'feature_vision': 0.2,    # Vision feature matching
        'feature_text': 0.15,     # Text feature matching
        'feature_fusion': 0.15,   # Fusion feature matching
        'contrastive': 0.1        # Contrastive distillation
    }
    
    # Hyperparameters
    MAX_Q_LEN = 48
    MAX_OUTPUT_LEN = 96  # Longer for XML format
    
    # Early stopping
    EARLY_STOP_PATIENCE = 4
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

# =========================
# DATASET
# =========================
class XMLDistillationDataset(Dataset):
    """Dataset that prepares XML-formatted targets"""
    
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, 
                 decoder_tokenizer, max_q_len=48, max_output_len=96):
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
# DISTILLATION MODEL
# =========================
class DistillationWrapper(nn.Module):
    """Wrapper for multi-stage distillation"""
    
    def __init__(self, student_model, teacher_model, cfg):
        super().__init__()
        self.student = student_model.to(cfg.device)  # Move student only
        self.teacher = teacher_model  # Teacher stays on meta / device_map
        self.cfg = cfg
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        
        # Projection layers
        self.vision_projector = nn.Linear(768, 1536)
        self.text_projector = nn.Linear(768, 1536)
        self.fusion_projector = nn.Linear(768, 1536)
        
    def extract_teacher_features(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            try:
                vision_outputs = self.teacher.visual(pixel_values)
                v_teacher = getattr(vision_outputs, "pooler_output", vision_outputs.last_hidden_state.mean(1))
                
                text_outputs = self.teacher.language_model.get_input_embeddings()(input_ids)
                t_teacher = text_outputs.mean(dim=1)
                
                fusion_teacher = torch.cat([v_teacher, t_teacher], dim=-1)
                if fusion_teacher.size(-1) != v_teacher.size(-1):
                    fusion_teacher = F.adaptive_avg_pool1d(fusion_teacher.unsqueeze(1), v_teacher.size(-1)).squeeze(1)
                
                return v_teacher, t_teacher, fusion_teacher
            except Exception as e:
                print(f"[WARNING] Teacher feature extraction failed: {e}")
                return None, None, None
    
    def forward(self, pixel_values, input_ids, attention_mask, labels, 
                image_path=None, question=None, xml_target=None):
        batch_size = pixel_values.size(0)
        vision_output = self.student.vision_encoder(pixel_values)
        v_student = vision_output.last_hidden_state.mean(dim=1)
        
        text_output = self.student.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_student = text_output.last_hidden_state.mean(dim=1)
        
        fusion_student = self.student.fusion(v_student, t_student)
        
        ce_loss, student_logits = self.student(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        losses = {'ce': ce_loss}
        
        v_teacher, t_teacher, fusion_teacher = self.extract_teacher_features(pixel_values, input_ids, attention_mask)
        if v_teacher is None:
            return losses, student_logits
        
        v_student_proj = self.vision_projector(v_student)
        losses['feature_vision'] = F.mse_loss(v_student_proj, v_teacher)
        
        t_student_proj = self.text_projector(t_student)
        losses['feature_text'] = F.mse_loss(t_student_proj, t_teacher)
        
        fusion_student_proj = self.fusion_projector(fusion_student)
        losses['feature_fusion'] = F.mse_loss(fusion_student_proj, fusion_teacher)
        
        fusion_student_norm = F.normalize(fusion_student, dim=-1)
        fusion_teacher_norm = F.normalize(fusion_teacher, dim=-1)
        logits_contrast = torch.matmul(fusion_student_norm, fusion_teacher_norm.T) / 0.5
        labels_contrast = torch.arange(batch_size, device=pixel_values.device)
        losses['contrastive'] = F.cross_entropy(logits_contrast, labels_contrast)
        
        return losses, student_logits
    
    def compute_total_loss(self, losses):
        total = 0.0
        for key, loss in losses.items():
            if key in self.cfg.LOSS_WEIGHTS:
                total += self.cfg.LOSS_WEIGHTS[key] * loss
        return total

# =========================
# GENERATION UTILITIES
# =========================
def parse_xml_output(text):
    answer, reasoning, reasoning_type = "", "", ""
    m1 = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m1:
        answer = m1.group(1).strip()
    m2 = re.search(r"<reasoning>\s*\[(\w+)\]\s*(.*?)</reasoning>", text, re.S)
    if m2:
        reasoning_type = m2.group(1).upper()
        reasoning = m2.group(2).strip()
    return answer, reasoning, reasoning_type

@torch.no_grad()
def generate_sample(model, sample, device):
    model.eval()
    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
    
    vision_output = model.student.vision_encoder(pixel_values)
    v_pooled = vision_output.last_hidden_state.mean(dim=1)
    
    text_output = model.student.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    t_pooled = text_output.last_hidden_state.mean(dim=1)
    
    fusion = model.student.fusion(v_pooled, t_pooled)
    generated_ids = model.student.decoder.generate(
        inputs_embeds=fusion.unsqueeze(1),
        max_length=cfg.MAX_OUTPUT_LEN,
        num_beams=4,
        early_stopping=True,
        pad_token_id=model.student.decoder_tokenizer.pad_token_id,
        eos_token_id=model.student.decoder_tokenizer.eos_token_id
    )
    return model.student.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# =========================
# TRAIN / VALIDATION
# =========================
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, cfg):
    model.train()
    model.teacher.eval()
    
    total_losses = {key: 0.0 for key in cfg.LOSS_WEIGHTS.keys()}
    total_loss_sum = 0.0
    
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress):
        batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=True):
            losses, logits = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                xml_target=batch.get('xml_target')
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
        
        for key, val in losses.items():
            total_losses[key] += val.item()
        total_loss_sum += total_loss.item() * cfg.GRADIENT_ACCUM_STEPS
        
        progress.set_postfix({
            'loss': f"{total_loss.item() * cfg.GRADIENT_ACCUM_STEPS:.4f}",
            'ce': f"{losses['ce'].item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    num_batches = len(loader)
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    avg_total = total_loss_sum / num_batches
    return avg_total, avg_losses

@torch.no_grad()
def validate(model, loader, cfg, num_samples=5):
    model.eval()
    print("\n" + "="*60)
    print("VALIDATION SAMPLES")
    print("="*60)
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        sample = {k: v[0] for k, v in batch.items() if torch.is_tensor(v)}
        generated_text = generate_sample(model, sample, cfg.device)
        answer, reasoning, rtype = parse_xml_output(generated_text)
        print(f"\nSample {i+1}:")
        print(f"Question: {batch['question'][0]}")
        print(f"Target: {batch['xml_target'][0]}")
        print(f"Generated: {generated_text}")
        print(f"  - Answer: {answer}")
        print(f"  - Reasoning: [{rtype}] {reasoning}")
    print("="*60 + "\n")

# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] Loss weights: {cfg.LOSS_WEIGHTS}")
    
    # Student
    student = VQAGenModel(vision_model_name="Salesforce/blip-vqa-base")
    if os.path.exists(cfg.STUDENT_CKPT):
        print(f"[INFO] Loading pretrained: {cfg.STUDENT_CKPT}")
        student.load_state_dict(torch.load(cfg.STUDENT_CKPT, map_location='cpu'))
    
    # Teacher
    print("[INFO] Loading teacher...")
    teacher = AutoModelForVision2Seq.from_pretrained(
        cfg.TEACHER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",   # important: keeps teacher distributed / meta-safe
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    teacher.eval()
    
    # Dataset
    vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    dataset = XMLDistillationDataset(
        jsonl_path=cfg.TEACHER_DATA,
        vision_processor=vision_processor,
        text_tokenizer=student.text_tokenizer,
        decoder_tokenizer=student.decoder_tokenizer,
        max_q_len=cfg.MAX_Q_LEN,
        max_output_len=cfg.MAX_OUTPUT_LEN
    )
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Wrap
    model = DistillationWrapper(student, teacher, cfg)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.student.decoder.parameters(), 'lr': cfg.LR},
        {'params': model.student.fusion.parameters(), 'lr': cfg.LR * 0.5},
        {'params': model.vision_projector.parameters(), 'lr': cfg.LR},
        {'params': model.text_projector.parameters(), 'lr': cfg.LR},
        {'params': model.fusion_projector.parameters(), 'lr': cfg.LR}
    ], weight_decay=0.01)
    
    num_steps = (len(loader) // cfg.GRADIENT_ACCUM_STEPS) * cfg.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg.WARMUP_RATIO * num_steps), num_training_steps=num_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    early_stop_counter = 0
    log_path = os.path.join(cfg.SAVE_DIR, f"{cfg.SAVE_NAME}_log.csv")
    log_df = pd.DataFrame(columns=['epoch', 'total_loss'] + list(cfg.LOSS_WEIGHTS.keys()))
    log_df.to_csv(log_path, index=False)
    
    print(f"[INFO] Training for {cfg.EPOCHS} epochs...")
    for epoch in range(cfg.EPOCHS):
        avg_total, avg_losses = train_epoch(model, loader, optimizer, scheduler, scaler, epoch, cfg)
        print(f"\n[INFO] Epoch {epoch+1} Summary: Total: {avg_total:.4f}")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")
        log_data = {'epoch': epoch+1, 'total_loss': avg_total}
        log_data.update(avg_losses)
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([log_data])], ignore_index=True)
        log_df.to_csv(log_path, index=False)
        
        if (epoch + 1) % 2 == 0:
            validate(model, val_loader, cfg)
        
        if avg_total < best_loss - 1e-4:
            best_loss = avg_total
            early_stop_counter = 0
            best_path = os.path.join(cfg.SAVE_DIR, f"{cfg.SAVE_NAME}_best.pt")
            torch.save(model.student.state_dict(), best_path)
            print(f"[INFO] ✅ Best model saved! Loss: {best_loss:.4f}")
        else:
            early_stop_counter += 1
        if early_stop_counter >= cfg.EARLY_STOP_PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break
    
    final_path = os.path.join(cfg.SAVE_DIR, f"{cfg.SAVE_NAME}_final.pt")
    torch.save(model.student.state_dict(), final_path)
    print(f"[INFO] ✅ Training done! Best: {best_loss:.4f}")

if __name__ == "__main__":
    main()
