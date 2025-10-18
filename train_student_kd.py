"""
Day 3 – Student Knowledge Distillation Training
Author: Han
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor
from model import VQAGenModel

# =====================
# CONFIG
# =====================
DATA_PATH = "/kaggle/input/teacher/Qwen2-VL-7B-Instruct.jsonl"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
SAVE_PATH = "/kaggle/working/vqa_student_kd.pt"
EPOCHS = 2
LR = 2e-5
ALPHA = 0.8
BATCH_SIZE = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# DATASET
# =====================
class DistillDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer, max_q_len=64, max_a_len=64):
        self.samples = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")]
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s["input"].split("Câu hỏi:")[-1].split("Ảnh:")[0].strip()
        img_path = s["input"].split("Ảnh:")[-1].strip()
        answer = s["output"]["answer"]
        reasoning = s["output"]["reasoning"]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values[0]
        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length", max_length=self.max_q_len, return_tensors="pt")
        label_text = f"{answer}. Giải thích: {reasoning}"
        labels = self.decoder_tokenizer(label_text, truncation=True, padding="max_length", max_length=self.max_a_len, return_tensors="pt").input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "labels": labels
        }

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading VQAGenModel...")
model = VQAGenModel(
    phobert_dir = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/phobert_tokenizer",
    vit5_dir = "/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/vit5_tokenizer"
)
model.load_state_dict(torch.load("/kaggle/input/checkpoints-data/tensorflow2/default/1/checkpoints/best_model.pth"))
model.to(device)

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
dataset = DistillDataset(
    jsonl_path=DATA_PATH,
    vision_processor=vision_processor,
    text_tokenizer=model.text_tokenizer,
    decoder_tokenizer=model.decoder_tokenizer
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# TRAIN LOOP
# =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        ce_loss, logits = model(**batch)
        loss = ALPHA * ce_loss  # KD loss nếu chưa có teacher logits
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[INFO] Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"[INFO] ✅ Saved fine-tuned student model → {SAVE_PATH}")


