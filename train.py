import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BlipImageProcessor
from tqdm import tqdm
import pandas as pd
from dataset import VQAGenDataset
from model import VQAGenModel

# === CONFIGURATION ===
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train'
CHECKPOINT_DIR = '/kaggle/input/checkpoint/transformers/default/1/checkpoints'
SAVE_DIR = '/kaggle/working/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 2e-4
VAL_SPLIT = 0.1  # 10% data for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_EPOCH = 0  # Set > 0 to resume

results = []

# === DATASET & DATALOADER ===
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip-vqa-base')
full_dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL & OPTIMIZER ===
model = VQAGenModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# === RESUME FROM CHECKPOINT ===
if RESUME_EPOCH > 0:
    model_path = os.path.join(CHECKPOINT_DIR, f'model_epoch{RESUME_EPOCH}.pth')
    optim_path = os.path.join(CHECKPOINT_DIR, f'optim_epoch{RESUME_EPOCH}.pth')
    print(f"[INFO] Resuming from epoch {RESUME_EPOCH}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    optimizer.load_state_dict(torch.load(optim_path, map_location=DEVICE))

# === TRAINING LOOP ===
best_val_loss = float("inf")

for epoch in range(RESUME_EPOCH, NUM_EPOCHS):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}]")

    for pixel_values, input_ids, attention_mask, labels in pbar:
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(train_loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)

    # --- VALIDATE ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for pixel_values, input_ids, attention_mask, labels in val_loader:
            pixel_values = pixel_values.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"[INFO] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- SAVE BEST MODEL ONLY ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(SAVE_DIR, 'best_optim.pth'))
        print(f"[INFO] New best model saved (Val Loss: {avg_val_loss:.4f})")

    # --- SAVE TOKENIZER ONCE ---
    if epoch == 0 and RESUME_EPOCH == 0:
        full_dataset.q_tokenizer.save_pretrained(os.path.join(SAVE_DIR, 'phobert_tokenizer'))
        full_dataset.a_tokenizer.save_pretrained(os.path.join(SAVE_DIR, 'vit5_tokenizer'))

print("[INFO] Training complete. Best model saved.")
