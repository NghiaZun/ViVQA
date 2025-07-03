import os
import torch
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from tqdm import tqdm
from dataset import VQAGenDataset
from model import VQAGenModel

# --- Config ---
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train'
CHECKPOINT_DIR = '/kaggle/input/checkpoint/transformers/default/1/checkpoints'
SAVE_DIR = '/kaggle/working/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 2e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESUME_EPOCH = 0

# --- Dataset ---
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip-vqa-base')
dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = VQAGenModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Resume checkpoint if needed
if RESUME_EPOCH > 0:
    model_path = os.path.join(CHECKPOINT_DIR, f'model_epoch{RESUME_EPOCH}.pth')
    optim_path = os.path.join(CHECKPOINT_DIR, f'optim_epoch{RESUME_EPOCH}.pth')
    print(f"Resuming from epoch {RESUME_EPOCH}...")
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optim_path))

# --- Training ---
model.train()
for epoch in range(RESUME_EPOCH, NUM_EPOCHS):
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for pixel_values, input_ids, attention_mask, labels in pbar:
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    if epoch == NUM_EPOCHS - 1:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pth'))

# Final model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vqagen_final.pth'))