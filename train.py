import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from tqdm import tqdm
from dataset_vqa_gen import VQAGenDataset
from vqa_vietnamese_gen import VQAGenModel

# --- Config ---
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train'
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 2e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Dataset ---
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = VQAGenModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --- Training ---
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for vision_feats, input_ids, attention_mask, labels in tqdm(dataloader):
        vision_feats = vision_feats.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        loss, _ = model(vision_feats, input_ids, attention_mask, labels=labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), 'vqagen_model.pth')