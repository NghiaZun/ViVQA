import os
import torch
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from tqdm import tqdm
from dataset import VQAGenDataset
from model import VQAGenModel
import pandas as pd

# === CONFIGURATION ===
CSV_PATH = '/kaggle/input/csv1000/mini_train.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train'
CHECKPOINT_DIR = '/kaggle/input/checkpoint/transformers/default/1/checkpoints'
SAVE_DIR = '/kaggle/working/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

results = []
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_EPOCH = 0  # Set > 0 to resume

# === DATASET & DATALOADER ===
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip-vqa-base')
dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
for epoch in range(RESUME_EPOCH, NUM_EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}]")

    for pixel_values, input_ids, attention_mask, labels in pbar:
        # Move tensors to device
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        model.zero_grad()
        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"[INFO] Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

# === FINAL SAVE ===
model.eval()
for idx in range(len(dataset)):
    with torch.no_grad():
        pixel_values, input_ids, attention_mask, labels = dataset[idx]
        pixel_values = pixel_values.unsqueeze(0).to(DEVICE)
        input_ids = input_ids.unsqueeze(0).to(DEVICE)
        attention_mask = attention_mask.unsqueeze(0).to(DEVICE)

        pred_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        decoded = dataset.a_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        question = dataset.q_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = dataset.a_tokenizer.decode(labels[labels != -100], skip_special_tokens=True)

        results.append({
            "question": question,
            "ground_truth": answer,
            "prediction": decoded
        })
df = pd.DataFrame(results)
df.to_csv("train_predictions.csv", index=False)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vqagen_final.pth'))
dataset.q_tokenizer.save_pretrained(os.path.join(SAVE_DIR, 'phobert_tokenizer'))
dataset.a_tokenizer.save_pretrained(os.path.join(SAVE_DIR, 'vit5_tokenizer'))

print("[INFO] Training complete. Final model and tokenizers saved.")
