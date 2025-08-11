import os
import torch
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from tqdm import tqdm
from dataset import VQAGenDataset
from model import VQAGenModel
import pandas as pd

# === CONFIGURATION ===
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train'
CHECKPOINT_DIR = '/kaggle/input/checkpoint/transformers/default/1/checkpoints'
SAVE_DIR = '/kaggle/working/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

results = []
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME_EPOCH = 0  # Set > 0 to resume

# === DATASET & DATALOADER ===
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip-vqa-base')
dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL & OPTIMIZER ===
model = VQAGenModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# === LEARNING RATE SCHEDULER ===
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True
)

# === BEST LOSS TRACKING ===
best_loss = float('inf')
patience_counter = 0
max_patience = 7  # Stop after 7 epochs without improvement

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
    
    optimizer.zero_grad()  # Zero gradients at start of epoch

    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        # Move tensors to device
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
        
        # Scale loss by accumulation steps
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Rescale for logging
        
        # Update weights every GRADIENT_ACCUMULATION_STEPS batches
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)

    avg_loss = total_loss / len(dataloader)
    
    # === CHECKPOINT SAVING ===
    if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
        checkpoint_path = os.path.join(SAVE_DIR, f'model_epoch{epoch+1}.pth')
        optim_path = os.path.join(SAVE_DIR, f'optim_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(optimizer.state_dict(), optim_path)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")
    
    # === BEST MODEL TRACKING ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0  # Reset patience
        best_path = os.path.join(SAVE_DIR, 'best_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, best_path)
        print(f"🏆 NEW BEST MODEL! Loss: {avg_loss:.4f}")
    else:
        patience_counter += 1
        print(f"⚠️  No improvement for {patience_counter}/{max_patience} epochs")
    
    # === LEARNING RATE SCHEDULER ===
    scheduler.step(avg_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"⏰ Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f}")
    
    # === EARLY STOPPING ===
    if patience_counter >= max_patience:
        print(f"🛑 EARLY STOPPING after {epoch+1} epochs (no improvement for {max_patience} epochs)")
        break

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
