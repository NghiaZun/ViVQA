"""
Kaggle Training Script với logging và monitoring
Chạy trong Kaggle Notebook
"""

# === IMPORTS ===
import torch
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime

# Import các class từ files khác
from model import VQAGenModel
from dataset import VQAGenDataset

# === KAGGLE CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {DEVICE}")

# Kaggle paths
DATASET_DIR = "/kaggle/input/vivqa-dataset"
OUTPUT_DIR = "/kaggle/working"
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4
SAVE_DIR = "/kaggle/working/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

class KaggleLogger:
    """Custom logger cho Kaggle"""
    def __init__(self, log_file):
        self.log_file = log_file
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

def kaggle_train():
    """Main training function for Kaggle"""
    logger = KaggleLogger(LOG_FILE)
    logger.log("🚀 Starting Kaggle ViVQA Training")
    
    # === DATASET ===
    logger.log("📂 Loading dataset...")
    dataset = VQAGenDataset(
        questions_file=f"{DATASET_DIR}/train_questions.txt",
        answers_file=f"{DATASET_DIR}/train_answers.txt",
        images_dir=f"{DATASET_DIR}/train_images"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.log(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # === MODEL & OPTIMIZER ===
    logger.log("🤖 Initializing model...")
    model = VQAGenModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # Tracking variables
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 7
    
    logger.log("✅ Model initialized")
    
    # === TRAINING LOOP ===
    logger.log("🎯 Starting training loop...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        optimizer.zero_grad()
        
        for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
            # Move to device
            pixel_values = pixel_values.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
            
            # Gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Update weights
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # === CHECKPOINT SAVING ===
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(SAVE_DIR, f'model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.log(f"💾 Checkpoint saved: epoch {epoch+1}")
        
        # === BEST MODEL TRACKING ===
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_path = os.path.join(SAVE_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_path)
            logger.log(f"🏆 NEW BEST MODEL! Loss: {avg_loss:.4f}")
        else:
            patience_counter += 1
            logger.log(f"⚠️  No improvement for {patience_counter}/{max_patience} epochs")
        
        # === LEARNING RATE SCHEDULER ===
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        logger.log(f"⏰ Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # === EARLY STOPPING ===
        if patience_counter >= max_patience:
            logger.log(f"🛑 EARLY STOPPING after {epoch+1} epochs")
            break
        
        # === KAGGLE MONITORING ===
        if epoch % 10 == 0:  # Plot every 10 epochs
            plot_kaggle_progress(LOG_FILE, epoch+1)
    
    logger.log("✅ Training completed!")
    
    # === FINAL SAVE ===
    final_path = os.path.join(SAVE_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizers': {
            'question': dataset.q_tokenizer,
            'answer': dataset.a_tokenizer
        }
    }, final_path)
    logger.log(f"💾 Final model saved: {final_path}")

def plot_kaggle_progress(log_file, current_epoch):
    """Plot training progress in Kaggle"""
    import re
    
    epochs = []
    losses = []
    best_losses = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'Epoch (\d+)/\d+ \| Loss: ([\d\.]+) \| Best: ([\d\.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    best_loss = float(match.group(3))
                    
                    epochs.append(epoch)
                    losses.append(loss)
                    best_losses.append(best_loss)
    
    if len(epochs) > 1:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, 'b-', alpha=0.7, label='Training Loss')
        plt.plot(epochs, best_losses, 'r-', linewidth=2, label='Best Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress (Epoch {current_epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(losses) > 1:
            improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
            plt.text(0.1, 0.8, f"Current Epoch: {current_epoch}", fontsize=12)
            plt.text(0.1, 0.6, f"Best Loss: {min(best_losses):.4f}", fontsize=12)
            plt.text(0.1, 0.4, f"Improvement: {improvement:.1f}%", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Statistics')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/training_progress_epoch_{current_epoch}.png', dpi=150)
        plt.show()

def kaggle_resume_training(checkpoint_path):
    """Resume training in Kaggle"""
    logger = KaggleLogger(LOG_FILE)
    
    if not os.path.exists(checkpoint_path):
        logger.log(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    logger.log(f"📂 Resuming from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model
    model = VQAGenModel().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    
    logger.log(f"✅ Resumed from epoch {start_epoch}, loss: {best_loss:.4f}")
    
    # Continue training...
    # (Implementation similar to kaggle_train but starting from start_epoch)

# === KAGGLE COMMANDS ===
def run_kaggle_commands():
    """Commands to run in Kaggle cells"""
    
    print("📋 KAGGLE TRAINING COMMANDS:")
    print("="*50)
    
    print("\n1️⃣ Start Training:")
    print("kaggle_train()")
    
    print("\n2️⃣ Monitor Progress:")
    print("plot_kaggle_progress('/kaggle/working/training.log', current_epoch)")
    
    print("\n3️⃣ Resume Training:")
    print("kaggle_resume_training('/kaggle/working/checkpoints/best_model.pth')")
    
    print("\n4️⃣ Check Saved Files:")
    print("!ls -la /kaggle/working/checkpoints/")
    
    print("\n5️⃣ View Training Log:")
    print("!tail -20 /kaggle/working/training.log")
    
    print("\n6️⃣ Download Results:")
    print("# Files will be in /kaggle/working/ - auto-downloadable")

if __name__ == "__main__":
    run_kaggle_commands()
