"""
Script để tiếp tục training từ checkpoint
"""
import torch
import os
from train import *  # Import all variables from train.py

def resume_training(checkpoint_path):
    """Resume training from checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # === LOAD MODEL & OPTIMIZER ===
    model = VQAGenModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    
    print(f"✅ Resumed from epoch {start_epoch}, loss: {best_loss:.4f}")
    
    # === DATASET ===
    dataset = VQAGenDataset(
        questions_file="/kaggle/input/vivqa-dataset/train_questions.txt",
        answers_file="/kaggle/input/vivqa-dataset/train_answers.txt", 
        images_dir="/kaggle/input/vivqa-dataset/train_images"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # === CONTINUE TRAINING ===
    patience_counter = 0
    max_patience = 7
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for pixel_values, input_ids, attention_mask, labels in pbar:
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
            patience_counter = 0
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
            print(f"🛑 EARLY STOPPING after {epoch+1} epochs")
            break
    
    print("✅ Training completed!")

if __name__ == "__main__":
    # Example usage:
    # resume_training("saved_models/best_model.pth")
    # resume_training("saved_models/model_epoch37.pth")
    
    checkpoint_file = input("Enter checkpoint path: ")
    resume_training(checkpoint_file)
