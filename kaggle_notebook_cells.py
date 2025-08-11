# =============================================
# KAGGLE NOTEBOOK CELLS
# Copy từng cell này vào Kaggle Notebook
# =============================================

# CELL 1: Setup và Import
"""
!pip install transformers accelerate

import torch
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Upload files: model.py, dataset.py, kaggle_train.py
print(f"🔥 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"📂 Working directory: {os.getcwd()}")
"""

# CELL 2: Import Custom Modules  
"""
# Import custom classes
from model import VQAGenModel
from dataset import VQAGenDataset  
from kaggle_train import kaggle_train, plot_kaggle_progress, KaggleLogger

print("✅ All modules imported successfully")
"""

# CELL 3: Start Training
"""
# Start training với full monitoring
kaggle_train()
"""

# CELL 4: Monitor Progress (chạy song song)
"""
# Chạy cell này trong tab khác để monitor
import time

while True:
    try:
        plot_kaggle_progress('/kaggle/working/training.log', 0)
        print("📊 Progress updated...")
        time.sleep(300)  # Update every 5 minutes
    except:
        print("⏸️ Monitoring paused...")
        break
"""

# CELL 5: Check Training Status
"""
# Kiểm tra training status
!tail -20 /kaggle/working/training.log
print("\n" + "="*50)
!ls -la /kaggle/working/checkpoints/
"""

# CELL 6: Resume Training (nếu cần)
"""
from kaggle_train import kaggle_resume_training

# Resume từ checkpoint cụ thể
kaggle_resume_training('/kaggle/working/checkpoints/best_model.pth')
"""

# CELL 7: Quick Evaluation
"""
# Test model đã train
import torch
from model import VQAGenModel

model = VQAGenModel()
checkpoint = torch.load('/kaggle/working/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Best model loaded. Loss: {checkpoint['loss']:.4f}")
"""

# CELL 8: Download Results  
"""
# Download training results
import zipfile
import os

# Tạo zip file chứa kết quả
with zipfile.ZipFile('/kaggle/working/training_results.zip', 'w') as zipf:
    # Add checkpoints
    for root, dirs, files in os.walk('/kaggle/working/checkpoints'):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, '/kaggle/working'))
    
    # Add logs and plots
    if os.path.exists('/kaggle/working/training.log'):
        zipf.write('/kaggle/working/training.log', 'training.log')
    
    for file in os.listdir('/kaggle/working'):
        if file.endswith('.png'):
            zipf.write(f'/kaggle/working/{file}', file)

print("📦 Results packaged in: /kaggle/working/training_results.zip")
print("💾 Download this file to save your training results!")
"""
