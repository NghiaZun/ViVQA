# KAGGLE CHECKPOINT & RESUME GUIDE

## 🎯 Giải Pháp Checkpoint Tự Động

### 1️⃣ **Auto-Checkpoint Mỗi Epoch**
Script đã được cập nhật để lưu checkpoint SAU MỖI EPOCH:
- File: `latest_checkpoint.pt`
- Chứa: model, optimizer, scheduler, epoch, best_loss
- Tự động ghi đè → luôn có checkpoint mới nhất

### 2️⃣ **Backup Checkpoints**
Checkpoint định kỳ (không bị ghi đè):
- **Ultimate**: Mỗi 10 epochs + epoch 15, 30
- **Fast**: Mỗi 5 epochs
- File: `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`...

### 3️⃣ **Best Model**
Lưu model tốt nhất (val_loss thấp nhất):
- File: `vqa_student_best_ultimate.pt` (hoặc `_fast.pt`)
- Chỉ chứa model weights (nhẹ hơn)

---

## 📦 Cách Sử Dụng

### **Khi Session Hết Giờ (12h):**

1. **Download 3 files quan trọng:**
   ```python
   # Trong Kaggle notebook
   from IPython.display import FileLink
   FileLink('/kaggle/working/latest_checkpoint.pt')  # QUAN TRỌNG NHẤT
   FileLink('/kaggle/working/vqa_student_best_ultimate.pt')
   FileLink('/kaggle/working/train_val_log_ultimate.csv')
   ```

2. **Upload checkpoint lên Kaggle Dataset:**
   - Vào Kaggle.com → Datasets → New Dataset
   - Upload `latest_checkpoint.pt`
   - Tên dataset: `vivqa-checkpoint-epoch20` (ví dụ)
   - Make public hoặc private

3. **Resume trong session mới:**
   ```python
   # Ở đầu train_student_ultimate.py, sửa dòng:
   RESUME_FROM = "/kaggle/input/vivqa-checkpoint-epoch20/latest_checkpoint.pt"
   
   # Rồi chạy lại:
   !python train_student_ultimate.py
   ```

---

## ⚡ Script Nhanh Hơn

### **train_student_fast.py** (MỚI)
Tối ưu cho thời gian training ngắn hơn:

| Tính năng | Ultimate (Gốc) | Fast (Mới) |
|-----------|---------------|-----------|
| Epochs | 100 | 30 |
| Batch size | 3 | 4 |
| Max length | 160 | 128 |
| Early stop | 15 | 8 |
| Stage 1 | 15 epochs | 5 epochs |
| Stage 2 | 30 epochs | 12 epochs |
| **Thời gian ước tính** | **~58 giờ** | **~12-15 giờ** |

### **Cách dùng:**
```python
!python train_student_fast.py
```

**Lưu ý:** Fast mode phù hợp cho:
- ✅ Testing nhanh
- ✅ Deadline gấp
- ✅ Chỉ cần model khả dụng

Ultimate mode cho kết quả tốt nhất (thesis).

---

## 🔄 Resume Workflow

### **Workflow tự động:**
```
Epoch 1-10 → Session 1 (12h)
  ↓ Download latest_checkpoint.pt
  ↓ Upload to Kaggle Dataset
Epoch 11-20 → Session 2 (12h) [RESUME_FROM = checkpoint]
  ↓ Download latest_checkpoint.pt
  ↓ Upload to new Dataset
Epoch 21-30 → Session 3 (12h) [RESUME_FROM = checkpoint]
  ↓ ...
```

### **Script tự động tạo checkpoint:**
```python
# SAU MỖI EPOCH tự động lưu:
torch.save({
    'epoch': epoch,                          # Epoch số bao nhiêu
    'model_state_dict': model.state_dict(),  # Trọng số model
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'scheduler_state_dict': scheduler.state_dict(),  # LR scheduler
    'best_val_loss': best_val_loss,          # Val loss tốt nhất
    'early_stop_counter': early_stop_counter # Đếm early stopping
}, '/kaggle/working/latest_checkpoint.pt')
```

### **Script tự động resume:**
```python
if RESUME_FROM and os.path.exists(RESUME_FROM):
    checkpoint = torch.load(RESUME_FROM)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Tiếp tục từ epoch tiếp theo
    best_val_loss = checkpoint['best_val_loss']
    print(f"✅ Resuming from epoch {start_epoch}")
```

---

## 💡 Tips Tối Ưu Training

### 1. **Giảm epochs nếu gấp:**
```python
# Trong train_student_ultimate.py
EPOCHS = 50  # Thay vì 100
```

### 2. **Tăng batch size nếu GPU còn trống:**
```python
BATCH_SIZE = 4  # Thay vì 3 (test trước!)
```

### 3. **Tắt validation thường xuyên:**
```python
# Validate mỗi 2 epochs thay vì mỗi epoch
if epoch % 2 == 0:
    val_loss = validate_epoch(...)
```

### 4. **Reduce logging overhead:**
```python
# Tắt ROUGE scoring trong validation (tốn thời gian)
# Chỉ dùng loss để track
```

### 5. **Sử dụng mixed precision tốt:**
```python
# Đã có sẵn trong script
with torch.cuda.amp.autocast():
    loss = compute_loss(...)
```

---

## 📊 Monitor Training

### **Trong Kaggle notebook:**
```python
# Cell riêng để monitor
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/working/train_val_log_ultimate.csv')
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.legend()
plt.title('Loss')

plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['lr'])
plt.title('Learning Rate')

plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['w_format'], label='Format')
plt.plot(df['epoch'], df['w_answer'], label='Answer')
plt.plot(df['epoch'], df['w_reason'], label='Reason')
plt.legend()
plt.title('Curriculum Weights')

plt.tight_layout()
plt.show()
```

---

## ⏱️ Ước Tính Thời Gian

### **T4 GPU (16GB):**
- **1 epoch Ultimate**: ~35 phút (batch=3)
- **1 epoch Fast**: ~25 phút (batch=4)
- **100 epochs Ultimate**: ~58 giờ → 5 sessions
- **30 epochs Fast**: ~12.5 giờ → 2 sessions

### **Chiến lược:**
1. **Session 1**: Chạy fast mode (30 epochs) → có model khả dụng
2. **Session 2-5**: Chạy ultimate mode (100 epochs) → model tốt nhất

---

## 🚨 Xử Lý Sự Cố

### **OOM Error:**
```python
BATCH_SIZE = 2  # Giảm xuống
accum_steps = 4  # Tăng lên để giữ effective batch = 8
```

### **Session timeout sắp đến:**
```python
# Trong cell riêng, check thời gian
import time
start_time = time.time()

# Sau khi training
elapsed = (time.time() - start_time) / 3600
print(f"Elapsed: {elapsed:.1f} hours")
if elapsed > 11.5:  # Gần 12h
    print("⚠️  SESSION SẮP HẾT! DOWNLOAD CHECKPOINT NGAY!")
```

### **Checkpoint bị lỗi:**
```python
# Kiểm tra checkpoint trước khi resume
checkpoint = torch.load('/kaggle/input/my-checkpoint/latest_checkpoint.pt')
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best val loss: {checkpoint['best_val_loss']}")
print(f"Keys: {checkpoint.keys()}")
```

---

## ✅ Checklist Trước Khi Training

- [ ] Bật GPU trong Settings (T4 x2 hoặc P100)
- [ ] Verify paths với `verify_kaggle_paths.py`
- [ ] Chọn script: `train_student_ultimate.py` (tốt nhất) hoặc `train_student_fast.py` (nhanh)
- [ ] Set `RESUME_FROM = None` cho lần đầu
- [ ] Monitor GPU memory: `!nvidia-smi`
- [ ] Chuẩn bị download checkpoint sau 11 giờ

---

## 📁 Files Cần Download Sau Training

**Bắt buộc:**
1. `latest_checkpoint.pt` - Resume training
2. `vqa_student_best_ultimate.pt` - Model tốt nhất
3. `train_val_log_ultimate.csv` - Training history

**Tùy chọn:**
4. `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`... - Backup
5. `vqa_student_final_ultimate.pt` - Model cuối cùng

**Dung lượng:**
- Checkpoint đầy đủ: ~2.5GB
- Model weights only: ~1.7GB
- Log CSV: <1MB
