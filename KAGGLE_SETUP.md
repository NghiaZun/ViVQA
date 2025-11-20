# 📦 Kaggle Environment Setup

## ✅ Kaggle đã có sẵn hầu hết packages

Kaggle notebooks đã pre-install hầu hết thư viện phổ biến. Bạn **KHÔNG CẦN cài đặt** các packages sau:

### Đã có sẵn trên Kaggle:
- ✅ `torch` (PyTorch 2.1+)
- ✅ `torchvision`
- ✅ `transformers` (HuggingFace)
- ✅ `pandas`
- ✅ `numpy`
- ✅ `Pillow` (PIL)
- ✅ `tqdm`
- ✅ `scikit-learn`
- ✅ `matplotlib`
- ✅ `seaborn`

## 🔧 CẦN CÀI THÊM

Chỉ cần cài **1-2 packages** này:

### Option 1: Cài từng package (Nhanh nhất)
```python
# Chỉ cài những cái thiếu
!pip install -q rouge-score sentencepiece
```

### Option 2: Cài từ requirements.txt
```python
!pip install -q -r requirements.txt
```

## 🚀 Setup Script cho Kaggle

Copy đoạn này vào cell đầu tiên của Kaggle notebook:

```python
# ==================== KAGGLE SETUP ====================

# 1. Verify PyTorch & CUDA
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# 2. Install missing packages
print("\n📦 Installing required packages...")
!pip install -q rouge-score sentencepiece

# 3. Verify installations
try:
    from rouge_score import rouge_scorer
    print("✅ rouge-score installed")
except:
    print("❌ rouge-score missing")

try:
    import sentencepiece
    print("✅ sentencepiece installed")
except:
    print("❌ sentencepiece missing")

try:
    from transformers import AutoTokenizer, AutoModel
    print("✅ transformers working")
except:
    print("❌ transformers missing")

print("\n✅ Setup complete!")

# ======================================================
```

## 📋 Complete Installation Commands

Nếu muốn cài đầy đủ (không bắt buộc):

```bash
# Minimal - Chỉ cài thiếu (KHUYẾN NGHỊ)
!pip install -q rouge-score sentencepiece

# Full - Cài tất cả từ requirements.txt
!pip install -q -r requirements.txt

# Specific versions (nếu cần)
!pip install -q transformers==4.35.0 torch==2.1.0
```

## ⚡ Quick Start Template

```python
# Cell 1: Setup
!pip install -q rouge-score sentencepiece

# Cell 2: Verify paths
!python verify_kaggle_paths.py

# Cell 3: Train
!python train_student_ultimate.py
```

## 🐛 Troubleshooting

### Issue: Import error for sentencepiece
```python
!pip install --upgrade sentencepiece
```

### Issue: Transformers version conflict
```python
!pip install --upgrade transformers
```

### Issue: Rouge-score not found
```python
!pip install rouge-score
```

### Issue: CUDA out of memory
```python
# Already optimized in scripts, but you can reduce:
# In train_student_ultimate.py:
# BATCH_SIZE = 2  # Instead of 3
```

## 📊 Verify Installation

```python
# Run this to check all packages
import sys
packages = [
    'torch', 'torchvision', 'transformers', 
    'sentencepiece', 'rouge_score', 'pandas',
    'PIL', 'tqdm', 'sklearn'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {pkg}: {version}")
    except ImportError:
        print(f"❌ {pkg}: NOT INSTALLED")
```

## 💡 Recommendations

**Tốt nhất:**
- Chỉ cài `rouge-score` và `sentencepiece` 
- Kaggle đã có sẵn phần còn lại
- Tiết kiệm thời gian khởi động

**Nếu có lỗi:**
- Cài đầy đủ từ requirements.txt
- Hoặc upgrade specific package bị lỗi

## ⏱️ Installation Time

- **Minimal** (rouge-score + sentencepiece): ~10-15 giây
- **Full** (requirements.txt): ~1-2 phút

## 🎯 Summary

```python
# Chỉ cần chạy dòng này là đủ:
!pip install -q rouge-score sentencepiece

# Xong! Ready to train 🚀
```
