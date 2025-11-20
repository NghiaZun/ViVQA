# 📍 Hướng dẫn sửa đường dẫn cho Kaggle

## 🎯 CÁC ĐƯỜNG DẪN CẦN SỬA

Khi clone repo lên Kaggle, bạn cần thêm các dataset sau vào Notebook Input và sửa đường dẫn:

---

## 1️⃣ DATASET CẦN THÊM VÀO KAGGLE

### Bước 1: Tạo/Upload các Dataset sau lên Kaggle:

```
📦 teacher-checkpoint-11k
   └── teacher_outputs.jsonl

📦 vivqa-dataset  
   ├── ViVQA-main/ViVQA-main/
   │   ├── train.csv
   │   └── test.csv
   └── drive-download-20220309T020508Z-001/
       ├── train/ (ảnh training)
       └── test/  (ảnh test)

📦 base-checkpoints
   └── transformers/default/1/checkpoints/
       ├── phobert_tokenizer/
       └── vit5_tokenizer/

📦 pretrained-checkpoint (optional - nếu có checkpoint cũ)
   └── vqa_student_best_multiKD.pt
```

### Bước 2: Add các dataset vào Notebook
- Click "Add Data" ở sidebar phải
- Tìm và add các dataset đã upload

---

## 2️⃣ SỬA ĐƯỜNG DẪN TRONG CÁC FILE

### 📄 **train_student_ultimate.py**

Sửa các dòng sau (dòng 28-30):
```python
# TRƯỚC (GitHub)
DATA_PATH = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"

# SAU (cần check tên dataset thực tế trên Kaggle của bạn)
DATA_PATH = "/kaggle/input/[TÊN-DATASET-CỦA-BẠN]/teacher_outputs.jsonl"
```

Sửa dòng 233-234 (load model):
```python
# TRƯỚC
phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer"
vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"

# SAU (check cấu trúc folder dataset của bạn)
phobert_dir="/kaggle/input/[TÊN-DATASET-MODEL]/phobert_tokenizer"
vit5_dir="/kaggle/input/[TÊN-DATASET-MODEL]/vit5_tokenizer"
```

Sửa dòng 255 (checkpoint cũ - nếu có):
```python
# TRƯỚC
CHECKPOINT_PATH = "/kaggle/input/v2/transformers/default/1/vqa_student_best_multiKD.pt"

# SAU
CHECKPOINT_PATH = "/kaggle/input/[TÊN-DATASET-CHECKPOINT]/vqa_student_best_multiKD.pt"
# Hoặc comment dòng này nếu train từ đầu
```

---

### 📄 **train_student_adaptive_v3.py**

Sửa các dòng tương tự:
- Dòng 22: `DATA_PATH`
- Dòng 177-178: `phobert_dir`, `vit5_dir`
- Dòng 195: `CHECKPOINT_PATH` (nếu có)

---

### 📄 **eval_adaptive_v3.py**

Sửa dòng 21-23:
```python
# TRƯỚC
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_BASE = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
STUDENT_CHECKPOINT = "/kaggle/working/vqa_student_best_adaptive_v3.pt"

# SAU (check cấu trúc dataset)
TEST_CSV = "/kaggle/input/[TÊN-DATASET-VIVQA]/ViVQA-main/ViVQA-main/test.csv"
IMAGE_BASE = "/kaggle/input/[TÊN-DATASET-VIVQA]/drive-download-20220309T020508Z-001/test"
STUDENT_CHECKPOINT = "/kaggle/working/vqa_student_best_adaptive_v3.pt"
# (checkpoint output giữ nguyên /kaggle/working)
```

Sửa dòng 224-225:
```python
# TRƯỚC
phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer"
vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"

# SAU
phobert_dir="/kaggle/input/[TÊN-DATASET-MODEL]/phobert_tokenizer"
vit5_dir="/kaggle/input/[TÊN-DATASET-MODEL]/vit5_tokenizer"
```

---

## 3️⃣ TỰ ĐỘNG TÌM ĐƯỜNG DẪN

Thêm đoạn code này vào đầu notebook để tự động tìm:

```python
import os
import glob

# Tìm tất cả input datasets
print("📂 Available datasets:")
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:  # Chỉ show 3 levels
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Show 5 files đầu
            print(f"{subindent}{file}")

# Tìm file cụ thể
teacher_files = glob.glob("/kaggle/input/**/teacher_outputs.jsonl", recursive=True)
print(f"\n📄 Teacher outputs found at:")
for f in teacher_files:
    print(f"   {f}")

test_csv = glob.glob("/kaggle/input/**/test.csv", recursive=True)
print(f"\n📄 Test CSV found at:")
for f in test_csv:
    print(f"   {f}")

phobert = glob.glob("/kaggle/input/**/phobert_tokenizer", recursive=True)
print(f"\n📄 PhoBERT tokenizer found at:")
for f in phobert:
    print(f"   {f}")
```

---

## 4️⃣ TEMPLATE ĐƯỜNG DẪN MẪU

Copy template này vào đầu notebook và sửa theo kết quả tìm được:

```python
# ==================== KAGGLE PATHS CONFIG ====================
# Sửa các đường dẫn này theo dataset của bạn

# Data paths
DATA_PATH = "/kaggle/input/YOUR-TEACHER-DATASET/teacher_outputs.jsonl"
TRAIN_CSV = "/kaggle/input/YOUR-VIVQA-DATASET/ViVQA-main/ViVQA-main/train.csv"
TEST_CSV = "/kaggle/input/YOUR-VIVQA-DATASET/ViVQA-main/ViVQA-main/test.csv"
TRAIN_IMAGE_DIR = "/kaggle/input/YOUR-VIVQA-DATASET/drive-download-20220309T020508Z-001/train"
TEST_IMAGE_DIR = "/kaggle/input/YOUR-VIVQA-DATASET/drive-download-20220309T020508Z-001/test"

# Model paths
PHOBERT_DIR = "/kaggle/input/YOUR-MODEL-DATASET/phobert_tokenizer"
VIT5_DIR = "/kaggle/input/YOUR-MODEL-DATASET/vit5_tokenizer"

# Checkpoint path (nếu có)
OLD_CHECKPOINT = None  # Hoặc "/kaggle/input/YOUR-CHECKPOINT/model.pt"

# Output paths (không đổi)
SAVE_DIR = "/kaggle/working"

print("✅ Paths configured!")
# ============================================================
```

Sau đó trong code, thay thế:
```python
# Thay vì
DATA_PATH = "/kaggle/input/..."

# Dùng
# DATA_PATH đã define ở trên
```

---

## 5️⃣ CHECKLIST TRƯỚC KHI TRAINING

```python
import os

def verify_paths():
    """Kiểm tra tất cả paths trước khi train"""
    
    paths = {
        "Teacher data": DATA_PATH,
        "PhoBERT tokenizer": PHOBERT_DIR,
        "VietT5 tokenizer": VIT5_DIR,
        "Train images": TRAIN_IMAGE_DIR,
        "Test images": TEST_IMAGE_DIR,
    }
    
    all_ok = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All paths verified! Ready to train!")
    else:
        print("\n⚠️ Some paths missing! Please fix before training.")
    
    return all_ok

# Chạy trước khi train
verify_paths()
```

---

## 6️⃣ QUICK FIX - SỬA NHANH TRONG NOTEBOOK

Nếu không muốn sửa file, thêm đoạn này vào đầu notebook:

```python
# Override paths
import sys
import types

# Tạo config module
config = types.ModuleType('kaggle_config')
config.DATA_PATH = "/kaggle/input/YOUR-DATASET/teacher_outputs.jsonl"
config.PHOBERT_DIR = "/kaggle/input/YOUR-MODEL/phobert_tokenizer"
config.VIT5_DIR = "/kaggle/input/YOUR-MODEL/vit5_tokenizer"
config.TEST_CSV = "/kaggle/input/YOUR-VIVQA/test.csv"
config.IMAGE_BASE = "/kaggle/input/YOUR-VIVQA/test"

sys.modules['kaggle_config'] = config

# Trong file train_student_ultimate.py, thêm ở đầu:
# try:
#     from kaggle_config import *
# except:
#     pass  # Use default paths
```

---

## 📋 TÓM TẮT

**Các file cần sửa:**
1. ✅ `train_student_ultimate.py` - 3 chỗ
2. ✅ `train_student_adaptive_v3.py` - 3 chỗ
3. ✅ `eval_adaptive_v3.py` - 2 chỗ

**Các đường dẫn cần check:**
- `/kaggle/input/[dataset-name]/teacher_outputs.jsonl`
- `/kaggle/input/[dataset-name]/phobert_tokenizer`
- `/kaggle/input/[dataset-name]/vit5_tokenizer`
- `/kaggle/input/[dataset-name]/ViVQA-main/ViVQA-main/*.csv`
- `/kaggle/input/[dataset-name]/drive-download-20220309T020508Z-001/train|test`

**Output paths (không đổi):**
- `/kaggle/working/*` - Tất cả outputs

---

🚀 **Sau khi sửa xong, chạy `verify_paths()` để check!**
