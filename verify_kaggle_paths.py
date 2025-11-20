"""
🔍 Kiểm tra đường dẫn Kaggle - Chạy script này TRƯỚC KHI TRAINING
"""

import os
import sys

print("="*70)
print("🔍 KAGGLE PATH VERIFICATION")
print("="*70)

# =====================
# PATHS TO CHECK
# =====================
paths_config = {
    "📄 Teacher Outputs": "/kaggle/input/teacher-outputs/teacher_outputs.jsonl",
    "📊 Test CSV": "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv",
    "📊 Train CSV": "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv",
    "🖼️ Train Images": "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train",
    "🖼️ Test Images": "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test",
    "🤖 PhoBERT Tokenizer": "/kaggle/input/base/transformers/default/1/phobert_tokenizer",
    "🤖 VietT5 Tokenizer": "/kaggle/input/base/transformers/default/1/vit5_tokenizer",
    "💾 Output Directory": "/kaggle/working"
}

# =====================
# CHECK EACH PATH
# =====================
all_ok = True
missing_paths = []

for name, path in paths_config.items():
    exists = os.path.exists(path)
    
    if exists:
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024**2)
            print(f"✅ {name}")
            print(f"   Path: {path}")
            print(f"   Size: {size:.1f} MB\n")
        else:
            # Directory - count files
            try:
                files = os.listdir(path)
                print(f"✅ {name}")
                print(f"   Path: {path}")
                print(f"   Files: {len(files)} items\n")
            except:
                print(f"✅ {name}")
                print(f"   Path: {path}\n")
    else:
        print(f"❌ {name}")
        print(f"   Path: {path}")
        print(f"   Status: NOT FOUND\n")
        all_ok = False
        missing_paths.append((name, path))

# =====================
# SUMMARY
# =====================
print("="*70)
if all_ok:
    print("🎉 ALL PATHS VERIFIED!")
    print("✅ Ready to train!")
else:
    print("⚠️ MISSING PATHS DETECTED!")
    print(f"\nMissing {len(missing_paths)} path(s):")
    for name, path in missing_paths:
        print(f"  • {name}: {path}")
    
    print("\n📋 TO FIX:")
    print("  1. Go to Kaggle Notebook → Add Data (right sidebar)")
    print("  2. Add these datasets:")
    print("     - teacher-outputs (for teacher_outputs.jsonl)")
    print("     - vivqa (for images and CSV files)")
    print("     - base (for PhoBERT and VietT5 tokenizers)")
    print("  3. Make sure dataset names match the paths above")

print("="*70)

# =====================
# SHOW ALL AVAILABLE DATASETS
# =====================
print("\n📂 AVAILABLE KAGGLE DATASETS:")
print("="*70)

if os.path.exists("/kaggle/input"):
    datasets = os.listdir("/kaggle/input")
    if datasets:
        for ds in sorted(datasets):
            ds_path = f"/kaggle/input/{ds}"
            print(f"📦 {ds}")
            
            # Show first level structure
            try:
                items = os.listdir(ds_path)
                for item in sorted(items)[:5]:  # Show first 5 items
                    item_path = os.path.join(ds_path, item)
                    if os.path.isdir(item_path):
                        print(f"   📁 {item}/")
                    else:
                        print(f"   📄 {item}")
                if len(items) > 5:
                    print(f"   ... and {len(items)-5} more items")
            except:
                pass
            print()
    else:
        print("⚠️ No datasets found in /kaggle/input")
        print("   Please add datasets to your notebook!")
else:
    print("⚠️ /kaggle/input not found")
    print("   Are you running on Kaggle?")

print("="*70)

# =====================
# DETAILED SEARCH
# =====================
print("\n🔎 SEARCHING FOR KEY FILES...")
print("="*70)

import glob

search_patterns = {
    "teacher_outputs.jsonl": "**/teacher_outputs.jsonl",
    "test.csv": "**/test.csv",
    "train.csv": "**/train.csv",
    "phobert_tokenizer": "**/phobert_tokenizer",
    "vit5_tokenizer": "**/vit5_tokenizer"
}

for name, pattern in search_patterns.items():
    print(f"\n🔍 Searching for: {name}")
    results = glob.glob(f"/kaggle/input/{pattern}", recursive=True)
    if results:
        for r in results:
            print(f"   ✅ Found: {r}")
    else:
        print(f"   ❌ Not found")

print("\n" + "="*70)

# =====================
# GPU CHECK
# =====================
print("\n🎮 GPU CHECK:")
print("="*70)

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ Total VRAM: {total_mem:.1f} GB")
    else:
        print("⚠️ GPU not available!")
        print("   Enable GPU in Notebook Settings → Accelerator → GPU T4 x2")
except ImportError:
    print("❌ PyTorch not installed")
    print("   Run: !pip install torch")

print("="*70)

# =====================
# FINAL STATUS
# =====================
print("\n" + "="*70)
if all_ok:
    print("✅ VERIFICATION COMPLETE - READY TO TRAIN! 🚀")
    print("\nNext steps:")
    print("  1. Run: !python train_student_ultimate.py")
    print("  2. Or: !python train_student_adaptive_v3.py")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED - FIX PATHS BEFORE TRAINING")
    print("\nRerun this script after fixing the paths.")
    sys.exit(1)
