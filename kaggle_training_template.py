"""
🚀 KAGGLE TRAINING TEMPLATE - VQA Student Model
Optimized for T4 GPU with 16GB VRAM

BEFORE RUNNING:
1. Enable GPU: Settings → Accelerator → GPU T4 x2
2. Enable Internet: Settings → Internet → On
3. Add datasets:
   - teacher-checkpoint-11k (teacher outputs)
   - base-checkpoints (model weights)
   - vivqa (images)
"""

# ===================================================================
# STEP 1: INSTALL DEPENDENCIES (if needed)
# ===================================================================
!pip install -q rouge-score transformers==4.35.0 torch==2.1.0

# ===================================================================
# STEP 2: VERIFY GPU
# ===================================================================
import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
else:
    print("❌ GPU not available! Enable GPU in settings.")

# ===================================================================
# STEP 3: CHECK DATA PATHS
# ===================================================================
import os

paths_to_check = {
    "Teacher outputs": "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl",
    "Train images": "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train",
    "Test images": "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test",
    "PhoBERT tokenizer": "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    "VietT5 tokenizer": "/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
}

all_exist = True
for name, path in paths_to_check.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {name}: {path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n⚠️ Some paths are missing! Update dataset inputs.")
else:
    print("\n✅ All paths verified!")

# ===================================================================
# STEP 4: VERIFY OUTPUT DIRECTORY
# ===================================================================
output_dir = "/kaggle/working"
os.makedirs(output_dir, exist_ok=True)
print(f"✅ Output directory: {output_dir}")

# ===================================================================
# STEP 5: UPLOAD YOUR SCRIPTS
# ===================================================================
print("\n📂 Upload these files to Kaggle notebook:")
print("   1. model.py")
print("   2. train_student_ultimate.py")
print("   3. eval_adaptive_v3.py")
print("\nOr add them as a dataset and copy here:")

# Example: Copy from dataset
# !cp /kaggle/input/vqa-scripts/*.py /kaggle/working/

# ===================================================================
# STEP 6: PRE-TRAINING CHECKS
# ===================================================================
def check_memory_before_training():
    """Verify memory is sufficient"""
    import torch
    import gc
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n🔍 Pre-training Memory Check:")
    print(f"   Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.1f}GB")
    
    # Test with small batch
    try:
        from model import VQAGenModel
        print("   Loading model...")
        model = VQAGenModel(
            vision_model_name="Salesforce/blip-vqa-base",
            phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
            vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
        ).to('cuda')
        
        mem_after_load = torch.cuda.memory_allocated()/1024**3
        print(f"   Model loaded: {mem_after_load:.2f}GB")
        
        # Test forward pass
        dummy_pix = torch.randn(2, 3, 384, 384).cuda()
        dummy_ids = torch.randint(0, 1000, (2, 64)).cuda()
        dummy_mask = torch.ones(2, 64).cuda()
        dummy_labels = torch.randint(0, 1000, (2, 128)).cuda()
        
        with torch.cuda.amp.autocast():
            out = model(
                pixel_values=dummy_pix,
                input_ids=dummy_ids,
                attention_mask=dummy_mask,
                labels=dummy_labels
            )
        
        mem_after_forward = torch.cuda.memory_allocated()/1024**3
        print(f"   After forward: {mem_after_forward:.2f}GB")
        
        if mem_after_forward < 14:
            print("   ✅ Memory check PASSED")
        else:
            print("   ⚠️ Memory usage high, reduce batch size!")
        
        del model, dummy_pix, dummy_ids, dummy_mask, dummy_labels
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"   ❌ Memory check FAILED: {e}")
        return False

# Run check
check_memory_before_training()

# ===================================================================
# STEP 7: TRAINING OPTIONS
# ===================================================================
print("\n" + "="*60)
print("TRAINING OPTIONS")
print("="*60)

OPTION = 1  # Change this to select option

if OPTION == 1:
    print("Option 1: Full Training (Ultimate)")
    print("   - 100 epochs with curriculum learning")
    print("   - ~10-12 Kaggle sessions needed")
    print("   - Best results")
    script = "train_student_ultimate.py"
    
elif OPTION == 2:
    print("Option 2: Adaptive Training")
    print("   - 80 epochs with smooth transitions")
    print("   - ~8-10 Kaggle sessions needed")
    print("   - Good results, faster")
    script = "train_student_adaptive_v3.py"
    
elif OPTION == 3:
    print("Option 3: Resume Training")
    print("   - Continue from checkpoint")
    print("   - Update START_EPOCH in script")
    script = "train_student_ultimate.py"  # or adaptive

# ===================================================================
# STEP 8: RUN TRAINING
# ===================================================================
print(f"\n🚀 Starting: {script}")
print("⏱️ Estimated time: 4-5 hours per session")
print("💾 Checkpoints saved every 10 epochs")
print("\n" + "="*60 + "\n")

# METHOD 1: Run as script
!python {script}

# METHOD 2: Import and run (for debugging)
# %run {script}

# ===================================================================
# STEP 9: MONITOR TRAINING
# ===================================================================
# Open another notebook cell and run:
# !watch -n 5 nvidia-smi

# Or check logs:
# !tail -f /kaggle/working/train_val_log_ultimate.csv

# ===================================================================
# STEP 10: AFTER TRAINING - SAVE OUTPUTS
# ===================================================================
print("\n" + "="*60)
print("SAVING OUTPUTS")
print("="*60)

import glob
import shutil

output_files = [
    "/kaggle/working/vqa_student_best_ultimate.pt",
    "/kaggle/working/train_val_log_ultimate.csv",
    "/kaggle/working/checkpoint_epoch*.pt"
]

for pattern in output_files:
    files = glob.glob(pattern)
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024**2
            print(f"✅ {os.path.basename(f)}: {size:.1f}MB")

print("\n💡 To save outputs:")
print("   1. Notebook Settings → Persistence → Always save output")
print("   2. Or create a new Dataset from /kaggle/working")
print("   3. Use in future sessions as input dataset")

# ===================================================================
# STEP 11: EVALUATION (Optional - Run in separate session)
# ===================================================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)
print("Run this in a NEW session to save GPU quota:")

eval_code = '''
# Load best model and evaluate
!python eval_adaptive_v3.py

# Check results
import pandas as pd
df = pd.read_csv("/kaggle/working/eval_adaptive_v3_results.csv")
print(f"Exact Match: {df['exact_match'].mean():.4f}")
print(f"Token F1: {df['token_f1'].mean():.4f}")
print(f"Format Correct: {df['has_correct_format'].mean():.1%}")
'''

print(eval_code)

# ===================================================================
# TROUBLESHOOTING
# ===================================================================
print("\n" + "="*60)
print("TROUBLESHOOTING")
print("="*60)

print("""
🐛 Common Issues:

1. OOM Error:
   - Reduce BATCH_SIZE to 2 in script
   - Reduce MAX_A_LEN to 128
   - Disable PIN_MEMORY

2. Session Timeout:
   - Save checkpoints frequently (every 5-10 epochs)
   - Resume from last checkpoint
   - Plan for multiple sessions

3. Data Not Found:
   - Check dataset is added to notebook inputs
   - Verify paths match /kaggle/input/[dataset-name]/

4. Slow Training:
   - Verify GPU is enabled (not CPU)
   - Check GPU utilization: !nvidia-smi
   - Ensure mixed precision is working

5. Model Not Learning:
   - Check loss is decreasing
   - Verify data is loading correctly
   - Check learning rate schedule

📖 Full documentation: See KAGGLE_GPU_OPTIMIZATION.md
""")

# ===================================================================
# MONITORING SCRIPT
# ===================================================================
print("\n📊 Run this in a separate cell to monitor training:\n")

monitor_code = '''
import time
import pandas as pd
from IPython.display import clear_output

while True:
    clear_output(wait=True)
    
    # GPU stats
    !nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    
    # Training progress
    if os.path.exists("/kaggle/working/train_val_log_ultimate.csv"):
        df = pd.read_csv("/kaggle/working/train_val_log_ultimate.csv")
        if len(df) > 0:
            print(f"\\nEpoch {df.iloc[-1]['epoch']}/{100}")
            print(f"Train Loss: {df.iloc[-1]['train_loss']:.4f}")
            print(f"Val Loss: {df.iloc[-1]['val_loss']:.4f}")
            print(f"Best Val Loss: {df['val_loss'].min():.4f}")
    
    time.sleep(30)  # Update every 30 seconds
'''

print(monitor_code)

print("\n✅ Setup complete! Ready to train on Kaggle GPU 🚀")
