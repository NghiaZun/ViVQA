# 🚀 Kaggle GPU Optimization Guide

## 📊 Kaggle Environment Specs

**Standard GPU (T4)**:
- VRAM: 16GB GDDR6
- CUDA Cores: 2560
- Tensor Cores: 320
- Max Session: 12 hours/week (30 hours with phone verification)

## ⚡ Optimizations Applied

### 1. **Memory-Optimized Batch Sizes**
```python
BATCH_SIZE = 3              # Reduced from 4
accum_steps = 3             # Effective batch = 9
PIN_MEMORY = False          # Disable if OOM
```

**Why**: T4 has 16GB, model + optimizer states ≈ 8GB, leaves ~8GB for batches

### 2. **Gradient Checkpointing** ✅
```python
USE_GRADIENT_CHECKPOINTING = True
```
- Trades compute for memory
- Saves ~30-40% VRAM
- Adds ~20% training time
- **Enabled by default in decoder**

### 3. **Mixed Precision Training** ✅
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(...)
```
- Uses FP16 for faster computation
- Reduces memory by ~40%
- Maintains FP32 for critical ops

### 4. **Periodic Memory Cleanup**
```python
EMPTY_CACHE_EVERY_N_STEPS = 50

if (step + 1) % EMPTY_CACHE_EVERY_N_STEPS == 0:
    clear_memory()
```
- Prevents memory fragmentation
- Clears unused cached tensors

### 5. **Efficient DataLoader**
```python
num_workers=2               # Optimal for Kaggle
prefetch_factor=2           # Prefetch 2 batches
persistent_workers=True     # Keep workers alive
```

### 6. **Checkpoint Loading Optimization**
```python
ckpt = torch.load(PATH, map_location='cpu')  # Load to CPU first
model.load_state_dict(ckpt)
del ckpt                    # Free memory immediately
clear_memory()
```

### 7. **cuDNN Benchmarking** ✅
```python
torch.backends.cudnn.benchmark = True
```
- Auto-selects fastest convolution algorithms
- ~5-10% speedup for fixed input sizes

## 📈 Memory Budget Breakdown

**Model Components** (approximate):
```
Vision Encoder (BLIP ViT):     ~300MB
Text Encoder (PhoBERT):        ~450MB
Decoder (VietT5):              ~900MB
Fusion Layer:                  ~50MB
─────────────────────────────────────
Model Total:                   ~1.7GB

Optimizer States (AdamW):      ~3.4GB
Gradients:                     ~1.7GB
─────────────────────────────────────
Training Total:                ~6.8GB

Batch (3 samples):             ~2.5GB
Activation Cache:              ~1.5GB
Mixed Precision Overhead:      ~0.5GB
─────────────────────────────────────
Peak Usage:                    ~11-12GB / 16GB ✅
```

## 🔧 If You Still Get OOM

### Level 1: Reduce Batch Size
```python
BATCH_SIZE = 2
accum_steps = 4  # Maintain effective batch = 8
```

### Level 2: Reduce Sequence Length
```python
MAX_A_LEN = 128  # Down from 160
```

### Level 3: Disable Pin Memory
```python
PIN_MEMORY = False
```

### Level 4: Reduce Beams (Evaluation)
```python
NUM_BEAMS = 3  # Down from 4-5
```

### Level 5: Process Val Set in Smaller Chunks
```python
# Split validation into multiple passes
val_loader_small = DataLoader(val_set, batch_size=2)
```

## ⏱️ Training Time Estimates

**On Kaggle T4 GPU**:

| Configuration | Time/Epoch | 100 Epochs |
|---------------|-----------|------------|
| Batch=3, Accum=3 | ~35 min | ~58 hours |
| Batch=2, Accum=4 | ~42 min | ~70 hours |
| With Checkpointing | +7 min | +12 hours |

**Recommendation**: 
- Use **Kaggle Sessions** in multiple runs
- Save checkpoints every 10 epochs
- Resume from last checkpoint if session expires

## 🔄 Resume Training After Timeout

Add this to your training script:

```python
# Find latest checkpoint
checkpoints = sorted(glob.glob(f"{SAVE_DIR}/checkpoint_epoch*.pt"))
if checkpoints:
    latest = checkpoints[-1]
    start_epoch = int(re.search(r'epoch(\d+)', latest).group(1))
    model.load_state_dict(torch.load(latest, map_location='cpu'))
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 0

for epoch in range(start_epoch, EPOCHS):
    # ... training code
```

## 📊 Monitor GPU Usage

### In Code (Already Added):
```python
print_gpu_memory()  # Shows allocation
```

### In Notebook:
```python
!nvidia-smi
```

### Continuous Monitoring:
```python
!watch -n 1 nvidia-smi  # Updates every 1 second
```

## 💡 Kaggle-Specific Tips

### 1. **Internet Access**
Enable "Internet" in Kaggle notebook settings for downloading models

### 2. **Persistent Storage**
```python
# Save to /kaggle/working (persists after session)
# NOT /tmp (deleted on restart)
```

### 3. **Dataset Versions**
Use Kaggle Datasets for teacher outputs:
```python
DATA_PATH = "/kaggle/input/teacher-checkpoint-11k/teacher_outputs.jsonl"
```

### 4. **Output Datasets**
Create output dataset from `/kaggle/working/`:
- Easier to share checkpoints
- Can use in future sessions

### 5. **GPU Quota Management**
- Each run counts towards 12hr/week quota
- Plan for 2-3 sessions of 4-5 hours each
- Save frequently!

## 🎯 Recommended Training Strategy

### **Session 1** (4-5 hours, ~8 epochs):
```bash
python train_student_ultimate.py
# Stage 1: ANSWER_FOCUS (epochs 0-15)
```
Output: `checkpoint_epoch10.pt`

### **Session 2** (4-5 hours, ~8 epochs):
Resume from epoch 10
```
# Stage 1→2 transition
```
Output: `checkpoint_epoch20.pt`

### **Session 3** (4-5 hours, ~8 epochs):
Resume from epoch 20
```
# Stage 2: FORMAT_LEARNING (epochs 15-30)
```
Output: `checkpoint_epoch30.pt`

### **Sessions 4-8**: Continue until convergence

**Total**: ~10-12 sessions = complete training

## 🚨 Emergency OOM Solutions

If you get OOM error mid-training:

```python
# 1. Reduce batch size immediately
BATCH_SIZE = 2

# 2. Clear all caches
import gc
gc.collect()
torch.cuda.empty_cache()

# 3. Restart kernel and reduce config

# 4. Disable gradient checkpointing if using old GPUs
USE_GRADIENT_CHECKPOINTING = False  # Only if T4
```

## 📈 Expected Performance

**Training Speed** (T4 GPU):
- Steps/second: ~1.2-1.5
- Samples/second: ~4-5
- Epoch time: ~35-40 minutes

**Memory Usage**:
- Baseline: 8-10GB
- Peak (during backward): 11-13GB
- After cleanup: 8-9GB

## ✅ Pre-Flight Checklist

Before starting training on Kaggle:

- [ ] GPU accelerator enabled in settings
- [ ] Internet enabled (for model downloads)
- [ ] Dataset paths verified
- [ ] Output directory is `/kaggle/working`
- [ ] Checkpoint resume logic in place
- [ ] Batch size appropriate (2-3)
- [ ] Mixed precision enabled
- [ ] Gradient checkpointing enabled
- [ ] Memory monitoring in place
- [ ] Periodic saves configured (every 5-10 epochs)

## 🔬 Verify Optimization

Run this before training:

```python
# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

# Load model
model = VQAGenModel(...).to('cuda')
print(f"Model loaded: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Test forward pass
dummy_batch = {k: v.to('cuda') for k, v in dummy_batch.items()}
with torch.cuda.amp.autocast():
    loss = model(**dummy_batch)
print(f"After forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Test backward pass
loss.backward()
print(f"After backward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Should be < 14GB
assert torch.cuda.memory_allocated() < 14 * 1024**3, "OOM risk!"
```

## 📞 Quick Commands

```bash
# Monitor GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Clear Python cache
rm -rf __pycache__

# Free disk space
du -h /kaggle/working
```

## 🎓 Summary

Your training scripts are now optimized for Kaggle T4 GPU with:
- ✅ 16GB memory budget managed
- ✅ Gradient checkpointing enabled
- ✅ Mixed precision training
- ✅ Efficient data loading
- ✅ Periodic memory cleanup
- ✅ GPU monitoring built-in

**You're ready to train on Kaggle! 🚀**
