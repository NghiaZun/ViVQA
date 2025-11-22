# ✅ Pre-Upload Checklist for Kaggle

## 🔧 Fixed Issues

### 1. RuntimeError: Shape Mismatch ✅
- **Error**: `The size of tensor a (160) must match the size of tensor b (320)`
- **Fix**: Flatten all tensors consistently from start
- **Code**: `logits_flat = logits.view(-1, vocab)`, `labels_flat = labels.view(-1)`

### 2. FutureWarning: Deprecated autocast ✅
- **Error**: `torch.cuda.amp.autocast(args...)` is deprecated
- **Fix**: Use new API `torch.amp.autocast('cuda')`
- **Benefit**: PyTorch 2.0+ compatible

### 3. PyTorch Version Compatibility ✅
- **Added**: Version check for label_smoothing parameter
- **Fallback**: Works with PyTorch 1.8+ (Kaggle uses 2.x)

## 🎯 Features Implemented

### Multi-Task Learning
- ✅ 3 separate objectives: Answer + Reasoning + Format
- ✅ Each component has independent loss signal
- ✅ Proper logging: F, A, R all have real values

### Anti-Overfitting
- ✅ Label smoothing (0.1)
- ✅ Image augmentation (flip, jitter, rotation)
- ✅ EMA (exponential moving average)
- ✅ Gradient clipping (1.0)
- ✅ Weight decay (1e-4)

### Format Enforcement
- ✅ `validate_format()` - Check XML structure
- ✅ `fix_format()` - Auto-repair malformed output
- ✅ Periodic validation every 5 epochs
- ✅ Final report on format accuracy

### Training Stability
- ✅ LR warmup (3 epochs)
- ✅ Cosine annealing scheduler
- ✅ Mixed precision (AMP)
- ✅ Gradient accumulation (4 steps)

## 📋 Kaggle Upload Instructions

### 1. Files to Upload
```
train_student_ultimate.py  ← Main training script
model.py                   ← Model architecture
```

### 2. Required Kaggle Inputs
```
Dataset inputs:
- teacher-outputs/teacher_outputs.jsonl
- vivqa/train images
- base checkpoints (phobert, vit5, blip)

Resume checkpoint (optional):
- vivqa-checkpoint/latest_checkpoint.pt
```

### 3. Expected GPU
```
T4 GPU (16GB VRAM)
- Batch size: 2
- Gradient accumulation: 4
- Effective batch: 8
```

### 4. Training Time
```
Per epoch: ~10-15 minutes
Total (100 epochs): ~16-25 hours
Early stopping: Usually stops around epoch 30-50
```

## 📊 Expected Logs

### Startup
```
[INFO] PyTorch version: 2.x.x
[INFO] Label smoothing: Enabled
[INFO] GPU: Tesla T4
[INFO] Total GPU Memory: 15.89GB
[INFO] Dataset: 11000 total, 9900 train, 1100 val
[INFO] Training augmentation: Horizontal flip, Color jitter, Rotation
[INFO] EMA enabled with decay=0.999
```

### During Training
```
Epoch 1/100 | Stage: ANSWER_MASTERY
Task Weights: Answer=0.500, Format=0.350, Reason=0.150
LR: 1.67e-06 [Warmup 1/3]

Train Loss: 0.1250 (F:0.0420 | A:0.0380 | R:0.0450)
Val Loss:   0.1420 (F:0.0480 | A:0.0430 | R:0.0510)

[FORMAT CHECK] Valid: 8/10 (80%)  ← Every 5 epochs
```

### Final Output
```
[SUMMARY]
  • Best Val Loss: 0.1420
  • Total Epochs: 45/100
  • Format Validation: 9/10 samples (90%) valid
  
[SAVED MODELS]
  • Best: vqa_student_best_ultimate.pt
  • Final: vqa_student_final_ultimate.pt
  • Latest: latest_checkpoint.pt
```

## ⚠️ Troubleshooting

### If OOM (Out of Memory)
```python
# Reduce batch size in train_student_ultimate.py
BATCH_SIZE = 1  # Line ~40
accum_steps = 8  # Line ~45 (keep effective batch = 8)
```

### If Training Too Slow
```python
# Disable some features temporarily
USE_EMA = False                    # Line ~51
VALIDATE_FORMAT_EVERY_N_EPOCHS = 10  # Line ~62
```

### If Format Validation < 80%
- Normal in early epochs (< 20)
- Should improve to 90%+ by epoch 40+
- If stuck < 80%, increase format weight:
  ```python
  format_weight=3.0  # Line ~510 (default 2.5)
  ```

## 🚀 Ready to Upload!

All fixes verified. Upload to Kaggle and start training!

**Reminder**: Save notebook frequently during training.
