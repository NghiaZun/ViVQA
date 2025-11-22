# Training Ultimate - Multi-Task Learning với Format Enforcement

## 🎯 Tổng quan

Phiên bản training tối ưu với **multi-task learning** để giảm overfitting và đảm bảo output format đúng chuẩn XML.

## ✨ Features chính

### 1. **Multi-Task Learning (3 objectives riêng biệt)**
```python
# 3 forward passes per batch
loss_answer   = model(labels=answer_ids)      # <answer>...</answer>
loss_reasoning = model(labels=reasoning_ids)  # <reasoning>[TYPE]...</reasoning>
loss_format   = model(labels=full_format)     # Full structure

total_loss = w_answer * loss_answer + w_reasoning * loss_reasoning + w_format * loss_format
```

**Lợi ích:**
- ✅ Mỗi component có signal riêng → học tốt hơn
- ✅ Giảm overfitting (separate objectives)
- ✅ Logging rõ ràng: F, A, R đều có giá trị thực

### 2. **Curriculum Learning (3 stages)**
```
Stage 1 (0-20):   Answer=50%, Format=35%, Reason=15%  → Master answers first
Stage 2 (20-40):  Answer=35%, Format=35%, Reason=30%  → Balance all
Stage 3 (40+):    Answer=20%, Format=20%, Reason=60%  → Reasoning quality
```

### 3. **Anti-Overfitting Techniques**

#### Label Smoothing (0.1)
```python
# Thay vì: true_label=1.0, others=0.0
# Dùng: true_label=0.9, others=0.1/(vocab-1)
```
→ Model không quá tự tin, generalize tốt hơn

#### Image Augmentation
```python
transforms.Compose([
    RandomHorizontalFlip(p=0.3),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomRotation(degrees=5)
])
```
→ Tăng diversity của training data

#### EMA (Exponential Moving Average)
```python
# Smooth model weights: θ_ema = 0.999 * θ_ema + 0.001 * θ_current
# Best model lưu EMA weights → stable hơn
```
→ Giảm variance, improve generalization

#### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
→ Prevent gradient explosion

### 4. **Format Validation**

#### Validate output format
```python
validation = validate_format(text)
# Check: <answer>...</answer> và <reasoning>[TYPE]...</reasoning>
```

#### Auto-fix malformed output
```python
fixed_text = fix_format(text)
# Tự động sửa output thiếu tags hoặc sai format
```

#### Periodic checking during training
- Mỗi 5 epochs check format quality
- Report % outputs valid
- Warning nếu có lỗi format

## 📊 Expected Results

### Trước (Single Loss)
```
Train Loss: 0.1439 (F:0.1439 | A:0.0000 | R:0.0000)  ← A, R không học
Val Loss:   0.2322 (F:0.2322 | A:0.0000 | R:0.0000)  ← Overfitting
```

### Sau (Multi-Task)
```
Train Loss: 0.1250 (F:0.0420 | A:0.0380 | R:0.0450)  ← Tất cả đều học
Val Loss:   0.1420 (F:0.0480 | A:0.0430 | R:0.0510)  ← Gap nhỏ hơn
Format Valid: 9/10 (90%)                               ← Có thể parse
```

## ⚙️ Hyperparameters

| Parameter | Value | Lý do |
|-----------|-------|-------|
| LR | 5e-6 | Thấp hơn cho 3x forward passes |
| Batch Size | 2 | Giảm để fit 3 models trong memory |
| Accum Steps | 4 | Effective batch = 8 |
| Label Smoothing | 0.1 | Standard cho generation tasks |
| EMA Decay | 0.999 | Smooth averaging |
| Warmup Epochs | 3 | Stabilize multi-task training |

## 🚀 Usage

### Training
```python
# Đã config sẵn, chỉ cần chạy:
python train_student_ultimate.py
```

### Inference với format validation
```python
# Generate
output = model.generate(...)
text = tokenizer.decode(output)

# Validate
validation = validate_format(text)
if not validation['valid']:
    text = fix_format(text)  # Auto-fix

# Parse
answer = validation['answer']
reasoning = validation['reasoning_text']
```

## 📝 Logs & Checkpoints

### Training logs
```csv
epoch,stage,train_loss,train_F,train_R,train_A,val_loss,val_F,val_R,val_A,...
1,ANSWER_MASTERY,0.15,0.05,0.04,0.06,0.18,0.06,0.05,0.07,...
```

### Checkpoints
- `latest_checkpoint.pt` - Auto-saved mỗi epoch (resume)
- `vqa_student_best_ultimate.pt` - Best model (EMA weights)
- `vqa_student_final_ultimate.pt` - Final model (EMA weights)

## ⚠️ Notes

1. **Training time**: ~3x chậm hơn (3 forward passes/batch)
2. **Memory**: Đã optimize với batch=2, accum=4
3. **First epochs**: Loss cao do warmup (bình thường)
4. **Format check**: Mỗi 5 epochs, đảm bảo 90%+ valid

## 🎓 Theory

### Why Multi-Task Learning?
- Shared representations học tốt hơn
- Regularization effect (prevent overfitting)
- Better generalization

### Why Label Smoothing?
- Prevent over-confidence
- Smoother loss landscape
- Better calibration

### Why EMA?
- Reduce variance
- More stable predictions
- Like ensemble of recent models

## 📚 References

- Label Smoothing: Szegedy et al., "Rethinking the Inception Architecture" (2016)
- EMA: Polyak & Juditsky, "Acceleration of Stochastic Approximation" (1992)
- Multi-Task Learning: Caruana, "Multitask Learning" (1997)
