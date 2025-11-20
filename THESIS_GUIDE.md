# Vietnamese Visual Question Answering with Prompt Enhancement and Knowledge Distillation

## 🎯 Thesis Overview

This thesis addresses the challenge of generating high-quality, reasoned answers for Visual Question Answering (VQA) in Vietnamese. Traditional VQA methods often produce inconsistent or illogical responses. Our approach uses:

1. **Prompt Enhancement Pipeline**: Iterative refinement of prompts through error detection and reconstruction
2. **Knowledge Distillation**: Training a student model from a large teacher model (vLLM)
3. **Multi-Objective Learning**: Simultaneous optimization of format, answer, and reasoning generation

## 📊 Problem Identified

Your model was generating outputs like:
```
❌ BAD OUTPUT:
<reasoning>Băng ghế trong ảnh có màu trắng.</reasoning>
(Missing the <answer> tag completely!)
```

Instead of the desired format:
```
✅ GOOD OUTPUT:
<answer>Đen</answer>
<reasoning>[DESCRIPTIVE] Màu sắc của chiếc váy trong hình ảnh là đen.</reasoning>
```

## 🔧 Root Causes & Solutions

### Problems:
1. **Imbalanced loss weights** - Reasoning weight (0.50) too high vs answer (0.27)
2. **No format enforcement** - Model not penalized for missing tags
3. **Static training** - Same weights throughout training
4. **No curriculum** - No progressive learning strategy

### Solutions Implemented:

#### 1. **Adaptive Weight Scheduling** (`train_student_adaptive_v3.py`)
- **Early epochs (0-20%)**: Emphasize format (0.50) and answer (0.35), low reasoning (0.15)
- **Mid epochs**: Smooth transition using cosine annealing
- **Late epochs**: Balance shifts to reasoning quality (0.50)

```python
Stage 1 (Epoch 0-15):   Format=0.60, Answer=0.35, Reason=0.05
Stage 2 (Epoch 15-30):  Format=0.50, Answer=0.30, Reason=0.20  
Stage 3 (Epoch 30+):    Format=0.20-0.30, Answer=0.20-0.30, Reason=0.45-0.60
```

#### 2. **Curriculum Learning** (`train_student_ultimate.py`)
Three-stage training strategy:
- **Stage 1 (Epochs 0-15)**: ANSWER_FOCUS
  - Student learns to generate core answers
  - High weight on answer loss
- **Stage 2 (Epochs 15-30)**: FORMAT_LEARNING
  - Student learns XML structure with tags
  - Emphasis on complete format
- **Stage 3 (Epochs 30+)**: REASONING_QUALITY
  - Polish reasoning quality and types
  - High weight on reasoning loss

#### 3. **Format-Aware Loss**
Custom loss function that gives **2.5x higher weight** to special tokens:
- `<answer>`, `</answer>`, `<reasoning>`, `</reasoning>`
- Forces model to predict format tags correctly

#### 4. **Enhanced Data Encoding**
Three distinct training objectives:
```python
format_ids:  "<answer>Đen</answer>\n<reasoning>[DESCRIPTIVE] Màu...</reasoning>"
answer_ids:  "<answer>Đen</answer>"
reason_ids:  "<reasoning>[DESCRIPTIVE] Màu...</reasoning>"
```

## 🚀 Training Scripts

### Option 1: Adaptive V3 (Recommended for stability)
```bash
python train_student_adaptive_v3.py
```
- Smooth weight transitions
- Cosine annealing LR schedule
- 80 epochs, early stopping

### Option 2: Ultimate (Best for format enforcement)
```bash
python train_student_ultimate.py
```
- 3-stage curriculum learning
- Format-aware loss function
- 100 epochs with stage checkpoints

## 📈 Evaluation

```bash
python eval_adaptive_v3.py
```

Enhanced evaluation features:
- **Robust XML parsing** with multiple fallback strategies
- **Format correctness** tracking
- Extracts answer even from malformed output
- Comprehensive metrics: EM, Token-F1, ROUGE-1, ROUGE-L

## 🔍 Key Architectural Improvements

### Model Architecture (`model.py`)
```
Input: Image + Vietnamese Question
  ↓
[BLIP ViT] → Vision features
[PhoBERT]  → Text features
  ↓
Fusion Layer (2-layer MLP)
  ↓
[VietT5 Decoder] → Structured output
  ↓
Output: <answer>X</answer>\n<reasoning>[TYPE] Y</reasoning>
```

### Special Tokens
All special tokens properly added to VietT5:
- Structure tags: `<answer>`, `</answer>`, `<reasoning>`, `</reasoning>`
- Reasoning types: `[DESCRIPTIVE]`, `[CAUSAL]`, `[SPATIAL]`, `[COUNTING]`, `[COMMONSENSE]`, etc.

## 📊 Expected Improvements

### Before (your current model):
- Format correctness: ~30-40%
- EM: ~0.15-0.20
- Model generates reasoning but forgets answer

### After (with adaptive/ultimate training):
- Format correctness: **>85%**
- EM: **>0.35-0.45**
- Model generates BOTH answer AND reasoning consistently

## 🎓 Thesis Contributions

1. **Prompt Enhancement for Vietnamese VQA**
   - Iterative error detection and prompt reconstruction
   - Improves teacher model output quality

2. **Multi-Objective Knowledge Distillation**
   - Student learns format, answer, and reasoning simultaneously
   - Adaptive weight scheduling based on training progress

3. **Curriculum Learning Strategy**
   - Progressive complexity: Answer → Format → Reasoning
   - Format-aware loss for structure enforcement

4. **Robust Evaluation Framework**
   - Multiple fallback strategies for parsing
   - Comprehensive metrics beyond exact match

## 📁 File Structure

```
ViVQA/
├── model.py                          # Student VQA model architecture
├── train_student_adaptive_v3.py      # Adaptive weight training ⭐
├── train_student_ultimate.py         # Curriculum learning training ⭐⭐
├── eval_adaptive_v3.py               # Enhanced evaluation
├── train_student_multi_kd_v2.py      # Old version (for comparison)
├── eval_distill.py                   # Basic evaluation
└── kaggle/
    └── teacher_outputs.jsonl         # Teacher-generated training data
```

## 🔬 Experimental Settings

### Hyperparameters (Optimized):
```python
EPOCHS = 80-100
LR = 1e-5 (with cosine annealing)
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2 (effective batch = 8)
MAX_ANSWER_LENGTH = 160 tokens
NUM_BEAMS = 5 (for inference)
```

### Weight Schedule:
- Dynamic adjustment based on training stage
- Answer priority in early training
- Reasoning quality in late training

## 📝 Usage for Thesis

1. **Training**:
   ```bash
   # Use ultimate version for best results
   python train_student_ultimate.py
   ```

2. **Evaluation**:
   ```bash
   python eval_adaptive_v3.py
   ```

3. **Generate predictions**:
   ```python
   from model import VQAGenModel
   model = VQAGenModel()
   model.load_state_dict(torch.load("vqa_student_best_ultimate.pt"))
   
   output_ids = model.generate(pixel_values, input_ids, attention_mask)
   text = decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
   ```

## 🎯 Key Results to Report

1. **Format Adherence**: % of outputs with correct XML structure
2. **Exact Match (EM)**: Strict answer correctness
3. **Token F1**: Partial answer correctness
4. **ROUGE Scores**: Reasoning quality
5. **By Reasoning Type**: Performance breakdown for DESCRIPTIVE, CAUSAL, SPATIAL, etc.

## 🔮 Future Work

1. Attention visualization for explainability
2. Multi-modal fusion improvements
3. Cross-lingual transfer learning
4. Real-time inference optimization

## 📖 Citation

```bibtex
@thesis{vivqa_prompt_enhancement,
  title={Prompt Enhancement for Vietnamese Visual Question Answering with Knowledge Distillation},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## ⚡ Quick Start

```bash
# 1. Train with curriculum learning
python train_student_ultimate.py

# 2. Evaluate on test set
python eval_adaptive_v3.py

# 3. Check results
cat /kaggle/working/train_val_log_ultimate.csv
cat /kaggle/working/eval_adaptive_v3_results.csv
```

## 🐛 Troubleshooting

### Issue: Model still generates reasoning only
- Use `train_student_ultimate.py` with format-aware loss
- Increase `format_token_weight` to 3.0 or higher
- Reduce reasoning weight in early epochs

### Issue: Low exact match
- Check if answers are being generated (format correctness)
- Evaluate token F1 instead of EM for partial credit
- Analyze error patterns in eval output

### Issue: Training unstable
- Reduce learning rate to 5e-6
- Increase gradient accumulation steps
- Use more warmup epochs

---

**Good luck with your thesis! 🎓🚀**
