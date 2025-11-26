"""
Test inference on sample images from test set
Check if model generates correct format: Answer: X / Reasoning: Y
"""

import os
import json
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor
from model import VQAGenModel
import re

# =====================
# CONFIG
# =====================
MODEL_PATH = "/kaggle/input/best-v1/transformers/default/1/vqa_best.pt"  # or vqa_final.pt
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"

NUM_SAMPLES = 10  # Number of test samples to check
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# LOAD MODEL
# =====================
print(f"[INFO] Loading model from: {MODEL_PATH}")
print(f"[INFO] Device: {device}")

model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)

# Add special tokens BEFORE loading checkpoint
print("[INFO] Adding special tokens...")
added_tokens = model.add_special_tokens_and_resize()
if added_tokens > 0:
    print(f"[INFO] Added {added_tokens} special tokens")

# Load trained weights
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("[INFO] Model loaded successfully!")

# Load processors
vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# =====================
# LOAD TEST DATA
# =====================
print(f"\n[INFO] Loading test data from: {TEST_CSV}")
test_df = pd.read_csv(TEST_CSV)
print(f"[INFO] Total test samples: {len(test_df)}")

# Sample random test cases
import random
random.seed(42)
sample_indices = random.sample(range(len(test_df)), min(NUM_SAMPLES, len(test_df)))

# =====================
# PARSER
# =====================
def parse_output(text: str):
    """Parse Answer: X / Reasoning: Y format"""
    answer = ""
    reasoning = ""
    
    # Regex extraction
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Fallback: line-based
    if not answer or not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip()
            elif line.lower().startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid': bool(answer and reasoning)
    }

# =====================
# INFERENCE
# =====================
print(f"\n{'='*80}")
print(f"TESTING {NUM_SAMPLES} SAMPLES")
print(f"{'='*80}\n")

valid_count = 0
results = []

for idx, sample_idx in enumerate(sample_indices, 1):
    row = test_df.iloc[sample_idx]
    
    img_name = str(row['img_id'])  # Convert to string
    question = row['question']
    ground_truth = row['answer']
    
    # Load image
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot load image: {img_path}")
        continue
    
    # Prepare inputs
    pixel_values = vision_processor(img, return_tensors="pt").pixel_values.to(device)
    
    q_enc = model.text_tokenizer(
        question, 
        truncation=True, 
        padding="max_length",
        max_length=64, 
        return_tensors="pt"
    )
    input_ids = q_enc.input_ids.to(device)
    attention_mask = q_enc.attention_mask.to(device)
    
    # Generate answer
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode
    generated_text = model.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Parse
    parsed = parse_output(generated_text)
    
    if parsed['valid']:
        valid_count += 1
    
    # Store result
    results.append({
        'idx': sample_idx,
        'image': img_name,
        'question': question,
        'ground_truth': ground_truth,
        'generated': generated_text,
        'parsed_answer': parsed['answer'],
        'parsed_reasoning': parsed['reasoning'],
        'valid_format': parsed['valid']
    })
    
    # Print
    print(f"{'='*80}")
    print(f"Sample {idx}/{NUM_SAMPLES} (Index: {sample_idx})")
    print(f"{'='*80}")
    print(f"📷 Image: {img_name}")
    print(f"❓ Question: {question}")
    print(f"✅ Ground Truth: {ground_truth}")
    print(f"\n🤖 Generated Output:")
    print(f"{generated_text}")
    print(f"\n📊 Parsed:")
    print(f"   Answer: {parsed['answer']}")
    print(f"   Reasoning: {parsed['reasoning']}")
    print(f"   Valid Format: {'✅ YES' if parsed['valid'] else '❌ NO'}")
    print()

# =====================
# SUMMARY
# =====================
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total Samples: {len(results)}")
print(f"Valid Format: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
print(f"{'='*80}\n")

# =====================
# SAVE RESULTS
# =====================
output_file = "/kaggle/working/test_inference_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[INFO] Results saved to: {output_file}")
print(f"[INFO] Format accuracy: {valid_count/len(results)*100:.1f}%")

# Show examples of invalid format (if any)
invalid_samples = [r for r in results if not r['valid_format']]
if invalid_samples:
    print(f"\n[WARN] {len(invalid_samples)} samples have invalid format:")
    for r in invalid_samples[:3]:
        print(f"\n  Sample {r['idx']}:")
        print(f"  Generated: {r['generated'][:100]}...")
