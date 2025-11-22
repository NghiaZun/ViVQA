"""
Quick inference test with best model on 5 random test samples
"""
import torch
from PIL import Image
from model import VQAGenModel
from transformers import BlipProcessor
import pandas as pd
import random

# =====================
# CONFIG
# =====================
BEST_MODEL_PATH = "/kaggle/input/v1/transformers/default/1/vqa_student_best_ultimate.pt"
TEST_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
TEST_IMG_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading model...")
model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/base/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/base/transformers/default/1/checkpoints/vit5_tokenizer"
)

# Load best model weights
print(f"[INFO] Loading weights from: {BEST_MODEL_PATH}")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
model = model.to(device)
model.eval()

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

print("[INFO] Model loaded successfully!\n")

# =====================
# LOAD TEST DATA
# =====================
print("[INFO] Loading test data...")
test_df = pd.read_csv(TEST_CSV)
print(f"[INFO] Total test samples: {len(test_df)}")

# Sample 5 random images
sample_indices = random.sample(range(len(test_df)), min(5, len(test_df)))

print("\n" + "="*80)
print("INFERENCE RESULTS")
print("="*80 + "\n")

# =====================
# INFERENCE
# =====================
with torch.no_grad():
    for idx, sample_idx in enumerate(sample_indices, 1):
        row = test_df.iloc[sample_idx]
        
        img_path = f"{TEST_IMG_DIR}/{row['img_id']}.jpg"
        question = row['question']
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Preprocess
            vision_inputs = vision_processor(images=image, return_tensors="pt")
            pixel_values = vision_inputs['pixel_values'].to(device)
            
            # Encode question
            q_inputs = model.phobert_tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            q_input_ids = q_inputs['input_ids'].to(device)
            q_attention_mask = q_inputs['attention_mask'].to(device)
            
            # Generate answer
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=q_input_ids,
                attention_mask=q_attention_mask,
                max_length=160,
                num_beams=3,
                early_stopping=True
            )
            
            # Decode
            pred_answer = model.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Print result
            print(f"Sample {idx}:")
            print(f"  Image: {row['img_id']}")
            print(f"  Question: {question}")
            print(f"  Predicted Answer: {pred_answer}")
            print()
            
        except Exception as e:
            print(f"Sample {idx} FAILED:")
            print(f"  Image: {row['img_id']}")
            print(f"  Error: {str(e)}")
            print()

print("="*80)
print("INFERENCE COMPLETE")
print("="*80)
