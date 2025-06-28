import torch
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from dataset_vqa_gen import VQAGenDataset
from vqa_vietnamese_gen import VQAGenModel
from evaluate import load
from tqdm import tqdm

# --- Config ---
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test'
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load BLEU and ROUGE ---
bleu = load("bleu")
rouge = load("rouge")

# --- Dataset ---
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# --- Model ---
model = VQAGenModel().to(DEVICE)
model.eval()

preds, refs = [], []

with torch.no_grad():
    for vision_feats, input_ids, attention_mask, labels in tqdm(dataloader):
        vision_feats = vision_feats.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(vision_feats, input_ids, attention_mask)
        decoded_preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_refs = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds.extend(decoded_preds)
        refs.extend(decoded_refs)

# --- Evaluation ---
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
rouge_score = rouge.compute(predictions=preds, references=refs)

print("BLEU:", bleu_score)
print("ROUGE:", rouge_score)
