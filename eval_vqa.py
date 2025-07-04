import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BlipImageProcessor
from dataset import VQAGenDataset
from model import VQAGenModel
from evaluate import load
from tqdm import tqdm

# --- Config ---
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test'
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/kaggle/input/checkpoints/pytorch/default/1/checkpoints/vqagen_final.pth'
vit5_tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/checkpoints/pytorch/default/1/checkpoints/vit5_tokenizer')

# --- Load BLEU and ROUGE ---
bleu = load("bleu")
rouge = load("rouge")

# --- Dataset ---
vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
dataset = VQAGenDataset(CSV_PATH, IMAGE_FOLDER, vision_processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# --- Model ---
model = VQAGenModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

preds, refs = [], []

with torch.no_grad():
    for batch in tqdm(dataloader):
        pixel_values, input_ids, attention_mask, labels = batch
        pixel_values = pixel_values.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        pred_ids = model(pixel_values, input_ids, attention_mask, labels=None)
        decoded_preds = vit5_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        preds.extend(decoded_preds)

        if labels is not None:
            labels = labels.to(DEVICE)
            decoded_refs = vit5_tokenizer.batch_decode(labels, skip_special_tokens=True)
            refs.extend(decoded_refs)

# --- Evaluation ---
if refs:
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_score = rouge.compute(predictions=preds, references=refs)
    print("BLEU:", bleu_score)
    print("ROUGE:", rouge_score)
else:
    print("No references found in test set. Only predictions are generated.")
    for i, pred in enumerate(preds[:10]):
        print(f"Sample {i+1}: {pred}")