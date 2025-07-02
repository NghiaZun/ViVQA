import torch
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from dataset import VQAGenDataset 
from model import VQAGenModel 
from evaluate import load
from tqdm import tqdm

# --- Config ---
CSV_PATH = '/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv'
IMAGE_FOLDER = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test'
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/kaggle/input/evaluate/pytorch/default/1/checkpoints/model_epoch1.pth'  # Sửa đường dẫn model nếu cần

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

# --- Tokenizer ---
tokenizer = model.language_model.tokenizer  # Sửa lại nếu model của bạn có thuộc tính tokenizer khác

preds, refs = [], []

with torch.no_grad():
    for batch in tqdm(dataloader):
        # Nếu dataset trả về 4 phần tử (có labels)
        if len(batch) == 4:
            vision_feats, input_ids, attention_mask, labels = batch
        else:
            vision_feats, input_ids, attention_mask = batch
            labels = None

        vision_feats = vision_feats.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        # Sinh câu trả lời bằng generate
        generated_ids = model(
            vision_feats, input_ids, attention_mask, labels=None
        )

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds.extend(decoded_preds)

        # Nếu có labels thì decode refs
        if labels is not None:
            labels = labels.to(DEVICE)
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
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