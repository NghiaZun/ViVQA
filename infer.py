from transformers import AutoTokenizer, T5Tokenizer, BlipImageProcessor
from PIL import Image
import torch
from model import VQAGenModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VQAGenModel().to(device)
model.load_state_dict(torch.load('PATH_TO_MODEL.pth', map_location=device))
model.eval()

vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

image = Image.open("sample.jpg").convert("RGB")
pixel_values = vision_processor(images=image, return_tensors='pt')['pixel_values'].to(device)
question = "Người trong ảnh đang làm gì?"
enc = phobert_tokenizer(question, return_tensors="pt", padding=True)
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

with torch.no_grad():
    pred_ids = model(pixel_values, input_ids, attention_mask, labels=None)
    answer = t5_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print("Answer:", answer)