from transformers import AutoTokenizer, BlipImageProcessor
from model import VQAGenModel
import torch

def answer_question(model, vision_processor, q_tokenizer, vit5_tokenizer, image_folder, question, img_id, device='cpu', max_q_len=32):
    import os
    from PIL import Image
    model.eval()
    img_path = os.path.join(image_folder, f"{img_id}.jpg")
    image = Image.open(img_path).convert('RGB')
    pixel_values = vision_processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    q_enc = q_tokenizer(question, truncation=True, padding='max_length', max_length=max_q_len, return_tensors='pt')
    input_ids = q_enc['input_ids'].to(device)
    attention_mask = q_enc['attention_mask'].to(device)
    with torch.no_grad():
        pred_ids = model.generate(pixel_values, input_ids, attention_mask, max_length=32)
        print(pred_ids)  # <-- Thêm dòng này để debug
        print(vit5_tokenizer.decode(pred_ids[0])) 
        answer = vit5_tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    return answer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VQAGenModel().to(device)
#model.load_state_dict(torch.load('PATH_TO_MODEL.pth', map_location=device))
model.eval()

vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip-vqa-base')
phobert_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
vit5_tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
image_folder = '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test'

# Gọi hàm
question = "màu của miếng vá là gì"
img_id = "557067"

answer = answer_question(
    model, vision_processor, phobert_tokenizer, vit5_tokenizer,
    image_folder, question, img_id, device
)
print("Answer:", answer)