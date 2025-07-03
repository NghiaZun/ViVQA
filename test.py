# Khởi tạo các thành phần cần thiết
from transformers import AutoTokenizer, T5Tokenizer, BlipImageProcessor
from model import VQAGenModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VQAGenModel().to(device)
#model.load_state_dict(torch.load('PATH_TO_MODEL.pth', map_location=device))
model.eval()

vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip-vqa-base')
phobert_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
image_folder = 'PATH_TO_IMAGE_FOLDER'

# Gọi hàm
question = "Người trong ảnh đang làm gì?"
img_id = "000123"  # thay bằng id ảnh thực tế

answer = answer_question(
    model, vision_processor, phobert_tokenizer, t5_tokenizer,
    image_folder, question, img_id, device
)
print("Answer:", answer)