import torch
from torch.utils.data import DataLoader
from transformers import BlipImageProcessor
from dataset import VQAGenDataset
from model import VQAGenModel
import pandas as pd
from tqdm import tqdm

def infer_test(model_path, csv_path, image_folder, save_path, batch_size=8):
    """
    Hàm infer để chạy tập test và lưu kết quả
    
    Args:
        model_path: Đường dẫn đến file model đã train
        csv_path: Đường dẫn đến file CSV test
        image_folder: Thư mục chứa ảnh test
        save_path: Đường dẫn lưu kết quả
        batch_size: Batch size cho inference
    """
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = VQAGenModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    test_dataset = VQAGenDataset(csv_path, image_folder, vision_processor, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Inference
    print("Starting inference...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Inferencing")):
            if len(batch) == 4:  # Training format với labels
                vision_feats, input_ids, attention_mask, _ = batch
            else:  # Test format không có labels
                vision_feats, input_ids, attention_mask = batch
            
            # Move to device
            vision_feats = vision_feats.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Generate answers
            generated_ids = model.generate(
                vision_feats=vision_feats,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=3,
                do_sample=False,
                pad_token_id=model.language_model.config.pad_token_id,
                eos_token_id=model.language_model.config.eos_token_id
            )
            
            # Decode generated answers
            for i, generated_seq in enumerate(generated_ids):
                # Skip input tokens để chỉ lấy phần answer được generate
                input_length = input_ids[i].shape[0]
                answer_ids = generated_seq[input_length:]
                
                # Decode to text
                answer = model.language_model.tokenizer.decode(
                    answer_ids, 
                    skip_special_tokens=True
                ).strip()
                
                results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'generated_answer': answer
                })
    
    # Save results
    print(f"Saving results to {save_path}...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    
    print(f"Inference completed! Results saved to {save_path}")
    print(f"Total samples processed: {len(results)}")
    
    return df_results

def infer_single_sample(model_path, image_path, question, vision_processor=None):
    """
    Infer cho một sample duy nhất
    
    Args:
        model_path: Đường dẫn model
        image_path: Đường dẫn ảnh
        question: Câu hỏi
        vision_processor: Vision processor (tùy chọn)
    
    Returns:
        answer: Câu trả lời được generate
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = VQAGenModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load vision processor
    if vision_processor is None:
        vision_processor = BlipImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    
    # Process image and question
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    vision_inputs = vision_processor(image, return_tensors="pt")
    vision_feats = vision_inputs['pixel_values'].to(device)
    
    # Tokenize question
    tokenizer = model.language_model.tokenizer
    prompt = f"Question: {question} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate answer
    with torch.no_grad():
        generated_ids = model.generate(
            vision_feats=vision_feats,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode answer
        input_length = input_ids.shape[1]
        answer_ids = generated_ids[0][input_length:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    
    return answer

# Sử dụng:
if __name__ == "__main__":
    # Infer tập test
    results = infer_test(
        model_path='/kaggle/input/evaluate/pytorch/default/1/checkpoints/model_epoch1.pth',
        csv_path='/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv',
        image_folder='/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test',
        save_path='/kaggle/working/test_results.csv',
        batch_size=8
    )
    
    # Xem một vài kết quả đầu tiên
    print("\nFirst 5 results:")
    print(results.head())
    
    # Infer một sample
    # answer = infer_single_sample(
    #     model_path='/kaggle/working/checkpoints/vqagen_final.pth',
    #     image_path='path/to/image.jpg',
    #     question='What is in the image?'
    # )
    # print(f"Answer: {answer}")