import os
import torch
from torch import nn
from transformers import (
    BlipForQuestionAnswering,
    AutoTokenizer, AutoModel,
    AutoModelForSeq2SeqLM
)

class VQAGenModel(nn.Module):
    """
    VQA student model:
    - Vision: BLIP ViT
    - Text: PhoBERT
    - Decoder: VietT5
    - Fusion: concat vision+text features -> linear layers
    """
    def __init__(self,
                 vision_model_name="Salesforce/blip-vqa-base",
                 phobert_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
                 vit5_dir="/kaggle/input/base-checkpoints/transformers/default/1/checkpoints/vit5_tokenizer",
                 hidden_dim=768):
        super().__init__()

        # -----------------------
        # Vision Encoder (BLIP ViT)
        # -----------------------
        print("[INFO] Loading BLIP vision encoder...")
        blip = BlipForQuestionAnswering.from_pretrained(vision_model_name)
        self.vision_encoder = blip.vision_model

        # -----------------------
        # Text Encoder (PhoBERT)
        # -----------------------
        print("[INFO] Loading PhoBERT tokenizer & model...")
        if not any(fname.endswith((".bin", ".safetensors")) for fname in os.listdir(phobert_dir)):
            print(f"[WARN] PhoBERT weights not found in {phobert_dir}, loading HF hub model...")
            self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        else:
            self.text_encoder = AutoModel.from_pretrained(phobert_dir)

        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(phobert_dir, use_fast=False)
        except Exception:
            print("[WARN] PhoBERT tokenizer fallback to HF hub...")
            self.text_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

        # -----------------------
        # Fusion Layer
        # -----------------------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # -----------------------
        # Decoder (VietT5)
        # -----------------------
        print("[INFO] Loading VietT5 tokenizer & model...")
        if not any(fname.endswith((".bin", ".safetensors")) for fname in os.listdir(vit5_dir)):
            print(f"[WARN] VietT5 weights not found in {vit5_dir}, loading HF hub model...")
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
        else:
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained(vit5_dir)

        try:
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(vit5_dir, use_fast=False)
        except Exception:
            print("[WARN] VietT5 tokenizer fallback to HF hub...")
            self.decoder_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", use_fast=False)

    # -----------------------
    # Forward pass
    # -----------------------
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for training/inference
        - pixel_values: image tensor (B,C,H,W)
        - input_ids, attention_mask: question tokenized
        - labels: optional for decoder loss
        Returns: decoder outputs (logits, loss if labels)
        """
        # Vision features
        vision_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        vision_feats = vision_out.mean(dim=1)  # (B, hidden_dim)

        # Text features (CLS token)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = text_out[:,0,:]  # (B, hidden_dim)

        # Fusion
        fused = torch.cat([vision_feats, text_feats], dim=-1)
        fused = self.fusion(fused).unsqueeze(1)  # (B,1,hidden_dim)
        fusion_mask = torch.ones(fused.shape[:-1], dtype=torch.long, device=fused.device)

        # Encode + decode
        encoder_outputs = self.decoder.get_encoder()(inputs_embeds=fused, attention_mask=fusion_mask)

        if labels is not None:
            return self.decoder(encoder_outputs=encoder_outputs, labels=labels, return_dict=True)
        else:
            # return token IDs
            return self.decoder.generate(
                inputs_embeds=fused,
                attention_mask=fusion_mask,
                max_length=32,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id
            )

    # -----------------------
    # Generate helper
    # -----------------------
    def generate(self, pixel_values, input_ids, attention_mask=None, max_length=32):
        vision_feats = self.vision_encoder(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = text_out[:, 0, :]
        fused = self.fusion(torch.cat([vision_feats, text_feats], dim=-1)).unsqueeze(1)
        fusion_mask = torch.ones(fused.shape[:-1], dtype=torch.long).to(fused.device)
    
        return self.decoder.generate(
            inputs_embeds=fused,
            attention_mask=fusion_mask,
            max_length=max_length,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id
        )


