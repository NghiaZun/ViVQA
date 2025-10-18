import os
import torch
from torch import nn
from transformers import (
    BlipForQuestionAnswering,
    AutoTokenizer, AutoModel,
    AutoModelForSeq2SeqLM
)

class VQAGenModel(nn.Module):
    def __init__(self,
                 vision_model_name="Salesforce/blip-vqa-base",
                 phobert_dir="/kaggle/input/vqagen-checkpoint/phobert",
                 vit5_dir="/kaggle/input/vqagen-checkpoint/vit5",
                 hidden_dim=768):
        super().__init__()

        # ============================
        # Vision Encoder (BLIP ViT)
        # ============================
        print("[INFO] Loading BLIP vision encoder...")
        blip = BlipForQuestionAnswering.from_pretrained(vision_model_name)
        self.vision_encoder = blip.vision_model

        # ============================
        # Text Encoder (PhoBERT)
        # ============================
        print(f"[INFO] Loading PhoBERT tokenizer & model...")

        # ✅ Nếu local chỉ có tokenizer → load model từ HF
        if not any(fname.endswith((".bin", ".safetensors")) for fname in os.listdir(phobert_dir)):
            print(f"[WARN] No model weights found in {phobert_dir}, loading PhoBERT from HuggingFace Hub...")
            self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        else:
            self.text_encoder = AutoModel.from_pretrained(phobert_dir)

        # ✅ Load tokenizer (ưu tiên local)
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(phobert_dir, use_fast=False)
        except Exception:
            print("[WARN] Failed to load PhoBERT tokenizer locally → using vinai/phobert-base")
            self.text_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

        # ============================
        # Fusion Layer
        # ============================
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ============================
        # Decoder (VietT5)
        # ============================
        print(f"[INFO] Loading VietT5 tokenizer & model...")

        # ✅ Nếu local chỉ có tokenizer → load model từ HF
        if not any(fname.endswith((".bin", ".safetensors")) for fname in os.listdir(vit5_dir)):
            print(f"[WARN] No model weights found in {vit5_dir}, loading VietT5 from HuggingFace Hub...")
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
        else:
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained(vit5_dir)

        # ✅ Load tokenizer (ưu tiên local)
        try:
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(vit5_dir, use_fast=False)
        except Exception:
            print("[WARN] Failed to load VietT5 tokenizer locally → using VietAI/vit5-base")
            self.decoder_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", use_fast=False)

    # ============================
    # Forward
    # ============================
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Encode image
        vision_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        vision_feats = vision_out.mean(dim=1)  # (B, hidden_dim)

        # Encode question (CLS token from PhoBERT)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = text_out[:, 0, :]  # CLS token

        # Fusion
        fused = torch.cat([vision_feats, text_feats], dim=-1)
        fused = self.fusion(fused).unsqueeze(1)
        fusion_mask = torch.ones(fused.shape[:-1], dtype=torch.long).to(fused.device)

        # Decode
        encoder = self.decoder.get_encoder()
        encoder_outputs = encoder(inputs_embeds=fused, attention_mask=fusion_mask)

        if labels is not None:
            outputs = self.decoder(
                encoder_outputs=encoder_outputs,
                labels=labels,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            outputs = self.decoder.generate(
                inputs_embeds=fused,
                attention_mask=fusion_mask,
                max_length=32,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id
            )
            return outputs

    # ============================
    # Generate
    # ============================
    def generate(self, pixel_values, input_ids, attention_mask=None, **gen_kwargs):
        vision_feats = self.vision_encoder(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = text_out[:, 0, :]
        fused = self.fusion(torch.cat([vision_feats, text_feats], dim=-1)).unsqueeze(1)
        fusion_mask = torch.ones(fused.shape[:-1], dtype=torch.long).to(fused.device)

        gen_kwargs.setdefault("max_length", 32)
        gen_kwargs.setdefault("num_beams", 4)
        gen_kwargs.setdefault("early_stopping", True)
        gen_kwargs.setdefault("pad_token_id", self.decoder_tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.decoder_tokenizer.eos_token_id)

        return self.decoder.generate(
            inputs_embeds=fused,
            attention_mask=fusion_mask,
            **gen_kwargs
        )
