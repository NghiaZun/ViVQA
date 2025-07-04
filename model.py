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
                 text_model_name="vinai/phobert-base",
                 decoder_model_name="VietAI/vit5-base",
                 hidden_dim=768):
        super().__init__()

        # Vision encoder (ViT-B from BLIP)
        blip = BlipForQuestionAnswering.from_pretrained(vision_model_name)
        self.vision_encoder = blip.vision_model

        # Text encoder (PhoBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Fusion layer: [vision_feats; text_feats] → hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Decoder (VietT5-base via AutoModel)
        self.decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_model_name)

        # Tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Encode image
        vision_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        vision_feats = vision_out.mean(dim=1)  # (B, hidden_dim)

        # Encode question (use CLS token from PhoBERT)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = text_out[:, 0, :]  # CLS token

        # Fusion: concat & project
        fused = torch.cat([vision_feats, text_feats], dim=-1)  # (B, hidden_dim*2)
        fused = self.fusion(fused).unsqueeze(1)  # (B, 1, hidden_dim)
        fusion_mask = torch.ones(fused.shape[:-1], dtype=torch.long).to(fused.device)  # (B, 1)

        # Encode for decoder
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

    def generate(self, pixel_values, input_ids, attention_mask, **gen_kwargs):
        # Encode image
        vision_feats = self.vision_encoder(pixel_values=pixel_values).last_hidden_state.mean(dim=1)

        # Encode question
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = text_out[:, 0, :]  # CLS token

        # Fusion
        fused = self.fusion(torch.cat([vision_feats, text_feats], dim=-1)).unsqueeze(1)
        fusion_mask = torch.ones(fused.shape[:-1], dtype=torch.long).to(fused.device)

        return self.decoder.generate(
            inputs_embeds=fused,
            attention_mask=fusion_mask,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            max_length=32,
            num_beams=4,
            early_stopping=True,
            **gen_kwargs
        )
