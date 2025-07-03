import torch
from torch import nn
from transformers import (
    BlipForQuestionAnswering,
    AutoTokenizer, AutoModel,
    T5ForConditionalGeneration, T5Tokenizer
)

class VQAGenModel(nn.Module):
    def __init__(self,
                 vision_model_name="Salesforce/blip-image-base",
                 text_model_name="vinai/phobert-base",
                 decoder_model_name="t5-base",
                 hidden_dim=768):
        super().__init__()
        # Vision encoder (ViT-B/16)
        blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.vision_encoder = blip.vision_model
        # Text encoder (PhoBERT-small)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Decoder (VietT5-base)
        self.decoder = T5ForConditionalGeneration.from_pretrained(decoder_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(decoder_model_name)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Encode image
        vision_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        vision_feats = vision_out.mean(dim=1)  # Mean pooling

        # Encode text
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = (text_out * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        # Fusion
        fused = torch.cat([vision_feats, text_feats], dim=-1)
        fused = self.fusion(fused).unsqueeze(1)  # (B, 1, hidden_dim)

        # Decode
        if labels is not None:
            outputs = self.decoder(
                inputs_embeds=fused,
                labels=labels,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            outputs = self.decoder(
                inputs_embeds=fused,
                return_dict=True
            )
            pred_ids = outputs.logits.argmax(-1)
            return pred_ids

    def generate(self, pixel_values, input_ids, attention_mask, **gen_kwargs):
        vision_feats = self.vision_encoder(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feats = (text_out * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        fused = self.fusion(torch.cat([vision_feats, text_feats], dim=-1)).unsqueeze(1)

        return self.decoder.generate(inputs_embeds=fused, **gen_kwargs)