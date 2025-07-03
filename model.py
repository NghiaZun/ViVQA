# ===============================
# 1. MÔ HÌNH VQA GENERATIVE
# ===============================

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

class VQAGenModel(nn.Module):
    def __init__(self, vision_dim=1408, hidden_dim=768, num_query_tokens=32,
                 qformer_layers=4, qformer_heads=8,
                 phoBERT_name='vinai/phobert-base',
                 decoder_name='VietAI/vit5-base'):
        super().__init__()

        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))

        self.qformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=qformer_heads, batch_first=True)
            for _ in range(qformer_layers)
        ])

        self.phobert = AutoModel.from_pretrained(phoBERT_name)
        self.decoder = T5ForConditionalGeneration.from_pretrained(decoder_name)

        self.pho_proj = nn.Linear(self.phobert.config.hidden_size, hidden_dim)
        self.tokenizer = T5Tokenizer.from_pretrained(decoder_name)

    def forward(self, image_feats, input_ids, attention_mask, labels=None):
        b = image_feats.size(0)
        image_feats = self.vision_proj(image_feats)  # (b, n, d)
        query = self.query_tokens.expand(b, -1, -1)

        for layer in self.qformer_layers:
            query = layer(query, src_key_padding_mask=None)

        phobert_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        phobert_proj = self.pho_proj(phobert_out)

        decoder_input = torch.cat([query, phobert_proj], dim=1)
        decoder_attention = torch.ones(decoder_input.size()[:-1], dtype=torch.long).to(decoder_input.device)

        if labels is not None:
            output = self.decoder(
                encoder_outputs=(decoder_input,),
                attention_mask=decoder_attention,
                labels=labels
            )
            return output.loss, output.logits
        else:
            output = self.decoder(
                encoder_outputs=(decoder_input,),
                attention_mask=decoder_attention
            )
            pred_ids = output.logits.argmax(-1)
            return pred_ids
