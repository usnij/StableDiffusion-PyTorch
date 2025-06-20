# models/clip_encoder.py
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn

class CLIPTextEmbedder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)

    def forward(self, input_ids):
        return self.text_model(input_ids).last_hidden_state
