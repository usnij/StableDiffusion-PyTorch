# models/lora.py

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1.0

        # 기본 선형 레이어 (pretrained weight 고정)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # LoRA 레이어
        if r > 0:
            self.lora_down = nn.Linear(in_features, r, bias=False)
            self.lora_up = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
            nn.init.zeros_(self.lora_up.weight)
        else:
            self.lora_down = None
            self.lora_up = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.lora_up(self.lora_down(self.dropout(x)))
            return base_out + self.scale * lora_out
        else:
            return base_out
