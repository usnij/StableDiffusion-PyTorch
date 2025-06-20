import torch
import torch.nn as nn
from models.lora import LoRALinear


class DownBlockWithLoRA(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1)
            ) for i in range(num_layers)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
            for _ in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        if attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.q_layers = nn.ModuleList([
                LoRALinear(out_channels, out_channels) for _ in range(num_layers)
            ])
            self.k_layers = nn.ModuleList([
                LoRALinear(out_channels, out_channels) for _ in range(num_layers)
            ])
            self.v_layers = nn.ModuleList([
                LoRALinear(out_channels, out_channels) for _ in range(num_layers)
            ])
            self.out_proj_layers = nn.ModuleList([
                LoRALinear(out_channels, out_channels) for _ in range(num_layers)
            ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x, t_emb=None):
        out = x
        for i in range(self.num_layers):
            res_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](res_input)

            if self.attn:
                b, c, h, w = out.shape
                normed = self.attention_norms[i](out)
                flat = normed.flatten(2).transpose(1, 2)

                q = self.q_layers[i](flat)
                k = self.k_layers[i](flat)
                v = self.v_layers[i](flat)

                attn_weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (c ** 0.5), dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = self.out_proj_layers[i](attn_output)

                out_attn = attn_output.transpose(1, 2).reshape(b, c, h, w)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out
