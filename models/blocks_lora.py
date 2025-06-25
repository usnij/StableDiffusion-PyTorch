import torch
import torch.nn as nn
from models.lora import LoRALinear
from models.blocks import get_time_embedding

# ---- Down Block with LoRA (self/cross attention 지원) ----
class DownBlockWithLoRA(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None, lora_r=4, lora_alpha=16):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1)
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
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            ) for _ in range(num_layers)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)
        ])

        if attn:
            self.attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.q_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.k_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.v_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.out_proj_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        if cross_attn:
            self.cross_attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.cross_q_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_k_layers = nn.ModuleList([LoRALinear(context_dim, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_v_layers = nn.ModuleList([LoRALinear(context_dim, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_out_proj_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context_hidden_states=None):
        out = x
        for i in range(self.num_layers):
            res_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None and t_emb is not None:
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

            if self.cross_attn and context_hidden_states is not None:
                b, c, h, w = out.shape
                normed = self.cross_attention_norms[i](out)
                flat = normed.flatten(2).transpose(1, 2)  # (B, HW, C)
                q = self.cross_q_layers[i](flat)
                k = self.cross_k_layers[i](context_hidden_states)
                v = self.cross_v_layers[i](context_hidden_states)
                attn_weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (c ** 0.5), dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = self.cross_out_proj_layers[i](attn_output)
                out_attn = attn_output.transpose(1, 2).reshape(b, c, h, w)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out

# ---- Mid Block with LoRA ----
class MidBlockWithLoRA(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=False, context_dim=None, lora_r=4, lora_alpha=16):
        super().__init__()
        self.num_layers = num_layers
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
            ) for i in range(num_layers + 1)
        ])
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
            for _ in range(num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            ) for _ in range(num_layers + 1)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers + 1)
        ])
        # Self-attention (LoRA)
        self.attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
        self.q_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.k_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.v_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.out_proj_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        # Cross-attention (LoRA)
        if cross_attn:
            self.cross_attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.cross_q_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_k_layers = nn.ModuleList([LoRALinear(context_dim, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_v_layers = nn.ModuleList([LoRALinear(context_dim, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_out_proj_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])

    def forward(self, x, t_emb=None, context_hidden_states=None):
        out = x
        # 첫 resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None and t_emb is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        # 반복: attention → resblock
        for i in range(self.num_layers):
            # Self-attention
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
            # Cross-attention (텍스트 임베딩)
            if self.cross_attn and context_hidden_states is not None:
                normed = self.cross_attention_norms[i](out)
                flat = normed.flatten(2).transpose(1, 2)
                q = self.cross_q_layers[i](flat)
                k = self.cross_k_layers[i](context_hidden_states)
                v = self.cross_v_layers[i](context_hidden_states)
                attn_weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (c ** 0.5), dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = self.cross_out_proj_layers[i](attn_output)
                out_attn = attn_output.transpose(1, 2).reshape(b, c, h, w)
                out = out + out_attn
            # 다음 resblock
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None and t_emb is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        return out

# ---- Up Block with LoRA ----
class UpBlockUnetWithLoRA(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, num_layers, norm_channels, cross_attn=False, context_dim=None, lora_r=4, lora_alpha=16):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1)
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
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            ) for _ in range(num_layers)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)
        ])
        self.attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
        self.q_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.k_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.v_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.out_proj_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        if cross_attn:
            self.cross_attention_norms = nn.ModuleList([nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)])
            self.cross_q_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_k_layers = nn.ModuleList([LoRALinear(context_dim, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_v_layers = nn.ModuleList([LoRALinear(context_dim, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
            self.cross_out_proj_layers = nn.ModuleList([LoRALinear(out_channels, out_channels, lora_r, lora_alpha) for _ in range(num_layers)])
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None, context_hidden_states=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            if x.shape[2:] != out_down.shape[2:]:
                min_h = min(x.shape[2], out_down.shape[2])
                min_w = min(x.shape[3], out_down.shape[3])
                x = x[:, :, :min_h, :min_w]
                out_down = out_down[:, :, :min_h, :min_w]
            x = torch.cat([x, out_down], dim=1)
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None and t_emb is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Self-attn
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
            # Cross-attn
            if self.cross_attn and context_hidden_states is not None:
                normed = self.cross_attention_norms[i](out)
                flat = normed.flatten(2).transpose(1, 2)
                q = self.cross_q_layers[i](flat)
                k = self.cross_k_layers[i](context_hidden_states)
                v = self.cross_v_layers[i](context_hidden_states)
                attn_weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (c ** 0.5), dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = self.cross_out_proj_layers[i](attn_output)
                out_attn = attn_output.transpose(1, 2).reshape(b, c, h, w)
                out = out + out_attn
        return out