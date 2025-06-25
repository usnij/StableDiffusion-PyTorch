import torch
import torch.nn as nn
from models.blocks_lora import DownBlockWithLoRA, MidBlockWithLoRA, UpBlockUnetWithLoRA
from models.blocks import get_time_embedding
# ---- 전체 UnetWithLoRA ----
class UnetWithLoRA(nn.Module):
    def __init__(self, im_channels, model_config, lora_r=4, lora_alpha=16):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        self.condition_config = model_config.get('condition_config', None)
        self.text_cond = False
        self.text_embed_dim = None
        if self.condition_config and 'condition_types' in self.condition_config and 'text' in self.condition_config['condition_types']:
            self.text_cond = True
            self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlockWithLoRA(
                self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                down_sample=self.down_sample[i],
                num_heads=self.num_heads,
                num_layers=self.num_down_layers,
                attn=self.attns[i],
                norm_channels=self.norm_channels,
                cross_attn=self.text_cond,
                context_dim=self.text_embed_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            ))
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlockWithLoRA(
                self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                num_heads=self.num_heads,
                num_layers=self.num_mid_layers,
                norm_channels=self.norm_channels,
                cross_attn=self.text_cond,
                context_dim=self.text_embed_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            ))
        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlockUnetWithLoRA(
                self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                self.t_emb_dim,
                up_sample=self.down_sample[i],
                num_heads=self.num_heads,
                num_layers=self.num_up_layers,
                norm_channels=self.norm_channels,
                cross_attn=self.text_cond,
                context_dim=self.text_embed_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            ))
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, 3, padding=1)

    def forward(self, x, t, cond_input=None):
        out = self.conv_in(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        context_hidden_states = None
        if self.text_cond and cond_input is not None:
            context_hidden_states = cond_input['text']   # shape: (B, N_ctx, text_embed_dim)
        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)
        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out