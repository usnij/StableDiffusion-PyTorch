import torch
import torch.nn as nn
from models.blocks import get_time_embedding
from models.blocks import DownBlock, MidBlock, UpBlockUnet, ControlNetDownBlock, ControlNetMidBlock, ControlNetUpBlock
from utils.config_utils import *
import torch.nn.functional as F
class ControlNetUnet(nn.Module):
    """
    기존 Unet 구조에 ControlNet branch(Condition branch)만 추가!
    기존 weight와 최대한 호환되게 설계
    """

    def __init__(self, im_channels, model_config):
        super().__init__()
        # --- Unet 부분 (기존 코드 거의 동일) ---
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
        
        # Conditioning config
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        self.image_cond = False
        if self.condition_config is not None:
            if 'image' in self.condition_config.get('condition_types', []):
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config']['image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config']['image_condition_output_channels']

        if self.image_cond:
            self.cond_conv_in = nn.Conv2d(self.im_cond_input_ch, self.im_cond_output_ch, kernel_size=1, bias=False)
            self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch, self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])
        self.mids = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.control_ups = nn.ModuleList([])

        # --- Unet Block 생성 ---
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                          down_sample=self.down_sample[i],
                          num_heads=self.num_heads,
                          num_layers=self.num_down_layers,
                          attn=self.attns[i], norm_channels=self.norm_channels,
                          cross_attn=False, context_dim=None)
            )

        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                         num_heads=self.num_heads,
                         num_layers=self.num_mid_layers,
                         norm_channels=self.norm_channels,
                         cross_attn=False, context_dim=None)
            )

        for i in reversed(range(len(self.down_channels) - 1)):
            in_ch = self.down_channels[i] * 2
            out_ch = self.down_channels[i - 1] if i != 0 else self.conv_out_channels
            self.control_ups.append(
                ControlNetUpBlock(in_ch, out_ch, self.t_emb_dim, up_sample=self.down_sample[i])
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, 512)
        self.conv_out = nn.Conv2d(512, im_channels, kernel_size=3, padding=1)
        self.control_proj = nn.Conv2d(self.conv_out_channels, 512, kernel_size=1)
        # --- ControlNet branch 추가 ---
        # 각각 Down/Mid/Up block 마다 ControlNet branch를 동일한 구조로 추가
        self.control_downs = nn.ModuleList([
            ControlNetDownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                               down_sample=self.down_sample[i]) for i in range(len(self.down_channels) - 1)
        ])
        self.control_mids = nn.ModuleList([
            ControlNetMidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim)
            for i in range(len(self.mid_channels) - 1)
        ])
        self.control_ups = nn.ModuleList([
            ControlNetUpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else self.conv_out_channels,
                              self.t_emb_dim, up_sample=self.down_sample[i]) 
            for i in reversed(range(len(self.down_channels) - 1))
        ])
        print("self.conv_out_channels:", self.conv_out_channels)

        
    def forward(self, x, t, cond_input=None):
        if self.image_cond:
            im_cond = cond_input['image']
            im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            x = torch.cat([x, im_cond], dim=1)
            out = self.conv_in_concat(x)
        else:
            out = self.conv_in(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        # --- ControlNet Branch ---
        control_feats = []
        out_control = out.clone()
        for idx, control_down in enumerate(self.control_downs):
            control_feats.append(out_control)
            out_control = control_down(out_control, t_emb)

        for control_mid in self.control_mids:
            out_control = control_mid(out_control, t_emb)

        for control_up in self.control_ups:
            feat = control_feats.pop()
            out_control = control_up(out_control, feat, t_emb)

        # --- Main Unet Forward ---
        down_outs = []
        out_main = out
        for idx, down in enumerate(self.downs):
            down_outs.append(out_main)
            out_main = down(out_main, t_emb, None)

        for mid in self.mids:
            out_main = mid(out_main, t_emb, None)

        for up in self.ups:
            down_out = down_outs.pop()
            out_main = up(out_main, down_out, t_emb, None)
            
        out_control = self.control_proj(out_control)
        if out_control.shape[-2:] != out_main.shape[-2:]:
            out_control = F.interpolate(out_control, size=out_main.shape[-2:], mode='bilinear', align_corners=False)

        out_main = out_main + 0.1 * out_control


        out_main = self.norm_out(out_main)
        out_main = nn.SiLU()(out_main)
        out_main = self.conv_out(out_main)
        return out_main
