import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, base_layer, r=4, alpha=1.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 동일한 device에 LoRA 계층 초기화
        device = self.base.weight.device

        if isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels
            self.lora_down = nn.Conv2d(in_channels, r, kernel_size=1, bias=False).to(device)
            self.lora_up = nn.Conv2d(r, out_channels, kernel_size=1, bias=False).to(device)

        elif isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            self.lora_down = nn.Linear(in_features, r, bias=False).to(device)
            self.lora_up = nn.Linear(r, out_features, bias=False).to(device)

        else:
            raise NotImplementedError(f"LoRA not supported for {type(base_layer)}")

    def forward(self, x):
        base_out = self.base(x)
        if isinstance(self.base, nn.Conv2d):
            lora_out = self.lora_up(self.lora_down(x))
            if lora_out.shape[-2:] != base_out.shape[-2:]:
                lora_out = F.interpolate(
                    lora_out, size=base_out.shape[-2:], mode='bilinear', align_corners=False
                )
        else:
            lora_out = self.lora_up(self.lora_down(x))

        return base_out + self.scaling * lora_out


def inject_lora_layers(model, r=4, alpha=1.0):
    """
    모델의 Linear/Conv2d 레이어에 LoRA를 삽입하여 수정된 모델 반환.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            continue

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            lora = LoRALayer(module, r=r, alpha=alpha)
            setattr(model, name, lora)
        else:
            inject_lora_layers(module, r=r, alpha=alpha)
    return model


def get_lora_params(model):
    """
    LoRA 레이어에서 학습 가능한 파라미터만 추출.
    """
    return [p for n, p in model.named_parameters() if 'lora_' in n or isinstance(p, nn.Parameter)]
