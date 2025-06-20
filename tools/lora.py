# 전체 코드 예시
import torch
from models.unet_cond_base import Unet
import yaml
from transformers import CLIPTokenizer, CLIPTextModel
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from models.vqvae import VQVAE
import numpy as np
import torch
import torch.nn as nn
import math
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("./config/celebhq_text_cond.yaml", 'r') as f:
    config = yaml.safe_load(f)


autoencoder_config = config['autoencoder_params']

ldm_config = config['ldm_params']
im_channels = autoencoder_config['z_channels']    # 예: 4

# 2. Text encoder & tokenizer 준비 + <me> 임베딩 bin파일 주입
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

# 2-1. <me> 토큰 추가 및 embedding resize
num_added = tokenizer.add_tokens(["<me>"])
if num_added > 0:
    text_encoder.resize_token_embeddings(len(tokenizer))

# 2-2. bin파일에서 <me> 임베딩 주입
embed_data = torch.load("./learned_embeds.bin", map_location="cpu")
me_id = tokenizer.encode("<me>", add_special_tokens=False)[0]
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[me_id] = embed_data["<me>"]
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features)
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = alpha / r
            nn.init.zeros_(self.lora_B.weight)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        else:
            self.lora_A, self.lora_B, self.scaling = None, None, 1.0

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(x)) * self.scaling
        return out

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


class UnetLora(Unet):
    def __init__(self, im_channels, model_config, lora_r=4, lora_alpha=16):
        super().__init__(im_channels, model_config)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.inject_lora()

    def inject_lora(self):
        # 예시: 주요 Linear 레이어에만 LoRA branch 적용
        # 실제로는 Attention, Conv2d 등에도 LoRA branch 적용 필요
        # 아래는 예시로 일부 레이어에만 처리
        def patch_lora(module):
            for name, child in module.named_children():
                # 예시: Linear에만 LoRA 부착
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child.in_features, child.out_features, self.lora_r, self.lora_alpha))
                else:
                    patch_lora(child)
        patch_lora(self)

    # forward는 기존 Unet과 동일하게 사용 가능

model = UnetLora(
    im_channels=im_channels,
    model_config=ldm_config,
    lora_r=4,
    lora_alpha=16
).to(device)

# 기존 .pth에는 LoRA 관련 weight가 없음
pretrained_dict = torch.load("./celebhq/ddpm_ckpt_text_cond_clip.pth", map_location="cpu")
model_dict = model.state_dict()

# 1. 이름/shape이 겹치는 weight만 불러옴
compatible_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(compatible_dict)
model.load_state_dict(model_dict)

for n, p in model.named_parameters():
    if 'lora_' not in n:
        p.requires_grad = False
lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
optimizer = torch.optim.Adam(lora_params, lr=1e-4)

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class MyFaceCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer):
        # caption_file: "my_face0.jpg|a photo of <me>\n..."
        self.samples = []
        with open(caption_file, encoding='utf-8') as f:
            for line in f:
                fname, caption = line.strip().split('|')
                self.samples.append((fname, caption))
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        return self.transform(img), caption

# 데이터셋 준비
image_dir = "./data/myfaceimages"
caption_file = "./data/captions.txt"  # 앞서 만든 파일 예시
dataset = MyFaceCaptionDataset(image_dir, caption_file, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# VQVAE, NoiseScheduler, 모델 선언 (YAML config 활용)
vae = VQVAE(
    im_channels=autoencoder_config['im_channels'],
    model_config=autoencoder_config
).to(device)
vae.eval()
vae.load_state_dict(torch.load("./celebhq/vqvae_autoencoder_ckpt.pth", map_location=device), strict=True)

scheduler = LinearNoiseScheduler(
    num_timesteps=config['diffusion_params']['num_timesteps'],
    beta_start=config['diffusion_params']['beta_start'],
    beta_end=config['diffusion_params']['beta_end']
)


# 5. 학습 루프 예시 (텍스트 임베딩 추출)
num_epochs = 10
for epoch in range(num_epochs):
    for imgs, captions in dataloader:
        imgs = imgs.to(device)
        # 텍스트 임베딩 추출
        tokens = tokenizer(list(captions), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            latents, _ = vae.encode(imgs)

        noise = torch.randn_like(latents).to(device)
        t = torch.randint(0, scheduler.num_timesteps, (latents.size(0),), device=device)
        noisy_latents = scheduler.add_noise(latents, noise, t)    
        tokens = tokenizer(list(captions), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            text_embeds = text_encoder(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state

        cond_input = {'text': text_embeds}
        noise_pred = model(noisy_latents, t, cond_input=cond_input)
        if noise_pred.shape != noise.shape:
            min_h = min(noise_pred.shape[2], noise.shape[2])
            min_w = min(noise_pred.shape[3], noise.shape[3])
            noise_pred = noise_pred[:, :, :min_h, :min_w]
            noise = noise[:, :, :min_h, :min_w]
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    print(f"Epoch {epoch+1} done")

# 본체(기존 UNet) freeze
for n, p in model.named_parameters():
    if 'lora_' not in n:
        p.requires_grad = False


# LoRA branch만 따로 저장하려면 (예시)
lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
torch.save(lora_state_dict, "lora_finetuned_weights.pth")

# 전체 저장도 가능
torch.save(model.state_dict(), "unet_with_lora_full.pth")