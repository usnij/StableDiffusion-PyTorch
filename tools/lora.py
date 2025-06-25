# 전체 코드 예시
import torch
from models.unet_cond_base import Unet
import yaml
from transformers import CLIPTokenizer, CLIPTextModel
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from models.vqvae import VQVAE
from models.unet_cond_lora import UnetWithLoRA
import numpy as np
import torch
import torch.nn as nn
import math
import os
from models.lora import LoRALinear

def apply_lora_to_linear(module, lora_r, lora_alpha):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            # LoRALinear 생성 (bias는 child의 bias 옵션 반영)
            lora_layer = LoRALinear(child.in_features, child.out_features, lora_r, lora_alpha, bias=(child.bias is not None))
            # weight/bias 복사
            lora_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lora_layer.bias.data = child.bias.data.clone()
            setattr(module, name, lora_layer)
        else:
            apply_lora_to_linear(child, lora_r, lora_alpha)


def freeze_except_lora(module):
    for name, param in module.named_parameters():
        # 'lora_'라는 이름이 붙어있으면 True (LoRA만 학습)
        if 'lora_' in name or 'lora_down' in name or 'lora_up' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터 수: {total:,}")
    print(f"학습되는(gradient 켜진) 파라미터 수: {trainable:,}")
    print(f"Frozen 비율: {100 * (1 - trainable/total):.2f}%")
    return total, trainable



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


model = Unet(
    im_channels=im_channels,
    model_config=ldm_config,
).to(device)

# 기존 .pth에는 LoRA 관련 weight가 없음
pretrained_dict = torch.load("./celebhq/ddpm_ckpt_text_cond_clip.pth", map_location="cpu")
model.load_state_dict(pretrained_dict, strict=False)  # 100% 동일해야 함

# 이후 원하는 부분에 LoRA patch 적용
apply_lora_to_linear(model, lora_r=4, lora_alpha=16)
freeze_except_lora(model)

# 사용 예시
print_trainable_params(model)

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
num_epochs = 5
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