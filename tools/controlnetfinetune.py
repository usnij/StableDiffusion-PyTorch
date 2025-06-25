import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import torch
from models.unet_cond_controlnet import ControlNetUnet

from dataset.celeb_dataset import CelebDataset
from models.unet_cond_base import Unet  # ControlNet이 별도로 구현됐다면 교체
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.diffusion_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def migrate_unet_to_controlnet(unet, controlnet):
    """
    기존 Unet의 state_dict를 ControlNet 구조로 복사 (겹치는 key만)
    """
    unet_state = unet.state_dict()
    controlnet_state = controlnet.state_dict()
    for k in controlnet_state:
        if k in unet_state and controlnet_state[k].shape == unet_state[k].shape:
            controlnet_state[k] = unet_state[k]
    controlnet.load_state_dict(controlnet_state)
    return controlnet


def fine_tune(args):
    # 1. Config 불러오기
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # 2. Noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    # 3. Dataset/DataLoader 준비 (semantic mask 조건)
    condition_config = diffusion_model_config.get('condition_config', None)
    assert condition_config is not None and 'image' in condition_config['condition_types']
    im_dataset = CelebDataset(
        split='train',
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        use_latents=True,
        latent_path=os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name']),
        condition_config=condition_config
    )
    data_loader = DataLoader(im_dataset, batch_size=train_config['ldm_batch_size'], shuffle=True)

    # 4. 모델 인스턴스 준비 (Unet or ControlNetUnet)
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
    ).to(device)
    model.train()

    # **5. 기존 학습된 weight 불러오기!**
    # 반드시 기존에 저장된 경로가 맞는지 확인할 것
    ckpt_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
    print(f"Loading pretrained model weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    controlnet = ControlNetUnet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
    ).to(device)
    controlnet.train()

    controlnet = migrate_unet_to_controlnet(model, controlnet)
    # 6. VAE 준비 (latent가 없는 경우만)
    vae = None
    if not im_dataset.use_latents:
        vae = VQVAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae_ckpt = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
        if os.path.exists(vae_ckpt):
            print("comple")
            vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        else:
            raise Exception('VAE checkpoint not found')
        for param in vae.parameters():
            param.requires_grad = False
    
    for name, param in controlnet.named_parameters():
        if not name.startswith('control_'):  # main branch는 freeze
            param.requires_grad = False

    
    # 7. Optimizer (파인튜닝이니까 learning rate 낮게 해도 됨)
    optimizer = Adam(controlnet.parameters(), lr=5e-6)
    criterion = torch.nn.MSELoss()
    num_epochs = train_config.get('finetune_epochs', 5)  # config에 finetune_epochs 옵션 추가 권장

    # 9. Fine-tuning Loop (ControlNet 기준으로!)
    for epoch_idx in range(num_epochs):
        losses = []
        for batch in tqdm(data_loader):
            x, cond_input = batch
            optimizer.zero_grad()
            x = x.float().to(device)
            mask = cond_input['image'].to(device)

            # Latent가 없는 경우 VAE 인코딩
            if not im_dataset.use_latents and vae is not None:
                with torch.no_grad():
                    x, _ = vae.encode(x)
            # Noise 및 timestep 샘플
            noise = torch.randn_like(x).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (x.shape[0],)).to(device)
            noisy_x = scheduler.add_noise(x, noise, t)

            # 마스크 컨디션을 넣어서 예측 (ControlNetUnet으로!)

            noise_pred = controlnet(noisy_x, t, cond_input={'image': mask})
            # noise_pred: (B, 4, h1, w1), noise: (B, 4, h2, w2)일 수 있음
            if noise_pred.shape != noise.shape:
                # 채널(B, C)는 반드시 같아야 하고, spatial shape만 resize
                noise_pred = torch.nn.functional.interpolate(
                    noise_pred, size=noise.shape[-2:], mode='bilinear', align_corners=False
                )

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'[Fine-tune Epoch {epoch_idx+1}/{num_epochs}] Loss: {np.mean(losses):.4f}')
        # 체크포인트 저장 (ControlNet용 파일명으로!)
        torch.save(controlnet.state_dict(), os.path.join(
            train_config['task_name'], f'finetune_controlnet_{train_config["ldm_ckpt_name"]}'))

    print('Fine-tuning done!')

if __name__ == '__main__':
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU NOT USED!!!")
    parser = argparse.ArgumentParser(description='Fine-tune DDPM model with ControlNet and semantic mask')
    parser.add_argument('--config', dest='config_path', default='config/celebhq_mask_cond.yaml', type=str)
    args = parser.parse_args()
    fine_tune(args)