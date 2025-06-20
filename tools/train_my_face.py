import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor

from models.vqvae import VQVAE
from models.unet_cond_lora import UnetWithLoRA
from dataset.celeb_dataset import CelebDataset
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value
from utils.diffusion_utils import drop_text_condition


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image
from torch.utils.data import Dataset
import os

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(caption_file, 'r') as f:
            for line in f:
                image_name, caption = line.strip().split('|')
                self.samples.append((image_name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'caption': caption
        }



def get_tokenizer_and_model(model_type, device):
    if model_type == 'clip':
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        return tokenizer, model
    else:
        raise ValueError("Only 'clip' model_type is supported.")


def train_textual_inversion(placeholder_token, image_path, text_tokenizer,
                             clip_model, output_path, num_steps=1000, lr=5e-4, device='cuda'):
    if placeholder_token not in text_tokenizer.get_vocab():
        text_tokenizer.add_tokens([placeholder_token])
        print(f"Added new token: {placeholder_token}")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)

    clip_model.eval()
    with torch.no_grad():
        target_features = clip_model.get_image_features(**image_inputs)

    embedding_dim = target_features.shape[-1]
    new_embedding = torch.nn.Parameter(torch.randn(1, embedding_dim).to(device) * 0.01)
    optimizer = torch.optim.Adam([new_embedding], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = 1 - torch.cosine_similarity(new_embedding, target_features, dim=-1).mean()
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"[TextInv Step {step+1}] Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_embedding.detach().cpu(), output_path)
    print(f"Saved embedding to: {output_path}")


def collect_lora_parameters(model):
    return [param for name, param in model.named_parameters() if "lora" in name and param.requires_grad]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/me_finetune.yaml')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)


    train_config = config['train_params']
    diffusion_config = config['diffusion_params']
    autoencoder_config = config['autoencoder_params']
    dataset_config = config['dataset_params']
    ldm_config = config['ldm_params']
    condition_config = ldm_config['condition_config']
    text_condition = condition_config['text_condition_config']

    placeholder_token = text_condition['placeholder_token']
    embedding_path = text_condition['embedding_path']
    image_path = text_condition['image_path']

    text_tokenizer, text_model = get_tokenizer_and_model(text_condition['text_embed_model'], device)
    train_textual_inversion(placeholder_token, image_path, text_tokenizer, text_model, embedding_path)
    learned_embedding = torch.load(embedding_path).to(device)

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name']), map_location=device))
    vae.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ImageTextDataset(
        image_dir=config['dataset_params']['im_path'],
        caption_file=config['dataset_params']['caption_file'],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UnetWithLoRA(im_channels=autoencoder_config['z_channels'], model_config=ldm_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']), map_location=device))
    model.train()

    optimizer = Adam(collect_lora_parameters(model), lr=5e-4)

    for step, (img,) in enumerate(tqdm(dataloader)):
        if step >= 500: break
        optimizer.zero_grad()
        img = img.to(device).float()
        with torch.no_grad():
            z, _ = vae.encode(img)
        noise = torch.randn_like(z).to(device)
        t = torch.randint(0, diffusion_config['num_timesteps'], (1,), dtype=torch.long).to(device)

        cond_input = {'text': learned_embedding.expand(z.size(0), -1)}
        noisy_z = scheduler.add_noise(z, noise, t)
        noise_pred = model(noisy_z, t, cond_input)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"[LoRA Step {step+1}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(train_config['task_name'], 'lora_finetuned_unet.pt'))
    print("LoRA fine-tuning complete.")


if __name__ == '__main__':
    main()
