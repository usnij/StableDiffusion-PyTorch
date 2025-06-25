import os
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer
from models.vqvae import VQVAE
from models.unet_cond_base import Unet
from models.clip_encoder import CLIPTextEmbedder
from lora_utils import inject_lora_layers, get_lora_params
from models.textual_inversion import EmbeddingManager
from torchvision.utils import make_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_reconstruction(vqvae, noisy_z, epoch, save_dir="C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/finetunig/text_condition_sample"):
    os.makedirs(save_dir, exist_ok=True)
    vqvae.eval()
    with torch.no_grad():
        x_recon = vqvae.decode(noisy_z).clamp(0, 1)
    grid = transforms.ToPILImage()(torchvision.utils.make_grid(x_recon.cpu(), nrow=2))
    grid.save(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))

def save_text_conditioned_sample(vqvae, unet, text_encoder, tokenizer, epoch, save_dir , prompt, placeholder_token="*me*"):
    os.makedirs(save_dir, exist_ok=True)

    device = next(unet.parameters()).device
    vqvae.eval()
    unet.eval()

    # 1. 텍스트 임베딩
    input_ids = tokenizer([prompt], return_tensors="pt", padding=True).input_ids.to(device)
    with torch.no_grad():
        text_embed = text_encoder(input_ids)
        if text_embed.ndim == 2:
            text_embed = text_embed.unsqueeze(1)

        # 2. 랜덤 noise latent 생성
        z_shape = (1, 4, 32, 32)  # 보통 VQ-VAE latent shape
        noisy_z = torch.randn(z_shape).to(device)

        # 3. UNet으로 noise 제거 (t=0 assumed)
        t = torch.zeros(1, dtype=torch.long).to(device)
        noise_pred = unet(noisy_z, t, {'text': text_embed})
        denoised_z = noisy_z - noise_pred * 0.1  # 간단한 reverse step

        # 4. 디코딩
        recon = vqvae.decode(denoised_z).clamp(0, 1)

    # 5. 저장
    img = transforms.ToPILImage()(make_grid(recon.cpu(), nrow=1))
    img.save(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))


# ----------------------- 1. DATASET -------------------------
class FaceImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [os.path.join(folder, fname) for fname in os.listdir(folder)
                      if fname.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)

    def __len__(self):
        return len(self.paths)

# ----------------------- 2. TRAIN FUNCTION -------------------
def train():
    # [👈 여기에 경로를 직접 설정]
    image_dir = "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/data/myfaceimages"
    vqvae_ckpt = "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/celebhq/vqvae_autoencoder_ckpt.pth"
    unet_ckpt = "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/celebhq/ddpm_ckpt_text_cond_clip.pth"
    output_unet_path = "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/finetunig/unet_lora_finetuned.pth"
    output_token_path = "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/finetunig/textual_inversion_token.pt"
    batch_size = 4
    epochs = 100


    autoencoder_config = {
        'z_channels': 4,
        'codebook_size': 8192,
        'down_channels': [64, 128, 256, 256],
        'mid_channels': [256, 256],
        'down_sample': [True, True, True],
        'attn_down': [False, False, False],
        'norm_channels': 32,
        'num_heads': 4,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2
    }

    diffusion_config = {
        'down_channels': [256, 384, 512, 768],
        'mid_channels': [768, 512],
        'down_sample': [True, True, True],
        'attn_down': [True, True, True],
        'time_emb_dim': 512,
        'norm_channels': 32,
        'num_heads': 16,
        'conv_out_channels': 128,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
        'condition_config': {
            'condition_types': ['text'],
            'text_condition_config': {
                'text_embed_model': 'clip',
                'text_embed_dim': 512,
                'cond_drop_prob': 0.1
            }
        }
    }


    vqvae = VQVAE(im_channels=3, model_config=autoencoder_config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae.eval()

    unet = Unet(im_channels=4, model_config=diffusion_config).to(device)
    unet_ckpt_data = torch.load(unet_ckpt, map_location=device)
    unet.load_state_dict(unet_ckpt_data.get("unet", unet_ckpt_data))
    unet = inject_lora_layers(unet, r=4, alpha=1.0)
    for name, param in unet.named_parameters():
        param.requires_grad = 'lora_' in name
    unet.train()


    text_encoder = CLIPTextEmbedder().to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False

    placeholder_token = "*me*"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer.add_tokens([placeholder_token])
    text_encoder.text_model.resize_token_embeddings(len(tokenizer))

    embedding_manager = EmbeddingManager(text_encoder)
    embedding_manager.init_embedding(placeholder_token, tokenizer)
    embedding = embedding_manager.embedding
    embedding.requires_grad = True
    
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    dataset = FaceImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam([
        {"params": get_lora_params(unet), "lr": 1e-4},
        {"params": [embedding], "lr": 5e-4},
    ])

    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)

            with torch.no_grad():
                z, _ = vqvae.encode(imgs)
                z_q, _, _ = vqvae.quantize(z)

            input_ids = tokenizer([f"a photo of {placeholder_token}"] * imgs.size(0),
                                  return_tensors="pt", padding=True).input_ids.to(device)
            text_embed = text_encoder(input_ids)
            if text_embed.ndim == 2:
                text_embed = text_embed.unsqueeze(1)

            noise = torch.randn_like(z_q)
            noisy_z = z_q + noise * 0.1
            pred_noise = unet(noisy_z, torch.zeros(imgs.size(0), dtype=torch.long).to(device),
                              {'text': text_embed})
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss {loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            #save_reconstruction(vqvae, noisy_z, epoch + 1)
            torch.save(unet.state_dict(), output_unet_path)
            embedding_manager.save_embeddings(output_token_path)
            save_text_conditioned_sample(
                vqvae, unet, text_encoder, tokenizer,
                epoch + 1,
                "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/finetunig/text_condition_sample",
                prompt=f"a photo of {placeholder_token} wearing sunglasses"
            )

    print("[✅] Training complete. Saved LoRA + Textual Inversion.")


if __name__ == "__main__":
    train()
