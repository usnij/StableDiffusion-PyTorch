import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

from models.vqvae import VQVAE
from models.unet_cond_base import Unet
from models.clip_encoder import CLIPTextEmbedder
from models.textual_inversion import EmbeddingManager
from transformers import CLIPTokenizer
from lora_utils import inject_lora_layers

# --- 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae_ckpt = "celebhq/vqvae_autoencoder_ckpt.pth"
unet_ckpt = "finetunig/unet_lora_finetuned.pth"
embedding_path = "finetunig/textual_inversion_token.pt"
output_dir = "C:/Users/puzo9/Documents/GitHub/StableDiffusion-PyTorch/finetunig/infer"
os.makedirs(output_dir, exist_ok=True)

# --- 모델 설정 ---
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

# --- 모델 로드 ---
vqvae = VQVAE(im_channels=3, model_config=autoencoder_config).to(device)
vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
vqvae.eval()

unet = Unet(im_channels=autoencoder_config['z_channels'], model_config=diffusion_config)
unet = inject_lora_layers(unet)
unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
unet = unet.to(device).eval()

# --- 텍스트 임베딩 ---
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
tokenizer.add_tokens(["*me*"])
text_encoder = CLIPTextEmbedder().to(device)
embedding_manager = EmbeddingManager(text_encoder)
embedding_manager.load_embeddings(embedding_path)
text_encoder.text_model.resize_token_embeddings(len(tokenizer))

# --- 프롬프트 설정 ---
prompt = "a photo of *me* with sunglasses and smiling"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
text_embed = text_encoder(input_ids).mean(dim=1).unsqueeze(1)  # [1, 1, 512]

# --- 입력 이미지 로드 ---
img_path = "data/myfaceimages/KakaoTalk_20250618_005717122_01.jpg"
img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])
img_tensor = transform(img).unsqueeze(0).to(device)

# --- z 추출 및 noise 추가 ---
with torch.no_grad():
    z_gt, _ = vqvae.encode(img_tensor)
    z_q, _, _ = vqvae.quantize(z_gt)
    noise = torch.randn_like(z_q) * 0.1
    noisy_z = z_q + noise

    # Denoising
    t_tensor = torch.tensor([0], dtype=torch.long).to(device)
    pred_noise = unet(noisy_z, t_tensor, {'text': text_embed})
    denoised_z = noisy_z - pred_noise * 0.1

    # 복원
    x_gen = vqvae.decode(vqvae.quantize(denoised_z)[0]).clamp(0, 1)

# --- 저장 ---
grid = torchvision.utils.make_grid(x_gen.cpu(), nrow=1)
image = transforms.ToPILImage()(grid)
save_path = os.path.join(output_dir, "generated_from_face_with_prompt.jpg")
image.save(save_path)
print(f"[✅] 생성 이미지 저장됨 → {save_path}")
