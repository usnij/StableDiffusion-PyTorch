import torch
from diffusers import DiffusionPipeline, AutoencoderTiny
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
)
vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


# 2. 이미지 불러오기 및 전처리
image = Image.open("1.jpg").convert("RGB")
transform = T.Compose([
    T.Resize(512),
    T.CenterCrop(512),
    T.ToTensor(),
])
img_tensor = transform(image).unsqueeze(0).to(pipe.vae.device)

if next(pipe.vae.parameters()).dtype == torch.float16:
    img_tensor = img_tensor.half()

# 3. Encode → Decode
with torch.no_grad():
    latents = pipe.vae.encode(img_tensor).latents
    recon = pipe.vae.decode(latents)
    out = recon.sample if hasattr(recon, 'sample') else recon
    out = out.clamp(0, 1)
    T.ToPILImage()(out[0].cpu()).save("taesd_recon.png")

# 4. 시각화
recon_img = recon[0].detach().cpu().permute(1, 2, 0).numpy()
input_img = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

