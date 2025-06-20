import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
import torch.nn.functional as F

from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel

# --- 1. 환경 및 경로 세팅 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dir = "./data/myfaceimages"      # <- 네 이미지 폴더로 수정
placeholder_token = "<me>"
initializer_token = "person"
prompt = f"a photo of {placeholder_token}"
num_epochs = 100
batch_size = 4
lr = 5e-4

# --- 2. 모델 및 토크나이저 로드 ---
clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
clip_text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)

# --- 3. 토큰 추가 및 임베딩 확장/초기화 ---
num_added = clip_tokenizer.add_tokens([placeholder_token])
if num_added > 0:
    clip_text_model.resize_token_embeddings(len(clip_tokenizer))

token_id = clip_tokenizer.encode([placeholder_token], add_special_tokens=False)[0]
init_token_id = clip_tokenizer.encode([initializer_token], add_special_tokens=False)[0]
with torch.no_grad():
    clip_text_model.get_input_embeddings().weight[token_id] = clip_text_model.get_input_embeddings().weight[init_token_id]

embed_weight = clip_text_model.get_input_embeddings().weight
optimizer = Adam([embed_weight], lr=lr)

# --- 4. 데이터셋/로더 준비 ---
class MyFaceDataset(Dataset):
    def __init__(self, img_dir, prompt):
        self.imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.prompt = prompt
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),      # CLIP 기본 해상도
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        return self.transform(img), self.prompt

dataset = MyFaceDataset(img_dir, prompt)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 학습 루프에서 "내 토큰"만 gradient가 남도록 직접 gradient mask를 적용!
for epoch in range(num_epochs):
    for imgs, prompts in dataloader:
        imgs = imgs.to(device)
        image_embeds = clip_model.get_image_features(pixel_values=imgs)
        inputs = clip_tokenizer([prompt]*imgs.size(0), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        text_embeds = clip_text_model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        text_embeds = text_embeds.mean(dim=1)
        loss = 1 - F.cosine_similarity(image_embeds, text_embeds).mean()

        optimizer.zero_grad()
        loss.backward()

        # --- 여기서 gradient mask 적용 (내 토큰만 업데이트) ---
        grad = embed_weight.grad
        mask = torch.zeros_like(grad)
        mask[token_id] = 1
        grad.mul_(mask)
        # --------------------------------------------------

        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}  Loss: {loss.item():.4f}")

# --- 6. 임베딩 저장 ---
save_path = "learned_embeds.bin"
torch.save({placeholder_token: embed_weight[token_id].detach().cpu()}, save_path)
print(f"Learned embedding saved at {save_path}")
