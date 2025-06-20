import torch
from transformers import CLIPProcessor
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os


def train_textual_inversion(placeholder_token, image_path, text_tokenizer, clip_model, output_path,
                             num_steps=1000, lr=5e-4, device='cuda'):
    
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
    new_embedding = nn.Parameter(torch.randn(1, embedding_dim).to(device) * 0.01)
    optimizer = optim.Adam([new_embedding], lr=lr)

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
