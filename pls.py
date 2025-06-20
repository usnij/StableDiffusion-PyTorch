from diffusers import StableDiffusionPipeline, TextualInversionTrainer, TextualInversionTrainingArguments
import torch

# 1. 파이프라인 준비 (사전학습 Stable Diffusion)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# 2. 데이터셋 준비
# 내 얼굴 사진이 들어있는 폴더: ./data/myfaceimages
# 폴더에 jpg/png만 있으면 됨

# 3. 학습 파라미터 세팅
args = TextualInversionTrainingArguments(
    output_dir="./output/textual_inversion_me",
    placeholder_token="<me>",
    initializer_token="person",
    resolution=512,
    train_batch_size=1,
    max_train_steps=2000,
    learning_rate=5e-4,
    gradient_accumulation_steps=4,
    seed=42,
)

# 4. 트레이너 생성 & 학습
trainer = TextualInversionTrainer(
    args=args,
    train_data_dir="./data/myfaceimages",
    pipeline=pipe,
)
trainer.train()   # 🚀 이 라인만 돌리면 학습 시작

# 5. 결과물: ./output/textual_inversion_me/learned_embeds.bin
