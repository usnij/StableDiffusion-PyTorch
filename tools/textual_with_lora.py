import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.text_utils import *
import torch
import torch.nn as nn
import math
from models.unet_cond_lora import UnetWithLoRA
from models.lora import LoRALinear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    # For not fixing generation with one sample
    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    print("[DDPM 샘플러] xt shape:", xt.shape)  # torch.Size([1, 4, 32, 32])가 나와야 함

    ############ Create Conditional input ###############
    text_prompt = ['<me>']
    neg_prompt = ['He is a man.']
    empty_prompt = ['']
    text_prompt_embed = get_text_representation(text_prompt,
                                                text_tokenizer,
                                                text_model,
                                                device)
    print("me embedding (mean/std):", text_prompt_embed.mean().item(), text_prompt_embed.std().item())

    # Can replace empty prompt with negative prompt
    empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
    assert empty_text_embed.shape == text_prompt_embed.shape
    
    uncond_input = {
        'text': empty_text_embed
    }
    cond_input = {
        'text': text_prompt_embed
    }
    ###############################################
    
    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        #print(f"[Step {i}] noise_pred mean={noise_pred.mean().item()}, std={noise_pred.std().item()}, min={noise_pred.min().item()}, max={noise_pred.max().item()}")

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # 샘플링 루프 중간에
        #print(f"[Step {i}] xt mean={xt.mean().item()}, std={xt.std().item()}, min={xt.min().item()}, max={xt.max().item()}")
        #print(f"[Step {i}] x0_pred mean={x0_pred.mean().item()}, std={x0_pred.std().item()}")


        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final iamge to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=1)
        img = torchvision.transforms.ToPILImage()(grid)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'cond_text_samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'cond_text_samples'))
        img.save(os.path.join(train_config['task_name'], 'cond_text_samples', 'x0_{}.png'.format(i)))
        img.close()
    ##############################################################


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    text_tokenizer = None
    text_model = None
    
    ############# Validate the config #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for text conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types, ("This sampling script is for text conditional "
                                        "but no text condition found in config")
    validate_text_config(condition_config)
    ###############################################
    
    ############# Load tokenizer and text model #################
    with torch.no_grad():
        # Load tokenizer and text model based on config
        # Also get empty text representation
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                             ['text_embed_model'], device=device)
    ###############################################
    ldm_config = config['ldm_params']
    autoencoder_config = config['autoencoder_params']
    im_channels = autoencoder_config['z_channels']  
    ########## Load Unet #############
    model = Unet(
        im_channels=im_channels,
        model_config=ldm_config,
    ).to(device)

    apply_lora_to_linear(model, lora_r=4, lora_alpha=16)


    state_dict = torch.load("unet_with_lora_full.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    dummy_img = torch.randn(1, 3, 256, 256).to(device)
    vae.eval()
    
    with torch.no_grad():
        latent, _ = vae.encode(dummy_img)
    print("[VQVAE encode] latent shape:", latent.shape)
    # Load vae if found
    vae.load_state_dict(torch.load("./celebhq/vqvae_autoencoder_ckpt.pth", map_location=device), strict=True)

    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae,text_tokenizer, text_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with only '
                                                 'text conditioning')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond.yaml', type=str)
    args = parser.parse_args()
    infer(args)
