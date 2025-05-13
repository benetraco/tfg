# Script that generates data for the evaluation of the brain generation model.
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Script that generates data for the evaluation of the brain generation model.

import os
import sys
import random
import yaml
import torch
from pathlib import Path
from diffusers import DDPMPipeline, AutoencoderKL, UNet2DModel, DDPMScheduler
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

# Path configurations
repo_path = Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path):  # while not in the root of the repo
    repo_path = repo_path.parent  # go up one level
sys.path.insert(0, str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve()

print(f"Repo Path: {repo_path}")
print(f"Experiment Path: {exp_path}")


def generate_ddpm_images(pipe, num_images=1):
    """Generate images using a DDPM Pipeline"""
    seed = random.randint(0, 2**32 - 1)  # Generate a random seed
    generator = torch.manual_seed(seed)
    images = pipe(batch_size=num_images, generator=generator).images
    return images


def generate_latent_images(model, noise_scheduler, vae, device, num_images=1, latent_resolution=32):
    """Generate images using a Latent Diffusion Pipeline"""

    # denoise images
    latent_inf = torch.randn(num_images,4,latent_resolution,latent_resolution).to(device)
    latent_inf *= noise_scheduler.init_noise_sigma # init noise is 1.0 in vanilla case
    # markov chain
    for t in tqdm(noise_scheduler.timesteps): # markov chain
        latent_inf = noise_scheduler.scale_model_input(latent_inf, t) # # Apply scaling, no change in vanilla case
        with torch.no_grad(): # predict the noise residual with the unet
            noise_pred = model(latent_inf, t).sample
        latent_inf = noise_scheduler.step(noise_pred, t, latent_inf).prev_sample # compute the previous noisy sample x_t -> x_t-1

    original_latent = latent_inf.clone() # save the original latent for later use

    # load vae    
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.requires_grad_(False)
    # send vae to accelerator
    vae.to(device)

    # first we unscale
    latent_inf = latent_inf/ vae.config.scaling_factor
    # decode
    with torch.no_grad():
        im = vae.decode(latent_inf).sample

    # select just the first channel (RGB for vae)
    im = im[:, 0, :, :].unsqueeze(1)

    return im, original_latent

def save_images(images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Count the number of images already in the folder to avoid overwriting
    current_images = len(os.listdir(folder_path))
    
    for i, image in enumerate(images):
        # if the image is a tensor, unormalize it and convert it to a PIL image
        if isinstance(image, torch.Tensor):
            image = (image + 1) / 2  # Scale to [0, 1]
            image = image.clamp(0, 1)  # Clamp to [0, 1]
            image = to_pil_image(image)  # Convert to PIL image
                
        
        image.save(os.path.join(folder_path, f"image_{current_images + i}.png"))

# def save_latent_images(images, folder_path):
#     # show latents
#     fig, ax = plt.subplots(1,4, figsize=(10,5))
#     for i in range(4):
#         ax[i].imshow(latent_inf[0,i].cpu().numpy(), cmap='gray')
#         ax[i].set_title(f'Latent {i}')

def main():
    ### General setups
    # load the config file
    config_path = exp_path / 'config.yaml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load the model type (ddpm or latent)
    model_type = config['model_type'].lower()
    num_images = config['num_images']
    batch_size = config['batch_size']
    num_batches = (num_images + batch_size - 1) // batch_size

    print(f"Model type: {model_type}")
    print(f"Generating {num_images} images in {num_batches} batches of size {batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "ddpm":
        # Load the DDPM pipeline
        pipe = DDPMPipeline.from_pretrained("benetraco/" + config['model_name'], torch_dtype=torch.float32)
        pipe.to(device)
        
        for i in range(num_batches):
            print(f"Generating batch {i + 1}/{num_batches} for DDPM")
            images = generate_ddpm_images(pipe, num_images=batch_size)
            save_images(images, os.path.join(exp_path, config['output_dir'], config['model_name']))

    elif model_type == "latent":
        # Load the VAE and UNet model
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        vae.requires_grad_(False)
        vae.to(device)
        
        pipe = DDPMPipeline.from_pretrained("benetraco/" + config['model_name'], torch_dtype=torch.float32)
        model = pipe.unet
        model.to(device)
        noise_scheduler = pipe.scheduler
        
        for i in range(num_batches):
            print(f"Generating batch {i + 1}/{num_batches} for Latent Diffusion")
            images, latents = generate_latent_images(model, noise_scheduler, vae, device, num_images=batch_size)
            save_images(images, os.path.join(exp_path, config['output_dir'], config['model_name']))

    else:
        raise ValueError(f"Model type '{model_type}' not recognized. Use 'ddpm' or 'latent'.")
    
    print("All images generated and saved")


if __name__ == "__main__":
    main()
