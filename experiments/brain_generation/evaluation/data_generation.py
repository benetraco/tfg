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


def generate_latent_images(pipe, vae, num_images=1, resolution=256):
    """Generate images using a Latent Diffusion Pipeline"""
    images = []
    for _ in range(num_images):
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed
        generator = torch.manual_seed(seed)
        latent_inf = pipe(batch_size=1, generator=generator).latent_dist.sample(generator=generator)
        latent_inf = latent_inf * vae.config.scaling_factor
        
        with torch.no_grad():
            reconstructed = vae.decode(latent_inf / vae.config.scaling_factor).sample
        images.append(reconstructed.squeeze(0).cpu())
    
    return images


def save_images(images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Count the number of images already in the folder to avoid overwriting
    current_images = len(os.listdir(folder_path))
    
    for i, image in enumerate(images):
        image.save(os.path.join(folder_path, f"image_{current_images + i}.png"))


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

    if model_type == "ddpm":
        # Load the DDPM pipeline
        pipe = DDPMPipeline.from_pretrained("benetraco/" + config['model_name'], torch_dtype=torch.float32)
        pipe.to("cuda")
        
        for i in range(num_batches):
            print(f"Generating batch {i + 1}/{num_batches} for DDPM")
            images = generate_ddpm_images(pipe, num_images=batch_size)
            save_images(images, os.path.join(exp_path, config['output_dir'], config['model_name']))

    elif model_type == "latent":
        # Load the VAE and UNet model
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda")
        vae.eval()
        
        pipe = DDPMPipeline.from_pretrained("benetraco/" + config['model_name'], torch_dtype=torch.float32)
        pipe.to("cuda")
        
        for i in range(num_batches):
            print(f"Generating batch {i + 1}/{num_batches} for Latent Diffusion")
            images = generate_latent_images(pipe, vae, num_images=batch_size, resolution=256)
            save_images(images, os.path.join(exp_path, config['output_dir'], config['model_name']))

    else:
        raise ValueError(f"Model type '{model_type}' not recognized. Use 'ddpm' or 'latent'.")
    
    print("All images generated and saved")


if __name__ == "__main__":
    main()
