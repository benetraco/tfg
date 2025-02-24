import wandb
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from fastcore.script import call_parse
from torchvision import transforms
from diffusers import DDPMPipeline, DDIMScheduler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Custom dataset class for grayscale images
class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.npy_files = [f for f in os.listdir(dataset_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.dataset_dir, self.npy_files[idx])
        image = np.load(npy_path)  # Load numpy array
        image = Image.fromarray(image.astype(np.uint8))  # Convert to PIL Image (grayscale)

        if self.transform:
            image = self.transform(image)

        return {"images": image}

@call_parse
def train(
    dataset_path="/home/benet/data/VH2D/npy/flair",  # Change to your dataset folder
    image_size=128,
    batch_size=4,
    grad_accumulation_steps=5,
    num_epochs=10,
    start_model="google/ddpm-bedroom-256",
    device="cuda",
    model_save_name="finetuned_model",
    wandb_project="dm_finetune",
    log_samples_every=50,
    save_model_every=100,
):
    # Initialize wandb for logging
    wandb.init(project=wandb_project, config=locals())

    # Load pretrained model
    image_pipe = DDPMPipeline.from_pretrained(start_model)
    image_pipe.to(device)

    # Get scheduler for sampling
    sampling_scheduler = DDIMScheduler.from_config(start_model)
    sampling_scheduler.set_timesteps(num_inference_steps=50)

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize grayscale
    ])

    # Load dataset
    dataset = CustomDataset(dataset_path, transform=preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer & LR scheduler
    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # Get clean images
            clean_images = batch["images"].to(device)

            # Ensure grayscale has 3 channels if necessary
            if clean_images.shape[1] == 1:
                clean_images = clean_images.repeat(1, 3, 1, 1)

            # Sample noise
            noise = torch.randn_like(clean_images).to(device)
            bs = clean_images.shape[0]

            # Sample random timestep
            timesteps = torch.randint(0, image_pipe.scheduler.num_train_timesteps, (bs,), device=device).long()

            # Add noise to images (Forward Diffusion)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            wandb.log({"loss": loss.item()})

            # Backpropagation
            loss.backward()

            # Gradient accumulation
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Occasionally log sample generations
            if (step + 1) % log_samples_every == 0:
                x = torch.randn(8, 3, image_size, image_size).to(device)
                for i, t in tqdm(enumerate(sampling_scheduler.timesteps), leave=False):
                    model_input = sampling_scheduler.scale_model_input(x, t)
                    with torch.no_grad():
                        noise_pred = image_pipe.unet(model_input, t)["sample"]
                    x = sampling_scheduler.step(noise_pred, t, x).prev_sample

                # Convert tensor to image for logging
                grid = torchvision.utils.make_grid(x, nrow=4)
                im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
                im = Image.fromarray((im.numpy() * 255).astype(np.uint8))
                wandb.log({"Sample generations": wandb.Image(im)})

            # Occasionally save model
            if (step + 1) % save_model_every == 0:
                image_pipe.save_pretrained(model_save_name + f"_step_{step+1}")

        # Update LR scheduler
        scheduler.step()

    # Final model save
    image_pipe.save_pretrained(model_save_name)

    # End wandb run
    wandb.finish()
