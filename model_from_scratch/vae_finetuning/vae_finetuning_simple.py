import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
import wandb
import os
from PIL import Image


# Custom Dataset
class MRIDataset3ch(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # Load as greyscale
        image = image.convert("RGB")  # Convert to 3-channel RGB
        if self.transform:
            image = self.transform(image)
        return image

# Configuration
image_size = 256
batch_size = 16
epochs = 20
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = "/home/benet/data/VH2D/images/flair"
save_path = "vae_checkpoints"
os.makedirs(save_path, exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load dataset
image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".png")]
dataset = MRIDataset3ch(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)





# Load pretrained VAE
# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)

vae.train()

# Optimizer
optimizer = optim.AdamW(vae.parameters(), lr=lr)
criterion = nn.MSELoss()

# Initialize Weights & Biases
wandb.init(project="vae-fine-tuning", config={"epochs": epochs, "batch_size": batch_size, "lr": lr})

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        latents = vae.encode(batch).latent_dist.sample()
        reconstructions = vae.decode(latents).sample
        loss = criterion(reconstructions, batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss})
    
    # Save checkpoint
    torch.save(vae.state_dict(), os.path.join(save_path, f"vae_epoch_{epoch+1}.pth"))

vae.save_pretrained(save_path)  # This ensures a proper config.json is created
wandb.finish()
print("Training completed!")
