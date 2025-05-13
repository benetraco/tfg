import torch
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Imports for CMMD
import sys
sys.path.append('/home/benet/scenic')
from scenic.projects.baselines.clip import model as clip
sys.path.append('/home/benet/tfg/experiments/brain_generation/evaluation/google-research')
import cmmd.main as cmmd_main

# Imports for FID
from pytorch_fid import fid_score

# Imports for LPIPS
module_path = Path().resolve() / "PerceptualSimilarity"
sys.path.append(str(module_path))
import lpips

# Define the compute_cmmd function
compute_cmmd = cmmd_main.compute_cmmd

# Load the LPIPS model
loss_fn = lpips.LPIPS(net='alex',version='0.1').cuda()

# Resolutions to evaluate
models = ['brain_ddpm_64', 'brain_ddpm_128', 'brain_ddpm_256', 'latent_scratch']
test_paths = ['64', '128', '256', '256']

# Containers for results
fid_scores = []
cmmd_scores = []
lpips_mean = []
lpips_std = []

# Create results folder if not exists
if not os.path.exists('evaluation_results'):
    os.makedirs('evaluation_results')

# CSV File Path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"evaluation_results/metrics_{timestamp}.csv"

# Write the headers
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "FID Score", "CMMD Score", "LPIPS mean", "LPIPS std"])

# Loop through models and calculate the metrics
for i, model in enumerate(models):
    print(f"\n--- Evaluation for {model} model ---")
    path1 = 'generated_images_orig/' + model #+ '/'
    path2 = 'test_images/' + test_paths[i] #+ '/'

    # Compute FID
    fid_value = fid_score.calculate_fid_given_paths([path1, path2],
                                                    batch_size=50,
                                                    device='cuda:0',
                                                    dims=2048)
    fid_scores.append(fid_value)
    print(f"FID score for {model} model: {fid_value}")

    # Compute CMMD
    cmmd_value = compute_cmmd(path1, path2, batch_size=32, max_count=-1)
    cmmd_scores.append(cmmd_value)
    print(f"CMMD score for {model} model: {cmmd_value}")

    # Compute LPIPS
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    distances = []

    for file1 in tqdm(files1, desc="Computing LPIPS distances"):
        for file2 in files2:
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(path1, file1))).cuda()
            img2 = lpips.im2tensor(lpips.load_image(os.path.join(path2, file2))).cuda()
            dist = loss_fn(img1, img2)
            distances.append(dist.item())
    # Convert to tensor for easier computation
    distances_tensor = torch.tensor(distances)
    avg_lpips = distances_tensor.mean().item()
    std_lpips = distances_tensor.std().item()
    lpips_mean.append(avg_lpips)
    lpips_std.append(std_lpips)
    print(f"LPIPS for {model} model: mean {avg_lpips}, std {std_lpips}")
    
    # Save to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model, fid_value, cmmd_value, avg_lpips, std_lpips])

print(f"\n✅ Evaluation complete! Results saved to: {csv_file}")

# Plot the results in subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# FID subplot
axes[0].plot(models, fid_scores, marker='o', color='blue', label='FID Score')
axes[0].set_title('FID Metrics Across Resolutions')
axes[0].set_xlabel('Resolution (px)')
axes[0].set_ylabel('FID Score')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# CMMD subplot
axes[1].plot(models, cmmd_scores, marker='s', color='green', label='CMMD Score')
axes[1].set_title('CMMD Metrics Across Resolutions')
axes[1].set_xlabel('Resolution (px)')
axes[1].set_ylabel('CMMD Score')
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

# LPIPS subplot
axes[2].errorbar(models, lpips_mean, yerr=lpips_std, fmt='o', color='orange', label='LPIPS Score')
axes[2].set_title('LPIPS Metrics Across Resolutions')
axes[2].set_xlabel('Resolution (px)')
axes[2].set_ylabel('LPIPS Score')
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()

# Set the main title for the entire figure
fig.suptitle('Evaluation Metrics for Generated Images', fontsize=16)
 
# Adjust layout and save
plt.tight_layout()
plt.savefig(f"evaluation_results/metrics_subplots_{timestamp}.png")
plt.show()
