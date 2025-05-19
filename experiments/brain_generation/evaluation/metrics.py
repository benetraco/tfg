import os, sys
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# CMMD
sys.path.append('/home/benet/scenic')
from scenic.projects.baselines.clip import model as clip
sys.path.append('/home/benet/tfg/experiments/brain_generation/evaluation/google-research')
import cmmd.main as cmmd_main
compute_cmmd = cmmd_main.compute_cmmd

# FID
from pytorch_fid import fid_score

# LPIPS
module_path = Path().resolve() / "PerceptualSimilarity"
sys.path.append(str(module_path))
import lpips
loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda()

# Settings
models = ['brain_ddpm_64', 'brain_ddpm_128', 'brain_ddpm_256', 'latent_scratch']
test_paths = ['64', '128', '256', '256']
num_samples = 234
num_rounds = 5

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("evaluation_results")
results_dir.mkdir(exist_ok=True)
csv_file = results_dir / f"robust_metrics_{timestamp}.csv"

# CSV Header
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "FID mean", "FID std", "CMMD mean", "CMMD std", "LPIPS mean", "LPIPS std"])

# Store for plotting
fid_means, fid_stds = [], []
cmmd_means, cmmd_stds = [], []
lpips_means, lpips_stds = [], []

for i, model in enumerate(models):
    print(f"\n🔍 Evaluating {model}")
    path_gen = Path("generated_images_orig") / model
    path_test = Path("test_images") / test_paths[i]

    files_gen = sorted(os.listdir(path_gen))
    files_test = sorted(os.listdir(path_test))

    assert len(files_test) == num_samples, "Test set must have 234 images"

    fids, cmmds, lpips_vals = [], [], []

    for round_num in range(num_rounds):
        print(f"Round {round_num+1}/{num_rounds}")

        # Random subsample
        subset_gen = random.sample(files_gen, num_samples)
        tmp_subfolder = path_gen / f"subsample_{round_num}"
        tmp_subfolder.mkdir(exist_ok=True)

        for idx, fname in enumerate(subset_gen):
            src = path_gen / fname
            dst = tmp_subfolder / f"{idx}.png"
            if not dst.exists():
                os.system(f"cp '{src}' '{dst}'")  # Fast copy

        # FID
        fid_val = fid_score.calculate_fid_given_paths([str(tmp_subfolder), str(path_test)],
                                                      batch_size=50, device='cuda:0', dims=2048)
        fids.append(fid_val)

        # CMMD
        cmmd_val = compute_cmmd(str(tmp_subfolder), str(path_test), batch_size=32, max_count=-1)
        cmmds.append(cmmd_val.item() if hasattr(cmmd_val, 'item') else cmmd_val)

        # LPIPS
        lpips_dist = []
        for file1, file2 in zip(sorted(os.listdir(tmp_subfolder)), files_test):
            img1 = lpips.im2tensor(lpips.load_image(str(tmp_subfolder / file1))).cuda()
            img2 = lpips.im2tensor(lpips.load_image(str(path_test / file2))).cuda()
            lpips_dist.append(loss_fn(img1, img2).item())
        lpips_vals.append(torch.tensor(lpips_dist).mean().item())

        # Clean subsample folder
        for file in tmp_subfolder.iterdir():
            file.unlink()
        tmp_subfolder.rmdir()

    # Aggregate stats
    fid_mean, fid_std = torch.tensor(fids).mean().item(), torch.tensor(fids).std().item()
    cmmd_mean, cmmd_std = torch.tensor(cmmds).mean().item(), torch.tensor(cmmds).std().item()
    lpips_mean, lpips_std = torch.tensor(lpips_vals).mean().item(), torch.tensor(lpips_vals).std().item()

    fid_means.append(fid_mean)
    fid_stds.append(fid_std)
    cmmd_means.append(cmmd_mean)
    cmmd_stds.append(cmmd_std)
    lpips_means.append(lpips_mean)
    lpips_stds.append(lpips_std)

    # Write CSV row
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model, fid_mean, fid_std, cmmd_mean, cmmd_std, lpips_mean, lpips_std])

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].errorbar(models, fid_means, yerr=fid_stds, fmt='o', color='blue', label='FID')
axes[0].set_title("FID Score (mean ± std)")
axes[0].set_ylabel("FID")
axes[0].grid(True)

axes[1].errorbar(models, cmmd_means, yerr=cmmd_stds, fmt='o', color='green', label='CMMD')
axes[1].set_title("CMMD Score (mean ± std)")
axes[1].set_ylabel("CMMD")
axes[1].grid(True)

axes[2].errorbar(models, lpips_means, yerr=lpips_stds, fmt='o', color='orange', label='LPIPS')
axes[2].set_title("LPIPS Score (mean ± std)")
axes[2].set_ylabel("LPIPS")
axes[2].grid(True)

for ax in axes:
    ax.set_xlabel("Model")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15)
    ax.legend()

plt.tight_layout()
plt.savefig(results_dir / f"robust_metrics_plot_{timestamp}.png")
plt.show()

print(f"\n✅ Robust evaluation complete. Results saved to {csv_file}")
