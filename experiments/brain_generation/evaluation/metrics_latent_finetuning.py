import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import sys
import csv
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# CMMD
sys.path.append('/home/benet/scenic')
from scenic.projects.baselines.clip import model as clip
sys.path.append('/home/benet/tfg/experiments/brain_generation/evaluation/google-research')
import cmmd.main as cmmd_main
compute_cmmd = cmmd_main.compute_cmmd

# LPIPS
module_path = Path().resolve() / "PerceptualSimilarity"
sys.path.append(str(module_path))
import lpips
loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda()

# FID
from pytorch_fid import fid_score

# Config
datasets = ["VH", "SHIFTS", "WMH2017"]
guidance_values = ['g0.0', 'g1.0', 'g2.0', 'g3.0', 'g5.0', 'g7.0', 'g10.0']
num_images = 234  # Number of generated/test images to compare

# Paths
base_path = Path("generated_images/latent_finetuning_train_embeddings")
test_base_path = Path("test_images")
output_dir = Path("evaluation_results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = output_dir / f"guidance_eval_{timestamp}.csv"

# CSV header
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['Dataset', 'Guidance', 'FID', 'CMMD', 'LPIPS_mean', 'LPIPS_std'])
    writer.writerow(['Dataset', 'Guidance', 'FID', 'CMMD', 'LPIPS'])

# Storage for plots
# results = {ds: {'guidance': [], 'fid': [], 'cmmd': [], 'lpips_mean': [], 'lpips_std': []} for ds in datasets}
results = {ds: {'guidance': [], 'fid': [], 'cmmd': [], 'lpips': []} for ds in datasets}

# Loop through datasets and guidance levels
for ds in datasets:
    test_path = test_base_path / ds
    test_all_imgs = sorted(os.listdir(test_path))

    for g in guidance_values:
        gen_path = base_path / ds / g
        gen_imgs = sorted(os.listdir(gen_path))

        # Subsample num_images test images randomly
        test_imgs = random.sample(test_all_imgs, num_images)

        print(f"\n🔍 Evaluating {ds} - {g} with {num_images} random test samples")

        # TEMP DIR for FID/CMMD with selected test images
        temp_test_dir = output_dir / f"temp_{ds}_{g}"
        temp_test_dir.mkdir(exist_ok=True)
        for i, fname in enumerate(test_imgs):
            src = test_path / fname
            dst = temp_test_dir / f"{i}.png"
            os.system(f"cp '{src}' '{dst}'")

        # --- FID ---
        fid = fid_score.calculate_fid_given_paths([str(gen_path), str(temp_test_dir)],
                                                  batch_size=50, device='cuda:0', dims=2048)

        # --- CMMD ---
        cmmd = compute_cmmd(str(gen_path), str(temp_test_dir), batch_size=32, max_count=50)
        cmmd = cmmd.item() if hasattr(cmmd, 'item') else cmmd

        # --- LPIPS (1-to-1) ---
        distances = []
        for file1, file2 in zip(gen_imgs, test_imgs):
            img1 = lpips.im2tensor(lpips.load_image(str(gen_path / file1))).cuda()
            img2 = lpips.im2tensor(lpips.load_image(str(test_path / file2))).cuda()
            distances.append(loss_fn(img1, img2).item())
        lpips_mean = torch.tensor(distances).mean().item()
        # lpips_std = torch.tensor(distances).std().item()

        # --- Cleanup temp dir ---
        for f in temp_test_dir.iterdir():
            f.unlink()
        temp_test_dir.rmdir()

        # --- Store results ---
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow([ds, g, fid, cmmd, lpips_mean, lpips_std])
            writer.writerow([ds, g, fid, cmmd, lpips_mean])

        results[ds]['guidance'].append(g)
        results[ds]['fid'].append(fid)
        results[ds]['cmmd'].append(cmmd)
        results[ds]['lpips'].append(lpips_mean)
        # results[ds]['lpips_mean'].append(lpips_mean)
        # results[ds]['lpips_std'].append(lpips_std)

print(f"\n✅ Evaluation complete! Results saved to: {csv_path}")

# --- Plotting ---
for metric in ['fid', 'cmmd', 'lpips']:
    plt.figure(figsize=(10, 5))
    for ds in datasets:
        # y_vals = results[ds][f'{metric}_mean'] if metric == 'lpips' else results[ds][metric]
        # y_errs = results[ds]['lpips_std'] if metric == 'lpips' else None
        # plt.errorbar(results[ds]['guidance'], y_vals, yerr=y_errs if metric == 'lpips' else None,
        #              label=ds, marker='o', capsize=5)
        plt.plot(results[ds]['guidance'], results[ds][metric], label=ds, marker='o')

    plt.title(f"{metric.upper()} Score Across Guidance Values")
    plt.xlabel("Guidance Value")
    plt.ylabel(f"{metric.upper()}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_guidance_plot_{timestamp}.png")
    plt.show()
