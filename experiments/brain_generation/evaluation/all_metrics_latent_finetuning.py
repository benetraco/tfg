import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# datasets = ["VH", "SHIFTS", "WMH2017"]
datasets = ["Siemens", "Philips", "GE"]
guidance_values = ['g0.0', 'g1.0', 'g2.0', 'g3.0', 'g5.0'] #, 'g7.0', 'g10.0']
num_images = 25  # Number of generated/test images to compare

# Paths
base_path = Path("generated_images/latent_finetuning_scanners_healthy")
test_base_path = Path("test_images")
output_dir = Path("evaluation_results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = output_dir / f"guidance_eval_{timestamp}.csv"

# ... [imports and initial setup unchanged] ...

# Updated results storage and CSV header
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['GenPrompt', 'TestDataset', 'Guidance', 'FID', 'CMMD', 'LPIPS_mean', 'LPIPS_std'])
    writer.writerow(['GenPrompt', 'TestDataset', 'Guidance', 'FID', 'CMMD', 'LPIPS'])

results = {}
for gen_ds in datasets:
    for test_ds in datasets:
        key = f"{gen_ds}_vs_{test_ds}"
        # results[key] = {'guidance': [], 'fid': [], 'cmmd': [], 'lpips_mean': [], 'lpips_std': []}
        results[key] = {'guidance': [], 'fid': [], 'cmmd': [], 'lpips': []}

# Evaluate every combination
for gen_ds in datasets:
    for test_ds in datasets:
        test_path = test_base_path / test_ds
        test_all_imgs = sorted(os.listdir(test_path))
        test_all_imgs = [img for img in test_all_imgs if img.endswith('.png') or img.endswith('.jpg')]

        for g in guidance_values:
            gen_path = base_path / gen_ds / g
            gen_imgs = sorted(os.listdir(gen_path))
            test_imgs = random.sample(test_all_imgs, num_images)

            print(f"\n🔍 Evaluating {gen_ds}→{test_ds} - {g}")

            # TEMP DIR for selected test samples
            temp_test_dir = output_dir / f"temp_{gen_ds}_{test_ds}_{g}"
            temp_test_dir.mkdir(exist_ok=True)
            for i, fname in enumerate(test_imgs):
                os.system(f"cp '{test_path / fname}' '{temp_test_dir / f'{i}.png'}'")

            # FID
            fid = fid_score.calculate_fid_given_paths([str(gen_path), str(temp_test_dir)],
                                                      batch_size=25, device='cuda:0', dims=2048)

            # CMMD
            cmmd = compute_cmmd(str(gen_path), str(temp_test_dir), batch_size=25, max_count=50)
            cmmd = cmmd.item() if hasattr(cmmd, 'item') else cmmd

            # LPIPS
            distances = []
            for file1, file2 in zip(gen_imgs, test_imgs):
                img1 = lpips.im2tensor(lpips.load_image(str(gen_path / file1))).cuda()
                img2 = lpips.im2tensor(lpips.load_image(str(test_path / file2))).cuda()
                distances.append(loss_fn(img1, img2).item())
            lpips_mean = torch.tensor(distances).mean().item()
            # lpips_std = torch.tensor(distances).std().item()

            # Cleanup
            for f in temp_test_dir.iterdir():
                f.unlink()
            temp_test_dir.rmdir()

            # Store and log
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                # writer.writerow([gen_ds, test_ds, g, fid, cmmd, lpips_mean, lpips_std])
                writer.writerow([gen_ds, test_ds, g, fid, cmmd, lpips_mean])

            results[f"{gen_ds}_vs_{test_ds}"]['guidance'].append(g)
            results[f"{gen_ds}_vs_{test_ds}"]['fid'].append(fid)
            results[f"{gen_ds}_vs_{test_ds}"]['cmmd'].append(cmmd)
            results[f"{gen_ds}_vs_{test_ds}"]['lpips'].append(lpips_mean)
            # results[f"{gen_ds}_vs_{test_ds}"]['lpips_mean'].append(lpips_mean)
            # results[f"{gen_ds}_vs_{test_ds}"]['lpips_std'].append(lpips_std)

print(f"\n✅ Cross-prompt evaluation complete! Results saved to: {csv_path}")

# --- Plotting ---
for metric in ['fid', 'cmmd', 'lpips']:
    plt.figure(figsize=(15, 8))
    for gen_ds in datasets:
        for test_ds in datasets:
            key = f"{gen_ds}_vs_{test_ds}"
            label = f"{gen_ds}→{test_ds}"
            # y_vals = results[key][f'{metric}_mean'] if metric == 'lpips' else results[key][metric]
            # y_errs = results[key]['lpips_std'] if metric == 'lpips' else None
            # plt.errorbar(results[key]['guidance'], y_vals, yerr=y_errs if metric == 'lpips' else None,
            #              label=label, marker='o', capsize=5)
            plt.plot(results[key]['guidance'], results[key][metric], label=label, marker='o')

    plt.title(f"{metric.upper()} Score Across Guidance Values (All Prompt/Test Combos)")
    plt.xlabel("Guidance Value")
    plt.ylabel(f"{metric.upper()}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_cross_guidance_plot_{timestamp}.png")
    plt.show()
