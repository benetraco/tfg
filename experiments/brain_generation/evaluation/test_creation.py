### This script preprocesses and saves MRI images for testing.
# It resizes and crops the images to a specified resolution and saves them in a designated folder.

import os
import sys
from pathlib import Path
import argparse
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode
    
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
    print(repo_path)
    
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve() # path to the experiment folder
print(f"Repo Path: {repo_path}")
print(f"Experiment Path: {exp_path}")

from dataset.build_dataset import MRIDataset

# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and save MRI images for testing.")
    parser.add_argument("--resolution", type=int, required=True, help="Resolution for image preprocessing (e.g., 64, 128, 256)")
    return parser.parse_args()

# ----------------------------
# Image Saving Function
# ----------------------------
def save_images(images, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    existing_images = len(list(output_folder.iterdir()))
    
    for idx, image in enumerate(images):
        image.save(output_folder / f"image_{existing_images + idx}.png")

# ----------------------------
# Main Processing Function
# ----------------------------
def main():
    args = parse_args()
    
    data_dir = repo_path / "/home/benet/data/lesion2D_VH_split/test/flair"
    output_folder = exp_path / "test_images" / str(args.resolution)

    # Image Transformations
    preprocess = Compose([
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution)
    ])

    # Dataset Initialization
    dataset = MRIDataset(data_dir, transform=preprocess)

    # Save Images
    save_images(dataset, output_folder)
    print(f"Images saved to {output_folder}")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
