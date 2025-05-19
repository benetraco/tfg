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
    parser.add_argument("--resolution", type=int, default=256, help="Resolution for image preprocessing (e.g., 64, 128, 256)")
    parser.add_argument("--dataset", type=str, default="lesion2D_VH_split", help="Dataset name")
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

    # Image Transformations
    preprocess = Compose([
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution)
    ])

    if args.dataset == "lesion2D_VH_split":
        data_dir = repo_path / "/home/benet/data/lesion2D_VH_split/test/flair"
        output_folder = exp_path / "test_images" / str(args.resolution)
        dataset = MRIDataset(data_dir, transform=preprocess)
        save_images(dataset, output_folder)
        print(f"Images saved to {output_folder}")

    elif args.dataset == "VH-SHIFTS-WMH2017_split":
        data_dir = repo_path / "/home/benet/data/VH-SHIFTS-WMH2017_split/test/flair"
        output_folder = exp_path / "test_images"
        subdatasets = ["VH", "SHIFTS", "WMH2017"]
        for subdataset in subdatasets:
            temp_subfolder = data_dir / subdataset
            temp_subfolder.mkdir(parents=True, exist_ok=True)

            for image in os.listdir(data_dir):
                image_path = data_dir / image
                if not image_path.is_file():
                    continue

                if subdataset in ["VH", "WMH2017"]:
                    if subdataset in image:
                        dst = temp_subfolder / image
                        if not dst.exists():
                            os.system(f"cp '{image_path}' '{dst}'")
                elif subdataset == "SHIFTS":
                    if "VH" not in image and "WMH2017" not in image:
                        dst = temp_subfolder / image
                        if not dst.exists():
                            os.system(f"cp '{image_path}' '{dst}'")
            
            output_subfolder = output_folder / subdataset
            dataset = MRIDataset(temp_subfolder, transform=preprocess)
            save_images(dataset, output_subfolder)
            print(f"Images saved to {output_subfolder}")
            # remove the temp_subfolder
            for file in temp_subfolder.iterdir():
                file.unlink()
            temp_subfolder.rmdir()

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
