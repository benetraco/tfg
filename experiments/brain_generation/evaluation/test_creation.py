### Preprocess and save MRI test images with multiple modes including biomarkem2D_by_scanner
import os
import sys
from pathlib import Path
import argparse
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode
from PIL import Image

repo_path = Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path):
    repo_path = repo_path.parent
    print(repo_path)

sys.path.insert(0, str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve()
print(f"Repo Path: {repo_path}")
print(f"Experiment Path: {exp_path}")

from dataset.build_dataset import MRIDataset


# ----------------------------
# Scanner Classifier
# ----------------------------
def get_scanner_from_filename(filename):
    parts = filename.split("_")
    if parts[0] == "WMH2017":
        idx = int(parts[1])
        if idx < 50:
            return "Philips"
        elif idx < 70:
            return "Siemens"
        elif idx < 145:
            return "GE"
    if "ISI" in filename:
        return "GE"
    if "TSI" in filename:
        return "Philips"
    if "VSI" in filename:
        return "Siemens"
    return None


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and save MRI images for testing.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution for image preprocessing (e.g., 64, 128, 256)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["lesion2D_VH_split", "VH-SHIFTS-WMH2017_split", "WMH2017_by_scanner", "biomarkem2D_by_scanner"],
                        help="Which test creation strategy to use")
    return parser.parse_args()


# ----------------------------
# Save PIL images to output folder
# ----------------------------
def save_images(images, names, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    for image, name in zip(images, names):
        image.save(output_folder / name)


# ----------------------------
# Mode: lesion2D_VH_split
# ----------------------------
def run_vh_split(resolution):
    data_dir = Path("/home/benet/data/lesion2D_VH_split/test/flair")
    output_folder = exp_path / "test_images" / str(resolution)

    preprocess = Compose([
        Resize(resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(resolution)
    ])

    dataset = MRIDataset(data_dir, transform=preprocess)
    save_images(dataset, [f"image_{i}.png" for i in range(len(dataset))], output_folder)
    print(f"Saved {len(dataset)} images to {output_folder}")


# ----------------------------
# Mode: VH-SHIFTS-WMH2017_split
# ----------------------------
def run_multidataset_split(resolution):
    data_dir = Path("/home/benet/data/VH-SHIFTS-WMH2017_split/test/flair")
    output_folder = exp_path / "test_images"
    subdatasets = ["VH", "SHIFTS", "WMH2017"]

    preprocess = Compose([
        Resize(resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(resolution)
    ])

    for subdataset in subdatasets:
        temp_subfolder = data_dir / subdataset
        temp_subfolder.mkdir(parents=True, exist_ok=True)

        for image in os.listdir(data_dir):
            image_path = data_dir / image
            if not image_path.is_file():
                continue

            if subdataset == "SHIFTS":
                if "VH" not in image and "WMH2017" not in image:
                    os.system(f"cp '{image_path}' '{temp_subfolder / image}'")
            elif subdataset in image:
                os.system(f"cp '{image_path}' '{temp_subfolder / image}'")

        dataset = MRIDataset(temp_subfolder, transform=preprocess)
        save_images(dataset, [f"image_{i}.png" for i in range(len(dataset))], output_folder / subdataset)
        print(f"Saved {len(dataset)} images to {output_folder / subdataset}")

        # Cleanup
        for f in temp_subfolder.iterdir():
            f.unlink()
        temp_subfolder.rmdir()


# ----------------------------
# Mode: WMH2017_by_scanner
# ----------------------------
def run_wmh2017_by_scanner(resolution):
    input_folder = Path("/home/benet/data/WMH2017_split/test/flair")
    output_base = exp_path / "test_images"

    preprocess = Compose([
        Resize(resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(resolution)
    ])

    grouped_images = {"Philips": [], "Siemens": [], "GE": []}
    grouped_names = {"Philips": [], "Siemens": [], "GE": []}

    for file in sorted(input_folder.glob("*.png")):
        scanner = get_scanner_from_filename(file.name)
        if scanner in grouped_images:
            image = Image.open(file).convert("RGB")
            preprocessed = preprocess(image)
            grouped_images[scanner].append(preprocessed)
            grouped_names[scanner].append(file.name)

    for scanner in grouped_images:
        output_folder = output_base / "WMH2017" / scanner
        save_images(grouped_images[scanner], grouped_names[scanner], output_folder)
        print(f"Saved {len(grouped_images[scanner])} images to {output_folder}")


# ----------------------------
# Mode: biomarkem2D_by_scanner
# ----------------------------
def run_biomarkem2d_by_scanner(resolution):
    input_folder = Path("/home/benet/data/biomarkem2D/test/flair")
    output_base = exp_path / "test_images"

    preprocess = Compose([
        Resize(resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(resolution)
    ])

    grouped_images = {"Philips": [], "Siemens": [], "GE": []}
    grouped_names = {"Philips": [], "Siemens": [], "GE": []}

    for file in sorted(input_folder.glob("*.png")):
        scanner = get_scanner_from_filename(file.name)
        if scanner in grouped_images:
            image = Image.open(file).convert("RGB")
            preprocessed = preprocess(image)
            grouped_images[scanner].append(preprocessed)
            grouped_names[scanner].append(file.name)

    for scanner in grouped_images:
        output_folder = output_base / "biomarkem2D" / scanner
        save_images(grouped_images[scanner], grouped_names[scanner], output_folder)
        print(f"Saved {len(grouped_images[scanner])} images to {output_folder}")


# ----------------------------
# Main Dispatcher
# ----------------------------
def main():
    args = parse_args()

    if args.mode == "lesion2D_VH_split":
        run_vh_split(args.resolution)
    elif args.mode == "VH-SHIFTS-WMH2017_split":
        run_multidataset_split(args.resolution)
    elif args.mode == "WMH2017_by_scanner":
        run_wmh2017_by_scanner(args.resolution)
    elif args.mode == "biomarkem2D_by_scanner":
        run_biomarkem2d_by_scanner(args.resolution)


if __name__ == "__main__":
    main()
