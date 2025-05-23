#Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
)
from diffusers import AutoencoderKL
from PIL import Image
from pathlib import Path


# Script that generates data for the evaluation of the brain generation model.
class MRIBiomarkemDatasetBuilder:
    def __init__(self, data_folder="/home/benet/data/biomarkem_flair_processed", input_folders=["ISI", "TSI", "VSI"], 
                output_folder="/home/benet/data/biomarkem2D", slices_per_example=5, slices_step=1,
                start_slice=[24,24,26], size=256, slice_mapping=None):
        self.data_folder = data_folder
        self.input_folders = input_folders
        self.output_folder = output_folder
        self.slices_per_example = slices_per_example
        self.slices_step = slices_step
        self.start_slice = start_slice
        self.size = size
        self.slice_mapping = slice_mapping or {}  # default to empty dict
        self._create_output_dirs()

    def _create_output_dirs(self):
        """Creates necessary output directories."""
        os.makedirs(self.output_folder, exist_ok=True)
        for folder in self.input_folders:
            os.makedirs(os.path.join(self.output_folder, folder), exist_ok=True)
    
    def build_dataset(self):
        """Processes all specified folders"""
        for idx, folder in enumerate(self.input_folders):
            input_folder = os.path.join(self.data_folder, folder)
            examples_folders = sorted(os.listdir(input_folder))
            for example_id in examples_folders:
                # if example is a csv file, skip it
                if example_id.endswith(".csv"):
                    continue
                example_folder = os.path.join(input_folder, example_id)
                self._process_example(example_folder, example_id, idx)

    def _process_example(self, example_folder, example_id, idx):
        """Processes a single example folder."""
        example_path = os.path.join(example_folder, "flair_biasfield_corrected.nii.gz")
        if not os.path.exists(example_path):
            print(f"File {example_path} does not exist. Skipping...")
            return
        data = nib.load(example_path).get_fdata()        
        self._save_slices(example_id, data, idx)

    def _save_slices(self, example_id, data, idx):
        """Extracts and saves slices as PNG."""
        modality = self.input_folders[idx]
        img_number = example_id.split("_")[0][-2:]
        start_slice = self.slice_mapping[img_number][self.input_folders[idx]]
        output_folder = os.path.join(self.output_folder, modality)
        end_slice = start_slice + self.slices_per_example * self.slices_step
        for j, i in enumerate(range(start_slice, end_slice, self.slices_step)):
            data_slice = np.rot90(data[:, :, i])
            data_slice = self._preprocess_slice(data_slice)
            self._save_image(data_slice, example_id, j, output_folder)

    def _preprocess_slice(self, slice_data):
        """Preprocess slice by cropping top-bottom brain region and resizing."""
        
        # Normalize and convert to uint8 for PIL
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype('uint8')

        # 1. Find vertical bounding box of non-black region
        threshold = 10
        mask = slice_data > threshold
        if not mask.any():
            y0, y1 = 0, slice_data.shape[0]
        else:
            y_coords = np.any(mask, axis=1)
            y0, y1 = np.where(y_coords)[0][[0, -1]] + [0, 1]

        # 2. Crop vertically, keep full width
        cropped = slice_data[y0:y1, :]

        # 3. Convert to PIL and resize/crop
        img = Image.fromarray(cropped)

        transform = Compose([
            Resize(self.size),
            CenterCrop(self.size),
        ])
        return transform(img)

    def _save_image(self, slice_data, example_id, index, output_folder):
        """Saves a single image slice as PNG."""
        path = os.path.join(output_folder, f"{example_id}_{index}.png")
        plt.imsave(path, slice_data, cmap="gray")
        
    def show_sample(self, image_number):
        """
        Displays a sample of the dataset for each of the three scans (ISI, TSI, VSI) for a given image number.

        Args:
            image_number (str): The number of the image to display, has to be a string in format "01", "02", ..., "14"
        """
        path_ISI = os.path.join(self.output_folder, self.input_folders[0])
        images_ISI = os.listdir(path_ISI)

        path_TSI = os.path.join(self.output_folder, self.input_folders[1])
        images_TSI = os.listdir(path_TSI)

        path_VSI = os.path.join(self.output_folder, self.input_folders[2])
        images_VSI = os.listdir(path_VSI)

        images = images_ISI + images_TSI + images_VSI

        if not images:
            print("No images found in the dataset.")
            return
        
        # image_numer
        image_name_ISI = self.input_folders[0] + str(image_number) + "_0"
        image_name_TSI = self.input_folders[1] + str(image_number) + "_0"
        image_name_VSI = self.input_folders[2] + str(image_number) + "_0"
        images_names_ISI = [f"{image_name_ISI}_{i}.png" for i in range(self.slices_per_example)]
        images_names_TSI = [f"{image_name_TSI}_{i}.png" for i in range(self.slices_per_example)]
        images_names_VSI = [f"{image_name_VSI}_{i}.png" for i in range(self.slices_per_example)]
        selected_images_ISI = [img_name for img_name in images_names_ISI if img_name in images]
        selected_images_TSI = [img_name for img_name in images_names_TSI if img_name in images]
        selected_images_VSI = [img_name for img_name in images_names_VSI if img_name in images]
        if not selected_images_ISI or not selected_images_TSI or not selected_images_VSI:
            print("No matching image slices found in the dataset. \n Image number should be a string in format '01', '02', ..., '15'")  
            print(f"Image name: {image_name_ISI}, {image_name_TSI}, {image_name_VSI}")
            return

        for img_name_ISI, img_name_TSI, img_name_VSI in zip(selected_images_ISI, selected_images_TSI, selected_images_VSI):
            # Load and display the three images one next to the other
            flair_img_ISI = plt.imread(os.path.join(path_ISI, img_name_ISI))
            flair_img_TSI = plt.imread(os.path.join(path_TSI, img_name_TSI))
            flair_img_VSI = plt.imread(os.path.join(path_VSI, img_name_VSI))

            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Example: {image_number}, Slice: {img_name_ISI.split('_')[-1].split('.')[0]}", fontsize=16)
            plt.subplot(1, 3, 1)
            plt.imshow(flair_img_ISI, cmap="gray")
            plt.axis("off")
            plt.title("ISI")

            plt.subplot(1, 3, 2)
            plt.imshow(flair_img_TSI, cmap="gray")
            plt.axis("off")
            plt.title("TSI")

            plt.subplot(1, 3, 3)
            plt.imshow(flair_img_VSI, cmap="gray")
            plt.axis("off")
            plt.title("VSI")

            plt.show()


if __name__ == "__main__":
    slice_mapping = {
        "01": {"ISI": 24, "TSI": 24, "VSI": 27},
        "02": {"ISI": 25, "TSI": 23, "VSI": 26},
        "03": {"ISI": 25, "TSI": 24, "VSI": 25},
        "04": {"ISI": 24, "TSI": 23, "VSI": 24},
        "05": {"ISI": 25, "TSI": 23, "VSI": 26},
        "06": {"ISI": 25, "TSI": 22, "VSI": 26},
        "07": {"ISI": 25, "TSI": 23, "VSI": 24},
        "08": {"ISI": 24, "TSI": 23, "VSI": 24},
        "09": {"ISI": 26, "TSI": 24, "VSI": 26},
        "10": {"ISI": 26, "TSI": 24, "VSI": 25},
        "11": {"ISI": 25, "TSI": 22, "VSI": 25},
        "12": {"ISI": 25, "TSI": 23, "VSI": 25},
        "13": {"ISI": 25, "TSI": 23, "VSI": 24},
        "14": {"ISI": 25, "TSI": 22, "VSI": 26},
        "15": {"ISI": 26, "TSI": 24, "VSI": 25},
    }

    builder = MRIBiomarkemDatasetBuilder(slice_mapping=slice_mapping)
    builder.build_dataset()
    # builder.show_sample("01")