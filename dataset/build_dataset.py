import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

class MRIDatasetBuilder:
    def __init__(self, data_folder="/home/benet/data", input_folder="VH", output_folder="VH2D", folders=["train", "test"], 
                 flair_image="flair.nii.gz", mask_image="lesionMask.nii.gz", slices_per_example=5, slices_step=3, start_slice=88):
        self.data_folder = data_folder
        self.input_folder = os.path.join(data_folder, input_folder)
        self.output_folder = os.path.join(data_folder, output_folder)
        self.folders = folders
        self.flair_image = flair_image
        self.mask_image = mask_image
        self.slices_per_example = slices_per_example
        self.slices_step = slices_step
        self.start_slice = start_slice
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Creates necessary output directories."""
        sub_dirs = ["images/flair", "images/mask", "npy/flair", "npy/mask"]
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(self.output_folder, sub_dir), exist_ok=True)
    
    def process_folders(self):
        """Processes all specified folders (train/test)."""
        for folder in self.folders:
            folder_path = os.path.join(self.input_folder, folder)
            examples = sorted(os.listdir(folder_path))
            
            for example in examples:
                self._process_example(folder_path, example)
    
    def _process_example(self, folder_path, example):
        """Processes a single example folder."""
        example_path = os.path.join(folder_path, example)
        if not os.path.isdir(example_path):
            print(f"Skipping {example_path}")
            return
        
        flair_path = os.path.join(example_path, self.flair_image)
        mask_path = os.path.join(example_path, self.mask_image)
        flair_data, mask_data = self._load_nifti_images(flair_path, mask_path)
        
        self._save_slices(example, flair_data, mask_data)
    
    def _load_nifti_images(self, flair_path, mask_path):
        """Loads NIfTI images and returns their data arrays."""
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        assert flair.shape == mask.shape, "Flair and Mask shapes do not match!"
        return flair, mask
    
    def _save_slices(self, example, flair_data, mask_data):
        """Extracts and saves slices as PNG and NPY files."""
        end_slice = self.start_slice + self.slices_per_example * self.slices_step
        for j, i in enumerate(range(self.start_slice, end_slice, self.slices_step)):
            flair_slice = np.rot90(flair_data[:, :, i])
            mask_slice = np.rot90(mask_data[:, :, i])
            
            self._save_image(flair_slice, "flair", example, j)
            self._save_image(mask_slice, "mask", example, j)
            self._save_npy(flair_slice, "flair", example, j)
            self._save_npy(mask_slice, "mask", example, j)
    
    def _save_image(self, slice_data, image_type, example, index):
        """Saves a single image slice as PNG."""
        path = os.path.join(self.output_folder, "images", image_type, f"{example}_{index}.png")
        plt.imsave(path, slice_data, cmap="gray")
    
    def _save_npy(self, slice_data, image_type, example, index):
        """Saves a single slice as an NPY file."""
        path = os.path.join(self.output_folder, "npy", image_type, f"{example}_{index}.npy")
        np.save(path, slice_data)

# Example Usage
data_folder = "/home/benet/data"
input_folder = "VH"
output_folder = "VH_slices"
folders = ["train", "test"]
flair_image = "flair.nii.gz"
mask_image = "lesionMask.nii.gz"
slices_per_example = 5
slices_step = 3
start_slice = 88

dataset_builder = MRIDatasetBuilder(data_folder, input_folder, output_folder, folders, flair_image, mask_image, slices_per_example, slices_step, start_slice)
dataset_builder.process_folders()
