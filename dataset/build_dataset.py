import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

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
    
    def build_dataset(self):
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
    
    def show_sample(self, image_name=None, num_images=1):
        """
        Displays one or more flair images with their corresponding masks.
        If image_name is provided, it will display the slices of that image, otherwise it will display num_images random images.
        """
        flair_path = os.path.join(self.output_folder, "images", "flair")
        mask_path = os.path.join(self.output_folder, "images", "mask")
        
        images = os.listdir(flair_path)
        if not images:
            print("No images found in the dataset.")
            return
        
        if image_name is None:
            selected_images = np.random.choice(images, size=min(num_images, len(images)), replace=False)
        else:
            # image name is something like "123"
            # images_names should be something like ["123_0.png", "123_1.png", ...] until the last slice of the image
            images_names = [f"{image_name}_{i}.png" for i in range(self.slices_per_example)]
            selected_images = [img_name for img_name in images_names if img_name in images]
            if not selected_images:
                print("No matching image slices found in the dataset. \n Image name should be in the following format: '123' where 123 is the example number and it should be in the dataset.")   
                return

        for img_name in selected_images:
            flair_img = plt.imread(os.path.join(flair_path, img_name))
            mask_img = plt.imread(os.path.join(mask_path, img_name))
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(flair_img, cmap="gray")
            axes[0].set_title(f"Flair Image: {img_name}")
            axes[1].imshow(mask_img, cmap="gray")
            axes[1].set_title(f"Mask Image: {img_name}")
            plt.show()


# Custom dataset
class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image


##### Example Usage of the MRIDatasetBuilder
# data_folder = "/home/benet/data"
# input_folder = "VH"
# output_folder = "VH2D""
# folders = ["train", "test"]
# flair_image = "flair.nii.gz"
# mask_image = "lesionMask.nii.gz"
# slices_per_example = 5
# slices_step = 3
# start_slice = 88

# dataset_builder = MRIDatasetBuilder(data_folder, input_folder, output_folder, folders, flair_image, mask_image, slices_per_example, slices_step, start_slice)
# dataset_builder.build_dataset()
# dataset_builder.show_sample()
# dataset_builder.show_sample(image_name="739")


##### Example Usage of the MRIDataset
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
# from torchvision.transforms.functional import InterpolationMode
# data_dir = "/home/benet/data/VH2D/images/flair"
# resolution = 256
# batch_size = 32
# num_workers = 4

# preprocess = Compose(
#     [
#         Resize(resolution, interpolation= InterpolationMode.BILINEAR), #getattr(InterpolationMode, config['processing']['interpolation'])),  # Smaller edge is resized to 256 preserving aspect ratio
#         CenterCrop(resolution),  # Center crop to the desired squared resolution
#         #RandomHorizontalFlip(),  # Horizontal flip may not be a good idea if we want generation only one laterality
#         ToTensor(),  # Convert to PyTorch tensor
#         Normalize(mean=[0.5], std=[0.5]),  # Map to (-1, 1) as a way to make data more similar to a Gaussian distribution
#     ]
# )

# # Create dataset with the defined transformations
# dataset = MRIDataset(data_dir, transform=preprocess)
# # Create the dataloader
# train_dataloader = DataLoader(
#     dataset, batch_size=batch_size, num_workers= num_workers, shuffle=True
# )
# print(f"Number of samples in the dataset: {len(dataset)}")
# print(f"Number of batches in the dataloader: {len(train_dataloader)}")
