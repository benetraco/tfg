#Libraries
import os
# Restrict PyTorch to use only GPU X
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from diffusers import AutoencoderKL
from PIL import Image
from pathlib import Path

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



class MRIHealthyDatasetBuilder:
    def __init__(self, data_folder="/home/benet/data", input_folder="CamCAN", output_folder="Healthy2D/CamCAN", 
                 slices_per_example=10, slices_step=1, start_slice=83):
        self.data_folder = data_folder
        self.input_folder = os.path.join(data_folder, input_folder)
        self.output_folder = os.path.join(data_folder, output_folder)
        self.slices_per_example = slices_per_example
        self.slices_step = slices_step
        self.start_slice = start_slice
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Creates necessary output directories."""
        os.makedirs(os.path.join(self.output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "npy"), exist_ok=True)
    
    def build_dataset(self):
        """Processes all specified folders (train/test)."""
        examples_folders = sorted(os.listdir(self.input_folder))
        print(len(examples_folders))
        for example_id in examples_folders:
            # if example is a csv file, skip it
            if example_id.endswith(".csv"):
                continue
            example_folder = os.path.join(self.input_folder, example_id)
            self._process_example(example_folder, example_id)

    def _process_example(self, example_folder, example_id):
        """Processes a single example folder."""
        # the example is the file in the folder
        for file in os.listdir(example_folder):
            if file.endswith(".nii.gz"):
                example = file

        example_path = os.path.join(example_folder, example)
        
        data = nib.load(example_path).get_fdata()
        
        self._save_slices(example_id, data)
    
    def _save_slices(self, example_id, data):
        """Extracts and saves slices as PNG and NPY files."""
        end_slice = self.start_slice + self.slices_per_example * self.slices_step
        for j, i in enumerate(range(self.start_slice, end_slice, self.slices_step)):
            data_slice = np.rot90(data[:, :, i])
            
            self._save_image(data_slice, example_id, j)
            self._save_npy(data_slice, example_id, j)
    
    def _save_image(self, slice_data, example_id, index):
        """Saves a single image slice as PNG."""
        path = os.path.join(self.output_folder, "images", f"{example_id}_{index}.png")
        plt.imsave(path, slice_data, cmap="gray")
    
    def _save_npy(self, slice_data, example_id, index):
        """Saves a single slice as an NPY file."""
        path = os.path.join(self.output_folder, "npy", f"{example_id}_{index}.npy")
        np.save(path, slice_data)
    
    def show_sample(self, image_name=None, num_images=1):
        """
        Displays one or more flair images with their corresponding masks.
        If image_name is provided, it will display the slices of that image, otherwise it will display num_images random images.
        """
        path = os.path.join(self.output_folder, "images")
        
        images = os.listdir(path)
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
                print("No matching image slices found in the dataset. \n Image name should be in the following format: 'sub-CC110033' where sub-CC110033 is the example number and it should be in the dataset.")   
                return

        for img_name in selected_images:
            flair_img = plt.imread(os.path.join(path, img_name))
            
            plt.figure(figsize=(10, 10))
            plt.imshow(flair_img, cmap="gray")
            plt.axis("off")
            plt.show()


# Custom dataset
class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None, latents=False, RGB=False, prompt=False):
        self.data_dir = data_dir
        self.transform = transform
        self.latents = latents
        self.RGB = RGB
        self.prompt = prompt
        if latents:
            self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        else:
            self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_files) // 4 if self.latents else len(self.image_files)
    
    def _get_prompt_from_filename(self, filename):
        """Derive a prompt based on filename patterns."""
        if "dev" in filename or "eval" in filename or "train" in filename:
            return "SHIFTS FLAIR MRI"
        elif "VH" in filename:
            return "VH FLAIR MRI"
        elif "WMH2017" in filename:
            return "WMH2017 FLAIR MRI"
        else:
            return "FLAIR MRI"

    def __getitem__(self, idx):
        if self.latents:
            # Load four consecutive images as one sample
            img_paths = self.image_files[idx * 4 : (idx + 1) * 4]
            images = [np.load(os.path.join(self.data_dir, img_path)) for img_path in img_paths]

            if self.transform:
                images = [self.transform(img) for img in images]
            # Stack images along the channel dimension (4, 64, 64)
            latents = torch.cat(images, dim=0)
            
            if self.prompt:
                prompt = self._get_prompt_from_filename(img_paths[0])

                return latents, prompt  # return both latent -Shape: (4, 64, 64)- and its corresponding prompt

            return latents  # Shape: (4, 64, 64)

        else:    
            img_path = os.path.join(self.data_dir, self.image_files[idx])
            image = Image.open(img_path).convert("L")  # Convert to grayscale

            if self.RGB:
                image = image.convert("RGB")  # Convert to 3-channel RGB
            if self.transform:
                image = self.transform(image)

            return image       


class MRIDatasetScanners(Dataset):
    def __init__(self, data_dir, transform=None, latents=False, RGB=False, prompt=False):
        self.data_dir = data_dir
        self.transform = transform
        self.latents = latents
        self.RGB = RGB
        self.prompt = prompt
        if latents:
            self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        else:
            self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_files) // 4 if self.latents else len(self.image_files)
    
    def _get_prompt_from_filename(self, filename):
        """Derive a prompt based on filename patterns."""
        filename_parts = filename.split("_")
        if filename_parts[0] == "WMH2017":
            # Philips: 0 - 49
            # Simens: 50 - 69
            # GE: 100 - 144
            if int(filename_parts[1]) < 50:
                return "Philips FLAIR MRI"
            elif int(filename_parts[1]) < 70:
                return "Siemens FLAIR MRI"
            elif int(filename_parts[1]) < 145:
                return "GE FLAIR MRI"
            else:
                return "Unknown Scanner FLAIR MRI"
        else:
            return None

    def __getitem__(self, idx):
        if self.latents:
            # Load four consecutive images as one sample
            img_paths = self.image_files[idx * 4 : (idx + 1) * 4]
            images = [np.load(os.path.join(self.data_dir, img_path)) for img_path in img_paths]

            if self.transform:
                images = [self.transform(img) for img in images]
            # Stack images along the channel dimension (4, 64, 64)
            latents = torch.cat(images, dim=0)
            
            if self.prompt:
                prompt = self._get_prompt_from_filename(img_paths[0])

                return latents, prompt  # return both latent -Shape: (4, 64, 64)- and its corresponding prompt

            return latents  # Shape: (4, 64, 64)

        else:    
            img_path = os.path.join(self.data_dir, self.image_files[idx])
            image = Image.open(img_path).convert("L")  # Convert to grayscale

            if self.RGB:
                image = image.convert("RGB")  # Convert to 3-channel RGB
            if self.transform:
                image = self.transform(image)

            return image       


class LatentImageProcessor:
    def __init__(self, repo_path, input_dir = "/home/benet/data/VH2D/images/flair", output_dir = "/home/benet/data/VH2D/latent_flair",
                  resolution = 256, finetuned_vae = True, scale = False):
        self.repo_path = repo_path
        self.resolution = resolution
        self.scale = scale
        
        # Dataset and Latent Directory Setup
        self.input_dir = input_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define the image preprocessing pipeline
        self.preprocess = Compose(
            [
                Resize(self.resolution, interpolation=InterpolationMode.BILINEAR), 
                CenterCrop(self.resolution),  # Center crop to the desired squared resolution
                ToTensor(),  # Convert to PyTorch tensor
                Normalize(mean=[0.5], std=[0.5]),  # Map to (-1, 1) as a way to make data more similar to a Gaussian distribution
            ]
        )

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load VAE model
        if finetuned_vae:
            pipeline_dir = self.repo_path / 'results/pipelines' / 'fintuned_vae'
            self.vae = AutoencoderKL.from_pretrained(pipeline_dir)
        else:
            self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.vae.to(self.device).eval()

    def process_images(self):
        # Get image files from the dataset directory
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(".png")]
        print(f"Found {len(image_files)} images in the dataset folder")

        # Process each image and save latents
        with torch.no_grad():
            for image_file in image_files:
                img_path = os.path.join(self.input_dir, image_file)
                # Remove .png extension and add '_latent' in the image name
                image_file = image_file.split('.')[0] + "_latent"
                image = Image.open(img_path).convert("RGB")

                image = self.preprocess(image).unsqueeze(0).to(self.device)

                latent = self.vae.encode(image).latent_dist.sample()
                if self.scale:
                    latent *= self.vae.config.scaling_factor
                latent = latent.squeeze(0)
                
                # Save the latent images
                latent = latent.cpu().numpy()
                for j in range(4):
                    np.save(self.output_dir / f"{image_file}_{j}.npy", latent[j])
            
            # count how many latent images were saved
            latent_files = [f for f in os.listdir(self.output_dir) if f.endswith(".npy")]
            print(f"{len(latent_files)} latent images saved in {self.output_dir}")


class MRILesionDatasetBuilder:
    def __init__(self, data_folder="/home/benet/data", input_folder="VH", output_folder="lesion2D", folders=["train", "test"], flair_image="flair.nii.gz",
                 mask_image="lesionMask.nii.gz", slices_per_example=13, slices_step=1, start_slice=85, train_split=0.7, seed=17844, skip_empty_masks=True,
                 fill_lesion=False):
        self.data_folder = data_folder
        self.input_folder = input_folder
        self.output_folder = os.path.join(data_folder, output_folder)
        self.folders = folders
        self.flair_image = flair_image
        self.mask_image = mask_image
        self.slices_per_example = slices_per_example
        self.slices_step = slices_step
        self.start_slice = start_slice
        self.train_split = train_split
        self.seed = seed
        np.random.seed(seed)
        self.skip_empty_masks = skip_empty_masks
        self.fill_lesion = fill_lesion
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Creates necessary output directories."""
        sub_dirs = ["train/flair", "train/mask", "test/flair", "test/mask"]
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(self.output_folder, sub_dir), exist_ok=True)
    
    def build_dataset(self):
        """Processes all specified folders (train/test)."""
        empty_masks, train_count, test_count = 0, 0, 0
        if self.input_folder == "VH":
            train_examples, test_examples = 0, 0
            total_examples = sum(len(os.listdir(os.path.join(self.data_folder, self.input_folder, folder))) for folder in self.folders)
            for folder in self.folders:
                folder_path = os.path.join(self.data_folder, self.input_folder, folder)
                examples = sorted(os.listdir(folder_path))
                
                for example in examples:
                    if train_examples >= total_examples * self.train_split:
                        train_test = "test"
                        example_empty_masks = self._process_example(folder_path, example, folder, train_test)
                    elif test_examples >= total_examples * (1 - self.train_split):
                        train_test = "train"
                        example_empty_masks = self._process_example(folder_path, example, folder, train_test)
                    else:
                        train_test = folder
                        example_empty_masks = self._process_example(folder_path, example, folder, train_test)
                    
                    empty_masks += example_empty_masks
                    increment = self.slices_per_example - example_empty_masks
                    train_count += (train_test == "train") * increment
                    test_count += (train_test == "test") * increment
                    train_examples += (train_test == "train")
                    test_examples += (train_test == "test")

        elif self.input_folder == "SHIFTS_preprocessedMNI":
            for folder in self.folders:
                folder_path = os.path.join(self.data_folder, self.input_folder, folder)
                examples = os.listdir(folder_path)
                np.random.shuffle(examples)
                examples_train = examples[:int(len(examples) * self.train_split)]
                examples_test = examples[int(len(examples) * self.train_split):]
                for example in examples_train:
                    example_empty_masks = self._process_example(folder_path, example, folder, "train")
                    empty_masks += example_empty_masks
                    train_count += self.slices_per_example - example_empty_masks
                for example in examples_test:
                    example_empty_masks = self._process_example(folder_path, example, folder, "test")
                    empty_masks += example_empty_masks
                    test_count += self.slices_per_example - example_empty_masks

        elif self.input_folder == "WMH2017_preprocessedMNI":
            folder_path = os.path.join(self.data_folder, self.input_folder)
            examples = os.listdir(folder_path)
            np.random.shuffle(examples)
            examples_train = examples[:int(len(examples) * self.train_split)]
            examples_test = examples[int(len(examples) * self.train_split):]
            for example in examples_train:
                example_empty_masks = self._process_example(folder_path, example, None, "train")
                empty_masks += example_empty_masks
                train_count += self.slices_per_example - example_empty_masks
            for example in examples_test:
                example_empty_masks = self._process_example(folder_path, example, None, "test")
                empty_masks += example_empty_masks
                test_count += self.slices_per_example - example_empty_masks
        
        else:
            print(f"Unknown input folder: {self.input_folder}, only 'VH' and 'SHIFTS_preprocessedMNI' and 'WMH2017_preprocessedMNI' are supported.")
            return
                
        print(f"Total empty masks skipped: {empty_masks}")

        print(f"In the preprocessed folder: Total examples: {train_count + test_count}, train examples: {train_count} ({train_count/(train_count + test_count) * 100:.2f}%), test examples: {test_count} ({test_count/(train_count + test_count) * 100:.2f}%)")

        train_count_total = len(os.listdir(os.path.join(self.output_folder, "train/flair")))
        test_count_total = len(os.listdir(os.path.join(self.output_folder, "test/flair")))
        print(f"In the hole dataset: Total examples: {train_count_total + test_count_total}, train examples: {train_count_total} ({train_count_total/(train_count_total + test_count_total) * 100:.2f}%), test examples: {test_count_total} ({test_count_total/(train_count_total + test_count_total) * 100:.2f}%)")

    def _process_example(self, folder_path, example, folder, train_test=None):
        """Processes a single example folder."""
        example_path = os.path.join(folder_path, example)
        if not os.path.isdir(example_path):
            print(f"Skipping {example_path}")
            return
        
        flair_path = os.path.join(example_path, self.flair_image)
        mask_path = os.path.join(example_path, self.mask_image)
        flair_data, mask_data = self._load_nifti_images(flair_path, mask_path)
        
        return self._save_slices(example, flair_data, mask_data, folder, train_test)   
      
    def _load_nifti_images(self, flair_path, mask_path):
        """Loads NIfTI images and returns their data arrays."""
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        assert flair.shape == mask.shape, "Flair and Mask shapes do not match!"
        return flair, mask
    
    def _save_slices(self, example, flair_data, mask_data, folder, train_test=None):
        """Extracts and saves slices as PNG files."""
        end_slice = self.start_slice + self.slices_per_example * self.slices_step
        empty_masks = 0
        for j, i in enumerate(range(self.start_slice, end_slice, self.slices_step)):
            flair_slice = np.rot90(flair_data[:, :, i])
            mask_slice = np.rot90(mask_data[:, :, i])
            
            # If mask_slice is empty, skip saving
            if self.skip_empty_masks and np.sum(mask_slice) == 0:
                print(f"Skipping empty mask for {folder} {example} at slice {i}")
                empty_masks += 1
                continue
        
            # If fill_lesion is True, fill the lesion in the flair image with a gray value (0-255)
            if self.fill_lesion:
                # mid = (np.max(flair_slice) - np.min(flair_slice)) / 2
                # mean of flair_slice excuding where pixels are 0
                # zero_pixels = np.where(flair_slice == 0)
                # print(f"Number of zero pixels in flair slice: {len(zero_pixels[0])}")
                mean = np.mean(flair_slice[flair_slice > 0])
                flair_slice[mask_slice > 0] = mean
                # print(f"Filling lesion for {folder} {example} at slice {i}")
                # print the maximum value in the flair slice and the minimum
                # print(f"Max value in flair slice: {np.max(flair_slice)}, Min value in flair slice: {np.min(flair_slice)}")

            self._save_image(flair_slice, "flair", example, j, folder, train_test)
            self._save_image(mask_slice, "mask", example, j, folder, train_test)

        return empty_masks
    
    def _save_image(self, slice_data, image_type, example, index, folder, train_test=None):
        """Saves a single image slice as PNG."""
        if self.input_folder == "VH": # VH dataset folder is already train/test
            path = os.path.join(self.output_folder, train_test, image_type, f"{self.input_folder}_{example}_{index}.png")
        elif self.input_folder == "SHIFTS_preprocessedMNI":
            path = os.path.join(self.output_folder, train_test, image_type, f"{folder}_{example}_{index}.png")
        elif self.input_folder == "WMH2017_preprocessedMNI":
            path = os.path.join(self.output_folder, train_test, image_type, f"WMH2017_{example}_{index}.png")
        else:
            raise ValueError(f"Unknown input folder: {self.input_folder}")
               
        plt.imsave(path, slice_data, cmap="gray")

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


##### Example Usage of the MRIDatasetBuilder
# data_folder = "/home/benet/data"
# input_folder = "VH"
# output_folder = "VH2D"
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


##### Example Usage of the LatentImageProcessor
# from pathlib import Path
# import os, sys
# repo_path= Path.cwd().resolve()
# while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
#     repo_path = repo_path.parent #go up one level
#     print(repo_path)
# processor = LatentImageProcessor(repo_path)
# processor.process_images()


### Example Usage of the MRIBiomarkemDatasetBuilder
# if __name__ == "__main__":
#     slice_mapping = {
#         "01": {"ISI": 24, "TSI": 24, "VSI": 27},
#         "02": {"ISI": 25, "TSI": 23, "VSI": 26},
#         "03": {"ISI": 25, "TSI": 24, "VSI": 25},
#         "04": {"ISI": 24, "TSI": 23, "VSI": 24},
#         "05": {"ISI": 25, "TSI": 23, "VSI": 26},
#         "06": {"ISI": 25, "TSI": 22, "VSI": 26},
#         "07": {"ISI": 25, "TSI": 23, "VSI": 24},
#         "08": {"ISI": 24, "TSI": 23, "VSI": 24},
#         "09": {"ISI": 26, "TSI": 24, "VSI": 26},
#         "10": {"ISI": 26, "TSI": 24, "VSI": 25},
#         "11": {"ISI": 25, "TSI": 22, "VSI": 25},
#         "12": {"ISI": 25, "TSI": 23, "VSI": 25},
#         "13": {"ISI": 25, "TSI": 23, "VSI": 24},
#         "14": {"ISI": 25, "TSI": 22, "VSI": 26},
#         "15": {"ISI": 26, "TSI": 24, "VSI": 25},
#     }

#     builder = MRIBiomarkemDatasetBuilder(slice_mapping=slice_mapping)
#     builder.build_dataset()
#     # builder.show_sample("01")