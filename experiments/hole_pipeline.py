import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from PIL import Image
import ants
import cv2
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
import torch
from typing import Optional, Tuple


class CreateSample:
    def __init__(self, scanner: str, lesion_bbox: Optional[Tuple[int, int, int, int]], real_brain: bool, seed: int = 17844):
        assert isinstance(scanner, str) and scanner in ["Siemens", "Philips", "GE"], "Scanner must be one of Siemens, Philips, or GE."
        assert lesion_bbox is None or (
            isinstance(lesion_bbox, tuple) and
            len(lesion_bbox) == 4 and
            all(isinstance(x, int) and 0 <= x < 256 for x in lesion_bbox)
        ), "Lesion bounding box must be None or a tuple of 4 integers between 0 and 256."
        
        self.scanner = scanner
        self.lesion_bbox = lesion_bbox
        self.real_brain = real_brain
        self.syntetic_scanner_path = f"/home/benet/tfg/experiments/brain_generation/evaluation/generated_images/latent_finetuning_scanners_healthy/{scanner}/g1.0" # the guidance value 1.0 has the best results in terms of FID and CMMD
        self.real_scanner_path = f"/home/benet/data/biomarkem2D/test/{scanner}"
        self.seed = seed
        self.wm_gm_mask = None
        self.lesion_mask = None
        self.healthy_brain = None
        self.image = self.generate_image()

    def generate_image(self):
        """
        Generates a synthetic brain image with a lesion.
        If `real_brain` is True, it uses a real brain image; otherwise, it generates a synthetic one.
        If `lesion_bbox` is None, it generates a random bounding box for the lesion.
        """
        self.healthy_brain = self.get_brain_image()
        
        
        if not self.in_mask_brain():
            self.lesion_bbox = self.random_bbox()
        
        return self.get_lesion_image()
    
    def get_brain_image(self):
        """
        Returns a healthy brain image from the specified scanner path.
        If `real_brain` is True, it uses a real brain image; otherwise, it uses a synthetic one.
        """
        np.random.seed(self.seed)
        
        if self.real_brain:
            images = os.listdir(self.real_scanner_path)
            image_name = np.random.choice(images)
            image_path = os.path.join(self.real_scanner_path, image_name)
            image = Image.open(image_path).convert("L")
        else:
            images = os.listdir(self.syntetic_scanner_path)
            image_name = np.random.choice(images)
            image_path = os.path.join(self.syntetic_scanner_path, image_name)
            image = Image.open(image_path).convert("L")
        
        assert image.size == (256, 256), "Image size must be 256x256 pixels."
        return image
    
    def in_mask_brain(self):
        """
        Checks if the lesion bounding box is within the white matter and gray matter mask.
        Returns True if the lesion is predominantly within the WM/GM mask, otherwise False.
        """
        if self.wm_gm_mask is None:
            self.wm_gm_mask = self.get_wm_gm_mask()
        
        if not self.lesion_bbox:
            return False
        
        # Check if lesion bbox is within the WM/GM mask
        x1, y1, x2, y2 = self.lesion_bbox
        lesion_mask = self.wm_gm_mask[y1:y2, x1:x2]

        # if a 90% of the lesion mask is within the WM/GM mask, return True
        lesion_mask = lesion_mask > 0
        return np.sum(lesion_mask) / lesion_mask.size >= 0.9
    
    def get_wm_gm_mask(self):
        img_arr = np.array(self.healthy_brain).astype(np.float32)

        # Segment using ANTs
        img = ants.from_numpy(img_arr)
        mask = ants.get_mask(img)
        seg = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask, verbose=0)

        # Classify tissues by intensity
        seg_img = seg["segmentation"]
        labels = np.unique(seg_img.numpy())
        means = [img.numpy()[seg_img.numpy() == l].mean() for l in labels]
        sorted_labels = [label for _, label in sorted(zip(means, labels))]

        # Combine WM and GM (highest two intensity classes)
        wm_gm_mask = np.isin(seg_img.numpy(), sorted_labels[-2:]).astype(np.uint8) * 255

        # do a closing operation to fill small holes
        wm_gm_mask = cv2.morphologyEx(wm_gm_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        return wm_gm_mask
    
    def random_bbox(self, min_size: int = 16, max_size: int = 64, max_attempts: int = 100):
        """
        Generate a random rectangular bounding box (width and height between min_size and max_size)
        that lies at least 90% inside the WM/GM mask.
        """
        if self.wm_gm_mask is None:
            raise ValueError("WM/GM mask must be computed before generating random bbox.")
        
        mask = self.wm_gm_mask
        h, w = mask.shape

        for attempt in range(max_attempts):
            box_w = np.random.randint(min_size, max_size + 1)
            box_h = np.random.randint(min_size, max_size + 1)

            if box_w >= w or box_h >= h:
                continue  # skip impossible sizes

            x1 = np.random.randint(0, w - box_w)
            y1 = np.random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            region = mask[y1:y2, x1:x2]
            ratio = np.sum(region > 0) / (box_w * box_h)

            if ratio >= 0.9:
                print(f"Valid bbox found on attempt {attempt + 1}: ({x1}, {y1}, {x2}, {y2})")
                return (x1, y1, x2, y2)

        raise RuntimeError(f"Could not find a valid bounding box within {max_attempts} attempts.")
    
    def get_lesion_image(self):
        
        if self.lesion_mask is None:
            self.generate_mask(self.healthy_brain.size)
        model = self.load_inpainting_pipeline()

        # if image in one channel, convert to RGB
        if self.healthy_brain.mode == 'L':
            self.healthy_brain = self.healthy_brain.convert('RGB')

        return model(
            prompt="SHIFTS multiple sclerosis lesion in a FLAIR MRI",
            image=self.healthy_brain,
            mask_image=self.lesion_mask,
            num_inference_steps=25,
            guidance_scale=5,
        ).images[0]

    def generate_mask(self, size:tuple):
        
        mask = np.zeros(size, dtype=np.uint8)
        x1, y1, x2, y2 = self.lesion_bbox
        mask[y1:y2, x1:x2] = 255
        
        # Convert to PIL Image
        mask_image = Image.fromarray(mask, mode='L')

        self.lesion_mask = mask_image
    
    def load_inpainting_pipeline(self, model_id:str="benetraco/ms-lesion-inpainting-vh-shifts", device:str="cuda"):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
        # pipe.set_progress_bar_config(disable=True)
        return pipe.to(device)
    
    def plot_image(self):
        """
        Displays the original image, the mask, and the generated image with the lesion.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(self.healthy_brain, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        if self.lesion_mask is not None:
            axes[1].imshow(self.lesion_mask, cmap='gray')
            axes[1].set_title("Lesion Mask")
            axes[1].axis('off')

        axes[2].imshow(self.image, cmap='gray')
        axes[2].set_title("Image with Lesion")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()