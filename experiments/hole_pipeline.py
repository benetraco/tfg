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
from torchvision.transforms.functional import to_tensor
from skimage import measure

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
        self.lesion_bbox_mask = None
        self.healthy_brain = None
        self.lesioned_brain = self.generate_image()
        self.lesion_mask, self.diff_bbox = self.calculate_difference()
        self.segmented_lesion = self.segment_lesion()

    def segment_lesion(self):
        """"""
        # in the image with lesion, paint a red line in the lesion area from the difference image
        segmented_lesion = np.array(self.lesioned_brain)
        lesion_mask = np.array(self.lesion_mask)
        contours = measure.find_contours(lesion_mask > 0, 0.5)
        for contour in contours:
            for y, x in contour.astype(np.int32):
                if 0 <= x < segmented_lesion.shape[1] and 0 <= y < segmented_lesion.shape[0]:
                    # paint the pixel red
                    segmented_lesion[y, x] = [255, 0, 0]
        # convert to PIL Image
        segmented_lesion = Image.fromarray(segmented_lesion)


        return segmented_lesion


    def calculate_difference(self):
        """
        Calculates the difference between the original healthy brain image and the generated image with the lesion.
        """
        if self.healthy_brain.size != self.lesioned_brain.size:
            # resize the generated image to match the healthy brain image size
            self.lesioned_brain = self.lesioned_brain.resize(self.healthy_brain.size, Image.LANCZOS)

        healty_tensor = to_tensor(self.healthy_brain)
        lesioned_tensor = to_tensor(self.lesioned_brain)

        # diff_tensor is the difference to white between the lesioned and healthy brain images
        diff_tensor = lesioned_tensor - healty_tensor
        diff_image = diff_tensor.mean(dim=0)

        diff_bbox = diff_image.numpy()
        # restrict to only the values inside the bounding box of the lesion
        assert self.lesion_bbox is not None, "Lesion bounding box must be defined to calculate the difference."
        x1, y1, x2, y2 = self.lesion_bbox
        diff_bbox = diff_bbox[y1:y2, x1:x2]

        # normalize the difference image to the range [0, 1]
        diff_bbox = (diff_bbox - diff_bbox.min()) / (diff_bbox.max() - diff_bbox.min())
        #plot an histogram of the difference image
        plt.hist(diff_bbox.ravel(), bins=256, color='blue', alpha=0.7)
        plt.title("Histogram of the Difference Image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        # binarize the difference image to create a mask
        # diff_bbox = (diff_bbox > 0.7).astype(np.uint8) * 255
        
        # Convert the normalized diff_bbox to 8-bit grayscale
        diff_bbox_8bit = (diff_bbox * 255).astype(np.uint8)
        # Apply Otsu's thresholding
        otsu_thresh, diff_bbox_bin = cv2.threshold(diff_bbox_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print(f"Otsu's threshold: {otsu_thresh}")
        diff_bbox = diff_bbox_bin

        # apply a opening operation to remove small noise
        diff_bbox = cv2.morphologyEx(diff_bbox, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # apply a closing operation to fill small holes
        # diff_bbox = cv2.morphologyEx(diff_bbox, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # create a image of the size of the lesion_bbox_mask with the difference image
        lesion_mask = np.zeros_like(np.array(self.lesion_bbox_mask))

        lesion_mask[y1:y2, x1:x2] = diff_bbox
        # convert to PIL Image
        diff_bbox_image = Image.fromarray(diff_bbox, mode='L')
        lesion_mask_image = Image.fromarray(lesion_mask, mode='L')

        return lesion_mask_image, diff_bbox_image
        # return diff_bbox, diff_bbox_image

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
        lesion_bbox_mask = self.wm_gm_mask[y1:y2, x1:x2]

        # if a 90% of the lesion mask is within the WM/GM mask, return True
        lesion_bbox_mask = lesion_bbox_mask > 0
        return np.sum(lesion_bbox_mask) / lesion_bbox_mask.size >= 0.9
    
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
        
        if self.lesion_bbox_mask is None:
            self.generate_mask(self.healthy_brain.size)
        model = self.load_inpainting_pipeline()

        # if image in one channel, convert to RGB
        if self.healthy_brain.mode == 'L':
            self.healthy_brain = self.healthy_brain.convert('RGB')

        lesioned_image = model(
            prompt="SHIFTS multiple sclerosis lesion in a FLAIR MRI",
            image=self.healthy_brain,
            mask_image=self.lesion_bbox_mask,
            num_inference_steps=25,
            guidance_scale=5,
            generator=torch.Generator(device="cuda").manual_seed(self.seed)
        ).images[0]

        return lesioned_image
    
    def generate_mask(self, size:tuple):
        
        mask = np.zeros(size, dtype=np.uint8)
        x1, y1, x2, y2 = self.lesion_bbox
        mask[y1:y2, x1:x2] = 255
        
        # Convert to PIL Image
        mask_image = Image.fromarray(mask, mode='L')

        self.lesion_bbox_mask = mask_image
    
    def load_inpainting_pipeline(self, model_id:str="benetraco/ms-lesion-inpainting-vh-shifts-wmh2017_v2", device:str="cuda"):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
        # pipe.set_progress_bar_config(disable=True)
        return pipe.to(device)
    
    def plot_image(self):
        """
        Displays the original image, the mask, the generated image with the lesion, the difference image, and the segmented lesion.
        """
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(self.healthy_brain, cmap='gray')
        axes[0].set_title("Healthy Brain")
        axes[0].axis('off')

        axes[1].imshow(self.lesion_bbox_mask, cmap='gray')
        axes[1].set_title("Lesion BBox Mask")
        axes[1].axis('off')

        axes[2].set_title("Lesioned Brain")
        axes[2].axis('off')

        axes[3].imshow(self.lesion_mask, cmap='gray')
        axes[3].set_title("Lesion Mask")
        axes[3].axis('off')

        axes[4].imshow(self.segmented_lesion)
        axes[4].set_title("Segmented Lesion")
        axes[4].axis('off')

        plt.tight_layout()
        plt.show()