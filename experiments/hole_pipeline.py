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
from pathlib import Path
from itertools import product
import time
import csv


# class CreateSample:
#     def __init__(self, scanner: str, real_brain: bool, model= None, lesion_bbox: Optional[Tuple[int, int, int, int]] = None, guidance_scale: int = 3, inference_steps: int = 25, seed: int = 17844, plot_lesion_histogram: bool = False):
#         assert isinstance(scanner, str) and scanner in ["Siemens", "Philips", "GE"], "Scanner must be one of Siemens, Philips, or GE."
#         assert lesion_bbox is None or (
#             isinstance(lesion_bbox, tuple) and
#             len(lesion_bbox) == 4 and
#             all(isinstance(x, int) and 0 <= x < 256 for x in lesion_bbox)
#         ), "Lesion bounding box must be None or a tuple of 4 integers between 0 and 256."
        
#         self.scanner = scanner
#         self.real_brain = real_brain
#         self.model = model
#         self.lesion_bbox = lesion_bbox
#         self.syntetic_scanner_path = f"/home/benet/tfg/experiments/brain_generation/evaluation/generated_images/latent_finetuning_scanners_healthy/{scanner}/g1.0" # the guidance value 1.0 has the best results in terms of FID and CMMD
#         self.real_scanner_path = f"/home/benet/data/biomarkem2D/test/{scanner}"
#         self.guidance_scale = guidance_scale
#         self.inference_steps = inference_steps
#         self.seed = seed
#         self.plot_lesion_histogram = plot_lesion_histogram
        
#         if self.model is None:
#             self.model = self.load_inpainting_pipeline()
#         self.healthy_brain = self.get_brain_image()
#         self.bkgnd_mask, self.csf_mask, self.wm_gm_mask = self.get_brain_tissues_mask()
#         if self.lesion_bbox is None:
#             self.lesion_bbox = self.random_bbox()
#         self.lesion_bbox_mask = self.generate_mask(self.healthy_brain.size)
#         self.lesioned_brain = self.get_lesion_image()
#         self.lesion_mask, self.diff_bbox = self.calculate_difference()
#         self.segmented_lesion = self.segment_lesion()

#     def segment_lesion(self):
#         """"""
#         # in the image with lesion, paint a red line in the lesion area from the difference image
#         segmented_lesion = np.array(self.lesioned_brain)
#         lesion_mask = np.array(self.lesion_mask)
#         contours = measure.find_contours(lesion_mask > 0, 0.5)
#         for contour in contours:
#             for y, x in contour.astype(np.int32):
#                 if 0 <= x < segmented_lesion.shape[1] and 0 <= y < segmented_lesion.shape[0]:
#                     # paint the pixel red
#                     segmented_lesion[y, x] = [255, 0, 0]
#         # convert to PIL Image
#         segmented_lesion = Image.fromarray(segmented_lesion)


#         return segmented_lesion

#     def calculate_difference(self):
#         """
#         Calculates the difference between the original healthy brain image and the generated image with the lesion.
#         """
#         if self.healthy_brain.size != self.lesioned_brain.size:
#             self.lesioned_brain = self.lesioned_brain.resize(self.healthy_brain.size, Image.LANCZOS)

#         healthy_tensor = to_tensor(self.healthy_brain)
#         lesioned_tensor = to_tensor(self.lesioned_brain)

#         diff_tensor = lesioned_tensor - healthy_tensor
#         diff_image = diff_tensor.mean(dim=0).numpy()

#         assert self.lesion_bbox is not None, "Lesion bounding box must be defined."
#         x1, y1, x2, y2 = self.lesion_bbox
#         diff_bbox = diff_image[y1:y2, x1:x2]

#         # Normalize to [0, 1], then scale to [0, 255] and convert to uint8
#         diff_bbox = (diff_bbox - diff_bbox.min()) / (diff_bbox.max() - diff_bbox.min())
#         diff_bbox_uint8 = (diff_bbox * 255).astype(np.uint8)

#         # Otsu thresholding
#         otsu_thresh, diff_bbox_bin = cv2.threshold(diff_bbox_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # Optional histogram plotting (call this outside if needed)
#         if self.plot_lesion_histogram:
#             self.plot_diff_histogram(diff_bbox_uint8, otsu_thresh)
#             print(f"Otsu's threshold: {otsu_thresh} ({otsu_thresh / 255:.3f} normalized)")

#         # Morphological operations
#         diff_bbox_bin = cv2.morphologyEx(diff_bbox_bin, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

#         # Paste into full-size mask
#         lesion_mask = np.zeros_like(np.array(self.lesion_bbox_mask))
#         lesion_mask[y1:y2, x1:x2] = diff_bbox_bin

#         # Convert to PIL images
#         diff_bbox_image = Image.fromarray(diff_bbox_bin, mode='L')
#         lesion_mask_image = Image.fromarray(lesion_mask, mode='L')

#         return lesion_mask_image, diff_bbox_image

#     def plot_diff_histogram(self, diff_bbox_uint8, otsu_thresh=None):
#         """
#         Plots the histogram of the difference image, optionally overlaying Otsu's threshold.
#         """
#         plt.hist(diff_bbox_uint8.ravel(), bins=256, color='blue', alpha=0.7)
#         if otsu_thresh is not None:
#             plt.axvline(otsu_thresh, color='red', linestyle='dashed', linewidth=1, label=f"Otsu's threshold = {otsu_thresh:.2f}")
#             plt.legend()
#         plt.title("Histogram of the Difference Image")
#         plt.xlabel("Pixel Value")
#         plt.ylabel("Frequency")
#         plt.show()
    
#     def get_brain_image(self):
#         """
#         Returns a healthy brain image from the specified scanner path.
#         If `real_brain` is True, it uses a real brain image; otherwise, it uses a synthetic one.
#         """
#         np.random.seed(self.seed)
        
#         if self.real_brain:
#             images = os.listdir(self.real_scanner_path)
#             image_name = np.random.choice(images)
#             image_path = os.path.join(self.real_scanner_path, image_name)
#             image = Image.open(image_path).convert("L")
#         else:
#             images = os.listdir(self.syntetic_scanner_path)
#             image_name = np.random.choice(images)
#             image_path = os.path.join(self.syntetic_scanner_path, image_name)
#             image = Image.open(image_path).convert("L")
        
#         assert image.size == (256, 256), "Image size must be 256x256 pixels."
#         return image
    
#     def in_mask_brain(self):
#         """
#         Checks if the lesion bounding box is within the white matter and gray matter mask.
#         Returns True if the lesion is predominantly within the WM/GM mask, otherwise False.
#         """        
#         if not self.lesion_bbox:
#             return False
        
#         # Check if lesion bbox is within the WM/GM mask
#         x1, y1, x2, y2 = self.lesion_bbox
#         lesion_bbox_mask = self.wm_gm_mask[y1:y2, x1:x2]

#         # return True if:
#         # a 90% of the lesion mask is within the WM/GM mask,
#         # 100% is outside the background mask,
#         # and 95% is outside the CSF mask.
#         ratio_wm_gm = np.sum(lesion_bbox_mask > 0) / lesion_bbox_mask.size
#         ratio_bkgnd = np.sum(self.bkgnd_mask[y1:y2, x1:x2] > 0) / lesion_bbox_mask.size
#         ratio_csf = np.sum(self.csf_mask[y1:y2, x1:x2] > 0) / lesion_bbox_mask.size
#         if ratio_wm_gm >= 0.9 and ratio_bkgnd == 0 and ratio_csf <= 0.05:
#             print(f"Lesion bbox {self.lesion_bbox} is predominantly within the WM/GM mask.")
#             return True
#         else:
#             print(f"Lesion bbox {self.lesion_bbox} is NOT predominantly within the WM/GM mask. Ratios: WM/GM={ratio_wm_gm:.2f}, Bkgnd={ratio_bkgnd:.2f}, CSF={ratio_csf:.2f}")
#             return False
    
#     def get_brain_tissues_mask(self):
#         img_arr = np.array(self.healthy_brain).astype(np.float32)

#         # Segment using ANTs
#         img = ants.from_numpy(img_arr)
#         mask = ants.get_mask(img)
#         seg = ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask, verbose=0)

#         # Classify tissues by intensity
#         seg_img = seg["segmentation"]
#         labels = np.unique(seg_img.numpy())
#         means = [img.numpy()[seg_img.numpy() == l].mean() for l in labels]
#         sorted_labels = [label for _, label in sorted(zip(means, labels))]

#         # Combine WM and GM (highest two intensity classes)
#         wm_gm_mask = np.isin(seg_img.numpy(), sorted_labels[-2:]).astype(np.uint8) * 255
#         # Create masks for CSF and background
#         bkgnd_mask = np.isin(seg_img.numpy(), sorted_labels[0]).astype(np.uint8) * 255
#         csf_mask = np.isin(seg_img.numpy(), sorted_labels[1]).astype(np.uint8) * 255

#         # do a closing operation to fill small holes in the WM/GM mask
#         wm_gm_mask_morph = cv2.morphologyEx(wm_gm_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

#         # do an opening operation to remove small noise in the CSF mask
#         csf_mask_morph = cv2.morphologyEx(csf_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

#         # do a dilation operation to have some more separation between the background and the brain
#         bkgnd_mask_morph = cv2.dilate(bkgnd_mask, np.ones((10, 10), np.uint8), iterations=1)
        
#         #plot the masks (and both original and morph WM/GM and CSF masks)
#         # fig, axes = plt.subplots(1, 6, figsize=(20, 4))
#         # axes[0].imshow(bkgnd_mask, cmap='gray')
#         # axes[0].set_title("Background Mask")
#         # axes[0].axis('off')

#         # axes[1].imshow(bkgnd_mask_morph, cmap='gray')
#         # axes[1].set_title("Background Mask (Morphological)")
#         # axes[1].axis('off')

#         # axes[2].imshow(csf_mask, cmap='gray')
#         # axes[2].set_title("CSF Mask")
#         # axes[2].axis('off')

#         # axes[3].imshow(csf_mask_morph, cmap='gray')
#         # axes[3].set_title("CSF Mask (Morphological)")
#         # axes[3].axis('off')

#         # axes[4].imshow(wm_gm_mask, cmap='gray')
#         # axes[4].set_title("WM/GM Mask (Original)")
#         # axes[4].axis('off')

#         # axes[5].imshow(wm_gm_mask_morph, cmap='gray')
#         # axes[5].set_title("WM/GM Mask (Morphological)")
#         # axes[5].axis('off')

#         # plt.tight_layout()
#         # plt.show()
        
#         return bkgnd_mask_morph, csf_mask_morph, wm_gm_mask_morph
    
#     def random_bbox(self, min_size: int = 16, max_size: int = 64, max_attempts: int = 100):
#         """
#         Generate a random rectangular bounding box (width and height between min_size and max_size)
#         that lies at least 90% inside the WM/GM mask.
#         """
#         if self.wm_gm_mask is None:
#             raise ValueError("WM/GM mask must be computed before generating random bbox.")
        
#         mask = self.wm_gm_mask
#         h, w = mask.shape

#         for attempt in range(max_attempts):
#             box_w = np.random.randint(min_size, max_size + 1)
#             box_h = np.random.randint(min_size, max_size + 1)

#             if box_w >= w or box_h >= h:
#                 continue  # skip impossible sizes

#             x1 = np.random.randint(0, w - box_w)
#             y1 = np.random.randint(0, h - box_h)
#             x2 = x1 + box_w
#             y2 = y1 + box_h

#             region = mask[y1:y2, x1:x2]

#             # Check if at least 90% of the region is within the WM/GM mask
#             # and 100% is outside the background mask and 95% is outside the CSF mask
#             ratio_wm_gm = np.sum(region > 0) / region.size
#             ratio_bkgnd = np.sum(self.bkgnd_mask[y1:y2, x1:x2] > 0) / region.size
#             ratio_csf = np.sum(self.csf_mask[y1:y2, x1:x2] > 0) / region.size
#             if ratio_wm_gm >= 0.9 and ratio_bkgnd == 0 and ratio_csf <= 0.05:
#                 print(f"Generated bounding box: {x1}, {y1}, {x2}, {y2} (Attempt {attempt + 1}). Ratios: WM/GM={ratio_wm_gm:.2f}, Bkgnd={ratio_bkgnd:.2f}, CSF={ratio_csf:.2f}")
#                 self.lesion_bbox = (x1, y1, x2, y2)
#                 # self.generate_mask(mask.shape)
#                 # self.lesion_bbox_mask = self.lesion_bbox_mask.resize((256, 256), Image.LANCZOS)
#                 return self.lesion_bbox
#         print(f"Failed to generate a valid bounding box after {max_attempts} attempts.")


#         raise RuntimeError(f"Could not find a valid bounding box within {max_attempts} attempts.")
    
#     def get_lesion_image(self):        

#         # if image in one channel, convert to RGB
#         if self.healthy_brain.mode == 'L':
#             self.healthy_brain = self.healthy_brain.convert('RGB')

#         lesioned_image = self.model(
#             prompt="SHIFTS multiple sclerosis lesion in a FLAIR MRI",
#             image=self.healthy_brain,
#             mask_image=self.lesion_bbox_mask,
#             num_inference_steps=self.inference_steps,
#             guidance_scale=self.guidance_scale,
#             generator=torch.Generator(device="cuda").manual_seed(self.seed)
#         ).images[0]

#         return lesioned_image
    
#     def generate_mask(self, size:tuple):
        
#         mask = np.zeros(size, dtype=np.uint8)
#         x1, y1, x2, y2 = self.lesion_bbox
#         mask[y1:y2, x1:x2] = 255
        
#         # Convert to PIL Image
#         mask_image = Image.fromarray(mask, mode='L')

#         return mask_image
    
#     def load_inpainting_pipeline(self, model_id:str="benetraco/ms-lesion-inpainting-vh-shifts-wmh2017_v2", device:str="cuda"):
#         pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
#         # pipe.set_progress_bar_config(disable=True)
#         return pipe.to(device)
    
#     def plot_image(self):
#         """
#         Displays the original image, the mask, the generated image with the lesion, the difference image, and the segmented lesion.
#         """
#         fig, axes = plt.subplots(1, 5, figsize=(20, 4))
#         axes[0].imshow(self.healthy_brain, cmap='gray')
#         axes[0].set_title("Healthy Brain")
#         axes[0].axis('off')

#         axes[1].imshow(self.lesion_bbox_mask, cmap='gray')
#         axes[1].set_title("Lesion BBox Mask")
#         axes[1].axis('off')

#         axes[2].imshow(self.lesioned_brain)
#         axes[2].set_title("Lesioned Brain")
#         axes[2].axis('off')

#         axes[3].imshow(self.lesion_mask, cmap='gray')
#         axes[3].set_title("Lesion Mask")
#         axes[3].axis('off')

#         axes[4].imshow(self.segmented_lesion)
#         axes[4].set_title("Segmented Lesion")
#         axes[4].axis('off')

#         plt.tight_layout()
#         plt.show()



class CreateSample:
    def __init__(self, scanner: str, real_brain: bool, model=None,
                lesion_bboxes: Optional[list] = None, num_bboxes: int = 1,
                guidance_scale: int = 3, inference_steps: int = 25,
                seed: int = 17844, plot_lesion_histogram: bool = False):
        assert isinstance(scanner, str) and scanner in ["Siemens", "Philips", "GE"], "Scanner must be one of Siemens, Philips, or GE."
        assert lesion_bboxes is None or (
            isinstance(lesion_bbox, tuple) and
            len(lesion_bbox) == 4 and
            all(isinstance(x, int) and 0 <= x < 256 for x in lesion_bbox) for lesion_bbox in lesion_bboxes
        ), "Lesion bounding box must be None or a list of tuples of 4 integers between 0 and 256."
        
        self.scanner = scanner
        self.real_brain = real_brain
        self.model = model
        self.lesion_bboxes = lesion_bboxes
        self.syntetic_scanner_path = f"/home/benet/tfg/experiments/brain_generation/evaluation/generated_images/latent_finetuning_scanners_healthy/{scanner}/g1.0" # the guidance value 1.0 has the best results in terms of FID and CMMD
        self.real_scanner_path = f"/home/benet/data/biomarkem2D/test/{scanner}"
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
        self.seed = seed
        self.plot_lesion_histogram = plot_lesion_histogram
        
        if self.model is None:
            self.model = self.load_inpainting_pipeline()
        self.healthy_brain = self.get_brain_image()
        self.bkgnd_mask, self.csf_mask, self.wm_gm_mask = self.get_brain_tissues_mask()
        if self.lesion_bboxes is None:
            self.lesion_bboxes = self.generate_multiple_bboxes(num_bboxes)
        self.lesion_bbox_mask = self.generate_combined_mask(self.healthy_brain.size)
        self.lesioned_brain = self.get_lesion_image()
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
            self.lesioned_brain = self.lesioned_brain.resize(self.healthy_brain.size, Image.LANCZOS)

        healthy_tensor = to_tensor(self.healthy_brain)
        lesioned_tensor = to_tensor(self.lesioned_brain)

        diff_tensor = lesioned_tensor - healthy_tensor
        diff_image = diff_tensor.mean(dim=0).numpy()

        # assert self.lesion_bbox is not None, "Lesion bounding box must be defined."
        # x1, y1, x2, y2 = self.lesion_bbox
        # diff_bbox = diff_image[y1:y2, x1:x2]

        # # Normalize to [0, 1], then scale to [0, 255] and convert to uint8
        # diff_bbox = (diff_bbox - diff_bbox.min()) / (diff_bbox.max() - diff_bbox.min())
        # diff_bbox_uint8 = (diff_bbox * 255).astype(np.uint8)

        # # Otsu thresholding
        # otsu_thresh, diff_bbox_bin = cv2.threshold(diff_bbox_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # Optional histogram plotting (call this outside if needed)
        # if self.plot_lesion_histogram:
        #     self.plot_diff_histogram(diff_bbox_uint8, otsu_thresh)
        #     print(f"Otsu's threshold: {otsu_thresh} ({otsu_thresh / 255:.3f} normalized)")

        # # Morphological operations
        # diff_bbox_bin = cv2.morphologyEx(diff_bbox_bin, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        # Paste into full-size mask
        lesion_mask = np.zeros_like(np.array(self.lesion_bbox_mask))
        
        for bbox in self.lesion_bboxes:
            x1, y1, x2, y2 = bbox
            diff_bbox = diff_image[y1:y2, x1:x2]
            diff_bbox = (diff_bbox - diff_bbox.min()) / (diff_bbox.max() - diff_bbox.min())
            diff_bbox_uint8 = (diff_bbox * 255).astype(np.uint8)
            _, diff_bbox_bin = cv2.threshold(diff_bbox_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            diff_bbox_bin = cv2.morphologyEx(diff_bbox_bin, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
            lesion_mask[y1:y2, x1:x2] = diff_bbox_bin

        return Image.fromarray(lesion_mask, mode='L'), Image.fromarray((diff_image * 255).astype(np.uint8))

        # return lesion_mask_image, diff_bbox_image

    def plot_diff_histogram(self, diff_bbox_uint8, otsu_thresh=None):
        """
        Plots the histogram of the difference image, optionally overlaying Otsu's threshold.
        """
        plt.hist(diff_bbox_uint8.ravel(), bins=256, color='blue', alpha=0.7)
        if otsu_thresh is not None:
            plt.axvline(otsu_thresh, color='red', linestyle='dashed', linewidth=1, label=f"Otsu's threshold = {otsu_thresh:.2f}")
            plt.legend()
        plt.title("Histogram of the Difference Image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
    
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
        if not self.lesion_bbox:
            return False
        
        # Check if lesion bbox is within the WM/GM mask
        x1, y1, x2, y2 = self.lesion_bbox
        lesion_bbox_mask = self.wm_gm_mask[y1:y2, x1:x2]

        # return True if:
        # a 90% of the lesion mask is within the WM/GM mask,
        # 100% is outside the background mask,
        # and 95% is outside the CSF mask.
        ratio_wm_gm = np.sum(lesion_bbox_mask > 0) / lesion_bbox_mask.size
        ratio_bkgnd = np.sum(self.bkgnd_mask[y1:y2, x1:x2] > 0) / lesion_bbox_mask.size
        ratio_csf = np.sum(self.csf_mask[y1:y2, x1:x2] > 0) / lesion_bbox_mask.size
        if ratio_wm_gm >= 0.9 and ratio_bkgnd == 0 and ratio_csf <= 0.05:
            print(f"Lesion bbox {self.lesion_bbox} is predominantly within the WM/GM mask.")
            return True
        else:
            print(f"Lesion bbox {self.lesion_bbox} is NOT predominantly within the WM/GM mask. Ratios: WM/GM={ratio_wm_gm:.2f}, Bkgnd={ratio_bkgnd:.2f}, CSF={ratio_csf:.2f}")
            return False
    
    def get_brain_tissues_mask(self):
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
        # Create masks for CSF and background
        bkgnd_mask = np.isin(seg_img.numpy(), sorted_labels[0]).astype(np.uint8) * 255
        csf_mask = np.isin(seg_img.numpy(), sorted_labels[1]).astype(np.uint8) * 255

        # do a closing operation to fill small holes in the WM/GM mask
        wm_gm_mask_morph = cv2.morphologyEx(wm_gm_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # do an opening operation to remove small noise in the CSF mask
        csf_mask_morph = cv2.morphologyEx(csf_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # do a dilation operation to have some more separation between the background and the brain
        bkgnd_mask_morph = cv2.dilate(bkgnd_mask, np.ones((10, 10), np.uint8), iterations=1)
        
        #plot the masks (and both original and morph WM/GM and CSF masks)
        # fig, axes = plt.subplots(1, 6, figsize=(20, 4))
        # axes[0].imshow(bkgnd_mask, cmap='gray')
        # axes[0].set_title("Background Mask")
        # axes[0].axis('off')

        # axes[1].imshow(bkgnd_mask_morph, cmap='gray')
        # axes[1].set_title("Background Mask (Morphological)")
        # axes[1].axis('off')

        # axes[2].imshow(csf_mask, cmap='gray')
        # axes[2].set_title("CSF Mask")
        # axes[2].axis('off')

        # axes[3].imshow(csf_mask_morph, cmap='gray')
        # axes[3].set_title("CSF Mask (Morphological)")
        # axes[3].axis('off')

        # axes[4].imshow(wm_gm_mask, cmap='gray')
        # axes[4].set_title("WM/GM Mask (Original)")
        # axes[4].axis('off')

        # axes[5].imshow(wm_gm_mask_morph, cmap='gray')
        # axes[5].set_title("WM/GM Mask (Morphological)")
        # axes[5].axis('off')

        # plt.tight_layout()
        # plt.show()
        
        return bkgnd_mask_morph, csf_mask_morph, wm_gm_mask_morph
    
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

            # Check if at least 90% of the region is within the WM/GM mask
            # and 100% is outside the background mask and 95% is outside the CSF mask
            ratio_wm_gm = np.sum(region > 0) / region.size
            ratio_bkgnd = np.sum(self.bkgnd_mask[y1:y2, x1:x2] > 0) / region.size
            ratio_csf = np.sum(self.csf_mask[y1:y2, x1:x2] > 0) / region.size
            if ratio_wm_gm >= 0.9 and ratio_bkgnd == 0 and ratio_csf <= 0.05:
                print(f"Generated bounding box: {x1}, {y1}, {x2}, {y2} (Attempt {attempt + 1}). Ratios: WM/GM={ratio_wm_gm:.2f}, Bkgnd={ratio_bkgnd:.2f}, CSF={ratio_csf:.2f}")
                self.lesion_bbox = (x1, y1, x2, y2)
                # self.generate_mask(mask.shape)
                # self.lesion_bbox_mask = self.lesion_bbox_mask.resize((256, 256), Image.LANCZOS)
                return self.lesion_bbox
        print(f"Failed to generate a valid bounding box after {max_attempts} attempts.")


        raise RuntimeError(f"Could not find a valid bounding box within {max_attempts} attempts.")
    
    def get_lesion_image(self):        

        # if image in one channel, convert to RGB
        if self.healthy_brain.mode == 'L':
            self.healthy_brain = self.healthy_brain.convert('RGB')

        lesioned_image = self.model(
            prompt="SHIFTS multiple sclerosis lesion in a FLAIR MRI",
            image=self.healthy_brain,
            mask_image=self.lesion_bbox_mask,
            num_inference_steps=self.inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(self.seed)
        ).images[0]

        return lesioned_image
    
    # def generate_mask(self, size:tuple):
        
    #     mask = np.zeros(size, dtype=np.uint8)
    #     x1, y1, x2, y2 = self.lesion_bbox
    #     mask[y1:y2, x1:x2] = 255
        
    #     # Convert to PIL Image
    #     mask_image = Image.fromarray(mask, mode='L')

    #     return mask_image

    def generate_combined_mask(self, size: tuple):
        mask = np.zeros(size, dtype=np.uint8)
        for bbox in self.lesion_bboxes:
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
        return Image.fromarray(mask, mode='L')

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

        axes[2].imshow(self.lesioned_brain)
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

    def generate_multiple_bboxes(self, num_bboxes: int):
        bboxes = []
        attempts = 0
        while len(bboxes) < num_bboxes and attempts < num_bboxes * 10:
            # bbox = self.random_bbox()
            bbox = self.random_periventricular_bbox()
            if bbox and all(self._bbox_iou(bbox, b) < 0.1 for b in bboxes):
                bboxes.append(bbox)
            attempts += 1
        return bboxes

    def _bbox_iou(self, boxA, boxB):
        # compute intersection over union
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
    
    def random_periventricular_bbox(self, min_size=16, max_size=64, max_attempts=1000):
        """
        Generates a bbox near the CSF mask but still inside WM/GM.
        """
        if self.csf_mask is None or self.wm_gm_mask is None:
            raise ValueError("CSF and WM/GM masks must be available")

        # Step 1: Dilate CSF mask to get surrounding area
        dilated_csf = cv2.dilate(self.csf_mask, np.ones((25, 25), np.uint8), iterations=1)

        # Step 2: Subtract CSF to get only the surrounding border (shell)
        csf_border = cv2.subtract(dilated_csf, self.csf_mask)

        # Step 3: Intersect with WM/GM
        near_csf_in_wm = cv2.bitwise_and(csf_border, self.wm_gm_mask)

        # plot the dilated CSF, csf border, and near CSF in WM/GM
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[0].imshow(dilated_csf, cmap='gray')
        # axes[0].set_title("Dilated CSF Mask")
        # axes[0].axis('off')
        # axes[1].imshow(csf_border, cmap='gray')
        # axes[1].set_title("CSF Border")
        # axes[1].axis('off')
        # axes[2].imshow(near_csf_in_wm, cmap='gray')
        # axes[2].set_title("Near CSF in WM/GM")
        # axes[2].axis('off')
        # plt.tight_layout()
        # plt.show()

        h, w = near_csf_in_wm.shape
        for attempt in range(max_attempts):
            box_w = np.random.randint(min_size, max_size + 1)
            box_h = np.random.randint(min_size, max_size + 1)

            if box_w >= w or box_h >= h:
                continue

            x1 = np.random.randint(0, w - box_w)
            y1 = np.random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h

            region = near_csf_in_wm[y1:y2, x1:x2]

            ratio_valid = np.sum(region > 0) / region.size
            if ratio_valid >= 0.7:
                print(f"[✓] Periventricular bbox: {(x1, y1, x2, y2)} attempt {attempt}")
                return (x1, y1, x2, y2)
            # else:
                # print(f"[✗] Periventricular bbox attempt {attempt} failed with ratio {ratio_valid:.2f}. Box: {(x1, y1, x2, y2)}")

        raise RuntimeError("Could not find a valid periventricular bbox.")




def run_and_save_samples(
    scanners=["Siemens", "Philips", "GE"],
    real_options=[True, False],
    guidance_values=[2.0],
    output_root="./lesion_results_mult_bboxes",
    samples_per_combination=50,
    model_id="benetraco/ms-lesion-inpainting-vh-shifts-wmh2017_v2"
):
    output_root = Path(output_root)
    os.makedirs(output_root, exist_ok=True)

    # Load model once
    print("Loading inpainting pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None).to("cuda")

    csv_path = output_root / "timing_results.csv"
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scanner", "Real", "Guidance", "AverageTime(s)", "NumSamples"])

        for scanner, real_brain, guidance in product(scanners, real_options, guidance_values):
            total_time = 0
            successes = 0

            for i in range(samples_per_combination):
                try:
                    seed = np.random.randint(0, 99999)
                    start = time.time()

                    sample = CreateSample(
                        scanner=scanner,
                        real_brain=real_brain,
                        guidance_scale=guidance,
                        seed=seed,
                        model=pipe,
                        num_bboxes=3
                    )

                    end = time.time()
                    total_time += (end - start)
                    successes += 1

                    # Naming scheme
                    tag = f"g{guidance}_s{i}"
                    save_dir = output_root / scanner / ('real' if real_brain else 'synthetic') / tag
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Save images
                    sample.healthy_brain.save(save_dir / "original.png")
                    sample.lesion_bbox_mask.save(save_dir / "bbox_mask.png")
                    sample.lesioned_brain.save(save_dir / "lesioned.png")
                    sample.lesion_mask.save(save_dir / "lesion_mask.png")
                    sample.segmented_lesion.save(save_dir / "segmented.png")
                    with open(save_dir / "seed.txt", "w") as f:
                        f.write(str(seed))

                    print(f"[✓] Saved: {save_dir} | scanner={scanner} | real={real_brain} | g={guidance} | sample={i}")
                except Exception as e:
                    print(f"[!] Failed {scanner} | real={real_brain} | g={guidance} | err={e}")
            
            if successes > 0:
                avg_time = total_time / successes
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write the results for this combination
                    writer.writerow([scanner, real_brain, guidance, round(avg_time, 3), successes])
                print(f"[⏱] Avg time: {avg_time:.2f}s over {successes} samples for {scanner}, real={real_brain}, g={guidance}")

# Run it
# run_and_save_samples()
