# Generation of Synthetic Lesions in Brain MRI (Magnetic Resonance Imaging) Images Using Generative Deep Learning Techniques

This repository contains the code and documentation for my Bachelor's Thesis (**Treball de Fi de Grau**) in Data Science and Engineering at the Universitat Politècnica de Catalunya (UPC), conducted in collaboration with the [ViCOROB](https://vicorob.udg.edu) research group at the University of Girona.

---

## Overview

Multiple Sclerosis (MS) is a chronic neurological disease characterized by white matter lesions visible in brain MRI scans. Accurate detection and segmentation of these lesions is critical for diagnosis and monitoring but is hindered by the scarcity of annotated data and scanner variability.

This project investigates **generative diffusion models** as a robust method for creating realistic synthetic brain MRI slices and controllable MS lesions. The goal is to provide high-quality, domain-adapted synthetic data to improve the robustness and generalizability of automated MS lesion segmentation models.

Key contributions include:
- **Whole-brain MRI synthesis:** Unconditional and conditional diffusion pipelines (DDPM, LDM, and fine-tuned Stable Diffusion) adapted for generating realistic brain slices.
- **MS lesion inpainting:** A diffusion-based inpainting model that inserts synthetic MS lesions into healthy brain regions using user-defined bounding boxes.
- **Automatic mask recovery:** A novel approach to extract accurate lesion masks from generated images for paired data creation.
- **Evaluation:** A comprehensive validation combining standard generative metrics (FID, CMMD, LPIPS) and a blinded visual assessment by radiologists and neurologists.

The full pipeline enables controllable, scanner-aware synthesis of paired brain MRI images and lesion masks, supporting robust training and evaluation of MS lesion segmentation algorithms.

---
## Repository Structure

- `dataset/`  
  Notebooks and scripts for preprocessing the raw MRI data into 2D slices, preparing them for model training and evaluation.

- `experiments/`
  - `brain_generation/`  
    Code for developing and training models for **whole brain MRI synthesis**:
    - `ddpm/` — Implementation of Denoising Diffusion Probabilistic Models (DDPMs) from scratch at multiple resolutions, with model cards.
    - `latent/` — Implementation of Latent Diffusion Models (LDMs) from scratch for more efficient high-resolution synthesis.
    - `latent_finetuning/` — Fine-tuning Stable Diffusion models for brain MRI generation using dataset-specific and scanner-specific prompts.
    - `evaluation/` — Scripts and notebooks for sampling generated images from trained models and computing quantitative metrics (FID, CMMD, LPIPS).

  - `lesion_inpainting/`  
    Code for **MS lesion inpainting**:
    - `dreambooth_inpaint/` — Training scripts for a DreamBooth-based Stable Diffusion inpainting model to insert synthetic lesions guided by masks.
    - `evaluation/` — Scripts to generate inpainted samples, recover lesion masks, and log results for expert validation.

  - `whole_pipeline/`  
    Notebook and scripts that run the **complete pipeline**: generate a healthy brain MRI slice, inpaint synthetic lesions, and automatically recover the ground truth lesion mask — producing paired images and masks for training or testing segmentation models.

- `report/`  
  LaTeX source files for the final bachelor’s thesis manuscript.

- `requirements.txt`  
  List of Python dependencies.

- `README.md`  
  This file.
