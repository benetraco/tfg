import yaml
import itertools
import math
import os
import logging
from pathlib import Path
import wandb
import random
# Restrict PyTorch to use only GPU X
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionInpaintPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

# Set the device to 0 (because it's now the only visible device)
torch.cuda.set_device(0)

def mse_lesion_loss(image, target, mask):
    """Compute the MSE loss between the image and target, only considering the lesion area."""
    return F.mse_loss(image * mask, target * mask)

def log_validation_dataset(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset):
    """Validates the model, logs 8 images from top 50 mask cases (randomly selected), and computes average loss over all data."""
    logger.info("Running validation... \n Generating 1 image per input. Logging 8 random examples among top 50 mask sizes.")

    # Setup pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Compute mask sums and select top 50
    mask_sums = [val_dataset[i]["lesion_masks"].sum().item() for i in range(len(val_dataset))]
    top_50_indices = sorted(range(len(mask_sums)), key=lambda i: mask_sums[i], reverse=True)[:50]
    
    # Deterministic random sampling for logging
    rng = random.Random(args.seed)
    selected_for_logging = rng.sample(top_50_indices, 8)

    total_mse_losses = []
    all_logs = []
    log_pbar = tqdm(total=len(val_dataset), desc="Validation", position=0, leave=True)

    for idx in range(len(val_dataset)):
        # Load tensors
        input_image_tensor = val_dataset[idx]["instance_images"].to(accelerator.device)
        mask_tensor = val_dataset[idx]["lesion_masks"].to(accelerator.device)

        # Use PIL input and mask
        input_image_pil = val_dataset[idx]["PIL_images"]
        mask_pil = transforms.ToPILImage()(mask_tensor.cpu())

        # Generate image
        with torch.autocast("cuda"):
            image = pipeline(
                args.validation_prompt,
                image=input_image_pil,
                mask_image=mask_pil,
                num_inference_steps=25,
                generator=generator
            ).images[0]

            # Resize if needed
            if image.size != input_image_pil.size:
                image = image.resize(input_image_pil.size)

        # Compute loss for all cases
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(accelerator.device)
        mse_loss = mse_lesion_loss(image_tensor, input_image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)).item()
        total_mse_losses.append(mse_loss)

        # Log image (no loss) only for selected 8
        if idx in selected_for_logging:
            logs = {
                f"Validation {len(all_logs)}": [
                    wandb.Image(input_image_pil, caption="Input"),
                    wandb.Image(mask_pil, caption="Mask"),
                    wandb.Image(image, caption="Generated")
                ]
            }
            all_logs.append(logs)

        log_pbar.update(1)
        log_pbar.set_postfix({"Validation loss": mse_loss})

    log_pbar.close()

    # Final average MSE loss over all images
    final_mse_loss = sum(total_mse_losses) / len(total_mse_losses)

    # Log to wandb
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            for logs in all_logs:
                tracker.log(logs, step=global_step)
            tracker.log({"Validation loss": final_mse_loss}, step=global_step)
        else:
            raise ValueError(f"Tracker '{accelerator.tracker.name}' is not supported for validation logging.")

    del pipeline
    torch.cuda.empty_cache()

def log_validation_guidance_effects(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset):
    """
    Logs validation images using different guidance values for a fixed subset of validation samples.
    """
    logger.info("Running classifier-free guidance validation... Logging 8 images across multiple guidance values.")

    # Create pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Use fixed seed for reproducibility
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Select top 50 by mask size and then sample 8
    mask_sums = [val_dataset[i]["lesion_masks"].sum().item() for i in range(len(val_dataset))]
    top_50_indices = sorted(range(len(mask_sums)), key=lambda i: mask_sums[i], reverse=True)[:50]
    rng = random.Random(args.seed)
    selected_for_logging = rng.sample(top_50_indices, 8)

    guidance_values = [0.0, 1.0, 3.0, 5.0]
    all_logs = []

    for idx in selected_for_logging:
        input_image_pil = val_dataset[idx]["PIL_images"]
        mask_pil = transforms.ToPILImage()(val_dataset[idx]["lesion_masks"].cpu())

        row_log = {}
        row_log["Input"] = wandb.Image(input_image_pil, caption="Input")
        row_log["Mask"] = wandb.Image(mask_pil, caption="Mask")

        for guidance in guidance_values:
            with torch.autocast("cuda"):
                image = pipeline(
                    args.validation_prompt,
                    image=input_image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=25,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]

            # Resize if needed
            if image.size != input_image_pil.size:
                image = image.resize(input_image_pil.size)

            row_log[f"Guidance {guidance}"] = wandb.Image(image, caption=f"Guidance {guidance}")

        all_logs.append({f"CF Guidance Sample {idx}": list(row_log.values())})

    # Log to wandb
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            for log_entry in all_logs:
                tracker.log(log_entry, step=global_step)

    del pipeline
    torch.cuda.empty_cache()

def log_validation_by_domain_guidance(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset):
    """
    Logs up to 3 validation samples per dataset (VH, SHIFTS, WMH2017) using the correct prompt
    and shows results across multiple guidance values in a single W&B row per image.
    """
    logger.info("Running multi-guidance validation per domain with grouped logging.")

    from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
    from torchvision import transforms
    import wandb

    # Build pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    guidance_values = [0.0, 1.0, 3.0, 5.0]
    dataset_prompts = {
        "SHIFTS": "SHIFTS multiple sclerosis lesion in a FLAIR MRI",
        "WMH2017": "WMH2017 multiple sclerosis lesion in a FLAIR MRI",
        "VH": "VH multiple sclerosis lesion in a FLAIR MRI"
    }

    lesion_candidates = {"SHIFTS": [], "VH": [], "WMH2017": []}
    for idx in range(len(val_dataset)):
        filename = val_dataset[idx]["image_paths"].name.lower()
        if "dev" in filename or "test" in filename or "eval" in filename:
            tag = "SHIFTS"
        elif "wmh2017" in filename:
            tag = "WMH2017"
        elif "vh" in filename:
            tag = "VH"
        else:
            continue
        lesion_sum = val_dataset[idx]["lesion_masks"].sum().item()
        lesion_candidates[tag].append((idx, lesion_sum))

    # Select up to 3 top lesions per domain
    selected_samples = {
        tag: [i for i, _ in sorted(lesion_candidates[tag], key=lambda x: x[1], reverse=True)[:3]]
        for tag in dataset_prompts
    }

    all_logs = []

    for tag, indices in selected_samples.items():
        prompt = dataset_prompts[tag]
        for idx in indices:
            input_image_pil = val_dataset[idx]["PIL_images"]
            mask_pil = transforms.ToPILImage()(val_dataset[idx]["lesion_masks"].cpu())

            row_log = {}
            row_log["Input"] = wandb.Image(input_image_pil, caption="Input")
            row_log["Mask"] = wandb.Image(mask_pil, caption="Mask")

            for guidance in guidance_values:
                with torch.autocast("cuda"):
                    result = pipeline(
                        prompt,
                        image=input_image_pil,
                        mask_image=mask_pil,
                        num_inference_steps=25,
                        guidance_scale=guidance,
                        generator=generator,
                    ).images[0]

                if result.size != input_image_pil.size:
                    result = result.resize(input_image_pil.size)

                row_log[f"Guidance {guidance}"] = wandb.Image(result, caption=f"{tag} | guidance={guidance}")

            all_logs.append({f"{tag} Sample {idx}": list(row_log.values())})

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            for log_row in all_logs:
                tracker.log(log_row, step=global_step)

    del pipeline
    torch.cuda.empty_cache()

def log_token_embedding_similarity(text_encoder, tokenizer, accelerator, step):
    """
    Logs cosine similarity between the dataset-specific prompts.
    """
    prompts = ["SHIFTS multiple sclerosis lesion in a FLAIR MRI",
               "VH multiple sclerosis lesion in a FLAIR MRI",
               "WMH2017 multiple sclerosis lesion in a FLAIR MRI"]

    with torch.no_grad():
        embedding_layer = text_encoder.get_input_embeddings()

        embeddings = []
        for prompt in prompts:
            token_ids = tokenizer(prompt, return_tensors="pt").input_ids
            token_ids = token_ids.to(accelerator.device)
            embedding = embedding_layer(token_ids).mean(dim=1)
            embeddings.append(embedding)

        # Compute cosine similarities
        sim_shifts_vh = F.cosine_similarity(embeddings[0], embeddings[1]).item()
        sim_shifts_wmh = F.cosine_similarity(embeddings[0], embeddings[2]).item()
        sim_vh_wmh = F.cosine_similarity(embeddings[1], embeddings[2]).item()

        if accelerator.is_main_process:
            log_data = {
                "sim/SHIFTS–VH": sim_shifts_vh,
                "sim/SHIFTS–WMH2017": sim_shifts_wmh,
                "sim/VH–WMH2017": sim_vh_wmh,
            }
            if accelerator.trackers:
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(log_data, step=step)
            else:
                print(f"[Step {step}] Cosine similarities:")
                for k, v in log_data.items():
                    print(f"  {k}: {v:.4f}")

def prepare_mask_and_masked_image(image, mask, black_mask=True, discretize_mask=True):
    image = np.array(image.convert("RGB"))
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    if discretize_mask:
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
    else:
        mask[mask < 0.0] = 0
        mask[mask >= 1.0] = 1
    mask = torch.from_numpy(mask)

    if black_mask:
        masked_image = image * (mask < 0.5) + (mask >= 0.5) * -1 if discretize_mask else image * (1 - mask)
    else:
        masked_image = image * (mask < 0.5)

    return mask, masked_image


def parse_args():
    config_path = "config_dreambooth_inpaint.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Convert values in config to the correct types (float or int)
    def convert_values(config_dict):
        for key, value in config_dict.items():
            # If the value is a string that can be converted to float or int, convert it
            if isinstance(value, str):
                try:
                    # Try converting to float first (for cases like '5e-6')
                    config_dict[key] = float(value) if '.' in value or 'e' in value else int(value)
                except ValueError:
                    # If conversion fails, leave as string
                    pass
            # Recursively process dicts (in case there are nested structures)
            elif isinstance(value, dict):
                convert_values(value)

    # Convert all values
    convert_values(config)

    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = Args(config)
    
    # Ensure local_rank consistency with environment variable
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if hasattr(args, "local_rank") and env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Validate required arguments
    if not hasattr(args, "instance_data_dir") or args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")
        
    return args


class MSInpaintingDataset(Dataset):
    """
    Custom dataset for MS lesion inpainting. Loads paired FLAIR MRI images and corresponding lesion masks.
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        image_paths, # List of image file paths
        mask_paths, # List of corresponding mask file paths
        instance_prompt,
        tokenizer,
        size=512,
        black_mask=True,
        discretize_mask=True,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.black_mask = black_mask
        self.discretize_mask = discretize_mask
        
        self.image_paths = image_paths  # List of image file paths
        self.mask_paths = mask_paths  # List of corresponding mask file paths


        # Ensure there are corresponding masks for each image
        assert all(mask.exists() for mask in self.mask_paths), "Some masks are missing for the images!"

        self.num_instance_images = len(self.image_paths)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_path = self.image_paths[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)
        
        # instance_prompt = self.instance_prompt
        # Assign prompt based on image name (you can customize this)
        filename = instance_image_path.name.lower()
        if "dev" in filename or "test" in filename or "eval" in filename:
            instance_prompt = f"SHIFTS multiple sclerosis lesion in a FLAIR MRI"
        elif "WMH2017" in filename:
            instance_prompt = f"WMH2017 multiple sclerosis lesion in a FLAIR MRI"
        elif "VH" in filename:
            instance_prompt = f"VH multiple sclerosis lesion in a FLAIR MRI"
        else:
            instance_prompt = f"multiple sclerosis lesion in a FLAIR MRI"


        mask = Image.open(self.mask_paths[index])
        if not mask.mode == "L":
            mask = mask.convert("L")
        mask = self.image_transforms_resize_and_crop(mask)
        # prepare mask and masked image
        mask, masked_image = prepare_mask_and_masked_image(instance_image, mask, black_mask=self.black_mask, discretize_mask=self.discretize_mask)

        example["lesion_masks"] = mask
        example["masked_images"] = masked_image
        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        example["image_paths"] = self.image_paths[index]


        return example


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Using {accelerator.device.type} device")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    
    # Add special tokens (dataset-specific) to tokenizer
    new_tokens = ["SHIFTS", "VH", "WMH2017"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize embeddings for new tokens from the "MRI" token
    embedding_layer = text_encoder.get_input_embeddings()
    base_token = "MRI"
    base_token_id = tokenizer.convert_tokens_to_ids(base_token)
    base_embedding = embedding_layer.weight[base_token_id].detach().clone()

    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        with torch.no_grad():
            # Initialize by adding slight noise to differentiate
            embedding_layer.weight[token_id] = base_embedding + 0.01 * torch.randn_like(base_embedding)
        print(f"Token '{token}' -> ID {token_id}, initialized from '{base_token}'")

    # Load models and create wrapper for stable diffusion
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    logger.info("Models loaded successfully")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    logger.info("Optimizer and scheduler created")

    # Load dataset
    train_image_paths = sorted(list(Path(args.instance_data_dir).iterdir()))
    train_mask_paths = sorted([Path(args.mask_data_dir) / img.name for img in train_image_paths])
    val_image_paths = sorted(list(Path(args.val_input_image_path).iterdir()))
    val_mask_paths = sorted([Path(args.val_mask_image_path) / img.name for img in val_image_paths])

    # Training dataset
    train_dataset = MSInpaintingDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        black_mask=args.black_mask,
        discretize_mask=args.discretize_mask,
    )

    # Validation dataset
    val_dataset = MSInpaintingDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        instance_prompt=args.validation_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        black_mask=args.black_mask,
        discretize_mask=args.discretize_mask,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        masks = [example["lesion_masks"] for example in examples]
        masked_images = [example["masked_images"] for example in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "masks": masks,
            "masked_images": masked_images,
            "instance_images": pixel_values,
        }
        return batch


    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    logger.info(f"Data loaded successfully. Length of train dataset: {len(train_dataset)}. Length of train dataloader: {len(train_dataloader)}. Length of val dataset: {len(val_dataset)}.")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Masked image black: {args.black_mask}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # log_validate before training
    # log_validation_dataset(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset)
    # log_validation_guidance_effects(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset)
    # log_validation_by_domain_guidance(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset)

    start_time = time.time()
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = [] # Store the loss for logging
        pbar = tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space

                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["masks"]  # Already a batch of masks (N, H, W)
                # Ensure masks have the correct shape (N, 1, H, W)
                if masks.dim() == 3:  # If shape is (N, H, W), add a channel dimension
                    masks = masks.unsqueeze(1)  # Convert (N, H, W) → (N, 1, H, W)
                # Resize the masks to match latent space dimensions
                masks = torch.nn.functional.interpolate(
                    masks, size=(args.resolution // 8, args.resolution // 8), mode="nearest"
                )
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # IMPORTANT: The model expects 9 channels: 4-noisy latents, 1-mask, 4-masked latents
                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, masks, masked_latents], dim=1)

                # Get the text embedding for conditioning
                # Randomly drop prompt with 10% chance
                bsz = batch["input_ids"].shape[0]
                use_uncond = torch.rand(bsz) < 0.1  # 10% unconditional
                # Build conditional and unconditional embeddings
                cond_embeds = text_encoder(batch["input_ids"])[0]
                uncond_input_ids = torch.full_like(batch["input_ids"], tokenizer.pad_token_id)
                uncond_embeds = text_encoder(uncond_input_ids)[0]
                # Choose per-sample embedding
                encoder_hidden_states = torch.stack([
                    uncond_embeds[i] if use_uncond[i] else cond_embeds[i]
                    for i in range(bsz)
                ])

                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Logging
                train_loss.append(accelerator.gather(loss.repeat(bsz)).mean().item())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                pbar.update(1)
                global_step += 1
                train_loss = np.mean(train_loss)
                accelerator.log({"loss": train_loss, "log-loss": np.log(train_loss)}, step=global_step)

                train_loss = [] # reset the loss for the next accumulation

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            
            if accelerator.sync_gradients: # and (global_step % 50 == 0):
                log_token_embedding_similarity(text_encoder, tokenizer, accelerator, global_step)

            pbar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        pbar.close()
        accelerator.wait_for_everyone()

        # Log images to validate the model
        # if (epoch + 1) % args.validation_epochs == 0:
        #     if accelerator.is_main_process:
        #         # log_validation_dataset(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset)
        #         # log_validation_guidance_effects(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset)
        #         log_validation_by_domain_guidance(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, val_dataset)

    logger.info("Training complete.")
    end_time = time.time()
    logger.info(f"Total training time: {end_time - start_time:.2f} seconds, {((end_time - start_time) / 60):.2f} minutes.")
    
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            tokenizer=tokenizer,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()