import yaml
import itertools
import math
import os
import logging
from pathlib import Path
import wandb

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
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    # DiffusionPipeline,
    # DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from sklearn.model_selection import train_test_split

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

# Restrict PyTorch to use only GPU X
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Set the device to 0 (because it's now the only visible device)
torch.cuda.set_device(0)

def log_images_wandb(accelerator, pixel_values, masks, masked_images, generated_image, global_step):
    batch_size = pixel_values.shape[0]
    for j in range(batch_size):
        images = []
        images.append(wandb.Image(pixel_values[j], mode='F', caption="GT"))
        images.append(wandb.Image(masks[j], mode='F', caption="Mask"))
        images.append(wandb.Image(masked_images[j], mode='F', caption="Masked"))
        images.append(wandb.Image(generated_image[j], mode='F', caption="Generated"))
        images.append(wandb.Image(generated_image[j][0], mode='F', caption="Generated one-channel"))
        accelerator.log({f"Validation Sample {j}": images}, step=global_step)
    
def validate_model(unet, vae, text_encoder, noise_scheduler, val_dataloader, global_step, accelerator, weight_dtype, logger, args):
    unet.eval()
    vae.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            pixel_values = batch["pixel_values"].to(accelerator.device)
            masks = batch["masks"].to(accelerator.device)
            masked_images = batch["masked_images"].to(accelerator.device)
            input_ids = batch["input_ids"].to(accelerator.device)

            # Encode masked images into latents
            masked_latents = vae.encode(masked_images.reshape(pixel_values.shape).to(dtype=weight_dtype)).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            if masks.dim() == 3:
                latent_masks = masks.unsqueeze(1)
            else:
                latent_masks = masks
            latent_masks = torch.nn.functional.interpolate(
                latent_masks, size=(masked_latents.shape[-2], masked_latents.shape[-1]), mode="nearest"
            )

            # Prepare input to U-Net
            noise = torch.randn_like(masked_latents)
            noisy_latents = noise_scheduler.add_noise(masked_latents, noise, timesteps=torch.tensor([50], device=accelerator.device))
            
            latent_model_input = torch.cat([noisy_latents, latent_masks, masked_latents], dim=1)

            # Get text embeddings
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict lesion generation
            predicted_noise = unet(latent_model_input, torch.tensor([50], device=accelerator.device), encoder_hidden_states).sample

            # Denoise the image using the VAE decoder
            generated_latents = masked_latents - predicted_noise  # Assuming noise prediction
            generated_image = vae.decode(generated_latents / vae.config.scaling_factor).sample

            # # Normalize for logging --> No need for normalization if mode='F' is used in wandb.Image
            # masked_images = (masked_images + 1) / 2  # Convert from [-1,1] to [0,1]
            # generated_image = (generated_image + 1) / 2  # Same normalization
            # pixel_values = (pixel_values + 1) / 2  # Same normalization

            # Save/log images
            if args.log_with == "wandb":
                log_images_wandb(accelerator, pixel_values.cpu(), masks.cpu(), masked_images.cpu(), generated_image.cpu(), global_step)
            else:
                ValueError(f"Logging with '{args.log_with}' is not supported yet. Please use 'wandb'.")

    unet.train()
    vae.train()


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

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
    ):
        self.size = size
        self.tokenizer = tokenizer

        # self.instance_data_root = Path(instance_data_root)
        # self.mask_data_root = Path(mask_data_root)
        # if not self.instance_data_root.exists() or not self.mask_data_root.exists():
        #     raise ValueError("Instance images root or mask images root doesn't exists.")

        # self.image_paths = sorted(list(self.instance_data_root.iterdir()))
        # self.mask_paths = sorted([self.mask_data_root / img.name for img in self.image_paths]) # Corresponding masks for each image
        
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

        # Load instance image
        # instance_image = Image.open(self.images_path[index % self.num_instance_images])
        instance_image = Image.open(self.image_paths[index]).convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        # Load lesion mask
        lesion_mask = Image.open(self.mask_paths[index]).convert("L")
        lesion_mask = self.image_transforms_resize_and_crop(lesion_mask)
        lesion_mask = transforms.ToTensor()(lesion_mask)
        lesion_mask = (lesion_mask > 0.5).float()

        # Create masked image
        instance_image_tensor = self.image_transforms(instance_image)
        lesion_mask_expanded = lesion_mask.expand_as(instance_image_tensor) # Expand mask to the same shape as the image (3 channels)
        masked_image = instance_image_tensor * (1 - lesion_mask_expanded)

        # Store processed data
        example["PIL_images"] = instance_image
        example["instance_images"] = instance_image_tensor
        example["lesion_masks"] = lesion_mask
        example["masked_images"] = masked_image

        # Encode text prompt
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

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
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
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

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
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
    all_image_paths = sorted(list(Path(args.instance_data_dir).iterdir()))
    all_mask_paths = sorted([Path(args.mask_data_dir) / img.name for img in all_image_paths])

    # calculate the test_size so there is only one batch of size args.val_batch_size
    val_split = args.val_batch_size / len(all_image_paths)
    # Split dataset into train and validation
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        all_image_paths, all_mask_paths, test_size=val_split, random_state=args.seed
    )

    # Training dataset
    train_dataset = MSInpaintingDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    # Validation dataset
    val_dataset = MSInpaintingDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
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
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        return batch

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Data loaded successfully. Length of dataset: {len(train_dataset)}. Length of dataloader: {len(train_dataloader)}")

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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.num_train_epochs), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Epochs")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = [] # Store the loss for logging
        pbar = tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)

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
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

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
                    if global_step % args.validation_steps == 0: # Validate the model
                        validate_model(unet, vae, text_encoder, noise_scheduler, val_dataloader, global_step, accelerator, weight_dtype, logger, args)
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            
            pbar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        progress_bar.update(1)
        pbar.close()
        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
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