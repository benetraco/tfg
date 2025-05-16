from pathlib import Path
import os, sys
# Restrict PyTorch to use only GPU X
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor
import wandb
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL, DiffusionPipeline, DDPMPipeline, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import logging
from accelerate.logging import get_logger
from accelerate import Accelerator
from huggingface_hub import HfApi, create_repo, upload_folder, get_full_repo_name
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from dataset.build_dataset import MRIDataset

check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# Restrict PyTorch to use only GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set the device to 0 (because it's now the only visible device)
torch.cuda.set_device(0)

class LatentFineTuning:
    def __init__(self, config_path):
        """Initialize the latent fine-tuning class."""
        self.repo_path, self.exp_path = self._get_repo_exp_path()
        self.config = self._load_config(config_path)
        self.pipeline_dir = self.repo_path / self.config['saving']['local']['outputs_dir'] / self.config['saving']['local']['pipeline_name']
        self.accelerator = self._setup_accelerator()
        self.dataset, self.train_dataloader = self._setup_dataset()
        self.model, self.vae, self.ldm = self._setup_model()
        self.optimizer, self.lr_scheduler, self.noise_scheduler, self.num_update_steps_per_epoch = self._setup_training()
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)
        self.global_step = 0


    def _get_repo_exp_path(self):
        """Find the root directory of the repository."""
        repo_path = Path.cwd().resolve()
        while '.gitignore' not in os.listdir(repo_path):
            repo_path = repo_path.parent
        sys.path.insert(0, str(repo_path)) if str(repo_path) not in sys.path else None
        exp_path = Path.cwd().resolve()
        print(f"Repo path: {repo_path}, experiment path: {exp_path}")
        return repo_path, exp_path


    def _load_config(self, config_path):
        """Load the configuration file."""
        with open(config_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader)


    def _setup_accelerator(self):
        """Initialize the accelerator and define basic logging configurations."""
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config['training']['gradient_accumulation']['steps'],
            mixed_precision=self.config['training']['mixed_precision']['type'],
            log_with=self.config['logging']['logger_name']
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        return accelerator


    def _setup_dataset(self):
        """Load and preprocess the dataset."""
        data_dir = self.repo_path / self.config['processing']['dataset']
        preprocess = Compose([ToTensor()])
        dataset = MRIDataset(data_dir, transform=preprocess, latents=True, prompt=True)
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.config['processing']['batch_size'],
            num_workers=self.config['processing']['num_workers'],
            shuffle=True
        )
        logger.info(f"Dataset loaded with {len(dataset)} images and {len(train_dataloader)} batches.")
        return dataset, train_dataloader


    def _setup_model(self):
        """Load the diffusion model and embeddings, and add new tokens for conditioning."""
        if self.config['logging']['log_reconstructions']:
            vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
            vae.eval().to(self.accelerator.device)

        # Load full pipeline
        ldm = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        ldm.to(self.accelerator.device)

        # Add new special tokens
        new_tokens = ["SHIFTS", "VH", "WMH2017"]
        num_added_tokens = ldm.tokenizer.add_tokens(new_tokens)
        ldm.text_encoder.resize_token_embeddings(len(ldm.tokenizer))

        # Freeze entire text encoder except for the new token embeddings
        for name, param in ldm.text_encoder.named_parameters():
            param.requires_grad = False

        embedding_layer = ldm.text_encoder.get_input_embeddings()
        for token in new_tokens:
            token_id = ldm.tokenizer.convert_tokens_to_ids(token)
            embedding_layer.weight[token_id].requires_grad = True
            print(f"Token '{token}' -> ID {token_id}, requires_grad: {embedding_layer.weight[token_id].requires_grad}")

        model = ldm.unet.to(self.accelerator.device)
        return model, vae, ldm


    def _get_embeddings(self, prompt, ldm):
        tokenizer = ldm.tokenizer
        text_encoder = ldm.text_encoder
        text_inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")

        # Mou els tensors a la GPU
        text_inputs = {k: v.to(self.accelerator.device) for k, v in text_inputs.items()}

        return text_encoder(**text_inputs).last_hidden_state


    def _setup_training(self):
        """Set up the optimizer, learning rate scheduler, and noise scheduler."""
        # Get the embedding layer with updated weights
        embedding_layer = self.ldm.text_encoder.get_input_embeddings()

        # Optimizer: include U-Net and the new token embeddings
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + [embedding_layer.weight],
            lr=self.config['training']['optimizer']['learning_rate'],
            betas=(self.config['training']['optimizer']['beta_1'], self.config['training']['optimizer']['beta_2']),
            weight_decay=self.config['training']['optimizer']['weight_decay'],
            eps=self.config['training']['optimizer']['eps']
        )

        num_epochs = self.config['training']['num_epochs']
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config['training']['gradient_accumulation']['steps'])
        max_train_steps = num_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name=self.config['training']['lr_scheduler']['name'],
            optimizer=optimizer,
            num_warmup_steps=self.config['training']['lr_scheduler']['num_warmup_steps'],
            num_training_steps=max_train_steps,
        )

        scheduler_type = self.config['training']['noise_scheduler']['type']
        if scheduler_type == 'DDPM':
            noise_scheduler = DDPMScheduler(
                beta_start=self.config['training']['noise_scheduler']['beta_start'],
                beta_end=self.config['training']['noise_scheduler']['beta_end'],
                num_train_timesteps=self.config['training']['noise_scheduler']['num_train_timesteps'],
                beta_schedule=self.config['training']['noise_scheduler']['beta_schedule'],
            )
        elif scheduler_type == 'DDIM':
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.config['training']['noise_scheduler']['num_train_timesteps'],
                beta_schedule=self.config['training']['noise_scheduler']['beta_schedule'],
            )
            noise_scheduler.set_timesteps(self.config['training']['noise_scheduler']['num_inference_timesteps'])
        else:
            raise ValueError("Noise scheduler type not recognized. Please choose between 'DDPM' and 'DDIM'.")

        return optimizer, lr_scheduler, noise_scheduler, num_update_steps_per_epoch


    def _init_tracker(self):
        """Initialize the tracking for the experiment."""
        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0] # get the name of the script
            self.accelerator.init_trackers(project_name=run) # intialize a run for all trackers
            wandb.save(str(config_path)) if self.config['logging']['logger_name'] == 'wandb' else None # save the self.config file in the wandb run

        total_batch_size = self.config['processing']['batch_size'] * self.accelerator.num_processes * self.config['training']['gradient_accumulation']['steps'] # considering accumulated and distributed training
        max_train_steps = self.config['training']['num_epochs'] * self.num_update_steps_per_epoch

        logger.info('The training is starting...\n')
        logger.info(f'The number of examples is: {len(self.dataset)}\n')
        logger.info(f'The number of epochs is: {self.config['training']['num_epochs']}\n')
        logger.info(f'The number of batches is: {len(self.train_dataloader)}\n')
        logger.info(f'The batch size is: {self.config["processing"]["batch_size"]}\n')
        logger.info(f'The number of update steps per epoch is: {self.num_update_steps_per_epoch}\n')
        logger.info(f'The gradient accumulation steps is: {self.config["training"]["gradient_accumulation"]["steps"]}\n')
        logger.info(f'The total batch size (accumulated, multiprocess) is: {total_batch_size}\n')
        logger.info(f'Total optimization steps: {max_train_steps}\n')
        logger.info(f'Using device: {self.accelerator.device} with {self.accelerator.num_processes} processes. {self.config["training"]["mixed_precision"]["type"]} mixed precision.\n')
        logger.info(f'The image resolution is: {self.config["processing"]["resolution"]}\n')
        logger.info(f'The model has {self.model.num_parameters()} parameters.\n')
        logger.info(f'The learning rate scheduler is: {self.config["training"]["lr_scheduler"]["name"]}\n')
        logger.info(f'The number of warmup steps is: {self.config["training"]["lr_scheduler"]["num_warmup_steps"]}\n')
        # logger.info(f'The prompt is: {self.config["prompt"]}\n')

    def _save_samples(self):
        """Save visual samples using multiple specific prompts."""
        prompts = [
            "SHIFTS FLAIR MRI",
            "VH FLAIR MRI",
            "WMH2017 FLAIR MRI",
        ]
        log_bs = self.config['logging']['images']['batch_size']

        for prompt in prompts:
            text_embeddings = self._get_embeddings(prompt, self.ldm).to(self.accelerator.device)

            latent_inf = torch.randn(
                log_bs, 4,
                self.config['processing']['resolution'], self.config['processing']['resolution'],
                generator=torch.manual_seed(17844)
            ).to(self.accelerator.device) * self.noise_scheduler.init_noise_sigma

            for t in tqdm(self.noise_scheduler.timesteps, desc=f"Sampling for prompt: {prompt}"):
                latent_inf = self.noise_scheduler.scale_model_input(latent_inf, t)
                with torch.no_grad():
                    noise_pred = self.model(latent_inf, t, encoder_hidden_states=text_embeddings.expand(log_bs, -1, -1)).sample
                latent_inf = self.noise_scheduler.step(noise_pred, t, latent_inf).prev_sample

            # Log latent channels
            if self.config['logging']['logger_name'] == 'wandb':
                # tag is the first part of the prompt
                tag = prompt.split()[0].lower()
                for i in range(4):
                    self.accelerator.get_tracker('wandb').log(
                        {f"latent_{i}_{tag}": [wandb.Image(latent_inf[b, i], mode='F') for b in range(log_bs)]},
                        step=self.global_step,
                    )

            # Decode and log reconstructions
            if self.config['logging']['log_reconstructions']:
                if self.config['logging']['images']['scaled']:
                    latent_inf /= self.vae.config.scaling_factor
                reconstructed = self.vae.decode(latent_inf).sample
                self.accelerator.get_tracker('wandb').log(
                    {f"reconstructed_{tag}": [wandb.Image(reconstructed[b][0], mode='F') for b in range(log_bs)]},
                    step=self.global_step,
                )

    def _save_samples_guidance(self):
        """Save samples with classifier-free guidance using multiple prompts."""
        prompts = [
            "SHIFTS FLAIR MRI",
            "VH FLAIR MRI",
            "WMH2017 FLAIR MRI",
        ]
        log_bs = self.config['logging']['images']['batch_size']
        guidance_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

        uncond_embeddings = self._get_embeddings("", self.ldm).to(self.accelerator.device)

        for prompt in prompts:
            text_embeddings = self._get_embeddings(prompt, self.ldm).to(self.accelerator.device)
            tag = prompt.split()[0].lower()

            latent_inf = torch.randn(
                log_bs, 4,
                self.config['processing']['resolution'], self.config['processing']['resolution'],
                generator=torch.manual_seed(17844)
            ).to(self.accelerator.device) * self.noise_scheduler.init_noise_sigma

            for guidance_scale in guidance_values:
                latent_guided = latent_inf.clone()

                for t in tqdm(self.noise_scheduler.timesteps, desc=f"Guided sampling for {prompt} | scale {guidance_scale}"):
                    latent_guided = self.noise_scheduler.scale_model_input(latent_guided, t)

                    with torch.no_grad():
                        noise_pred_uncond = self.model(latent_guided, t, encoder_hidden_states=uncond_embeddings.expand(log_bs, -1, -1)).sample
                        noise_pred_text = self.model(latent_guided, t, encoder_hidden_states=text_embeddings.expand(log_bs, -1, -1)).sample
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latent_guided = self.noise_scheduler.step(noise_pred, t, latent_guided).prev_sample

                if self.config['logging']['images']['scaled']:
                    latent_guided /= self.vae.config.scaling_factor
                reconstructed = self.vae.decode(latent_guided).sample.cpu()

                if self.config['logging']['logger_name'] == 'wandb':
                    self.accelerator.get_tracker('wandb').log(
                        {f"reconstructed_guidance_{tag}_scale_{guidance_scale}":
                            [wandb.Image(reconstructed[b][0], mode='F') for b in range(log_bs)]
                        },
                        step=self.global_step,
                    )

                del latent_guided, reconstructed, noise_pred, noise_pred_uncond, noise_pred_text
                torch.cuda.empty_cache()
                gc.collect()


    
    def _save_model(self):
        """Save the model to the pipeline directory."""
        # create pipeline # unwrap the model
        pipeline = DDPMPipeline(unet=self.accelerator.unwrap_model(self.model), scheduler=self.noise_scheduler)
        pipeline.save_pretrained(str(self.pipeline_dir))
        logger.info(f"Saving model to {self.pipeline_dir}")


    def _push_to_hub(self):
        """Push the fine-tuned Stable Diffusion model with all components to Hugging Face Hub."""
        from huggingface_hub import HfApi, create_repo, upload_folder, get_full_repo_name
        from diffusers import StableDiffusionPipeline

        config = self.config
        repo_id = get_full_repo_name(config['saving']['hf']['repo_name'])

        # Create repo if it doesn't exist
        create_repo(repo_id, exist_ok=True)

        # Save model components to a temp directory
        output_dir = self.pipeline_dir / "hf_pipeline"
        output_dir.mkdir(parents=True, exist_ok=True)


        print("Saving components...")
        
        # # Set scheduler config values to ensure future compatibility
        # self.noise_scheduler.config.steps_offset = 1
        # self.noise_scheduler.config.clip_sample = False

        # # Save the scheduler config before saving the pipeline
        # self.noise_scheduler.save_pretrained(output_dir)

        # Create pipeline and save config (links all components together)
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
            unet=self.accelerator.unwrap_model(self.model),
            text_encoder=self.accelerator.unwrap_model(self.ldm.text_encoder),
            vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae"),
            scheduler=self.noise_scheduler,
            tokenizer=self.ldm.tokenizer,
            safety_checker=None,
        )
        pipe.save_pretrained(output_dir)

        # Upload everything
        print(f"Uploading to Hub at {repo_id}...")
        upload_folder(
            repo_id=repo_id,
            folder_path=str(output_dir),
            commit_message="Final upload: all components and pipeline",
            ignore_patterns=["checkpoint-*"]
        )

        # Upload model card
        model_card_path = self.exp_path / config['saving']['hf']['model_card_name']
        if model_card_path.exists():
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=repo_id
            )
            logger.info("Model card uploaded as README.md")
        else:
            logger.warning(f"Model card not found at {model_card_path}")

        logger.info(f"Model successfully pushed to https://huggingface.co/{repo_id}")



    def train(self):
        """Training loop for fine-tuning the model."""
        self._init_tracker()
        self.model.enable_gradient_checkpointing()
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            train_loss = []
            pbar = tqdm(total=self.num_update_steps_per_epoch, desc=f"Epoch {epoch}")
            for latents, prompts in self.train_dataloader:
                torch.cuda.empty_cache()
                with self.accelerator.accumulate(self.model):
                    # Generate noise
                    noise = torch.randn_like(latents)
                    bs = latents.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()
                    

                    # **Randomly use empty prompt embeddings for a portion of the batch**
                    use_uncond = torch.rand(bs) < 0.1  # 10% of the batch uses unconditional embeddings
                    text_embeddings_batch = []
                    for i in range(bs):
                        if use_uncond[i]:
                            prompt = ""
                        else:
                            prompt = prompts[i]
                        embedding = self._get_embeddings(prompt, self.ldm).to(self.accelerator.device)
                        text_embeddings_batch.append(embedding)

                    text_embeddings_batch = torch.stack(text_embeddings_batch).squeeze(1)

                    # Forward pass
                    noisy_images = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=text_embeddings_batch.expand(bs, -1, -1)).sample
                    # noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=self.text_embeddings.expand(bs, -1, -1)).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                    
                    # Logging
                    avg_loss = self.accelerator.gather(loss.repeat(bs)).mean()
                    train_loss.append(avg_loss.item())
                    
                    # Backpropagation
                    self.accelerator.backward(loss, retain_graph=True)
                    if self.accelerator.sync_gradients: # gradient accumulation
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.config['training']['gradient_clip']['max_norm'])
                    
                    # Update parameters
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.accelerator.sync_gradients:
                    pbar.update(1)
                    self.global_step += 1
                    train_loss = np.mean(train_loss)
                    self.accelerator.log({"loss": train_loss, "log-loss": np.log(train_loss)}, step=self.global_step)

                    train_loss = [] # reset the loss for the next accumulation
                    
                    # Save the checkpoint
                    if self.global_step % self.config['saving']['local']['checkpoint_frequency'] == 0:
                        if self.accelerator.is_main_process:
                            save_path = self.pipeline_dir / f"checkpoint-{self.global_step}"
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saving checkpoint to {save_path}")
                
                # step logging
                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                self.accelerator.log(values=logs, step=self.global_step)
                pbar.set_postfix(**logs) # add to the end of the progress bar
            
            # Close the progress bar at the end of the epoch
            pbar.close()
            self.accelerator.wait_for_everyone()

            
            # Save the model and visual samples
            if self.accelerator.is_main_process:
                # if epoch  % self.config['logging']['images']['freq_epochs'] == 0 or epoch == self.config['training']['num_epochs'] - 1:
                #     if self.config['logging']['guidance']:
                #         self._save_samples_guidance()
                #     else:
                #         self._save_samples()
                    
                if epoch % self.config['saving']['local']['saving_frequency'] == 0 or epoch == self.config['training']['num_epochs'] - 1:
                    self._save_model()
            
        logger.info("Training complete.")

        self._push_to_hub()

        self.accelerator.end_training()



if __name__ == "__main__":
    config_path = "config_latent_finetuning.yaml"
    latent_finetuning = LatentFineTuning(config_path)
    latent_finetuning.train()
