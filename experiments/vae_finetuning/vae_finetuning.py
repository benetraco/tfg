#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
    print(repo_path)
    
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve() # path to the experiment folder
print(f"Repo Path: {repo_path}")
print(f"Experiment Path: {exp_path}")

#Libraries
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
import wandb
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import logging
from accelerate.logging import get_logger
from accelerate import Accelerator

# import the MRIDataset class from the dataset folder
from dataset.build_dataset import MRIDataset


# Check the diffusers version
check_min_version("0.15.0.dev0")

# set the logger
logger = get_logger(__name__, log_level="INFO") # allow from info level and above

######MAIN######
def main():

    ### 0. General setups
    # load the config file
    config_path = exp_path / 'config_vae_finetunning.yaml'
    with open(config_path) as file: # expects the config file to be in the same directory
        config = yaml.load(file, Loader=yaml.FullLoader)

    # define logging directory
    pipeline_dir = repo_path / config['saving']['local']['outputs_dir'] / config['saving']['local']['pipeline_name']

    # start the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation']['steps'],
        mixed_precision=config['training']['mixed_precision']['type'],
        log_with=config['logging']['logger_name'],  # Keep this if you're using `log_with`
    )

    # define basic logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", # format of the log message. # name is the logger name.
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )



    ### 1. Dataset loading and preprocessing
    # Dataset loading
    data_dir = repo_path / config['processing']['dataset']
    # Define the transformations to apply to the images
    preprocess = Compose(
        [
            Resize(config['processing']['resolution'], interpolation= InterpolationMode.BILINEAR), #getattr(InterpolationMode, config['processing']['interpolation'])),  # Smaller edge is resized to 256 preserving aspect ratio
            CenterCrop(config['processing']['resolution']),  # Center crop to the desired squared resolution
            #RandomHorizontalFlip(),  # Horizontal flip may not be a good idea if we want generation only one laterality
            ToTensor(),  # Convert to PyTorch tensor
            Normalize(mean=[0.5], std=[0.5]),  # Map to (-1, 1) as a way to make data more similar to a Gaussian distribution
        ]
    )

    # Create dataset with the defined transformations
    dataset = MRIDataset(data_dir, transform=preprocess, latents=True) # create the dataset
    logger.info(f"Dataset loaded with {len(dataset)} images") # show info about the dataset
    # Create the dataloader
    train_dataloader = DataLoader(
        dataset, batch_size=config['processing']['batch_size'], num_workers= config['processing']['num_workers'], shuffle=True
    )


    ### 2. Model loading 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(accelerator.device)

    ### 3. Training
    ## 3.1 Setup the training
    num_epochs = config['training']['num_epochs']
    
    # global trackers
    total_batch_size = config['processing']['batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation']['steps'] # considering accumulated and distributed training
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['training']['gradient_accumulation']['steps']) # take into account the gradient accumulation (divide)
    max_train_steps = num_epochs * num_update_steps_per_epoch # total number of training steps
    
    
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr= config['training']['optimizer']['learning_rate'], # learning rate of the optimizer
        betas= (config['training']['optimizer']['beta_1'], config['training']['optimizer']['beta_2']), # betas according to the AdamW paper
        weight_decay= config['training']['optimizer']['weight_decay'], # weight decay according to the AdamW paper
        eps= config['training']['optimizer']['eps'] # epsilon according to the AdamW paper
    )
    lr_scheduler = get_scheduler(
        name= config['training']['lr_scheduler']['name'], # name of the scheduler
        optimizer= optimizer, # optimizer to use
        num_warmup_steps= config['training']['lr_scheduler']['num_warmup_steps'], #* config['training']['gradient_accumulation']['steps'],
        num_training_steps= max_train_steps, 
    )

    # prepare with the accelerator
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    # init wandb tracker and save config file
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0] # get the name of the script
        accelerator.init_trackers(project_name=run) # intialize a run for all trackers
        wandb.save(str(config_path)) if config['logging']['logger_name'] == 'wandb' else None # save the config file in the wandb run

    logger.info('The training is starting...\n')
    logger.info(f'The number of examples is: {len(dataset)}\n')
    logger.info(f'The number of epochs is: {num_epochs}\n')
    logger.info(f'The number of batches is: {len(train_dataloader)}\n')
    logger.info(f'The batch size is: {config["processing"]["batch_size"]}\n')
    logger.info(f'The number of update steps per epoch is: {num_update_steps_per_epoch}\n')
    logger.info(f'The gradient accumulation steps is: {config["training"]["gradient_accumulation"]["steps"]}\n')
    logger.info(f'The total batch size (accumulated, multiprocess) is: {total_batch_size}\n')
    logger.info(f'Total optimization steps: {max_train_steps}\n')
    logger.info(f'Using device: {accelerator.device} with {accelerator.num_processes} processes. {config["training"]["mixed_precision"]["type"]} mixed precision.\n')
    logger.info(f'The image resolution is: {config["processing"]["resolution"]}\n')
    logger.info(f'The vae has {vae.num_parameters()} parameters.\n')
    logger.info(f'The learning rate scheduler is: {config["training"]["lr_scheduler"]["name"]}\n')
    logger.info(f'The number of warmup steps is: {config["training"]["lr_scheduler"]["num_warmup_steps"]}\n')

    ## Log images before training to check the initial performance
    # global variables (mainly useful for checkpointing)
    global_step = 0

    # Choose num_images random images from the dataset
    num_images = config['logging']['images']['batch_size']
    np.random.seed(17844)
    indices = np.random.choice(len(dataset), num_images, replace=False)
    images = [dataset[i] for i in indices]
    images = torch.stack(images).to(accelerator.device)
    
    # Encode and decode the images using the pretrained VAE
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        reconstructions = vae.decode(latents).sample
    
    # Log the original and reconstruction images (before training)
    if config['logging']['logger_name'] == 'wandb':
        wandb.log(
            {
                "original": [wandb.Image(img) for img in images.cpu()],
                "reconstructions": [wandb.Image(img) for img in reconstructions.cpu()],
                "latents": [wandb.Image(img) for img in latents.cpu()],
            },
            step=global_step,
        )
    logger.info(f"Initial images saved and logged at global_step {global_step}")

    ## Prepare the model for training
    vae = accelerator.prepare(vae) # prepare the model with the accelerator
    vae = accelerator.unwrap_model(vae) # unwrap the model to allow training
    vae.train()

    #### 3.2 Training loop
    for epoch in range(num_epochs): # Loop over the epochs
        vae.train()
        train_loss = [] # accumulated loss list
        pbar = tqdm(total=num_update_steps_per_epoch)
        pbar.set_description(f"Epoch {epoch}")
        for batch in train_dataloader: # Loop over the batches
            with accelerator.accumulate(vae):
                # Encode the batch
                latents = vae.encode(batch).latent_dist.sample()
                # Decode the batch
                reconstructions = vae.decode(latents).sample
                # Calculate the loss
                loss = F.mse_loss(reconstructions, batch)
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config['processing']['batch_size'])).mean()
                # append the loss to the train loss
                train_loss.append(avg_loss.item())
                
                # Backpropagate the loss
                accelerator.backward(loss) #loss is used as a gradient, coming from the accumulation of the gradients of the loss function
                if accelerator.sync_gradients: # gradient clipping
                    accelerator.clip_grad_norm_(vae.parameters(), config['training']['gradient_clip']['max_norm'])
                # Update
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # updates and checkpoint saving happens only if the gradients are synced
            if accelerator.sync_gradients:
                # Update the progress bar
                pbar.update(1)
                global_step += 1
                # take the mean of the accumulated loss
                train_loss = np.mean(train_loss)
                accelerator.log({"loss": train_loss, "log-loss": np.log(train_loss)}, step=global_step) #accumulated loss
                train_loss = [] # reset the train for next accumulation
                # Save the checkpoint
                if global_step % config['saving']['local']['checkpoint_frequency'] == 0: # if saving time
                    if accelerator.is_main_process: # only if in main process
                        save_path = pipeline_dir / f"checkpoint-{global_step}" # create the path
                        accelerator.save_state(save_path) # save the state
                        logger.info(f"Saving checkpoint to {save_path}") # let the user know
            # step logging
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(values=logs, step=global_step)
            pbar.set_postfix(**logs)
        # Close the progress bar at the end of the epoch
        pbar.close()
        accelerator.wait_for_everyone() # wait for all processes to finish before saving the model



        ##### 4. Saving the model and visual samples
        # generate visual samples to track training performance and save when in saving epoch
        if accelerator.is_main_process:
            if epoch % config['logging']['images']['freq_epochs'] == 0 or epoch == num_epochs - 1: # if in saving epoch or last one
                # unwrape the model 
                vae = accelerator.unwrap_model(vae)
                num_images = config['logging']['images']['batch_size']
                # choose num_images random images
                np.random.seed(17844)
                indices = np.random.choice(len(dataset), num_images, replace=False)
                images = [dataset[i] for i in indices]
                images = torch.stack(images).to(accelerator.device)
                # encode the images
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    reconstructions = vae.decode(latents).sample
                # log the images (original, latents and reconstructions)
                if config['logging']['logger_name'] == 'wandb':
                    wandb.log(
                        {
                            "original": [wandb.Image(img) for img in images.cpu()],
                            "reconstructions": [wandb.Image(img) for img in reconstructions.cpu()],
                            "latents": [wandb.Image(img) for img in latents.cpu()],
                        },
                        step=global_step,
                    )
            # save model
            if epoch % config['saving']['local']['saving_frequency'] == 0 or epoch == num_epochs - 1:
                vae.save_pretrained(str(pipeline_dir))
                logger.info(f"Saving VAE to {pipeline_dir}")
    
    logger.info("Finished training!\n")
    # stop tracking
    accelerator.end_training()


############################################################################################################

if __name__ == "__main__":
    main()