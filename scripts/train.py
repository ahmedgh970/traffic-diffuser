import os
import argparse
import logging
from glob import glob
from time import time
import importlib
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import numpy as np
from accelerate import Accelerator

from diffusion import create_diffusion


#################################################################################
#                             Helper Functions                                  #
#################################################################################
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    
    return logger

def load_model(module_name, class_name):
    """
    Dynamically load the model class from the specified module.
    """
    module = importlib.import_module(f"models.backbones.{module_name}")
    model_class = getattr(module, class_name)
    return model_class

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

#################################################################################
#                                Dataset                                        #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_path, max_agent, hist_length, seq_length, dim_size):
        self.data_path = data_path
        self.max_agent = max_agent
        self.hist_length = hist_length
        self.seq_length = seq_length
        self.dim_size = dim_size   
        self.data_files = sorted(os.listdir(data_path))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):      
        data_file = self.data_files[idx]
        data_npy = np.load(os.path.join(self.data_path, data_file))
        data_npy = data_npy[:self.max_agent, :, :self.dim_size]
        data_tensor = torch.tensor(data_npy, dtype=torch.float32)
        assert data_tensor.shape == (self.max_agent, self.hist_length + self.seq_length, self.dim_size), \
            f"Unexpected shape {data_tensor.shape} at index {idx}"
        return data_tensor



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a diffusion model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(config['train']['results_dir'], exist_ok=True)
        experiment_index = len(glob(f"{config['train']['results_dir']}/*"))
        experiment_dir = f"{config['train']['results_dir']}/{experiment_index:03d}-{config['model']['name']}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    
    # Setup Dataloader
    dataset = CustomDataset(
        data_path=config['train']['train_dir'],
        max_agent=config['model']['max_num_agents'],
        hist_length=config['model']['hist_length'],
        seq_length=config['model']['seq_length'],
        dim_size=config['model']['dim_size'],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config['train']['global_batch_size'] // accelerator.num_processes),
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} scenarios")
    
    # Create model:
    # Note that parameter initialization is done within the model constructor
    model_class = load_model(config['model']['module'], config['model']['class'])
    model = model_class[config['model']['name']](
        max_num_agents=config['model']['max_num_agents'],
        seq_length=config['model']['seq_length'],
        hist_length=config['model']['hist_length'],
        dim_size=config['model']['dim_size'],
        map_channels=config['model']['map_channels'],
        use_map_embed=config['model']['use_map_embed'],
        use_ckpt_wrapper=config['model']['use_ckpt_wrapper'],
    ).to(device)
    
    # Model summary:
    if accelerator.is_main_process:
        logger.info(f"Model summary:\n{model}")
    
    # Create diffusion
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule="linear",
        diffusion_steps=config['train']['diffusion_steps']
    )
    if accelerator.is_main_process:
        logger.info(f"{config['model']['name']} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=0)
    
    # Prepare models for training:
    model.train()
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    num_epochs = config['train']['epochs']
    
    if accelerator.is_main_process:
        logger.info(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for data in loader:
            x = data[:, :, config['model']['hist_length']:, :].to(device)
            h = data[:, :, :config['model']['hist_length'], :].to(device)
            model_kwargs = dict(h=h)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save model checkpoint:
            if train_steps % config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    if accelerator.is_main_process:
        logger.info("Done!")


# To launch TrafficDiffuser-S training with one or multiple GPUs on one node:
# accelerate launch scripts/train.py --config configs/config_train.yaml
# accelerate launch --num-processes=1 --gpu_ids 1 --main_process_port 29502 scripts/train.py --config configs/config_train.yaml

# Load configuration and run main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_train.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)