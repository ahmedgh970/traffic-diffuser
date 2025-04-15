import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import os
import sys
import argparse
import logging
from glob import glob
from time import time
import importlib
import yaml
import numpy as np
from accelerate import Accelerator
from diffusion import create_diffusion
from torch.optim.lr_scheduler import LambdaLR


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
    module = importlib.import_module(f"{module_name}")
    model_class = getattr(module, class_name)
    return model_class

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

#################################################################################
#                         Custom Dataset and Dataloader                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_path, map_path, num_agents, hist_length, seq_length, dim_size):
        self.data_path = data_path
        self.map_path = map_path
        self.num_agents = num_agents
        self.hist_length = hist_length
        self.seq_length = seq_length
        self.dim_size = dim_size   
        self.filenames = sorted(os.listdir(data_path))
        self.mean = np.array([2273.33535086, 512.75471819]) # merged trainset
        self.std = np.array([4258.49158442, 4678.78819673]) # merged trainset
        self.scale = 100

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):      
        filename = self.filenames[idx]
        data_npy = np.load(os.path.join(self.data_path, filename))
        data_npy = data_npy[:self.num_agents, :, :self.dim_size]
        mask_data = np.all(data_npy == 0.0, axis=(2))
        data_npy = (data_npy - self.mean) / self.std  # Standardization
        data_npy *= self.scale # Rescaling
        data_npy[mask_data] = 0
        data_tensor = torch.tensor(data_npy, dtype=torch.float32)
        assert data_tensor.shape == (self.num_agents, self.hist_length + self.seq_length, self.dim_size), \
            f"Unexpected shape {data_tensor.shape} at index {idx}"
        map_npy = np.load(os.path.join(self.map_path, filename))
        map_npy = map_npy[:self.num_agents, :, :, :]
        mask_map = np.all(map_npy == 0.0, axis=(3))
        map_npy = (map_npy - self.mean) / self.std  # Standardization
        map_npy *= self.scale # Rescaling
        map_npy[mask_map] = 0
        map_tensor = torch.tensor(map_npy, dtype=torch.float32)
        assert map_tensor.shape == (self.num_agents, 16, 128, 2), \
            f"Unexpected shape {map_tensor.shape} at index {idx}"
        return data_tensor, map_tensor

class DynamicSubsetSampler(Sampler):
    """
    Custom Sampler for dynamic subset sampling each epoch.
    This sampler generates a new random set of indices from the dataset on every call
    to __iter__. We also add a set_epoch() method so that different epochs yield different
    subsets in a reproducible manner.
    """
    def __init__(self, data_source, subset_size, seed=0):
        self.data_source = data_source
        self.subset_size = subset_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = rng.choice(len(self.data_source), size=self.subset_size, replace=False)
        return iter(indices)

    def __len__(self):
        return self.subset_size

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(config):
    """
    Trains a diffusion model.
    """
    np.random.seed(config['train']['seed'])
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    
    # Initialize vars from config file:
    num_agents = config['model']['num_agents']
    seq_length = config['model']['seq_length']
    hist_length = config['model']['hist_length']
    dim_size = config['model']['dim_size']
    use_map_embed = config['model']['use_map_embed']   
    model_name = config['model']['name']
    experiments_dir = config['train']['experiments_dir']
    
    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(experiments_dir, exist_ok=True)
        experiment_index = len(glob(f"{experiments_dir}/*"))
        experiment_dir = f"{experiments_dir}/{experiment_index:03d}-{model_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = logging.getLogger(__name__)
    
    # Setup Dataloader:
    dataset = CustomDataset(
        data_path=config['data']['tracks_path'],
        map_path=config['data']['maps_path'],
        num_agents=num_agents,
        hist_length=hist_length,
        seq_length=seq_length,
        dim_size=dim_size,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} scenarios")
    # Generate a new random subset of indices each epoch.
    sampler = DynamicSubsetSampler(dataset, subset_size=config['data']['size'], seed=config['train']['seed'])
    loader = DataLoader(
        dataset,
        batch_size=int(config['train']['global_batch_size'] // accelerator.num_processes),
        sampler=sampler,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    # Create model:
    # Note that parameter initialization is done within the model constructor
    model_class = load_model(config['model']['module'], config['model']['class'])
    model = model_class[model_name](
        num_agents=num_agents,
        seq_length=seq_length,
        hist_length=hist_length,
        dim_size=dim_size,
        use_map_embed=use_map_embed,
        use_ckpt_wrapper=config['model']['use_ckpt_wrapper'],
    ).to(device)
    
    if accelerator.is_main_process:
        logger.info(f"Model summary:\n{model}")
    
    # Create diffusion:
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule="linear",
        diffusion_steps=config['train']['diffusion_steps']
    )
    if accelerator.is_main_process:
        logger.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['train']['learning_rate']),
        weight_decay=float(config['train']['weight_decay']),
    )

    # Calculate total steps and create a warmup + linear decay scheduler:
    total_steps = config['train']['epochs'] * len(loader)
    warmup_steps = config['train']['warmup_steps']
    scheduler = LambdaLR(
        opt,
        lr_lambda=lambda step: min(
            (step + 1) / warmup_steps,
            max(0.0, (total_steps - step) / (total_steps - warmup_steps))
        )
    )
    
    # Resume training if specified:
    start_epoch = 0
    train_steps = 0
    if config['train'].get('resume', False):
        resume_ckpt_path = config['train'].get('resume_ckpt', None)
        if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
            ckpt = torch.load(resume_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            scheduler.load_state_dict(ckpt.get("scheduler", {}))
            start_epoch = ckpt.get("epoch", 0)
            train_steps = ckpt.get("global_step", 0)
            if accelerator.is_main_process:
                logger.info(f"Resumed training from checkpoint {resume_ckpt_path} at epoch {start_epoch}, step {train_steps}.")
        else:
            if accelerator.is_main_process:
                logger.info("Resume flag set but no valid checkpoint found. Starting from scratch.")
    
    # Prepare models for training:
    model.train()
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Variables for monitoring/logging:
    log_steps = 0
    running_loss = 0
    start_time = time()
    num_epochs = config['train']['epochs']
    
    if accelerator.is_main_process:
        logger.info(f"Training for {num_epochs} epochs, starting at epoch {start_epoch}...")
    for epoch in range(start_epoch, num_epochs):
        # Update the sampler's epoch so that a new subset of data will be sampled.
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for data, mp in loader:
            x = data[:, :, hist_length:, :].to(device)
            h = data[:, :, :hist_length, :].to(device)
            mp = mp.to(device)
            model_kwargs = dict(h=h, mp=mp)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            scheduler.step()
            
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % config['train']['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save model checkpoint:
            if train_steps % config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": getattr(model, "module", model).state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "global_step": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    if accelerator.is_main_process:
        logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_train.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
