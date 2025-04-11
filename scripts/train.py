import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
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
#                                Dataset                                        #
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
    
    # Initialize var from config file
    num_agents = config['model']['num_agents']
    seq_length = config['model']['seq_length']
    hist_length = config['model']['hist_length']
    dim_size = config['model']['dim_size']
    use_map_embed=config['model']['use_map_embed']   
    model_name = config['model']['name']
    results_dir = config['train']['results_dir']
    
    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(results_dir, exist_ok=True)
        experiment_index = len(glob(f"{results_dir}/*"))
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    
    # Setup Dataloader
    dataset = CustomDataset(
        data_path=config['data']['tracks_path'],
        map_path=config['data']['maps_path'],
        num_agents=num_agents,
        hist_length=hist_length,
        seq_length=seq_length,
        dim_size=dim_size,
    )
    sampler = RandomSampler(dataset, replacement=True, num_samples=config['data']['size'])
    loader = DataLoader(
        dataset,
        batch_size=int(config['train']['global_batch_size'] // accelerator.num_processes),
        sampler=sampler,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} scenarios")
    
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
                        "config": config
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