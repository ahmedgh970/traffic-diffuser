import os
import argparse
import logging
from copy import deepcopy
from glob import glob
from time import time

import torch
import torch
import torch.nn as nn
import torch.optim as optim
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from collections import OrderedDict
from accelerate import Accelerator
from PIL import Image

from models.model_td import TrafficDiffuser_models
from diffusion import create_diffusion



#################################################################################
#                             Training Helper Functions                         #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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


#################################################################################
#                                Dataloader                                     #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_path, map_path):
        self.data_path = data_path
        self.map_path = map_path   
        self.data_files = sorted(os.listdir(data_path))
        self.map_files = sorted(os.listdir(map_path))
        self.transform = transforms.Compose([
            #transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Using ImageNet statistics
        ])

    def __len__(self):
        assert len(self.data_files) == len(self.map_files), \
            "Number of data files and map files should be same"
        return len(self.data_files)

    def __getitem__(self, idx):      
        data_file = self.data_files[idx]
        map_file = self.map_files[idx]

        data_npy = np.load(os.path.join(self.data_path, data_file))
        map_rgb = Image.open(os.path.join(self.map_path, map_file)).convert('RGB')
        
        return torch.tensor(data_npy, dtype=torch.float32), self.transform(map_rgb)
    
    
# Adapt the max_agents from dataset to other
def collate_fn(batch, max_agents=234, hist_length=100):
    padded_batch = []
    map_images = []
    masks = []
    
    for data, map_image in batch:
        pad_n = max_agents - data.size(0)
        pad_size = (0, 0, 0, 0, 0, pad_n)  # padding only the first dimension
        padded_data = nn.functional.pad(data, pad_size, "constant", 0.0)
        padded_batch.append(padded_data)

        # Create a mask for valid agents
        mask = torch.ones((data.size(0), data.size(1), data.size(2)), dtype=torch.float32)
        mask = nn.functional.pad(mask, pad_size, "constant", 0.0)
        masks.append(mask)

        map_images.append(map_image)
    
    padded_x, padded_hist = torch.stack(padded_batch)[:, :, hist_length:, :], torch.stack(padded_batch)[:, :, :hist_length, :]
    masks = torch.stack(masks)
    map_images = torch.stack(map_images)

    return padded_x, padded_hist, map_images, masks
    
    
    
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
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.model}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    model = TrafficDiffuser_models[args.model](
        max_num_agents=args.max_agents,
        seq_length=args.seq_length,
        hist_length=args.hist_length,
        dim_size=args.dim_size,
        use_map=args.use_map,
        use_history=args.use_history,
    )
    
    # Note that parameter initialization is done within the model constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"{args.model} Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    # Dataloader
    # The dataloader will return a batch of shape [data, mask, map]
    # of shape [(batch_size, max_agents, seq_length, dim), (batch_size, max_agents, seq_length, dim), (batch_size, H, W, C)]
    dataset = CustomDataset(data_path=args.data_path, map_path=args.map_path)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} scenarios ({args.data_path})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, hist, mp, mask in loader:
            x = x.to(device)
            hist = hist.to(device)
            mp = mp.to(device)
            mask = mask.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(hist=hist, mp=mp, mask=mask)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
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
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    
    if accelerator.is_main_process:
        logger.info("Done!")


# To launch TrafficDiffuser-S training with multiple GPUs on one node:
# accelerate launch --multi_gpu train.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data/tii/data/nuscenes_trainval_npy")
    parser.add_argument("--map-path", type=str, default="/data/tii/data/nuscenes_maps/nuscenes_trainval_maps_png1")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(TrafficDiffuser_models.keys()), default="TrafficDiffuser-S")
    parser.add_argument("--max-agents", type=int, default=234)
    parser.add_argument("--seq-length", type=int, default=56)
    parser.add_argument("--hist-length", type=int, default=100)
    parser.add_argument("--dim-size", type=int, default=8)
    parser.add_argument("--use-map", type=bool, default=True)
    parser.add_argument("--use-history", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=13)
    parser.add_argument("--ckpt-every", type=int, default=1300)
    args = parser.parse_args()
    main(args)