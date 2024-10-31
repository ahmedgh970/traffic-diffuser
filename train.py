import os
import argparse
import logging
from glob import glob
from time import time
import importlib

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

def load_model(module_name):
    """
    Dynamically load the model class from the specified module.
    """
    module = importlib.import_module(f"models.{module_name}")
    model_class = getattr(module, 'TrafficDiffuser_models')
    return model_class


#################################################################################
#                                Dataloader                                     #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_path, max_agent, dim_size, map_path, map_size):
        self.data_path = data_path
        self.max_agent = max_agent
        self.dim_size = dim_size
        self.map_path = map_path
        self.map_size = map_size
        self.data_files = sorted(os.listdir(data_path))
        self.map_files = sorted(os.listdir(map_path))
        self.map_transform = T.Resize(self.map_size)

    def __len__(self):
        return len(self.data_files)

    def pad_or_crop_agents(self, data_npy):
        # Crop to the desired dim_size
        data_tensor = torch.tensor(data_npy[:, :, :self.dim_size], dtype=torch.float32)
        num_agents = data_tensor.shape[0]
        
        # Crop or pad to the desired max_agent
        if num_agents >= self.max_agent:
            return data_tensor[:self.max_agent]
        else:
            # If fewer agents, pad with zeros using torch.cat for efficiency
            padding = torch.zeros((self.max_agent - num_agents, data_tensor.shape[1], data_tensor.shape[2]), dtype=torch.float32)
            return torch.cat((data_tensor, padding), dim=0)

    def __getitem__(self, idx):      
        data_file = self.data_files[idx]
        data_npy = np.load(os.path.join(self.data_path, data_file))
        
        # crop to max num agents and dim size
        #data_tensor = torch.tensor(data_npy[:self.max_agent, :, :self.dim_size], dtype=torch.float32)
        data_tensor = self.pad_or_crop_agents(data_npy)
        
        map_file = self.map_files[idx]
        map_npy = np.load(os.path.join(self.map_path, map_file))
        map_tensor = torch.tensor(map_npy, dtype=torch.float32)
        map_tensor = self.map_transform(map_tensor)
        return data_tensor, map_tensor



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
    
    # Setup Dataloader
    # The dataloader will return a batch of tensor x of shape (B, L, D)
    dataset = CustomDataset(data_path=args.train_dir, max_agent=args.max_num_agents, dim_size=args.dim_size, map_path=args.map_train_dir, map_size=args.map_size)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} scenarios ({args.train_dir})")
        
    # Create model:
    model_class = load_model(args.model_module)
    model = model_class[args.model](
        max_num_agents = args.max_num_agents,
        seq_length=args.seq_length,
        hist_length=args.hist_length,
        dim_size=args.dim_size,
        map_channels=args.map_channels,
        use_gmlp=args.use_gmlp,
        use_map_embed=args.use_map_embed,
        use_ckpt_wrapper=args.use_ckpt_wrapper,
    ).to(device)
    
    # Model summary:
    if accelerator.is_main_process:
        logger.info(f"Model summary:\n{model}")
    
    # Note that parameter initialization is done within the model constructor
    diffusion = create_diffusion(timestep_respacing="", noise_schedule="linear", diffusion_steps=args.diffusion_steps)
    if accelerator.is_main_process:
        logger.info(f"{args.model} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Prepare models for training:
    model.train()
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
        for data, m in loader:
            x = data[:, :, args.hist_length:, :].to(device)
            h = data[:, :, :args.hist_length, :].to(device)
            if args.use_map_embed:
                m = m.to(device)
                model_kwargs = dict(h=h, m=m)
            else:
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
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    if accelerator.is_main_process:
        logger.info("Done!")


# To launch TrafficDiffuser-S training with one or multiple GPUs on one node:
# accelerate launch train.py --model TrafficDiffuser-S --max-num-agents 46 --hist-length 8 --seq-length 5 --use-map-embed --use-ckpt-wrapper
# accelerate launch --num-processes=1 --gpu_ids 1 --main_process_port 29502 train.py --model TrafficDiffuser-S --max-num-agents 46 --hist-length 8 --seq-length 5 --use-map-embed --use-ckpt-wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="/data/tii/data/nuscenes/nuscenes_trainval_clean_train/")
    parser.add_argument("--map-train-dir", type=str, default="/data/tii/data/nuscenes/nuscenes_maps/nuscenes_trainval_raster_train")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model-module", type=str, default="model_td")
    parser.add_argument("--model", type=str, default="TrafficDiffuser-S", help='choose from TrafficDiffuser-{S, B, L}')
    parser.add_argument("--max-num-agents", type=int, default=46) # 46 for full and 19 for veh
    parser.add_argument("--seq-length", type=int, default=5)
    parser.add_argument("--hist-length", type=int, default=8)
    parser.add_argument("--dim-size", type=int, default=2)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--map-channels", type=int, default=4)
    parser.add_argument("--use-gmlp", action='store_true', help='using gated mlp instead of mlp')
    parser.add_argument("--use-map-embed", action='store_true', help='using map embedding conditioning')
    parser.add_argument("--use-ckpt-wrapper", action='store_true', help='using checkpoint wrapper for memory saving during training')
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=21)  #-- 698 for full and 664 for veh
    parser.add_argument("--ckpt-every", type=int, default=21_000)
    args = parser.parse_args()
    main(args)