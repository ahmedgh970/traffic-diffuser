"""
This script is a modification of work originally created by Bill Peebles, Saining Xie, and Ikko Eltociear Ashimine

Original work licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

Original source: https://github.com/facebookresearch/DiT
License: https://creativecommons.org/licenses/by-nc/4.0/
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchvision.transforms as transforms

from diffusion import create_diffusion
from models.model_td import TrafficDiffuser_models



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = TrafficDiffuser_models[args.model](
        max_num_agents=args.max_agents,
        seq_length=args.seq_length,
        hist_length=args.hist_length,
        dim_size=args.dim_size,
        use_map=args.use_map,
        use_history=args.use_history,
    ).to(device)
    
    # Load a TrafficDiffuser checkpoint:
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    if "ema" in state_dict: # supports checkpoints from train.py
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # Create sampling noise
    z = torch.randn(args.num_sampling, args.max_agents, args.seq_length, args.dim_size, device=device)
    print('shape of z:', z.shape)
    
    # Load scenario history
    data = np.load(args.data_path)
    data = torch.tensor(data, dtype=torch.float32)
    data = data.unsqueeze(0).expand(args.num_sampling, data.size(0), data.size(1), data.size(2))
    print('shape of data:', data.shape)
    
    pad_n = args.max_agents - data.size(1)
    pad_size = (0, 0, 0, 0, 0, pad_n, 0, 0)  # padding only in the agent dimension
    padded_data = nn.functional.pad(data, pad_size, "constant", 0.0)
    h = padded_data[:, :, :args.hist_length, :].to(device)
    print('shape of h:', h.shape)
    
    # Create a mask for valid agents
    mask = torch.ones((data.size(0), data.size(1), data.size(2), data.size(3)), dtype=torch.float32)
    mask = nn.functional.pad(mask, pad_size, "constant", 0.0)
    mask_x = mask[:, :, args.hist_length:, :].to(device)
    mask_h = mask[:, :, :args.hist_length, :].to(device)
    print('shape of mask_x:', mask_x.shape)
    print('shape of mask_h:', mask_h.shape)
    
    # Load scenario map
    map_rgb = Image.open(args.map_path).convert('RGB')
    mp = transforms.ToTensor()(map_rgb).to(device)
    mp = mp.unsqueeze(0).expand(args.num_sampling, mp.size(0), mp.size(1), mp.size(2))
    print('shape of map:', mp.shape)

    model_kwargs = dict(h=h, mp=mp, mask_x=mask_x, mask_h=mask_h)

    # Sample trajectories:
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples = samples[:, :data.size(1), :, :]
    print('shape of pred samples', samples.shape)
    h = h[:, :data.size(1), :, :]
    print('shape of hist samples', h.shape)
    samples = torch.cat((h, samples), dim=2)
    print('shape of full samples', samples.shape)
    
    # Save sampled trajectories
    samples_path = args.ckpt.replace("checkpoints", "samples")
    samples_path = os.path.splitext(samples_path)[0] + ".npy"
    samples_dir = os.path.dirname(samples_path)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    np.save(samples_path, samples.cpu().numpy())


# To sample from the EMA weights of a custom TrafficDiffuser-L model, run:
# python sample_wp.py --model TrafficDiffuser-S --use-history --use-map --ckpt results/000-TrafficDiffuser-S/checkpoints/0000052.pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(TrafficDiffuser_models.keys()), default="TrafficDiffuser-L")
    parser.add_argument("--data-path", type=str, default="/data/tii/data/nuscenes_trainval_npy/sd_nuscenes_v1.0-trainval_scene-0001.npy")
    parser.add_argument("--map-path", type=str, default="/data/tii/data/nuscenes_maps/nuscenes_trainval_maps_png1/sd_nuscenes_v1.0-trainval_scene-0001.png")
    parser.add_argument("--max-agents", type=int, default=234)
    parser.add_argument("--seq-length", type=int, default=56)
    parser.add_argument("--hist-length", type=int, default=100)
    parser.add_argument("--dim-size", type=int, default=8)
    parser.add_argument("--use-map", action='store_true', help='using map conditioning')
    parser.add_argument("--use-history", action='store_true', help='using agent history conditioning')
    parser.add_argument("--num-sampling", type=int, default=4)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
