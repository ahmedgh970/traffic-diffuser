import os
import argparse
import math
import numpy as np
from statistics import mean
import logging
from datetime import datetime

import torch
import torch.distributed as dist

import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusion import create_diffusion
from models.model_td import TrafficDiffuser_models
from metrics import *



def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    model = TrafficDiffuser_models[args.model](
        max_num_agents = args.max_num_agents,
        seq_length=args.seq_length,
        hist_length=args.hist_length,
        dim_size=args.dim_size,
        map_channels=args.map_channels,
        use_gmlp=args.use_gmlp,
        use_map_embed=args.use_map_embed,
        use_ckpt_wrapper=args.use_ckpt_wrapper,
    ).to(device)

    # Load a TrafficDiffuser checkpoint:
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"])
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{model_string_name}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .npy samples at {sample_folder_dir}")
    dist.barrier()
    
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_sampling / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of trajectories that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    for _ in pbar:
        for scenario, mp in zip(sorted(os.listdir(args.test_dir)), sorted(os.listdir(args.map_test_dir))):
            data = np.load(os.path.join(args.test_dir, scenario))
            data = torch.tensor(data[:args.max_num_agents, :, :], dtype=torch.float32).to(device)
            data = data.unsqueeze(0).expand(args.num_sampling, data.size(0), data.size(1), data.size(2))
            h = data[:, :, :args.hist_length, :]        
            if args.use_map_embed:
                m = np.load(os.path.join(args.map_test_dir, mp))
                m = torch.tensor(m, dtype=torch.float32).to(device)
                m = m.unsqueeze(0).expand(args.num_sampling, m.size(0), m.size(1), m.size(2))
                model_kwargs = dict(h=h, m=m)
            else:
                model_kwargs = dict(h=h)
            
            # Create sampling noise:
            x = torch.randn(args.num_sampling, args.max_num_agents, args.seq_length, args.dim_size, device=device)
            
            # Sample trajectories:
            samples = diffusion.p_sample_loop(
                model.forward, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples = samples.cpu().numpy()
            
            # Save sampled trajectories
            samples_path = args.ckpt.replace("checkpoints", "samples")
            file_name = scenario.split('/')[-1].split('.npy')[0]      # from absolute path to data file name without extension
            samples_path = os.path.splitext(samples_path)[0] + "_" + file_name + ".npy"
            samples_dir = os.path.dirname(samples_path)
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
            np.save(samples_path, samples)
            print(f'Generated samples saved in {samples_path}')


#-- torchrun --nnodes=1 --nproc_per_node=8  sample.py --use-map-embed --use-ckpt-wrapper --ckpt /data/ahmed.ghorbel/workdir/autod/TrafficDiffuser/results/006-TrafficDiffuser-B/checkpoints/0084000.pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="/data/tii/data/nuscenes_trainval_clean_test")  # 149 scenarios and 19 ag
    parser.add_argument("--map-test-dir", type=str, default="/data/tii/data/nuscenes_maps/nuscenes_trainval_raster_test")
    parser.add_argument("--model", type=str, choices=list(TrafficDiffuser_models.keys()), default="TrafficDiffuser-B")
    parser.add_argument("--max-num-agents", type=int, default=46)
    parser.add_argument("--seq-length", type=int, default=5)
    parser.add_argument("--hist-length", type=int, default=8)
    parser.add_argument("--dim-size", type=int, default=2)
    parser.add_argument("--map-channels", type=int, default=4)
    parser.add_argument("--use-gmlp", action='store_true', help='using gated mlp in place of mlp')
    parser.add_argument("--use-map-embed", action='store_true', help='using history embedding conditioning')
    parser.add_argument("--use-ckpt-wrapper", action='store_true', help='using checkpoint wrapper for memory saving during training')
    parser.add_argument("--num-sampling", type=int, default=128)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
