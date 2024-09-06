import os
import argparse
import numpy as np
from statistics import mean
import logging
from datetime import datetime

import torch

from diffusion import create_diffusion
from models.model_td import TrafficDiffuser_models
from metrics import *

#torch.manual_seed(1234)



# Function to create a unique log file name
def create_log_file(log_dir='logs'):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{log_dir}/evaluation_{timestamp}.log"
    return log_filename


def evaluate_trajectory(valid_agent_gen, valid_agent_future, num_timesteps=10, kind='linear'):
    """
    Evaluate a single agent's generated trajectory against the future ground truth.
    """
    # Extrapolate trajs to the desired num_timesteps
    valid_agent_gen = interpolate_to_fixed_length(valid_agent_gen, num_timesteps, kind)
    valid_agent_future = interpolate_to_fixed_length(valid_agent_future, num_timesteps, kind)
    
    # Calculate metrics
    area = calculate_polygone_area(valid_agent_future, valid_agent_gen)
    diff_dist_travelled = calculate_diff_distance_traveled(valid_agent_future, valid_agent_gen)
    frechet_dist = calculate_frechet_distance(valid_agent_future, valid_agent_gen)
    ade = calculate_ade(valid_agent_future, valid_agent_gen)
    fde = calculate_fde(valid_agent_future, valid_agent_gen)
    
    return area, diff_dist_travelled, frechet_dist, ade, fde
    

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = TrafficDiffuser_models[args.model](
        max_num_agents = args.max_num_agents,
        seq_length=args.seq_length,
        hist_length=args.hist_length,
        dim_size=args.dim_size,
        map_size=args.map_size,
        use_gmlp=args.use_gmlp,
        use_map_embed=args.use_history_embed,
        use_ckpt_wrapper=args.use_ckpt_wrapper,
    ).to(device)
    
    # Load a TrafficDiffuser checkpoint:
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"])
    model.eval()  # important!
    
    # Create diffusion with the desired number of sampling steps 
    diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))
    
    # Set up logging to file
    log_filename = create_log_file(log_dir=os.path.dirname(os.path.dirname(args.ckpt)))
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    # Sample trajectories from testset:
    average_metrics = []
    for scenario in sorted(os.listdir(args.test_dir)):
        data = np.load(os.path.join(args.test_dir, scenario))
        data = torch.tensor(data[:args.max_num_agents, :, :], dtype=torch.float32).to(device)
        data = data.unsqueeze(0).expand(args.num_sampling, data.size(0), data.size(1), data.size(2))
        h = data[:, :, :args.hist_length, :]
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
        
        # Evaluation loop
        data_future = data[:, :, args.hist_length-1:, :].cpu().numpy()   # (N, L_seq_length + 1, D)
        epsilon, num_timesteps, kind = 0.1, 10, 'linear'

        # Initialize storage for all agents
        area_agent, diff_dist_travelled_agent, frechet_dist_agent, ade_agent, fde_agent = [], [], [], [], []
        
        for ag in range(samples.shape[1]):
            metrics = []
            for sample_idx in range(samples.shape[0]):
                
                agent_future = data_future[sample_idx, ag, :, :]
                agent_gen = samples[sample_idx, ag, :, :]
                agent_gen = np.concatenate((agent_future[0:1, :], agent_gen), axis=0)
                
                # Filter out points that are too close to (0,0)
                valid_agent_gen = agent_gen[(np.abs(agent_gen[:, 0]) > epsilon) & (np.abs(agent_gen[:, 1]) > epsilon)]
                valid_agent_future = agent_future[(agent_future[:, 0] != 0.0) & (agent_future[:, 1] != 0.0)]
                
                if valid_agent_gen.shape[0] != 0 or valid_agent_future.shape[0] != 0:
                    # Evaluate each agent's trajectory
                    metrics.append(evaluate_trajectory(valid_agent_gen, valid_agent_future, num_timesteps, kind))
            
            if metrics != []:
                # Unpack metrics
                area, diff_dist_travelled, frechet_dist, ade, fde = zip(*metrics)
            
                # Store the averaged metrics
                area_agent.append(mean(area))
                diff_dist_travelled_agent.append(mean(diff_dist_travelled))
                frechet_dist_agent.append(mean(frechet_dist))
                ade_agent.append(mean(ade))
                fde_agent.append(mean(fde))
        
        # Logging the results
        logging.info(f"This evaluation is conducted for scenario {scenario} and involves averaging across all samples and agents:")
        logging.info(f" - Polygone Area (PA): {mean(area_agent)}")
        logging.info(f" - Absolute Traveled Distance Difference (ATDD): {mean(diff_dist_travelled_agent)}")
        logging.info(f" - Frechet Distance (FD): {mean(frechet_dist_agent)}")
        logging.info(f" - Average Displacement Error (ADE): {mean(ade_agent)}")
        logging.info(f" - Final Displacement Error (FDE): {mean(fde_agent)} \n")

        print(f"Logs have been saved to {log_filename}")
        
        average_metrics.append((mean(area_agent), mean(diff_dist_travelled_agent), mean(frechet_dist_agent), mean(ade_agent), mean(fde_agent)))
    
    dataset_area, dataset_diff_dist_travelled, dataset_frechet_dist, dataset_ade, dataset_fde = zip(*average_metrics)
    
    # Logging the results
    logging.info(f"The average evaluation results across all scenarios:")
    logging.info(f" - Polygone Area (PA): {mean(dataset_area)}")
    logging.info(f" - Absolute Traveled Distance Difference (ATDD): {mean(dataset_diff_dist_travelled)}")
    logging.info(f" - Frechet Distance (FD): {mean(dataset_frechet_dist)}")
    logging.info(f" - Average Displacement Error (ADE): {mean(dataset_ade)}")
    logging.info(f" - Final Displacement Error (FDE): {mean(dataset_fde)} \n")


# To sample from the EMA weights of a custom TrafficDiffuser-L model, run:
# python sample.py --use-map-embed --use-ckpt-wrapper --ckpt /data/ahmed.ghorbel/workdir/autod/TrafficDiffuser/results/006-TrafficDiffuser-B/checkpoints/0084000.pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="/data/tii/data/nuscenes_trainval_clean_test")  # 149 scenarios
    parser.add_argument("--model", type=str, choices=list(TrafficDiffuser_models.keys()), default="TrafficDiffuser-B")
    parser.add_argument("--max-num-agents", type=int, default=46)
    parser.add_argument("--seq-length", type=int, default=5)
    parser.add_argument("--hist-length", type=int, default=8)
    parser.add_argument("--dim-size", type=int, default=2)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--use-gmlp", action='store_true', help='using gated mlp in place of mlp')
    parser.add_argument("--use-map-embed", action='store_true', help='using history embedding conditioning')
    parser.add_argument("--use-ckpt-wrapper", action='store_true', help='using checkpoint wrapper for memory saving during training')
    parser.add_argument("--num-sampling", type=int, default=10)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
