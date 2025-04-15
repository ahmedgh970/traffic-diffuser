import torch
import torch.nn as nn
import os
import time
import yaml
import random
import argparse
import numpy as np
from statistics import mean
import logging
from datetime import datetime
import importlib
from fvcore.nn import FlopCountAnalysis
from diffusion import create_diffusion
from utils import ade, fde, interpolate 



#################################################################################
#                             Helper Functions                                  #
#################################################################################
def create_log_file(log_dir='logs'):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{log_dir}/evaluation_{timestamp}.log"
    return log_filename

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

def dataset_stats(dataset_name):
    scale_factor = 100
    dataset_stats = {
        'train': ([2273.33535086, 512.75471819], [4258.49158442, 4678.78819673]),
        'val_av2': ([2697.18031371, 1104.79493649], [3204.85198704, 1708.79929148]),
        'test_waymo': ([3226.6213927, -1227.95431973], [4762.12227595, 7409.94028107]),
        'val_waymo': ([1814.67110415,  176.52220831], [5094.52135561, 6194.25890569]),
    } 
    mn, std = dataset_stats[dataset_name]
    return np.array(mn), np.array(std), scale_factor
        
def evaluate_trajectory(gen_traj, gt_traj, num_timesteps=10, kind='linear'):
    """
    Evaluate a single generated trajectory against the future ground truth.
    """
    # Extrapolate trajs to the desired num_timesteps
    gen_traj = interpolate(gen_traj, num_timesteps, kind)
    gt_traj = interpolate(gt_traj, num_timesteps, kind)
    
    # Calculate metrics
    ADE = ade(gt_traj, gen_traj)
    FDE = fde(gt_traj, gen_traj)
    MR = 1 if FDE > 2 else 0
    return ADE, FDE, MR
    

#################################################################################
#                       Sampling and Evaluation Loop                            #
#################################################################################
def main(config):
    # Setup PyTorch:
    torch.manual_seed(config['sample']['seed'])
    random.seed(config['sample']['seed'])
    torch.set_grad_enabled(False)
    device = f"cuda:{config['sample']['cuda_device']}" if torch.cuda.is_available() else "cpu"
    
    # Initialize model args
    num_agents=config['model']['num_agents']
    seq_length=config['model']['seq_length']
    hist_length=config['model']['hist_length']
    dim_size=config['model']['dim_size']
    use_map_embed=config['model']['use_map_embed']
    
    # Initialize the model:
    # Note that parameter initialization is done within the model constructor
    model_class = load_model(config['model']['module'], config['model']['class'])
    model_name = config['model']['name']
    model = model_class[model_name](
        num_agents=num_agents,
        seq_length=seq_length,
        hist_length=hist_length,
        dim_size=dim_size,
        use_map_embed=use_map_embed,
        use_ckpt_wrapper=False,
    ).to(device)
    
    # Load a TrafficDiffuser checkpoint:
    state_dict = torch.load(config['model']['ckpt'], weights_only=True)
    model.load_state_dict(state_dict["model"])
    model.eval()  # important!
    print('===> Model initialized !')
    
    # Create diffusion with the desired number of sampling steps 
    diffusion = create_diffusion(timestep_respacing=str(config['sample']['num_sampling_steps']))
    
    # Set up logging to file
    log_filename = create_log_file(log_dir=os.path.dirname(os.path.dirname(config['model']['ckpt'])))
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    
    # Print model parameters, and summary
    logging.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    logging.info(f"{model_name} Model summary:\n{model}\n")
        
    # Choose a subset of scenarios from testset:
    dataset_name = config['data']['name']
    dataset_path = os.path.join(config['data']['path'], dataset_name)
    test_files = sorted(random.sample(sorted(os.listdir(dataset_path)), config['data']['size'][dataset_name]))

    # Make samples directory
    samples_dir = os.path.dirname(config['model']['ckpt']).replace("checkpoints", "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Sample and evaluate scenarios from test_files
    logging.info("The metrics are computed as the average over all agents within each scenario.\n")
    mean_xy, std_xy, scale_factor = dataset_stats(dataset_name)
    metrics_testset = []
    idx = 1
    num_sampling = config['sample']['num_sampling']
    for filename in test_files:
        # full data
        data = np.load(os.path.join(dataset_path, filename))
        data = data[:num_agents]   
        # history
        h = data[:, :hist_length, :]
        mask = np.all(h == 0.0, axis=(2))# Mask for padded agents and ts
        h = (h - mean_xy) / std_xy  # Standardization
        h *= scale_factor # Rescaling
        h[mask] = 0
        h = torch.tensor(h, dtype=torch.float32).to(device)
        h = h.unsqueeze(0).expand(num_sampling, h.size(0), h.size(1), h.size(2))     
        # map
        if use_map_embed:
            num_segments = 16
            mp = torch.zeros(num_sampling, num_agents, num_segments, 128, 2).to(device)
        else:
            mp = None
        # Create sampling noise:
        x = torch.randn(num_sampling, num_agents, seq_length, dim_size).to(device)
    
        # Sample trajectories:
        model_kwargs = dict(h=h, mp=mp)
        samples = diffusion.p_sample_loop(
            model.forward, x.shape, x,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device
        )
        samples = samples.cpu().numpy()

        # Prepare boolean masks for padded agents and near zero noise
        epsilon = 0.99
        mask0 = np.abs(samples) < epsilon
        mask1 = np.all(data == 0.0, axis=(1, 2))
        mask1 = np.expand_dims(mask1, axis=0) 
        mask1 = np.broadcast_to(mask1, (num_sampling, mask1.shape[1]))

        # Rescale and unstandardize
        samples /= scale_factor
        samples = (samples * std_xy) + mean_xy
        
        # Mask
        samples[mask0] = 0 
        samples[mask1] = 0
        
        #####################################
        # Sampled scenario evaluation loop
        #####################################
        num_timesteps, kind = 10, 'linear'
        ADE_scenario, FDE_scenario, MR_scenario  = [], [], []

        # Evaluate each agent's trajectories
        for ag in range(num_agents):
            metrics = []
            agent_future = data[ag, hist_length-1:, :]
            valid_agent_future = agent_future[(agent_future[:, 0] != 0) & (agent_future[:, 1] != 0)]
            for sample_idx in range(num_sampling):   
                agent_gen = np.concatenate((
                        agent_future[hist_length-1:hist_length, :],
                        samples[sample_idx, ag, :, :]), 
                    axis=0,)
                valid_agent_gen = agent_gen[(agent_gen[:, 0] != 0) & (agent_gen[:, 1] != 0)]                              
                # Evaluate each agent's sampled trajectory
                if (valid_agent_gen.shape[0] > 1) & (valid_agent_future.shape[0] > 1):
                    s_ade, s_fde, s_mr = evaluate_trajectory(valid_agent_gen, valid_agent_future, num_timesteps, kind)
                    if s_ade < 5:
                        metrics.append((s_ade, s_fde, s_mr))        
            if metrics != []:
                # Unpack metrics and store the agent minimum ADE/FDE and agent average MR
                ADE, FDE, MR = zip(*metrics)
                ADE_scenario.append(min(ADE))
                FDE_scenario.append(min(FDE))
                MR_scenario.append(mean(MR))
        
        # Logging the results
        if ADE_scenario != []:
            logging.info(f"{idx:05}- {filename}: minADE_{num_sampling}={mean(ADE_scenario):.3f}, minFDE_{num_sampling}={mean(FDE_scenario):.3f}, MR_{num_sampling}={mean(MR_scenario):.3f}")
            metrics_testset.append((mean(ADE_scenario), mean(FDE_scenario), mean(MR_scenario)))
            print(f'===> Scenario ({idx:05}) {filename} evaluated !')
            # Save best sampled trajectories
            if mean(ADE_scenario) < 0.8 :
                np.save(os.path.join(samples_dir, filename), samples)
                print(f'==> Scenario {filename} sampled and saved !')
        else:
            logging.info(f"This evaluation is conducted for scenario {filename} and all the agent's sampled trajectories are NOT VALID !")
            print(f'===> Scenario ({idx:05}) {filename} discarded !')       
        idx += 1
    
    # Logging the average result on the full testset
    ADE_testset, FDE_testset, MR_testset = zip(*metrics_testset)
    logging.info(
        f"\nThe average evaluation results across all scenarios:\n"
        f"- Average minADE_{num_sampling}={mean(ADE_testset):.3f}\n"
        f"- Average minFDE_{num_sampling}={mean(FDE_testset):.3f}\n"
        f"- Average MR_{num_sampling}={mean(MR_testset):.3f}"
    )
    print(f'====> End of sampling and evaluation.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_sample.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
