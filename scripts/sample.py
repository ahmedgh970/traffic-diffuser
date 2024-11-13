import torch
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
from utils import atdd, fd, ade, fde, interpolate 



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

def testset_stats(test_files):
    """
    Count occurrences of each dataset type
    """
    dataset_counts = {'av2': 0, 'nuscenes': 0, 'waymo': 0}
    for file_name in test_files:
        for dataset in dataset_counts:
            if dataset in file_name:
                dataset_counts[dataset] += 1
                break
    stats =""
    for dataset, count in dataset_counts.items():
        stats = stats + f" - {dataset}: {count}\n"
    return stats

def dataset_stats(file_name):
    scale_factor = 100
    dataset_stats = {
        'av2': ([2677.9026, 1098.3357], [3185.974, 1670.7698]),
        'waymo': ([1699.1744, 305.3823], [5284.104, 6511.814]),
        'nuscenes': ([998.90979829, 1372.90628199], [539.07656177, 463.67307649])
    } 
    for dataset_name, (mn, std) in dataset_stats.items():
        if dataset_name in file_name:
            return mn, std, scale_factor
    else:
        raise ValueError(f"Unrecognized dataset for file name: {file_name}")
        
def evaluate_trajectory(gen_traj, gt_traj, num_timesteps=10, kind='linear'):
    """
    Evaluate a single generated trajectory against the future ground truth.
    """
    # Extrapolate trajs to the desired num_timesteps
    gen_traj = interpolate(gen_traj, num_timesteps, kind)
    gt_traj = interpolate(gt_traj, num_timesteps, kind)
    
    # Calculate metrics
    FD = fd(gt_traj, gen_traj)
    ATDD = atdd(gt_traj, gen_traj)
    ADE = ade(gt_traj, gen_traj)
    FDE = fde(gt_traj, gen_traj)
    return FD, ATDD, ADE, FDE
    


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
    max_num_agents=config['model']['max_num_agents']
    seq_length=config['model']['seq_length']
    hist_length=config['model']['hist_length']
    dim_size=config['model']['dim_size']
    
    # Initialize the model:
    # Note that parameter initialization is done within the model constructor
    model_class = load_model(config['model']['module'], config['model']['class'])
    model_name = config['model']['name']
    model = model_class[model_name](
        max_num_agents=max_num_agents,
        seq_length=seq_length,
        hist_length=hist_length,
        dim_size=dim_size,
        map_channels=config['model']['map_channels'],
        use_map_embed=config['model']['use_map_embed'],
        use_ckpt_wrapper=False,
    ).to(device)
    
    # Load a TrafficDiffuser checkpoint:
    state_dict = torch.load(config['model']['ckpt'], map_location=lambda storage, loc: storage)
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
    
    # Print model flops
    batch_size = 1 # to ensure flops and inference time are calculated for a single scenario
    dummy_x = torch.randn(batch_size, max_num_agents, seq_length, dim_size, device=device)
    dummy_t = torch.randn(batch_size, device=device)
    dummy_h = torch.randn(batch_size, max_num_agents, hist_length, dim_size, device=device)
    dummy_m = None
    flops = FlopCountAnalysis(model, (dummy_x, dummy_t, dummy_h, dummy_m))
    gflops = flops.total() / 1e9
    logging.info(f"{model_name} GFLOPs: {gflops:.4f}\n")
    
    # Print model sampling time
    model_kwargs = dict(h=dummy_h, m=dummy_m)
    num_trials = 10
    avg_sampling_time = 0
    for _ in range(num_trials):
        torch.cuda.synchronize()
        tic = time.time()
        samples = diffusion.p_sample_loop(
                model.forward, dummy_x.shape, dummy_x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        torch.cuda.synchronize()
        toc = time.time()
        avg_sampling_time += (toc - tic)
    avg_sampling_time /= num_trials
    logging.info(f"{model_name} Sampling time: {avg_sampling_time:.2f} s\n")
    print('===> Sampling time calculated !')
        
    # Choose a subset of scenarios from testset:            
    test_files = sorted(random.sample(sorted(os.listdir(config['data']['test_dir'])), config['data']['subset_size']))
    logging.info(f"The occurence of each dataset in the testset are:\n{testset_stats(test_files)}\n")
    
    # Make samples directory
    samples_dir = os.path.dirname(config['model']['ckpt']).replace("checkpoints", "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Sample and evaluate scenarios from test_files
    metrics_testset = []
    for filename in test_files:
        data = np.load(os.path.join(config['data']['test_dir'], filename))
        data = torch.tensor(data[:max_num_agents, :, :dim_size], dtype=torch.float32).to(device)        
        data = data.unsqueeze(0).expand(config['sample']['num_sampling'], data.size(0), data.size(1), data.size(2))
        h = data[:, :, :hist_length, :]        
        model_kwargs = dict(h=h)
        
        # Create sampling noise:
        x = torch.randn(config['sample']['num_sampling'], max_num_agents, seq_length, dim_size, device=device)
        
        # Sample trajectories:
        samples = diffusion.p_sample_loop(
            model.forward, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        samples = samples.cpu().numpy()
        
        # Save sampled trajectories
        np.save(os.path.join(samples_dir, filename), samples)
        print(f'===> Scenario {filename} sampled and saved !')
        
        # Scenario evaluation loop
        data_future = data[:, :, hist_length-1:, :].cpu().numpy()   # (N, L_seq_length + 1, D)
        epsilon, num_timesteps, kind = 0.1, seq_length*2, 'linear'
        FD_scenario, ATDD_scenario, ADE_scenario, FDE_scenario  = [], [], [], []

        # Find the matching dataset statistics based on the file name
        mean_xy, std_xy, scale_factor = dataset_stats(filename)

        # Evaluate each agent's trajectories
        for ag in range(samples.shape[1]):
            metrics = []
            for sample_idx in range(samples.shape[0]):   
                agent_future = data_future[sample_idx, ag, :, :]
                agent_gen = samples[sample_idx, ag, :, :]
                agent_gen = np.concatenate((agent_future[0:1, :], agent_gen), axis=0)
                # Filter out points that are too close to (0,0)
                valid_agent_gen = agent_gen[(np.abs(agent_gen[:, 0]) > epsilon) & (np.abs(agent_gen[:, 1]) > epsilon)]
                valid_agent_future = agent_future[(agent_future[:, 0] != 0.0) & (agent_future[:, 1] != 0.0)]              
                # Evaluate each agent's sampled trajectory
                if valid_agent_gen.shape[0] > 1 and valid_agent_future.shape[0] > 1:
                    # unscale and unstanderdize
                    valid_agent_future = (valid_agent_future / scale_factor) * std_xy + mean_xy 
                    valid_agent_gen = (valid_agent_gen / scale_factor) * std_xy + mean_xy
                    metrics.append(evaluate_trajectory(valid_agent_gen, valid_agent_future, num_timesteps, kind))           
            if metrics != []:
                # Unpack metrics and store the average
                FD, ATDD, ADE, FDE = zip(*metrics)
                # select average (mean) or minimum (min) by number of samples               
                FD_scenario.append(mean(FD))
                ATDD_scenario.append(mean(ATDD))
                ADE_scenario.append(mean(ADE))
                FDE_scenario.append(mean(FDE))
        
        # Logging the results
        if FD_scenario == []:
            logging.info(f"This evaluation is conducted for scenario {filename} and all the agent's sampled trajectories are NOT VALID !")
        else:
            logging.info(f"This evaluation is conducted for scenario {filename} and involves averaging across all agents and samples:")
            logging.info(f" - Frechet Distance (FD): {mean(FD_scenario)}")
            logging.info(f" - Absolute Traveled Distance Difference (ATDD): {mean(ATDD_scenario)}")
            logging.info(f" - Average Displacement Error (ADE): {mean(ADE_scenario)}")
            logging.info(f" - Final Displacement Error (FDE): {mean(FDE_scenario)} \n")
            metrics_testset.append((mean(FD_scenario), mean(ATDD_scenario), mean(ADE_scenario), mean(FDE_scenario)))
        print(f'===> Scenario {filename} evaluated !')
    
    # Logging the average result on the full testset
    FD, ATDD, ADE, FDE = zip(*metrics_testset)
    logging.info(f"The average evaluation results across all scenarios:")
    logging.info(f" - Frechet Distance (FD): {mean(FD)}")
    logging.info(f" - Absolute Traveled Distance Difference (ATDD): {mean(ATDD)}")
    logging.info(f" - Average Displacement Error (ADE): {mean(ADE)}")
    logging.info(f" - Final Displacement Error (FDE): {mean(FDE)} \n")
    print(f'===> End of sampling and evaluation.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_sample.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
