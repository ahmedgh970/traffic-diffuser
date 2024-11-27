import os
import numpy as np
from statistics import mean
from scipy.interpolate import interp1d



def calculate_ade(agent_traj, feature_traj):
    """
    Calculate the Average Displacement Error (ADE) between an agent's trajectory
    and a map feature trajectory.
    
    Parameters:
    - agent_traj (np.ndarray): Agent's trajectory of shape (L, 2).
    - feature_traj (np.ndarray): Map feature trajectory of shape (L, 2).
    
    Returns:
    - ade (float): Average Displacement Error.
    """
    return np.mean(np.linalg.norm(agent_traj - feature_traj, axis=1))


def interpolate_to_fixed_length(trajectory, num_timesteps, kind):
    """
    Interpolate a variable-length trajectory to a fixed number of timesteps.

    Parameters:
    trajectory (np.ndarray): Input trajectory of shape (L, 2), where L is the number of timesteps.
    num_timesteps (int): The number of timesteps to interpolate to (default is 10).
    kind: choose from 'linear', 'quadratic', 'cubic'.
    
    Returns:
    np.ndarray: Interpolated trajectory of shape (num_timesteps, 2).
    """
    L = trajectory.shape[0]  # Original length of the trajectory

    # Original timesteps
    original_timesteps = np.linspace(0, L - 1, L)

    # New timesteps to interpolate to
    new_timesteps = np.linspace(0, L - 1, num_timesteps)

    # Interpolated trajectory
    interpolated_trajectory = np.zeros((num_timesteps, 2))

    # Interpolate x and y separately
    for i in range(2):  # for each dimension x and y
        interp_func = interp1d(original_timesteps, trajectory[:, i], kind=kind, fill_value='extrapolate')
        interpolated_trajectory[:, i] = interp_func(new_timesteps)

    return interpolated_trajectory


def select_closest_map_features(scenario, map_features, num_selected_features):
    """
    Selects the S closest map features for each agent in the scenario based on Average Displacement Error (ADE).

    Parameters:
    - scenario (np.ndarray): The scenario tensor of shape (N_max, L, 2).
    - map_features (np.ndarray): The map features tensor of shape (Mi, 128, 2).
    - num_selected_features (int): The number of closest map features to select (S).

    Returns:
    - fixed_map_features (np.ndarray): Selected map features of shape (N_max, S, 128, 2).
    """
    N_max, L, _ = scenario.shape
    M_i, seq_len, _ = map_features.shape
    
    # Initialize the fixed map features array
    fixed_map_features = np.zeros((N_max, num_selected_features, seq_len, 2), dtype=map_features.dtype)
    
    for agent_idx in range(N_max):
        # Get the agent's trajectory
        agent_traj = scenario[agent_idx]  # Shape: (L, 2)
        
        # Calculate ADE for the agent's trajectory with all map features        
        ades = np.array([calculate_ade(interpolate_to_fixed_length(agent_traj, seq_len, kind='linear'), map_features[feature_idx]) for feature_idx in range(M_i)])  # Shape: (Mi,)
        
        # Get indices of the closest `num_selected_features` map features based on ADE
        closest_indices = np.argsort(ades)[:num_selected_features]
        
        # Select the closest map features
        fixed_map_features[agent_idx] = map_features[closest_indices]
    
    return fixed_map_features


if __name__ == "__main__":
    scene_dir = '/data/ahmed.ghorbel/workdir/autod/backup/data/nuscenes_trainval_clean_train'
    map_dir = '/data/ahmed.ghorbel/workdir/autod/backup/data/nuscenes_trainval_maps_train'
    output_dir = '/data/ahmed.ghorbel/workdir/autod/backup/data/nuscenes_trainval_maps_clean_train'
    os.makedirs(output_dir, exist_ok=True)
    num_selected_features = 32
    
    for filename in os.listdir(map_dir):
        scene = np.load(os.path.join(scene_dir, filename))
        map_features = np.load(os.path.join(map_dir, filename))
        fixed_map_features = select_closest_map_features(scene, map_features, num_selected_features)
        np.save(os.path.join(output_dir,filename), fixed_map_features)
        print(f'Scenario map {filename} filtered and saved.')
    print("Processing complete.")            
        