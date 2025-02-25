import os
import numpy as np
from statistics import mean
from scipy.interpolate import interp1d


def calculate_ade(traj1, traj2):
    return np.mean(np.linalg.norm(traj1 - traj2, axis=1))
    
def interpolate_to_fixed_length(traj, length, kind='linear'):
    current_length = traj.shape[0]
    x = np.linspace(0, 1, current_length)
    f = interp1d(x, traj, axis=0, kind=kind)
    x_new = np.linspace(0, 1, length)
    return f(x_new)

def select_closest_segments(trajectory, map_segments, num_selected_segments):
    # Filter out padded values in the trajectory
    non_padded_mask = np.any(trajectory != 0, axis=1)
    filtered_trajectory = trajectory[non_padded_mask]
    
    # Interpolate the trajectory to match the length of the map segments
    interpolated_trajectory = interpolate_to_fixed_length(filtered_trajectory, map_segments.shape[1])
    
    # Calculate ADE for each map segment
    ades = np.array([calculate_ade(interpolated_trajectory, segment) for segment in map_segments])
    
    # Get indices of the closest `num_selected_segments` map segments
    closest_indices = np.argsort(ades)[:num_selected_segments]
    
    # Select the closest map segments
    closest_segments = map_segments[closest_indices]
    
    return closest_segments


def select_vmap(scenario, map_features, num_selected_features):
    """
    Selects the S closest map features for each agent in the scenario based on Average Displacement Error (ADE).

    Parameters:
    - scenario (np.ndarray): The scenario tensor of shape (N_max, L, 2).
    - map_features (np.ndarray): The map features tensor of shape (Si, 128, 2).
    - num_selected_features (int): The number of closest map features to select (S).

    Returns:
    - selected_vmap (np.ndarray): Selected map features of shape (N_max, S, 128, 2).
    """

    # Initialize the fixed map features array
    selected_vmap = np.zeros((scenario.shape[0], num_selected_features, map_features.shape[1], map_features.shape[2]), dtype=map_features.dtype)
    
    # Filter out the padded agents from the scenario
    non_padded_mask = np.any(scenario != 0, axis=(1, 2))
    filtered_scenario = scenario[non_padded_mask]
    print('Filtered scenario shape', filtered_scenario.shape)

    for agent_idx in range(filtered_scenario.shape[0]):
        # Get the agent's closest map segments ==> output shape (S, P, 2)
        agent_closest_segments = select_closest_segments(scenario[agent_idx], map_features, num_selected_features)
        selected_vmap[agent_idx] = agent_closest_segments  
    return selected_vmap


if __name__ == "__main__":
    scene_dir = '/data/ahmed.ghorbel/workdir/autod/traffic-diffuser/data/tracks/train_nag4'
    vmap_dir = '/data/ahmed.ghorbel/workdir/autod/traffic-diffuser/data/maps/full'
    output_dir = '/data/ahmed.ghorbel/workdir/autod/traffic-diffuser/data/maps/multi_nag4_s16'
    os.makedirs(output_dir, exist_ok=True)
    num_selected_seg = 16
    
    for filename in os.listdir(scene_dir):
        scene = np.load(os.path.join(scene_dir, filename))
        vmap = np.load(os.path.join(vmap_dir, filename))
        print('Initial shape of the vector map', vmap.shape)
        scene = scene[:, :, :]
        selected_vmap = select_vmap(scene, vmap, num_selected_seg)
        print('Shape of the filtered vector map', selected_vmap.shape)
        np.save(os.path.join(output_dir, filename), selected_vmap)
        print(f'Scenario vector map {filename} filtered and saved.')
    print("Processing complete.")            
        