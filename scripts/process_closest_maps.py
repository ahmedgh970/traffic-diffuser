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

def select_closest_segments(trajectory, map_segments):
    # Filter out padded values in the trajectory
    non_padded_mask = np.any(trajectory != 0, axis=1)
    filtered_trajectory = trajectory[non_padded_mask]
    # Interpolate the trajectory to match the length of the map segments
    interpolated_trajectory = interpolate_to_fixed_length(filtered_trajectory, map_segments.shape[1])
    # Calculate ADE for each map segment
    ades = np.array([calculate_ade(interpolated_trajectory, segment) for segment in map_segments])
    return ades


def select_vmap(scenario, map_segments, num_selected_segments):
    """
    Selects the S closest map features for each agent in the scenario based on Average Displacement Error (ADE).

    Parameters:
    - scenario (np.ndarray): The scenario tensor of shape (N_max, L, 2).
    - map_segments (np.ndarray): The map features tensor of shape (Si, 128, 2).
    - num_selected_segments (int): The number of closest map features to select (S).

    Returns:
    - selected_vmap (np.ndarray): Selected map features of shape (N_max, S, 128, 2).
    """
        
    # Filter out the padded agents from the scenario
    non_padded_mask = np.any(scenario != 0, axis=(1, 2))
    filtered_scenario = scenario[non_padded_mask]
    print('Filtered scenario shape', filtered_scenario.shape)
    
    all_ades = []
    for agent_idx in range(filtered_scenario.shape[0]):
        # Get the agent's closest map segments ==> output shape (S, P, 2)
        ades = select_closest_segments(scenario[agent_idx], map_segments)
        all_ades.append(ades)
    
    # Combine ADEs across agents
    all_ades = np.array(all_ades)  # Shape: (N_valid_agents, Si)
    mean_ades = all_ades.mean(axis=0)  # Shape: (Si,)

    # Select the indices of the closest map segments
    closest_indices = np.argsort(mean_ades)[:num_selected_segments]

    # Return the selected map segments
    selected_segments = map_segments[closest_indices]
    return selected_segments


if __name__ == "__main__":
    scene_dir = '/data/ahmed.ghorbel/workdir/autod/backup/data/tracks/train_nag4'
    vmap_dir = '/data/ahmed.ghorbel/workdir/autod/backup/data/maps/full'
    output_dir = '/data/ahmed.ghorbel/workdir/autod/backup/data/maps/filtered/filtered_nag4_46'
    os.makedirs(output_dir, exist_ok=True)
    num_selected_seg = 46
    
    for filename in os.listdir(scene_dir):
        scene = np.load(os.path.join(scene_dir, filename))
        vmap = np.load(os.path.join(vmap_dir, filename))
        print('Initial shape of the vector map', vmap.shape)
        selected_vmap = select_vmap(scene, vmap, num_selected_seg)
        print('Shape of the filtered vector map', selected_vmap.shape)
        np.save(os.path.join(output_dir, filename), selected_vmap)
        print(f'Scenario vector map {filename} filtered and saved.')
    print("Processing complete.")            
        