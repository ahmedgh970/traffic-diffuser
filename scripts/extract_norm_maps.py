import os
import numpy as np
from scipy.interpolate import interp1d
from scenarionet import read_dataset_summary, read_scenario



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


def slice_array_based_on_condition(arr, epsilon):
    slices = []
    start_idx = 0

    # Iterate through the array to find break points
    for i in range(len(arr) - 1):
        x_diff = abs(arr[i, 0] - arr[i + 1, 0])
        y_diff = abs(arr[i, 1] - arr[i + 1, 1])

        if x_diff > epsilon or y_diff > epsilon:
            # If the condition is met, slice the array
            slices.append(arr[start_idx:i+1])
            start_idx = i + 1

    # Add the last slice
    if start_idx < len(arr):
        slices.append(arr[start_idx:])
    
    return slices


def extract_process_map(scenario_name, mapping, dataset_path, output_dir, num_timesteps_interm, num_timesteps_out):
    scenario = read_scenario(dataset_path=dataset_path, mapping=mapping, scenario_file_name=scenario_name)
    print(f'Map of the scenario {scenario_name} is being processed...')

    # Determine the min max position of agents in the scenario 
    tracks = scenario['tracks']
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0., 0.
    for idx, (id, track) in enumerate(tracks.items()):
        xyz = track['state']['position']  #-- array of shape (seq_length, 3)
        # Filter out invalid points
        valid_points = xyz[(xyz[:, 0] != 0.0) & (xyz[:, 1] != 0.0)]
        if valid_points.shape[0] > 0:
            if max(valid_points[:, 0]) > max_x:
                max_x = max(valid_points[:, 0])
            if max(valid_points[:, 1]) > max_y:
                max_y = max(valid_points[:, 1])
            if min(valid_points[:, 0]) < min_x:
                min_x = min(valid_points[:, 0])
            if min(valid_points[:, 1]) < min_y:
                min_y = min(valid_points[:, 1])

    # Crop the map feature of the scenario and filter undesired points 
    arr_list = []
    for k in scenario['map_features'].keys():
        for kk in scenario['map_features'][k].keys():
            if kk in ['polyline', 'polygon']:
                arr = scenario['map_features'][k][kk]
                arr = arr[:, :2]
                if arr.shape[0]>1 :
                    # interpolate 
                    arr = interpolate_to_fixed_length(arr, num_timesteps=num_timesteps_interm, kind='linear')
                    # crop to max_x max_y
                    cropped_arr = arr[
                        (arr[:, 0] >= min_x - 50) & 
                        (arr[:, 0] <= max_x + 50) &
                        (arr[:, 1] >= min_y - 50) &
                        (arr[:, 1] <= max_y + 50)]
                    
                    if cropped_arr.shape[0] > 1:
                        # slice the cropped array to avoid weird cnx
                        cropped_arr_slices = slice_array_based_on_condition(cropped_arr, epsilon=50)
                        for arr_slice in cropped_arr_slices:
                            if arr_slice.shape[0] > 1:
                                # interpolate
                                arr_slice = interpolate_to_fixed_length(arr_slice, num_timesteps=num_timesteps_out, kind='linear')
                                arr_list.append(arr_slice)
    
    arr_map = np.array(arr_list)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, scenario_name.replace('.pkl', '.npy')), arr_map)
    print(f'Scenario map of shape {arr_map.shape} and id "{scenario_name}" has been saved!')


def standardize_scale(input_dir, output_dir, mean, std, scale_factor):
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize each scenario map
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        scenario = np.load(file_path)

        # Standardize scenario map
        scenario = (scenario - mean) / std
        
        # Scale scenario map
        scenario = scenario * scale_factor
        
        # Save processed scenario
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, scenario)
        print(f'Scenario map {filename} standardized, scaled, and saved.')
    print("Processing complete.")

    

def main():
    dataset_path = '/data/tii/data/nuscenes/pkl'
    output_dir = '/data/tii/data/nuscenes/maps'
    mean, std = [998.90979829, 1372.90628199], [539.07656177, 463.67307649]
    _, scenario_ids, mapping = read_dataset_summary(dataset_path=dataset_path)
    
    for scenario_name in scenario_ids:
        extract_process_map(scenario_name, mapping, dataset_path, output_dir, num_timesteps_interm=50000, num_timesteps_out=128)
    
    standardize_scale(output_dir, output_dir, mean, std, scale_factor=100)
    
    dataset_path = '/data/tii/data/waymo/pkl'
    output_dir = '/data/tii/data/waymo/maps'
    mean, std = [1699.1744,  305.3823], [5284.104, 6511.814]
    _, scenario_ids, mapping = read_dataset_summary(dataset_path=dataset_path)
    
    for scenario_name in scenario_ids:
        extract_process_map(scenario_name, mapping, dataset_path, output_dir, num_timesteps_interm=50000, num_timesteps_out=128)    
    
    standardize_scale(output_dir, output_dir, mean, std, scale_factor=100)
    
    dataset_path = '/data/tii/data/argoverse/pkl/train_pkl'
    output_dir = '/data/tii/data/argoverse/maps'
    mean, std = [2677.9026, 1098.3357], [3185.974,  1670.7698]
    _, scenario_ids, mapping = read_dataset_summary(dataset_path=dataset_path)
    
    for scenario_name in scenario_ids:
        extract_process_map(scenario_name, mapping, dataset_path, output_dir, num_timesteps_interm=50000, num_timesteps_out=128)
        
    dataset_path = '/data/tii/data/argoverse/pkl/val_pkl'
    output_dir = '/data/tii/data/argoverse/maps'
    _, scenario_ids, mapping = read_dataset_summary(dataset_path=dataset_path)
    
    for scenario_name in scenario_ids:
        extract_process_map(scenario_name, mapping, dataset_path, output_dir, num_timesteps_interm=50000, num_timesteps_out=128)            
    
    standardize_scale(output_dir, output_dir, mean, std, scale_factor=100)
    
if __name__ == "__main__":
    main()