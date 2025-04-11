import os
import numpy as np
from scipy.interpolate import interp1d
from scenarionet import read_dataset_summary, read_scenario


# --- Function: Interpolate trajectory to fixed length ---
def interpolate_to_fixed_length(trajectory, num_timesteps, kind='linear'):
    """
    Interpolate a variable-length trajectory to a fixed number of timesteps.

    Parameters:
        trajectory (np.ndarray): Input trajectory of shape (L, 2).
        num_timesteps (int): The number of timesteps to interpolate to.
        kind (str): Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
        np.ndarray: Interpolated trajectory of shape (num_timesteps, 2).
    """
    L = trajectory.shape[0]  # Original length of the trajectory
    original_timesteps = np.linspace(0, L - 1, L)
    new_timesteps = np.linspace(0, L - 1, num_timesteps)

    interpolated_trajectory = np.zeros((num_timesteps, 2))
    for i in range(2):
        interp_func = interp1d(original_timesteps, trajectory[:, i], kind=kind, fill_value='extrapolate')
        interpolated_trajectory[:, i] = interp_func(new_timesteps)

    return interpolated_trajectory

# --- Function: Slice array based on spatial distance ---
def slice_array_based_on_condition(arr, epsilon):
    """
    Slice an array into segments based on spatial continuity.

    Parameters:
        arr (np.ndarray): Input array of shape (N, 2).
        epsilon (float): Maximum allowed distance between consecutive points.

    Returns:
        list: List of array slices.
    """
    slices = []
    start_idx = 0
    for i in range(len(arr) - 1):
        x_diff = abs(arr[i, 0] - arr[i + 1, 0])
        y_diff = abs(arr[i, 1] - arr[i + 1, 1])
        if x_diff > epsilon or y_diff > epsilon:
            slices.append(arr[start_idx:i + 1])
            start_idx = i + 1

    if start_idx < len(arr):
        slices.append(arr[start_idx:])

    return slices

# --- Function: Process map data ---
def extract_process_map(scenario_name, mapping, dataset_path,
                        num_ts_interm, num_ts_out):
    """
    Extract and preprocess map features for a given scenario.

    Parameters:
        scenario_name (str): Name of the scenario file.
        mapping (dict): Dataset mapping information.
        dataset_path (str): Path to the dataset.
        num_ts_interm (int): Number of intermediate timesteps for interpolation.
        num_ts_out (int): Number of final timesteps for interpolation.

    Returns:
        np.ndarray: Processed map data.
    """
    scenario = read_scenario(dataset_path=dataset_path, mapping=mapping, scenario_file_name=scenario_name)

    tracks = scenario['tracks']
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')
    for track in tracks.values():
        xyz = track['state']['position']
        valid_points = xyz[(xyz[:, 0] != 0.0) & (xyz[:, 1] != 0.0)]
        if valid_points.shape[0] > 0:
            min_x, min_y = min(min_x, np.min(valid_points[:, 0])), min(min_y, np.min(valid_points[:, 1]))
            max_x, max_y = max(max_x, np.max(valid_points[:, 0])), max(max_y, np.max(valid_points[:, 1]))

    arr_list = []
    for k, features in scenario['map_features'].items():
        for kk, feature in features.items():
            if kk in ['polyline', 'polygon']:
                arr = feature[:, :2]
                if arr.shape[0] > 1:
                    arr = interpolate_to_fixed_length(arr, num_timesteps=num_ts_interm, kind='linear')
                    cropped_arr = arr[
                        (arr[:, 0] >= min_x - 50) & (arr[:, 0] <= max_x + 50) &
                        (arr[:, 1] >= min_y - 50) & (arr[:, 1] <= max_y + 50)
                    ]
                    if cropped_arr.shape[0] > 1:
                        cropped_arr_slices = slice_array_based_on_condition(cropped_arr, epsilon=50)
                        for arr_slice in cropped_arr_slices:
                            if arr_slice.shape[0] > 1:
                                arr_slice = interpolate_to_fixed_length(arr_slice, num_timesteps=num_ts_out, kind='linear')
                                arr_list.append(arr_slice)

    map_array = np.array(arr_list)
    # Standardize and scale processed map
    #map_array = (map_array - mean) / std * scale_factor
    return map_array


# --- Main Pipeline ---
def main():
    """
    Main pipeline for processing map data from different datasets.
    """
    dataset_path = "/data/tii/data/waymo/converted/test"
    reference_path = "/data/tii/data/waymo/tracks/test"
    output_path = "/data/tii/data/waymo/maps/scene"
    os.makedirs(output_path, exist_ok=True)
    # default map processing settings
    num_ts_interm, num_ts_out = 25000, 128
    #scale_factor = 100
    # nuScenes statistics
    #mean = [998.90979829, 1372.90628199]
    #std = [539.07656177, 463.67307649]
    
    _, scenario_ids, mapping = read_dataset_summary(dataset_path=dataset_path)
    i = 0
    for scenario_name in os.listdir(reference_path):
        scenario_pkl = scenario_name.replace('.npy', '.pkl')
        processed_map = extract_process_map(
            scenario_pkl, mapping, dataset_path,
            num_ts_interm, num_ts_out,
        )
        file_path = os.path.join(output_path, scenario_name)
        np.save(file_path, processed_map)
        print(f"Processed map {i} and saved to {file_path}")
        i += 1

if __name__ == "__main__":
    main()
