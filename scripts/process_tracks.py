import os
import numpy as np
from scenarionet import read_dataset_summary, read_scenario


def interpolate_to_fixed_length(data, fixed_length=11):
    """
    Interpolates a variable-length numpy array to a fixed length.

    Parameters:
        data (np.ndarray): The input array of shape (N, L, 2) where L is between 90 and 94.
        fixed_length (int): The desired fixed length (default is 11).

    Returns:
        np.ndarray: Interpolated array of shape (N, fixed_length, 2).
    """
    # Extract dimensions
    N, L, D = data.shape
    
    # Prepare the interpolated array
    interpolated_data = np.zeros((N, fixed_length, D))
    
    # Loop over each sequence (agent)
    for i in range(N):
        # Generate the fixed-length indices
        fixed_indices = np.linspace(0, L - 1, fixed_length)

        # Interpolate for each dimension (x and y)
        for j in range(D):
            interpolated_data[i, :, j] = np.interp(fixed_indices, np.arange(L), data[i, :, j])
    
    return interpolated_data

# --- Function: Preprocess scenarios ---
def extract_process_scenario(scenario_name, mapping, dataset_path, min_tdist, seq_length):
    """
    Extract, sample, and filter tracks for a given scenario.

    Parameters:
        scenario_name (str): Name of the scenario file.
        mapping (dict): Dataset mapping information.
        dataset_path (str): Path to the dataset.
        min_tdist (float): Minimum travel distance to filter stationary agents.
        seq_length (int): The desired sequence length to interpolate to.

    Returns:
        np.ndarray: Preprocessed tracks.
    """
    scenario = read_scenario(dataset_path=dataset_path, mapping=mapping, scenario_file_name=scenario_name)

    # Extract relevant tracks
    tracks_list = [
        np.column_stack((track_data['state']['position'][:, 0], track_data['state']['position'][:, 1]))
        for track_data in scenario['tracks'].values()
        if track_data['type'] in {'PEDESTRIAN', 'CYCLIST', 'VEHICLE'}
    ]
    tracks = np.array(tracks_list)

    # Sample tracks to fixed sequence length
    print('original tracks shape:', tracks.shape)
    tracks_data = interpolate_to_fixed_length(tracks, seq_length)
    tracks_data = tracks_data[:, :11, :]
    print('interp tracks shape:', tracks_data.shape)
    
    # Fct to calculate the agents traveled distances in a scenario
    def traveled_distances(tracks):
        non_padded_mask = ~(np.all(tracks == 0.0, axis=2))
        diffs = np.diff(tracks, axis=1)
        valid_diffs_mask = non_padded_mask[:, :-1] & non_padded_mask[:, 1:]
        distances = np.linalg.norm(diffs * valid_diffs_mask[:, :, np.newaxis], axis=2)
        return distances.sum(axis=1)
    
    # Remove stationary agents
    h3_tdist = traveled_distances(tracks_data[:, :3, :])
    h6_tdist = traveled_distances(tracks_data[:, :6, :])
    h8_tdist = traveled_distances(tracks_data[:, :8, :])
    f8_tdist = traveled_distances(tracks_data[:, 3:, :])
    f5_tdist = traveled_distances(tracks_data[:, 6:, :])
    f3_tdist = traveled_distances(tracks_data[:, 8:, :]) 

    non_stationary = (h3_tdist >= 3*min_tdist) & (h6_tdist >= 6*min_tdist) & (h8_tdist >= 8*min_tdist) & \
        (f3_tdist >= 3*min_tdist) & (f5_tdist >= 5*min_tdist) & (f8_tdist >= 8*min_tdist)

    tracks_filtered = tracks_data[non_stationary]
    print(tracks_filtered.shape)
    return tracks_filtered

# --- Function: Calculate dataset statistics ---
def calculate_dataset_statistics(dataset_path):
    """
    Calculate Mean/Std values for x and y positions throughout the dataset, excluding zero padding.

    Parameters:
        dataset_path (str): Path to the processed dataset.

    Returns:
        tuple: Mean and standard deviation for the dataset.
    """
    scenarios_list = []
    for filename in os.listdir(dataset_path):
        scenario = np.load(os.path.join(dataset_path, filename))
        mask = np.any(scenario != 0, axis=-1)
        scenarios_list.append(scenario[mask])   
         
    dataset = np.concatenate(scenarios_list, axis=0)
    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0)
    return mean, std

# --- Function: Standardize and scale scenario ---
def standardize_scale_scenario(tracks, mean, std, scale_factor):
    """
    Standardize and scale tracks using dataset-specific statistics.

    Parameters:
        tracks (np.ndarray): Tracks to be standardized and scaled.
        mean (np.ndarray): Mean values for the dataset.
        std (np.ndarray): Standard deviation values for the dataset.
        scale_factor (float): Scale factor to apply after standardization.

    Returns:
        np.ndarray: Standardized and scaled tracks.
    """
    mask = np.any(tracks != 0, axis=-1)
    tracks[mask] = ((tracks[mask] - mean) / std) * scale_factor
    return tracks
   
# --- Main Pipeline ---
def main():
    """
    Main pipeline for preprocessing scenarios, calculating dataset statistics,
    and standardizing/scaling tracks.
    """
    dataset_path = '/data/tii/data/argoverse/converted/train'
    output_path = '/data/tii/data/argoverse/tracks/train'
    os.makedirs(output_path, exist_ok=True)
    
    min_tdist = 1.4       # Average humain walking speed is approximately 1.4 meters per second (m/s)
    seq_length = 11       # Desired sequence length to train and test on (can be padded)
    scale_factor = 100    # Empirically determined for all datasets

    _, scenario_ids, mapping = read_dataset_summary(dataset_path)
    
    # Preprocess
    for scenario_name in scenario_ids:
        processed_scenario = extract_process_scenario(
            scenario_name, mapping, dataset_path, min_tdist=min_tdist, seq_length=seq_length
        )
        if processed_scenario.shape[0] != 0:
            np.save(os.path.join(output_path, scenario_name.replace('.pkl', '.npy')), processed_scenario)
            print(f"Preprocessed and saved {scenario_name.replace('.pkl', '.npy')} with shape {processed_scenario.shape}")
    
    # Standardize and scale
    mean, std = calculate_dataset_statistics(output_path)
    print(f'Mean and Std of {dataset_path} are {mean}, {std}')

    #for scenario_name in os.listdir(output_path):
    #    tracks = np.load(os.path.join(output_path, scenario_name))
    #    tracks = standardize_scale_scenario(tracks, mean, std, scale_factor)
    #    np.save(os.path.join(output_path, scenario_name), tracks)
    #    print(f"Standardized and saved {scenario_name}")

if __name__ == "__main__":
    main()
