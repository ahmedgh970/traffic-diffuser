import os
import numpy as np
from scenarionet import read_dataset_summary, read_scenario

# --- Function: Preprocess scenarios ---
def extract_process_scenario(scenario_name, mapping, dataset_path, min_distance, step):
    """
    Extract, sample, and filter tracks for a given scenario.

    Parameters:
        scenario_name (str): Name of the scenario file.
        mapping (dict): Dataset mapping information.
        dataset_path (str): Path to the dataset.
        min_distance (float): Minimum travel distance to filter stationary agents.
        step (int): Step size for sampling sequence length.

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
    tracks_data = np.array(tracks_list)

    # Sample tracks to fixed sequence length
    print('original tracks shape:', tracks_data.shape)
    tracks_sampled = tracks_data[:, ::step, :]
    print('sampled tracks shape:', tracks_sampled.shape)
    
    # Remove stationary agents
    def calculate_traveled_distance(tracks):
        non_padded_mask = ~(np.all(tracks == 0.0, axis=2))
        diffs = np.diff(tracks, axis=1)
        valid_diffs_mask = non_padded_mask[:, :-1] & non_padded_mask[:, 1:]
        distances = np.linalg.norm(diffs * valid_diffs_mask[:, :, np.newaxis], axis=2)
        return distances.sum(axis=1)

    tracks_sampled_history, tracks_sampled_future = tracks_sampled[:, :8, :], tracks_sampled[:, 8:, :]
    history_traveled_distance = calculate_traveled_distance(tracks_sampled_history)
    future_traveled_distance = calculate_traveled_distance(tracks_sampled_future)

    non_stationary = (history_traveled_distance > min_distance) & (future_traveled_distance > min_distance)
    return tracks_sampled[non_stationary]

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
    dataset_path = '/data/tii/data/waymo/pkl'
    output_path = '/data/tii/data/waymo/npy'
    os.makedirs(output_path, exist_ok=True)
    
    min_distance = 4      # Empirically determined for all datasets
    step = 16             # Depends on the original dataset scenario length
    scale_factor = 100    # Empirically determined for all datasets

    _, scenario_ids, mapping = read_dataset_summary(dataset_path)
    
    # Preprocess
    for scenario_name in scenario_ids:
        processed_scenario = extract_process_scenario(
            scenario_name, mapping, dataset_path, min_distance=min_distance, step=step
        )
        if processed_scenario.shape[0] != 0:
            np.save(os.path.join(output_path, scenario_name.replace('.pkl', '.npy')), processed_scenario)
            print(f"Preprocessed and saved {scenario_name.replace('.pkl', '.npy')} with shape {processed_scenario.shape}")
    
    # Standardize and scale
    mean, std = calculate_dataset_statistics(output_path)
    for scenario_name in os.listdir(output_path):
        tracks = np.load(os.path.join(output_path, scenario_name))
        tracks = standardize_scale_scenario(tracks, mean, std, scale_factor)
        np.save(os.path.join(output_path, scenario_name), tracks)
        print(f"Standardized and saved {scenario_name}")

if __name__ == "__main__":
    main()
