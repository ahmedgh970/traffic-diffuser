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


def has_padded_timestep(tracks_data):
    # Check if any timestep has both position values (x, y) as 0 or 0.0
    return np.any(np.all(tracks_data == 0, axis=-1))


# --- Main Pipeline ---
def main():
    """
    Main pipeline for preprocessing scenarios, calculating dataset statistics,
    and standardizing/scaling tracks.
    """
    dataset_path = '/data/tii/data/argoverse/converted/train'

    _, scenario_ids, mapping = read_dataset_summary(dataset_path)
    
    # Preprocess
    for scenario_name in scenario_ids[0:10]:
        scenario = read_scenario(dataset_path=dataset_path, mapping=mapping, scenario_file_name=scenario_name)

        # Extract relevant tracks
        tracks_list = [
            np.column_stack((track_data['state']['position'][:, 0], track_data['state']['position'][:, 1]))
            for track_data in scenario['tracks'].values()
            if track_data['type'] in {'PEDESTRIAN', 'CYCLIST', 'VEHICLE'}
        ]
        tracks_data = np.array(tracks_list)

        test = has_padded_timestep(tracks_data)
        print('has padded timestep:', test)

        #print(tracks_data)

        # Sample tracks to fixed sequence length
        print('original tracks shape:', tracks_data.shape)
        

if __name__ == "__main__":
    main()
