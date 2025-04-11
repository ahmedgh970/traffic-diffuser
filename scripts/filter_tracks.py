import os
import shutil
import random

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage
import cv2
from statistics import mean
import importlib
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from collections import Counter
from scipy.stats import zscore


# --- Utility Functions ---
def is_padded(position):
    """Check if a given position is padded (zeroed)."""
    return np.all(position == 0.0)

def traveled_distances(tracks):
    """Calculate the traveled distances while ignoring padded values."""
    non_padded_mask = ~(np.all(tracks == 0.0, axis=2))
    diffs = np.diff(tracks, axis=1)
    valid_diffs_mask = non_padded_mask[:, :-1] & non_padded_mask[:, 1:]
    distances = np.linalg.norm(diffs * valid_diffs_mask[:, :, np.newaxis], axis=2)
    return distances.sum(axis=1)

# --- Data Processing Functions ---
def process_scenario(tracks_data, min_tdist):
    """
    Extract, sample, and filter tracks for a given scenario.
    Remove stationary agents based on traveled distance thresholds.
    """
    h3_tdist = traveled_distances(tracks_data[:, :3, :])
    h6_tdist = traveled_distances(tracks_data[:, :6, :])
    h8_tdist = traveled_distances(tracks_data[:, :8, :])
    f8_tdist = traveled_distances(tracks_data[:, 3:, :])
    f5_tdist = traveled_distances(tracks_data[:, 6:, :])
    f3_tdist = traveled_distances(tracks_data[:, 8:, :]) 

    non_stationary = (
        (h3_tdist >= 3 * min_tdist) & 
        (h6_tdist >= 6 * min_tdist) & 
        (h8_tdist >= 8 * min_tdist) & 
        (f3_tdist >= 3 * min_tdist) & 
        (f5_tdist >= 5 * min_tdist) & 
        (f8_tdist >= 8 * min_tdist)
    )

    tracks_filtered = tracks_data[non_stationary]
    return tracks_filtered

def calculate_dataset_statistics(dataset_path):
    """
    Calculate Mean/Std values for x and y positions throughout the dataset, excluding zero padding.
    """
    scenarios_list = []
    for filename in os.listdir(dataset_path):
        scenario = np.load(os.path.join(dataset_path, filename))
        mask = np.any(scenario != 0, axis=-1)
        scenarios_list.append(scenario[mask])
         
    dataset = np.concatenate(scenarios_list, axis=0)
    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0)
    return mean, std

def clean_trajectory(data, threshold):
    """
    Remove outliers from the trajectory based on z-scores, while ignoring padded values.
    """
    for agent_idx in range(data.shape[0]):
        agent_traj = data[agent_idx, :, :]
        valid_mask = ~np.all(agent_traj == 0.0, axis=1)
        valid_traj = agent_traj[valid_mask]

        if valid_traj.shape[0] > 1:
            z_scores = np.abs(zscore(valid_traj, axis=0))
            outlier_mask = (z_scores > threshold)
            valid_traj[outlier_mask] = 0.0
            agent_traj[valid_mask] = valid_traj
        
        data[agent_idx, :, :] = agent_traj
    return data

def pad_save_data(data, filename, N_MAX, OUTPUT_DIRECTORY):
    """
    Save the cleaned data to the output directory after processing.
    """
    N = data.shape[0]
    if N != 0:
        if N < N_MAX:
            padding = np.zeros((N_MAX - N, 11, 2))
            data = np.vstack((data, padding))
        else:
            data = data[:N_MAX]
        
        output_path = os.path.join(OUTPUT_DIRECTORY, filename)
        np.save(output_path, data)
        print(f"Processed and saved cleaned data for file: {filename}")
    else:
        print(f"Skipping file {filename} due to zero-length data after processing.")

# --- Main Function ---
def main():
    # Constants
    INPUT_DIRECTORY = '/data/tii/data/argoverse/tracks_with_stas_and_ood/val'
    OUTPUT_DIRECTORY = '/data/tii/data/merged/tracks/val_av2'
    THRESHOLD = 2.0
    MIN_TDIST = 1.4
    N_MAX = 20

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    print("Starting data cleaning process...")
    for filename in os.listdir(INPUT_DIRECTORY):
        file_path = os.path.join(INPUT_DIRECTORY, filename)
        data = np.load(file_path)
        print(f"Processing file: {filename}")
        
        # Clean the trajectory by removing outliers
        cleaned_data = clean_trajectory(data, THRESHOLD)

        # Remove stationary agents
        final_data = process_scenario(cleaned_data, MIN_TDIST)
        
        # Save the cleaned data
        pad_save_data(final_data, filename, N_MAX, OUTPUT_DIRECTORY)

    mean, std = calculate_dataset_statistics(OUTPUT_DIRECTORY)
    print(f'Mean and Std of the dataset are {mean}, {std}')

if __name__ == "__main__":
    main()
