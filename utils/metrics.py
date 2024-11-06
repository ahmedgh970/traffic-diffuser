import numpy as np
from scipy.spatial.distance import euclidean


def calculate_diff_distance_traveled(trajectory1, trajectory2):
    """
    Calculate the absolute difference of total distances traveled by two trajectories.

    Parameters:
    trajectory1 (np.ndarray): Trajectory of shape (L, 2).
    trajectory2 (np.ndarray): Generated trajectory of shape (L, 2).

    Returns:
    float: Total distance traveled.
    """
    distances_gen = np.linalg.norm(np.diff(trajectory1, axis=0), axis=1)
    distances_gt = np.linalg.norm(np.diff(trajectory2, axis=0), axis=1)
    
    return np.abs(np.sum(distances_gen) - np.sum(distances_gt))


def calculate_frechet_distance(trajectory1, trajectory2):
    """
    Calculate the Frechet Distance between two trajectories.

    Parameters:
    trajectory1 (np.ndarray): Ground truth trajectory of shape (L, 2).
    trajectory2 (np.ndarray): Generated trajectory of shape (L, 2).

    Returns:
    float: Frechet distance.
    """
    ca = np.full((len(trajectory1), len(trajectory2)), -1.0)

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = euclidean(trajectory1[0], trajectory2[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i-1, 0), euclidean(trajectory1[i], trajectory2[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j-1), euclidean(trajectory1[0], trajectory2[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)), euclidean(trajectory1[i], trajectory2[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return _c(len(trajectory1) - 1, len(trajectory2) - 1)


def calculate_ade(trajectory1, trajectory2):
    """
    Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Parameters:
    trajectory1 (np.ndarray): Ground truth trajectory of shape (L, 2).
    trajectory2 (list of np.ndarray): List of generated trajectories, each of shape (L, 2).

    Returns:
    float, float: ADE.
    """
    de = np.linalg.norm(trajectory1 - trajectory2, axis=1)
    ade = np.mean(de)
    return ade


def calculate_fde(trajectory1, trajectory2):
    """
    Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Parameters:
    trajectory1 (np.ndarray): Ground truth trajectory of shape (L, 2).
    trajectory2 (list of np.ndarray): List of generated trajectories, each of shape (L, 2).

    Returns:
    float, float: FDE.
    """
    fde = np.linalg.norm(trajectory1[-1] - trajectory2[-1])
    return fde