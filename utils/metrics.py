import torch 

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance, norm


def calculate_polygone_area(trajectory1, trajectory2):
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Parameters:
    trajectory1 (np.ndarray): Ground truth trajectory of shape (L, 2).
    trajectory2 (np.ndarray): Generated trajectory of shape (L, 2).
    
    Returns:
    float: The area of the polygon.
    
    Problems:
    Works with a polygon that does not intersect itself
    """
    points = np.concatenate((trajectory1, trajectory2[::-1]), axis=0)
    
    # Ensure that the polygon is closed
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    x = points[:, 0]
    y = points[:, 1]
    
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return area

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
    ade = np.mean([np.mean(np.linalg.norm(trajectory1 - trajectory2, axis=1))])
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
    fde = np.mean([np.linalg.norm(trajectory1[-1] - trajectory2[-1])])
    return fde


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



def sliced_wasserstein(dist_1, dist_2, n_slices=100):
    """Compute sliced Wasserstein distance between two distributions.
    Assumes that both ``dist_1`` and ``dist_2`` have the same dimension.
    Parameters
    ----------
    dist_1 : Tensor
    dist_2 : Tensor
    n_slices : int, default=100
        The number of the considered random projections.
    Return
    ------
    sw_distance : float
    """
    if dist_1.ndim > 2:
        dist_1 = dist_1.reshape(dist_1.shape[0], -1)
        dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    projections = torch.randn(size=(n_slices, dist_1.shape[1]), device=dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T

    dist_1_projected = dist_1_projected.cpu().numpy()
    dist_2_projected = dist_2_projected.cpu().numpy()
    return np.mean(
        [wasserstein_distance(u_values=d1, v_values=d2) for d1, d2 in zip(dist_1_projected, dist_2_projected)]
    )