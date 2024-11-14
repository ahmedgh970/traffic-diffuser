import numpy as np
from scipy.interpolate import interp1d


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