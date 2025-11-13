
import torch
import numpy as np

def normalize(data):
    """
    Normalizes the input data to the range (-1, 1) for each feature.
    The input data can be a PyTorch tensor or a NumPy ndarray with shape (N, T, F).
    
    Returns:
        normalized_data: data normalized to (-1, 1) with the same type as input.
        min_vals: per-feature minimum values (shape: (F,))
        max_vals: per-feature maximum values (shape: (F,))
    """
    # Determine whether we are working with a torch tensor or numpy array
    if isinstance(data, torch.Tensor):
        # Compute the min and max over axes 0 and 1 (across N and T)
        min_vals = data.amin(dim=(0, 1))
        max_vals = data.amax(dim=(0, 1))
        
        # Reshape to (1, 1, F) for broadcasting
        min_vals_broadcast = min_vals.view(1, 1, -1)
        max_vals_broadcast = max_vals.view(1, 1, -1)
        
        # Avoid division by zero: if max equals min, set denominator to 1.
        range_vals = (max_vals_broadcast - min_vals_broadcast)
        range_vals[range_vals == 0] = 1.0
        
        normalized = 2 * (data - min_vals_broadcast) / range_vals - 1
        
        return normalized, min_vals, max_vals

    elif isinstance(data, np.ndarray):
        # Compute min and max over axis 0 and 1 for each feature
        min_vals = data.min(axis=(0, 1))
        max_vals = data.max(axis=(0, 1))
        
        # Reshape for broadcasting
        min_vals_broadcast = min_vals.reshape(1, 1, -1)
        max_vals_broadcast = max_vals.reshape(1, 1, -1)
        
        # Avoid division by zero
        range_vals = (max_vals_broadcast - min_vals_broadcast)
        range_vals[range_vals == 0] = 1.0
        
        normalized = 2 * (data - min_vals_broadcast) / range_vals - 1
        
        return normalized, min_vals, max_vals

    else:
        raise TypeError("Input data must be a torch.Tensor or a numpy.ndarray.")

def inverse_normalize(normalized_data, min_vals, max_vals):
    """
    Recovers the original data from the normalized data given per-feature min and max.
    The normalized_data should be in the range (-1, 1).
    
    Args:
        normalized_data: Normalized data (same type as original input) of shape (N, T, F).
        min_vals: Per-feature minimum values (shape: (F,))
        max_vals: Per-feature maximum values (shape: (F,))
    
    Returns:
        original_data: Data in the original scale.
    """
    if isinstance(normalized_data, torch.Tensor):
        min_vals_broadcast = min_vals.view(1, 1, -1).to(normalized_data.device)
        max_vals_broadcast = max_vals.view(1, 1, -1).to(normalized_data.device)
        range_vals = (max_vals_broadcast - min_vals_broadcast)
        normalized_data = torch.clamp(normalized_data,-1,1)
        original = (normalized_data + 1) / 2 * range_vals + min_vals_broadcast
        return original

    elif isinstance(normalized_data, np.ndarray):
        min_vals_broadcast = min_vals.reshape(1, 1, -1)
        max_vals_broadcast = max_vals.reshape(1, 1, -1)
        range_vals = (max_vals_broadcast - min_vals_broadcast)
        normalized_data = np.clip(normalized_data,-1,1)
        original = (normalized_data + 1) / 2 * range_vals + min_vals_broadcast
        return original

    else:
        raise TypeError("Normalized data must be a torch.Tensor or a numpy.ndarray.")
