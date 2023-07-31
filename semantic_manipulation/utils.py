import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import memory_allocated
import numpy as np

def check_batch_fit(model, device, batch_size):
    try:
        model = model.to(device)
        dummy_input = torch.randn(batch_size, 3, 64, 64, device=device)
        model.encode(dummy_input)
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e):
            return False
        else:
            raise e

def find_max_batch(model, device):
    batch_size = 1024  # Start from a reasonably large number
    while batch_size > 0:
        if check_batch_fit(model, device, batch_size):
            return batch_size
        else:
            batch_size = batch_size // 2

    raise ValueError('Unable to find a batch size that fits the memory.')



def cartesian_to_spherical(cartesian):
    """
    cartesian: numpy array (N, D)
    """
    sqr = cartesian ** 2
    cum_r = np.sqrt(np.cumsum(sqr[:,::-1], axis=1))
    angles = np.arccos(cartesian[:,:-1] / cum_r[:,:0:-1])
    angles[cartesian[:, -1] < 0, -1] = 2 * np.pi - angles[cartesian[:, -1] < 0, -1]
    r = cum_r[:,-1][:,None]
    return np.concatenate([r, angles], axis=1)

def spherical_to_cartesian(spherical):
    """
    spherical: numpy array (N, D)
    """
    r = spherical[:, 0][:,None]
    angles = spherical[:, 1:]
    cos = np.cos(angles)
    cos = np.concatenate([cos, np.ones((len(cos), 1))], axis=1)
    sin = np.sin(angles)
    cum_sin = np.cumprod(sin, axis=1)
    cum_sin = np.concatenate([np.ones((len(cum_sin), 1)), cum_sin], axis=1)
    return r * cum_sin * cos 
