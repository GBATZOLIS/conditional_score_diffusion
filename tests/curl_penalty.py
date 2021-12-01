#%%
import torch

#%% 
def compute_curl(vector_field):
    """
    Args:
    vector_field
    Returns:
    curl: curl of the vectorfield.
    """
    vx = 