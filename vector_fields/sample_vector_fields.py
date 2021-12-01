import torch

def constant_curl_vf(x,t):
    """
    x: tensor (B,2)
    t: tensor (B,1)
    """
    return torch.stack([x[:,1], -x[:,0]],axis=1)