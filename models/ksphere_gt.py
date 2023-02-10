import torch 
import numpy as np
from scipy.special import ive
from . import utils
import pytorch_lightning as pl

@utils.register_model(name='ksphere_gt')
class KSphereGT(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.dummy = torch.nn.Linear(1,1)
     
    def forward(self, batch, sigmas):
        batch = batch.detach().cpu().numpy()
        sigmas = sigmas.detach().cpu().numpy()
        
        def von_mises_grad_log(p,k):
            result = - ive(p/2, k) / ive(p/2-1, k)
            return result
        
        scores = []
        for x, sigma in zip(batch,sigmas):
            r = np.linalg.norm(x)
            scores.append((x / r) * (-von_mises_grad_log(self.config.data.manifold_dim, r/sigma**2) - r) / sigma **2)
        scores = np.stack(scores)
        return torch.from_numpy(scores).to(self.device)