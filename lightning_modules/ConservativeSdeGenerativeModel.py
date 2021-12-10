import losses
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn, get_general_sde_loss_fn
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
from utils import compute_grad
import pytorch_lightning as pl
import sde_lib
from sampling.unconditional import get_sampling_fn
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from . import utils
import torch.optim as optim
import os
import torch
import numpy as np


@utils.register_lightning_module(name='curl_penalty')
class ConservativeSdeGenerativeModel(BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.LAMBDA = config.training.LAMBDA


    def training_step(self, batch, batch_idx):
        curl_penalty = self.curl_penalty(batch)
        loss = self.train_loss_fn(self.score_model, batch) + 0.01*(self.current_epoch * 1e-3)**2 *curl_penalty
        self.log('curl_penalty', curl_penalty, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def curl_penalty(self, batch, eps=1e-5):
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
        perturbed_data.requires_grad = True
        #i = torch.randint(0, perturbed_data.shape[1],(1,)).item()
        #j=i
        #while j==i:
        #    j = torch.randint(0, perturbed_data.shape[1],(1,)).item()
        #dvy_dx = compute_grad(lambda x,t: score(x,t)[:,i],perturbed_data,t)[:,j]
        #dvx_dy = compute_grad(lambda x,t: score(x,t)[:,j],perturbed_data,t)[:,i]

        dvy_dx = compute_grad(self.vy, perturbed_data, t)[:,0]
        dvx_dy = compute_grad(self.vx, perturbed_data, t)[:,1]
        return torch.mean((dvy_dx - dvx_dy)**2)

    def vx(self, x,t):
        return self.score_model(x,t)[:,0]
    def vy(self, x,t):
        return self.score_model(x,t)[:,1]