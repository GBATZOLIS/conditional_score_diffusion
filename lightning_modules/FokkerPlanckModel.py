import losses
from losses import get_general_sde_loss_fn
from utils import compute_grad, compute_divergence
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
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel

@utils.register_lightning_module(name='fokker-planck')
class FokkerPlanckModel(BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

    def compute_fp_loss(self, batch):
        
        eps=1e-5
        # sample times
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
        # sample x_t form perturbation kernel
        x_t = self.sde.perturb(batch, t) # WARNING P_1 or P_t

        # get the diffusion coeff
        g = self.sde.sde_coefficients(torch.zeros_like(batch), t)[1]
        # get the drift function    
        if self.config.model.fp_mdoe == 'forward':
            f_x =  lambda y: self.sde.sde_coefficients(y, t)[0]
        elif self.config.model.fp_mdoe == 'reverse':
            f_x =  lambda y: self.sde.sde_coefficients(y, t)[0] - g[...,None]**2 * self.score_model(y, t)

        def fp_loss(f_x, g, x, t):
            '''
            Args:
            x - batch of x_t's (B, ...)
            t - batch of corresponding t's (B,)
            f_x - drift function. Takes y and returns f(y,t) (B,D) -> (B,D)
            g - diffucion coeffictent at t. (B,)

            Returns:
                difference - difference of RHS and LHS of corresponding Fokker-Planck (B,)
            '''

            B = x.shape[0] # batch size
            D = np.prod(x.shape[1:])
            score_norm_sq = torch.linalg.norm(self.score_model.score(x, t).view(B,-1), dim=1)**2 # (B,)
            score_x = lambda y: self.score_model.score(y,t)
            score_divergence = compute_divergence(score_x, x, hutchinson=self.config.training.hutchinson) # (B,)
            f_divergence = compute_divergence(f_x, x, hutchinson=self.config.training.hutchinson) # (B,)
            f = f_x(x) # (B,D)
            f_dot_s = torch.bmm(f.view(B, 1, D), self.score_model.score(x, t).view(B, D, 1)).squeeze() # (B,)
            log_energy_t = lambda s: self.score_model.log_energy(x, s) 
            time_derivative = compute_grad(log_energy_t, t).squeeze(1) # (B,)
            if self.config.model.fp_mdoe == 'reverse':
                difference = -time_derivative - (0.5*g**2*(score_divergence + score_norm_sq) - f_dot_s - f_divergence) #(B,)
            elif self.config.model.fp_mdoe == 'forward':
                difference = time_derivative - (0.5*g**2*(score_divergence + score_norm_sq) - f_dot_s - f_divergence) #(B,)
            return difference
            
        loss_fp = fp_loss(f_x, g, x_t, t).abs().mean() # should we apply the likelihood weighting? 
        
        return loss_fp

    def compute_ballance_loss(self, batch):
        eps=1e-5
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
        diffusion = self.sde.sde_coefficients(torch.zeros_like(batch), t)[1]
        perturbed_data = self.sde.perturb(batch, t)
        norm_fp_model = torch.linalg.norm(self.score_model.score(perturbed_data, t, weight_corerctor=0), dim=1)**2
        norm_correction_model =torch.linalg.norm(self.score_model.score(perturbed_data, t, weight_fp=0), dim=1)**2
        return (norm_correction_model/norm_fp_model).mean()

    def training_step(self, batch, batch_idx):
        loss_fp = self.compute_fp_loss(batch)
        self.log('train_fokker_planck_loss', loss_fp, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #ballance_loss = self.compute_ballance_loss(batch)
        #self.log('train_ballance_loss', ballance_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        loss_dsm = self.train_loss_fn(self.score_model, batch)
        self.log('train_denoising_loss', loss_dsm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        


        N = self.config.training.num_epochs
        t = self.current_epoch / N
        # For projection
        #t = ((self.current_epoch - (N-1)) / N)
        
        # constant weight
        if self.config.training.schedule == 'constant':
            weight = self.config.training.alpha 
        # geometric schedule
        elif self.config.training.schedule == 'geometric':
            weight = self.config.training.alpha_min * (self.config.training.alpha_max / self.config.training.alpha_min) ** t
        # linear schedule
        elif self.config.training.schedule == 'linear':
            weight = (1 - t) * self.config.training.alpha_min  + t * self.config.training.alpha_max

        # maxout
        #weight = max(weight, self.config.training.alpha_max)

        # loss
        loss = loss_dsm + weight * loss_fp #+ weight * ballance_loss
        self.log('train_full_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_loss_fn(self.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


