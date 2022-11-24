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

@utils.register_lightning_module(name='fokker-planck')
class FokkerPlanckModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the score model
        self.config = config
        self.score_model = mutils.create_model(config)
        self.configure_default_sampling_shape(config)
        # Placeholder to store samples
        self.samples = None

    def configure_sde(self, config):
        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            if config.data.use_data_mean:
                data_mean_path = os.path.join(config.data.base_dir, 'datasets_mean', '%s_%d' % (config.data.dataset, config.data.image_size), 'mean.pt')
                data_mean = torch.load(data_mean_path)
            else:
                data_mean = None
            self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=data_mean)
            self.sampling_eps = 1e-5
        elif config.training.sde.lower() == 'snrsde':
            self.sde = sde_lib.SNRSDE(N=config.model.num_scales)
            self.sampling_eps = 1e-3
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    def configure_loss_fn(self, config, train):
        if config.training.continuous:
            loss_fn = get_general_sde_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)
        return loss_fn

    def configure_default_sampling_shape(self, config):
        #Sampling settings
        self.data_shape = config.data.shape
        self.default_sampling_shape = [config.training.batch_size] +  self.data_shape


    def compute_fp_loss(self, batch):
        eps=1e-5
        #x = batch
        #n_chunks = self.config.training.n_chunks
        loss_fp = 0
        # t=[]
        # x_chunked = torch.chunk(x, n_chunks, dim=0)
        # for x_chunk in x_chunked:
        #     chunk_size = x_chunk.shape[0]
        #     t_chunk = torch.rand(1).repeat(chunk_size)
        #     t.append(t_chunk)
        # t = torch.cat(t, dim=0).to(self.device)

        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps

        diffusion = self.sde.sde(torch.zeros_like(batch), t)[1]
        # oneish = (1.0- eps) * torch.ones_like(t).to(self.device)
        perturbed_data = self.sde.perturb(batch, t) # WARNING P_1 or P_t

        def fp_loss(x,t):
            B = x.shape[0] # batch size
            grad_norm_2 = torch.linalg.norm(self.score_model.score(x, t).view(B,-1), dim=1)**2

            score_x = lambda y: self.score_model.score(y,t)
            divergence = compute_divergence(score_x, x, hutchinson=self.config.training.hutchinson)    
            #self.score_model.trace_hessian_log_energy(x, t) 
            
            log_energy_t = lambda s: self.score_model.log_energy(x, s) 
            time_derivative = compute_grad(log_energy_t, t).squeeze(1)
            #self.score_model.time_derivative_log_energy(x,t)

            difference = (time_derivative - (diffusion**2 / 2) * (grad_norm_2 + divergence))
            difference = diffusion**2 * difference # apply weighting
            return difference

        #difference = fp_loss(perturbed_data, t)
        
        #x_grad_fp_loss = compute_grad(lambda x: fp_loss(x,t), perturbed_data)
        #loss_fp = (torch.linalg.norm(x_grad_fp_loss, dim=1)).mean()
        
        loss_fp = fp_loss(perturbed_data, t).abs().mean()
        
        #diff_chunked = torch.chunk(difference, n_chunks, dim=0)
        # for chunk in diff_chunked:
        #     chunk = chunk.view(1,chunk.shape[0],1)
        #     #loss_fp += torch.cdist(chunk, chunk).mean() / n_chunks
        #     loss_fp += torch.std(chunk) / n_chunks
        
        return loss_fp

    def compute_ballance_loss(self, batch):
        eps=1e-5
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
        diffusion = self.sde.sde(torch.zeros_like(batch), t)[1]
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
    
    def sample(self, show_evolution=False, num_samples=None):
        # Construct the sampling function
        if num_samples is None:
            sampling_shape = self.default_sampling_shape
        else:
            sampling_shape = [num_samples] +  self.config.data.shape
        sampling_fn = get_sampling_fn(self.config, self.sde, sampling_shape, self.sampling_eps)

        return sampling_fn(self.score_model, show_evolution=show_evolution)

    def configure_optimizers(self):
        class scheduler_lambda_function:
            def __init__(self, warm_up):
                self.use_warm_up = True if warm_up > 0 else False
                self.warm_up = warm_up

            def __call__(self, s):
                if self.use_warm_up:
                    if s < self.warm_up:
                        return s / self.warm_up
                    else:
                        return 1
                else:
                    return 1
        

        optimizer = losses.get_optimizer(self.config, self.score_model.parameters())
        
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer,scheduler_lambda_function(self.config.optim.warmup)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]

    
    

