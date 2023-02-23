from . import BaseSdeGenerativeModel
from losses import get_scoreVAE_loss_fn, get_old_scoreVAE_loss_fn
import pytorch_lightning as pl
import sde_lib
from models import utils as mutils
from . import utils
import torch.optim as optim
import os
import torch
from sde_lib import cVPSDE, csubVPSDE, cVESDE
from sampling.conditional import get_conditional_sampling_fn
import torchvision
import numpy as np
import losses

@utils.register_lightning_module(name='score_vae')
class ScoreVAEmodel(BaseSdeGenerativeModel.BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.variational = config.training.variational
        self.encoder = mutils.create_encoder(config)

    def configure_sde(self, config):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.cVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.csubVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':            
            self.sde = sde_lib.cVESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales, data_mean=data_mean)
            self.sampling_eps = 1e-5           
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde.sampling_eps = self.sampling_eps
    
    def configure_loss_fn(self, config, train):
        if hasattr(config.training, 'cde_loss'):
            if config.training.cde_loss:
                loss_fn = get_old_scoreVAE_loss_fn(self.sde, train, 
                                        variational=config.training.variational, 
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        eps=self.sampling_eps)
            else:
                loss_fn = get_scoreVAE_loss_fn(self.sde, train, 
                                            variational=config.training.variational, 
                                            likelihood_weighting=config.training.likelihood_weighting,
                                            eps=self.sampling_eps,
                                            t_batch_size=config.training.t_batch_size,
                                            kl_weight=config.training.kl_weight)
        else:
            loss_fn = get_scoreVAE_loss_fn(self.sde, train, 
                                        variational=config.training.variational, 
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        eps=self.sampling_eps,
                                        t_batch_size=config.training.t_batch_size,
                                        kl_weight=config.training.kl_weight)
        return loss_fn
    
    def training_step(self, batch, batch_idx):
        loss = self.train_loss_fn(self.encoder, self.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_loss_fn(self.encoder, self.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0 and self.current_epoch % self.config.training.visualisation_freq == 1:
            reconstruction = self.sample(batch)

            reconstruction =  reconstruction.cpu()
            grid_reconstruction = torchvision.utils.make_grid(reconstruction, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('reconstruction', grid_reconstruction, self.current_epoch)
            
            batch = batch.cpu()
            grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('real', grid_batch)

            difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(batch, start_dim=1)
            L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
            avg_L2norm = torch.mean(L2norm)

            self.log('reconstruction_loss', avg_L2norm, logger=True)

        return loss

    def sample(self, x, show_evolution=False, predictor='default', corrector='default', p_steps='default', c_steps='default', snr='default', denoise='default'):
        if self.variational:
            mean_y, log_var_y = self.encoder(x)
            y = mean_y + torch.sqrt(log_var_y.exp()) * torch.randn_like(mean_y)
        else:
            y = self.encoder(x)

        sampling_shape = [x.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps, 
                                                              predictor=predictor, corrector=corrector, 
                                                              p_steps=p_steps, c_steps=c_steps, snr=snr, 
                                                              denoise=denoise, use_path=False)

        return conditional_sampling_fn(self.score_model, y, show_evolution)
    
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
        

        optimizer = losses.get_optimizer(self.config, self.parameters())
        
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer,scheduler_lambda_function(self.config.optim.warmup)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]