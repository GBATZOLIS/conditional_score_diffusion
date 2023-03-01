from . import BaseSdeGenerativeModel
from losses import get_scoreVAE_loss_fn, get_old_scoreVAE_loss_fn, get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from models import utils as mutils
from . import utils
import torch.optim as optim
import os
import torch
from sde_lib import cVPSDE, csubVPSDE, cVESDE
from sampling.conditional import get_conditional_sampling_fn
from sampling.unconditional import get_sampling_fn
import torchvision
import numpy as np
import losses
import torch

@utils.register_lightning_module(name='pretrained_score_vae')
class PretrainedScoreVAEmodel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the score model
        self.config = config

        #latent correction model
        config.model.input_channels = 2*config.model.output_channels
        self.latent_correction_model = mutils.create_model(config)
        
        #unconditional score model
        config.model.input_channels = config.model.output_channels
        config.model.name = config.model.unconditional_score_model_name
        self.unconditional_score_model = mutils.create_model(config)

        #encoder
        self.encoder = mutils.create_encoder(config)

    def configure_sde(self, config):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.cVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.usde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.csubVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.usde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':            
            self.sde = sde_lib.cVESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales)
            self.usde = sde_lib.VESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales)
            self.sampling_eps = 1e-5           
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde.sampling_eps = self.sampling_eps
        self.usde.sampling_eps = self.sampling_eps
    
    def configure_loss_fn(self, config, train):
        if hasattr(config.training, 'cde_loss'):
            if config.training.cde_loss:
                loss_fn = get_old_scoreVAE_loss_fn(self.sde, train, 
                                        variational=config.training.variational, 
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        eps=self.sampling_eps,
                                        use_pretrained=True)
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
        
        unconditional_loss_fn = get_general_sde_loss_fn(self.usde, train, conditional=False, reduce_mean=True,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)

        return {0:unconditional_loss_fn, 1:loss_fn}
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss = self.train_loss_fn[0](self.unconditional_score_model, batch)
            self.logger.experiment.add_scalars('train_loss', {'unconditional': loss}, self.global_step)

        elif optimizer_idx == 1:
            loss = self.train_loss_fn[1](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.logger.experiment.add_scalars('train_loss', {'conditional': loss}, self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_loss_fn[0](self.unconditional_score_model, batch)
        self.logger.experiment.add_scalars('val_loss', {'unconditional': loss}, self.global_step)
        
        loss = self.eval_loss_fn[1](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
        self.logger.experiment.add_scalars('val_loss', {'conditional': loss}, self.global_step)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx == 0 and (self.current_epoch+1) % self.config.training.visualisation_freq == 5:
            reconstruction = self.encoder_n_decode(batch)

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

            sample, _ = self.unconditional_sample()
            sample = sample.cpu()
            grid_batch = torchvision.utils.make_grid(sample, nrow=int(np.sqrt(sample.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('unconditional_sample', grid_batch, self.current_epoch)
            

        return loss

    def unconditional_sample(self, num_samples=None):
        if num_samples is None:
            sampling_shape = [self.config.training.batch_size] +  self.config.data.shape
        else:
            sampling_shape = [num_samples] +  self.config.data.shape
        sampling_fn = get_sampling_fn(self.config, self.usde, sampling_shape, self.sampling_eps)
        return sampling_fn(self.unconditional_score_model)

    def encoder_n_decode(self, x, show_evolution=False, predictor='default', corrector='default', p_steps='default', \
                     c_steps='default', snr='default', denoise='default', use_pretrained=True):

        if self.config.training.variational:
            mean_y, log_var_y = self.encoder(x)
            y = mean_y + torch.sqrt(log_var_y.exp()) * torch.randn_like(mean_y)
        else:
            y = self.encoder(x)

        sampling_shape = [x.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps, 
                                                              predictor=predictor, corrector=corrector, 
                                                              p_steps=p_steps, c_steps=c_steps, snr=snr, 
                                                              denoise=denoise, use_path=False, use_pretrained=use_pretrained)
        
        model = {'unconditional_score_model':self.unconditional_score_model,
                'latent_correction_model': self.latent_correction_model}

        return conditional_sampling_fn(model, y, show_evolution)
    
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
        

        unconditional_score_optimizer = optim.Adam(self.unconditional_score_model.parameters(), 
                                        lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                                        weight_decay=self.config.optim.weight_decay)
        
        unconditional_score_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(unconditional_score_optimizer, scheduler_lambda_function(self.config.optim.warmup)),
                                         'interval': 'step'}  # called after each training step

        ae_params = list(self.encoder.parameters())+list(self.latent_correction_model.parameters())
        ae_optimizer = optim.Adam(ae_params, lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                           weight_decay=self.config.optim.weight_decay)
        ae_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(ae_optimizer, scheduler_lambda_function(self.config.optim.slowing_factor*self.config.optim.warmup)),
                        'interval': 'step'}  # called after each training step

        
                    
        return [unconditional_score_optimizer, ae_optimizer], [unconditional_score_scheduler, ae_scheduler]