from . import BaseSdeGenerativeModel
from losses import get_scoreVAE_loss_fn, get_old_scoreVAE_loss_fn, get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from models import utils as mutils
from . import utils
from utils import get_named_beta_schedule
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
import lpips
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import numpy as np
from torch.distributions import Uniform

@utils.register_lightning_module(name='disentangled_score_vae')
class DisentangledScoreVAEmodel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.learning_rate = config.optim.lr
        self.save_hyperparameters()

        # Initialize the score model
        self.config = config
        
        #unconditional score model
        self.score_model = mutils.load_prior_model(config) #can be either conditional or unconditional
        self.score_model.freeze()

        #load attribute encoder
        #Create a ligtning module similar to the Base Lightning module that trains the attribute encoder.
        #self.attribute_encoder = mutils.load_attribute_encoder(config)
        #self.attribute_encoder.freeze()

        #encoder
        self.encoder = mutils.create_encoder(config) #it encodes image latent characteristics independent from the encoded attributes

        #instantiate the model that approximates 
        #the mutual information between the attibutes encoding and the complementary encoding
        #self.MI_diffusion_model = mutils.create_model(config)

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
        elif config.training.sde.lower() == 'snrsde':
            self.sampling_eps = 1e-3

            if hasattr(config.training, 'beta_schedule'):
                #DISCRETE QUANTITIES
                N = config.model.num_scales
                betas = get_named_beta_schedule('linear', N)
                alphas = 1.0 - betas
                alphas_cumprod = np.cumprod(alphas, axis=0)
                discrete_snrs = alphas_cumprod/(1.0 - alphas_cumprod)

                #Monotonic Bicubic Interpolation
                snr = PchipInterpolator(np.linspace(self.sampling_eps, 1, len(discrete_snrs)), discrete_snrs)
                d_snr = snr.derivative(nu=1)

                def logsnr(t):
                    device = t.device
                    snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                    return torch.log(snr_val)

                def d_logsnr(t):
                    device = t.device
                    dsnr_val = torch.from_numpy(d_snr(t.cpu().numpy())).float().to(device)
                    snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                    return dsnr_val/snr_val

                self.sde = sde_lib.cSNRSDE(N=N, gamma=logsnr, dgamma=d_logsnr)
                self.usde = sde_lib.SNRSDE(N=N, gamma=logsnr, dgamma=d_logsnr)
            else:
                self.sde = sde_lib.cSNRSDE(N=config.model.num_scales)
                self.usde = sde_lib.SNRSDE(N=config.model.num_scales)

        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde.sampling_eps = self.sampling_eps
        self.usde.sampling_eps = self.sampling_eps

        self.t_dist = Uniform(self.sampling_eps, 1)
    
    def _handle_batch(self, batch):
        if type(batch) == list:
            x = batch[0]
        else:
            x = batch
        return x

    def get_conditional_score_fn(self, train=False):
        def get_latent_correction_fn(encoder):
            def get_log_density_fn(encoder):
                def log_density_fn(x, z, t):
                    latent_distribution_parameters = encoder(x, t)
                    channels = latent_distribution_parameters.size(1) // 2
                    mean_z = latent_distribution_parameters[:, :channels]
                    log_var_z = latent_distribution_parameters[:, channels:]

                    # Flatten mean_z and log_var_z for consistent shape handling
                    mean_z_flat = mean_z.view(mean_z.size(0), -1)
                    log_var_z_flat = log_var_z.view(log_var_z.size(0), -1)
                    z_flat = z.view(z.size(0), -1)

                    logdensity = -0.5 * torch.sum(torch.square(z_flat - mean_z_flat) / log_var_z_flat.exp(), dim=1)
                    return logdensity

                return log_density_fn

            def latent_correction_fn(x, z, t):
                  if not train: 
                    torch.set_grad_enabled(True)

                  log_density_fn = get_log_density_fn(encoder)
                  device = x.device
                  x.requires_grad=True
                  ftx = log_density_fn(x, z, t)
                  grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                      grad_outputs=torch.ones(ftx.size()).to(device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                  assert grad_log_density.size() == x.size()

                  if not train:
                    torch.set_grad_enabled(False)

                  return grad_log_density

            return latent_correction_fn

        def conditional_score_fn(x, cond, t):
            y, z = cond
            latent_correction_fn = get_latent_correction_fn(self.encoder)
            pretrained_score_fn = mutils.get_score_fn(self.sde, self.score_model, conditional=True, 
                                                       train=train, continuous=True)
            conditional_score = pretrained_score_fn({'x':x, 'y':y}, t) + latent_correction_fn(x, z, t)
            return conditional_score

        return conditional_score_fn


    def configure_loss_fn(self, config, train):
        score_loss_fn = get_scoreVAE_loss_fn(self.sde, train, 
                                        variational=config.training.variational, 
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        eps=self.sampling_eps,
                                        t_batch_size=config.training.t_batch_size,
                                        kl_weight=config.training.kl_weight,
                                        use_pretrained=config.training.use_pretrained,
                                        encoder_only = config.training.encoder_only,
                                        t_dependent = config.training.t_dependent,
                                        attibute_encoder = True)
        
        #trains a conditional+unconditional score model on the latent encodings. 
        #This model is used to estimate the mutual information.
        #MI_estimator_loss_fn = get_MI_estimator_loss_fn 

        return {0:score_loss_fn}
    
    def training_step(self, *args, **kwargs):
        batch, batch_idx = args[0], args[1]
        if hasattr(self.config, 'debug') and self.config.debug.skip_training and batch_idx > 1:
            return None
        batch = self._handle_batch(batch)

        if self.config.training.use_pretrained:
            if self.config.training.encoder_only:
                loss = self.train_loss_fn[0](self.encoder, self.unconditional_score_model, batch, self.t_dist)
            else:
                loss = self.train_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self._handle_batch(batch)
        if hasattr(self.config, 'debug') and self.config.debug.skip_validation and batch_idx > 1:
            return None
        if self.config.training.use_pretrained:
            if self.config.training.encoder_only:
                loss = self.eval_loss_fn[0](self.encoder, self.unconditional_score_model, batch, self.t_dist)
            else:
                loss = self.eval_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = self.eval_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.logger.experiment.add_scalars('val_loss', {'conditional': loss}, self.global_step)
            self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            loss = self.eval_loss_fn[1](self.unconditional_score_model, batch)
            self.logger.experiment.add_scalars('val_loss', {'unconditional': loss}, self.global_step)

        if batch_idx == 2 and self.current_epoch+1 == 2:
            sample, _ = self.unconditional_sample(p_steps=250)
            sample = sample.cpu()
            grid_batch = torchvision.utils.make_grid(sample, nrow=int(np.sqrt(sample.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('unconditional_sample', grid_batch, self.current_epoch)

        # visualize reconstruction
        if batch_idx == 2 and (self.current_epoch+1) % self.config.training.visualisation_freq == 0:
            if torch.all(self.val_batch == 0) or batch.size(0) != self.val_batch.size(0):
                self.val_batch = batch
            else:
                batch = self.val_batch
                
            reconstruction = self.encode_n_decode(batch, p_steps=250,
                                                         use_pretrained=self.config.training.use_pretrained,
                                                         encoder_only=self.config.training.encoder_only,
                                                         t_dependent=self.config.training.t_dependent)

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

    def unconditional_sample(self, num_samples=None, p_steps='default'):
        if num_samples is None:
            sampling_shape = [self.config.training.batch_size] +  self.config.data.shape
        else:
            sampling_shape = [num_samples] +  self.config.data.shape
        sampling_fn = get_sampling_fn(self.config, self.usde, sampling_shape, self.sampling_eps, p_steps)
        return sampling_fn(self.unconditional_score_model)
    
    def encode(self, x, y, use_latent_mean=False, encode_x_T=False):
        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = self.encoder(x, t0)
        latent_dim = latent_distribution_parameters.size(1)//2
        mean_z = latent_distribution_parameters[:, :latent_dim]
        log_var_z = latent_distribution_parameters[:, latent_dim:]
        if use_latent_mean:
            z = mean_z
        else:
            z = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)
        
        #--new code--
        if encode_x_T:
            conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=x.size(), eps=self.sampling_eps, 
                                                              p_steps=128,
                                                              predictor='conditional_heun',
                                                              direction='forward', 
                                                              x_boundary=x)

            score_fn = self.get_conditional_score_fn(train=False)
            cond = [y, z]
            x_T = conditional_sampling_fn(self.score_model, cond, score_fn=score_fn)
            return z, x_T
        else:
            return z, None

    def decode(self, y, z, x_T=None):
        sampling_shape = [z.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps,
                                                              p_steps=128,
                                                              predictor='conditional_heun',
                                                              x_boundary=x_T)
        cond = [y, z]
        score_fn = self.get_conditional_score_fn(train=False)
        return conditional_sampling_fn(self.score_model, cond, score_fn=score_fn)

    def encode_n_decode(self, x, y, encode_x_T=False):
        z, x_T = self.encode(x, y, encode_x_T=encode_x_T)
        x_hat = self.decode(y, z, x_T)
    
    def flip_attributes(self, y, attributes=None):
        pass

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
        
        if self.config.training.use_pretrained:
            if self.config.training.encoder_only:
                ae_params = self.encoder.parameters()
            else:
                ae_params = list(self.encoder.parameters())+list(self.latent_correction_model.parameters())
            
            ae_optimizer = optim.Adam(ae_params, lr=self.learning_rate, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                            weight_decay=self.config.optim.weight_decay)
            ae_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(ae_optimizer, scheduler_lambda_function(self.config.optim.warmup)),
                            'interval': 'step'}  # called after each training step
            return [ae_optimizer], [ae_scheduler]
        else:
            ae_params = list(self.encoder.parameters())+list(self.latent_correction_model.parameters())
            ae_optimizer = optim.Adam(ae_params, lr=self.learning_rate, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                            weight_decay=self.config.optim.weight_decay)
            ae_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(ae_optimizer, scheduler_lambda_function(self.config.optim.slowing_factor*self.config.optim.warmup)),
                            'interval': 'step'}  # called after each training step
                            
            unconditional_score_optimizer = optim.Adam(self.unconditional_score_model.parameters(), 
                                            lr=self.learning_rate, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                                            weight_decay=self.config.optim.weight_decay)
            unconditional_score_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(unconditional_score_optimizer, scheduler_lambda_function(self.config.optim.warmup)),
                                            'interval': 'step'}  # called after each training step
            
            return [ae_optimizer, unconditional_score_optimizer], [ae_scheduler, unconditional_score_scheduler]