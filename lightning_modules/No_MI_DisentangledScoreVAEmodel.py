from . import BaseSdeGenerativeModel
from losses import get_simplified_scoreVAE_loss_fn
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

@utils.register_lightning_module(name='no_MI_disentangled_score_vae')
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
        score_loss_fn = get_simplified_scoreVAE_loss_fn(self.sde,
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        kl_weight=config.training.kl_weight)
        
        
        #Mutual Information Estimation Loss
        #trains a conditional+unconditional score model on the latent encodings. 
        #This model is used to estimate the mutual information.
        #MI_estimator_loss_fn = get_MI_estimator_loss_fn 

        return {0:score_loss_fn}
    
    def training_step(self, batch, batch_idx):
        cond_score_fn = self.get_conditional_score_fn(train=True)
        loss = self.train_loss_fn[0](cond_score_fn, self.encoder, batch, self.t_dist)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        cond_score_fn = self.get_conditional_score_fn(train=False)
        loss = self.eval_loss_fn[0](cond_score_fn, self.encoder, batch, self.t_dist)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
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
        return x_hat
    
    def flip_attributes(self, y, attributes=None, attribute_to_index_map=None):
        # Check if an attribute_to_index_map is provided, if not, fetch it from the data module
        if attribute_to_index_map is None:
            # Ensure self.trainer is available and has been set
            if self.trainer and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                data_module = self.trainer.datamodule
                attribute_to_index_map = data_module.get_attribute_to_index_map()
            else:
                # Handle the case where the data module or trainer is not available
                raise RuntimeError("Data module is not accessible. Ensure this method is called within the LightningModule lifecycle.")

        if attributes is None:
            # If no attributes specified, assume flipping all attributes
            indices_to_flip = list(range(y.shape[1]))
        else:
            # Find the column indices for the specified attributes using the provided or fetched map
            indices_to_flip = [attribute_to_index_map[attr] for attr in attributes]

        # Flip the specified attributes in y
        for index in indices_to_flip:
            y[:, index] = 1 - y[:, index]
        
        return y

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
        
        # Since we are only training the encoder, we define parameters accordingly
        ae_params = self.encoder.parameters()  # Encoder parameters
        
        # Set up the optimizer for the encoder
        ae_optimizer = optim.Adam(ae_params, lr=self.learning_rate, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                                  weight_decay=self.config.optim.weight_decay)
        
        # Set up the learning rate scheduler for the encoder's optimizer
        ae_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(ae_optimizer, scheduler_lambda_function(self.config.optim.warmup)),
                        'interval': 'step'}  # called after each training step
        
        # Return the optimizer and scheduler for the encoder only
        return [ae_optimizer], [ae_scheduler]
