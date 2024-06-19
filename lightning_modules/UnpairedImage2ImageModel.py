import torch
import torch.optim as optim
from torch.distributions import Uniform
import pytorch_lightning as pl
import sde_lib
from . import utils
from utils import get_named_beta_schedule
from models import utils as mutils
from scipy.interpolate import PchipInterpolator
import numpy as np
from losses import get_attribute_classifier_loss_fn, get_attribute_classifier_scoreVAE_loss_fn
import torch.nn.functional as F
from sampling.conditional import get_conditional_sampling_fn

@utils.register_lightning_module(name='UnpairedImage2Image')
class UnpairedImage2ImageModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.learning_rate = config.optim.lr
        self.save_hyperparameters()
        self.config = config

        #unconditional score model
        self.score_model_A = mutils.load_prior_model(config, domain='A')
        self.score_model_A.freeze()
        self.score_model_B = mutils.load_prior_model(config, domain='B')
        self.score_model_B.freeze()
        
        #encoders
        self.shared_encoder = mutils.create_encoder(config, type='shared')
        self.domain_A_encoder = mutils.create_encoder(config, type='domain_A')
        self.domain_B_encoder = mutils.create_encoder(config, type='domain_B')

        #MI estimators
        self.MI_score_model_domain_A = mutils.create_model(config, type='MI')
        self.MI_score_model_domain_B = mutils.create_model(config, type='MI')

    def get_conditional_score_fn(self, train=False, gamma=1):
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
    
    def configure_loss_fn(self, config, train):
        if config.training.loss_type == 'scoreVAE':
            loss_fn = get_attribute_classifier_scoreVAE_loss_fn(
                        self.sde, config.training.likelihood_weighting,
                        self.sampling_eps)
        elif config.training.loss_type == 'crossentropy':
            loss_fn = get_attribute_classifier_loss_fn(self.sde, train, self.sampling_eps)
        return loss_fn
    
    def training_step(self, batch, batch_idx):
        if self.config.training.loss_type == 'scoreVAE':
            cond_score_fn = self.get_conditional_score_fn(train=True)
            loss = self.train_loss_fn(self.attribute_encoder, cond_score_fn, batch, self.t_dist)
        elif self.config.training.loss_type == 'crossentropy':
            loss = self.train_loss_fn(self.attribute_encoder, batch, self.t_dist)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.config.training.loss_type == 'scoreVAE':
            cond_score_fn = self.get_conditional_score_fn(train=False)
            loss = self.eval_loss_fn(self.attribute_encoder, cond_score_fn, batch, self.t_dist)
        elif self.config.training.loss_type == 'crossentropy':
            loss = self.eval_loss_fn(self.attribute_encoder, batch, self.t_dist)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def sample(self, y):
        sampling_shape = [y.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps,
                                                              p_steps=128)
        score_fn = self.get_conditional_score_fn()
        return conditional_sampling_fn(self.unconditional_score_model, y, score_fn=score_fn)
        
        
    def configure_optimizers(self):
        class SchedulerLambdaFunction:
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

        # Instantiating scheduler_lambda_function with the warm-up period
        warm_up_function = SchedulerLambdaFunction(self.config.optim.warmup)

        # Optimizer for the attribute encoder
        optimizer = optim.Adam(
            self.attribute_encoder.parameters(), 
            lr=self.learning_rate, 
            betas=(self.config.optim.beta1, 0.999), 
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )

        # Scheduler configuration using the instantiated lambda function
        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_function),
            'interval': 'step',  # Apply scheduler after each step
        }

        return [optimizer], [scheduler]
