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
from losses import get_attribute_corrector_loss_fn
from disentanglement_losses import modify_labels_for_classifier_free_guidance
import torch.nn.functional as F
from sampling.conditional import get_conditional_sampling_fn

@utils.register_lightning_module(name='attribute_conditional')
class AttributeConditionalModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.learning_rate = config.optim.lr
        self.save_hyperparameters()
        self.config = config

        #unconditional score model
        use_prior = config.training.get('use_prior', False)
        if use_prior:
            self.unconditional_score_model = mutils.load_prior_model(config)
            self.unconditional_score_model.freeze()
            
        self.attribute_correction_model = mutils.create_model(config)

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
    
    def get_attribute_correction(self, train=False):
        #derive the conditional score contribution of the attribute encoder
        attribute_correction = mutils.get_score_fn(self.sde, self.attribute_correction_model, conditional=True, train=train, continuous=True)
        return attribute_correction

    def get_noise_fn(self, train=False):
        use_prior = self.config.training.get('use_prior', False)
        classifier_free = self.config.training.get('classifier_free', False)
        
        assert use_prior==False, 'use_prior should be false for this experiment'
        assert classifier_free==True, 'classifier_free should be True for this experiment'
        
        raw_noise_fn = mutils.get_noise_fn(self.sde, self.attribute_correction_model, train)
        def noise_fn(x, y, t):
            return raw_noise_fn({'x':x, 'y':y}, t)

        return noise_fn

    def get_conditional_score_fn(self, train=False, gamma=1, sampling=False):
        use_prior = self.config.training.get('use_prior', False)
        classifier_free = self.config.training.get('classifier_free', False)

        if use_prior:
            unconditional_score_fn = mutils.get_score_fn(self.usde, self.unconditional_score_model, conditional=False, train=False, continuous=True)
            attribute_encoder_correction = self.get_attribute_correction(train=train)
            def conditional_score_fn(x, y, t):
                return unconditional_score_fn(x, t) + gamma*attribute_encoder_correction({'x':x, 'y':y}, t)
        else:
            if classifier_free:
                attribute_encoder_correction = self.get_attribute_correction(train=train)

                if not sampling:
                    def conditional_score_fn(x, y, t):
                        return attribute_encoder_correction({'x':x, 'y':y}, t)
                else:
                    def conditional_score_fn(x, y, t):
                        batchsize = x.size(0)
                        y_uncond = torch.ones_like(y) * -1
                        x_concat = torch.cat([x, x], dim=0)
                        y_concat = torch.cat([y_uncond, y], dim=0)
                        t_concat = torch.cat([t, t], dim=0)
                        out = attribute_encoder_correction({'x':x_concat, 'y':y_concat}, t_concat) #(2*batchsize, *x.shape[1:])
                        unconditional_score, conditional_score = out[:batchsize], out[batchsize:]
                        return (1-gamma) * unconditional_score + gamma * conditional_score
            else:
                attribute_encoder_correction = self.get_attribute_correction(train=train)
                def conditional_score_fn(x, y, t):
                    return attribute_encoder_correction({'x':x, 'y':y}, t) 
                
        return conditional_score_fn

    def configure_loss_fn(self, config, train):
        loss_fn = get_attribute_corrector_loss_fn(self.sde, config.training.likelihood_weighting, 
                                                  use_simple_loss=True)
        return loss_fn
    
    def training_step(self, batch, batch_idx):
        classifier_free = self.config.training.get('classifier_free', False)
        if classifier_free:
            x, y = batch
            class_free_y = modify_labels_for_classifier_free_guidance(y)
            batch = (x, class_free_y)

        #OLD
        #cond_score_fn = self.get_conditional_score_fn(train=True)
        #loss = self.train_loss_fn(cond_score_fn, batch, self.t_dist)

        #NEW
        noise_fn = self.get_noise_fn(train=True)
        loss = self.train_loss_fn(noise_fn, batch, self.t_dist)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        classifier_free = self.config.training.get('classifier_free', False)
        #model_output_fn = self.get_conditional_score_fn(train=False)
        model_output_fn = self.get_noise_fn(train=False)

        if classifier_free:
            conditional_loss = self.eval_loss_fn(model_output_fn, batch, self.t_dist)
            self.log('val_conditional_loss', conditional_loss)

            x, y = batch
            y_uncond = torch.ones_like(y) * -1
            unconditional_batch = (x, y_uncond)
            unconditional_loss = self.eval_loss_fn(model_output_fn, unconditional_batch, self.t_dist)
            self.log('val_unconditional_loss', unconditional_loss)

            loss = 1/2*(conditional_loss + unconditional_loss)
        else:
            loss = self.eval_loss_fn(model_output_fn, batch, self.t_dist)

        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def sample(self, y, gamma=1):
        y = y.float()
        sampling_shape = [y.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps,
                                                              p_steps=128)
        score_fn = self.get_conditional_score_fn(train=False, gamma=gamma, sampling=True)
        return conditional_sampling_fn(self.attribute_correction_model, y, score_fn=score_fn)
        
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
            self.attribute_correction_model.parameters(), 
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
