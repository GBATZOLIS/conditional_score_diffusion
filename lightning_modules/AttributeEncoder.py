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
from losses import get_attribute_classifier_loss_fn
import torch.nn.functional as F
from sampling.conditional import get_conditional_sampling_fn

@utils.register_lightning_module(name='attribute_encoder')
class AttributeEncodermodel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.learning_rate = config.optim.lr
        self.save_hyperparameters()
        self.config = config

        #unconditional score model
        self.unconditional_score_model = mutils.load_prior_model(config)
        self.unconditional_score_model.freeze()
        
        self.attribute_encoder = mutils.create_encoder(config)

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
    
    def get_attribute_encoder_correction(self,):
        #derive the conditional score contribution of the attribute encoder
        def attribute_encoder_correction(x, y, t):
            # Ensure `y` is provided with shape (batchsize, num_features) where each element is an integer
            # representing the class index for the corresponding feature.
            torch.set_grad_enabled(True)
            x_in = x.detach().requires_grad_(True)

            # Obtain logits from the encoder; shape: (batchsize, num_features, num_classes)
            logits = self.attribute_encoder(x_in, t)

            # Calculate log probabilities across the num_classes dimension
            log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batchsize, num_features, num_classes)

            # Use `y` to select the log probability of the correct class for each feature.
            # This requires gathering the relevant log_probs using `y` as the index.
            # First, prepare `y` indices for gathering. It should have the same dimension as `log_probs`.
            y_indices = y.unsqueeze(-1)  # Shape: (batchsize, num_features, 1)
            selected_log_probs = log_probs.gather(dim=-1, index=y_indices)  # Shape: (batchsize, num_features, 1)

            # Since we're only interested in the selected log_probs, squeeze the last dimension
            selected_log_probs = selected_log_probs.squeeze(-1)  # Shape: (batchsize, num_features)

            log_probs = torch.sum(selected_log_probs, dim=1)

            # Compute gradients with respect to inputs
            conditional_score = torch.autograd.grad(outputs=log_probs, inputs=x_in,
                                      grad_outputs=torch.ones(log_probs.size()).to(x.device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            torch.set_grad_enabled(False)

            return conditional_score

        return attribute_encoder_correction

    def get_conditional_score_fn(self,):
        unconditional_score_fn = mutils.get_score_fn(self.sde, self.unconditional_score_model, 
                                                    conditional=False, train=False, continuous=True)
        attribute_encoder_correction = self.get_attribute_encoder_correction()

        def conditional_score_fn(x, y, t):
            return unconditional_score_fn(x, t) + attribute_encoder_correction(x, y, t)

        return conditional_score_fn

    def configure_loss_fn(self, config, train):
        loss_fn = get_attribute_classifier_loss_fn(self.sde, train, self.sampling_eps)
        return loss_fn
    
    def training_step(self, batch, batch_idx):
        loss = self.train_loss_fn(self.attribute_encoder, batch, self.t_dist)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
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
