import losses
from losses import get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from sampling.unconditional import get_sampling_fn
from models import utils as mutils
from . import utils
import torch.optim as optim
import os
import torch
import copy

@utils.register_lightning_module(name='base')
class BaseSdeGenerativeModel(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        loss = self.train_loss_fn(self.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_loss_fn(self.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def sample(self, show_evolution=False, num_samples=None, ode=False):
        sampling_config = copy.deepcopy(self.config)
        if ode:
            sampling_config.sampling.method = 'ode'
        # Construct the sampling function
        if num_samples is None:
            sampling_shape = self.default_sampling_shape
        else:
            sampling_shape = [num_samples] +  sampling_config.data.shape
        sampling_fn = get_sampling_fn(sampling_config, self.sde, sampling_shape, self.sampling_eps)

        if ode:
            return sampling_fn(self.score_model)
        else:
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

    
    

