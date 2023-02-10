import losses
from losses import get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from sampling.unconditional import get_sampling_fn
from scipy.special import ive
from models import utils as mutils
from . import utils
import torch.optim as optim
import os
import torch
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel

@utils.register_lightning_module(name='ksphere_gt')
class KSphereGroundTruthModel(BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

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
        loss = torch.tensor(42., requires_grad=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = torch.tensor(42., requires_grad=True)
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

    
    

