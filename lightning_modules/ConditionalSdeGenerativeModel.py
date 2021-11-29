from . import BaseSdeGenerativeModel
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn, get_inverse_problem_smld_loss_fn, get_inverse_problem_ddpm_loss_fn, get_general_sde_loss_fn
from sde_lib import VESDE, VPSDE, cVESDE
from sampling.conditional import get_conditional_sampling_fn
import sde_lib
from . import utils
import torch
from iunets.layers import InvertibleDownsampling2D
import torch.nn as nn
import os

@utils.register_lightning_module(name='conditional')
class ConditionalSdeGenerativeModel(BaseSdeGenerativeModel.BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

    def configure_sde(self, config):
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

            sde_x = sde_lib.cVESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales, data_mean=data_mean)
            self.sampling_eps = 1e-5

            if config.training.conditioning_approach == 'sr3':
                self.sde = sde_x
            else:
                sde_y = sde_lib.VESDE(sigma_min=config.model.sigma_min_y, sigma_max=config.model.sigma_max_y, N=config.model.num_scales)
                self.sde = {'x':sde_x, 'y':sde_y}
            
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    def configure_loss_fn(self, config, train):
        if config.training.continuous:
            loss_fn = get_general_sde_loss_fn(self.sde, train, conditional=True, reduce_mean=config.training.reduce_mean,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)
        else:
            #assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, dict):
                #this part of the code needs to be improved. We should check all elements of the sde list. We might have a mixture.
                if isinstance(self.sde['y'], VESDE) and isinstance(self.sde['x'], cVESDE) and len(self.sde.keys())==2:
                    loss_fn = get_inverse_problem_smld_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean, \
                        likelihood_weighting=config.training.likelihood_weighting)
                elif isinstance(self.sde['y'], VPSDE) and isinstance(self.sde['x'], VPSDE) and len(self.sde.keys())==2:
                    loss_fn = get_inverse_problem_ddpm_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
                else:
                    raise NotImplementedError('This combination of sdes is not supported for discrete training yet.')
                
            elif isinstance(self.sde, VESDE):
                loss_fn = get_smld_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
            elif isinstance(self.sde, VPSDE):
                loss_fn = get_ddpm_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")
        
        return loss_fn
    
    def test_step(self, batch, batch_idx):
        print('Test batch %d' % batch_idx)
        
    def sample(self, y, show_evolution=False, predictor='default', corrector='default', p_steps='default', c_steps='default', snr='default', denoise='default', use_path='default'):
        sampling_shape = [y.size(0)]+self.config.data.shape_x
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps, 
                                                              predictor=predictor, corrector=corrector, 
                                                              p_steps=p_steps, c_steps=c_steps, snr=snr, 
                                                              denoise=denoise, use_path=use_path)

        return conditional_sampling_fn(self.score_model, y, show_evolution)

@utils.register_lightning_module(name='deprecated_conditional_decreasing_variance')
class DecreasingVarianceConditionalSdeGenerativeModel(ConditionalSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.register_buffer('sigma_max_y', torch.tensor(config.model.sigma_max_x).float())

    def configure_sde(self, config, sigma_max_y = None):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            if sigma_max_y is None:
                sigma_max_y = torch.tensor(config.model.sigma_max_y).float()
            else:
                sigma_max_y = torch.tensor(sigma_max_y).float()
            
            self.sigma_max_y = sigma_max_y
            sde_y = sde_lib.VESDE(sigma_min=config.model.sigma_min_y, sigma_max=sigma_max_y.cpu(), N=config.model.num_scales)
            
            if config.data.use_data_mean:
                data_mean_path = os.path.join(config.data.base_dir, 'datasets_mean', '%s_%d' % (config.data.dataset, config.data.image_size), 'mean.pt')
                data_mean = torch.load(data_mean_path)
            else:
                data_mean = None
            sde_x = sde_lib.cVESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales, data_mean=data_mean)
            
            self.sde = {'x':sde_x, 'y':sde_y}
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    def reconfigure_conditioning_sde(self, config, sigma_max_y = None):
        if config.training.sde.lower() == 'vesde':
            if sigma_max_y is None:
                sigma_max_y = torch.tensor(config.model.sigma_max_y).float()
            else:
                sigma_max_y = torch.tensor(sigma_max_y).float()
            
            self.sigma_max_y = sigma_max_y
            self.sde['y'] = sde_lib.VESDE(sigma_min=config.model.sigma_min_y, sigma_max=sigma_max_y.cpu(), N=config.model.num_scales)
        else:
            raise NotImplementedError(f"Conditioning SDE {config.training.sde} not supported yet.")
    
    def test_step(self, batch, batch_idx):
        print('Test batch %d' % batch_idx)

@utils.register_lightning_module(name='conditional_decreasing_variance')
class DecreasingVarianceConditionalSdeGenerativeModel(ConditionalSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.register_buffer('sigma_max_y', torch.tensor(config.model.sigma_max_y).float())
        self.register_buffer('sigma_min_y', torch.tensor(config.model.sigma_min_y).float())

    def configure_sde(self, config, sigma_min_y = None, sigma_max_y = None):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            if sigma_max_y is None:
                sigma_max_y = torch.tensor(config.model.sigma_max_y).float()
            else:
                sigma_max_y = torch.tensor(sigma_max_y).float()
            
            if sigma_min_y is None:
                sigma_min_y = torch.tensor(config.model.sigma_min_y).float()
            else:
                sigma_min_y = torch.tensor(sigma_min_y).float()
            
            self.sigma_max_y = sigma_max_y
            self.sigma_min_y = sigma_min_y

            sde_y = sde_lib.VESDE(sigma_min=sigma_min_y.cpu(), sigma_max=sigma_max_y.cpu(), N=config.model.num_scales)
            
            if config.data.use_data_mean:
                data_mean_path = os.path.join(config.data.base_dir, 'datasets_mean', '%s_%d' % (config.data.dataset, config.data.image_size), 'mean.pt')
                data_mean = torch.load(data_mean_path)
            else:
                data_mean = None
                
            sde_x = sde_lib.cVESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales, data_mean=data_mean)
            
            self.sde = {'x':sde_x, 'y':sde_y}
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    def reconfigure_conditioning_sde(self, config, sigma_min_y=None, sigma_max_y = None):
        if config.training.sde.lower() == 'vesde':
            if sigma_max_y is None:
                sigma_max_y = torch.tensor(config.model.sigma_max_y).float()
            else:
                sigma_max_y = torch.tensor(sigma_max_y).float()
            
            if sigma_min_y is None:
                sigma_min_y = torch.tensor(config.model.sigma_min_y).float()
            else:
                sigma_min_y = torch.tensor(sigma_min_y).float()
            
            self.sigma_max_y = sigma_max_y
            self.sigma_min_y = sigma_min_y
            
            self.sde['y'] = sde_lib.VESDE(sigma_min=sigma_min_y.cpu(), sigma_max=sigma_max_y.cpu(), N=config.model.num_scales)
        else:
            raise NotImplementedError(f"Conditioning SDE {config.training.sde} not supported yet.")
    
    def test_step(self, batch, batch_idx):
        print('Test batch %d' % batch_idx)

@utils.register_lightning_module(name='haar_conditional_decreasing_variance')
class HaarDecreasingVarianceConditionalSdeGenerativeModel(DecreasingVarianceConditionalSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)
    
    def haar_forward(self, x):
        x = self.haar_transform(x)
        x = permute_channels(x)
        return x
    
    def haar_backward(self, x):
        x = permute_channels(x, forward=False)
        x = self.haar_transform.inverse(x)
        return x
    
    def get_dc_coefficients(self, x):
        return self.haar_forward(x)[:,:3,::]
    
    def get_hf_coefficients(self, x):
        return self.haar_forward(x)[:,3:,::]

def permute_channels(haar_image, forward=True):
        permuted_image = torch.zeros_like(haar_image)
        if forward:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                for j in range(3):
                    permuted_image[:, 3*k+j, :, :] = haar_image[:, 4*j+i, :, :]
        else:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                
                for j in range(3):
                    permuted_image[:,4*j+k,:,:] = haar_image[:, 3*i+j, :, :]

        return permuted_image