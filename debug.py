import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
from models.fcn import FCN
from models.ddpm import DDPM
import pickle
import sde_lib

config_path = 'logs/celebA/real_celebA_crop_ampere/config.pkl'
with open(config_path, 'rb') as file:
    config = pickle.load(file)
config.model.checkpoint_path = 'logs/celebA/real_celebA_crop_ampere/checkpoints/best/last.ckpt'

config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def sample(config):
    from sampling.unconditional import get_sampling_fn
    ckpt_path = config.model.checkpoint_path
    pl_module = BaseSdeGenerativeModel.load_from_checkpoint(ckpt_path)
    pl_module.to(config.device)
    score_model = pl_module.score_model
    pl_module.configure_sde(config)
    pl_module = pl_module.eval()

    num_samples = 10  
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=None) #pl_module.sde
    sampling_eps = 1e-5
    sampling_shape = [num_samples] +  config.data.shape
    sampling_fn = get_sampling_fn(config, sde, sampling_shape, sampling_eps)
    score_model = pl_module.score_model
    samples, _ = sampling_fn(score_model)#, show_evolution=False)
    return samples

config.sampling.method = 'pc'
config.sampling.predictor = 'reverse_diffusion'
config.sampling.corrector = 'none'
config.sampling.n_steps_each = 1
config.sampling.noise_removal = True
config.sampling.probability_flow = False
config.sampling.snr = 0.0001
config.model.num_scales=1000

samples = sample(config)
samples_normalized = samples
print(samples.min())
print(samples.max())

import torchvision
from matplotlib import pyplot as plt
grid_images = torchvision.utils.make_grid(samples_normalized.cpu(), normalize=True, scale_each=True)
plt.figure(figsize=(10,10))
plt.imshow(grid_images.permute(1,2,0))
plt.show(block=True)