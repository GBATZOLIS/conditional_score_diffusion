#%%
# Import required modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import sde_lib
import ot
import os
import itertools
import json
from importlib import reload
from models import utils as mutils
from vector_fields.vector_utils import  calculate_centers,curl
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from lightning_data_modules.SyntheticDataset import SyntheticDataModule
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
from lightning_modules.ConservativeSdeGenerativeModel import ConservativeSdeGenerativeModel
from models.fcn_potential import FCN_Potential
from models.fcn import FCN
from models.utils import get_score_fn
from sampling.unconditional import get_sampling_fn
from likelihood import get_likelihood_fn
from configs.jan.circles.potential.circles_potential import get_config as get_config_potential

    
def calculate_wasserstein(batch1, batch2):
    n_batch = batch1.shape[0]
    cost_matrix = torch.cdist(batch1, batch2)
    weights = torch.ones(n_batch)/n_batch
    wasserstein = float(ot.emd2(weights, weights, cost_matrix, numItermax=int(1e6)))
    return wasserstein

def calculate_regularised_wasserstein(batch1, batch2, epsilon = 1.0):
    n_batch = batch1.shape[0]
    C = torch.cdist(batch1, batch2)
    w = np.ones(n_batch)/n_batch
    d = float(ot.sinkhorn2(w, w, C, reg=epsilon))
    return d

def calculate_sinkhorn_np(batch1, batch2, epsilon = 1.0):
  return calculate_regularised_wasserstein(batch1, batch2, epsilon) - .5*(calculate_regularised_wasserstein(batch1, batch1, epsilon) + calculate_regularised_wasserstein(batch2, batch2, epsilon))

#%%
def visualise_samples(samples, title='samples', force_range=False):
    samples_np =  samples.cpu().numpy()
    plt.figure(figsize=(10,10))
    image = plt.scatter(samples_np[:,0],samples_np[:,1])
    if force_range:
        plt.ylim(-1.5,1.5)
        plt.xlim(-1.5,1.5)
    plt.gca().set_aspect('equal')
    plt.savefig('figures/'+title, dpi=300)
# %%
def plot_energy(score_model, t=0.0):
    d=6
    n=500
    dx = 2*d/n
    c=[0,0]
    x = np.linspace(-d + c[0], d + c[0], n)
    y = np.linspace(-d + c[1], d + c[1], n)
    # Meshgrid
    X,Y = np.meshgrid(x,y)
    XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)#%%
    XYpairs_tensor = torch.from_numpy(XYpairs).float()

    plt.figure(figsize=(10, 10))
    plt.title('Energy at t=' + str(t))
    t=torch.tensor([t]*len(XYpairs_tensor)).float()
    Z=np.log(score_model.energy(XYpairs_tensor,t).detach().numpy().reshape(n,n))
    plt.contourf(X, Y, Z, levels =100)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.show()
#%%
def get_config(snr=0.075, predictor='reverse_diffusion', corrector = 'none', 
                n_steps_each=1, probability_flow=False, num_scales=1000):
    config = get_config_potential()
    
    sampling = config.sampling
    sampling.corrector = 'pc'
    sampling.predictor =  predictor
    sampling.corrector = corrector
    sampling.n_steps_each = n_steps_each
    sampling.noise_removal = True
    sampling.probability_flow = probability_flow
    sampling.snr = snr

    model = config.model
    model.num_scales = num_scales

    return config
#%%
#ckpt_path = 'logs/mala/vanilla_ve/checkpoints/test.ckpt'
#ckpt_path = 'logs/mala/potential_ve2/checkpoints/test.ckpt'
#ckpt_path = 'logs/fokker_planck/fp_ve/checkpoints/test.ckpt'
ckpt_path = 'logs/fokker_planck/fp_v2_a0_01_n1/checkpoints/test.ckpt'
model = BaseSdeGenerativeModel.load_from_checkpoint(ckpt_path)
score_model = model.score_model
#%%
snr=0.2
predictor='none' #'heun'
corrector='mala'
n_steps_each=10
num_scales=10
probability_flow=True
config = get_config(snr,predictor, corrector, n_steps_each, probability_flow, num_scales)

# %%
num_samples = 1000           
sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=None)
sampling_eps = 1e-5
sampling_shape = [num_samples] +  config.data.shape
sampling_fn = get_sampling_fn(config, sde, sampling_shape, sampling_eps)
# %%
samples, _ = sampling_fn(score_model)#, show_evolution=False)
visualise_samples(samples, corrector, force_range=True)
#%%
visualise_samples(samples, corrector, force_range=False)
# %%
data_module = SyntheticDataModule(config)
data_module.setup()
data = data_module.data.data[:num_samples,:]
# %%
calculate_wasserstein(samples, data)
# %%
plot_energy(score_model, t=0.05)
# %%
likelihood_fn = get_likelihood_fn(sde, exact=True, rtol=1e-5, atol=1e-5,  t=0.99)
def log_likelihood(x):
    log_likelihoods, _, _ = likelihood_fn(score_model, x)
    return log_likelihoods
#%%
d=1.5
n=100
dx = 2*d/n
c=[0,0]
x = np.linspace(-d + c[0], d + c[0], n)
y = np.linspace(-d + c[1], d + c[1], n)

# Meshgrid
X,Y = np.meshgrid(x,y)
XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
XYpairs_tensor = torch.from_numpy(XYpairs).float()
log_likelihoods= -log_likelihood(XYpairs_tensor)
Z=log_likelihoods.detach().numpy().reshape(n,n)
#Z=torch.exp(log_likelihoods).detach().numpy().reshape(n,n)
#%%
#Z=np.exp(log_likelihoods / 20).reshape(n,n)
plt.figure(figsize=(10, 10))
plt.title('Energy')
plt.contourf(X, Y, Z, levels =100)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()
# %%
visualise_samples(data)
# %%
visualise_samples(samples)
# %%
calculate_wasserstein(samples, data)
# %%
snrs=[0.001, 0.005, 0.01, 0.05 ,0.1, 0.5]
steps=[1, 3, 5, 10, 20]
corrector='mala'
reverse_sde=False
wassersteins = dict()
for snr, n_steps_each in itertools.product(snrs, steps):
    print(' snr: ', snr,'\n n_steps: ', n_steps_each)
    config = get_config(snr, corrector, reverse_sde,n_steps_each)
    num_samples = 1000           
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=None)
    sampling_eps = 1e-5
    sampling_shape = [num_samples] +  config.data.shape
    sampling_fn = get_sampling_fn(config, sde, sampling_shape, sampling_eps)
    samples, _ = sampling_fn(score_model, show_evolution=False)
    if not os.path.exists('figures/'+ corrector):
        os.makedirs('figures/'+ corrector)
    path = corrector + '/snr' + str(snr).replace('.','_') + 'n_steps' + str(n_steps_each)
    visualise_samples(samples, path)
    plt.show()
    wassersteins[(str(snr), str(n_steps_each))] = calculate_wasserstein(samples, data)
    print('wasserstein: ', wassersteins[(str(snr), str(n_steps_each))])
# %%
with open('figures/langevin/wassersteins.json', 'w') as file:
    dict_copy = {'snr: '+key[0]+', n_steps: '+key[1]: value for key, value in wassersteins.items()}
    json.dump(dict_copy, file, indent=4)