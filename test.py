#%%
# Import required modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from vector_fields.vector_utils import  calculate_centers,curl
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from lightning_data_modules.SyntheticDataset import SyntheticDataModule
from configs.jan.GaussianBubbles import get_config
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
from lightning_modules.ConservativeSdeGenerativeModel import ConservativeSdeGenerativeModel
from models.fcn import FCN
from models.utils import get_score_fn

config = get_config()
data_m = SyntheticDataModule(config)
data_m.setup()
gaussian_bubbles = data_m.data

d=2
n=100
dx = 2*d/n
c=[0,0]
x = np.linspace(-d + c[0], d + c[0], n)
y = np.linspace(-d + c[1], d + c[1], n)
# Meshgrid
X,Y = np.meshgrid(x,y)
XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)#%%
XYpairs_tensor = torch.from_numpy(XYpairs)
t = torch.tensor([1.]*len(XYpairs_tensor))

plt.figure(figsize=(10, 10))
Z=data_m.data.log_prob(XYpairs_tensor, t).detach().numpy().reshape(n,n)
plt.contourf(X, Y, Z)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()