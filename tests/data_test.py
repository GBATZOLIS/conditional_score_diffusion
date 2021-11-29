#%% 
import os
os.chdir('..')
# %% 
from lightning_data_modules.Synthetic1DConditionalDataset import Synthetic1DConditionalDataModule 
from configs.jan.holiday.Synthetic1DDataset import get_config

#%%
config = get_config()
#print(config)
data_module = Synthetic1DConditionalDataModule(config)
data_module.setup()

# %%
train_loader = data_module.train_dataloader()
for item in train_loader:
    x=item
    break
# %%
import torch
ys =torch.tensor([1,2])
for y in ys:
    print(y.repeat(100))
# %%

# %%
pts=x[1]
from matplotlib import pyplot as plt
plt.scatter(pts[:,0],pts[:,1])
# %%
import torch.distributions as D
import torch
centers = torch.tensor([[-1],[1]]).float()
dim=1
n=2
data_samples=500
comp = D.independent.Independent(D.Normal(centers, 0.2*torch.ones(n,dim)), 1)
mix = D.categorical.Categorical(torch.ones(n,))
gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
data = gmm.sample(torch.Size([data_samples])).float()
# %%

from matplotlib import pyplot as plt
import numpy as np


s=data.numpy().squeeze()
plt.hist(s, density=True, bins=30)

kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde.fit(s.reshape(-1,1))
mn, mx = s.min(), s.max()
t=np.linspace(mn,mx,100).reshape(-1,1)
scores=np.exp(kde.score_samples(t))
plt.plot(t,scores)

# %%
plt.hist(x, density=True, bins=10) 
# %%
from sklearn.neighbors import KernelDensity
x=np.random.normal(0,1,(1000,1))
kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde.fit(x.reshape(-1,1))
mn, mx = x.min(), x.max()
t=np.linspace(mn,mx,100).reshape(-1,1)
scores=np.exp(kde.score_samples(t))
plt.plot(t,scores)
# %%
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde.fit(s.reshape(-1,1))
mn, mx = s.min(), s.max()
t=np.linspace(mn,mx,100).reshape(-1,1)
scores=np.exp(kde.score_samples(t))
plt.plot(t,scores)

# %%
