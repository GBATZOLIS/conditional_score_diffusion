import numpy as np
import torch
import torch.distributions as D
from utils import compute_grad
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
#%%
def calculate_centers(num_mixtures):
                if num_mixtures==1:
                    return np.zeros(1,2)
                else:
                    centers=[]
                    theta=0
                    for i in range(num_mixtures):
                        center=[np.cos(theta), np.sin(theta)]
                        centers.append(center)
                        theta+=2*np.pi/num_mixtures
                    return centers

# def normal_score(x, mus, sigmas):
#     # to be improved
#     n=len(mus)
#     mix = D.categorical.Categorical(torch.ones(n,))
#     nomral_distro = D.independent.Independent(D.normal.Normal(
#                 mus, sigmas), 1)
#     gmm = D.mixture_same_family.MixtureSameFamily(mix, nomral_distro)
#     summands=[] # x centered using different mus
#     for mu, sigma in zip(mus, sigmas):
#         gaussian_prob=torch.exp(nomral_distro.log_prob(x))
#         x_centred = gaussian_prob * (- (x - mu)/ (sigma**2))
#         summands.append(x_centred)
#     summands=torch.stack(summands, axis=0) # n_mix x 2
#     return  torch.sum(summands, axis=0)) / torch.exp(gmm.log_prob(x))

def curl(vx, vy, dx):
    dvy_dx = np.gradient(vy, dx, axis=1)
    dvx_dy = np.gradient(vx, dx, axis=0)

    return dvy_dx - dvx_dy

def curl_backprop(f, xs, ts):
    dvy_dx = compute_grad(lambda x,t: f(x,t)[:,1],xs,ts)[:,0]
    dvx_dy = compute_grad(lambda x,t: f(x,t)[:,0],xs,ts)[:,1]
    return (dvy_dx - dvx_dy)