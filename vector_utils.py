import numpy as np
from utils import compute_grad
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
#%%
def normal_score(x, mus, sigmas):
    summands=[] # x centered using different mus
    for mu, sigma in zip(mus, sigmas):
        x_centred = gaussian_density2D(x,mu,sigma) * (- (x - mu)/ (sigma**2))
        summands.append(x_centred)
    summands=np.stack(summands, axis=0) # n_mix x 2
    return  (1/len(mus) * np.sum(summands, axis=0))/mixture_density(x,mus,sigmas)


def gaussian_density2D(x, mu=[0,0], sigma=1):
    norm = np.linalg.norm(x - mu)
    constant = 1/(2*np.pi * sigma**2)
    return constant * np.exp(-norm/(2*sigma**2))

def mixture_density(x,mus,sigmas):
    densities=[]
    for mu,sigma in zip(mus, sigmas):
        densities.append(gaussian_density2D(x,mu,sigma))
    densities=np.stack(densities, axis=0)
    return 1/len(mus) * np.sum(densities,axis=0)

def curl(vx, vy, dx):
    dvy_dx = np.gradient(vy, dx, axis=1)
    dvx_dy = np.gradient(vx, dx, axis=0)

    return dvy_dx - dvx_dy

def curl_backprop(f, xs, ts):
    dvy_dx = compute_grad(lambda x,t: f(x,t)[1],xs,ts)[:,0]
    dvx_dy = compute_grad(lambda x,t: f(x,t)[0],xs,ts)[:,1]
    return (dvy_dx - dvx_dy).detach().numpy()