import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def intrinsic_dim_sample_wise(X, k=5, dist = None):
    """
    Returns Levina-Bickel dimensionality estimation
    
    Input parameters:
    X    - data
    k    - number of nearest neighbours (Default = 5)
    dist - matrix of distances to the k nearest neighbors of each point (Optional)
    
    Returns: 
    dimensionality estimation for the k 
    """
    if dist is None:
        neighb = NearestNeighbors(n_neighbors=k+1, n_jobs=1, algorithm='ball_tree').fit(X)
        dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:(k+1)]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample


def intrinsic_dim_scale_interval(X, k1=10, k2=20, dist = None):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2, k1 < k2
    
    Input parameters:
    X    - data
    k1   - minimal number of nearest neighbours (Default = 10)
    k2   - maximal number of nearest neighbours (Default = 20)
    dist - matrix of distances to the k nearest neighbors of each point (Optional)
    
    Returns: 
    list of Levina-Bickel dimensionality estimation for k = k1..k2
    """
    intdim_k = []
    if dist is None:
        neighb = NearestNeighbors(n_neighbors=k2+1, n_jobs=1, algorithm='ball_tree').fit(X)
        dist, ind = neighb.kneighbors(X)
        
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k,dist).mean()
        intdim_k.append(m)
    return intdim_k


def bootstrap_intrinsic_dim_scale_interval(X, nb_iter=100, random_state=None,
                                           k1 = 10, k2 = 20, average = False):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples
    
    Input parameters:
    X            - data
    nb_iter      - number of bootstrap iterations (Default = 100)
    random_state - random state (Optional)
    k1           - minimal number of nearest neighbours (Default = 10)
    k2           - maximal number of nearest neighbours (Default = 20)
    average      - if False returns array of shape (nb_iter, k2-k1+1) of the estimations for each bootstrap samples (Default = True)
    
    Returns: 
    array of shape (k2-k1+1,) of Levina-Bickel dimensionality estimation for k = k1..k2 averaged over bootstrap samples
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
        
    nb_examples = X.shape[0]
    results = []
    
    neighb = NearestNeighbors(n_neighbors=k2+1, n_jobs=1, algorithm='ball_tree').fit(X)
    dist, ind = neighb.kneighbors(X)    
    
    Rs = []
    for i in range(k1, k2 + 1):
        Rs.append(np.max(dist[:,:i]))
    
    for i in range(nb_iter):
        idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
        results.append(intrinsic_dim_scale_interval(X.iloc[idx], k1, k2, dist[idx,:]))
    results = np.array(results)
    
    if average:
        return results.mean(axis = 0),Rs
    else:
        return results,Rs