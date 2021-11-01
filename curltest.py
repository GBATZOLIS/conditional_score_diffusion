#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from vector_utils import curl

#%%
def toy_vector_field(point, high_curl=False):
    if high_curl:
        return np.array([point[1],-point[0]])
    else:
        return np.array([point[1],point[0]])
# %%
d=1
n=100
dx = 2*d/n
c=[0,0]
x = np.linspace(-d + c[0], d + c[0], n)
y = np.linspace(-d + c[1], d + c[1], n)
# Meshgrid
X,Y = np.meshgrid(x,y)
XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
# %%
v=np.array([toy_vector_field(x, high_curl=True) for x in XYpairs]).reshape(n,n,2)
vx = v[:,:,0]
vy= v[:,:,1]
# %%
np.mean(np.abs(curl(vx,vy, dx)))
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vx,vy, density=1)
plt.grid()

# %%
