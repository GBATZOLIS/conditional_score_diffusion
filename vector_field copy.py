#%%
# Import required modules
import numpy as np
import matplotlib.pyplot as plt
from vector_fields.vector_utils import normal_score, calculate_centers,curl
#%%
d=2
n=500
dx = 2*d/n
c=[0,0]
x = np.linspace(-d + c[0], d + c[0], n)
y = np.linspace(-d + c[1], d + c[1], n)
# Meshgrid
X,Y = np.meshgrid(x,y)
XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
# %%
mus=calculate_centers(4)
sigmas=np.array([.2]*4)
vector=np.array([normal_score(x,mus,sigmas) for x in XYpairs]).reshape(n,n,2)
vector_X = vector[:,:,0]
vector_Y = vector[:,:,1]
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vector_X,vector_Y)
plt.grid()
plt.title('Ground Truth')
plt.savefig('figures/plot', dpi=300)
# %%
print("Curl:", np.mean(np.abs(curl(vector_X, vector_Y, dx))))
