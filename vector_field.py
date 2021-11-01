#%%
# Import required modules
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import fcn, fcn_potential
from vector_utils import calculate_centers, normal_score, curl
from utils import compute_grad
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel

#%%
d=1
n=100
c=[0,0]
x = np.linspace(-d + c[0], d + c[0], n)
y = np.linspace(-d + c[1], d + c[1], n)
# Meshgrid
X,Y = np.meshgrid(x,y)
XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)

# %%
# non conservative
model = BaseSdeGenerativeModel.load_from_checkpoint('/home/js2164/jan/repos/diffusion_score/lightning_logs/version_28/checkpoints/epoch=4367-step=349439.ckpt')
score = model.score_model
score = score.eval()
# conservative
model_conservative = BaseSdeGenerativeModel.load_from_checkpoint('/home/js2164/jan/repos/diffusion_score/lightning_logs/version_42/checkpoints/epoch=1604-step=128399.ckpt')
score_conservative = model_conservative.score_model
score_conservative= score_conservative.eval()
# %%
xs = torch.tensor(XYpairs, dtype=torch.float)
ts = torch.tensor([0.] * n**2, dtype=torch.float)
# %%
out = score(xs, ts).view(n,n,-1)
out_X = out[:,:,0].detach().numpy()
out_Y = out[:,:,1].detach().numpy()
#%%
out_conservative =score_conservative(xs, ts).view(n,n,-1)
out_conservative_X = out_conservative[:,:,0].detach().numpy()
out_conservative_Y = out_conservative[:,:,1].detach().numpy()

#%%
np.mean(np.abs(curl(out_X, out_Y)))
#%%
np.mean(np.abs(curl(out_conservative_X, out_conservative_Y)))
#%%
def curl_backprop(f, xs, ts):
    dvy_dx = compute_grad(lambda x,t: f(x,t)[1],xs,ts)[:,0]
    dvx_dy = compute_grad(lambda x,t: f(x,t)[0],xs,ts)[:,1]
    return (dvy_dx - dvx_dy).detach().numpy()
#%%
np.mean(np.abs(curl_backprop(score,xs, ts)))
#%%
np.mean(np.abs(curl(out_conservative_X, out_conservative_Y)))
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,out_X,out_Y, density=1)
plt.grid()
plt.title('Standard')
plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,out_conservative_X,out_conservative_Y, density=1)
#plt.quiver(X,Y,out_X,out_Y)
plt.grid()
plt.title('Learning the potential')
plt.show()

#%%
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
plt.show()

#%%
np.mean(curl(vector_X,vector_Y))
# %%
np.linalg.norm(out.detach()-vector) / n**2
# %%
np.linalg.norm(out_conservative.detach()-vector) / n**2
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vector_X,vector_Y, density=1, color ='r')
plt.streamplot(X,Y,out_X,out_Y, density=1, color='b')
plt.streamplot(X,Y,out_conservative_X,out_conservative_Y, density=1, color='g')
plt.grid()
plt.show()
# %%
from models import fcn_potential
from types import SimpleNamespace

model_dict = {
        'state_size' : 2,
        'hidden_layers' : 3,
        'hidden_nodes' : 64,
        'dropout' : 0.25}
config = SimpleNamespace(model=SimpleNamespace(**model_dict))
test_model = fcn_potential.FCN_Potential(config)
test_model=test_model.eval()
#%%
curl_backprop(test_model,xs,ts)
# %%
