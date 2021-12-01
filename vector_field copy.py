#%%
# Import required modules
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import fcn, fcn_potential
from vector_utils import calculate_centers, normal_score, curl, curl_backprop
from utils import compute_grad
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel

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
# non conservative
model = BaseSdeGenerativeModel.load_from_checkpoint('/home/js2164/jan/repos/diffusion_score/potential/lightning_logs/fcn/checkpoints/epoch=6249-step=499999.ckpt')
score = model.score_model
score = score.eval()
# conservative
model_conservative = BaseSdeGenerativeModel.load_from_checkpoint('potential/lightning_logs/fcn_potential/checkpoints/epoch=6249-step=499999.ckpt')
score_conservative = model_conservative.score_model
score_conservative= score_conservative.eval()
# %%
xs = torch.tensor(XYpairs, dtype=torch.float, requires_grad=True)
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
print(np.mean(np.abs(curl(out_X, out_Y,dx))))
print(np.mean(np.abs(curl(out_conservative_X, out_conservative_Y,dx))))
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,out_X,out_Y, density=1)
plt.grid()
plt.title('Standard')
plt.show()
#%%
Z=curl(out_X,out_Y,dx)
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, np.abs(Z))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
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
Z=curl(out_conservative_X,out_conservative_Y,dx)
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, np.abs(Z))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
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
print(np.mean(np.abs(curl(vector_X, vector_Y,dx))))

# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vector_X,vector_Y, density=1, color ='r')
plt.streamplot(X,Y,out_X,out_Y, density=1, color='b')
plt.streamplot(X,Y,out_conservative_X,out_conservative_Y, density=1, color='g')
plt.grid()
plt.show()
# %%
from models import fcn_potential, fcn
from types import SimpleNamespace

model_dict = {
        'state_size' : 2,
        'hidden_layers' : 1,
        'hidden_nodes' : 500,
        'dropout' : 0.0}
config = SimpleNamespace(model=SimpleNamespace(**model_dict))
test_model = fcn_potential.FCN_Potential(config)
#test_model = fcn.FCN(config)
test_model=test_model.eval()
# %%
out_test =test_model(xs, ts).view(n,n,-1)
out_test_X = out_test[:,:,0].detach().numpy()
out_test_Y = out_test[:,:,1].detach().numpy()
#%%
Z=curl(out_test_X,out_test_Y,dx)
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, np.abs(Z))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,out_test_X, out_test_Y, density=1, color='g')
plt.grid()
plt.show()
#%%
print(np.mean(np.abs(curl(out_test_X, out_test_Y, dx))))
#print(np.linalg.norm(curl(out_test_X, out_test_Y, dx)))
#print(np.mean(np.abs(curl_backprop(test_model,xs,ts))))
# %%
Z=curl(vector_X, vector_Y,dx)
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, np.abs(Z))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
from scipy.stats import norm
def mixture_density(x, y, mus, sigma=.2):
    distros=[]
    for mu in mus:
        distros.append(norm.pdf(x, loc=mu[0], scale=sigma)*norm.pdf(y, loc=mu[1], scale=sigma))
    np.stack(distros, axis=0)
    return 1/len(mus) * np.sum(distros, axis=0)
# %%
Z=mixture_density(X, Y, mus)
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, Z)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
v=np.gradient(np.log(Z), dx)
vx=v[0]
vy=v[1]
#%%
Z=test_model(xs, ts).view(n,n).detach()
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, Z)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
v=np.gradient(Z, dx)
vx=v[0]
vy=v[1]

# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vx, vy, density=1, color='g')
plt.grid()
plt.show()

#%%
Z=curl(vx,vy,dx)
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, np.abs(Z))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
out=test_model(xs, ts)
gradients = torch.autograd.grad(outputs=out, inputs=xs,
                                grad_outputs=torch.ones(out.size()),
                                create_graph=True, retain_graph=True,
                                only_inputs=True,)[0]
gradients = gradients.view(n,n,2).detach()
# %%
vx=gradients[:,:,0]
vy=gradients[:,:,1]
# %%
out=score_conservative(xs, ts)
gradients = torch.autograd.grad(outputs=out, inputs=xs,
                                grad_outputs=torch.ones(out.size()),
                                create_graph=True, retain_graph=True,)[0]

vx = gradients[:,0]
vy = gradients[:,1]

vx_grad= torch.autograd.grad(outputs=vx, inputs=xs,
                    grad_outputs=torch.ones(vx.size()), retain_graph=True)[0]

vy_grad= torch.autograd.grad(outputs=vy, inputs=xs,
                    grad_outputs=torch.ones(vy.size()), retain_graph=True)[0]

print(vx_grad.max())
#vy_grad[:,0] - vx_grad[:,1]

#%%
Z=vx.view(n,n).detach()
plt.figure(figsize=(10, 10))
plt.contourf(X, Y, np.abs(Z))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
