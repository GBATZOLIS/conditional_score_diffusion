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
#%%
config = get_config()
data_m = SyntheticDataModule(config)
# %%
data_m.setup()
gaussian_bubbles = data_m.data
#%%
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
XYpairs_tensor.requires_grad=True
sigmas_t = torch.tensor([1.]*len(XYpairs_tensor))
s=gaussian_bubbles.ground_truth_score(XYpairs_tensor, sigmas_t=sigmas_t).detach().numpy()
vector_X=s[:,0].reshape(n,n)
vector_Y=s[:,1].reshape(n,n)
#%%
XYpairs_tensor = torch.from_numpy(XYpairs)
XYpairs_tensor.requires_grad=True
sigmas_t = torch.tensor([0.]*len(XYpairs_tensor)) #torch.tensor([0.01]*len(XYpairs_tensor))
s_backprop=gaussian_bubbles.ground_truth_score_backprop(XYpairs_tensor, sigmas_t=sigmas_t).detach().numpy()
vector_X_backprop=s_backprop[:,0].reshape(n,n)
vector_Y_backprop=s_backprop[:,1].reshape(n,n)
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vector_X,vector_Y)
plt.grid()
plt.title('Ground Truth')
plt.savefig('figures/plot', dpi=300)
# %%
print("Curl:", np.mean(np.abs(curl(vector_X, vector_Y, dx))))
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vector_X_backprop,vector_Y_backprop)
plt.grid()
plt.title('Ground Truth')
plt.savefig('figures/plot', dpi=300)
# %%
plt.figure(figsize=(10, 10))
Z=gmm.log_prob(XYpairs_tensor).detach().numpy().reshape(n,n)
plt.contourf(X, Y, Z)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()
# %%
def curl_backprop(f, xs):
    dvy_dx = compute_grad(lambda x: f(x)[:,1],xs)[:,0]
    dvx_dy = compute_grad(lambda x: f(x)[:,0],xs)[:,1]
    return (dvy_dx - dvx_dy)

def compute_grad(f, x):
  """
  Args:
      - f - function 
      - x - tensor shape (B, ...) where B is batch size
  Retruns:
      - grads - tensor of gradients for each x
  """
  torch_grad_enabled =torch.is_grad_enabled()
  torch.set_grad_enabled(True)
  device = x.device
  ftx =f(x)
  assert len(ftx.shape)==1
  gradients = torch.autograd.grad(outputs=ftx, inputs=x,
                                  grad_outputs=torch.ones(ftx.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  torch.set_grad_enabled(torch_grad_enabled)
  return gradients



# %%
torch.mean(curl_backprop(score, XYpairs_tensor)**2).item()
# %%
model = BaseSdeGenerativeModel.load_from_checkpoint('/home/js2164/jan/repos/diffusion_score/logs/gaussian_bubbles/lightning_logs/version_2/checkpoints/epoch=3673-step=293919.ckpt')
score = model.score_model
score = score.eval()
model.configure_sde(config)
score_fn = get_score_fn(model.sde, score, train=False, continuous=True)
# %%
xs = torch.tensor(XYpairs, dtype=torch.float)
ts = torch.tensor([1.] * n**2, dtype=torch.float)
sigmas = model.sde.marginal_prob(torch.zeros_like(xs), ts)[1]
out_flat = score(xs, sigmas)
out = out_flat.view(n,n,-1)
out_X = out[... ,0].detach().numpy()
out_Y = out[... ,1].detach().numpy()
out_np = out.detach().numpy()
s_approx = out_flat / sigmas[..., None]
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,vector_X,vector_Y)
plt.grid()
plt.title('Ground Truth')
plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,out_X,out_Y)
plt.grid()
plt.title('Model')
plt.show()
# %%
print("Curl:", np.mean(np.abs(curl(out_X, out_Y, dx))))
#%%
gaussian_bubbles.centres

#%%

# %%
consine_similarity = torch.nn.CosineSimilarity()
# %%
out_flat = score(xs, ts)
s_tensor = torch.from_numpy(s)
torch.mean(consine_similarity(s_tensor, out_flat))
# %%
batch = xs
eps = 1e-5
t = torch.rand(batch.shape[0], device=batch.device) * (model.sde.T - eps) + eps
z = torch.randn_like(batch)
mean, std = model.sde.marginal_prob(batch, t)
perturbed_data = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
perturbed_data.requires_grad = True
g2 = model.sde.sde(torch.zeros_like(batch), t)[1] ** 2
model_score = score(perturbed_data, t) / sigmas[..., None]
gt_score = gaussian_bubbles.ground_truth_score(perturbed_data, std)
fisher_div = torch.mean(g2 * torch.linalg.norm(gt_score - model_score, dim=1)**2)
print(fisher_div)
# %%
