import numpy as np
import torch
import os
import logging
from matplotlib import pyplot as plt
import io
import PIL
from torch._C import device
import torchvision.transforms as transforms
import numpy as np
import cv2
import math
from sklearn.neighbors import KernelDensity
from models.utils import get_score_fn

REPO_PATH_HOLIDAY = '/home/js2164/jan/repos/diffusion/score_sde_pytorch'

def generate_grid(n=500, d=2, c=[0,0], tensor=False):
    x = np.linspace(-d + c[0], d + c[0], n)
    y = np.linspace(-d + c[1], d + c[1], n)
    # Meshgrid
    X,Y = np.meshgrid(x,y)
    if tensor:
      XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
      XYpairs_tensor = torch.from_numpy(XYpairs) + 1e-10 # for numerical stability
      XYpairs_tensor=XYpairs_tensor.float()
      return XYpairs_tensor

    return X, Y

def extract_vector_field(pl_module, X, Y, t=0.):
    device = pl_module.device
    score_fn = get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)
    n = len(X[0])
    XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float, requires_grad=True, device=device)
    ts = torch.tensor([t] * n**2, dtype=torch.float, device=device)
    out = score_fn(xs, ts).view(n,n,-1)
    out_X = out[:,:,0].cpu().detach().numpy()
    out_Y = out[:,:,1].cpu().detach().numpy()
    return out_X, out_Y

def hist(data):
  s=data.detach().cpu().numpy().squeeze()
  fig = plt.figure()
  plt.hist(s, density=True, bins=30)
  kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
  kde.fit(s.reshape(-1,1))
  mn, mx = s.min(), s.max()
  t=np.linspace(mn,mx,100).reshape(-1,1)
  scores=np.exp(kde.score_samples(t))
  plt.plot(t,scores)
  buf = io.BytesIO()
  plt.savefig(buf, format='jpeg')
  buf.seek(0)
  image = PIL.Image.open(buf)
  image = transforms.ToTensor()(image)
  plt.close()
  return image

def scatter(x, y, **kwargs):
  fig = plt.figure()
  if 'title' in kwargs.keys():
    title = kwargs['title']
    plt.title(title)
  if 'xlim' in kwargs.keys():
    xlim = kwargs['xlim']
    plt.xlim(xlim)
  if 'ylim' in kwargs.keys():  
    ylim = kwargs['ylim']
    plt.ylim(ylim)
  plt.scatter(x, y)
  plt.gca().set_aspect('equal')
  buf = io.BytesIO()
  plt.savefig(buf, format='jpeg')
  buf.seek(0)
  image = PIL.Image.open(buf)
  image = transforms.ToTensor()(image)
  plt.close()
  return image

def plot(x, y, title):
  fig = plt.figure()
  plt.title(title)
  plt.plot(x, y)
  buf = io.BytesIO()
  plt.savefig(buf, format='jpeg')
  buf.seek(0)
  image = PIL.Image.open(buf)
  image = transforms.ToTensor()(image)
  plt.close()
  return image

def create_video(evolution, **kwargs):
  video_tensor = []
  for samples in evolution:
    samples_np =  samples.cpu().numpy()
    image = scatter(samples_np[:,0],samples_np[:,1], **kwargs)
    video_tensor.append(image)
  video_tensor = torch.stack(video_tensor)
  return video_tensor.unsqueeze(0)

def compute_grad(f,x):
  """
  Args:
      - fx - function 
      - x - inputs
  Retruns:
      - grads - tensor of gradients for each x
  """
  with torch.enable_grad():
    x = x.requires_grad_(True)
    out = f(x)
    gradients = torch.autograd.grad(outputs=out, inputs=x,
                            grad_outputs=torch.ones(out.size()).to(x.device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
  return gradients

def compute_divergence(f, x, hutchinson=False):
  """
  Args:
    - f - vector field function
    - x - inputs
  Returns:
    - div - divergence of the vector field f at each x
  """
  if hutchinson:
    with torch.enable_grad():
      #eps = torch.randn_like(x)
      eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
      x.requires_grad_(True)
      fn_eps = torch.sum(f(x) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
  else:
    with torch.enable_grad():
      x = x.requires_grad_(True)
      out = f(x)
      grads = []
      for i, v in enumerate(torch.eye(x.shape[1], device=x.device)):
          gradients = torch.autograd.grad(outputs=out, inputs=x,
                              grad_outputs=v.repeat(x.shape[0],1),
                              create_graph=True, retain_graph=True, only_inputs=True)[0][:,i]
          grads.append(gradients)
      grads = torch.stack(grads,dim=1)
    return torch.sum(grads, dim=1)


def compute_curl(f, xs):
  with torch.enable_grad():
    dvy_dx = compute_grad(lambda x: f(x)[:,1],xs)[:,0]
    dvx_dy = compute_grad(lambda x: f(x)[:,0],xs)[:,1]
    return (dvy_dx - dvx_dy)


def fisher_divergence(pl_module, data_module, t=0.01, grid=True):
  from models.utils import get_score_fn
  score_fn = get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)
  
  if grid:
      X, Y = generate_grid(n=25)
      XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
      XYpairs_tensor = torch.from_numpy(XYpairs) + 1e-10 # for numerical stability
      XYpairs_tensor = XYpairs_tensor.float()

      t = torch.tensor([t]*len(XYpairs_tensor))

      s_gt = data_module.dataset.ground_truth_score(XYpairs_tensor , t)

      
      s_model = score_fn(XYpairs_tensor, t)

      diff = torch.linalg.norm(s_gt - s_model, dim=1) ** 2
  else:
      eps=1e-5
      sde = pl_module.sde
      x = next(iter(data_module.val_dataloader()))
      t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps
      x_t = sde.perturb(x, t)

      diffusion = sde.sde(torch.zeros_like(x), t)[1]

      s_gt = data_module.dataset.ground_truth_score(x_t , t)
      s_model = score_fn(x_t, t)

      diff = diffusion**2 * torch.linalg.norm(s_gt - s_model, dim=1) ** 2
  return diff.mean().item()