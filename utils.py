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

def compute_grad(f,x,t):
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
  ftx =f(x,t)
  assert len(ftx.shape)==1
  gradients = torch.autograd.grad(outputs=ftx, inputs=x,
                                  grad_outputs=torch.ones(ftx.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  torch.set_grad_enabled(torch_grad_enabled)
  return gradients

def compute_grad2(f, x):
    # CLEAN THE NAME UP (MULTIPLE DISPATCH)
    torch_grad_enabled =torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    device = x.device
    log_px = f(x)
    assert len(log_px.shape )==1
    gradients = torch.autograd.grad(outputs=log_px, inputs=x,
                                    grad_outputs=torch.ones(log_px.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    torch.set_grad_enabled(torch_grad_enabled)
    return gradients

