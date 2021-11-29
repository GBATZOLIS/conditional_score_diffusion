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
  device = x.device
  x.requires_grad=True
  ftx =f(x,t)
  gradients = torch.autograd.grad(outputs=ftx, inputs=x,
                                  grad_outputs=torch.ones(ftx.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  return gradients


