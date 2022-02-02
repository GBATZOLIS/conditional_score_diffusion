from unicodedata import decimal
import torch
import io
import torchvision.transforms as transforms
import PIL
import numpy as np
import copy
from matplotlib import pyplot as plt
from models.utils import get_score_fn
from vector_fields.vector_utils import curl, curl_backprop

def generate_grid():
    d=2
    n=500
    c=[0,0]
    x = np.linspace(-d + c[0], d + c[0], n)
    y = np.linspace(-d + c[1], d + c[1], n)
    # Meshgrid
    X,Y = np.meshgrid(x,y)
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

def plot_streamlines(pl_module, title='Stream plot', t=0.):
    X,Y = generate_grid()
    out_X, out_Y = extract_vector_field(pl_module, X, Y, t)
    plt.figure(figsize=(10, 10))
    plt.streamplot(X,Y,out_X,out_Y, density=1)
    plt.grid()
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image


def plot_curl(pl_module, title='Curl'):
    X,Y = generate_grid()
    out_X, out_Y = extract_vector_field(pl_module, X, Y)
    n = len(X[0])
    dx = 2*2/n
    Z=curl(out_X,out_Y,dx)
    plt.figure(figsize=(10, 10))
    plt.contourf(X, Y, np.abs(Z))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image

def plot_curl_backprop(pl_module, title='Curl', t=0.):
    device = pl_module.device
    score_fn = get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)
    X,Y = generate_grid()
    n = len(X[0])
    XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float, requires_grad=True, device=device)
    ts = torch.tensor([t] * n**2, dtype=torch.float, device=device)
    Z=curl_backprop(score_fn,xs, ts).cpu().detach().numpy().reshape(n,n)
    plt.figure(figsize=(10, 10))
    plt.contourf(X, Y, np.abs(Z))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image