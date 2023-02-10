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
from utils import generate_grid, extract_vector_field, compute_curl

def plot_vector_field(pl_module, title='Stream plot', t=0., lines=False):
    n = 500 if lines else 25
    X,Y = generate_grid(n=n)
    out_X, out_Y = extract_vector_field(pl_module, X, Y, t)
    plt.figure(figsize=(10, 10))
    if lines:
        plt.streamplot(X,Y,out_X,out_Y, density=1)
    else:
        plt.quiver(X,Y,out_X,out_Y)
    plt.grid()
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image


# def plot_curl(pl_module, title='Curl'):
#     X,Y = generate_grid()
#     out_X, out_Y = extract_vector_field(pl_module, X, Y)
#     n = len(X[0])
#     dx = 2*2/n
#     Z=curl(out_X,out_Y,dx)
#     plt.figure(figsize=(10, 10))
#     plt.contourf(X, Y, np.abs(Z))
#     plt.colorbar()
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.gca().set_aspect('equal')
#     plt.title(title)
#     buf = io.BytesIO()
#     plt.savefig(buf, format='jpeg')
#     buf.seek(0)
#     image = PIL.Image.open(buf)
#     image = transforms.ToTensor()(image)
#     plt.close()
#     return image

def plot_curl(pl_module, title='Curl', t=0.):
    device = pl_module.device
    score_fn = get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)
    X,Y = generate_grid()
    n = len(X[0])
    XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float, requires_grad=True, device=device)
    ts = torch.tensor([t] * n**2, dtype=torch.float, device=device)
    fn = lambda x: score_fn(x,ts)
    Z=compute_curl(fn, xs).cpu().detach().numpy().reshape(n,n)
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

# def plot_samples(pl_module, title='Samples', t=0):
#     device = pl_module.device
#     score_fn = get_score_fn(pl_module.sde, pl_module.score_model, train=False, continuous=True)

def plot_log_energy(score_model, t=0.0, X=None, Y=None):
    if X is None and Y is None:
        d=6
        n=500
        dx = 2*d/n
        c=[0,0]
        x = np.linspace(-d + c[0], d + c[0], n)
        y = np.linspace(-d + c[1], d + c[1], n)
        # Meshgrid
        X,Y = np.meshgrid(x,y)
    n = X.shape[0]
    XYpairs = np.stack([ X.reshape(-1), Y.reshape(-1) ], axis=1)#%%
    XYpairs_tensor = torch.from_numpy(XYpairs).float()

    plt.figure(figsize=(10, 10))
    plt.title('Energy at t=' + str(t))
    t=torch.tensor([t]*len(XYpairs_tensor)).float()
    Z=score_model.log_energy(XYpairs_tensor,t).detach().numpy().reshape(n,n)
    plt.contourf(X, Y, Z, levels =100)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.show()
    return Z


def plot_spectrum(singular_values, return_tensor=False):
    sing_vals = (np.array(singular_values)).mean(axis=0)
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(15,10))
    plt.bar(list(range(1, len(sing_vals)+1)),sing_vals)
    plt.grid(alpha=0.5)
    plt.title('Score Spectrum')
    plt.xticks(np.arange(0, len(sing_vals)+1, 10))
    if return_tensor:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image
    else:
        plt.show()

def plot_norms(samples, return_tensor=False):
    norms=torch.linalg.norm(samples, dim=1).cpu().detach().numpy()
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10,10))
    plt.grid(alpha=0.5)
    plt.hist(norms, bins=50)
    if return_tensor:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image
    else:
        plt.show()