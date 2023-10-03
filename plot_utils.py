from unicodedata import decimal
import torch
import io
import torchvision.transforms as transforms
import PIL
import pickle
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


def plot_spectrum(svd, return_tensor=False, mode='first', title='Score Spectrum', ground_truth=None):
    singular_values = extract_sing_vals(svd, mode)
    sing_vals = singular_values[0]
    
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(15,10))
    plt.grid(alpha=0.5)
    plt.title(title)
    plt.xticks(np.arange(0, len(sing_vals)+1, 10))
    if ground_truth:
        if isinstance(ground_truth, list):
            for gt in ground_truth:
                plt.axvline(x=len(sing_vals)-gt, color='red', ls='--')
        else:
            plt.axvline(x=len(sing_vals)-ground_truth, color='red', ls='--')
    for sing_vals in singular_values:
        #plt.bar(list(range(1, len(sing_vals)+1)),sing_vals)
        plt.plot(list(range(1, len(sing_vals)+1)),sing_vals)

    if return_tensor:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image
    else:
        return plt.gcf()

def plot_norms(samples, return_tensor=False):
    norms=torch.linalg.norm(samples, dim=1).cpu().detach().numpy()
    plt.rcParams.update({'font.size': 24})
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

def plot_distribution(svd, mode='first', return_tensor=False, tail=None):

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    singular_values = extract_sing_vals(svd, mode)
    sing_vals = singular_values[0]
    
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(15,10))
    plt.grid(alpha=0.5)
    plt.title('Dimension distribution')
    dims=[]
    for sing_vals in singular_values:
        s=sing_vals
        norm_factor = s[1]-s[2]
        diff = [(s[i]-s[i+1])/norm_factor for i in range(1, len(s)-1)]
        soft = softmax(diff)
        if tail:
            soft = soft[-tail:]
        plt.plot(list(range(1,1+len(soft)))[::-1],soft)
        #plt.xticks(np.arange(0, len(sing_vals)+1, 10))
        dim = len(soft)-soft.argmax()
        dims.append(dim)

    if return_tensor:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image, dims
    else:
        plt.show()
        return dims       

def extract_sing_vals(svd, mode='first'):
    print(f'Aggregation mode: {mode}')
    singular_vals = svd['singular_values']
    if mode == 'first':
        return [singular_vals[0]]
    elif mode == 'all':
        return singular_vals
    elif mode == 'mean':
        return [np.mean(singular_vals, axis=0)]
    
def plot_dims(svd, title='Histogram of dimensions', tick_step=2, tick_start=1):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    singular_values = extract_sing_vals(svd, 'all')
    sing_vals = singular_values[0]
    
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.autolayout"] = True
    plt.figure(figsize=(15,10))
    plt.grid(alpha=0.5)
    plt.xlabel('dimension')
    plt.ylabel('count')
    plt.title(title)
    dims=[]
    for sing_vals in singular_values:
        s=sing_vals
        norm_factor = s[1]-s[2]
        diff = [(s[i]-s[i+1])/norm_factor for i in range(1, len(s)-1)]
        soft = softmax(diff)
        dim = len(soft)-soft.argmax()
        dims.append(dim)      
    n, bins, patches = plt.hist(dims, bins=np.arange(tick_start,max(dims)+1,0.5))
    ticklabels = (bins[1:] + bins[:-1]) // 2 ## or ticks
    ticklabels = [int(label) for label in ticklabels][::tick_step]
    ticks = [(patch.get_x() + (patch.get_x() + patch.get_width()))/2 for patch in patches][::tick_step] ## or ticklabels

    plt.xticks(ticks, np.round(ticklabels, 2))

    return plt.gcf(), dims
