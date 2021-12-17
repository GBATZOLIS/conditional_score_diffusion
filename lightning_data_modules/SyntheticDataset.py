import torch.distributions as D
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset, DataLoader 
import numpy as np
from PIL import Image
#helper function for plotting samples from a 2D distribution.
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import io
from . import utils
from torchvision.transforms.functional import normalize
from sklearn.datasets import make_circles
from utils import compute_grad2

class SyntheticDataset(Dataset):
    def __init__(self, config):
        super(SyntheticDataset, self).__init__()
        self.return_labels = config.data.return_labels
        self.data, self.labels = self.create_dataset(config)
   
    def create_dataset(self, config):
        raise NotImplemented
        # return data, labels

    def __getitem__(self, index):
        if self.return_labels:
            item = self.data[index], self.labels[index]
        else:
            item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)

class GaussianBubbles(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def create_dataset(self, config):
        data_samples = config.data.data_samples
        self.mixtures = config.data.mixtures
        n=self.mixtures
        categorical = D.categorical.Categorical(torch.ones(n,)/n)
        distributions = []
        self.centres = self.calculate_centers(n)
        for center in self.centres:
            distributions.append(D.normal.Normal(loc=center, scale=0.2))
        mixtures_indices = categorical.sample(torch.Size([data_samples]))
        data = []
        for index in mixtures_indices:
            data.append(distributions[index].sample().to(torch.float32))
        data = torch.stack(data)
        if normalize:
            data[:,0] = data[:,0] / torch.max(torch.abs(data[:,0]))
            data[:,1] = data[:,1] / torch.max(torch.abs(data[:,1]))
        return data, mixtures_indices

    def calculate_centers(self, num_mixtures):
                if num_mixtures==1:
                    return torch.zeros(1,2)
                else:
                    centers=[]
                    theta=0
                    for i in range(num_mixtures):
                        center=[np.cos(theta), np.sin(theta)]
                        centers.append(center)
                        theta+=2*np.pi/num_mixtures
                    centers=torch.tensor(centers)
                    return centers

    def gmm_log_prob(self, x, sigma_t=0):
        n=self.mixtures
        mus=torch.tensor(self.calculate_centers(n)).type_as(x)
        sigmas=torch.tensor([[.2 + sigma_t, .2 + sigma_t]]*n).type_as(x)
        mix = D.categorical.Categorical(torch.ones(n,).type_as(x))
        comp = D.independent.Independent(D.normal.Normal(
                    mus, sigmas), 1)
        gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm.log_prob(x)

    def ground_truth_score_backprop(self, batch, sigmas_t=None):
        if sigmas_t is None:
            sigmas_t = torch.zeros_like(batch)
        
        prob_fn = lambda batch: torch.stack(
                                            [self.gmm_log_prob(x, simga_t) 
                                                for (x, simga_t) in zip(batch, sigmas_t)], 
                                            dim=0)

        score = compute_grad2(prob_fn, batch)
        return score

    def ground_truth_score(self, batch, sigmas_t):
        def normal_density_2D(x, mu, sigma):
            const = 2 * np.pi * sigma**2
            return torch.exp(-torch.linalg.norm(x - mu)**2 / (2 * sigma**2)) / const
        def grad_normal_density_2D(x, mu, sigma):
            return normal_density_2D(x, mu, sigma) / (sigma**2) * (x - mu)
        def gmm_density(x, mus, sigma):
            mixture_dinsities = [normal_density_2D(x, mu, sigma) for mu in mus]
            return torch.mean(torch.stack(mixture_dinsities, dim=0))
        def grad_gmm_density(x, mus ,sigma):
            grad_mixture_dinsities = [grad_normal_density_2D(x, mu, sigma) for mu in mus]
            return torch.mean(torch.stack(grad_mixture_dinsities, dim=0), dim=0)
        def gmm_score(x, mus, sigma):
            return grad_gmm_density(x, mus, sigma) / gmm_density(x, mus, sigma)

        mus = self.centres
        sigma = 0.2
        scores = [gmm_score(x, mus, sigma + sigma_t) for (x, sigma_t) in zip (batch, sigmas_t)]
        return torch.stack(scores, dim=0)
    

class Circles(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)

    def create_dataset(self, config):
        data_samples = config.data.data_samples
        noise = config.data.noise
        factor = config.data.factor
        points, labels = make_circles(n_samples=data_samples, noise=noise, factor=factor)
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).float()
        return points, labels

@utils.register_lightning_datamodule(name='Synthetic')
class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config): 
        super().__init__()
        #Synthetic Dataset arguments
        self.config = config
        self.split = config.data.split
        self.dataset_type = self.config.data.dataset_type

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size
        
    def setup(self, stage=None): 
        if self.dataset_type == 'GaussianBubbles':
            self.data = GaussianBubbles(self.config)
        elif self.dataset_type == 'Circles':
            self.data = Circles(self.config)
        else:
            raise NotImplemented
        l=len(self.data)
        self.train_data, self.valid_data, self.test_data = random_split(self.data, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
    

def scatter_plot(x, x_lim=None, y_lim=None, labels=None, save=False):
    assert len(x.shape)==2, 'x must have 2 dimensions to create a scatter plot.'
    fig = plt.figure()
    x1 = x[:,0].cpu().numpy()
    x2 = x[:,1].cpu().numpy()
    plt.scatter(x1, x2, c=labels, s=8)
    if x_lim is not None and y_lim is not None:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    if save:
        plt.savefig('out.jpg', dpi=300)
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image