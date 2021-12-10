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

class SyntheticDataset(Dataset):
    def __init__(self, config):
        super(SyntheticDataset, self).__init__()
        self.return_labels = config.data.return_labels
        self.data, self.labels = self.create_dataset(config)

    def create_dataset(self, config):
        #normalize = config.data.normalize
        data_samples = config.data.data_samples
        dataset_type = config.data.dataset_type
        if dataset_type == 'GaussianBubbles':
            self.mixtures = config.data.mixtures
            def calculate_centers(num_mixtures):
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
            n=self.mixtures
            categorical = D.categorical.Categorical(torch.ones(n,)/n)
            distributions = []
            for center in calculate_centers(n):
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

        elif dataset_type == 'Circles':
            noise = config.data.noise
            factor = config.data.factor
            points, labels = make_circles(n_samples=data_samples, noise=noise, factor=factor)
            points = torch.from_numpy(points).float()
            labels = torch.from_numpy(labels).float()
            return points, labels

    def __getitem__(self, index):
        if self.return_labels:
            item = self.data[index], self.labels[index]
        else:
            item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)

    
    def ground_truth_score(self, x, t):
        return None

@utils.register_lightning_datamodule(name='Synthetic')
class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config): 
        super().__init__()
        #Synthetic Dataset arguments
        self.config = config
        self.split = config.data.split

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size
        
    def setup(self, stage=None): 
        data = SyntheticDataset(self.config)
        l=len(data)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
    
