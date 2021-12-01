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

class SyntheticConditionalDataset(Dataset):
    def __init__(self, data_samples, dataset_type='ConditionalGaussianBubbles', mixtures=4, return_distances=True, normalize=False, y_min=0, y_max=1):
        super(SyntheticConditionalDataset, self).__init__()
        #self.data, self.labels = self.read_dataset(filename)
        #self.transform = transforms.Compose([convert_to_robust_range])
        self.y_min = y_min
        self.y_max = y_max
        self.normalize = normalize
        self.data_samples = data_samples
        self.dataset_type = dataset_type
        self.mixtures = mixtures
        self.return_distances = return_distances
        self.data, self.distances = self.create_dataset(self.dataset_type, self.mixtures, self.data_samples)

    def create_dataset(self, dataset_type, mixtures, data_samples):
        if dataset_type == 'ConditionalGaussianBubbles':
            
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
           
            n=mixtures
            possible_centres = calculate_centers(n)
            possible_distances = torch.linspace(self.y_min, self.y_max, 100)

            mixtures_indices = torch.randint(len(possible_centres), (data_samples,))
            centres = possible_centres[mixtures_indices]
            distances = possible_distances[torch.randint(len(possible_distances), (data_samples,))]
            data = []
            for center, distance in zip(centres, distances):
                distribution = D.normal.Normal(loc=distance * center, scale=0.2)
                data.append(distribution.sample().to(torch.float32))
            data = torch.stack(data)

            #mix = D.categorical.Categorical(torch.ones(n,))
            #comp = D.independent.Independent(D.Normal(calculate_centers(n), 0.2*torch.ones(n,2)), 1)
            #gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
            #data = gmm.sample(torch.Size([data_samples])).float()
            if normalize:
                data[:,0] = data[:,0] / torch.max(torch.abs(data[:,0]))
                data[:,1] = data[:,1] / torch.max(torch.abs(data[:,1]))
            return data, distances
    
    def __getitem__(self, index):
        if self.return_distances:
            item = self.distances[index], self.data[index]
        else:
            item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)


@utils.register_lightning_datamodule(name='ConditionalSynthetic')
class SyntheticConditionalDataModule(pl.LightningDataModule):
    def __init__(self, config): 
        super().__init__()
        #Synthetic Dataset arguments
        self.data_samples=config.data.data_samples
        self.dataset_type=config.data.dataset_type
        self.mixtures = config.data.mixtures
        self.y_min = config.data.y_min
        self.y_max = config.data.y_max
        self.return_distances = config.data.return_distances
        self.split = config.data.split

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size


        #self.normalize = config.normalize
        
    def setup(self, stage=None): 
        data = SyntheticConditionalDataset(self.data_samples, self.dataset_type, self.mixtures, self.return_distances, y_min=self.y_min, y_max=self.y_max)
        l=len(data)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=True) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers,  shuffle=True) 
    
