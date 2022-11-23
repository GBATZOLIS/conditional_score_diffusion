import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader 
from . import utils

class KSphereDataset(Dataset):

    def __init__(self, config) -> None:
        super().__init__()
        self.data = self.generate_data(config.data.data_samples, config.data.n_spheres, config.data.ambient_dim, config.data.manifold_dim, config.data.noise_std)

    def generate_data(self, n_samples, n_spheres, ambient_dim, manifold_dim, noise_std):
            data = []
            for _ in range(n_spheres):
                    # sample N(0, I) and normalize
                    new_data = torch.randn((n_samples, manifold_dim+1))
                    norms = torch.linalg.norm(new_data, dim=1)
                    new_data = new_data / norms[:,None]

                    # random isometric embedding
                    embedding_matrix = torch.randn((ambient_dim, manifold_dim+1))
                    q, r = np.linalg.qr(embedding_matrix)
                    q = torch.from_numpy(q)

                    # embed the data in the ambient space
                    new_data = (q @ new_data.T).T

                    # add noise
                    new_data = new_data + noise_std * torch.randn_like(new_data)
                    data.append(new_data)

            data = torch.cat(data, dim=0)
            return data

    def __getitem__(self, index):
        item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)


@utils.register_lightning_datamodule(name='KSphere')
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

        self.dataset = KSphereDataset(self.config)
        l=len(self.dataset)
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True)  
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=True) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 