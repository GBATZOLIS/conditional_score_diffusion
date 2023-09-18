import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset, DataLoader 
import pickle
from . import utils

class GanDataset(Dataset):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.data_path = self.config.data.data_path
        self.latent_dim = self.config.data.latent_dim
        
        if hasattr(self.config.data, 'style_gan'):
            if self.config.data.style_gan:
                self.data = np.load(os.path.join(self.data_path, f'style_gan_horvat/gan_{self.latent_dim}d_train.npy'))
                self.data = torch.from_numpy(self.data).float()
            else:
                self.data = torch.load(os.path.join(self.data_path, f'latent_dim_{self.latent_dim}/data.pt'))

    def __getitem__(self, index):
        item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)

@utils.register_lightning_datamodule(name='Gan')
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
        self.dataset = GanDataset(self.config)

        l=len(self.dataset)
        train_len = int(self.split[0]*l)
        val_len =  int(self.split[1]*l)
        test_len = l - train_len - val_len
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [train_len, val_len, test_len]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
    