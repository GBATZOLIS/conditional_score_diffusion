import os
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

    def __getitem__(self, index):
        i = index // int(1e4)
        j = index % int(1e4)
        with open(os.path.join(self.data_path, f'latent_dim_{self.latent_dim}_part_{i}'), 'rb') as f:
            X = pickle.load(f)
        item = X[j]
        
        return item 

    def __len__(self):
        return int(1e5)

@utils.register_lightning_datamodule(name='Gan')
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
        self.dataset = GanDataset(self.config)

        l=len(self.dataset)
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
    