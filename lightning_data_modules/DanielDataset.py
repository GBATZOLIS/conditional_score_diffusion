import pytorch_lightning as pl
import torch
import numpy as np
import json
from torch.utils.data import random_split, Dataset, DataLoader 
import lightning_data_modules.utils as utils

class DanielDataset(Dataset):

    def __init__(self, config) -> None:
        super().__init__()
        self.data = self.get_data(config.data.data_path)

    def __getitem__(self, index):
        item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)

    def get_data(self, path):
        x = np.load(path)
        # normalize to (-1,1) range
        x = x - x.min(0)
        i = x.max(0) - x.min(0)
        x = x / i * 2 - 1
        return torch.tensor(x, dtype=torch.float32)

@utils.register_lightning_datamodule(name='Daniel')
class DanielDataModule(pl.LightningDataModule):
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

        self.dataset = DanielDataset(self.config)
        l=len(self.dataset)
        l_train, l_val = int(self.split[0]*l), int(self.split[1]*l)
        l_test = l - (l_train + l_val)
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [l_train, l_val, l_test]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True)  
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=True) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 