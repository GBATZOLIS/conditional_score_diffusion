import pytorch_lightning as pl
import torch
import numpy as np
import json
from torch.utils.data import random_split, Dataset, DataLoader 
import lightning_data_modules.utils as utils

class MammothDataset(Dataset):

    def __init__(self, config) -> None:
        super().__init__()
        self.data = self.generate_data(
                                        config.data.get('ambient_dim', 3), 
                                        config.data.get('noise_std', 0), 
                                        config.data.get('embedding_type', 'first'),
                                        )

    def __getitem__(self, index):
        item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)

    def generate_data(self, ambient_dim, 
                        noise_std, 
                        embedding_type):

        with open('mammoth.json', 'rb') as f:
            mammoth_list = json.load(f)
        
        mammoth = torch.tensor(mammoth_list)
        mammoth = mammoth - mammoth.mean(0) 
        mammoth = mammoth / (mammoth.max() - mammoth.min())
        
        n_samples = mammoth.shape[0]
        manifold_dim = 2

        if embedding_type == 'random_isometry':
            # random isometric embedding
            randomness_generator = torch.Generator().manual_seed(0)
            embedding_matrix = torch.randn(size=(ambient_dim, manifold_dim+1), generator=randomness_generator)
            q, r = np.linalg.qr(embedding_matrix)
            q = torch.from_numpy(q)
            mammoth = (q @ mammoth.T).T
        elif embedding_type == 'first':
            # embedding into first manifold_dim + 1 dimensions
            suffix_zeros = torch.zeros([n_samples, ambient_dim - mammoth.shape[1]])
            mammoth = torch.cat([mammoth, suffix_zeros], dim=1)
        else:
            raise RuntimeError('Unknown embedding type.')
            
        # add noise
        mammoth = mammoth + noise_std * torch.randn_like(mammoth)
        return mammoth


@utils.register_lightning_datamodule(name='Mammoth')
class MammothDataModule(pl.LightningDataModule):
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

        self.dataset = MammothDataset(self.config)
        l=len(self.dataset)
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True)  
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=True) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 