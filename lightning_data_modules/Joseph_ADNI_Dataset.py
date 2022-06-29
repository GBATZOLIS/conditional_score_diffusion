from torch.utils.data import  Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
import torch
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from . import utils
import pytorch_lightning as pl
from glob import glob 
import scipy

def listdir_nothidden_filenames(path, filetype=None):
    if not filetype:
        paths = glob(os.path.join(path, '*'))
    else:
        paths = glob(os.path.join(path, '*.%s' % filetype))
    files = [os.path.basename(path) for path in paths]
    return files

def load_data(path):
    IDs = listdir_nothidden_filenames(path)
    data = {}
    for i, ID in enumerate(IDs):
        ID_data = {}
        for quantity in listdir_nothidden_filenames(os.path.join(path, ID)):
            ID_data[quantity.split('.')[0]] = np.load(os.path.join(path, ID, quantity))
        data[i] = ID_data
    
    return data

class JosephADNI_Dataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self,  config, phase):
        # get the image paths of your dataset;
        self.phase = phase
        self.data = load_data(os.path.join(config.data.base_dir, config.data.dataset, phase))
        self.use_data_augmentation = config.data.use_data_augmentation

    def __getitem__(self, index):
        mri = self.data[index]['img_mri']
        pet = self.data[index]['img_pet']

        if self.use_data_augmentation and self.phase=='train':
            available_dims = np.arange(0, len(mri.shape))
            flipped_dims = []
            for dim in available_dims:
                if np.random.randint(2) == 0:
                    flipped_dims.append(dim)
            
            mri = np.flip(mri, tuple(flipped_dims)).copy()
            pet = np.flip(pet, tuple(flipped_dims)).copy()

            '''
            if np.random.randint(2) == 0:
                angle = np.random.randint(0, 360)
                axes_combo = (0, 2)
                rotated_mri = scipy.ndimage.rotate(mri, angle=angle, axes=axes_combo, reshape=False)
                rotated_pet = scipy.ndimage.rotate(pet, angle=angle, axes=axes_combo, reshape=False)
                assert rotated_mri.shape == mri.shape and rotated_pet.shape == pet.shape, 'rotated shapes do not match initial shapes'
                mri = rotated_mri
                pet = rotated_pet
            '''
        
        mri = torch.tensor(mri, dtype=torch.float32).unsqueeze(0)
        pet = torch.tensor(pet, dtype=torch.float32).unsqueeze(0)

        return mri, pet
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.data.keys())

@utils.register_lightning_datamodule(name='JosephADNI')
class JosephADNIDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        #DataLoader arguments
        self.config = config
        self.train_workers = config.training.workers
        self.val_workers = config.eval.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.eval.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None): 
        self.train_dataset = JosephADNI_Dataset(self.config, phase='train')
        self.val_dataset = JosephADNI_Dataset(self.config, phase='validation')
        self.test_dataset = JosephADNI_Dataset(self.config, phase='validation')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

