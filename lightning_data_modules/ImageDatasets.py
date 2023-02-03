import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision  import transforms, datasets

import PIL.Image as Image
from . import utils
import os
import glob

class MNISTDataset(datasets.MNIST):
    def __init__(self, config):
        super().__init__(root=config.data.base_dir, train=True, download=True)
        transforms_list=[transforms.ToTensor(), transforms.Pad(2, fill=0)] #left and right 2+2=4 padding
        self.transform_my = transforms.Compose(transforms_list)

        self.return_labels = config.data.return_labels
    
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x = self.transform_my(x)
        if self.return_labels:
            return x, y
        else:
            return x

def load_file_paths(dataset_base_dir):
    listOfFiles = [os.path.join(dataset_base_dir, f) for f in os.listdir(dataset_base_dir)]
    return listOfFiles

#the code should become more general for the ImageDataset class.
class ImageDataset(Dataset):
    def __init__(self, config):
        path = os.path.join(config.data.base_dir, config.data.dataset)
        res_x, res_y = config.data.shape[1], config.data.shape[2]
        if config.data.crop:
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            croper = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Lambda(croper),
                transforms.ToPILImage(),
                transforms.Resize(size=(res_x, res_y),  interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize(size=(res_x, res_y))])
            
        self.image_paths = load_file_paths(path)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


@utils.register_lightning_datamodule(name='image')
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split = config.data.split

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.eval.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.eval.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None):
        if self.config.data.dataset == 'mnist':
            data = MNISTDataset(self.config)
        else:
            data = ImageDataset(self.config)
        
        print(len(data))
        l=len(data)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), l - int(self.split[0]*l) - int(self.split[1]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
