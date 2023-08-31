import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision  import transforms, datasets
import torch
import PIL.Image as Image
from . import utils
import os
import glob
import pickle
from tqdm import tqdm

class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, config, train):
        super().__init__(root=config.data.base_dir, train=train, download=True)
        transforms_list=[transforms.ToTensor()]
        self.transform_my = transforms.Compose(transforms_list)
        self.return_labels = config.data.return_labels
    
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x = self.transform_my(x)
        if self.return_labels:
            return x, y
        else:
            return x

@utils.register_lightning_datamodule(name='cifar10')
class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None):
        train_dataset = CIFAR10Dataset(self.config, train=True)
        torch.manual_seed(22)
        self.train_data, self.valid_data = torch.utils.data.random_split(train_dataset, [45000, 5000])
        self.test_data = CIFAR10Dataset(self.config, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=False) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers, shuffle=False) 



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
    listOfFiles = [os.path.join(dataset_base_dir, f) for f in os.listdir(dataset_base_dir) if os.path.isfile(os.path.join(dataset_base_dir, f))]
    return listOfFiles

class MNISTLatentDataset(MNISTDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
        dataset_path = os.path.join(config.data.base_dir, 'MNIST_latents')
        #from janutils.models.autoencoder import AutoEncoder
        #self.autoencoder = AutoEncoder.load_from_checkpoint(config.data.encoder_path)
        if not os.path.exists(dataset_path):
            from janutils.models.autoencoder import AutoEncoder
            autoencoder = AutoEncoder.load_from_checkpoint(config.data.encoder_path)
            autoencoder.to(config.device)
            latents = []
            print('Generating latents')
            for index in tqdm(range(super().__len__())):

                if self.return_labels:
                    x, _ = super().__getitem__(index)
                else:
                    x = super().__getitem__(index)
                x = x.to(config.device)
                z = autoencoder.encode(x.unsqueeze(0)).squeeze().detach()
                latents.append(z)
            self.latents = torch.stack(latents)
            os.makedirs(dataset_path)
            with open(os.path.join(dataset_path, 'latents'), 'wb') as f:
                pickle.dump(self.latents, f)
        else:
            print('Loading latents')
            with open(os.path.join(dataset_path, 'latents'), 'rb') as f:
                self.latents = pickle.load(f)


    def __getitem__(self, index):
        
        #z = self.latents(index)
        if self.return_labels:
            x, y = super().__getitem__(index)
        else:
            x = super().__getitem__(index)
        
        z = self.latents[index]
        #z = self.autoencoder.encode(x.unsqueeze(0)).squeeze()
        return x, z

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
        elif self.config.data.dataset == 'mnist_latent':
            data = MNISTLatentDataset(self.config)
        else:
            data = ImageDataset(self.config)
        
        print(len(data))
        l=len(data)
        torch.manual_seed(0)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), l - int(self.split[0]*l) - int(self.split[1]*l)]) 
        torch.manual_seed(torch.initial_seed())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
