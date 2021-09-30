import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
import PIL.Image as Image


def load_dataset(dataset_name, resolution=64, n_channels=3):
    # Define data transform
    transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Resize(size=(resolution, resolution)),
                    transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels)]
        )
    if dataset_name == 'MNIST':
        dataset = get_MNIST(transform)
    if dataset_name == 'CelebA':
        dataset = get_CelebA(transform) 
    if dataset_name == 'CIFAR10':
        dataset = get_CIFAR10(transform)
    
    return dataset

def get_Gaussian(n=int(1e5)):
    return generate_8gaussians(n)

def get_MNIST(transform):
    PATH='/store/CIA/js2164/data/mnist'
    dataset = datasets.MNIST(root=PATH, train=True, 
                                download=True, transform=transform)
    return dataset

def get_CIFAR10(transform):
    PATH='/store/CIA/js2164/data/cifar10'
    dataset = datasets.CIFAR10(root=PATH, train=True, 
                                download=True, transform=transform)
    return dataset

def get_CelebA(transform=None, crop =False, resolution=128):

    PATH='/store/CIA/js2164/data/celeba'
    # If you want to crop images to remove background redefine transform (64x64 only)
    if crop:
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        croper = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(croper),
            transforms.ToPILImage(),
            transforms.Resize(size=(resolution, resolution),  interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    # Create pytorch dataloader

    celeba_data = datasets.ImageFolder(PATH, transform=transform)

    return celeba_data


class GaussianMixture2D(Dataset):
  def __init__(self, n, mus, covs):
    self.n = n
    self.mus = mus
    self.covs = covs
    r = []
    for mu, cov in zip(mus, covs):
      x = np.random.multivariate_normal(mu, cov, n)
      x = torch.tensor(x, dtype=torch.float32)
      r.append(x)
    self.X = torch.cat(r, dim=0)
    self.X = self.X[torch.randperm(self.X.size(0))]

  def __len__(self):
    return len(self.mus) * self.n

  def __getitem__(self, idx):
    return (self.X[idx])

def generate_8gaussians(n, std=.01):    
    centers = [
     (1, 0),
     (-1, 0),
     (0, 1),
     (0, -1),
     (1. / np.sqrt(2), 1. / np.sqrt(2)),
     (1. / np.sqrt(2), -1. / np.sqrt(2)),
     (-1. / np.sqrt(2), 1. / np.sqrt(2)),
     (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    covs = [std * np.eye(2) for i in centers]
    data = GaussianMixture2D(n, centers, covs)
    return data

