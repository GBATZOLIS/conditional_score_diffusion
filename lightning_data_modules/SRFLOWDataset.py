import os
# import subprocess
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import time
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode, rgb_to_grayscale
from iunets.layers import InvertibleDownsampling2D

import pickle
import pytorch_lightning as pl
from . import utils

def get_exact_paths(config, phase):
    if config.data.dataset == 'DF2K':  
        if phase == 'train':
            LQ_file = 'DF2K-tr_X4.pklv4'
            GT_file = 'DF2K-tr.pklv4'
        elif phase == 'val':
            LQ_file = 'DIV2K-va_X4.pklv4'
            GT_file = 'DIV2K-va.pklv4'
        elif phase == 'test':
            LQ_file = 'DIV2K-teFullMod8_X4.pklv4'
            GT_file = 'DIV2K-teFullMod8.pklv4'
        else:
            return NotImplementedError('%s is not supported.' % phase)
    elif config.data.dataset == 'celebA-HQ-160':
        if phase == 'train':
            LQ_file = 'CelebAHq_160_MBic_tr_X8.pklv4'
            GT_file = 'CelebAHq_160_MBic_tr.pklv4'
        elif phase == 'val':
            LQ_file = 'CelebAHq_160_MBic_va_X8.pklv4'
            GT_file = 'CelebAHq_160_MBic_va.pklv4'
        elif phase == 'test':
            LQ_file = 'CelebAHq_160_MBic_va_X8.pklv4'
            GT_file = 'CelebAHq_160_MBic_va.pklv4'
            
        else:
            return NotImplementedError('%s is not supported.' % phase)
    else:
        return NotImplementedError('%s is not supported.' % config.data.dataset)
    
    full_path_LQ = os.path.join(config.data.base_dir, config.data.dataset, LQ_file)
    full_path_GT = os.path.join(config.data.base_dir, config.data.dataset, GT_file)

    return {'LQ':full_path_LQ, 'GT':full_path_GT}

class LRHR_PKLDataset(data.Dataset):
    def __init__(self, config, phase):
        super(LRHR_PKLDataset, self).__init__()
        self.target_size = config.data.target_resolution #overall target size
        self.crop_size = config.data.image_size #target image size for this scale
        self.scale = config.data.scale

        hr_file_path = get_exact_paths(config, phase)['GT']
        lr_file_path = get_exact_paths(config, phase)['LQ']

        self.use_flip = config.data.use_flip
        self.use_rot = config.data.use_rot
        self.use_crop = config.data.use_crop
        self.upscale_lr = config.data.upscale_lr

        t = time.time()
        self.lr_images = self.load_pkls(lr_file_path, n_max=int(1e9))
        self.hr_images = self.load_pkls(hr_file_path, n_max=int(1e9))

        min_val_hr = np.min([i.min() for i in self.hr_images[:20]])
        max_val_hr = np.max([i.max() for i in self.hr_images[:20]])

        min_val_lr = np.min([i.min() for i in self.lr_images[:20]])
        max_val_lr = np.max([i.max() for i in self.lr_images[:20]])

        t = time.time() - t
        print("Loaded {} HR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.hr_images), min_val_hr, max_val_hr, t, hr_file_path))
        print("Loaded {} LR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.lr_images), min_val_lr, max_val_lr, t, lr_file_path))

    def load_pkls(self, path, n_max):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        images = images[:n_max]
        images = [np.transpose(image, [2, 0, 1]) for image in images]
        return images

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):

        hr = self.hr_images[item]
        lr = self.lr_images[item]

        if self.scale == hr.shape[1] // lr.shape[1]:
            if self.use_crop:
                hr, lr = random_crop(hr, lr, self.crop_size, self.scale)

            #if self.center_crop_hr_size:
            #    hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size // self.scale)

            if self.use_flip:
                hr, lr = random_flip(hr, lr)

            if self.use_rot:
                hr, lr = random_rotation(hr, lr)

            hr = hr / 255.0
            lr = lr / 255.0

            hr = torch.Tensor(hr)
            lr = torch.Tensor(lr)

            if self.upscale_lr:
                resize_to_hr = Resize(self.crop_size, interpolation=InterpolationMode.NEAREST)
                lr = resize_to_hr(lr)
        
        elif self.scale < hr.shape[1] // lr.shape[1]:
            if self.crop_size == self.scale * lr.shape[1]:
                a_priori_scale = hr.shape[1] // lr.shape[1]
                hr, lr = random_crop(hr, lr, self.target_size, a_priori_scale)
                
                #convert hr, lr to tensors
                hr, lr = hr / 255.0, lr / 255.0
                hr, lr = torch.Tensor(hr), torch.Tensor(lr)

                #resize hr to the right scale
                resize = Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC)
                hr = resize(hr)
            else:
                size_hr_x, size_hr_y = hr.shape[1], hr.shape[2]
                start_x_hr = np.random.randint(low=0, high=(size_hr_x - self.target_size) + 1) if size_hr_x > self.target_size else 0
                start_y_hr = np.random.randint(low=0, high=(size_hr_y - self.target_size) + 1) if size_hr_y > self.target_size else 0
                hr = hr[:, start_x_hr:start_x_hr + self.target_size, start_y_hr:start_y_hr + self.target_size]
                
                #convert hr to tensor
                hr = hr / 255.0
                hr = torch.Tensor(hr)
                
                #resize hr_patch to the crop size (target image size at that scale).
                resize = Resize(self.crop_size, interpolation=InterpolationMode.BICUBIC)
                hr = resize(hr)
                
                #resize hr to lr using the provided scale
                resize = Resize(self.crop_size//self.scale, interpolation=InterpolationMode.BICUBIC)
                lr = resize(hr)

        return lr, hr

class Haar_PKLDataset(data.Dataset):
    def __init__(self, config, phase):
        super(Haar_PKLDataset, self).__init__()
        self.haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)

        self.target_size = config.data.target_resolution #overall target size
        self.crop_size = config.data.image_size #target image size for this scale
        self.level = config.data.level
        self.scale = config.data.scale
        self.map = config.data.map
        self.random_scale_list = [1]

        hr_file_path = get_exact_paths(config, phase)['GT']
        lr_file_path = get_exact_paths(config, phase)['LQ']

        self.use_flip = config.data.use_flip
        self.use_rot = config.data.use_rot
        self.use_crop = config.data.use_crop
        #self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        #n_max = opt["n_max"] if "n_max" in opt.keys() else int(1e8)
        self.lr_images = self.load_pkls(lr_file_path, n_max=int(1e9))
        self.hr_images = self.load_pkls(hr_file_path, n_max=int(1e9))

    def load_pkls(self, path, n_max):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        images = images[:n_max]
        images = [np.transpose(image, [2, 0, 1]) for image in images]
        return images

    def haar_forward(self, x):
        x = self.haar_transform(x)
        x = permute_channels(x)
        return x

    def multi_level_haar_forward(self, x, level):
        approx_cf = x.unsqueeze(0)
        for _ in range(int(level)):
            haar = self.haar_forward(approx_cf)
            approx_cf, detail_cf = haar[:,:3,::], haar[:,3:,::]
        return torch.squeeze(approx_cf, 0), torch.squeeze(detail_cf, 0)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        hr = self.hr_images[item]
        lr = self.lr_images[item]

        if self.use_crop:   
            hr, lr = random_crop(hr, lr, self.target_size, hr.shape[1] // lr.shape[1])
        
        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = hr / 255.0
        lr = lr / 255.0

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        approx_cf, detail_cf = self.multi_level_haar_forward(hr, level=self.level+1)

        if self.map == 'approx to detail':
            return approx_cf, detail_cf
        elif self.map == 'bicubic to approx':
            return  lr, approx_cf
        elif self.map == 'bicubic to haar':
            return lr, torch.cat((approx_cf, detail_cf), dim=1)
        else:
            raise NotImplementedError('Mapping <<%s>> is not supported' % self.map)

class General_PKLDataset(data.Dataset):
    def __init__(self, config, phase):
        super(General_PKLDataset, self).__init__()
        self.image_size = config.data.image_size #target image size for this scale
        self.task = config.data.task

        self.scale = config.data.scale #used for SR
        self.mask_coverage = config.data.mask_coverage #used for inpainting
        self.use_flip = config.data.use_flip

        hr_file_path = get_exact_paths(config, phase)['GT']
        self.hr_images = self.load_pkls(hr_file_path, n_max=int(1e9))

    def load_pkls(self, path, n_max):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        images = images[:n_max]
        images = [np.transpose(image, [2, 0, 1]) for image in images]
        return images

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        hr = self.hr_images[item]

        if self.use_flip:
            random_choice = np.random.choice([True, False])
            hr = hr if random_choice else np.flip(hr, 2).copy()
        
        hr = hr / 255.0
        hr = torch.Tensor(hr)

        resize_to_target = Resize(self.image_size, interpolation=InterpolationMode.BICUBIC)
        hr = resize_to_target(hr)

        if self.task == 'super-resolution':
            resize_to_lr = Resize(self.image_size//self.scale, interpolation=InterpolationMode.BICUBIC)
            lr = resize_to_lr(hr)
            resize_to_hr = Resize(self.image_size, interpolation=InterpolationMode.NEAREST)
            lr_nn = resize_to_hr(lr)
            return lr_nn, hr

        elif self.task == 'colorization':
            gray = rgb_to_grayscale(hr)
            return gray, hr

        elif self.task == 'inpainting':
            masked_img = hr.clone()
            mask_size = int(np.sqrt(self.mask_coverage * hr.shape[1] * hr.shape[2]))
            size_x, size_y = masked_img.shape[1], masked_img.shape[2]
            start_x = np.random.randint(low=0, high=(size_x - mask_size) + 1) if size_x > mask_size else 0
            start_y = np.random.randint(low=0, high=(size_y - mask_size) + 1) if size_y > mask_size else 0
            masked_img[:, start_x:start_x + mask_size, start_y:start_y + mask_size] = 0.
            return masked_img, hr


        
def permute_channels(haar_image, forward=True):
    permuted_image = torch.zeros_like(haar_image)
    if forward:
        for i in range(4):
            if i == 0:
                k = 1
            elif i == 1:
                k = 0
            else:
                k = i
            for j in range(3):
                permuted_image[:, 3*k+j, :, :] = haar_image[:, 4*j+i, :, :]
    else:
        for i in range(4):
            if i == 0:
                k = 1
            elif i == 1:
                k = 0
            else:
                k = i
                
            for j in range(3):
                permuted_image[:,4*j+k,:,:] = haar_image[:, 3*i+j, :, :]

    return permuted_image

def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg


def random_crop(hr, lr, size_hr, scale):
    if size_hr == hr.shape[1] and size_hr == hr.shape[2]:
        return hr, lr
    else:
        size_lr = size_hr // scale

        size_lr_x = lr.shape[1]
        size_lr_y = lr.shape[2]

        start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
        start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

        # LR Patch
        lr_patch = lr[:, start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]

        # HR Patch
        start_x_hr = start_x_lr * scale
        start_y_hr = start_y_lr * scale
        hr_patch = hr[:, start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr]

        return hr_patch, lr_patch      

def center_crop(img, size):
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, border:-border, border:-border]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]


@utils.register_lightning_datamodule(name='LRHR_PKLDataset')
class PairedDataModule(pl.LightningDataModule):
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
        self.train_dataset = LRHR_PKLDataset(self.config, phase='train')
        self.val_dataset = LRHR_PKLDataset(self.config, phase='val')
        self.test_dataset = LRHR_PKLDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

@utils.register_lightning_datamodule(name='Haar_PKLDataset')
class PairedDataModule(pl.LightningDataModule):
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
        self.train_dataset = Haar_PKLDataset(self.config, phase='train')
        self.val_dataset = Haar_PKLDataset(self.config, phase='val')
        self.test_dataset = Haar_PKLDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

@utils.register_lightning_datamodule(name='General_PKLDataset')
class PairedDataModule(pl.LightningDataModule):
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
        self.train_dataset = General_PKLDataset(self.config, phase='train')
        self.val_dataset = General_PKLDataset(self.config, phase='val')
        self.test_dataset = General_PKLDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

