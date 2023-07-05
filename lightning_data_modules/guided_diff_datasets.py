import math
import random

from PIL import Image
import blobfile as bf
#from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import multiprocessing
import torch
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
from tqdm import tqdm
import pytorch_lightning as pl
import os
from . import utils

class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, args, train):
        super().__init__(root=args.data_dir, train=train, download=True)
        transforms_list = [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # tranform to [-1, 1] range
        ]
        self.transform_my = transforms.Compose(transforms_list)
        self.return_labels = args.class_cond
    
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x = self.transform_my(x)
        if self.return_labels:
            return x, y
        else:
            return x

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        #DataLoader arguments
        self.train_workers = args.workers
        self.val_workers = args.workers
        self.test_workers = args.workers

        self.train_batch = args.batch_size
        self.val_batch = args.batch_size
        self.test_batch = args.batch_size

    def setup(self, stage=None):
        train_dataset = CIFAR10Dataset(self.args, train=True)
        self.train_data, self.valid_data = torch.utils.data.random_split(train_dataset, [45000, 5000])
        self.test_data = CIFAR10Dataset(self.args, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True, pin_memory=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=False, pin_memory=True) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers, shuffle=False, pin_memory=True) 


# ... existing functions and classes (load_data, ImageDataset, etc.) ...
@utils.register_lightning_datamodule(name='guided_diff_dataset')
class ImageDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data.base_dir
        self.dataset = config.data.dataset
        self.percentage_use = 100 #percentage of the original dataset to be used
        self.batch_size = config.training.batch_size
        self.image_size = config.data.image_size
        self.class_cond = config.data.class_cond
        self.random_crop = config.data.random_crop
        self.random_flip = config.data.random_flip
        self.num_workers = config.training.workers

    def setup(self, stage=None):

        # Load all image files
        all_files = _list_image_files_recursively(os.path.join(self.data_dir, self.dataset))
        print('%s contains %d datapoints.' % (self.dataset, len(all_files)))
        all_files = random.sample(all_files, int(self.percentage_use/100*len(all_files)))  # select the percentage of files to use
        all_images = load_images(all_files, random_crop=self.random_crop, resolution=self.image_size)

        classes = None
        if self.class_cond:
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        # Create dataset
        full_dataset = ImageDataset(
            self.image_size,
            all_images,
            classes=classes,
            random_crop=self.random_crop,
            random_flip=self.random_flip,
        )

        # Ensure reproducibility
        torch.manual_seed(42)

        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)
    all_files = random.sample(all_files, 100000) #select 1000 to make the loading and the debugging faster
    all_images = load_images(all_files, random_crop, image_size)

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    dataset = ImageDataset(
        image_size,
        all_images,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_image(params):
    path, random_crop, resolution = params
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        if random_crop:
            arr = random_crop_arr(pil_image, resolution)
        else:
            arr = center_crop_arr(pil_image, resolution)
        arr = arr.astype(np.float32) / 127.5 - 1  # Normalization
    return arr

def load_images(image_paths, random_crop, resolution):
    with multiprocessing.Pool() as pool:
        params = [(path, random_crop, resolution) for path in image_paths]
        images = list(tqdm(pool.imap(load_image, params), total=len(image_paths)))
    return images


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        images,
        classes=None,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = images
        self.local_classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        arr = self.local_images[idx]

        if self.random_flip and random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1])

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]