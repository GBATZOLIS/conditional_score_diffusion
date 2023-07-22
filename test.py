import run_lib
import torch
import pandas as pd
import numpy as np
import pickle
import os
from lightning_modules.VAE import VAE
from utils import fix_rds_path
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from lightning_data_modules.ImageDatasets import CelebAAnnotatedDataset
from lightning_modules.utils import create_lightning_module
from utils import fix_rds_path
from sklearn.linear_model import LogisticRegression

with open('/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/paper/pretrained/celebA_64/only_ResNetEncoder_VAE_KLweight_0.01/config.pkl', 'rb') as f:
  config = pickle.load(f)
# config.model.time_conditional = True
# model = create_lightning_module(config)
# pl_module = model.load_from_checkpoint(fix_rds_path(config.model.checkpoint_path), config=config)

# get home dir
home = os.path.expanduser('~')
config.data.base_dir = f'{home}/rds_work/datasets/'
config.data.dataset = 'celebA-HQ-160'
config.data.attributes = ['Male']
train_dataset = CelebAAnnotatedDataset(config, phase='train')
val_dataset = CelebAAnnotatedDataset(config, phase='val')
test_dataset = CelebAAnnotatedDataset(config, phase='test')
train_dataloader = DataLoader(train_dataset, batch_size=int(16), shuffle=False, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=int(16), shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=int(16), shuffle=False, num_workers=0)
# get a batch from each 
data_iter = iter(train_dataloader)
train_data = next(data_iter)
data_iter = iter(val_dataloader)
val_data = next(data_iter)
data_iter = iter(test_dataloader)
test_data = next(data_iter)

#plot a bach images as grid with labels as captions
import matplotlib.pyplot as plt
import torchvision.utils as vutils
def show_batch(batch, title):
    im = vutils.make_grid(batch[0], nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.title(title)
    plt.show()

train_labels = train_data[1].flatten().numpy().reshape(4, 4)
print(train_labels)
show_batch(train_data, 'Train Batch')
val_labels = val_data[1].flatten().numpy().reshape(4, 4)
print(val_labels)
show_batch(val_data, 'Val Batch')
test_labels = test_data[1].flatten().numpy().reshape(4, 4)
print(test_labels)
show_batch(test_data, 'Test Batch')

# TEST IS SAME AS VAL