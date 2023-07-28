import run_lib
import torch
import pandas as pd
import numpy as np
import pickle
import os
from lightning_modules.VAE import VAE
from dim_reduction import inspect_VAE
from configs.utils import fix_rds_path
from torch.utils.data import DataLoader
from lightning_data_modules.ImageDatasets import CelebAAnnotatedDataset
from lightning_data_modules.guided_diff_datasets import ImageDataModule
from configs.utils import fix_config
from matplotlib import pyplot as plt

# load config
path = '/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/gd_ffhq/only_encoder_ddpm_plus_smld_VAE_KLweight_0.01_DiffDecoders_continuous_prior/config.pkl'
with open(path, 'rb') as f:
  config = pickle.load(f)
config = fix_config(config)
config.data.dataset = 'celebA-HQ-160'
config.data.attributes = ['Male']
config.data.normalization_mode = 'gd'
dataset = CelebAAnnotatedDataset(config, phase='train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
data_iter = iter(dataloader)
train_data = next(data_iter)
test_data = next(data_iter)

dataset.normalization_mode
x = train_data[0][0]
# plot it
plt.imshow(x.permute(1,2,0))
plt.show()