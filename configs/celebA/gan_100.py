import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
#from configs.celebA.ddpm import get_config as default_celeba
from configs.jan.celebA.default import get_config as default_celeba

def get_config():
  config = default_celeba()

  # data
  config.data = data = ml_collections.ConfigDict()
  data.datamodule = 'Gan'
  data.data_path = '/rds/user/js2164/hpc-work/data/gan_data' #'datasets' ->put the directory where you have the dataset: /datasets/. It will load .../datasets/celebA
  data.latent_dim = 100
  data.use_data_mean = False
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 64
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.centered = False
  data.random_flip = False
  data.crop = True
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  #logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = 'logs/celebA/'#'/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/dimension_detection/experiments/celebA/'
  logging.log_name = f'gan_{data.latent_dim}'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  return config