import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
import os
from configs.dimension_estimation.styleGAN.style_gan_base import get_config as get_base_config

# get home directory
from pathlib import Path
home = str(Path.home())

def get_config():
  config = get_base_config()

  # data
  data = config.data 
  data.latent_dim = 2

  #logging
  logging = config.logging 
  logging.log_name = str(data.latent_dim)
  
  #model 
  model = config.model
  model.checkpoint_path = f'{home}/rds_work/projects/dimension_detection/experiments/style_gan/{logging.log_name}/checkpoints/best/last.ckpt'

  return config
