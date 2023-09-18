import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
import os
from configs.dimension_estimation.styleGAN.style_gan_base import get_config as get_base_config


def get_config():
  config = get_base_config()

  #logging
  logging = config.logging 
  logging.log_path = '~/rds_work/projects/dimension_detection/experiments/style_gan/'
  logging.log_name = '64'
  
  # data
  data = config.data 
  data.latent_dim = 64

  return config
