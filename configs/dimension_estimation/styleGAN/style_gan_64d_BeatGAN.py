import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
import os
from configs.dimension_estimation.styleGAN.style_gan_BeatGAN import get_config as get_base_config

# get home directory
from pathlib import Path
home = str(Path.home())

def get_config():
  config = get_base_config()

  # data
  data = config.data 
  data.latent_dim = 64

  #model
  model = config.model
  model.embed_channels = data.latent_dim
  model.checkpoint_path = None
  model.dropout = 0.3

  #logging
  logging = config.logging 
  logging.log_name = str(data.latent_dim) + "_BeatGANsUNetModel" + f"_dropout_{model.dropout}" 

  return config
