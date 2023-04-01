import ml_collections
import torch
import math
import numpy as np
from configs.jan.default import get_default_configs


def get_config():
  config = get_default_configs()

  config.training.sde = 'snrsde'
  
  return config