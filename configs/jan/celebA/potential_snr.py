import ml_collections
import torch
import math
import numpy as np
import configs.jan.celebA.potential as base

def get_config():
  config = base.get_config()
  config.training.sde = 'snrsde'
  return config