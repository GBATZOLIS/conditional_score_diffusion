import ml_collections
import torch
import math
import numpy as np
from configs.jan.celebA.default import get_default_configs


def get_config():
  config = get_default_configs()

  data = config.data
  data.base_dir = '/rds/user/js2164/hpc-work/data'
  data.dataset = 'celeba'
  return config