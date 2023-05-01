import ml_collections
import torch
import math
import numpy as np
import configs.jan.celebA.default as default


def get_config():
  config = default.get_config()

  data = config.data
  data.base_dir = '/store/CIA/js2164/data/'
  data.dataset = 'celeba'
  return config