# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Config file for synthetic dataset."""

import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta

from configs.dimension_estimation.ksphere.vesde import get_config as veconfig

def get_config():
  config = veconfig()

  # data
  data = config.data
  data.n_spheres = 1
  data.manifold_dim=50
  
  data.embedding_type = 'random_isometry'
  data.angle_std = -1

  # dim_estimation
  config.dim_estimation.num_datapoints = 1000

  # model
  model = config.model
  model.sigma_min = 1e-2
  model.sigma_max = 4
  model.hidden_layers = 5
  model.hidden_nodes = 2048

  #logging
  logging = config.logging
  logging.log_path = f'logs/ksphere/dim_{data.manifold_dim}/n_{data.n_spheres}/{data.embedding_type}/'
  logging.log_name = f'uniform_random'

  return config
