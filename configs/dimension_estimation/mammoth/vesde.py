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

from configs.jan.default import get_default_configs

def get_config():
  config = get_default_configs()

  #logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = 'logs/mammoth/' #'logs/ksphere/'
  logging.log_name = 'new'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)
  logging.svd_frequency = 1000
  logging.save_svd = False

  # training
  training = config.training
  training.mode = 'train'
  training.gpus = 1
  training.lightning_module = 'base' 
  training.batch_size = 500
  training.num_epochs = int(1e20)
  training.n_iters = int(1e20)
  training.likelihood_weighting = True
  training.continuous = True
  training.sde = 'vesde'
  # callbacks
  training.visualization_callback = ['ScoreSpectrumVisualization']
  training.show_evolution = False 

  # validation
  validation = config.validation
  validation.batch_size = 500

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.15 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

   # data
  config.data = data = ml_collections.ConfigDict()
  data.datamodule = 'Mammoth'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.data_samples = 50000
  data.use_data_mean=False 

  data.ambient_dim=100
  data.noise_std = 0
  data.embedding_type = 'random_isometry'
  data.dim = data.ambient_dim
  data.num_channels = 0 
  data.shape = [data.dim]
  
  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None #'/home/gb511/projects/manifold_dimension/ksphere/ve/checkpoints/last.ckpt'
  model.sigma_max = 4 #data.manifold_dim * math.sqrt(2)
  model.sigma_min = 1e-2

  model.name = 'fcn'
  model.state_size = data.dim
  model.hidden_layers = 5
  model.hidden_nodes = 2048
  model.dropout = 0.0
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999

  # optimization
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


  return config
