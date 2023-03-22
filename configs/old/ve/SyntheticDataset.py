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

def get_config():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 500
  training.workers = 4
  training.num_epochs = 10000
  training.n_iters = 500000
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.eval_freq = 2500
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False #look more for that setting
  training.sde = 'vesde'

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = 500
  validation.workers = 4

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 512
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'Synthetic'
  data.dataset_type = 'GaussianBubbles'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.data_samples = 50000
  data.mixtures = 4
  data.return_mixtures = False #whether to return the mixture class of each point in the mixture.
  data.shape = [2]
  data.dim = 2
  data.num_channels = 0 
  

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 4
  model.sigma_min = 0.01
  model.beta_min = 0.1
  # We use an adjusted beta max 
  # because the range is doubled in each level starting from the first level
  model.beta_max = 25 #take the value range into consideration - consider the final perturbation kernels.

  model.name = 'fcn'
  model.state_size = data.dim
  model.hidden_layers = 3
  model.hidden_nodes = 64
  model.dropout = 0.25
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999


  # optimization
  config.optim = optim = ml_collections.ConfigDict()
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
