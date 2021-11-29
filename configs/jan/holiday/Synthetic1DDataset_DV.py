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
  training.lightning_module = 'conditional'
  training.conditioning_approach = 'ours_NDV'
  training.gpus = 1
  training.num_nodes = 1
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4
  training.batch_size = 500
  training.num_epochs = 10000
  training.n_iters = 100000000
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.eval_freq = 2500
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = False #look more for that setting
  training.sde = 'vesde'
  # callbacks
  training.visualization_callback = 'Conditional1DVisualization'
  training.show_evolution = False



  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = 500
  validation.workers = 4

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'conditional_reverse_diffusion'
  sampling.corrector = 'conditional_none'
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
  data.datamodule = 'Conditional1DSynthetic'
  data.dataset_type = 'Conditional1DGaussians'
  data.use_data_mean = False # What is this?
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.data_samples = int(1e5)
  data.mixtures = 2
  data.y_min = 0
  data.y_max = 2
  data.return_distances = True 
  data.shape_x = [1]
  data.shape_y = [1]
  data.dim = 1
  data.num_channels = 0 
  

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None
  #SIGMA INFORMATION FOR THE VE SDE
  model.reach_target_steps = 500000
  model.sigma_max_x = np.sqrt(np.prod(data.shape_x))
  model.sigma_max_y = 0.05#np.sqrt(np.prod(data.shape_y))
  model.sigma_max_y_target = model.sigma_max_y#np.sqrt(np.prod(data.shape_y))
  model.sigma_min_x = 0.01#5e-3
  model.sigma_min_y = 0.01#5e-3
  model.sigma_min_y_target = 0.01#5e-3
  #model.sigma_max = 4
  #model.sigma_min = 0.01
  model.beta_min = 0.1
  # We use an adjusted beta max 
  # because the range is doubled in each level starting from the first level
  model.beta_max = 25 #take the value range into consideration - consider the final perturbation kernels.

  model.name = 'fcn_joint'
  model.state_size = data.dim
  model.hidden_layers = 3
  model.hidden_nodes = 64
  model.dropout = 0.0
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