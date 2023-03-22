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
"""Config file for reproducing the results of DDPM on bedrooms."""

import ml_collections
import torch
import math
import numpy as np

def get_config():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'haar_conditional_decreasing_variance'
  config.training.batch_size = 25
  training.num_nodes = 1
  training.gpus = 2
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4
  training.n_iters = 2400001
  training.visualization_callback = 'conditional_haar_multiscale'
  training.show_evolution = False
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = False
  training.reduce_mean = True 
  training.sde = 'vesde'

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'conditional_reverse_diffusion'
  sampling.corrector = 'conditional_langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 25
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = 'datasets'
  data.dataset = 'celebA'
  data.use_data_mean = False
  data.datamodule = 'haar_multiscale'
  data.create_dataset = False
  data.split = [0.925, 0.05, 0.025]
  data.target_resolution = 160 #this should remain constant for an experiment
  data.image_size = 160 #we vary this for training on different resolutions
  data.level = math.log(data.target_resolution // data.image_size, 2)
  data.effective_image_size = data.image_size // 2 #actual image size after preprocessing. Divided by two when using haar tranform.
  data.max_haar_depth = 2 #maximum depth of multi-level haar tranform -> 1+data.max_haar_depth resolution levels.
  data.centered = False
  data.random_flip = False
  data.uniform_dequantization = False
  data.num_channels = 12 #because of the haar tranform we have 12 channels.
  data.shape_x = [9, data.effective_image_size, data.effective_image_size]

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = 'experiments/beatSRFLOW/160/37M_450sigma_max_50_batch/lightning_logs/version_1/checkpoints/epoch=59-step=219359.ckpt'
  model.num_scales = 1000
  model.sigma_max_x = 450 #change it to 480 for future experiments (not a big difference)
  #we do not want to perturb y a lot. 
  #A slight perturbation will result in better approximation of the conditional time-dependent score.
  model.sigma_max_y = model.sigma_max_x
  #-------The three subsequent settings configure the reduction schedule of sigma_max_y
  model.reduction = 'inverse_exponentional' #choices=['linear', 'inverse_exponentional']
  model.reach_target_in_epochs = 64 #matters only if reduction = linear
  model.starting_transition_iterations = 8000 #matters only if reduction = inverse exponential

  model.sigma_min_x = 0.01
  model.sigma_min_y = 0.01

  model.beta_min = 0.1
  # We use an adjusted beta max 
  # because the range is doubled in each level starting from the first level
  model.beta_max = 20. + 4*(data.level+1)*np.log(2) #take the doubling value range into consideration.
  model.dropout = 0.1
  model.embedding_type = 'fourier'


  model.name = 'ddpm'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2)
  model.num_res_blocks = 3
  model.attn_resolutions = (20, 10)
  model.resamp_with_conv = True
  model.conditional = True
  model.conv_size = 3

  # optimization
  config.optim = optim = ml_collections.ConfigDict()

  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


  return config