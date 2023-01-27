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

def get_config():
  config = ml_collections.ConfigDict()

  #logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/dimension_detection/experiments/fixed_gaussians_manifold/' 
  logging.log_name = '100_deep'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  config.training = training = ml_collections.ConfigDict()
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = None if training.gpus <= 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.lightning_module = 'base' 
  config.training.batch_size = 128
  training.workers = 4
  training.num_epochs = 10000
  training.n_iters = 2500000
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.eval_freq = 2500
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = True #look more for that setting
  training.sde = 'vesde'
  training.visualization_callback = 'base'
  training.show_evolution = False

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = 128
  validation.workers = 4

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.callback = None
  evaluate.workers = 4
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 128
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.datamodule = 'Synthetic'
  data.dataset_type = 'FixedGaussiansManifold'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.data_samples = 250000
  data.image_size = 32
  data.effective_image_size = data.image_size
  data.centered = False
  data.use_data_mean = False
  data.num_gaussians = 100
  data.std_range = [1, 5]
  data.mixtures = 4
  data.return_labels = False
  data.return_mixtures = False #whether to return the mixture class of each point in the mixture.
  data.shape = [1, data.image_size, data.image_size]
  data.num_channels = 1
  
  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None #'/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/dimension_detection/experiments/fixed_gaussians_manifold/%s/checkpoints/best/last.ckpt' % logging.log_name
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  model.name = 'ddpm' #'ncsnpp'
  model.input_channels = model.output_channels = data.num_channels
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 256
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16, 8)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
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
