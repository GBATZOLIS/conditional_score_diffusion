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
  config.training.batch_size = 128
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  #----- to be removed -----
  training.n_iters = 2400001
  #------              --------
  training.visualization_callback = 'conditional_haar_multiscale'
  training.show_evolution = False
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
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
  evaluate.workers = 4*training.gpus
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
  data.base_dir = 'datasets' #'/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/datasets'
  data.dataset = 'celebA-HQ-160'
  data.use_data_mean = False
  data.datamodule = 'Haar_PKLDataset'
  data.coordinate_space = 'haar'
  data.map = 'approx to detail'
  data.create_dataset = False
  data.target_resolution = 160 #this should remain constant for an experiment
  data.image_size = 40 #we vary this for training on different resolutions
  data.effective_image_size = data.image_size//2 #because we squeeze the 2X HR image to concatenate it with the LR image.
  data.scale = 2 #we address 4x super-resolution directly
  data.centered = False
  data.level = math.log(data.target_resolution // data.image_size, 2)
  data.range_x = [-2**data.level, 2**data.level]
  data.range_y = [0, 2**(data.level+1)]
  data.shape_x = [9, data.image_size // 2, data.image_size // 2]
  data.shape_y = [3, data.image_size // 2, data.image_size // 2]
  data.num_channels = data.shape_x[0]+ data.shape_y[0] #because of the squeezing and the concatenation -> important information for construction of the score based model.

  #data augmentation settings
  data.use_flip = True
  data.use_rot = False
  data.use_crop = True
  data.uniform_dequantization = False
  

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = '/home/gb511/saved_checkpoints/celebA/sequential/haar/20to40/checkpoints/epoch=710-step=904391.ckpt'
  model.num_scales = 1000
  
  #SIGMA INFORMATION FOR THE VE SDE
  model.reach_target_steps = 8000
  model.sigma_max_x = np.sqrt(np.prod(data.shape_x))*(data.range_x[1] - data.range_x[0])
  model.sigma_max_y = np.sqrt(np.prod(data.shape_y))*(data.range_y[1] - data.range_y[0])
  model.sigma_max_y_target = model.sigma_max_y/2
  model.sigma_min_x = 5e-3
  model.sigma_min_y = 5e-3
  model.sigma_min_y_target = 5e-3

  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'


  model.name = 'ddpm_paired'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 96
  model.ch_mult = (1, 1, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (20, 10, 5)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.input_channels = data.num_channels
  model.output_channels = data.num_channels

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