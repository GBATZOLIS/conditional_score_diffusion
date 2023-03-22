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

def get_config():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
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
  sampling.corrector = 'langevin'
  sampling.probability_flow = False
  sampling.snr = 0.15
  sampling.n_steps_each = 1
  sampling.noise_removal = True

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
  data.path = '/store/CIA/js2164/data/celeba'
  data.split = [0.8, 0.1, 0.1]
  data.dataset = 'CELEBA'
  data.image_size = 64
  data.random_flip = False
  data.uniform_dequantization = False
  data.centered = False
  data.channels = 3
  data.shape = [data.channels, data.image_size, data.image_size]

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 90.
  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  model.name = 'ncsnv2_64'
  model.scale_by_sigma = True
  model.num_scales = 500
  model.ema_rate = 0.999
  model.normalization = 'InstanceNorm++'
  model.nonlinearity = 'elu'
  model.nf = 128
  model.interpolation = 'bilinear'
  
  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = -1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
