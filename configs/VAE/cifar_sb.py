import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
from configs.utils import get_path

def get_config():
  config = ml_collections.ConfigDict()

  config.log_name = 'standard'
  config.VAE_config_path = '/home/gb511/projects/scoreVAE/code/configs/VAE/cifar_simple.py'
  config.stochastic_decoder = True

  # training 
  config.training = training = ml_collections.ConfigDict()
  training.sigma = 1
  training.visualisation_freq = 10
  training.sb_latent_conditioned = False

  #reference process settings
  training.beta_max = 2e-2
  training.beta_min = 1e-4

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = get_path('data_path')
  data.dataset = 'cifar10'
  data.datamodule = data.dataset
  data.return_labels = False
  data.use_data_mean = False
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 32
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.centered = False
  data.use_flip = False
  data.crop = False
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.
  data.range = [0, 1]

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None

  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.time_conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3
  model.dropout = 0.3
  model.input_channels = 3
  model.output_channels = 3

  model.latent_dim = 384

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4
  optim.use_scheduler=True
  optim.sch_factor = 0.25
  optim.sch_patience = 40
  optim.sch_min_lr = 1e-5

  return config