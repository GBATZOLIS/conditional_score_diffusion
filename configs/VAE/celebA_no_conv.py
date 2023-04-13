import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
from configs.utils import get_path

def get_config():
  config = ml_collections.ConfigDict()

  # logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_freq = 5

  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 128
  training.workers = 4

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = 128
  validation.workers = 4
  
  # test
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = validation.batch_size
  evaluate.workers = 4
  
  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = get_path('data_path')
  data.dataset = 'celeba'
  data.task = 'generation'
  data.datamodule = 'unpaired_PKLDataset'
  data.scale = 4 
  data.use_data_mean = False
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 64
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.centered = False
  data.use_flip = True
  data.crop = True
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] 

  # model
  config.model = model = ml_collections.ConfigDict()
  model.nonlinearity = 'swish'
  model.latent_dim = 512
  data.latent_dim = model.latent_dim
  model.kl_weight = 0.01

  # encoder model
  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.name = 'time_dependent_DDPM_encoder'

  model.checkpoint_path = None
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  model.unconditional_score_model_name = 'ddpm'
  model.name = 'ddpm_mirror_decoder'
  model.input_channels = data.num_channels
  model.output_channels = data.num_channels
  model.scale_by_sigma = True
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 3)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = False
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


  # decoder model
  config.decoder = decoder = ml_collections.ConfigDict()
  decoder.name = 'half_U_decoder_no_conv'
  decoder.scale_by_sigma = False
  decoder.ema_rate = 0.9999
  decoder.dropout = 0.0
  decoder.normalization = 'GroupNorm'
  decoder.nonlinearity = model.nonlinearity
  decoder.nf = 128
  decoder.ch_mult = (1, 1, 2, 2, 3)
  decoder.num_res_blocks = 2
  decoder.attn_resolutions = (16,)
  decoder.resamp_with_conv = True
  decoder.conditional = False
  decoder.init_scale = 0.
  decoder.embedding_type = 'positional'
  decoder.conv_size = 3
  decoder.input_channels = model.ch_mult[-1] * model.nf
  decoder.output_channels = data.num_channels
  decoder.latent_dim = model.latent_dim

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4
  optim.use_scheduler=True
  optim.sch_factor = 0.25
  optim.sch_patience = 5
  optim.sch_min_lr = 1e-5

  return config