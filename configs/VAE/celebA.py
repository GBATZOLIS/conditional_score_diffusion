import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta

def get_config():
  config = ml_collections.ConfigDict()

  # logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_freq = 1

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
  data.base_dir = '/store/CIA/js2164/data'
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
  model.latent_dim = 100
  model.kl_weight = 0.01

  # encoder model
  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.name = 'half_U_encoder'
  encoder.scale_by_sigma = False
  encoder.ema_rate = 0.9999
  encoder.dropout = 0.1
  encoder.normalization = 'GroupNorm'
  encoder.nonlinearity = model.nonlinearity
  encoder.nf = 128
  encoder.ch_mult = (1, 1, 2, 2)
  encoder.num_res_blocks = 3
  encoder.attn_resolutions = (16,)
  encoder.resamp_with_conv = True
  encoder.conditional = False
  encoder.fir = False
  encoder.fir_kernel = [1, 3, 3, 1]
  encoder.skip_rescale = True
  encoder.resblock_type = 'biggan'
  encoder.progressive = 'none'
  encoder.progressive_input = 'none'
  encoder.progressive_combine = 'sum'
  encoder.attention_type = 'ddpm'
  encoder.init_scale = 0.
  encoder.embedding_type = 'positional'
  encoder.fourier_scale = 16
  encoder.conv_size = 3
  encoder.input_channels = data.num_channels
  encoder.output_channels = 3
  encoder.latent_dim = model.latent_dim


  # decoder model
  config.decoder = decoder = ml_collections.ConfigDict()
  decoder.name = 'half_U_decoder'
  decoder.scale_by_sigma = False
  decoder.ema_rate = 0.9999
  decoder.dropout = 0.1
  decoder.normalization = 'GroupNorm'
  decoder.nonlinearity = model.nonlinearity
  decoder.nf = 128
  decoder.ch_mult = (1, 1, 2, 2)
  decoder.num_res_blocks = 3
  decoder.attn_resolutions = (16,)
  decoder.resamp_with_conv = True
  decoder.conditional = False
  decoder.fir = False
  decoder.fir_kernel = [1, 3, 3, 1]
  decoder.skip_rescale = True
  decoder.resblock_type = 'biggan'
  decoder.progressive = 'none'
  decoder.progressive_input = 'none'
  decoder.progressive_combine = 'sum'
  decoder.attention_type = 'ddpm'
  decoder.init_scale = 0.
  decoder.embedding_type = 'positional'
  decoder.fourier_scale = 16
  decoder.conv_size = 3
  decoder.input_channels = encoder.output_channels
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