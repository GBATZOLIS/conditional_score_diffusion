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
  data.dataset = 'mnist'
  data.datamodule = data.dataset
  data.return_labels = False
  data.use_data_mean = False
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 32
  data.effective_image_size = data.image_size
  data.shape = [1, data.image_size, data.image_size]
  data.centered = False
  data.use_flip = False
  data.crop = False
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.nonlinearity = 'swish'
  model.latent_dim = 10
  model.kl_weight = 0.01
  model.use_config_translator = True
  model.variational = False

# encoder model
  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.name = 'half_U_encoder_no_conv'
  encoder.scale_by_sigma = False
  encoder.ema_rate = 0.9999
  encoder.dropout = 0.1
  encoder.normalization = 'GroupNorm'
  encoder.nonlinearity = model.nonlinearity
  encoder.nf = 128
  encoder.ch_mult = (1, 2, 2, 4)
  encoder.num_res_blocks = 4
  encoder.attn_resolutions = (16,)
  encoder.resamp_with_conv = True
  encoder.time_conditional = False
  encoder.init_scale = 0.
  encoder.embedding_type = 'positional'
  encoder.conv_size = 3
  encoder.input_channels = data.num_channels
  encoder.output_channels = 128
  encoder.latent_dim = model.latent_dim


  # decoder model
  config.decoder = decoder = ml_collections.ConfigDict()
  decoder.name = 'half_U_decoder_no_conv'
  decoder.scale_by_sigma = False
  decoder.ema_rate = 0.9999
  decoder.dropout = 0.1
  decoder.normalization = 'GroupNorm'
  decoder.nonlinearity = model.nonlinearity
  decoder.nf = 128
  decoder.ch_mult = (1, 2, 2, 2)
  decoder.num_res_blocks = 4
  decoder.attn_resolutions = (16,)
  decoder.resamp_with_conv = True
  decoder.time_conditional = False
  decoder.init_scale = 0.
  decoder.embedding_type = 'positional'
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