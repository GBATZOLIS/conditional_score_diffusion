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
  config.log_path = '/home/gb511/projects/scoreVAE/experiments'
  logging.log_freq = 5

  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 128
  training.workers = 4
  training.variational = True

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

  # model
  config.model = model = ml_collections.ConfigDict()
  model.nonlinearity = 'swish'
  model.latent_dim = 384
  model.kl_weight = 0.01
  model.time_conditional = False

  model.checkpoint_path = None
  model.dropout = 0.1
  model.input_channels = 2*data.num_channels
  model.output_channels = data.num_channels
  model.encoder_input_channels = data.num_channels
  model.encoder_latent_dim = model.latent_dim 
  model.encoder_base_channel_size = 64

  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.name = 'simple_encoder'
  config.decoder = decoder = ml_collections.ConfigDict()
  decoder.name = 'simple_decoder'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4
  optim.use_scheduler=True
  optim.sch_factor = 0.25
  optim.sch_patience = 5
  optim.sch_min_lr = 1e-5

  return config