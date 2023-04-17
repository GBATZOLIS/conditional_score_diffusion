import ml_collections
import torch
import math
import numpy as np
import configs.jan.celebA.default as default
from datetime import timedelta


def get_config():
  config = default.get_config()

  data =  config.data
  sampling = config.sampling

  sampling.corrector = 'none'

  # training
  training = config.training
  training.lightning_module = 'fokker-planck'
  training.schedule = 'constant'
  training.alpha = 1e-3
  training.alpha_min=0#1e-5
  training.alpha_max=0#1e-3
  training.hutchinson = True
  #training.n_chunks=50
  training.batch_size = 64

  # logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = 'logs/celebA/fokker_planck2/'
  logging.log_name = 'fp_1e-3'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)


  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None #'/home/js2164/rds/hpc-work/repos/score_sde_pytorch/logs/celebA/fokker_planck/fp_1e-3/checkpoints/best/last.ckpt'
  model.num_scales = 1000
  model.sigma_max = np.sqrt(np.prod(data.shape)) #input range is [0,1] and resolution is 64^2
  print('model.sigma_max: %.4f' % model.sigma_max)

  #-------The three subsequent settings configure the reduction schedule of sigma_max_y
  #model.reduction = 'inverse_exponentional' #choices=['linear', 'inverse_exponentional']
  #model.reach_target_in_epochs = 64
  #model.starting_transition_iterations = 2000
  #-------

  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  model.name = 'ddpm_potential'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 96
  model.ch_mult = (1, 1, 2, 2, 3)
  model.num_res_blocks = 2
  model.attn_resolutions = (16, 8, 4)
  model.resamp_with_conv = True
  model.time_conditional = True
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

  return config