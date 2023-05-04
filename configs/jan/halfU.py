import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta

def get_config():
  config = ml_collections.ConfigDict()

  #logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = 'logs/celebA/'
  logging.log_name = 'real_celebA_crop_ampere_2'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'base'
  training.batch_size = 128
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4
  training.num_epochs = 10000
  training.n_iters = 2500000

  #----- to be removed -----
  training.snapshot_freq = 5000
  training.log_freq = 250
  training.eval_freq = 2500
  #------              --------
  
  training.visualization_callback = 'base'
  training.visualisation_freq = 1
  training.show_evolution = False
  
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000 #to be removed

  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'vesde'

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = 128
  validation.workers = 4

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4
  evaluate.batch_size = training.batch_size
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.callback = None

  # data
  # config.data = data = ml_collections.ConfigDict()
  # data.datamodule = 'image'
  # data.base_dir = '/rds/user/js2164/hpc-work/data' #'datasets' ->put the directory where you have the dataset: /datasets/. It will load .../datasets/celebA
  # data.dataset = 'celeba'
  # data.use_data_mean = False
  # data.create_dataset = False
  # data.split = [0.8, 0.1, 0.1]
  # data.image_size = 64
  # data.effective_image_size = data.image_size
  # data.shape = [3, data.image_size, data.image_size]
  # data.centered = False
  # data.random_flip = False
  # data.crop = True
  # data.uniform_dequantization = False
  # data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = '/store/CIA/js2164/data'
  data.dataset = 'celeba'
  data.task = 'generation'
  data.datamodule = 'unpaired_PKLDataset'
  data.scale = 4 #?
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
  data.num_channels = data.shape[0] #the number of channels the model sees as input.


  config.model = model = ml_collections.ConfigDict()
  model.nonlinearity = 'swish'
  


  # model
  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.checkpoint_path = None
  encoder.num_scales = 1000
  encoder.sigma_max = np.sqrt(np.prod(data.shape))
  encoder.sigma_min = 0.01
  encoder.beta_min = 0.1
  encoder.beta_max = 20.
  encoder.dropout = 0.1
  encoder.embedding_type = 'fourier'

   # encoder architecture
  encoder.name = 'half_U_encoder'
  encoder.scale_by_sigma = False
  encoder.ema_rate = 0.9999
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
  encoder.output_channels = data.num_channels
  encoder.latent_dim = 100


  # model
  config.decoder = decoder = ml_collections.ConfigDict()
  decoder.checkpoint_path = None
  decoder.num_scales = 1000
  decoder.sigma_max = np.sqrt(np.prod(data.shape))
  decoder.sigma_min = 0.01
  decoder.beta_min = 0.1
  decoder.beta_max = 20.
  decoder.dropout = 0.1
  decoder.embedding_type = 'fourier'

   # decoder architecture
  decoder.name = 'half_U_decoder'
  decoder.scale_by_sigma = False
  decoder.ema_rate = 0.9999
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
  decoder.latent_dim = 100

  # optimization
  config.optim = optim = ml_collections.ConfigDict()

  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000 #set it to 0 if you do not want to use warm up.
  optim.grad_clip = 1 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  config.seed = 42

  return config