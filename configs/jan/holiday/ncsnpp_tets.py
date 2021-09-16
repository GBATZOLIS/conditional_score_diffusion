import ml_collections
import torch
import math
import numpy as np

def get_config():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'base'
  training.batch_size = 50
  training.gpus = 1
  training.accumulate_grad_batches = 1
  training.workers = 4
  training.num_epochs = 10000
  training.n_iters = 2400001

  #----- to be removed -----
  training.snapshot_freq = 5000
  training.log_freq = 250
  training.eval_freq = 2500
  #------              --------
  
  training.visualization_callback = 'base'
  training.show_evolution = False
  
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000 #to be removed

  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = False
  training.reduce_mean = True 
  training.sde = 'vesde'
  

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4
  evaluate.batch_size = 50
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = '/store/CIA/js2164/data'
  data.dataset = 'celeba/img_align_celeba'
  data.datamodule = 'image'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 64
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.centered = False
  data.random_flip = False
  data.crop = False
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None

  #-------The three subsequent settings configure the reduction schedule of sigma_max_y
  #model.reduction = 'inverse_exponentional' #choices=['linear', 'inverse_exponentional']
  #model.reach_target_in_epochs = 64
  #model.starting_transition_iterations = 2000
  #-------

  model.sigma_max = 90.
  print('model.sigma_max: %.4f' % model.sigma_max)
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_begin = 90
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
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
  model.init_scale = 0.0
  model.conv_size = 3
  model.embedding_type = 'positional'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()

  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 0 #set it to 0 if you do not want to use warm up.
  optim.grad_clip = 1 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  config.seed = 42

  return config