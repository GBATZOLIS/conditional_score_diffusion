import ml_collections
import torch
import math
import numpy as np
from configs.jan.default import get_default_configs
from datetime import timedelta

def get_config():
  config = get_default_configs()

  # logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = 'logs/crypto/'
  logging.log_name = 'btc_d_returns'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  training = config.training 
  training.lightning_module = 'conditional'
  training.conditioning_approach = 'sr3'
  training.batch_size = 16
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

  training.sde = 'vesde'
  
  training.visualization_callback = [] # CHECK
  training.show_evolution = False
  
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000 #to be removed

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'conditional_reverse_diffusion'
  sampling.corrector = 'conditional_none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.15 #0.15 in VE sde (you t

  # evaluation (this file is not modified at all - subject to change)
  evaluate = config.eval 
  evaluate.workers = 4
  evaluate.batch_size = 50
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.datamodule = 'Crypto'
  data.dataset = 'Crypto'
  data.use_data_mean = False
  data.create_dataset = False
  data.coin_name = 'BTC'
  data.time_frequency = 'd'
  data.L_1 = 30
  data.L_2 = 1
  data.shape = [data.L_2, 1] # (L_2, D)
  data.num_channels = 0 #the number of channels the model sees as input.


  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None
  model.num_scales = 1000
  model.sigma_max_x = np.sqrt(np.prod(data.shape)) #input range is [0,1] and resolution is 64^2
  print('model.sigma_max: %.4f' % model.sigma_max_x)

  #-------The three subsequent settings configure the reduction schedule of sigma_max_y
  #model.reduction = 'inverse_exponentional' #choices=['linear', 'inverse_exponentional']
  #model.reach_target_in_epochs = 64
  #model.starting_transition_iterations = 2000
  #-------

  model.sigma_min_x = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  model.name = 'csdi_conditional'
  model.num_channels = 64
  model.num_scales = 50
  model.diff_embedding_dim = 128
  model.time_embedding_dim = 128
  model.feature_embedding_dim = 16
  model.nheads = 8
  model.num_layers = 4 
  model.ema_rate = 0.999

  # optimization
  config.optim = optim = ml_collections.ConfigDict()

  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2.0e-5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 0 #set it to 0 if you do not want to use warm up.
  optim.grad_clip = 1 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  config.seed = 42


  return config