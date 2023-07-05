import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta

def get_config():
  config = ml_collections.ConfigDict()

  #logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = '/home/gb511/projects/scoreVAE/experiments/cifar10'
  logging.log_name = 'continuous_conversion_encoder'
  logging.top_k = 3
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'encoder_only_pretrained_score_vae'
  training.use_pretrained = True
  training.prior_config_path = '/home/gb511/projects/scoreVAE/experiments/cifar10/continuous_conversion_linear/config.pkl'
  training.prior_checkpoint_path = '/home/gb511/projects/scoreVAE/experiments/cifar10/continuous_conversion_linear/checkpoints/best/epoch=54--eval_loss_epoch=0.014.ckpt'
  training.encoder_only = True
  training.t_dependent = True
  training.conditioning_approach = 'sr3'
  training.batch_size = 32
  training.t_batch_size = 1
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  #----- to be removed -----
  training.num_epochs = 10000
  training.n_iters = 2500000
  training.snapshot_freq = 5000
  training.log_freq = 250
  training.eval_freq = 2500
  #------              --------
  
  training.visualisation_freq = 50
  training.visualization_callback = None
  training.show_evolution = False

  training.likelihood_weighting = False #irrelevant for this config
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'snrsde'
  training.beta_schedule = 'linear'

  ##new related to the training of Score VAE
  training.variational = True
  training.cde_loss = False #only difference -> this allows us to use the VAE loss
  training.kl_weight = 0.01 #KL penalty

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = training.batch_size
  validation.workers = training.workers

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'conditional_euler_maruyama'
  sampling.corrector = 'conditional_none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.15 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.callback = None
  evaluate.workers = training.workers
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = validation.batch_size
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = '/home/gb511/datasets' #'/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/datasets'
  data.dataset = 'cifar10'
  data.datamodule = data.dataset
  data.return_labels = False
  data.use_data_mean = False
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 32
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.latent_dim = 384
  data.centered = False
  data.use_flip = False
  data.crop = False
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.ema_rate = 0.999
  model.num_scales = 1000
  model.checkpoint_path = None
  model.encoder_name = 'time_dependent_simple_encoder'
  model.encoder_input_channels = data.num_channels
  model.encoder_latent_dim = data.latent_dim
  model.encoder_base_channel_size = 64
  model.encoder_split_output=False


  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 2500
  optim.slowing_factor = 10
  optim.grad_clip = 1.

  config.seed = 42
  return config