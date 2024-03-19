import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta

def get_config():
  config = ml_collections.ConfigDict()

  config.server = 'CIA'

  #logging
  config.logging = logging = ml_collections.ConfigDict()
  logging.log_path = '/home/gb511/rds_work/projects/scoreVAE/experiments/CelebA_64' if config.server=='hpc' else '/store/CIA/gb511/projects/scoreVAE/experiments/CelebA_64' 
  logging.log_name = 'prior'
  logging.top_k = 5
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'base'
  training.conditioning_approach = 'sr3'
  training.batch_size = 64
  training.t_batch_size = 1
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = 'gpu'
  training.accumulate_grad_batches = 2
  training.workers = 4*training.gpus
  #----- to be removed -----
  training.num_epochs = 10000
  training.n_iters = 2500000
  training.snapshot_freq = 5000
  training.log_freq = 250
  training.eval_freq = 2500
  #------              --------
  
  training.visualisation_freq = 10
  training.visualization_callback = 'base'
  training.show_evolution = False

  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'snrsde'
  training.beta_schedule = 'linear'

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = training.batch_size
  validation.workers = training.workers

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'ddim'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.15 

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
  data.base_dir = '/home/gb511/rds_work/datasets/' if config.server=='hpc' else '/store/CIA/gb511/datasets' 
  data.dataset = 'celebA-HQ-160'
  data.datamodule = 'CelebA_Annotated_PKLDataset'
  data.normalization_mode = 'gd'
  data.attributes = ['Eyeglasses', 'Male', 'Smiling', 'Wearing_Hat', 'Young']
  data.total_num_attributes = 40
  data.num_classes = 2
  data.return_labels = False
  data.use_data_mean = False
  data.create_dataset = False
  #data.split = [0.9, 0.05, 0.05]
  data.image_size = 64
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.class_cond = False
  data.centered = True
  data.random_crop = False
  data.random_flip = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.num_scales = 1000
  model.discrete_checkpoint_path = None
  model.checkpoint_path = None

  model.name = 'BeatGANsUNetModel'
  model.ema_rate = 0.9999
  model.image_size = data.image_size
  model.in_channels = data.num_channels
  model.model_channels: int = 128
  model.out_channels = data.num_channels
  model.num_res_blocks: int = 2
  model.num_input_res_blocks: int = None
  model.embed_channels = 512
  model.attention_resolutions = (16, )
  model.time_embed_channels: int = None
  model.dropout: float = 0.
  model.channel_mult = (1, 1, 2, 3)
  model.input_channel_mult = None
  model.conv_resample: bool = True
  model.dims: int = 2
  model.num_classes: int = None
  model.use_checkpoint: bool = False
  model.num_heads: int = 1
  model.num_head_channels: int = -1
  model.num_heads_upsample: int = -1
  model.resblock_updown: bool = True
  model.use_new_attention_order: bool = False
  model.resnet_two_cond: bool = False
  model.resnet_cond_channels: int = None
  model.resnet_use_zero_module: bool = True
  model.attn_checkpoint: bool = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 8e-5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  return config