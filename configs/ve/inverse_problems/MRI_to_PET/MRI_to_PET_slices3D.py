import ml_collections
import torch
import math
import numpy as np

def get_config():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'conditional_decreasing_variance'
  training.batch_size = 4
  training.num_nodes = 1
  training.gpus = 2
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  #----- to be removed -----
  training.num_epochs = 10000
  training.n_iters = 2400001
  training.snapshot_freq = 5000
  training.log_freq = 250
  training.eval_freq = 2500
  #------              --------
  
  training.visualization_callback = 'paired3D'
  training.show_evolution = False
  
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = False
  training.reduce_mean = True 
  training.sde = 'vesde'
  

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'conditional_reverse_diffusion'
  sampling.corrector = 'conditional_langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4*training.gpus
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 4
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = 'datasets'
  data.dataset = 'mri_to_pet'
  data.use_data_mean = False
  data.datamodule = 'paired'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 96
  data.effective_image_size = data.image_size
  data.shape_x = [1, data.image_size, data.image_size, 16]
  data.shape_y = [1, data.image_size, data.image_size, 16]
  data.range_x = [0,50000] #[0, 1226428]
  data.range_y = [0,7000] #[0, 7190]
  
  data.centered = False
  data.random_flip = False
  data.uniform_dequantization = False
  data.num_channels = data.shape_x[0]+data.shape_y[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None
  model.num_scales = 1000

  #SIGMA INFORMATION FOR THE VE SDE
  model.reach_target_steps = 6e4

  model.sigma_max_x = np.sqrt(np.prod(data.shape_x))
  model.sigma_max_y = np.sqrt(np.prod(data.shape_y))
  model.sigma_max_y_target = model.sigma_max_y/2**4
  
  model.sigma_min_x = 1e-4
  model.sigma_min_y = 1e-4
  model.sigma_min_y_target = model.sigma_min_y #SET it equal to model.sigma_min_y if you do not want to reduce sigma_min_y

  model.beta_min = 0.1
  # We use an adjusted beta max 
  # because the range is doubled in each level starting from the first level
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'


  model.name = 'ddpm3D_paired'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 64
  model.ch_mult = (1, 1, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = () #(24, 12, 6) -> attention is not supported for ddpm3D yet.
  model.resamp_with_conv = False #code modifications needed in the downsample and upsample functions to make this True.
  model.conditional = True
  model.conv_size = 3
  model.input_channels = data.num_channels
  model.output_channels = data.num_channels

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
  #config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


  return config