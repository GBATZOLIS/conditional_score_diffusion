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
  logging.log_path = '/home/gb511/rds_work/projects/scoreVAE/experiments/CelebAHQ' if config.server=='hpc' else '/store/CIA/gb511/projects/scoreVAE/experiments/CelebAHQ' 
  logging.log_name = 'AttributeEncoder_5attributes'
  logging.top_k = 3
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'attribute_encoder'
  training.prior_config_path = '/home/gb511/rds_work/projects/scoreVAE/experiments/gd_ffhq/DiffDecoders_continuous_prior/config.pkl' if config.server=='hpc' else '/store/CIA/gb511/projects/scoreVAE/experiments/ffhq/prior/config.pkl' 
  training.prior_checkpoint_path = '/home/gb511/rds_work/projects/scoreVAE/experiments/gd_ffhq/DiffDecoders_continuous_prior/checkpoints/best/epoch=141--eval_loss_epoch=0.014.ckpt' if config.server=='hpc' else '/store/CIA/gb511/projects/scoreVAE/experiments/ffhq/prior/epoch=141--eval_loss_epoch=0.014.ckpt' 
  training.conditioning_approach = 'sr3'
  training.batch_size = 32
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
  training.visualization_callback = ['AttributeEncoder']
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
  sampling.predictor = 'conditional_heun'
  sampling.corrector = 'conditional_none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.15 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.callback = None
  evaluate.workers = training.workers
  evaluate.batch_size = validation.batch_size
  evaluate.num_samples = 50000

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
  data.image_size = 128
  #data.percentage_use = 100 #default:100
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  #data.latent_dim = 1024
  data.class_cond = False
  data.centered = True
  data.random_crop = False
  data.random_flip = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  #encoder
  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.name = 'AttributeEncoderModel'
  encoder.enc_use_time_condition = True
  encoder.model_channels = 64
  encoder.in_channels = data.num_channels  # Ensure 'data.num_channels' is defined elsewhere in your code
  encoder.image_size = data.image_size  # Ensure 'data.image_size' is defined elsewhere in your code
  encoder.enc_channel_mult = (1, 1, 2, 3, 4, 4)
  encoder.enc_num_res_blocks = 2
  encoder.dropout = 0.2
  encoder.dims = 2
  encoder.use_checkpoint = False
  encoder.num_heads = 1
  encoder.num_head_channels = -1  # Default setting, could be changed if needed
  encoder.use_new_attention_order = False
  encoder.resblock_updown = False
  encoder.conv_resample = True
  encoder.enc_attn_resolutions = ()  # Empty tuple indicates no specific resolutions for attention
  encoder.enc_pool = 'flatten-linear'
  encoder.resolution_before_flattening = data.image_size // 2**(len(encoder.enc_channel_mult)-1)  # Calculation based on 'enc_channel_mult'
  
  encoder.enc_out_channels = data.num_classes*data.total_num_attributes if data.attributes == 'all' else data.num_classes*len(data.attributes)
  encoder.encoder_split_output = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None
  model.num_scales = 1000
  model.ema_rate = 0.999 #0.9999

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 1000
  optim.grad_clip = 1.

  config.seed = 42
  return config