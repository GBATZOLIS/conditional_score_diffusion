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
  logging.log_name = 'disentangled_encoder'
  logging.top_k = 3
  logging.every_n_epochs = 1000
  logging.envery_timedelta = timedelta(minutes=1)

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'disentangled_score_vae'
  training.use_pretrained = True
  training.prior_config_path = '/home/gb511/rds_work/projects/scoreVAE/experiments/gd_ffhq/DiffDecoders_continuous_prior/config.pkl' if config.server=='hpc' else '/store/CIA/gb511/projects/scoreVAE/experiments/CelebA_64/attribute_conditional/config.pkl' 
  training.prior_checkpoint_path = '/home/gb511/rds_work/projects/scoreVAE/experiments/gd_ffhq/DiffDecoders_continuous_prior/checkpoints/best/epoch=141--eval_loss_epoch=0.014.ckpt' if config.server=='hpc' else '/store/CIA/gb511/projects/scoreVAE/experiments/CelebA_64/attribute_conditional/checkpoints/best/epoch=211--eval_loss_epoch=0.016.ckpt'
  training.encoder_only = True
  training.t_dependent = True
  training.conditioning_approach = 'sr3'
  training.batch_size = 32
  training.t_batch_size = 1
  training.num_nodes = 1
  training.gpus = 2
  training.accelerator = 'gpu'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus

  #----- to be removed -----
  training.num_epochs = 10000
  training.n_iters = 2500000
  training.snapshot_freq = 5000
  training.log_freq = 250
  training.eval_freq = 2500
  #------              --------
  
  training.visualisation_freq = 10
  training.visualization_callback = ['attribute_conditional_encoder']
  training.show_evolution = False

  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'snrsde'
  training.beta_schedule = 'linear'

  ##new related to the training of Score VAE
  training.variational = True
  training.cde_loss = False
  training.kl_weight = 1e-3

  #DISENTANGLED REPRESENTATION LEARNING TRAINING PARAMETERS (new)
  training.disentanglement_factor = 0 #controls the degree of disentanglement.

  # validation
  config.validation = validation = ml_collections.ConfigDict()
  validation.batch_size = training.batch_size
  validation.workers = training.workers

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'conditional_ddim'
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
  data.base_dir = '/home/gb511/rds_work/datasets/' if config.server=='hpc' else '/store/CIA/gb511/datasets' 
  data.dataset = 'celebA-HQ-160'
  data.datamodule = 'CelebA_Annotated_PKLDataset'
  data.normalization_mode = 'gd'
  data.attributes = 'all'
  data.total_num_attributes = 40
  data.num_classes = 2

  data.return_labels = False
  data.use_data_mean = False
  data.create_dataset = False
  #data.split = [0.9, 0.05, 0.05]
  data.image_size = 64
  #data.percentage_use = 100 #default:100
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.latent_dim = 512
  data.class_cond = False
  data.centered = True
  data.random_crop = False
  data.random_flip = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None
  model.num_scales = 1000
  model.ema_rate = 0.999 #0.9999

  config.encoder = encoder = ml_collections.ConfigDict()
  encoder.name = 'BeatGANsEncoderModel'
  
  encoder.image_size = data.image_size
  encoder.in_channels = data.num_channels
  encoder.out_channels = data.num_channels
  encoder.model_channels = 128
  encoder.enc_num_res_blocks = 2
  encoder.latent_dim = data.latent_dim
  encoder.enc_attn_resolutions = ()
  encoder.enc_use_time_condition = True
  encoder.enc_channel_mult = (1, 1, 2, 3, 4)
  encoder.enc_pool = 'flatten-linear'
  encoder.resolution_before_flattening = data.image_size // 2 ** (len(encoder.enc_channel_mult) - 1)
  encoder.conv_resample = True
  encoder.dims = 2
  encoder.use_checkpoint = False
  encoder.num_heads = 1
  encoder.num_head_channels = -1
  encoder.use_new_attention_order = False
  encoder.dropout = 0.1
  encoder.enc_out_channels = 2 * data.latent_dim
  encoder.encoder_split_output = False
  encoder.resblock_updown = False

  # Redundant encoder attributes
  # encoder.num_input_res_blocks = None
  # encoder.embed_channels = 512
  # encoder.time_embed_channels = None
  # encoder.input_channel_mult = None
  # encoder.num_classes = None
  # encoder.num_heads_upsample = -1
  # encoder.resnet_two_cond = False
  # encoder.resnet_cond_channels = None
  # encoder.resnet_use_zero_module = True
  # encoder.attn_checkpoint = False
  

  config.MI_estimator = MI_estimator = ml_collections.ConfigDict()
  MI_estimator.name = 'MLPSkipNet'
  MI_estimator.output_channels = data.latent_dim
  MI_estimator.attribute_channels = data.total_num_attributes if data.attributes == 'all' else len(data.attributes)
  MI_estimator.input_channels = data.latent_dim + MI_estimator.attribute_channels
  MI_estimator.num_channels = MI_estimator.output_channels  # latent dimension
  MI_estimator.num_layers = 10
  MI_estimator.skip_layers = list(range(1, 10))
  MI_estimator.num_hid_channels = 1024
  MI_estimator.num_time_emb_channels = 64
  MI_estimator.activation = 'silu'  # Assuming Activation enum or similar, replace with actual class or enum if needed
  MI_estimator.use_norm = True
  MI_estimator.condition_bias = 1
  MI_estimator.dropout = 0.2
  MI_estimator.last_act = 'none'  # Assuming Activation enum or similar, replace with actual class or enum if needed
  MI_estimator.num_time_layers = 2
  MI_estimator.time_last_act = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.use_ema = False
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.encoder_lr = 5e-5
  optim.MI_diffusion_lr = 1e-4
  optim.MI_update_steps = 5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 1000
  optim.slowing_factor = 1
  optim.grad_clip = 0
  optim.manual_grad_clip = 1.

  config.seed = 42
  return config
