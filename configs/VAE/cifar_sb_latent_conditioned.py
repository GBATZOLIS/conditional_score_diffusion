import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
from configs.utils import get_path

def get_config():
  config = ml_collections.ConfigDict()

  config.log_name = 'beta_max_2e-2_test'
  config.VAE_config_path = '/home/gb511/projects/scoreVAE/code/configs/VAE/cifar_simple.py'
  config.stochastic_decoder = True

  # training 
  config.training = training = ml_collections.ConfigDict()
  training.visualisation_freq = 10
  training.sb_latent_conditioned = True

  #reference process settings
  training.beta_max = 2e-2 #2e-1
  training.beta_min = 1e-4

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
  data.latent_dim = 384
  data.range = [0, 1]

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = '/home/gb511/projects/scoreVAE/experiments/VAE/cifar10/kl_0.01/SBVAE/beta_max/2e-2/copy_stochastic_decoder_latent_condition/sb_checkpoints/epoch=142--val_loss=0.399.ckpt'
  
  model.name = 'BeatGANsLatentScoreModel'
  model.ema_rate = 0.9999
  model.image_size = data.image_size
  model.in_channels = data.num_channels
  # base channels, will be multiplied
  model.model_channels: int = 128
  # output of the unet
  # suggest: 3
  # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
  model.out_channels = data.num_channels
  # how many repeating resblocks per resolution
  # the decoding side would have "one more" resblock
  # default: 2
  model.num_res_blocks: int = 2
  # you can also set the number of resblocks specifically for the input blocks
  # default: None = above
  model.num_input_res_blocks: int = None
  # number of time embed channels and style channels
  model.embed_channels = data.latent_dim 
  # at what resolutions you want to do self-attention of the feature maps
  # attentions generally improve performance
  # default: [16]
  # beatgans: [32, 16, 8]
  model.attention_resolutions = (16, )
  # number of time embed channels
  model.time_embed_channels: int = None
  # dropout applies to the resblocks (on feature maps)
  model.dropout: float = 0.3
  model.channel_mult = (1, 1, 2, 2)
  model.input_channel_mult = None
  model.conv_resample: bool = True
  # always 2 = 2d conv
  model.dims: int = 2
  # don't use this, legacy from BeatGANs
  model.num_classes: int = None
  model.use_checkpoint: bool = False
  # number of attention heads
  model.num_heads: int = 1
  # or specify the number of channels per attention head
  model.num_head_channels: int = -1
  # what's this?
  model.num_heads_upsample: int = -1
  # use resblock for upscale/downscale blocks (expensive)
  # default: True (BeatGANs)
  model.resblock_updown: bool = True
  # never tried
  model.use_new_attention_order: bool = False
  model.resnet_two_cond: bool = True
  model.resnet_cond_channels: int = None
  # init the decoding conv layers with zero weights, this speeds up training
  # default: True (BeattGANs)
  model.resnet_use_zero_module: bool = True
  # gradient checkpoint the attention operation
  model.attn_checkpoint: bool = False

  model.latent_dim = data.latent_dim

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4
  optim.use_scheduler=True
  optim.sch_factor = 0.25
  optim.sch_patience = 30
  optim.sch_min_lr = 1e-5

  return config