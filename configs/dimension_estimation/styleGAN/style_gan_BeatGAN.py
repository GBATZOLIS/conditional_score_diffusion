import ml_collections
import torch
import math
import numpy as np
from datetime import timedelta
import os
from configs.dimension_estimation.styleGAN.style_gan_base import get_config as get_base_config

# get home directory
from pathlib import Path
home = str(Path.home())

def get_config():
  config = get_base_config()  

  # data
  data = config.data
  
  #model 
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None #'/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/scoreVAE/experiments/paper/pretrained/FFHQ_128/prior/checkpoints/best/last.ckpt'

  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.

  model.name = 'BeatGANsUNetModel'
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
  model.embed_channels = None
  # at what resolutions you want to do self-attention of the feature maps
  # attentions generally improve performance
  # default: [16]
  # beatgans: [32, 16, 8]
  model.attention_resolutions = (16, )
  # number of time embed channels
  model.time_embed_channels: int = None
  # dropout applies to the resblocks (on feature maps)
  model.dropout: float = 0.1
  model.channel_mult = (1, 1, 2, 3, 4)
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
  model.resnet_two_cond: bool = False
  model.resnet_cond_channels: int = None
  # init the decoding conv layers with zero weights, this speeds up training
  # default: True (BeattGANs)
  model.resnet_use_zero_module: bool = True
  # gradient checkpoint the attention operation
  model.attn_checkpoint: bool = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.slowing_factor = 1
  optim.grad_clip = 1.

  return config
