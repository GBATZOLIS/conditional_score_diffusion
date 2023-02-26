import torch
import torch.nn as nn
import functools
import pytorch_lightning as pl
from . import utils, layers, normalization
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize
import numpy as np

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ddpm_encoder')
class DDPMencoder(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.latent_dim = config.data.latent_dim #new
    self.nf = nf = config.model.encoder_nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.encoder_num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.effective_image_size // (2 ** i) for i in range(num_resolutions)] #80,40,20,10

    AttnBlock = functools.partial(layers.AttnBlock)
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)

    self.centered = config.data.centered
    input_channels = config.model.encoder_input_channels
    output_channels = config.model.encoder_output_channels

    modules = []
    # ddpm_conv3x3
    modules.append(conv3x3(input_channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, output_channels, init_scale=0.))

    modules.append(nn.Linear(np.prod(config.data.shape), self.latent_dim)) #new
    modules.append(nn.Tanh()) #new

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x):
    modules = self.all_modules
    m_idx = 0
    temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h.view(h.shape[0], -1))
    m_idx += 1
    assert m_idx == len(modules)

    return h

@utils.register_model(name='simple_encoder')
class Encoder(pl.LightningModule):
    def __init__(self, config):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        num_input_channels = config.model.encoder_input_channels
        base_channel_size = config.model.encoder_base_channel_size
        self.latent_dim = latent_dim = config.model.encoder_latent_dim
        act_fn = nn.GELU
        self.variational = config.training.variational

        if self.variational:
          out_dim = 2*latent_dim
        else:
          out_dim = latent_dim

        c_hid = base_channel_size

        if config.data.image_size == 32:
          self.net = nn.Sequential(
              nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
              act_fn(),
              nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
              act_fn(),
              nn.Flatten(), # Image grid to single feature vector
              nn.Linear(2*16*c_hid, out_dim)
          )
        
        elif config.data.image_size == 128:
          self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 128x128 => 64x64
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.Conv2d(4*c_hid, 8*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(8*c_hid, 8*c_hid, kernel_size=3, padding=1),
            act_fn(),

            nn.Conv2d(8*c_hid, 8*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),

            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(8*16*c_hid, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.net(x)
        if self.variational:
          return out[:,:self.latent_dim], out[:,self.latent_dim:]
        else:
          return out