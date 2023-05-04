import torch
import torch.nn as nn
import functools
import pytorch_lightning as pl
from . import utils, layers, normalization
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

class SqueezeBlock(nn.Module):
  def forward(self, z, reverse=False):
      B, C, H, W = z.shape
      if not reverse:
          # Forward direction: H x W x C => H/2 x W/2 x 4C
          z = z.reshape(B, C, H//2, 2, W//2, 2)
          z = z.permute(0, 1, 3, 5, 2, 4)
          z = z.reshape(B, 4*C, H//2, W//2)
      else:
          # Reverse direction: H/2 x W/2 x 4C => H x W x C
          z = z.reshape(B, C//4, 2, 2, H, W)
          z = z.permute(0, 1, 4, 2, 5, 3)
          z = z.reshape(B, C//4, H*2, W*2)
      return z

def permute_channels(haar_image, forward=True):
        permuted_image = torch.zeros_like(haar_image)
        if forward:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                for j in range(3):
                    permuted_image[:, 3*k+j, :, :] = haar_image[:, 4*j+i, :, :]
        else:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                
                for j in range(3):
                    permuted_image[:,4*j+k,:,:] = haar_image[:, 3*i+j, :, :]

        return permuted_image

@utils.register_model(name='half_U_encoder')
class HalfUEncoder(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.latent_dim = config.encoder.latent_dim
    self.act = act = get_act(config)

    self.nf = nf = config.encoder.nf
    ch_mult = config.encoder.ch_mult
    self.num_res_blocks = num_res_blocks = config.encoder.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.encoder.attn_resolutions
    dropout = config.encoder.dropout
    resamp_with_conv = config.encoder.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.effective_image_size // (2 ** i) for i in range(num_resolutions)] #80,40,20,10

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.encoder.time_conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)
    else:
      modules = []

    self.centered = config.data.centered
    input_channels = config.encoder.input_channels
    output_channels = config.encoder.output_channels

    if self.latent_dim:
        self.last_hidden_dim = self.all_resolutions[-1]**2 * output_channels
        self.latent_projection = nn.Linear(self.last_hidden_dim, 2 * self.latent_dim)

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

    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, output_channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, labels=None):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
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

    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)
    
    if self.latent_dim:
        h = h.reshape((h.shape[0], -1))
        h = self.latent_projection(h)
    return h[:, :self.latent_dim], h[:, :self.latent_dim]
  

@utils.register_model(name='half_U_decoder')
class HalfUDecoder(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.latent_dim = config.encoder.latent_dim

    self.nf = nf = config.decoder.nf
    ch_mult = config.decoder.ch_mult
    self.num_res_blocks = num_res_blocks = config.decoder.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.decoder.attn_resolutions
    dropout = config.decoder.dropout
    resamp_with_conv = config.decoder.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.effective_image_size // (2 ** i) for i in range(num_resolutions)] #80,40,20,10


    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.decoder.time_conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)
    else:
      modules = []

    self.centered = config.data.centered
    self.input_channels = config.decoder.input_channels
    output_channels = config.decoder.output_channels
    
    if self.latent_dim:
        self.last_hidden_dim = self.all_resolutions[-1]**2 * self.input_channels
        self.latent_projection = nn.Linear(self.latent_dim, self.last_hidden_dim)

    # ddpm_conv3x3
    modules.append(conv3x3(self.input_channels, nf))
    in_ch = nf
   # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, output_channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, labels=None):

    if self.latent_dim:
       h = self.latent_projection(x)
       h = h.reshape(h.shape[0], self.input_channels, self.all_resolutions[-1], self.all_resolutions[-1])
    else:
       h=x

    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None


    h = modules[m_idx](h)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](h, temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    return h
  

@utils.register_model(name='half_U_decoder_no_conv')
class HalfUDecoderNoConv(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.latent_dim = config.model.latent_dim

    self.nf = nf = config.decoder.nf
    ch_mult = config.decoder.ch_mult
    self.num_res_blocks = num_res_blocks = config.decoder.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.decoder.attn_resolutions
    dropout = config.decoder.dropout
    resamp_with_conv = config.decoder.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.effective_image_size // (2 ** i) for i in range(num_resolutions)] #80,40,20,10


    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.decoder.time_conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)
    else:
      modules = []

    self.centered = config.data.centered
    self.input_channels = config.decoder.input_channels
    output_channels = config.decoder.output_channels
    
    if self.latent_dim:
        self.last_hidden_dim = self.all_resolutions[-1]**2 * self.input_channels
        self.latent_projection = nn.Linear(self.latent_dim, self.last_hidden_dim)

    # ddpm_conv3x3
    #modules.append(conv3x3(self.input_channels, nf))
    in_ch = self.input_channels
   # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, output_channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, labels=None):

    if self.latent_dim:
       h = self.latent_projection(x)
       h = h.reshape(h.shape[0], self.input_channels, self.all_resolutions[-1], self.all_resolutions[-1])
    else:
       h=x

    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None


    #h = modules[m_idx](h)
    #m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](h, temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    return h