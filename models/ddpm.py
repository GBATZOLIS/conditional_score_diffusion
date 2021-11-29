# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.
This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
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

@utils.register_model(name='ddpm')
class DDPM(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.effective_image_size // (2 ** i) for i in range(num_resolutions)] #80,40,20,10

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    input_channels = config.model.input_channels
    output_channels = config.model.output_channels

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
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, labels):
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
    assert m_idx == len(modules)

    return h

@utils.register_model(name='ddpm_multi_speed_haar')
class DDPM_multi_speed_haar(DDPM):
  def __init__(self, config, *args, **kwargs):
      super().__init__(config)
      self.haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)
      self.max_haar_depth = config.data.max_haar_depth
  
  def haar_forward(self, x):
      x = self.haar_transform(x)
      x = permute_channels(x)
      return x
    
  def haar_backward(self, x):
      x = permute_channels(x, forward=False)
      x = self.haar_transform.inverse(x)
      return x
    
  def get_dc_coefficients(self, x):
      return self.haar_forward(x)[:,:3,::]
    
  def get_hf_coefficients(self, x):
      return self.haar_forward(x)[:,3:,::]

  def convert_to_haar_space(self, x, max_depth=None):
      if max_depth is None:
        max_depth = self.max_haar_depth
      
      haar_x = {}
      for i in range(max_depth):
        x = self.haar_forward(x)
        if i < max_depth - 1:
          haar_x['d%d'%(i+1)] = x[:,3:,::]
        elif i == max_depth - 1:
          haar_x['d%d'%(i+1)] = x[:,3:,::]
          haar_x['a%d'%(i+1)] = x[:,:3,::]
  
  def detect_haar_depth(self, haar_x : dict):
    for key in haar_x.keys():
      if key.startswith('a'):
        approx_key = key
        break
    return int(approx_key[1])

  def convert_to_image_space(self, haar_x):
    depth = self.detect_haar_depth(haar_x)

    a = haar_x['a%d'%depth]
    for i in range(depth):
      d = haar_x['d%d'%(depth-i)]
      concat = torch.cat((a,d), dim=1)
      a = self.haar_backward(concat)
    
    return a

  def forward(self, haar_x:dict, labels):
    x = self.convert_to_image_space(haar_x)
    image_output = super().forward(x, labels)
    haar_output = self.convert_to_haar_space(image_output)
    return haar_output

@utils.register_model(name='ddpm_paired_SR3')
class DDPM_paired_SR3(DDPM):
  def __init__(self, config, *args, **kwargs):
        super().__init__(config)
  
  def forward(self, input_dict, labels):
    x, y = input_dict['x'], input_dict['y']
    x_channels = x.size(1)
    concat = torch.cat((x, y), dim=1)
    score_x = super().forward(concat, labels)
    return score_x

@utils.register_model(name='ddpm_paired')
class DDPM_paired(DDPM):
  def __init__(self, config, *args, **kwargs):
        super().__init__(config)
  
  def forward(self, input_dict, labels):
    x, y = input_dict['x'], input_dict['y']
    x_channels = x.size(1)
    concat = torch.cat((x, y), dim=1)
    output = super().forward(concat, labels)
    return {'x': output[:,:x_channels,::], \
            'y':output[:,x_channels:,::]}

@utils.register_model(name='ddpm_2xSR')
class DDPM_2xSR(DDPM):
  def __init__(self, config, *args, **kwargs):
      super().__init__(config)
      self.squeeze_block = SqueezeBlock()

  def forward(self, input_dict, labels):
    x, y = input_dict['x'], input_dict['y']
    x = self.squeeze_block(x)
    x_channels = x.size(1)
    concat = torch.cat((x,y), dim=1)
    output = super().forward(concat, labels)
    
    return {'x':self.squeeze_block(output[:,:x_channels,::], reverse=True),\
            'y':output[:,x_channels:,::]}

@utils.register_model(name='ddpm_KxSR')
class ddpm_KxSR(DDPM):
  def __init__(self, config, *args, **kwargs):
      super().__init__(config)
      self.resize_to_GT = Resize(config.data.target_resolution, interpolation=InterpolationMode.BILINEAR)
      self.resize_to_LQ = Resize(config.data.target_resolution//config.data.scale, interpolation=InterpolationMode.BILINEAR)
  
  def forward(self, input_dict, labels):
    x, y = input_dict['x'], input_dict['y']
    y = self.resize_to_GT(y)
    x_channels = x.size(1)
    concat = torch.cat((x,y), dim=1)
    output = super().forward(concat, labels)
    
    return {'x':output[:,:x_channels,::],\
            'y':self.resize_to_LQ(output[:,x_channels:,::])}
