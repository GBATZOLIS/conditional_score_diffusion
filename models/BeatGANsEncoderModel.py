import math
from numbers import Number
from typing import NamedTuple, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from .BeatGANs_choices import *
from .config_base import BaseConfig
from .BeatGANsblocks import *
from . import utils
import pytorch_lightning as pl
from .BeatGANs_nn import (conv_nd, linear, normalization, timestep_embedding,
                 torch_checkpoint, zero_module)
from . import utils

@utils.register_model(name='BeatGANsEncoderModel')
class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """
    def __init__(self, config):
        super().__init__()
        self.conf = config.model
        self.dtype = th.float32

        if self.conf.enc_use_time_condition:
            time_embed_dim = self.conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(self.conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(self.conf.enc_channel_mult[0] * self.conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(self.conf.dims, self.conf.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = self.conf.image_size
        for level, mult in enumerate(self.conf.enc_channel_mult):
            for _ in range(self.conf.enc_num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        self.conf.dropout,
                        out_channels=int(mult * self.conf.model_channels),
                        dims=self.conf.dims,
                        use_condition=self.conf.enc_use_time_condition,
                        use_checkpoint=self.conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * self.conf.model_channels)
                if resolution in self.conf.enc_attn_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.conf.use_checkpoint,
                            num_heads=self.conf.num_heads,
                            num_head_channels=self.conf.num_head_channels,
                            use_new_attention_order=self.conf.use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.conf.enc_channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            self.conf.dropout,
                            out_channels=out_ch,
                            dims=self.conf.dims,
                            use_condition=self.conf.enc_use_time_condition,
                            use_checkpoint=self.conf.use_checkpoint,
                            down=True,
                        ).make_model() if (
                            self.conf.resblock_updown
                        ) else Downsample(ch,
                                          self.conf.conv_resample,
                                          dims=self.conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                self.conf.dropout,
                dims=self.conf.dims,
                use_condition=self.conf.enc_use_time_condition,
                use_checkpoint=self.conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=self.conf.use_checkpoint,
                num_heads=self.conf.num_heads,
                num_head_channels=self.conf.num_head_channels,
                use_new_attention_order=self.conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                self.conf.dropout,
                dims=self.conf.dims,
                use_condition=self.conf.enc_use_time_condition,
                use_checkpoint=self.conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch
        if self.conf.enc_pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(self.conf.dims, ch, self.conf.enc_out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {self.conf.enc_pool} pooling")

    def forward(self, x, t=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if self.conf.enc_use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.conf.model_channels))
        else:
            emb = None

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
            if self.conf.enc_pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb=emb)
        if self.conf.enc_pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)

        #h_2d = h
        h = self.out(h)

        if self.conf.encoder_split_output:
            return h[:,:self.conf.latent_dim], h[:,self.conf.latent_dim:]
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h