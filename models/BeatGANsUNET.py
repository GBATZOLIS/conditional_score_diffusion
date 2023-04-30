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


@utils.register_model(name='BeatGANsUNetModel')
class BeatGANsUNetModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.conf = config.model

        if self.conf.num_heads_upsample == -1:
            self.num_heads_upsample = self.conf.num_heads

        #self.dtype = th.float32

        self.time_emb_channels = self.conf.time_embed_channels or self.conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, self.conf.embed_channels),
            nn.SiLU(),
            linear(self.conf.embed_channels, self.conf.embed_channels),
        )

        if self.conf.num_classes is not None:
            self.label_emb = nn.Embedding(self.conf.num_classes,
                                          self.conf.embed_channels)

        ch = input_ch = int(self.conf.channel_mult[0] * self.conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(self.conf.dims, self.conf.in_channels, ch, 3, padding=1))
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=self.conf.resnet_two_cond,
            use_zero_module=self.conf.resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=self.conf.resnet_cond_channels,
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(self.conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(self.conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(self.conf.channel_mult))]

        ds = 1
        resolution = self.conf.image_size
        for level, mult in enumerate(self.conf.input_channel_mult
                                     or self.conf.channel_mult):
            for _ in range(self.conf.num_input_res_blocks or self.conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        self.conf.embed_channels,
                        self.conf.dropout,
                        out_channels=int(mult * self.conf.model_channels),
                        dims=self.conf.dims,
                        use_checkpoint=self.conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * self.conf.model_channels)
                if resolution in self.conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.conf.use_checkpoint
                            or self.conf.attn_checkpoint,
                            num_heads=self.conf.num_heads,
                            num_head_channels=self.conf.num_head_channels,
                            use_new_attention_order=self.conf.use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(self.conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            self.conf.embed_channels,
                            self.conf.dropout,
                            out_channels=out_ch,
                            dims=self.conf.dims,
                            use_checkpoint=self.conf.use_checkpoint,
                            down=True,
                            **kwargs,
                        ).make_model() if self.conf.
                        resblock_updown else Downsample(ch,
                                                        self.conf.conv_resample,
                                                        dims=self.conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                self.conf.embed_channels,
                self.conf.dropout,
                dims=self.conf.dims,
                use_checkpoint=self.conf.use_checkpoint,
                **kwargs,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=self.conf.use_checkpoint or self.conf.attn_checkpoint,
                num_heads=self.conf.num_heads,
                num_head_channels=self.conf.num_head_channels,
                use_new_attention_order=self.conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                self.conf.embed_channels,
                self.conf.dropout,
                dims=self.conf.dims,
                use_checkpoint=self.conf.use_checkpoint,
                **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.conf.channel_mult))[::-1]:
            for i in range(self.conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=self.conf.embed_channels,
                        dropout=self.conf.dropout,
                        out_channels=int(self.conf.model_channels * mult),
                        dims=self.conf.dims,
                        use_checkpoint=self.conf.use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(self.conf.model_channels * mult)
                if resolution in self.conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.conf.use_checkpoint
                            or self.conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.conf.num_head_channels,
                            use_new_attention_order=self.conf.use_new_attention_order,
                        ))
                if level and i == self.conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            self.conf.embed_channels,
                            self.conf.dropout,
                            out_channels=out_ch,
                            dims=self.conf.dims,
                            use_checkpoint=self.conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            self.conf.resblock_updown
                        ) else Upsample(ch,
                                        self.conf.conv_resample,
                                        dims=self.conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if self.conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(self.conf.dims,
                            input_ch,
                            self.conf.out_channels,
                            3,
                            padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(self.conf.dims, input_ch, self.conf.out_channels, 3, padding=1),
            )

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return pred
        #return Return(pred=pred)

'''
class Return(NamedTuple):
    pred: th.Tensor
'''


'''
@dataclass
class BeatGANsEncoderConfig(BaseConfig):
    image_size: int
    in_channels: int
    model_channels: int
    out_hid_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    dropout: float = 0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = 'adaptivenonzero'

    def make_model(self):
        return BeatGANsEncoderModel(self)

class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """
    def __init__(self, conf: BeatGANsEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32

        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,
                        use_checkpoint=conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Downsample(ch,
                                          conf.conv_resample,
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
            if self.conf.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb=emb)
        if self.conf.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)

        h_2d = h
        h = self.out(h)

        if return_2d_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h
'''
