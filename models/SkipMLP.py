import torch
import torch.nn as nn
import torch.nn.init as init
from enum import Enum
from typing import NamedTuple, Tuple
import math
from . import utils

class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    #timesteps = timesteps.squeeze(1)

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class LatentNetType(Enum):
    none = 'none'
    skip = 'skip'

class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None

@utils.register_model(name='MLPSkipNet')
class MLPSkipNet(nn.Module):
    """
    Concat x to hidden layers.
    """
    def __init__(self, config):
        super().__init__()
        self.conf = config.MI_estimator if hasattr(config, 'MI_estimator') else config.model
        self.input_channels = input_channels = self.conf.input_channels
        self.output_channels = output_channels = self.conf.output_channels
        self.num_channels = num_channels = self.conf.num_channels
        self.num_layers = num_layers = self.conf.num_layers
        self.skip_layers = skip_layers = self.conf.skip_layers
        self.num_hid_channels = num_hid_channels = self.conf.num_hid_channels
        self.num_time_emb_channels = num_time_emb_channels = self.conf.num_time_emb_channels
        self.activation = activation = Activation.silu
        self.use_norm = use_norm = self.conf.use_norm
        self.condition_bias = condition_bias = self.conf.condition_bias
        self.dropout = dropout = self.conf.dropout
        self.last_act = last_act = Activation.none
        self.num_time_layers = num_time_layers = self.conf.num_time_layers
        self.time_last_act = time_last_act = self.conf.time_last_act

        # Time embedding layers
        time_layers = []
        for i in range(num_time_layers):
            in_channels = num_time_emb_channels if i == 0 else num_channels
            time_layers.append(nn.Linear(in_channels, num_channels))
            if i < num_time_layers - 1 or time_last_act:
                time_layers.append(activation.get_act())
        self.time_embed = nn.Sequential(*time_layers)

        # Main layers
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else num_hid_channels
            out_channels = num_channels if i == num_layers - 1 else num_hid_channels
            act = last_act if i == num_layers - 1 else activation
            norm = use_norm and i != num_layers - 1
            cond = i != num_layers - 1
            if i in skip_layers:
                in_channels += num_channels
            self.layers.append(
                MLPLNAct(
                    in_channels,
                    out_channels,
                    norm=norm,
                    activation=act,
                    cond_channels=num_channels,
                    use_cond=cond,
                    condition_bias=condition_bias,
                    dropout=dropout,
                ))

    def forward(self, input, t):
        if type(input)==dict:
            x, y = input['x'], input['y']
        else:
            x, y = input, None

        t = timestep_embedding(t, self.num_time_emb_channels)
        cond = self.time_embed(t)
        h = torch.cat([x, y], dim=1) if y is not None else x
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = layer(h, cond)
        return h


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: Activation,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x