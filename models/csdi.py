import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import numpy as np
from . import utils


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_dim=128, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_dim // 2) * scale, requires_grad=False)
    self.linear1 = nn.Linear(embedding_dim, embedding_dim)
    self.linear2 = nn.Linear(embedding_dim, embedding_dim)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    x = self.linear1(x)
    x = F.silu(x)
    x = self.linear2(x)
    x = F.silu(x)
    return x

class diff_CSDI(pl.LightningModule):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.L_1 = config.data.L_1
        self.L_2 = config.data.L_2
        self.L = self.L_1 + self.L_2
        self.K = config.data.shape[1]
        self.channels = config.model.num_channels

        self.diffusion_embedding = GaussianFourierProjection(
            embedding_dim=config.model.diff_embedding_dim,
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        self.output_projection3 = nn.Linear(self.K * self.L, self.L_2)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config.model.time_embedding_dim + config.model.feature_embedding_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config.model.diff_embedding_dim,
                    nheads=config.model.nheads,
                )
                for _ in range(config.model.num_layers)
            ]
        )

    def forward(self,  x, side_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = self.output_projection3(x.squeeze(1)).reshape(B, K, self.L_2) # (B, K, L_2)
        return x

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, cond_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

@utils.register_model(name='csdi_conditional')
class CSDI(pl.LightningModule):

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = diff_CSDI(config)
        self.emb_time_dim = config.model.time_embedding_dim
        self.emb_feature_dim = config.model.feature_embedding_dim
        self.target_dim = config.data.shape[-1]
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )


    def forward(self, input_dict, t):
        labels = input_dict['x'].permute(0,2,1) # (B, K, L_2)
        y = input_dict['y']
        inpt = y['input'].permute(0,2,1) # (B, K, L_1)
        timesteps = y['timesteps'] # (L)

        B = inpt.shape[0]
        K = inpt.shape[1]
        L_1 = inpt.shape[2]
        L_2 = labels.shape[2]
        L = L_1 + L_2

        mask = torch.cat([torch.ones((B,K, L_1)), torch.zeros((B,K, L_2))], dim=2).to(self.device) # (B, K, L)
        observed_data = torch.cat([inpt, labels], dim=2) # wrong name
        condition = (mask * observed_data).unsqueeze(1)
        perturbed_labels = ((1-mask) * observed_data).unsqueeze(1)
        side_info = self.get_side_info(timesteps, mask) # (B,4,K,L)
        total_input = torch.cat([condition, perturbed_labels], dim=1) # (B,2,K,L)
        score = self.model(total_input, side_info, t) # (B, K, L_2)
        return score.permute(0,2,1) #(B, L_2, K)

    def get_side_info(self, observed_tp, cond_mask):
        # time_embedding, feature_embedding, cond_mask

        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

