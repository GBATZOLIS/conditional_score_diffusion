import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from models.half_U import HalfUEncoder, HalfUDecoder, HalfUDecoderNoConv
from models.encoder import DDPMEncoder, Encoder
from models.decoder import MirrorDecoder
import lpips

class VAE(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.model.latent_dim
        net_options = {
            'simple_encoder': Encoder,
            'simple_decoder': MirrorDecoder,
            'half_U_encoder' : HalfUEncoder,
            'half_U_encoder_no_conv': DDPMEncoder,
            'half_U_decoder': HalfUDecoder,
            'half_U_decoder_no_conv': HalfUDecoderNoConv
        }
        self.encoder = net_options[config.encoder.name](config)
        self.decoder = net_options[config.decoder.name](config)
        self.decoder_sigmoid = nn.Sigmoid()
        self.save_hyperparameters()

    def forward(self, x):
        return x

    def encode(self, x):
        if self.config.model.variational:
            mean_z, log_var_z = self.encoder(x)
            return mean_z, log_var_z
        else:
            z = self.encoder(x)
            return z, None
        

    def decode(self, z):
        out = self.decoder(z)
        mean_x = out
        # Normalization of output. Only makes sense for image data.
        mean_x = self.decoder_sigmoid(mean_x)
        return mean_x, None

    def compute_loss(self, batch):
        B = batch.shape[0]
        D_Z = self.latent_dim
        if self.config.model.variational:
            mean_z, log_var_z = self.encode(batch) # (B, D_Z), (B, D_Z) assuming Sigma_z is diagonal
            z = torch.randn((B, self.latent_dim), device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
            mean_x, _ = self.decode(z) # (B, D_X), (B, D_X) assuming Sigma_x is the identity matrix. (fine when using the kl weight)

            kl_loss =  -0.5 * torch.sum(1 + log_var_z - mean_z ** 2 - log_var_z.exp(), dim=1)
            kl_loss = kl_loss.mean()
            rec_loss = torch.linalg.norm(batch.view(B,-1) - mean_x.view(B,-1), dim=1)
            rec_loss = rec_loss.mean()
            kl_weight = self.config.model.kl_weight
            loss = rec_loss + kl_weight * kl_loss
        else:
            z, _ = self.encode(batch)
            mean_x, _ = self.decode(z)
            rec_loss = torch.linalg.norm(batch.view(B,-1) - mean_x.view(B,-1), dim=1)
            rec_loss = rec_loss.mean()
            kl_loss = torch.tensor(0.)
            loss = rec_loss

        return loss, rec_loss.detach() , kl_loss.detach()


    def training_step(self, batch, batch_idx):
        loss, rec_loss, kl_loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        self.log('train_rec_loss', rec_loss)
        self.log('train_kl_loss', kl_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, rec_loss, kl_loss = self.compute_loss(batch)
        self.log('val_loss', loss)
        self.log('val_rec_loss', rec_loss)
        self.log('val_kl_loss', kl_loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.lpips_distance_fn = lpips.LPIPS(net='vgg').to(self.device) 
            
        z, _ = self.encode(batch)
        reconstruction, _ = self.decode(z)

        avg_lpips_score = torch.mean(self.lpips_distance_fn(reconstruction, batch))

        difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(batch, start_dim=1)
        L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
        avg_L2norm = torch.mean(L2norm)

        self.log("LPIPS", avg_lpips_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("L2", avg_L2norm, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output = dict({
        'LPIPS': avg_lpips_score,
        'L2': avg_L2norm,
        })
        return output
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.config.optim.lr)
        if self.config.optim.use_scheduler:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                            factor=self.config.optim.sch_factor, 
                                                            patience=self.config.optim.sch_patience, 
                                                            min_lr=self.config.optim.sch_min_lr)
            return {'optimizer': optim, 
                    "lr_scheduler" : {
                        "scheduler" : sch,
                        "monitor" : "train_loss",
                    }
                }
        else:
            return optim