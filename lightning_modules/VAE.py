import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from models.half_U import HalfUEncoder, HalfUDecoder, HalfUDecoderNoConv
from models.encoder import DDPMEncoder, Encoder
from models.decoder import MirrorDecoder
import lpips
from models import utils as mutils
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

# for loading old models
#def fix_config(config):

class MPVisualizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.lpips_distance_fn = lpips.LPIPS(net='vgg')
    
    def setup(self, trainer, pl_module, stage):
        self.lpips_distance_fn = self.lpips_distance_fn.to(pl_module.device)

    def generate_visualize(self, trainer, pl_module, batch, discretisation, current_epoch):
        if pl_module.config.training.sb_latent_conditioned:
            x, x_hat, z = pl_module.get_pair(batch)
            x_hat_sb = pl_module.ddpm_sample(x_hat, z, discretisation=discretisation)
        else:
            x, x_hat = pl_module.get_pair(batch)
            x_hat_sb = pl_module.ddpm_sample(x_hat, discretisation=discretisation)

        #calculate the average LPIPS score
        avg_lpips_score = torch.mean(self.lpips_distance_fn(x, x_hat_sb))
        trainer.logger.experiment.add_scalar('avg_lpips_score_%d' % discretisation, avg_lpips_score, global_step=current_epoch)

        # Visualize the first 16 images in the batch
        x = x[:16]
        x_hat = x_hat[:16]
        x_hat_sb = x_hat_sb[:16]
            
        # Concatenate tensors along a new dimension and log them
        img_grid = torch.stack((x, x_hat, x_hat_sb), dim=1)  # shape: [16, 3, H, W]
        img_grid = img_grid.view(-1, 3, 32, 32) # reshape to [48, H, W]
        grid = make_grid(img_grid, nrow=3, normalize=True, scale_each=True)  # Creates a grid with 3 images in each row
        trainer.logger.experiment.add_image('conditional_%d' % discretisation, grid, current_epoch)


    def on_validation_epoch_end(self, trainer, pl_module):
        self.lpips_distance_fn = self.lpips_distance_fn.to(pl_module.device)
        current_epoch = trainer.current_epoch
        if hasattr(pl_module.config.training, 'visualisation_freq'):
            freq = pl_module.config.training.visualisation_freq
        else:
            freq = 10
        
        if (trainer.current_epoch+1) % freq == 0:
            dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(dataloader))
            batch = batch.to(pl_module.device)

            self.generate_visualize(trainer, pl_module, batch, 4, current_epoch)
            self.generate_visualize(trainer, pl_module, batch, 8, current_epoch)
            self.generate_visualize(trainer, pl_module, batch, 16, current_epoch)
            self.generate_visualize(trainer, pl_module, batch, 32, current_epoch)
            self.generate_visualize(trainer, pl_module, batch, 64, current_epoch)

        if (trainer.current_epoch+1) % 50 == 0:
            unconditional_sample = pl_module.unconditional_sample(batch.size(0),  discretisation=32)
            grid = make_grid(unconditional_sample, nrow=int(np.sqrt(unconditional_sample.size(0))), normalize=True, scale_each=True)
            trainer.logger.experiment.add_image('unconditional', grid, current_epoch)


class FullMarkovianProjector(pl.LightningModule):
    def __init__(self, config, vae_model):
        super(FullMarkovianProjector, self).__init__()
        self.config = config
        self.vae = vae_model
        self.vae.freeze()
        self.model = mutils.create_model(config)

        self.t0 = torch.tensor(0).to(self.device)
        self.t1 = torch.tensor(1).to(self.device)
    
    def vae_decode(self, z):
        mean_x, _ = self.vae.decode(z)
        if self.config.stochastic_decoder:
            x_hat = mean_x + 0.1*torch.randn_like(mean_x)
        else:
            x_hat = mean_x
        return x_hat

    def get_pair(self, batch):
        B = batch.shape[0]
        mean_z, log_var_z = self.vae.encode(batch)
        z = torch.randn((B, mean_z.size(1)), device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
        x_hat = self.vae_decode(z)
        return batch, x_hat, z

    def _f(self, x, t):
        return torch.zeros_like(x)
    
    def _beta(self, t):
        beta_min = self.config.training.beta_min
        beta_max = self.config.training.beta_max
        increase = beta_min + 2 * t * (beta_max - beta_min)
        decrease = beta_max - 2 * (t - 0.5) * (beta_max - beta_min)
        return torch.where(t <= 0.5, increase, decrease)

    def _g(self, t):
        return torch.sqrt(self._beta(t))
    
    def forward_variance_accumulation(self, t):
        #int_0_t_{beta(t')dt'}
        beta_min = self.config.training.beta_min
        beta_max = self.config.training.beta_max
        first_part = beta_min * t + (beta_max - beta_min) * t**2

        first_part_integral = beta_min * 0.5 + (beta_max - beta_min) * 0.5**2
        second_part = first_part_integral + beta_max * (t-0.5) - (beta_max - beta_min) * (t-0.5)**2

        return torch.where(t <= 0.5, first_part, second_part)

    def backward_variance_accumulation(self, t):
        #int_t_1_{beta(t')dt'}
        beta_min = self.config.training.beta_min
        beta_max = self.config.training.beta_max
        first_part = beta_max * (1-t) - (beta_max - beta_min) * (0.5**2-(t-0.5)**2)
        first_part_integral = beta_max * 0.5 - (beta_max - beta_min) * 0.5**2
        second_part = first_part_integral + beta_min * (0.5-t) + (beta_max - beta_min) * (0.5**2-t**2)
        return torch.where(t >= 0.5, first_part, second_part)
    
    def get_sample_from_posterior_given_pair(self, t, x0, x1):
        s_t_2 = self.forward_variance_accumulation(t)
        bar_s_t_2 = self.backward_variance_accumulation(t)
        var_sum = s_t_2 + bar_s_t_2
        x0_factor = bar_s_t_2/var_sum
        x1_factor = s_t_2/var_sum
        mean_t = x0_factor[(...,) + (None,) * len(x0.shape[1:])] * x0 + x1_factor[(...,) + (None,) * len(x1.shape[1:])] * x1
        std_t = torch.sqrt(s_t_2 * bar_s_t_2 / var_sum)
        x_t = mean_t + std_t[(...,) + (None,) * len(x0.shape[1:])] * torch.randn_like(x0).to(self.device)
        return x_t
        
    def training_step(self, batch, batch_idx):
        x, x_hat, z = self.get_pair(batch)
        loss = self.compute_loss(x, x_hat, z)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_hat, z = self.get_pair(batch)
        loss = self.compute_loss(x, x_hat, z)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def compute_loss(self, x0, x1, z):
        t0 = self.t0 + torch.tensor(1e-6)
        t1 = self.t1
        t = torch.rand(x0.size(0)).to(self.device) * (t1 - t0) + t0 
        xt = self.get_sample_from_posterior_given_pair(t, x0, x1)
        sigma_t = torch.sqrt(self.forward_variance_accumulation(t))
        model_input = {'x':xt, 'y':z}
        epsilon = self.model(model_input, t)
        loss = epsilon - (xt - x0)/sigma_t[(...,) + (None,) * len(x0.shape[1:])]
        loss = torch.mean(torch.square(loss))
        return loss
    
    @torch.no_grad()
    def ddpm_sample(self, x_hat, z, discretisation, clip_denoise=True): #sample from p(x|x_hat, z)
        data_min, data_max = self.config.data.range[0], self.config.data.range[1]
        x1 = x_hat.to(self.device)
        ts = torch.linspace(self.t1, self.t0, discretisation+1).to(self.device)
        dt = ts[1]-ts[0] #negative dtimestep
        for t in ts[:-1]:
            s_t = torch.sqrt(self.forward_variance_accumulation(t))
            #print(t.size())
            model_input = {'x':x1, 'y':z}
            x0_e = x1 - s_t*self.model(model_input, t.repeat(x1.size(0)))
            if clip_denoise: x0_e.clamp_(data_min, data_max)

            if torch.abs((t+dt) - self.t0) <= torch.tensor(1e-6):
                x_t_plus_dt = x0_e
            else:
                x_t_plus_dt = self.get_sample_from_posterior_given_pair(t+dt, x0_e, x1)
            
            x1 = x_t_plus_dt
        return x1
    
    @torch.no_grad()
    def unconditional_sample(self, num_samples, discretisation):
        z = torch.randn((num_samples, self.vae.latent_dim)).to(self.device)
        x_hat = self.vae_decode(z)
        x = self.ddpm_sample(x_hat, z, discretisation)
        return x
    
    @torch.no_grad()
    def encoder_n_decode(self, x, discretisation):
        mean_z, log_var_z = self.vae.encode(x)
        z = torch.randn_like(mean_z).to(self.device) * torch.sqrt(log_var_z.exp()) + mean_z
        x_hat = self.vae_decode(z)
        x = self.ddpm_sample(x_hat, z, discretisation)
        return x
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.config.optim.lr)
        if self.config.optim.use_scheduler:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                            factor=self.config.optim.sch_factor, 
                                                            patience=self.config.optim.sch_patience, 
                                                            min_lr=self.config.optim.sch_min_lr)
            return {'optimizer': optim, 
                    "lr_scheduler" : {
                        "scheduler" : sch,
                        "monitor" : "val_loss",
                    }
                }
        else:
            return optim

class MarkovianProjector(pl.LightningModule):
    def __init__(self, config, vae_model):
        super(MarkovianProjector, self).__init__()
        self.config = config
        self.vae = vae_model
        self.vae.freeze()
        self.model = mutils.create_model(config)

        self.t0 = torch.tensor(0).to(self.device)
        self.t1 = torch.tensor(1).to(self.device)
        self.sigma = config.training.sigma
    
    def vae_decode(self, z):
        mean_x, _ = self.vae.decode(z)
        if self.config.stochastic_decoder:
            x_hat = mean_x + 0.1*torch.randn_like(mean_x)
        else:
            x_hat = mean_x
        return x_hat

    def get_pair(self, batch):
        B = batch.shape[0]
        mean_z, log_var_z = self.vae.encode(batch)
        z = torch.randn((B, mean_z.size(1)), device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
        x_hat = self.vae_decode(z)
        return batch, x_hat

    def _f(self, x, t):
        return torch.zeros_like(x)
    
    def _beta(self, t):
        beta_min = self.config.training.beta_min
        beta_max = self.config.training.beta_max
        increase = beta_min + 2 * t * (beta_max - beta_min)
        decrease = beta_max - 2 * (t - 0.5) * (beta_max - beta_min)
        return torch.where(t <= 0.5, increase, decrease)

    def _g(self, t):
        return torch.sqrt(self._beta(t))
    
    def forward_variance_accumulation(self, t):
        #int_0_t_{beta(t')dt'}
        beta_min = self.config.training.beta_min
        beta_max = self.config.training.beta_max
        first_part = beta_min * t + (beta_max - beta_min) * t**2

        first_part_integral = beta_min * 0.5 + (beta_max - beta_min) * 0.5**2
        second_part = first_part_integral + beta_max * (t-0.5) - (beta_max - beta_min) * (t-0.5)**2

        return torch.where(t <= 0.5, first_part, second_part)

    def backward_variance_accumulation(self, t):
        #int_t_1_{beta(t')dt'}
        beta_min = self.config.training.beta_min
        beta_max = self.config.training.beta_max
        first_part = beta_max * (1-t) - (beta_max - beta_min) * (0.5**2-(t-0.5)**2)
        first_part_integral = beta_max * 0.5 - (beta_max - beta_min) * 0.5**2
        second_part = first_part_integral + beta_min * (0.5-t) + (beta_max - beta_min) * (0.5**2-t**2)
        return torch.where(t >= 0.5, first_part, second_part)
    
    def get_sample_from_posterior_given_pair(self, t, x0, x1):
        s_t_2 = self.forward_variance_accumulation(t)
        bar_s_t_2 = self.backward_variance_accumulation(t)
        var_sum = s_t_2 + bar_s_t_2
        x0_factor = bar_s_t_2/var_sum
        x1_factor = s_t_2/var_sum
        mean_t = x0_factor[(...,) + (None,) * len(x0.shape[1:])] * x0 + x1_factor[(...,) + (None,) * len(x1.shape[1:])] * x1
        std_t = torch.sqrt(s_t_2 * bar_s_t_2 / var_sum)
        x_t = mean_t + std_t[(...,) + (None,) * len(x0.shape[1:])] * torch.randn_like(x0).to(self.device)
        return x_t
        
    def training_step(self, batch, batch_idx):
        x, x_hat = self.get_pair(batch)
        loss = self.compute_loss(x, x_hat)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_hat = self.get_pair(batch)
        loss = self.compute_loss(x, x_hat)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def compute_loss(self, x0, x1):
        t0 = self.t0 + torch.tensor(1e-6)
        t1 = self.t1
        t = torch.rand(x0.size(0)).to(self.device) * (t1 - t0) + t0 
        xt = self.get_sample_from_posterior_given_pair(t, x0, x1)
        sigma_t = torch.sqrt(self.forward_variance_accumulation(t))
        epsilon = self.model(xt, t)
        loss = epsilon - (xt - x0)/sigma_t[(...,) + (None,) * len(x0.shape[1:])]
        loss = torch.mean(torch.square(loss))
        return loss
    
    @torch.no_grad()
    def ddpm_sample(self, x_hat, discretisation, clip_denoise=True): #sample from p(x|x_hat)
        data_min, data_max = self.config.data.range[0], self.config.data.range[1]
        x1 = x_hat.to(self.device)
        ts = torch.linspace(self.t1, self.t0, discretisation+1).to(self.device)
        dt = ts[1]-ts[0] #negative dtimestep
        for t in ts[:-1]:
            s_t = torch.sqrt(self.forward_variance_accumulation(t))
            x0_e = x1 - s_t*self.model(x1, t.repeat(x1.size(0)))
            if clip_denoise: x0_e.clamp_(data_min, data_max)

            if torch.abs((t+dt) - self.t0) <= torch.tensor(1e-6):
                x_t_plus_dt = x0_e
            else:
                x_t_plus_dt = self.get_sample_from_posterior_given_pair(t+dt, x0_e, x1)
            
            x1 = x_t_plus_dt
        return x1
    
    @torch.no_grad()
    def unconditional_sample(self, num_samples, discretisation):
        z = torch.randn((num_samples, self.vae.latent_dim)).to(self.device)
        x_hat = self.vae_decode(z)
        x = self.ddpm_sample(x_hat, discretisation)
        return x
    
    @torch.no_grad()
    def encoder_n_decode(self, x, discretisation):
        mean_z, log_var_z = self.vae.encode(x)
        z = torch.randn_like(mean_z).to(self.device) * torch.sqrt(log_var_z.exp()) + mean_z
        x_hat = self.vae_decode(z)
        x = self.ddpm_sample(x_hat, discretisation)
        return x
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.config.optim.lr)
        if self.config.optim.use_scheduler:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                            factor=self.config.optim.sch_factor, 
                                                            patience=self.config.optim.sch_patience, 
                                                            min_lr=self.config.optim.sch_min_lr)
            return {'optimizer': optim, 
                    "lr_scheduler" : {
                        "scheduler" : sch,
                        "monitor" : "val_loss",
                    }
                }
        else:
            return optim

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
        if self.config.training.variational:
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

    def reconstruct(self, x, use_latent_mean=True):
        mean_z, log_var_z = self.encode(x)
        if use_latent_mean:
            z = mean_z
        else:
            z = torch.randn_like(mean_z, device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
        mean_x, _ = self.decode(z)
        return mean_x

    def compute_loss(self, batch):
        B = batch.shape[0]
        D_Z = self.latent_dim
        if self.config.training.variational:
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
                        "monitor" : "val_loss",
                    }
                }
        else:
            return optim