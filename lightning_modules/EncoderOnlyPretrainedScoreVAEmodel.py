from . import BaseSdeGenerativeModel
from losses import get_scoreVAE_loss_fn, get_old_scoreVAE_loss_fn, get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from models import utils as mutils
from . import utils
import torch.optim as optim
import os
import torch
from sde_lib import cVPSDE, csubVPSDE, cVESDE
from sampling.conditional import get_conditional_sampling_fn
from sampling.unconditional import get_sampling_fn
import torchvision
import numpy as np
import losses
import torch
import lpips
from pathlib import Path
from utils import get_named_beta_schedule
from scipy.interpolate import PchipInterpolator
import numpy as np
from torch.distributions import Uniform

@utils.register_lightning_module(name='encoder_only_pretrained_score_vae')
class EncoderOnlyPretrainedScoreVAEmodel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the score model
        self.config = config
        
        if not config.training.encoder_only:
            #latent correction model
            config.model.input_channels = 2*config.model.output_channels
            self.latent_correction_model = mutils.create_model(config)
        
        #unconditional score model
        if config.training.use_pretrained:
            self.unconditional_score_model = mutils.load_prior_model(config)
            self.unconditional_score_model.freeze()
        else:
            config.model.input_channels = config.model.output_channels
            config.model.name = config.model.unconditional_score_model_name
            self.unconditional_score_model = mutils.create_model(config)

        #encoder
        self.encoder = mutils.create_encoder(config)

        # validation batch
        # register buffer 
        self.register_buffer('val_batch', torch.zeros(tuple([config.validation.batch_size]+config.data.shape)))

    def on_load_checkpoint(self, checkpoint):
        # Load the existing state dictionary for all model components
        self.load_state_dict(checkpoint['state_dict'])

        # Get the size of the 'val_batch' tensor from the checkpoint
        val_batch_size = checkpoint['state_dict']['val_batch'].size()

        # Now, re-register the 'val_batch' buffer with the new size
        self.register_buffer('val_batch', torch.zeros(val_batch_size))

        # Finally, load the state dictionary again, this time it should work without size mismatch errors
        self.load_state_dict(checkpoint['state_dict'])

    def configure_sde(self, config):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.cVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.usde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.csubVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.usde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':            
            self.sde = sde_lib.cVESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales)
            self.usde = sde_lib.VESDE(sigma_min=config.model.sigma_min_x, sigma_max=config.model.sigma_max_x, N=config.model.num_scales)
            self.sampling_eps = 1e-5   
        elif config.training.sde.lower() == 'snrsde':
            self.sampling_eps = 1e-3

            if hasattr(config.training, 'beta_schedule'):
                #DISCRETE QUANTITIES
                N = config.model.num_scales
                betas = get_named_beta_schedule('linear', N)
                alphas = 1.0 - betas
                alphas_cumprod = np.cumprod(alphas, axis=0)
                discrete_snrs = alphas_cumprod/(1.0 - alphas_cumprod)

                #Monotonic Bicubic Interpolation
                snr = PchipInterpolator(np.linspace(self.sampling_eps, 1, len(discrete_snrs)), discrete_snrs)
                d_snr = snr.derivative(nu=1)

                def logsnr(t):
                    device = t.device
                    snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                    return torch.log(snr_val)

                def d_logsnr(t):
                    device = t.device
                    dsnr_val = torch.from_numpy(d_snr(t.cpu().numpy())).float().to(device)
                    snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                    return dsnr_val/snr_val

                self.sde = sde_lib.cSNRSDE(N=N, gamma=logsnr, dgamma=d_logsnr)
                self.usde = sde_lib.SNRSDE(N=N, gamma=logsnr, dgamma=d_logsnr)
            else:
                self.sde = sde_lib.cSNRSDE(N=config.model.num_scales)
                self.usde = sde_lib.SNRSDE(N=config.model.num_scales)

        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde.sampling_eps = self.sampling_eps
        self.usde.sampling_eps = self.sampling_eps

        self.t_dist = Uniform(self.sampling_eps, 1)
    
    def _handle_batch(self, batch):
        if type(batch) == list:
            x = batch[0]
        else:
            x = batch
        return x

    def configure_loss_fn(self, config, train):
        if hasattr(config.training, 'cde_loss'):
            if config.training.cde_loss:
                loss_fn = get_old_scoreVAE_loss_fn(self.sde, train, 
                                        variational=config.training.variational, 
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        eps=self.sampling_eps,
                                        use_pretrained=config.training.use_pretrained,
                                        encoder_only = config.training.encoder_only,
                                        t_dependent = config.training.t_dependent)
            else:
                loss_fn = get_scoreVAE_loss_fn(self.sde, train, 
                                            variational=config.training.variational, 
                                            likelihood_weighting=config.training.likelihood_weighting,
                                            eps=self.sampling_eps,
                                            t_batch_size=config.training.t_batch_size,
                                            kl_weight=config.training.kl_weight,
                                            use_pretrained=config.training.use_pretrained,
                                            encoder_only = config.training.encoder_only,
                                            t_dependent = config.training.t_dependent)
        else:
            loss_fn = get_scoreVAE_loss_fn(self.sde, train, 
                                        variational=config.training.variational, 
                                        likelihood_weighting=config.training.likelihood_weighting,
                                        eps=self.sampling_eps,
                                        t_batch_size=config.training.t_batch_size,
                                        kl_weight=config.training.kl_weight,
                                        use_pretrained=config.training.use_pretrained,
                                        encoder_only = config.training.encoder_only,
                                        t_dependent = config.training.t_dependent)
        
        if config.training.use_pretrained:
            return {0:loss_fn}
        else:
            unconditional_loss_fn = get_general_sde_loss_fn(self.usde, train, conditional=False, reduce_mean=True,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)
            return {0:loss_fn, 1:unconditional_loss_fn}
    
    def training_step(self, *args, **kwargs):
        batch, batch_idx = args[0], args[1]
        if hasattr(self.config, 'debug') and self.config.debug.skip_training and batch_idx > 1:
            return None
        batch = self._handle_batch(batch)

        if self.config.training.use_pretrained:
            if self.config.training.encoder_only:
                loss = self.train_loss_fn[0](self.encoder, self.unconditional_score_model, batch, self.t_dist)
            else:
                loss = self.train_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            optimizer_idx = args[2]
            if optimizer_idx == 0:
                loss = self.train_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
                if self.config.training.use_pretrained:
                    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                else:
                    self.logger.experiment.add_scalars('train_loss', {'conditional': loss}, self.global_step)

            elif optimizer_idx == 1:
                loss = self.train_loss_fn[1](self.unconditional_score_model, batch)
                self.logger.experiment.add_scalars('train_loss', {'unconditional': loss}, self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self._handle_batch(batch)
        if hasattr(self.config, 'debug') and self.config.debug.skip_validation and batch_idx > 1:
            return None
        if self.config.training.use_pretrained:
            if self.config.training.encoder_only:
                loss = self.eval_loss_fn[0](self.encoder, self.unconditional_score_model, batch, self.t_dist)
            else:
                loss = self.eval_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = self.eval_loss_fn[0](self.encoder, self.latent_correction_model, self.unconditional_score_model, batch)
            self.logger.experiment.add_scalars('val_loss', {'conditional': loss}, self.global_step)
            self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            loss = self.eval_loss_fn[1](self.unconditional_score_model, batch)
            self.logger.experiment.add_scalars('val_loss', {'unconditional': loss}, self.global_step)

        if batch_idx == 2 and self.current_epoch+1 == 2:
            sample, _ = self.unconditional_sample(p_steps=250)
            sample = sample.cpu()
            grid_batch = torchvision.utils.make_grid(sample, nrow=int(np.sqrt(sample.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('unconditional_sample', grid_batch, self.current_epoch)

        # visualize reconstruction
        if batch_idx == 2 and (self.current_epoch+1) % self.config.training.visualisation_freq == 0:
            if torch.all(self.val_batch == 0) or batch.size(0) != self.val_batch.size(0):
                self.val_batch = batch
            else:
                batch = self.val_batch
                
            reconstruction = self.encode_n_decode(batch, p_steps=250,
                                                         use_pretrained=self.config.training.use_pretrained,
                                                         encoder_only=self.config.training.encoder_only,
                                                         t_dependent=self.config.training.t_dependent)

            reconstruction =  reconstruction.cpu()
            grid_reconstruction = torchvision.utils.make_grid(reconstruction, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('reconstruction', grid_reconstruction, self.current_epoch)
            
            batch = batch.cpu()
            grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
            self.logger.experiment.add_image('real', grid_batch)

            difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(batch, start_dim=1)
            L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
            avg_L2norm = torch.mean(L2norm)
            self.log('reconstruction_loss', avg_L2norm, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        batch = self._handle_batch(batch)
        reconstruction = self.encode_n_decode(batch, use_pretrained=self.config.training.use_pretrained,
                                                          encoder_only=self.config.training.encoder_only,
                                                          t_dependent=self.config.training.t_dependent)
        if batch_idx == 0:
            #save the first batch and its reconstruction
            log_path = self.config.logging.log_path
            log_name = self.config.logging.log_name

            base_save_path = os.path.join(log_path, log_name, 'images')
            Path(base_save_path).mkdir(parents=True, exist_ok=True)

            original_save_path = os.path.join(log_path, log_name, 'images', 'original')
            Path(original_save_path).mkdir(parents=True, exist_ok=True)

            reconstruction_save_path = os.path.join(log_path, log_name, 'images', 'reconstructions')
            Path(reconstruction_save_path).mkdir(parents=True, exist_ok=True)

            for i in range(batch.size(0)):
                torchvision.utils.save_image(batch[i, :, :, :], os.path.join(original_save_path,'{}.png'.format(i+1)))
            
            for i in range(batch.size(0)):
                a = reconstruction[i, :, :, :]
                min_a, max_a = a.min(), a.max()
                a -= min_a
                a /= max_a - min_a
                torchvision.utils.save_image(a, os.path.join(reconstruction_save_path,'{}.png'.format(i+1)))

        self.lpips_distance_fn = lpips.LPIPS(net='vgg').to(self.device)
        avg_lpips_score = torch.mean(self.lpips_distance_fn(reconstruction.to(self.device), batch))

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


    def unconditional_sample(self, num_samples=None, p_steps='default'):
        if num_samples is None:
            sampling_shape = [self.config.training.batch_size] +  self.config.data.shape
        else:
            sampling_shape = [num_samples] +  self.config.data.shape
        sampling_fn = get_sampling_fn(self.config, self.usde, sampling_shape, self.sampling_eps, p_steps)
        return sampling_fn(self.unconditional_score_model)
    
    def interpolate(self, x, num_points):
        def slerp(low, high, val):
            low_norm = low/torch.linalg.norm(low, ord=2)
            high_norm = high/torch.linalg.norm(high, ord=2)
            omega = torch.acos((low_norm*high_norm).sum())
            so = torch.sin(omega)
            res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
            return res

        assert x.size(0) == 2

        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = self.encoder(x, t0)
        latent_dim = latent_distribution_parameters.size(1)//2
        mean_y = latent_distribution_parameters[:, :latent_dim]
        log_var_y = latent_distribution_parameters[:, latent_dim:]
        y = mean_y + torch.sqrt(log_var_y.exp()) * torch.randn_like(mean_y)

        weights = torch.linspace(0, 1, steps=num_points+2).to(self.device)
        z = torch.zeros(size=(num_points, y.size(1))).type_as(x)    
        for i in range(weights.size(0)-2):
            z[i] = slerp(y[0], y[1], weights[i+1])
        
        sampling_shape = [z.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps, 
                                                              predictor='default', corrector='default', 
                                                              p_steps='default', c_steps='default', snr='default', 
                                                              denoise='default', use_path=False, 
                                                              use_pretrained=self.config.training.use_pretrained, 
                                                              encoder_only=self.config.training.encoder_only, 
                                                              t_dependent=self.config.training.t_dependent)

        model = {'unconditional_score_model':self.unconditional_score_model,
                 'encoder': self.encoder}
        
        interpolation = conditional_sampling_fn(model, z, False)

        augmented = torch.zeros(size=tuple([num_points+2]+self.config.data.shape))
        augmented[0] =  x[0]
        augmented[1:-1] = interpolation
        augmented[-1] = x[1]
        return augmented

    def encode(self, x, use_latent_mean=False):
        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = self.encoder(x, t0)
        latent_dim = latent_distribution_parameters.size(1)//2
        mean_y = latent_distribution_parameters[:, :latent_dim]
        log_var_y = latent_distribution_parameters[:, latent_dim:]
        if use_latent_mean:
            y = mean_y
        else:
            y = mean_y + torch.sqrt(log_var_y.exp()) * torch.randn_like(mean_y)
        return y
    
    def decode(self, z):
        use_pretrained=self.config.training.use_pretrained,
        encoder_only=self.config.training.encoder_only,
        t_dependent=self.config.training.t_dependent
        show_evolution=False
        sampling_shape = [z.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps, 
                                                              use_pretrained=use_pretrained, encoder_only=encoder_only, 
                                                              use_path=False, 
                                                              t_dependent=t_dependent)
        if self.config.training.encoder_only:
            model = {'unconditional_score_model':self.unconditional_score_model,
                     'encoder': self.encoder}
        else:
            model = {'unconditional_score_model':self.unconditional_score_model,
                     'latent_correction_model': self.latent_correction_model}

        return conditional_sampling_fn(model, z, show_evolution)


    def encode_n_decode(self, x, show_evolution=False, predictor='default', corrector='default', p_steps='default', \
                     c_steps='default', snr='default', denoise='default', use_pretrained=True, encoder_only=False, t_dependent=True, gamma=1.):

        if self.config.training.variational:
            if self.config.training.encoder_only:
                if t_dependent:
                    t0 = torch.zeros(x.shape[0]).type_as(x)
                    latent_distribution_parameters = self.encoder(x, t0)
                else:
                    latent_distribution_parameters = self.encoder(x)

                latent_dim = latent_distribution_parameters.size(1)//2
                mean_y = latent_distribution_parameters[:, :latent_dim]
                log_var_y = latent_distribution_parameters[:, latent_dim:]
                y = mean_y + torch.sqrt(log_var_y.exp()) * torch.randn_like(mean_y)

                #y = torch.sqrt(torch.tensor(latent_dim)) * (y / torch.linalg.norm(y, dim=1, keepdim=True)) #this must be removed.
            else:
                mean_y, log_var_y = self.encoder(x)
                y = mean_y + torch.sqrt(log_var_y.exp()) * torch.randn_like(mean_y)
        else:
            y = self.encoder(x)
        

        

        sampling_shape = [x.size(0)]+self.config.data.shape
        conditional_sampling_fn = get_conditional_sampling_fn(config=self.config, sde=self.sde, 
                                                              shape=sampling_shape, eps=self.sampling_eps, 
                                                              predictor=predictor, corrector=corrector, 
                                                              p_steps=p_steps, c_steps=c_steps, snr=snr, 
                                                              denoise=denoise, use_path=False, 
                                                              use_pretrained=use_pretrained, encoder_only=encoder_only, 
                                                              t_dependent=t_dependent, gamma=gamma)
        if self.config.training.encoder_only:
            model = {'unconditional_score_model':self.unconditional_score_model,
                     'encoder': self.encoder}
        else:
            model = {'unconditional_score_model':self.unconditional_score_model,
                     'latent_correction_model': self.latent_correction_model}

        return conditional_sampling_fn(model, y, show_evolution)
    
    def configure_optimizers(self):
        class scheduler_lambda_function:
            def __init__(self, warm_up):
                self.use_warm_up = True if warm_up > 0 else False
                self.warm_up = warm_up

            def __call__(self, s):
                if self.use_warm_up:
                    if s < self.warm_up:
                        return s / self.warm_up
                    else:
                        return 1
                else:
                    return 1
        
        if self.config.training.use_pretrained:
            if self.config.training.encoder_only:
                ae_params = self.encoder.parameters()
            else:
                ae_params = list(self.encoder.parameters())+list(self.latent_correction_model.parameters())
            
            ae_optimizer = optim.Adam(ae_params, lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                            weight_decay=self.config.optim.weight_decay)
            ae_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(ae_optimizer, scheduler_lambda_function(self.config.optim.warmup)),
                            'interval': 'step'}  # called after each training step
            return [ae_optimizer], [ae_scheduler]
        else:
            ae_params = list(self.encoder.parameters())+list(self.latent_correction_model.parameters())
            ae_optimizer = optim.Adam(ae_params, lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                            weight_decay=self.config.optim.weight_decay)
            ae_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(ae_optimizer, scheduler_lambda_function(self.config.optim.slowing_factor*self.config.optim.warmup)),
                            'interval': 'step'}  # called after each training step
                            
            unconditional_score_optimizer = optim.Adam(self.unconditional_score_model.parameters(), 
                                            lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999), eps=self.config.optim.eps,
                                            weight_decay=self.config.optim.weight_decay)
            unconditional_score_scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(unconditional_score_optimizer, scheduler_lambda_function(self.config.optim.warmup)),
                                            'interval': 'step'}  # called after each training step
            
            return [ae_optimizer, unconditional_score_optimizer], [ae_scheduler, unconditional_score_scheduler]