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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE, cVESDE
import math
from torch.autograd.functional import vjp

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn

def get_scoreVAE_loss_fn(sde, train, variational=False, likelihood_weighting=True, eps=1e-5, t_batch_size=1, kl_weight=1, 
                          use_pretrained=False, encoder_only=True, t_dependent=True, latent_correction=False):
  if not variational:
    def loss_fn(encoder, score_model, batch):
      x = batch
      y = encoder(x)

      score_fn = mutils.get_score_fn(sde, score_model, conditional=True, train=train, continuous=True)
      
      t_losses = torch.zeros(size=(x.size(0),)).type_as(x)
      for _ in range(t_batch_size):
        t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
        perturbed_data = {'x':perturbed_x, 'y':y}

        score = score_fn(perturbed_data, t)

        f, g = sde.sde(torch.zeros_like(x), t, True)
        g2 = g ** 2
        grad_log_pert_kernel = -1 * z / std[(...,) + (None,) * len(x.shape[1:])]
        losses = torch.square(score - grad_log_pert_kernel)-torch.square(grad_log_pert_kernel)
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        losses -= 2*torch.sum(f.reshape(f.shape[0], -1), dim=-1)
        losses *= 1/2

        t_losses+=losses
      
      loss = torch.mean(t_losses)/t_batch_size
      return loss
    
  else:
    if not use_pretrained:
      def loss_fn(encoder, score_model, batch):
        x = batch
        score_fn = mutils.get_score_fn(sde, score_model, conditional=True, train=train, continuous=True)
        
        #reparametrisation trick
        mean_z, log_var_z = encoder(x)
        y = mean_z + torch.sqrt(log_var_z.exp()) * torch.randn_like(mean_z)

        if kl_weight > 1e-9:
          kl_loss = -0.5 * torch.sum(1 + log_var_z - mean_z ** 2 - log_var_z.exp(), dim=1).mean()

        #t_losses = torch.zeros(size=(x.size(0),)).type_as(x)
        #for _ in range(t_batch_size):
        t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
        perturbed_data = {'x':perturbed_x, 'y':y}

        score = score_fn(perturbed_data, t)

        f, g = sde.sde(torch.zeros_like(x), t, True)
        g2 = g ** 2
        grad_log_pert_kernel = -1 * z / std[(...,) + (None,) * len(x.shape[1:])]
        losses = torch.square(score - grad_log_pert_kernel) #-torch.square(grad_log_pert_kernel)
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        #losses -= 2*torch.sum(f.reshape(f.shape[0], -1), dim=-1)
        losses *= 1/2
        rec_loss = torch.mean(losses)

        #t_losses+=losses
        #rec_loss = torch.mean(t_losses)/t_batch_size
        if kl_weight > 1e-9:
          loss = rec_loss + kl_weight * kl_loss
        else:
          loss = rec_loss

        return loss
      
    else:
      if latent_correction:
        assert encoder_only and t_dependent
        def loss_fn(encoder, latent_correction_model, unconditional_score_model, batch):
            def get_encoder_latent_correction_fn(encoder, z):
              def get_log_density_fn(encoder):
                def log_density_fn(z, x, t):
                  latent_distribution_parameters = encoder(x, t)
                  latent_dim = latent_distribution_parameters.size(1)//2
                  mean_z = latent_distribution_parameters[:, :latent_dim]
                  log_var_z = latent_distribution_parameters[:, latent_dim:]
                  logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
                  return logdensity
                
                return log_density_fn

              def latent_correction_fn(x, t):
                  if not train: 
                    torch.set_grad_enabled(True)

                  log_density_fn = get_log_density_fn(encoder)
                  device = x.device
                  x.requires_grad=True
                  ftx = log_density_fn(z, x, t)
                  grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                      grad_outputs=torch.ones(ftx.size()).to(device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                  assert grad_log_density.size() == x.size()

                  if not train:
                    torch.set_grad_enabled(False)

                  return grad_log_density

              return latent_correction_fn

            x = batch
            unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=train, continuous=True)
            latent_correction_fn = mutils.get_score_fn(sde, latent_correction_model, conditional=True, train=train, continuous=True)

            t0 = torch.zeros(x.shape[0]).type_as(x)
            latent_distribution_parameters = encoder(x, t0)
            latent_dim = latent_distribution_parameters.size(1)//2
            mean_z = latent_distribution_parameters[:, :latent_dim]
            log_var_z = latent_distribution_parameters[:, latent_dim:]

            kl_loss = -0.5 * torch.sum(1 + log_var_z - mean_z ** 2 - log_var_z.exp(), dim=1).mean()
            latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)

            encoder_correction_fn = get_encoder_latent_correction_fn(encoder, latent)

            t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
            z = torch.randn_like(x)
            mean, std = sde.marginal_prob(x, t)
            perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
            
            unconditional_score = unconditional_score_fn(perturbed_x, t)
            encoder_correction = encoder_correction_fn(perturbed_x, t)
            latent_correction = latent_correction_fn({'x':perturbed_x, 'y':latent}, t)

            score = latent_correction + encoder_correction + unconditional_score
            
            grad_log_pert_kernel = -1 * z / std[(...,) + (None,) * len(x.shape[1:])]
            losses = torch.square(score - grad_log_pert_kernel)
            
            if likelihood_weighting:
              _, g = sde.sde(torch.zeros_like(x), t, True)
              w2 = g ** 2
            else:
              w2 = std ** 2
            
            losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2
            losses *= 1/2
            rec_loss = torch.mean(losses)

            loss = rec_loss + kl_weight * kl_loss

            return loss
      else:
        assert encoder_only and t_dependent
        def loss_fn(encoder, unconditional_score_model, batch, t_dist):
            def get_latent_correction_fn(encoder, z):
              def get_log_density_fn(encoder):
                def log_density_fn(z, x, t):
                  latent_distribution_parameters = encoder(x, t)
                  latent_dim = latent_distribution_parameters.size(1)//2
                  mean_z = latent_distribution_parameters[:, :latent_dim]
                  log_var_z = latent_distribution_parameters[:, latent_dim:]
                  logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
                  return logdensity
                
                return log_density_fn

              def latent_correction_fn(x, t):
                  if not train: 
                    torch.set_grad_enabled(True)

                  log_density_fn = get_log_density_fn(encoder)
                  device = x.device
                  x.requires_grad=True
                  ftx = log_density_fn(z, x, t)
                  grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                      grad_outputs=torch.ones(ftx.size()).to(device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                  assert grad_log_density.size() == x.size()

                  if not train:
                    torch.set_grad_enabled(False)

                  return grad_log_density

              return latent_correction_fn

            x = batch
            unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=train, continuous=True)

            t0 = torch.zeros(x.shape[0]).type_as(x)
            latent_distribution_parameters = encoder(x, t0)
            latent_dim = latent_distribution_parameters.size(1)//2
            mean_z = latent_distribution_parameters[:, :latent_dim]
            log_var_z = latent_distribution_parameters[:, latent_dim:]

            kl_loss = -0.5 * torch.sum(1 + log_var_z - mean_z ** 2 - log_var_z.exp(), dim=1).mean()
            latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)

            conditional_correction_fn = get_latent_correction_fn(encoder, latent)

            t = t_dist.sample((x.shape[0],)).type_as(x)
            z = torch.randn_like(x)
            mean, std = sde.marginal_prob(x, t)
            perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
            
            unconditional_score = unconditional_score_fn(perturbed_x, t)
            conditional_correction = conditional_correction_fn(perturbed_x, t)

            score = conditional_correction + unconditional_score
            
            grad_log_pert_kernel = -1 * z / std[(...,) + (None,) * len(x.shape[1:])]
            losses = torch.square(score - grad_log_pert_kernel)
            
            if likelihood_weighting:
              _, g = sde.sde(torch.zeros_like(x), t, True)
              w2 = g ** 2
            else:
              w2 = std ** 2
            
            importance_weight = torch.exp(-1*t_dist.log_prob(t).type_as(t))
            losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
            losses *= 1/2
            rec_loss = torch.mean(losses)

            loss = rec_loss + kl_weight * kl_loss

            return loss



  
  return loss_fn

def get_old_scoreVAE_loss_fn(sde, train, variational=False, likelihood_weighting=True, eps=1e-5, use_pretrained=False, encoder_only=False, t_dependent=True):
  if not use_pretrained:
    def loss_fn(encoder, score_model, batch):
      score_fn = mutils.get_score_fn(sde, score_model, conditional=True, train=train, continuous=True)
      
      x = batch
      y = encoder(x)
      t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
      z = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
      perturbed_data = {'x':perturbed_x, 'y':y}

      score = score_fn(perturbed_data, t)

      if not likelihood_weighting:
        losses = torch.square(score * std[(...,) + (None,) * len(x.shape[1:])] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
      else:
        g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
        losses = torch.square(score + z / std[(...,) + (None,) * len(x.shape[1:])])
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1) * g2

      loss = torch.mean(losses)
      return loss
  else:
    if encoder_only:
      if t_dependent:
        def loss_fn(encoder, unconditional_score_model, batch):
          def get_latent_correction_fn(encoder, z):
            def get_log_density_fn(encoder):
              def log_density_fn(z, x, t):
                latent_distribution_parameters = encoder(x, t)
                latent_dim = latent_distribution_parameters.size(1)//2
                mean_z = latent_distribution_parameters[:, :latent_dim]
                log_var_z = latent_distribution_parameters[:, latent_dim:]
                logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
                return logdensity
              
              return log_density_fn

            def latent_correction_fn(x, t):
                if not train: 
                  torch.set_grad_enabled(True)

                log_density_fn = get_log_density_fn(encoder)
                device = x.device
                x.requires_grad=True
                #z.requires_grad=True
                #t.requires_grad=True
                ftx = log_density_fn(z, x, t)
                grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                    grad_outputs=torch.ones(ftx.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                assert grad_log_density.size() == x.size()

                if not train:
                  torch.set_grad_enabled(False)

                return grad_log_density

            return latent_correction_fn

          x = batch
          unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=train, continuous=True)

          t0 = torch.zeros(x.shape[0]).type_as(x)
          latent_distribution_parameters = encoder(x, t0)
          latent_dim = latent_distribution_parameters.size(1)//2
          mean_z = latent_distribution_parameters[:, :latent_dim]
          log_var_z = latent_distribution_parameters[:, latent_dim:]
          latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)

          conditional_correction_fn = get_latent_correction_fn(encoder, latent)

          t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
          z = torch.randn_like(x)
          mean, std = sde.marginal_prob(x, t)
          perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
          
          unconditional_score = unconditional_score_fn(perturbed_x, t)
          conditional_correction = conditional_correction_fn(perturbed_x, t)

          score = conditional_correction + unconditional_score

          if not likelihood_weighting:
            losses = torch.square(score * std[(...,) + (None,) * len(x.shape[1:])] + z)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
          else:
            g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
            losses = torch.square(score + z / std[(...,) + (None,) * len(x.shape[1:])])
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1) * g2

          loss = torch.mean(losses)
          return loss
      
      else:
        #implementation for the use of a time-independent encoder
        #this implementation is based on the idea of the denoising - score parametrisation equivalence.
        def loss_fn(encoder, unconditional_score_model, batch):
          def get_denoiser_fn(unconditional_score_model):
              score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=train, continuous=True)
              def denoiser_fn(x, t):
                score = score_fn(x, t)
                a_t, sigma_t = sde.kernel_coefficients(x, t)
                denoised = ((sigma_t**2)[(...,)+(None,)*len(x.shape[1:])]*score + x) / a_t[(...,)+(None,)*len(x.shape[1:])]
                return denoised
              return denoiser_fn

          def get_latent_correction_fn(encoder, unconditional_score_model, z):
            denoiser_fn = get_denoiser_fn(unconditional_score_model)

            def get_log_density_fn(encoder, denoiser_fn):
              def log_density_fn(z, x, t):
                denoised_x = denoiser_fn(x, t)
                latent_distribution_parameters = encoder(denoised_x)
                latent_dim = latent_distribution_parameters.size(1)//2
                mean_z = latent_distribution_parameters[:, :latent_dim]
                log_var_z = latent_distribution_parameters[:, latent_dim:]
                logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
                return logdensity
              
              return log_density_fn

            def latent_correction_fn(x, t):
                if not train: 
                  torch.set_grad_enabled(True)

                log_density_fn = get_log_density_fn(encoder, denoiser_fn)
                device = x.device
                x.requires_grad=True
                #z.requires_grad=True
                #t.requires_grad=True
                ftx = log_density_fn(z, x, t)
                grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                    grad_outputs=torch.ones(ftx.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                assert grad_log_density.size() == x.size()

                if not train:
                  torch.set_grad_enabled(False)

                return grad_log_density

            return latent_correction_fn

          x = batch
          unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=train, continuous=True)

          latent_distribution_parameters = encoder(x)
          latent_dim = latent_distribution_parameters.size(1)//2
          mean_z = latent_distribution_parameters[:, :latent_dim]
          log_var_z = latent_distribution_parameters[:, latent_dim:]
          latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)

          conditional_correction_fn = get_latent_correction_fn(encoder, unconditional_score_model, latent)

          t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
          z = torch.randn_like(x)
          mean, std = sde.marginal_prob(x, t)
          perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
          
          unconditional_score = unconditional_score_fn(perturbed_x, t)
          conditional_correction = conditional_correction_fn(perturbed_x, t)

          score = conditional_correction + unconditional_score

          if not likelihood_weighting:
            losses = torch.square(score * std[(...,) + (None,) * len(x.shape[1:])] + z)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
          else:
            g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
            losses = torch.square(score + z / std[(...,) + (None,) * len(x.shape[1:])])
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1) * g2

          loss = torch.mean(losses)
          return loss



    else:
      def loss_fn(encoder, latent_correction_model, unconditional_score_model, batch):
        unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=train, continuous=True)
        conditional_correction_fn = mutils.get_score_fn(sde, latent_correction_model, conditional=True, train=train, continuous=True)
        
        x = batch
        y = encoder(x)
        t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
        perturbed_data = {'x':perturbed_x, 'y':y}

        unconditional_score = unconditional_score_fn(perturbed_x, t)
        conditional_correction = conditional_correction_fn(perturbed_data, t)
        
        score = conditional_correction + unconditional_score

        if not likelihood_weighting:
          losses = torch.square(score * std[(...,) + (None,) * len(x.shape[1:])] + z)
          losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
          g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
          losses = torch.square(score + z / std[(...,) + (None,) * len(x.shape[1:])])
          losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss
  
  return loss_fn

def get_general_sde_loss_fn(sde, train, conditional=False, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  
  if conditional:
    if isinstance(sde, dict):
      if len(sde.keys()) == 2:
        assert likelihood_weighting, 'For the variance reduction technique in inverse problems, we only support likelihood weighting for the time being.'
        
        def loss_fn(model, batch):
          y, x = batch
          score_fn = mutils.get_score_fn(sde, model, conditional=conditional, train=train, continuous=continuous)
          t = torch.rand(x.shape[0]).type_as(x) * (sde['x'].T - eps) + eps

          z_y = torch.randn_like(y)
          mean_y, std_y = sde['y'].marginal_prob(y, t)
          perturbed_data_y = mean_y + std_y[(...,) + (None,) * len(y.shape[1:])] * z_y

          z_x = torch.randn_like(x)
          mean_x, std_x = sde['x'].marginal_prob(x, t)
          perturbed_data_x = mean_x + std_x[(...,) + (None,) * len(x.shape[1:])] * z_x
          
          perturbed_data = {'x':perturbed_data_x, 'y':perturbed_data_y}
          score = score_fn(perturbed_data, t)
          
          g2_y = sde['y'].sde(torch.zeros_like(y), t)[1] ** 2
          g2_x = sde['x'].sde(torch.zeros_like(x), t)[1] ** 2
          
          losses_y = torch.square(score['y'] + z_y / std_y[(...,) + (None,) * len(y.shape[1:])])*g2_y[(...,) + (None,) * len(y.shape[1:])]
          losses_y = losses_y.reshape(losses_y.shape[0], -1)
          losses_x = torch.square(score['x'] + z_x / std_x[(...,) + (None,) * len(x.shape[1:])])*g2_x[(...,) + (None,) * len(x.shape[1:])]
          losses_x = losses_x.reshape(losses_x.shape[0], -1)
          losses = torch.cat((losses_x, losses_y), dim=-1)
          losses = reduce_op(losses, dim=-1)
          loss = torch.mean(losses)
          return loss

      elif len(sde.keys()) >= 3:
        assert likelihood_weighting, 'For multi-speed diffussion, we support only likelihood weighting.'
        def loss_fn(model, batch):
          #batch is a dictionary of tensors
          score_fn = mutils.get_score_fn(sde, model, conditional=conditional, train=train, continuous=continuous)
          
          key = list(batch.keys())[0]
          t = torch.rand(batch[key].shape[0]).type_as(batch[key]) * (sde[key].T - eps) + eps

          perturbed_data_dict = {}
          noise_dict = {}
          std_dict = {}
          for diff_quantity in batch.keys():
            z = torch.randn_like(batch[diff_quantity])
            noise_dict[diff_quantity] = z

            mean, std = sde[diff_quantity].marginal_prob(batch[diff_quantity], t)
            std_dict[diff_quantity] = std

            perturbed_data = mean + std[(...,) + (None,) * len(batch[diff_quantity].shape[1:])] * z
            perturbed_data_dict[diff_quantity] = perturbed_data
          
          score = score_fn(perturbed_data, t) #score is a dictionary

          losses = []
          for diff_quantity in batch.keys():
            g2 = sde[diff_quantity].sde(torch.zeros_like(batch[diff_quantity]), t)[1] ** 2
            diff_quantity_losses = torch.square(score[diff_quantity] + noise_dict[diff_quantity] / std_dict[diff_quantity][(...,) + (None,) * len(batch[diff_quantity].shape[1:])])*g2[(...,) + (None,) * len(batch[diff_quantity].shape[1:])]
            diff_quantity_losses = diff_quantity_losses.reshape(diff_quantity_losses.shape[0], -1)
            losses.append(diff_quantity_losses)
          
          losses = torch.cat(losses, dim=-1)
          losses = reduce_op(losses, dim=-1)
          loss = torch.mean(losses)
          return loss
    else:
      #SR3 estimator
      def loss_fn(model, batch):
        y, x = batch
        score_fn = mutils.get_score_fn(sde, model, conditional=conditional, train=train, continuous=continuous)
        t = torch.rand(x.shape[0]).type_as(x) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
        perturbed_data = {'x':perturbed_x, 'y':y}

        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
          losses = torch.square(score * std[(...,) + (None,) * len(x.shape[1:])] + z)
          losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
          g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
          losses = torch.square(score + z / std[(...,) + (None,) * len(x.shape[1:])])
          losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss
  
  # UNCONDITIONAL
  else:
    def loss_fn(model, batch):
      """Compute the loss function.
      Args:
        model: A score model.
        batch: A mini-batch of training data.
      Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
      """
      score_fn = mutils.get_score_fn(sde, model, conditional=conditional, train=train, continuous=continuous)
      t = torch.rand(batch.shape[0]).type_as(batch) * (sde.T - eps) + eps
      z = torch.randn_like(batch)
      mean, std = sde.marginal_prob(batch, t)
      perturbed_data = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
      score = score_fn(perturbed_data, t)

      if not likelihood_weighting:
        losses = torch.square(score * std[(...,) + (None,) * len(batch.shape[1:])] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
      else:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[(...,) + (None,) * len(batch.shape[1:])])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

      loss = torch.mean(losses)
      return loss

  return loss_fn