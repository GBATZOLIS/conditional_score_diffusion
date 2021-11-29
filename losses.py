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


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
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

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
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

def get_smld_loss_fn(vesde, train, reduce_mean=False, likelihood_weighting=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = vesde.discrete_sigmas
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    score_fn = mutils.get_score_fn(vesde, model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    score_fn_labels = labels/(vesde.N - 1)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[(..., ) + (None, ) * len(batch.shape[1:])]
    perturbed_data = noise + batch
    score = score_fn(perturbed_data, score_fn_labels)
    target = -noise / (sigmas ** 2)[(..., ) + (None, ) * len(batch.shape[1:])]
    losses = torch.square(score - target)

    if likelihood_weighting:
      losses = losses*sigmas[(..., ) + (None, ) * len(batch.shape[1:])]**2
      losses = losses.reshape(losses.shape[0], -1)
      losses = reduce_op(losses, dim=-1)
    else:
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2

    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_inverse_problem_smld_loss_fn(sde, train, reduce_mean=False, likelihood_weighting=True):
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  # Previous SMLD models assume descending sigmas
  smld_sigma_array_y = sde['y'].discrete_sigmas #observed
  smld_sigma_array_x = sde['x'].discrete_sigmas #unobserved
  

  def loss_fn(model, batch):
    y, x = batch
    score_fn = mutils.get_score_fn(sde, model, train=train)
    labels = torch.randint(0, sde['x'].N, (x.shape[0],), device=x.device)
    score_fn_labels = labels/(sde['x'].N - 1)

    sigmas_y = smld_sigma_array_y.type_as(y)[labels]
    sigmas_x = smld_sigma_array_x.type_as(x)[labels]
    
    noise_y = torch.randn_like(y) * sigmas_y[(..., ) + (None, ) * len(y.shape[1:])]
    perturbed_data_y = noise_y + y
    noise_x = torch.randn_like(x) * sigmas_x[(..., ) + (None, ) * len(x.shape[1:])]
    perturbed_data_x = noise_x + x

    perturbed_data = {'x':perturbed_data_x, 'y':perturbed_data_y}
    score = score_fn(perturbed_data, score_fn_labels)

    target_x = -noise_x / (sigmas_x ** 2)[(..., ) + (None, ) * len(x.shape[1:])]
    target_y = -noise_y / (sigmas_y ** 2)[(..., ) + (None, ) * len(y.shape[1:])]
    
    losses_x = torch.square(score['x']-target_x)
    losses_y = torch.square(score['y']-target_y)

    if likelihood_weighting:
      losses_x = losses_x*sigmas_x[(..., ) + (None, ) * len(x.shape[1:])]**2
      losses_x = losses_x.reshape(losses_x.shape[0], -1)
      losses_y = losses_y*sigmas_y[(..., ) + (None, ) * len(y.shape[1:])]**2
      losses_y = losses_y.reshape(losses_y.shape[0], -1)
      losses = torch.cat((losses_x, losses_y), dim=-1)
      losses = reduce_op(losses, dim=-1)
    else:
      losses_x = losses_x.reshape(losses_x.shape[0], -1)
      losses_y = losses_y.reshape(losses_y.shape[0], -1)
      losses = torch.cat((losses_x, losses_y), dim=-1)
      smld_weighting = (sigmas_x**2*sigmas_y**2)/(sigmas_x**2+sigmas_y**2) #smld weighting
      losses = reduce_op(losses, dim=-1) * smld_weighting
    
    loss = torch.mean(losses)
    
    return loss
  
  return loss_fn


#I need to revisit this function when I implement conditional SDE with VPSDEs and discrete training procedure.
def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_inverse_problem_ddpm_loss_fn(vpsdes, train, reduce_mean=True):
  return NotImplementedError

def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    #assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, list):
      #this part of the code needs to be improved. We should check all elements of the sde list. We might have a mixture.
      if isinstance(sde['y'], VESDE) and isinstance(sde['x'], cVESDE) and len(sde.keys())==2:
        loss_fn = get_inverse_problem_smld_loss_fn(sde, train, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)
      elif isinstance(sde['y'], VPSDE) and isinstance(sde['x'], VPSDE) and len(sde.keys())==2:
        loss_fn = get_inverse_problem_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
      else:
        raise NotImplementedError('This combination of sdes is not supported for discrete training yet.')
    
    elif isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn