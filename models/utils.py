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

"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def divide_by_sigmas(h, labels, sde, continuous=False):
  #inputs:
  #h:the output of model_fn
  #labels -> point to the right sigmas
  #sde: the sdes used for diffusing the input tensor(s)
  #outputs: the scaled activation

  #calculate sigmas using sde and divide by sigmas
  if not continuous:
    if isinstance(sde, dict) and isinstance(h, dict):
      for domain in h.keys():
        domain_sigmas = sde[domain].discrete_sigmas.type_as(h[domain])
        h[domain] = h[domain] / domain_sigmas[(labels,) + (None,) * len(h[domain].shape[1:])] 
    else:
      sigmas = sde.discrete_sigmas.type_as(h)
      h = h / sigmas[(labels,) + (None,) * len(h.shape[1:])]
  elif continuous:
    if isinstance(sde, dict) and isinstance(h, dict):
      for domain in h.keys():
        domain_std = sde[domain].marginal_prob(torch.zeros_like(h[domain]), labels)[1]
        h[domain] = h[domain] / domain_std[(...,) + (None,) * len(h[domain].shape[1:])] 
    else:
      std = sde.marginal_prob(torch.zeros_like(h), labels)[1]
      h = h / std[(...,) + (None,) * len(h.shape[1:])]
  return h

def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  #score_model = score_model.to(config.device)
  #score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn



def get_score_fn(sde, model, conditional=False, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  if conditional:
    """COVERS OUR CONDITIONAL SCORE ESTIMATOR"""
    if isinstance(sde, dict):
      if isinstance(sde['y'], sde_lib.VPSDE) or isinstance(sde['y'], sde_lib.subVPSDE):
        raise NotImplementedError('This combination of sdes is not supported for conditional SDEs yet.')
      elif isinstance(sde['y'], sde_lib.VESDE) and isinstance(sde['x'], sde_lib.cVESDE) and len(sde)==2:
        def score_fn(x, t):
          if continuous:
            labels = t * (sde['x'].N - 1)
            score = model_fn(x, labels)
            score = divide_by_sigmas(score, t, sde, continuous)
          else:
            # For VE-trained models, t=0 corresponds to the highest noise level
            labels = t*(sde['x'].N - 1)
            labels = torch.round(labels.float()).long()
            score = model_fn(x, labels)
            score = divide_by_sigmas(score, labels, sde, continuous)

          return score
      else:
        raise NotImplementedError('This combination of SDEs is not supported for conditional SDEs yet.')
    else:
      """COVERS THE SR3 CONDITIONAL SCORE ESTIMATOR"""
      if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
          # Scale neural network output by standard deviation and flip sign
          if continuous or isinstance(sde, sde_lib.subVPSDE):
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            labels = t * (sde.N - 1)
            score = model_fn(x, labels)
            std = sde.marginal_prob(torch.zeros_like(score), t)[1]
          else:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            labels = t * (sde.N - 1)
            score = model_fn(x, labels)
            std = sde.sqrt_1m_alphas_cumprod.type_as(labels)[labels.long()]

          score = score / std[(...,)+(None,)*len(score.shape[1:])] #-> why do they scale the output of the network by std ??
          return score

      elif isinstance(sde, sde_lib.VESDE) or isinstance(sde, sde_lib.cVESDE):
        def score_fn(x, t):
          if continuous:
            labels = t * (sde.N - 1)
            score = model_fn(x, labels)
            score = divide_by_sigmas(score, t, sde, continuous)
          else:
            labels = t*(sde.N - 1)
            labels = torch.round(labels.float()).long()
            score = model_fn(x, labels)
            score = divide_by_sigmas(score, labels, sde, continuous)
          return score
      else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  else:
    """COVERS THE BASIC UNCONDITIONAL CASE"""
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      def score_fn(x, t):
        # Scale neural network output by standard deviation and flip sign
        if continuous or isinstance(sde, sde_lib.subVPSDE):
          # For VP-trained models, t=0 corresponds to the lowest noise level
          # The maximum value of time embedding is assumed to 999 for
          # continuously-trained models.
          labels = t * (sde.N - 1)
          score = model_fn(x, labels)
          std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VP-trained models, t=0 corresponds to the lowest noise level
          labels = t * (sde.N - 1)
          score = model_fn(x, labels)
          std = sde.sqrt_1m_alphas_cumprod.type_as(labels)[labels.long()]

        score = score / std[(...,)+(None,)*len(x.shape[1:])] #-> why do they scale the output of the network by std ??
        return score

    elif isinstance(sde, sde_lib.VESDE) or isinstance(sde, sde_lib.cVESDE):
      def score_fn(x, t):
        if continuous:
          #raise NotImplementedError('Continuous training for VE SDE is not checked. Division by std should be included. Not completed yet.')
          std = labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
          time_embedding = torch.log(labels) if model.embedding_type == 'fourier' else labels
          score = model_fn(x, time_embedding)
          score = score / std[(...,)+(None,)*len(x.shape[1:])]
        else:
          # For VE-trained models, t=0 corresponds to the lowest noise level
          labels = t*(sde.N - 1)
          labels = torch.round(labels).long()
          std = sigma_labels = sde.discrete_sigmas.type_as(x)[labels]
          score = model_fn(x, sigma_labels)
          score = score / std[(...,)+(None,)*len(x.shape[1:])]

        return score

    else:
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

#needs shape generalisation
def get_conditional_score_fn(score_fn, target_domain): #for standard inverse problems. It should be modified for general inverse problems (different resolutions etc.).
  def conditional_score_fn(x, y, t):
    score = score_fn({'x':x, 'y':y}, t)
    if isinstance(score, dict):
      return score[target_domain]
    else:
      return score

  return conditional_score_fn

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))