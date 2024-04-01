import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE, cVESDE
import math
from torch.autograd.functional import vjp
import torch.nn as nn
import torch.nn.functional as F

   
def get_MI_loss_fn(sde):
  #Refer to Theorem 4 in 'Maximum likelihood training of score-based models' by Song et al.
  def get_unscaled_entropy(score_fn, batch, t_dist):
      x, y = batch
      t = t_dist.sample((x.shape[0],)).type_as(x)
      n = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
      
      cond = y
      score = score_fn(perturbed_x, cond, t)
      losses = torch.square(score)

      #we must use likelihood weighting for entropy estimation  
      _, g = sde.sde(torch.zeros_like(x), t, True)
      w2 = g ** 2
            
      importance_weight = torch.exp(-1*t_dist.log_prob(t).type_as(t))
      losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
      losses *= 1/2
      loss = torch.mean(losses)
      return -1*loss
      
  def mutual_information_fn(score_fn, batch, t_dist):
      #x will be the latent encoding for disentangled representation learning and y the label
      x, y = batch 

      #conditional entropy (up to a constant C)
      conditional_entropy = get_unscaled_entropy(score_fn, batch, t_dist)

      #unconditional entropy (up to the same constant C)
      y_unlabeled = torch.ones_like(y) * -1
      batch = [x, y_unlabeled]
      unconditional_entropy = get_unscaled_entropy(score_fn, batch, t_dist)

      mutual_information = unconditional_entropy - conditional_entropy
      return mutual_information
  
  return mutual_information_fn

def modify_labels_for_classifier_free_guidance(y):
    """
    Randomly choose half of the indices in `y` and set their value to -1.
    Args:
    - y (Tensor): The labels tensor.

    Returns:
    - Modified labels tensor with half of the values set to -1.
    """
    indices = torch.randperm(len(y))[:len(y) // 2]  # Randomly select half of the indices
    y[indices] = -1  # Set selected labels to -1
    return y

def get_DSM_loss_fn(sde, likelihood_weighting):
  def loss_fn(score_fn, batch, t_dist):
      x, y = batch
      t = t_dist.sample((x.shape[0],)).type_as(x)
      n = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
      
      cond = y
      score = score_fn(perturbed_x, cond, t)
      grad_log_pert_kernel = -1 * n / std[(...,) + (None,) * len(x.shape[1:])]
      losses = torch.square(score - grad_log_pert_kernel)
            
      if likelihood_weighting:
        _, g = sde.sde(torch.zeros_like(x), t, True)
        w2 = g ** 2
      else:
        w2 = std ** 2
            
      importance_weight = torch.exp(-1*t_dist.log_prob(t).type_as(t))
      losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
      losses *= 1/2
      loss = torch.mean(losses)
      return loss
  return loss_fn
   
def get_classifier_free_loss_fn(sde, likelihood_weighting=True):
  def loss_fn(score_fn, batch, t_dist):
      x, y = batch
      t = t_dist.sample((x.shape[0],)).type_as(x)
      n = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
      
      cond = modify_labels_for_classifier_free_guidance(y)
      score = score_fn(perturbed_x, cond, t)
      grad_log_pert_kernel = -1 * n / std[(...,) + (None,) * len(x.shape[1:])]
      losses = torch.square(score - grad_log_pert_kernel)
            
      if likelihood_weighting:
        _, g = sde.sde(torch.zeros_like(x), t, True)
        w2 = g ** 2
      else:
        w2 = std ** 2
            
      importance_weight = torch.exp(-1*t_dist.log_prob(t).type_as(t))
      losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
      losses *= 1/2
      loss = torch.mean(losses)
      return loss
   
  return loss_fn

def scoreVAE_loss_fn(score_fn, x, y, mean_z, log_var_z, latent, t_dist, likelihood_weighting, kl_weight, sde):
  # Compute KL loss using directly flattened tensors
  kl_loss = -0.5 * torch.sum(
                  1 + log_var_z.view(log_var_z.size(0), -1) - mean_z.view(mean_z.size(0), -1).pow(2) - log_var_z.view(log_var_z.size(0), -1).exp(),dim=1).mean()
      
  t = t_dist.sample((x.shape[0],)).type_as(x)
  n = torch.randn_like(x)
  mean, std = sde.marginal_prob(x, t)
  perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
      
  cond = [y, latent]
  score = score_fn(perturbed_x, cond, t)
  grad_log_pert_kernel = -1 * n / std[(...,) + (None,) * len(x.shape[1:])]
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

def get_disentangled_scoreVAE_loss_fn(sde, likelihood_weighting=True, kl_weight=1, disentanglement_factor=1):
    #idea: disentangled representations
    #motivation: train the encoder in such a way that it optimises the ELBO, but also make sure that 
    #the latent encoding has as low mutual information with the interpretable attributes y as possible.
    #goal: y and latent are independent random vectors that fully describe the original image x.
    def loss_fn(score_fn, MI_diffusion_model_score_fn, encoder, batch, t_dist):
      x, y = batch

      #get the encoded latent and its associated parameters (mean_z, log_var_z)
      t0 = torch.zeros(x.shape[0]).type_as(x)
      latent_distribution_parameters = encoder(x, t0)
      channels = latent_distribution_parameters.size(1)//2
      mean_z = latent_distribution_parameters[:, :channels]
      log_var_z = latent_distribution_parameters[:, channels:]
      latent = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)

      scoreVAE_loss = scoreVAE_loss_fn(score_fn, x, y, mean_z, log_var_z, latent, t_dist, likelihood_weighting, kl_weight, sde)
      
      MI_loss_fn = get_MI_loss_fn(sde)
      MI_batch = [latent, y]
      mutual_information = MI_loss_fn(MI_diffusion_model_score_fn, MI_batch, t_dist)

      loss = scoreVAE_loss + disentanglement_factor * mutual_information
      return loss
    
    return loss_fn
