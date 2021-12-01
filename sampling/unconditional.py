from sampling.predictors import get_predictor, ReverseDiffusionPredictor, NonePredictor
from sampling.correctors import get_corrector, NoneCorrector
from tqdm import tqdm
import functools
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

def get_sampling_fn(config, sde, shape, eps):
  """Create a sampling function.
  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn

def get_inpainting_fn(config, sde, eps, n_steps_each=1):
  '''create the inpainting function'''
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())
  sampling_fn = get_pc_inpainter(sde=sde, 
                                 predictor=predictor, 
                                 corrector=corrector, 
                                 snr=config.sampling.snr,
                                 n_steps=n_steps_each, 
                                 probability_flow=config.sampling.probability_flow, 
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal, 
                                 eps=eps)
  return sampling_fn

def get_ode_sampler(sde, shape,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3):
  """Probability flow ODE sampler with the black-box ODE solver.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, conditional=False, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, conditional=False, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.
    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(model.device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(model.device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(model.device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      return x, nfe

  return ode_sampler

def get_pc_sampler(sde, shape, predictor, corrector, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3):
  """Create a Predictor-Corrector (PC) sampler.
  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer. -> not used anymore
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model, show_evolution=False):
    """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    if show_evolution:
      evolution = []

    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(model.device).type(torch.float32)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=model.device)

      for i in tqdm(range(sde.N)):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
        
        if show_evolution:
          evolution.append(x.cpu())

      samples = x_mean if denoise else x

      if show_evolution:
        sampling_info = {'evolution': torch.stack(evolution), 
                        'times':timesteps, 'steps':sde.N * (n_steps + 1)}
        return samples, sampling_info
      else:
        sampling_info = {'times':timesteps, 'steps':sde.N * (n_steps + 1)}
        return samples, sampling_info

  return pc_sampler

def get_pc_inpainter(sde, predictor, corrector, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create an image inpainting function that uses PC samplers.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.
  Returns:
    An inpainting function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_inpaint_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def inpaint_update_fn(model, data, mask, x, t):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0]).type_as(data) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        
        masked_data_mean, std = sde.marginal_prob(data, vec_t)
        masked_data = masked_data_mean + torch.randn_like(x) * std[(...,) + (None,) * len(x.shape[1:])]
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask

        return x, x_mean

    return inpaint_update_fn

  projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(model, data, mask, show_evolution=False):
    """Predictor-Corrector (PC) sampler for image inpainting.
    Args:
      model: A score model.
      data: A PyTorch tensor that represents a mini-batch of images to inpaint.
      mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
        and value `0` marks pixels that require inpainting.
    Returns:
      Inpainted (complete) images.
    """

    '''
    def corrections_steps(i):
      changing_points = [500, 750, 875, 937, 969, 985, 992]
      if i < changing_points[0]:
        return 1
      elif i >= changing_points[-1]:
        return 2**len(changing_points)
      else:
        for j in range(len(changing_points)-1):
          if i >= changing_points[j] and i<changing_points[j+1]:
            break
        exp = j+1
        return 2**exp
    '''

    def corrections_steps(i):
      return 1


    with torch.no_grad():
      # Initial sample

      #GB suggestion:
      #vec_t = torch.ones(data.shape[0], device=data.device) * sde.T
      #masked_data_mean, std = sde.marginal_prob(data, vec_t)
      #masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
      #x = masked_data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)
      #
      #Song code:
      #x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask) 

      x = data * mask + sde.prior_sampling(data.shape).type_as(data) * (1. - mask) 
      
      if show_evolution:
        evolution = [x.cpu()]

      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in tqdm(range(sde.N)):
        t = timesteps[i]

        for _ in range(corrections_steps(i)):
          x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)

        x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t)

        if show_evolution:
          evolution.append(x.cpu())
      
      if show_evolution:
        sampling_info = {'evolution': torch.stack(evolution)}
        return x_mean if denoise else x, sampling_info
      else:
        return x_mean if denoise else x, {}

  return pc_inpainter

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, conditional=False, train=False, continuous=continuous)

  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper that configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, conditional=False, train=False, continuous=continuous)

  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)