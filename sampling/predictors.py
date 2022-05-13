import abc
import torch
import sde_lib
import numpy as np

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

def get_predictor(name):
  return _PREDICTORS[name]

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.
    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[(...,) + (None,) * len(x.shape[1:])] * np.sqrt(-dt) * z
    return x, x_mean

@register_predictor(name='heun') #Heun's method (pc-Adams-11-pece)
class PC_Adams_11_Predictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=True):
    super().__init__(sde, score_fn, probability_flow)
    #we implement the PECE method here. This should give us quadratic order of accuracy.

  def f(self, x, t):
    if isinstance(self.sde, dict):
      score = self.score_fn(x, t)
      f = {}
      for name in self.sde.keys():
        f_drift, f_diffusion = self.sde[name].sde(x[name], t)
        r_drift = f_drift - f_diffusion[(..., ) + (None, ) * len(x[name].shape[1:])] ** 2 * score[name] * 0.5
        f[name] = r_drift
      return f
    else:
      drift, diffusion = self.sde.sde(x, t)
      score = self.score_fn(x, t)
      drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * 0.5
      return drift

  def predict(self, x, f_0, h):
    if isinstance(self.sde, dict):
      prediction = {}
      for name in self.sde.keys():
        prediction[name] = x[name] + f_0[name] * h
    else:
      prediction = x + f_0 * h

    return prediction
  
  def correct(self, x, f_1, f_0, h):
    if isinstance(self.sde, dict):
      correction = {}
      for name in x.keys():
        correction[name] = x[name] + h/2 * (f_1[name] + f_0[name])
    else:
      correction = x + h/2 * (f_1 + f_0)
    return correction

  def update_fn(self, x, t):
      h = -1. / self.rsde.N #torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)
      
      #evaluate
      f_0 = self.f(x, t)
      #predict
      x_1 = self.predict(x, f_0, h)
      #evaluate
      f_1 = self.f(x_1, t+h)
      #correct once
      x_2 = self.correct(x, f_1, f_0, h)

      return x_2, x_2

@register_predictor(name='conditional_euler_maruyama')
class conditionalEulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, y, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, y, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[(...,) + (None,) * len(x.shape[1:])] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[(...,) + (None,) * len(x.shape[1:])] * z
    return x, x_mean
  

@register_predictor(name='conditional_reverse_diffusion')
class conditionalReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, y, t):
    f, G = self.rsde.discretize(x, y, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[(...,) + (None,) * len(x.shape[1:])] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[(...,) + (None,) * len(x.shape[1:])]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[(...,) + (None,) * len(x.shape[1:])] * score) / torch.sqrt(1. - beta)[(...,) + (None,) * len(x.shape[1:])]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)

@register_predictor(name='conditional_ancestral_sampling')
class conditionalAncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE) \
      and not isinstance(sde, sde_lib.cVESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, y, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, y, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[(...,) + (None,) * len(x.shape[1:])]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, y, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, y, t)
    x_mean = (x + beta[(...,) + (None,) * len(x.shape[1:])] * score) / torch.sqrt(1. - beta)[(...,) + (None,) * len(x.shape[1:])]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x

@register_predictor(name='conditional_none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, y, t):
    return x, x
