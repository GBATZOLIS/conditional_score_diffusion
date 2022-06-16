import abc
import torch
import sde_lib

_CORRECTORS = {}

def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_corrector(name):
  return _CORRECTORS[name]

class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.
    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = 1 #torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[(...,) + (None,) * len(x.shape[1:])] * grad
      x = x_mean + torch.sqrt(step_size * 2)[(...,) + (None,) * len(x.shape[1:])] * noise

    return x, x_mean


@register_corrector(name='mala')
class MetropolisAdjustedLangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, energy_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)

    self.energy_fn = energy_fn
    
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
  
  def proposal_density(self, x, x_0, step_size, t):
      grad = self.score_fn(x_0, t)
      c = -0.25 * (1/step_size) 
      norm = torch.linalg.norm(x - x_0 - step_size[(...,) + (None,) * len(x.shape[1:])] * grad, dim=1)**2
      return torch.exp(c * norm)

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    energy_fn = self.energy_fn
    proposal_density = self.proposal_density
    n_steps = self.n_steps
    target_snr = self.snr


    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      # calculate a proposal update according to langevin dynamics
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = 1 #torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = 1 #torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[(...,) + (None,) * len(x.shape[1:])] * grad
      x_new = x_mean + torch.sqrt(step_size * 2)[(...,) + (None,) * len(x.shape[1:])] * noise

      # calculate the acceptance threshold
      numerator = energy_fn(x_new, t).squeeze() * proposal_density(x, x_new, step_size, t) 
      denumerator = energy_fn(x, t).squeeze() * proposal_density(x_new, x, step_size, t)
      quotient = numerator / denumerator
      rejection_threshold = torch.min(torch.ones_like(quotient), quotient) 
      
      # accept or reject the update
      u = torch.rand_like(rejection_threshold)
      condition = (u < rejection_threshold).int()[(...,) + (None,) * len(x.shape[1:])]
      ones_minus_condition = torch.ones_like(condition) - condition
      x = ones_minus_condition * x + condition * x_new
      x_mean = ones_minus_condition * x + condition * x_mean

    #print(condition.sum().item()/10)
    return x, x_mean


    

@register_corrector(name='conditional_langevin')
class conditionalLangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) and not isinstance(sde, sde_lib.cVESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, y, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, y, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[(...,) + (None,) * len(x.shape[1:])] * grad
      x = x_mean + torch.sqrt(step_size * 2)[(...,) + (None,) * len(x.shape[1:])] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.
  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[(...,) + (None,) * len(x.shape[1:])] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[(...,) + (None,) * len(x.shape[1:])]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x

#use this none corrector for conditional settings
@register_corrector(name='conditional_none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, y, t):
    return x, x