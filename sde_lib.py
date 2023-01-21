"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.
    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  def perturb(self, x_0, t):
    """Retruns x from pt(x|x_0)"""
    z = torch.randn_like(x_0)
    mean, std = self.marginal_prob(x_0, t)
    perturbed_data = mean + std[(...,) + (None,) * len(x_0.shape[1:])] * z
    return perturbed_data

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.
    Useful for computing the log-likelihood via probability flow ODE.
    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.
    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)
    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

class cSDE(SDE): #conditional setting. Allow for conditional time-dependent score.
  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, y, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score_x = score_fn(x, y, t) #conditional score on y
        drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_x * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, y, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_fn(x, y, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()



class SNRSDE(SDE):
  def __init__(self, N, gamma=None, a=2, b=3, c=6, minus_log_SNR_0 = -10, minus_log_SNR_1 = 5):
    super().__init__(N)
    if gamma is None:
      gamma = lambda t: a * t + b * t**c
      d_gamma = lambda t: a + b*c * t**(c-1)
    else:
      #Use atuograd
      pass
    # Gamma has to be normalized to have correct start and end points (cf. Appendix D of VDM paper)
    normalizing_consant = (minus_log_SNR_1 - minus_log_SNR_0)/(gamma(1)-gamma(0))
    log_SNR = lambda t: - (minus_log_SNR_0 +  normalizing_consant * (gamma(t) - gamma(0)))
    self.d_log_SNR = lambda t: -normalizing_consant * d_gamma(t)
    self.log_SNR = log_SNR
  
  @property
  def T(self):
    return 1

  def sde(self, x, t):
    SNR = lambda t: torch.exp(self.log_SNR(t))
    d_log_SNR = self.d_log_SNR
    std = torch.sqrt(1 / (1 + SNR(t)))
    drift = 0.5 * std[(...,)+(None,)*len(x.shape[1:])]**2 * d_log_SNR(t)[(...,)+(None,)*len(x.shape[1:])] * x
    diffusion_squared = - std**2 * d_log_SNR(t)
    diffusion = torch.sqrt(diffusion_squared)
    return drift, diffusion

  def marginal_prob(self, x, t): 
    SNR = lambda t: torch.exp(self.log_SNR(t))
    alpha = torch.sqrt(SNR(t) / (1 + SNR(t)))[(...,)+(None,)*len(x.shape[1:])]
    mean = alpha * x
    std = torch.sqrt(1 / (1 + SNR(t)))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    dims_to_reduce=tuple(range(len(z.shape))[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=dims_to_reduce) / 2.
    return logps

class VVSDE(SDE):
  """Construct a Variance Vanishing SDE.
  Args:
    N: number of discretization steps
  """
  def __init__(self, N):
      super().__init__(N)
  
  @property
  def T(self):
    return 1

  def sde(self, x, t):
    return super().sde(x, t)

  def marginal_prob(self, x, t):
    return super().marginal_prob(x, t)

  def prior_sampling(self, shape):
    return super().prior_sampling(shape)

  def prior_logp(self, z):
    return super().prior_logp(z)

class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[(...,)+(None,)*len(x.shape[1:])] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t): #perturbation kernel
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[(...,)+(None,)*len(x.shape[1:])]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[(...,)+(None,)*len(x.shape[1:])] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[(...,)+(None,)*len(x.shape[1:])] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[(...,)+(None,)*len(x.shape[1:])] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])

    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, data_mean=None):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

    self.diffused_mean = data_mean #new

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))).type_as(t))
    return drift, diffusion

  def marginal_prob(self, x, t): #perturbation kernel P(X(t)|X(0)) parameters
    sigma_min = torch.tensor(self.sigma_min).type_as(t)
    sigma_max = torch.tensor(self.sigma_max).type_as(t)
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std
  
  def compute_backward_kernel(self, x0, x_tplustau, t, tau):
    #x_forward = x(t+\tau)
    #compute the parameters of p(x(t)|x(0), x(t+\tau)) - the reverse kernel of width tau at time step t.
    sigma_min, sigma_max = torch.tensor(self.sigma_min).type_as(t), torch.tensor(self.sigma_max).type_as(t)

    sigma_t_square = (sigma_min * (sigma_max / sigma_min) ** t)**2
    sigma_tplustau_square = (sigma_min * (sigma_max / sigma_min) ** (t+tau))**2

    std_backward = torch.sqrt(sigma_t_square * (sigma_tplustau_square - sigma_t_square) / sigma_tplustau_square)

    #backward scaling factor for the mean
    s_b_0 = (sigma_tplustau_square - sigma_t_square) / sigma_tplustau_square
    s_b_tplustau = sigma_t_square / sigma_tplustau_square

    mean_backward = x0 * s_b_0[(...,) + (None,) * len(x0.shape[1:])] + x_tplustau * s_b_tplustau[(...,) + (None,) * len(x0.shape[1:])]

    return mean_backward, std_backward

  def prior_sampling(self, shape):
    if self.diffused_mean is not None:
      repeat_tuple = tuple([shape[0]]+[1 for _ in shape[1:]])
      diffused_mean = self.diffused_mean.unsqueeze(0).repeat(repeat_tuple)
      return torch.randn(*shape) * self.sigma_max + diffused_mean
    else:
      return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    dims_to_reduce=tuple(range(len(z.shape))[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=dims_to_reduce) / (2 * self.sigma_max ** 2)

  # def discretize(self, x, t):
  #   """SMLD(NCSN) discretization."""
  #   timestep = (t * (self.N - 1) / self.T).long()
  #   sigma = self.discrete_sigmas.to(t.device)[timestep]
  #   adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
  #                                self.discrete_sigmas[timestep - 1].to(t.device))
  #   f = torch.zeros_like(x)
  #   G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
  #   return f, G

class cVESDE(cSDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, data_mean=None):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N
    self.diffused_mean = data_mean #new

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t): #perturbation kernel P(X(t)|X(0)) parameters 
    sigma_min = torch.tensor(self.sigma_min).type_as(t)
    sigma_max = torch.tensor(self.sigma_max).type_as(t)
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    if self.diffused_mean is not None:
      repeat_tuple = tuple([shape[0]]+[1 for _ in shape[1:]])
      diffused_mean = self.diffused_mean.unsqueeze(0).repeat(repeat_tuple)
      return torch.randn(*shape) * self.sigma_max + diffused_mean
    else:
      return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G
  