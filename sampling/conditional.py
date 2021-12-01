from sampling.predictors import get_predictor, NonePredictor
from sampling.correctors import get_corrector, NoneCorrector
import functools
import torch
from tqdm import tqdm
from models import utils as mutils

def get_conditional_sampling_fn(config, sde, shape, eps, 
                          predictor='default', corrector='default', p_steps='default', 
                          c_steps='default', snr='default', denoise='default', use_path='default'):

    if predictor == 'default':
      predictor = get_predictor(config.sampling.predictor.lower())
    else:
      predictor = get_predictor(predictor.lower())

    if corrector == 'default':
      corrector = get_corrector(config.sampling.corrector.lower())
    else:
      corrector = get_corrector(corrector.lower())

    if p_steps == 'default':
      p_steps = config.model.num_scales
    if c_steps == 'default':
      c_steps = config.sampling.n_steps_each
    if snr == 'default':
      snr = config.sampling.snr
    if denoise == 'default':
      denoise = config.sampling.noise_removal
    if use_path =='default':
      use_path = config.sampling.use_path
    
    sampling_fn = get_pc_conditional_sampler(sde=sde, 
                                            shape = shape,
                                            predictor=predictor, 
                                            corrector=corrector, 
                                            snr=snr,
                                            p_steps=p_steps,
                                            c_steps=c_steps, 
                                            probability_flow=config.sampling.probability_flow, 
                                            continuous=config.training.continuous,
                                            denoise = denoise,
                                            use_path = use_path,
                                            eps=eps)
    return sampling_fn

def get_pc_conditional_sampler(sde, shape, predictor, corrector, snr, p_steps,
                   c_steps=1, probability_flow=False, continuous=False, 
                   denoise=True, use_path=False, eps=1e-5):

  """Create a Predictor-Corrector (PC) sampler.
  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    c_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.
  Returns:
    A conditional sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(conditional_shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)

  corrector_update_fn = functools.partial(conditional_shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=c_steps)

  def get_conditional_update_fn(update_fn, sampler_type):
    """Modify the update function of predictor & corrector to incorporate data information."""

    if isinstance(sde, dict) and len(sde.keys())==2:
      if use_path:
        if sampler_type == 'predictor':
          def conditional_update_fn(x, y, t, model, y_tplustau, tau):
            with torch.no_grad():
              vec_t = torch.ones(x.shape[0]).to(model.device) * t
              vec_tau = torch.ones(x.shape[0]).to(model.device) * tau
              y_t_mean, y_t_std = sde['y'].compute_backward_kernel(y, y_tplustau, vec_t, vec_tau)
              y_t_perturbed = y_t_mean + torch.randn_like(y) * y_t_std[(...,) + (None,) * len(y.shape[1:])]
              x, x_mean = update_fn(x=x, y=y_t_perturbed, t=vec_t, model=model)
              return x, x_mean, y_t_perturbed
        elif sampler_type == 'corrector':
          #y_t is the sample from P(y_t|y_{t+tau}, y0). We do not resample that. 
          #The mean and std that created y_t should be saved before. No need to return them with this function.
          def conditional_update_fn(x, y_t, t, model):
            vec_t = torch.ones(x.shape[0]).to(model.device) * t
            return update_fn(x=x, y=y_t, t=vec_t, model=model)
        else:
          return NotImplementedError('Sampler type %s is not supported. Available sampler type options: [predictor, corrector]' % sampler_type)
      else:
        def conditional_update_fn(x, y, t, model):
          with torch.no_grad():
            vec_t = torch.ones(x.shape[0]).to(model.device) * t
            y_mean, y_std = sde['y'].marginal_prob(y, vec_t)
            y_perturbed = y_mean + torch.randn_like(y) * y_std[(...,) + (None,) * len(y.shape[1:])]
            x, x_mean = update_fn(x=x, y=y_perturbed, t=vec_t, model=model)
            return x, x_mean, y_perturbed, y_mean
    else:
      def conditional_update_fn(x, y, t, model):
        with torch.no_grad():
          vec_t = torch.ones(x.shape[0]).to(model.device) * t
          x, x_mean = update_fn(x=x, y=y, t=vec_t, model=model)
        return x, x_mean, y, y

    return conditional_update_fn

  predictor_conditional_update_fn = get_conditional_update_fn(predictor_update_fn, sampler_type='predictor')
  corrector_conditional_update_fn = get_conditional_update_fn(corrector_update_fn, sampler_type='corrector')

  if use_path:
    def pc_conditional_sampler(model, y, show_evolution=False):
      """ The PC conditional sampler function.
      Args:
        model: A score model.
      Returns:
        Samples, number of function evaluations.
      """

      #this function can be used for the exploration of variable correction schemes
      #i.e. the num of correction steps is a function of the diffusion time. 
      #This can possibly increase sampling speed or sampling quality. To be explored.
      def corrections_steps(i):
          return 1

      c_sde = sde['x'] if isinstance(sde, dict) else sde

      with torch.no_grad():
        # Initial sample
        x = c_sde.prior_sampling(shape).to(model.device)

        timesteps = torch.linspace(c_sde.T, eps, p_steps, device=model.device)
        tau = timesteps[0]-timesteps[1]
        T = timesteps[0]
        y_tplustau_mean, y_tplustau_std  = sde['y'].marginal_prob(y, torch.ones(x.shape[0]).to(model.device) * (T+tau))
        y_tplustau = y_tplustau_mean + torch.randn_like(y)*y_tplustau_std[(...,) + (None,) * len(y.shape[1:])]

        if show_evolution:
          evolution = {'x':[], 'y':[]}

        for i in tqdm(range(p_steps)):
          t = timesteps[i]
          
          x, x_mean, y_tplustau = predictor_conditional_update_fn(x, y, t, model, y_tplustau, tau)

          for _ in range(corrections_steps(i)):
            x, x_mean = corrector_conditional_update_fn(x, y_tplustau, t, model)
          
          if show_evolution:
            evolution['x'].append(x.cpu())
            evolution['y'].append(y_tplustau.cpu())

        print('torch.mean(torch.abs(y-y_tplustau)): %.8f' % torch.mean(torch.abs(y-y_tplustau)))

        if show_evolution:
          #check the effect of denoising
          #evolution['x'].append(x_mean.cpu())
          #evolution['y'].append(y_mean.cpu())

          evolution['x'], evolution['y'] = torch.stack(evolution['x']), torch.stack(evolution['y'])
          sampling_info = {'evolution': evolution}
          return x_mean if denoise else x, sampling_info
        else:
          return x_mean if denoise else x, {}
          
    return pc_conditional_sampler
  else:
    def pc_conditional_sampler(model, y, show_evolution=False):
      """ The PC conditional sampler function.
      Args:
        model: A score model.
      Returns:
        Samples, number of function evaluations.
      """

      #this function can be used for the exploration of variable correction schemes
      #i.e. the num of correction steps is a function of the diffusion time. 
      #This can possibly increase sampling speed or sampling quality. To be explored.
      def corrections_steps(i):
          return 1

      c_sde = sde['x'] if isinstance(sde, dict) else sde

      with torch.no_grad():
        # Initial sample
        x = c_sde.prior_sampling(shape).to(model.device)
        if show_evolution:
          evolution = {'x':[], 'y':[]}

        timesteps = torch.linspace(c_sde.T, eps, p_steps, device=model.device)

        for i in tqdm(range(p_steps)):
          t = timesteps[i]
          #vec_t = torch.ones(shape[0], device=model.device) * t
          x, x_mean, y_perturbed, y_mean = predictor_conditional_update_fn(x, y, t, model)

          for _ in range(corrections_steps(i)):
            x, x_mean, y_perturbed, y_mean = corrector_conditional_update_fn(x, y, t, model)
          
          if show_evolution:
            evolution['x'].append(x.cpu())
            evolution['y'].append(y_perturbed.cpu())

        if show_evolution:
          #check the effect of denoising
          #evolution['x'].append(x_mean.cpu())
          #evolution['y'].append(y_mean.cpu())

          evolution['x'], evolution['y'] = torch.stack(evolution['x']), torch.stack(evolution['y'])
          sampling_info = {'evolution': evolution}
          return x_mean if denoise else x, sampling_info
        else:
          return x_mean if denoise else x, {}
          
    return pc_conditional_sampler

def conditional_shared_predictor_update_fn(x, y, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, conditional=True, train=False, continuous=continuous)
  score_fn = mutils.get_conditional_score_fn(score_fn, target_domain='x')

  c_sde = sde['x'] if isinstance(sde, dict) else sde
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(c_sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(c_sde, score_fn, probability_flow)

  return predictor_obj.update_fn(x, y, t)

def conditional_shared_corrector_update_fn(x, y, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper that configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, conditional=True, train=False, continuous=continuous)
  score_fn = mutils.get_conditional_score_fn(score_fn, target_domain='x')

  c_sde = sde['x'] if isinstance(sde, dict) else sde
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(c_sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(c_sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, y, t)