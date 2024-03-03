from sampling.predictors import get_predictor, NonePredictor
from sampling.correctors import get_corrector, NoneCorrector
import functools
import torch
from tqdm import tqdm
from models import utils as mutils

def setup_score_fn(sde, model, continuous, use_pretrained, encoder_only, t_dependent, latent_correction, gamma):
    if not use_pretrained:
      score_fn = mutils.get_score_fn(sde, model, conditional=True, train=False, continuous=continuous)
      score_fn = mutils.get_conditional_score_fn(score_fn, target_domain='x')
      device = model.device
    else:
      if not encoder_only:
        unconditional_score_model = model['unconditional_score_model']
        latent_correction_model = model['latent_correction_model']
        unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=continuous)
        conditional_correction_fn = mutils.get_score_fn(sde, latent_correction_model, conditional=True, train=False, continuous=continuous)
          
        score_fn = mutils.get_conditional_score_fn_with_prior_diffusion_model(unconditional_score_fn, conditional_correction_fn)
        device = unconditional_score_model.device
      else:
        if t_dependent:
          if not latent_correction: #BASIC CONFIG
            def get_latent_correction_fn(encoder):
              def get_log_density_fn(encoder):
                def log_density_fn(x, z, t):
                    latent_distribution_parameters = encoder(x, t)
                    channels = latent_distribution_parameters.size(1) // 2
                    mean_z = latent_distribution_parameters[:, :channels]
                    log_var_z = latent_distribution_parameters[:, channels:]

                    # Flatten mean_z and log_var_z for consistent shape handling
                    mean_z_flat = mean_z.view(mean_z.size(0), -1)
                    log_var_z_flat = log_var_z.view(log_var_z.size(0), -1)
                    z_flat = z.view(z.size(0), -1)

                    logdensity = -0.5 * torch.sum(torch.square(z_flat - mean_z_flat) / log_var_z_flat.exp(), dim=1)
                    return logdensity
                  
                return log_density_fn

              def latent_correction_fn(x, z, t):
                  torch.set_grad_enabled(True)
                  log_density_fn = get_log_density_fn(encoder)
                  device = x.device
                  x.requires_grad=True
                  ftx = log_density_fn(x, z, t)
                  grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                        grad_outputs=torch.ones(ftx.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                  assert grad_log_density.size() == x.size()
                  torch.set_grad_enabled(False)
                  return grad_log_density

              return latent_correction_fn
              

            unconditional_score_model = model['unconditional_score_model']
            unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=continuous)

            encoder = model['encoder']
            conditional_correction_fn = get_latent_correction_fn(encoder)

            def get_conditional_score_fn_with_prior(unconditional_score_fn, conditional_correction_fn):
              def conditional_score_fn(x, y, t):
                conditional_score = gamma*conditional_correction_fn(x, y, t) + unconditional_score_fn(x, t)
                return conditional_score
              return conditional_score_fn

            score_fn = get_conditional_score_fn_with_prior(unconditional_score_fn, conditional_correction_fn)
            device = unconditional_score_model.device
            
          else:
            def get_encoder_latent_correction_fn(encoder):
              def get_log_density_fn(encoder):
                def log_density_fn(x, z, t):
                    latent_distribution_parameters = encoder(x, t)
                    channels = latent_distribution_parameters.size(1) // 2
                    mean_z = latent_distribution_parameters[:, :channels]
                    log_var_z = latent_distribution_parameters[:, channels:]

                    # Flatten mean_z and log_var_z for consistent shape handling
                    mean_z_flat = mean_z.view(mean_z.size(0), -1)
                    log_var_z_flat = log_var_z.view(log_var_z.size(0), -1)
                    z_flat = z.view(z.size(0), -1)

                    logdensity = -0.5 * torch.sum(torch.square(z_flat - mean_z_flat) / log_var_z_flat.exp(), dim=1)
                    return logdensity
                  
                return log_density_fn

              def latent_correction_fn(x, z, t):
                  torch.set_grad_enabled(True)
                  log_density_fn = get_log_density_fn(encoder)
                  device = x.device
                  x.requires_grad=True
                  ftx = log_density_fn(x, z, t)
                  grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                        grad_outputs=torch.ones(ftx.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                  assert grad_log_density.size() == x.size()
                  torch.set_grad_enabled(False)
                  return grad_log_density

              return latent_correction_fn
              
            unconditional_score_model = model['unconditional_score_model']
            unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=continuous)
              
            latent_correction_model = model['latent_correction_model']
            latent_correction_fn = mutils.get_score_fn(sde, latent_correction_model, conditional=False, train=False, continuous=continuous)

            encoder = model['encoder']
            encoder_correction_fn = get_encoder_latent_correction_fn(encoder)

            def get_conditional_score_fn_with_prior(unconditional_score_fn, encoder_correction_fn, latent_correction_fn):
              def conditional_score_fn(x, y, t):
                conditional_score = latent_correction_fn({'x':x, 'y':y}, t) + encoder_correction_fn(x, y, t) + unconditional_score_fn(x, t)
                return conditional_score
              return conditional_score_fn

            score_fn = get_conditional_score_fn_with_prior(unconditional_score_fn, encoder_correction_fn, latent_correction_fn)
            device = unconditional_score_model.device

        else:
          #t-independent encoder implementation
          def get_denoiser_fn(unconditional_score_model):
            score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=True)
            def denoiser_fn(x, t):
              score = score_fn(x, t)
              a_t, sigma_t = sde.kernel_coefficients(x, t)
              denoised = ((sigma_t**2)[(...,)+(None,)*len(x.shape[1:])]*score + x) / a_t[(...,)+(None,)*len(x.shape[1:])]
              return denoised
            return denoiser_fn
            
          def get_latent_correction_fn(encoder, unconditional_score_model):
            denoiser_fn = get_denoiser_fn(unconditional_score_model)

            def get_log_density_fn(encoder, denoiser_fn):
              def log_density_fn(x, z, t):
                denoised_x = denoiser_fn(x, t)
                latent_distribution_parameters = encoder(denoised_x)
                latent_dim = latent_distribution_parameters.size(1)//2
                mean_z = latent_distribution_parameters[:, :latent_dim]
                log_var_z = latent_distribution_parameters[:, latent_dim:]
                logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
                return logdensity
                
              return log_density_fn

            def latent_correction_fn(x, z, t):
                torch.set_grad_enabled(True)
                log_density_fn = get_log_density_fn(encoder, denoiser_fn)
                device = x.device
                x.requires_grad=True
                ftx = log_density_fn(x, z, t)
                grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                      grad_outputs=torch.ones(ftx.size()).to(device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                assert grad_log_density.size() == x.size()
                torch.set_grad_enabled(False)
                return grad_log_density

            return latent_correction_fn
            
          unconditional_score_model = model['unconditional_score_model']
          unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=continuous)

          encoder = model['encoder']
          conditional_correction_fn = get_latent_correction_fn(encoder, unconditional_score_model)
          def get_conditional_score_fn_with_prior(unconditional_score_fn, conditional_correction_fn):
            def conditional_score_fn(x, y, t):
              conditional_score = conditional_correction_fn(x, y, t) + unconditional_score_fn(x, t)
              return conditional_score
            return conditional_score_fn

          score_fn = get_conditional_score_fn_with_prior(unconditional_score_fn, conditional_correction_fn)
          device = unconditional_score_model.device

    return score_fn, device


def get_conditional_sampling_fn(config, sde, shape, eps, 
                          predictor='default', corrector='default', p_steps='default', 
                          c_steps='default', snr='default', denoise='default', use_path='default', use_pretrained=False, 
                          encoder_only=False, t_dependent=True, latent_correction=False, gamma=1,
                          direction='backward', x_boundary=None):

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
      use_path = getattr(config.sampling, 'use_path', False)
    
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
                                            eps=eps, 
                                            use_pretrained=use_pretrained,
                                            encoder_only=encoder_only,
                                            t_dependent=t_dependent,
                                            latent_correction=latent_correction,
                                            gamma=gamma,
                                            direction=direction,
                                            x_boundary=x_boundary)
    return sampling_fn

def get_pc_conditional_sampler(sde, shape, predictor, corrector, snr, p_steps,
                   c_steps=1, probability_flow=False, continuous=False, 
                   denoise=True, use_path=False, eps=1e-5, use_pretrained=False, encoder_only=False, t_dependent=True, 
                   latent_correction=False, gamma=1., direction='backward', x_boundary=None):

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
  

  def get_conditional_update_fn(update_fn, sampler_type):
    """Modify the update function of predictor & corrector to incorporate data information."""
    def conditional_update_fn(x, y, t, score_fn):
      with torch.no_grad():
        vec_t = torch.ones(x.shape[0]).to(t.device) * t
        x, x_mean = update_fn(x=x, y=y, t=vec_t, score_fn=score_fn)
      return x, x_mean, y, y

    return conditional_update_fn

  
  def pc_conditional_sampler(model, y, show_evolution=False, score_fn=None):
    """ The PC conditional sampler function.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    
    #Set the score function if it is not provided
    if score_fn is None:
      score_fn, device = setup_score_fn(sde, model, continuous, use_pretrained, encoder_only, t_dependent, latent_correction, gamma)
    else:
      device = model.device #set the device

    c_sde = sde['x'] if isinstance(sde, dict) else sde
 
    if direction == 'backward':
      timesteps = torch.linspace(c_sde.T, eps, p_steps+1, device=device)
    elif direction == 'forward':
      timesteps = torch.linspace(eps, c_sde.T, p_steps+1, device=device)

    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(conditional_shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            discretisation=timesteps)
                                            
    predictor_conditional_update_fn = get_conditional_update_fn(predictor_update_fn, sampler_type='predictor')

    with torch.no_grad():
      # Initial sample
      if x_boundary is None:
        x = c_sde.prior_sampling(shape).to(device)
      else:
        x = x_boundary

      if show_evolution:
        evolution = {'x':[], 'y':[]}

      for i in tqdm(range(p_steps)):
        t = timesteps[i]
        x, x_mean, y_perturbed, y_mean = predictor_conditional_update_fn(x, y, t, score_fn)
          
        if show_evolution:
          evolution['x'].append(x.cpu())
          evolution['y'].append(y_perturbed.cpu())

      if show_evolution:
        evolution['x'], evolution['y'] = torch.stack(evolution['x']), torch.stack(evolution['y'])
        sampling_info = {'evolution': evolution}
        return x_mean if denoise else x
      else:
        return x_mean if denoise else x
          
  return pc_conditional_sampler

def conditional_shared_predictor_update_fn(x, y, t, sde, score_fn, predictor, probability_flow, continuous, discretisation):
  """A wrapper that configures and returns the update function of predictors."""

  c_sde = sde['x'] if isinstance(sde, dict) else sde
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(c_sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(c_sde, score_fn, probability_flow, discretisation)

  return predictor_obj.update_fn(x, y, t)

def conditional_shared_corrector_update_fn(x, y, t, sde, score_fn, corrector, continuous, snr, n_steps):
  """A wrapper that configures and returns the update function of correctors."""

  c_sde = sde['x'] if isinstance(sde, dict) else sde
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(c_sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(c_sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, y, t)