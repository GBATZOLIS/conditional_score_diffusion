import torch 
from pathlib import Path
import os
from lightning_modules.utils import create_lightning_module
from lightning_data_modules.utils import create_lightning_datamodule
from models import utils as mutils
import math
from tqdm import tqdm
import pickle
import numpy as np
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision
import lpips

def get_conditional_manifold_dimension(config, name=None):
  #---- create the setup ---
  log_path = config.logging.log_path
  log_name = config.logging.log_name
  save_path = os.path.join(log_path, log_name, 'svd')
  Path(save_path).mkdir(parents=True, exist_ok=True)

  DataModule = create_lightning_datamodule(config)
  DataModule.setup()
  train_dataloader = DataModule.train_dataloader()
    
  pl_module = create_lightning_module(config)
  pl_module = pl_module.load_from_checkpoint(config.model.checkpoint_path)
  pl_module.configure_sde(config)

  device = config.device
  pl_module = pl_module.to(device)
  pl_module.eval()
  
  score_model = pl_module.score_model
  sde = pl_module.sde
  score_fn = mutils.get_score_fn(sde, score_model, conditional=False, train=False, continuous=True)
  #---- end of setup ----

  num_datapoints = config.get('dim_estimation.num_datapoints', 2500)
  singular_values = []
  labels = []
  idx = 0
  with tqdm(total=num_datapoints) as pbar:
    for orig_batch, orig_labels in train_dataloader:
      #orig_batch = orig_batch.to(device)
      batchsize = orig_batch.size(0)
      
      if idx+1 >= num_datapoints:
          break
        
      for x, y in zip(orig_batch, orig_labels):
        if idx+1 >= num_datapoints:
          break
        
        x = x.to(device)
        ambient_dim = math.prod(x.shape[1:])
        x = x.repeat([batchsize,]+[1 for i in range(len(x.shape))])

        num_batches = ambient_dim // batchsize + 1
        extra_in_last_batch = ambient_dim - (ambient_dim // batchsize) * batchsize
        num_batches *= 4

        t = pl_module.sampling_eps
        vec_t = torch.ones(x.size(0), device=device) * t

        scores = []
        for i in range(1, num_batches+1):
          batch = x.clone()

          mean, std = sde.marginal_prob(batch, vec_t)
          z = torch.randn_like(batch)
          batch = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
          score = score_fn(batch, vec_t).detach().cpu()

          if i < num_batches:
            scores.append(score)
          else:
            scores.append(score[:extra_in_last_batch])
        
        scores = torch.cat(scores, dim=0)
        scores = torch.flatten(scores, start_dim=1)

        means = scores.mean(dim=0, keepdim=True)
        normalized_scores = scores - means

        u, s, v = torch.linalg.svd(normalized_scores)
        s = s.tolist()
        singular_values.append(s)
        labels.append(y.item())

        idx+=1
        pbar.update(1)

  with open(os.path.join(save_path, 'svd.pkl'), 'wb') as f:
    info = {'singular_values':singular_values}
    pickle.dump(info, f)
  
  with open(os.path.join(save_path, 'labels.pkl'), 'wb') as f:
    info = {'labels': labels}
    pickle.dump(info, f)

def create_vae_setup(config):
  #---- create the setup ---
  log_path = config.logging.log_path
  log_name = config.logging.log_name
  save_path = os.path.join(log_path, log_name, 'vae_inspection')
  Path(save_path).mkdir(parents=True, exist_ok=True)
  DataModule = create_lightning_datamodule(config)
  DataModule.setup()
  test_dataloader = DataModule.test_dataloader()

  pl_module = create_lightning_module(config)

  if config.training.gpus == 0:
    device = torch.device('cpu')
    checkpoint = torch.load(config.model.checkpoint_path, map_location=device)
    pl_module.load_state_dict(checkpoint['state_dict'])
  else:
    device = torch.device('cuda:0')
    pl_module = pl_module.load_from_checkpoint(config.model.checkpoint_path, config=config)

  pl_module.configure_sde(config)
  sde = pl_module.sde

  pl_module = pl_module.to(device)
  pl_module.eval()

  return pl_module, test_dataloader, sde, device, save_path

def scoreVAE_fidelity(config):
  pl_module, test_dataloader, sde, device, save_path = create_vae_setup(config)
  save_path = os.path.join(save_path, 'fidelity')
  Path(save_path).mkdir(parents=True, exist_ok=True)
  images_save_path = os.path.join(save_path, 'images')
  Path(images_save_path).mkdir(parents=True, exist_ok=True)

  lpips_distance_fn = lpips.LPIPS(net='vgg').to(device)

  gamma_to_rec = {}
  for i, batch in enumerate(test_dataloader):
    if i >= 1:
      break
    
    grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
    torchvision.utils.save_image(grid_batch, os.path.join(images_save_path, 'original.png'))
    
    batch = batch.to(device)


    for gamma in [0.8, 0.9, 1., 1.1, 1.2]:
      if gamma not in gamma_to_rec:
        gamma_to_rec[gamma] = {'LPIPS': 0, 'L2': 0}

      reconstruction = pl_module.encode_n_decode(batch, p_steps=512,
                                                  use_pretrained=config.training.use_pretrained,
                                                  encoder_only=config.training.encoder_only,
                                                  t_dependent=config.training.t_dependent, 
                                                  gamma=gamma)

      grid_reconstruction = torchvision.utils.make_grid(reconstruction, nrow=int(np.sqrt(reconstruction.size(0))), normalize=True, scale_each=True)
      torchvision.utils.save_image(grid_reconstruction, os.path.join(images_save_path,'%.1f.png' % gamma))

      avg_lpips_score = torch.mean(lpips_distance_fn(reconstruction.to(device), batch))
      gamma_to_rec[gamma]['LPIPS'] = avg_lpips_score
      
      difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(batch, start_dim=1)
      L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
      avg_L2norm = torch.mean(L2norm)
      gamma_to_rec[gamma]['L2'] = avg_L2norm
  
      with open(os.path.join(save_path, 'fidelity.pkl'), 'wb') as f:
        pickle.dump(gamma_to_rec, f)

def inspect_corrected_VAE(config):
  pl_module, val_dataloader, sde, device, save_path = create_vae_setup(config)

  def get_encoding_fn(encoder):
    def encoding_fn(x):
      t0 = torch.zeros(x.shape[0]).type_as(x)
      latent_distribution_parameters = encoder(x, t0)
      latent_dim = latent_distribution_parameters.size(1)//2
      mean_z = latent_distribution_parameters[:, :latent_dim]
      log_var_z = latent_distribution_parameters[:, latent_dim:]
      latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)
      return latent
    return encoding_fn

  def get_encoder_latent_correction_fn(encoder):
    def get_log_density_fn(encoder):
      def log_density_fn(x, z, t):
        latent_distribution_parameters = encoder(x, t)
        latent_dim = latent_distribution_parameters.size(1)//2
        mean_z = latent_distribution_parameters[:, :latent_dim]
        log_var_z = latent_distribution_parameters[:, latent_dim:]
        logdensity = -1/2*torch.sum(torch.square(z - mean_z)/log_var_z.exp(), dim=1)
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

  encoder = pl_module.encoder
  unconditional_score_model = pl_module.unconditional_score_model
  latent_correction_model = pl_module.latent_correction_model

  encoding_fn = get_encoding_fn(encoder)
  encoder_latent_correction_fn = get_encoder_latent_correction_fn(encoder)
  unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=True)
  auxiliary_correction_fn = mutils.get_score_fn(sde, latent_correction_model, conditional=True, train=False, continuous=True)
  unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=True)

  iterations = 1
  encoder_correction_percentages = {}
  auxiliary_correction_percentages = {}
  unconditional_score_percentages = {}
  for i, x in enumerate(val_dataloader):
    if i == iterations:
      break
    
    x = x.to(device)
    latent = encoding_fn(x)

    ts = torch.linspace(start=sde.sampling_eps, end=sde.T, steps=200)
    for t_ in tqdm(ts):
      n_t = t_.item() 
      if n_t not in encoder_correction_percentages.keys():
        encoder_correction_percentages[n_t] = []
        auxiliary_correction_percentages[n_t] = []
        unconditional_score_percentages[n_t] = []
        
      t = torch.ones(x.shape[0]).type_as(x)*t_
      z = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z

      encoder_correction = encoder_latent_correction_fn(perturbed_x, latent, t)
      auxiliary_correction = auxiliary_correction_fn({'x':perturbed_x, 'y':latent}, t)
      unconditional_score = unconditional_score_fn(perturbed_x, t)

      total_score = encoder_correction + auxiliary_correction + unconditional_score

      encoder_correction_norm = torch.linalg.norm(encoder_correction.reshape(encoder_correction.shape[0], -1), dim=1)
      auxiliary_correction_norm = torch.linalg.norm(auxiliary_correction.reshape(auxiliary_correction.shape[0], -1), dim=1)
      unconditional_score_norm = torch.linalg.norm(unconditional_score.reshape(unconditional_score.shape[0], -1), dim=1)

      total_score_norm = torch.linalg.norm(total_score.reshape(total_score.shape[0], -1), dim=1)


      encoder_correction_percentages[n_t].extend(torch.flatten(encoder_correction_norm / total_score_norm).numpy().tolist())
      auxiliary_correction_percentages[n_t].extend(torch.flatten(auxiliary_correction_norm / total_score_norm).numpy().tolist())
      unconditional_score_percentages[n_t].extend(torch.flatten(unconditional_score_norm / total_score_norm).numpy().tolist())
  
  with open(os.path.join(save_path, 'contribution_pascal.pkl'), 'wb') as f:
    contribution = {}
    contribution['encoder'] = encoder_correction_percentages
    contribution['auxiliary'] = auxiliary_correction_percentages
    contribution['pretrained'] = unconditional_score_percentages
    pickle.dump(contribution, f)

  '''
  order_times = sorted(list(encoder_correction_percentages.keys()))
  plt.figure()
  plt.title('Norm contribution of different latent score components')
  plt.plot([np.mean(encoder_correction_percentages[t]) for t in order_times], label='encoder')
  plt.plot([np.mean(auxiliary_correction_percentages[t]) for t in order_times], label='auxiliary_correction')
  plt.plot([np.mean(unconditional_score_percentages[t]) for t in order_times], label='unconditional_score')
  plt.legend()
  plt.show()
  '''

def inspect_VAE(config):
  pl_module, val_dataloader, sde, device, save_path = create_vae_setup(config)

  encoder = pl_module.encoder
  unconditional_score_model = pl_module.unconditional_score_model

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
      torch.set_grad_enabled(True)
      log_density_fn = get_log_density_fn(encoder)
      device = x.device
      x.requires_grad=True
      ftx = log_density_fn(z, x, t)
      grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                      grad_outputs=torch.ones(ftx.size()).to(device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
      assert grad_log_density.size() == x.size()
      torch.set_grad_enabled(False)
      return grad_log_density

    return latent_correction_fn

  def get_conditional_score_fn(encoder, unconditional_score_model, latent):
    unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=True)
    conditional_correction_fn = get_latent_correction_fn(encoder, latent)
    def score_fn(x, t):
      return conditional_correction_fn(x, t) + unconditional_score_fn(x, t)
    return score_fn

  encoder = pl_module.encoder
  unconditional_score_model = pl_module.unconditional_score_model

  iterations = 8
  num_interpolations = 1
  latents = []
  
  interpolations_path = os.path.join(save_path, 'interpolation')
  Path(interpolations_path).mkdir(parents=True, exist_ok=True)
  for i, x in tqdm(enumerate(val_dataloader)):
    if i == iterations:
      break

    x = x.to(device)

    ### INTERPOLATION
    interpolations = pl_module.interpolate(x[4:6], num_points=20)
    grid_interpolations = torchvision.utils.make_grid(interpolations, nrow=interpolations.size(0), normalize=True, scale_each=True)
    save_image(grid_interpolations, os.path.join(interpolations_path, '%d.png' % num_interpolations))
    num_interpolations+=1

    ### SAVE LATENT VALUES (FOR KOLMOGOROV-SMIRNOV TEST) 
    '''
    t0 = torch.zeros(x.shape[0]).type_as(x)
    latent_distribution_parameters = encoder(x, t0)
    latent_dim = latent_distribution_parameters.size(1)//2
    mean_z = latent_distribution_parameters[:, :latent_dim]
    log_var_z = latent_distribution_parameters[:, latent_dim:]
    latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)
    latents.append(latent)
    '''

    ### EVALUATE THE CONTRIBUTION RATIO
    '''
    conditional_correction_fn = get_latent_correction_fn(encoder, latent)
    #unconditional_score_fn = mutils.get_score_fn(sde, unconditional_score_model, conditional=False, train=False, continuous=True)  
    total_score_fn = get_conditional_score_fn(encoder, unconditional_score_model, latent)

    
    ts = torch.linspace(start=sde.sampling_eps, end=sde.T, steps=10)
    ratios = []
    corrections = []
    for t_ in tqdm(ts):
      t = torch.ones(x.shape[0]).type_as(x)*t_

      z = torch.randn_like(x)
      mean, std = sde.marginal_prob(x, t)
      perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z

      #unconditional_score = unconditional_score_fn(perturbed_x, t)
      conditional_correction = conditional_correction_fn(perturbed_x, t)
      total_score = total_score_fn(perturbed_x, t)

      conditional_correction_norm = torch.linalg.norm(conditional_correction.reshape(conditional_correction.shape[0], -1), dim=1)
      total_score_norm = torch.linalg.norm(total_score.reshape(total_score.shape[0], -1), dim=1)
      
      ratio = conditional_correction_norm / total_score_norm
      mean_ratio = torch.mean(ratio)
      ratios.append(mean_ratio.item()) 
      corrections.append(torch.mean(conditional_correction_norm).item())
    
    plt.figure()
    plt.plot(ts, corrections)
    plt.show()
    '''


  ### DISPLAY THE KS HISTOGRAM
  '''
  latents = torch.cat(latents, dim=0) #(num_samples, latent_dimension)
  latent_dim = latents.size(1)
  ks_stats = []
  for i in range(latent_dim):
    ks = stats.ks_1samp(latents[:,i].detach().numpy(), stats.norm.cdf)
    ks_stats.append(ks.statistic)
  
  plt.figure()
  plt.hist(ks_stats, bins=50)
  plt.show()
  '''
  



def get_manifold_dimension(config, name=None):
  #---- create the setup ---
  log_path = config.logging.log_path
  log_name = config.logging.log_name
  save_path = os.path.join(log_path, log_name, 'svd')
  Path(save_path).mkdir(parents=True, exist_ok=True)

  DataModule = create_lightning_datamodule(config)
  DataModule.setup()
  train_dataloader = DataModule.train_dataloader()
    
  pl_module = create_lightning_module(config)
  pl_module = pl_module.load_from_checkpoint(config.model.checkpoint_path)
  pl_module.configure_sde(config)

  #get the ema parameters for evaluation
  #pl_module.ema.store(pl_module.parameters())
  #pl_module.ema.copy_to(pl_module.parameters()) 

  device = config.device
  pl_module = pl_module.to(device)
  pl_module.eval()
  
  score_model = pl_module.score_model
  sde = pl_module.sde
  score_fn = mutils.get_score_fn(sde, score_model, conditional=False, train=False, continuous=True)
  #---- end of setup ----

  num_datapoints = config.get('dim_estimation.num_datapoints', 100)
  singular_values = []
  normalized_scores_list = []
  idx = 0
  with tqdm(total=num_datapoints) as pbar:
    for _, orig_batch in enumerate(train_dataloader):

      orig_batch = orig_batch.to(device)
      batchsize = orig_batch.size(0)
      
      if idx+1 >= num_datapoints:
          break
        
      for x in orig_batch:
        if idx+1 >= num_datapoints:
          break
        
        ambient_dim = math.prod(x.shape[1:])
        x = x.repeat([batchsize,]+[1 for i in range(len(x.shape))])

        num_batches = ambient_dim // batchsize + 1
        extra_in_last_batch = ambient_dim - (ambient_dim // batchsize) * batchsize
        num_batches *= 8

        t = pl_module.sampling_eps
        vec_t = torch.ones(x.size(0), device=device) * t

        scores = []
        for i in range(1, num_batches+1):
          batch = x.clone()

          mean, std = sde.marginal_prob(batch, vec_t)
          z = torch.randn_like(batch)
          batch = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
          score = score_fn(batch, vec_t).detach().cpu()

          if i < num_batches:
            scores.append(score)
          else:
            scores.append(score[:extra_in_last_batch])
        
        scores = torch.cat(scores, dim=0)
        scores = torch.flatten(scores, start_dim=1)

        means = scores.mean(dim=0, keepdim=True)
        normalized_scores = scores - means
        #normalized_scores_list.append(normalized_scores.tolist())

        u, s, v = torch.linalg.svd(normalized_scores)
        s = s.tolist()
        singular_values.append(s)

        idx+=1
        pbar.update(1)

  #if name is None:
  #  name = 'svd'

  with open(os.path.join(save_path, 'svd.pkl'), 'wb') as f:
    info = {'singular_values':singular_values}
    pickle.dump(info, f)
  
  #with open(os.path.join(save_path, 'normalized_scores.pkl'), 'wb') as f:
  #  info = {'normalized_scores': normalized_scores_list}
  #  pickle.dump(info, f)