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
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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
  val_dataloader = DataModule.val_dataloader()

  pl_module = create_lightning_module(config)

  if config.training.gpus == 0:
    device = torch.device('cpu')
    checkpoint = torch.load(config.model.checkpoint_path, map_location=device)
    pl_module.load_state_dict(checkpoint['state_dict'])
  else:
    device = torch.device('cuda:0')
    pl_module = pl_module.load_from_checkpoint(config.model.checkpoint_path)

  pl_module.configure_sde(config)
  sde = pl_module.sde

  pl_module = pl_module.to(device)
  pl_module.eval()

  return pl_module, val_dataloader, sde, device, save_path

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
    return encoder_fn

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

  iterations = 10
  encoder_correction_percentages = {}
  auxiliary_correction_percentages = {}
  unconditional_score_percentages = {}
  for i, x in tqdm(enumerate(val_dataloader)):
    if i == iterations:
      break
    
    x = x.to(device)
    latent = encoding_fn(x)

    ts = torch.linspace(start=sde.sampling_eps, end=sde.T, steps=25)
    for t_ in ts:
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

      encoder_correction_percentages[n_t].append(torch.mean(encoder_correction_norm / total_score_norm).item())
      auxiliary_correction_percentages[n_t].append(torch.mean(auxiliary_correction_norm / total_score_norm).item())
      unconditional_score_percentages[n_t].append(torch.mean(unconditional_score_norm / total_score_norm).item())
  
  with open(os.path.join(save_path, 'contribution.pkl'), 'wb') as f:
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

  iterations = 1
  latents = []
  for i, x in tqdm(enumerate(val_dataloader)):
    if i == iterations:
      break

    x = x.to(device)

    t0 = torch.zeros(x.shape[0]).type_as(x)
    latent_distribution_parameters = encoder(x, t0)
    latent_dim = latent_distribution_parameters.size(1)//2
    mean_z = latent_distribution_parameters[:, :latent_dim]
    log_var_z = latent_distribution_parameters[:, latent_dim:]
    latent = mean_z + torch.sqrt(log_var_z.exp())*torch.randn_like(mean_z)
    latents.append(latent)

    
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
    

  
  latents = torch.cat(latents, dim=0)

  flattened = torch.flatten(latents).cpu().detach().numpy()
  

  plt.figure()
  plt.hist(flattened, bins=500, density=True, alpha=0.6, color='b')

  # Plot the PDF.
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 200)
  mu = 0
  std = 1
  p = norm.pdf(x, mu, std)
    
  plt.plot(x, p, 'k', linewidth=2)
  title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
  plt.title(title)
    
  plt.show()
  



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