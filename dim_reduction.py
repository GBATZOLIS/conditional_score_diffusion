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

def get_conditional_manifold_dimension(config, name=None):
  #---- create the setup ---
  log_path = config.logging.log_path
  log_name = config.logging.log_name
  save_path = os.path.join(log_path, log_name, 'svd')
  Path(save_path).mkdir(parents=True, exist_ok=True)

  config.data.return_labels = True
  DataModule = create_lightning_datamodule(config)
  DataModule.setup()
  train_dataloader = DataModule.val_dataloader()
    
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

  num_datapoints = config.get('dim_estimation.num_datapoints', 26)

  times = torch.linspace(pl_module.sampling_eps, 0.3, 12)
  for t_slice in times:
    t_save_path = os.path.join(log_path, log_name, 'svd', '%.3f' % t_slice.item())
    Path(t_save_path).mkdir(parents=True, exist_ok=True)

    singular_values = []
    labels = []
    imgs = []
    idx = 0
    with tqdm(total=num_datapoints) as pbar:
      for orig_batch, orig_labels in train_dataloader:
        #orig_batch = orig_batch.to(device)
        batchsize = orig_batch.size(0)
        
        if idx+1 >= num_datapoints:
            break
          
        for x, y in zip(orig_batch, orig_labels):
          if y.item() != 1:
            continue

          if idx+1 >= num_datapoints:
            break
          
          imgs.append(x.permute(1, 2, 0))
          x = x.to(device)
          ambient_dim = np.prod(x.shape[1:])
          x = x.repeat([batchsize,]+[1 for i in range(len(x.shape))])

          num_batches = ambient_dim // batchsize + 1
          extra_in_last_batch = ambient_dim - (ambient_dim // batchsize) * batchsize
          num_batches *= 4

          t = t_slice #10*pl_module.sampling_eps
          vec_t = torch.ones(x.size(0), device=device) * t

          scores = []
          for i in tqdm(range(1, num_batches+1)):
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

    imgs = torch.stack(imgs).numpy()
    with open(os.path.join(t_save_path, 'images.pkl'), 'wb') as f:
      info = {'images':imgs}
      pickle.dump(info, f)

    with open(os.path.join(t_save_path, 'labels_svd.pkl'), 'wb') as f:
      info = {'singular_values':singular_values}
      pickle.dump(info, f)
    
    with open(os.path.join(t_save_path, 'labels.pkl'), 'wb') as f:
      info = {'labels': labels}
      pickle.dump(info, f)

def get_manifold_dimension(config, name=None, return_svd=False):
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

  if hasattr(config, 'dim_estimation.num_datapoints'):
    num_datapoints = config.dim_estimation.num_datapoints
  elif hasattr(config, 'logging.svd_points'):
    num_datapoints = config.logging.svd_points
  
  
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
        #normalized_scores_list.append(normalized_scores.tolist())

        u, s, v = torch.linalg.svd(normalized_scores)
        s = s.tolist()
        singular_values.append(s)

        idx+=1
        pbar.update(1)

  #if name is None:
  #  name = 'svd'
  info = {'singular_values':singular_values}
  if return_svd:
    return info
  else:
    with open(os.path.join(save_path, f'{name}.pkl'), 'wb') as f:
      pickle.dump(info, f)
  
  #with open(os.path.join(save_path, 'normalized_scores.pkl'), 'wb') as f:
  #  info = {'normalized_scores': normalized_scores_list}
  #  pickle.dump(info, f)