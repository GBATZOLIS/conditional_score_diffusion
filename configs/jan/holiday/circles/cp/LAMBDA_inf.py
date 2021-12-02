"""Config file for synthetic dataset."""
import ml_collections
from configs.jan.holiday.circles.cp import default_cp

def get_config():
  config = default_cp.get_config()

  # training
  training = config.training
  training.LAMBDA=0

  # model
  model = config.model
  model.name = 'fcn_potential'
  model.state_size = 2
  model.hidden_layers = 3
  model.hidden_nodes = 128
  model.dropout = 0.25
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999

  return config
