"""Config file for synthetic dataset."""
import ml_collections
from configs.jan.holiday.circles.cp import default_cp

def get_config():
  config = default_cp.get_config()

  # training
  training = config.training
  training.LAMBDA = 0.1

  return config
