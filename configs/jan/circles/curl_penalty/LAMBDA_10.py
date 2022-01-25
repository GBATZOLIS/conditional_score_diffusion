"""Config file for synthetic dataset."""
import ml_collections
from configs.jan.circles.curl_penalty import default_cp

def get_config():
  config = default_cp.get_config()

  # training
  training = config.training
  training.LAMBDA=10
  config.model.curl_penalty_type = 'L2'
  training.adaptive = False

  return config
