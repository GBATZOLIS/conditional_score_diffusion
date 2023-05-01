"""Config file for synthetic dataset."""
import ml_collections
from configs.jan.circles.curl_penalty import default_cp

def get_config():
  config = default_cp.get_config()

  # training
  training = config.training
  training.sde = 'snrsde'
  training.LAMBDA=0.0
  training.adaptive = False
  training.visualization_callback = ['2DSamplesVisualization', '2DCurlVisualization', '2DVectorFieldVisualization']
  training.show_evolution = True

  model = config.model
  model.curl_penalty_type = 'Linfty' #'L2'

  optim = config.optim
  optim.lr = 2e-4
  return config
