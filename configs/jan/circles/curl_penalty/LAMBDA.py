"""Config file for synthetic dataset."""
import ml_collections
from configs.jan.circles.curl_penalty import default_cp

def get_config():
  config = default_cp.get_config()

  # training
  training = config.training
  training.LAMBDA=0.0
  training.adaptive = False
  training.visualization_callback = ['2DSamplesVisualization', '2DCurlVisualization', '2DVectorFieldVisualization']

  return config
