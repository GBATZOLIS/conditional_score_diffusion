"""Config file for synthetic dataset."""
import ml_collections
from configs.jan.circles.cp import default_cp

def get_config():
  config = default_cp.get_config()

  # training
  training = config.training
  training.num_epochs = 50000 #7000
  training.LAMBDA=0

  model = config.model
  model.checkpoint_path = "potential/lightning_logs/version_4/checkpoints/epoch=24999-step=1999999.ckpt"

  return config
