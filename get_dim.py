
import run_lib
from configs.utils import read_config
from dim_reduction import get_manifold_dimension

config_path = 'configs/daniel/daniel.py'
log_path = 'logs/daniel/test/'
config = read_config(config_path)
config.model.checkpoint_path = log_path + 'checkpoints/best/epoch=15616--eval_loss_epoch=109.561.ckpt'
#config.model.checkpoint_path = log_path + 'checkpoints/best/last.ckpt'
config.dim_estimation.num_datapoints = 100
get_manifold_dimension(config)