import run_lib
from configs.utils import read_config
from dim_reduction import get_manifold_dimension
import glob

#ns = [10, 100, 250, 500]
ns=[250]
for n in ns:
    path = f'logs/ksphere/sample_complexity/n={n}/config.pkl'
    config = read_config(path)
    config.model.checkpoint_path = glob.glob(f'logs/ksphere/sample_complexity/n={n}/checkpoints/best/epoch*')[0] #config.logging.log_path  + config.logging.log_name + "/checkpoints/best/last.ckpt"
    config.dim_estimation.num_datapoints = 100
    get_manifold_dimension(config)