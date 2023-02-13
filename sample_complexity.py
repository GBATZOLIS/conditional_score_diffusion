import run_lib
from configs.utils import read_config
from dim_reduction import get_manifold_dimension

ns = [500, 1000, 10000, 100000, 1000000]
for n in ns:
    path = f'logs/ksphere/sample_complexity/n={n}/config.pkl'
    config = read_config(path)
    config.model.checkpoint_path = config.logging.log_path  + config.logging.log_name + "/checkpoints/best/last.ckpt"
    config.dim_estimation.num_datapoints = 100
    get_manifold_dimension(config)