import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_lib
from dim_reduction import get_manifold_dimension
from plot_utils import plot_spectrum, plot_dims

#get home path
from pathlib import Path
home = str(Path.home())

config_path = f'{home}/rds_work/projects/dimension_detection/experiments/style_gan/2_BeatGANsUNetModel/config.pkl'
with open(config_path, 'rb') as f:
    config = pickle.load(f) 
    log_path = config.logging.log_path
    log_name = config.logging.log_name
    save_path = os.path.join(log_path, log_name, 'svd')
    
for i in range(10):
    config.logging.svd_points = 10
    get_manifold_dimension(config, name=f'big_svd_{i}')