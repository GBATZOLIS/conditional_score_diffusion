import run_lib
from lightning_data_modules.SyntheticDataset import SyntheticDataModule
from configs.utils import read_config

config = read_config('configs/jan/circles/fokker_planck/VESDE/0_small.py')
config.data.n_circles = 3
data_m = SyntheticDataModule(config)
data_m.setup()