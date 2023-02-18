import run_lib
from lightning_data_modules.ImageDatasets import MNISTLatentDataset, MNISTDataset
from configs.utils import read_config

config = read_config('configs/mnist/unconditional_jan.py')
config.data.encoder_path = 'logs/latent_dim_100.ckpt'
dataset =MNISTLatentDataset(config)
print(dataset.__getitem__(0))
