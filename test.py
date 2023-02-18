import run_lib
from lightning_data_modules.ImageDatasets import MNISTLatentDataset, MNISTDataset
from configs.utils import read_config


config = read_config('configs/mnist/unconditional_jan.py')
config.data.encoder_path = 'latent_dim_100.ckpt'
MNISTLatentDataset(config)
#dataset = MNISTDataset(config)
#dataset.__getitem__(0).shape