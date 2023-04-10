
import os
import sys
from pathlib import Path
repo_path=Path('.').absolute()
sys.path.append(f'{repo_path}/janutils')
from configs.utils import read_config
from argparse import ArgumentParser

from vae_run_lib import train, evaluate

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = read_config(args.config)
    print(f'Latent dim: {config.model.latent_dim}')
    if args.mode == 'train':
        train(config)
    elif args.mode == 'eval':
        #config.eval_path  = 'logs/VAE/celeba/kl_0.01/latent_dim_512/checkpoints/epoch=139--val_loss=12.000.ckpt'
        evaluate(config)