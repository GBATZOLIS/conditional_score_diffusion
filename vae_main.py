
import os
import sys
from pathlib import Path
repo_path=Path('.').absolute()
sys.path.append(f'{repo_path}/janutils')
from configs.utils import read_config
from argparse import ArgumentParser

from vae_run_lib import train, train_sb, test_sb, evaluate

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=str)
    parser.add_argument('--eval_path', type=str, default='none')
    parser.add_argument('--latent_dim', type=str, default='none')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.latent_dim != 'none':
        config.model.latent_dim = int(args.latent_dim)
        config.encoder.latent_dim = int(args.latent_dim)
        config.decoder.latent_dim = int(args.latent_dim)
    print(f'Latent dim: {config.model.latent_dim}')
    if args.mode == 'train':
        train(config)
    elif args.mode == 'train_sb':
        train_sb(config)
    elif args.mode == 'test_sb':
        test_sb(config)
    elif args.mode == 'eval':
        if args.eval_path == 'none':
            evaluate(config)
        else:
            evaluate(config, args.eval_path)