
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
    parser.add_argument('--eval_path', type=str, default='none')
    args = parser.parse_args()
    config = read_config(args.config)
    print(f'Latent dim: {config.model.latent_dim}')
    if args.mode == 'train':
        train(config)
    elif args.mode == 'eval':
        if args.eval_path == 'none':
            evaluate(config)
        else:
            evaluate(config, args.eval_path)