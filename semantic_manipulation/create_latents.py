import sys
import os
from configs.utils import fix_rds_path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import run_lib
import torch
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from lightning_modules.VAE import VAE
from configs.utils import fix_config
from torch.utils.data import DataLoader
from lightning_data_modules.ImageDatasets import CelebAAnnotatedDataset
from lightning_modules.utils import create_lightning_module
import argparse

# define args
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
# default_path = '/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/paper/pretrained/celebA_64/only_ResNetEncoder_VAE_KLweight_0.01/config.pkl'
default_path = '/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/gd_ffhq/only_encoder_ddpm_plus_smld_VAE_KLweight_0.01_DiffDecoders_continuous_prior/config.pkl'
parser.add_argument('--model_path', default=default_path)
parser.add_argument('--save_labels', default=True)


def main(args): 

    # load pretrained model
    print('loading pretrained model')
    with open(args.model_path, 'rb') as f:
        config = pickle.load(f)
    config = fix_config(config)
    model = create_lightning_module(config)
    pl_module = model.load_from_checkpoint(config.model.checkpoint_path, config=config)
    pl_module.eval()

    # edit config
    home = os.path.expanduser('~')
    config.data.base_dir = f'{home}/rds_work/datasets/'
    config.data.dataset = 'celebA-HQ-160'
    config.data.attributes = ['Male']
    config.data.normalization_mode = 'gd'

    train_dataset = CelebAAnnotatedDataset(config, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = CelebAAnnotatedDataset(config, phase='val') #test and val are the same
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # create latent dataset
    print('creating latent dataset')
    with torch.no_grad():
        device = 'cuda'        
        pl_module = pl_module.to(device)
        Z_train = []
        y_train = []
        for x, y in tqdm(train_dataloader):
            Z_train.append(pl_module.encode(x.to(device)))
            y_train.append(y)
        Z_train = torch.cat(Z_train, dim=0)
        y_train = torch.cat(y_train, dim=0)

        Z_test = []
        y_test = []
        for x, y in tqdm(test_dataloader):
            Z_test.append(pl_module.encode(x.to(device)))
            y_test.append(y)
        Z_test = torch.cat(Z_test, dim=0)
        y_test = torch.cat(y_test, dim=0)

    # convert to numpy
    Z_train = Z_train.detach().cpu().numpy()
    Z_test = Z_test.detach().cpu().numpy()


    # create save directory
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # save labels
    if args.save_labels:
        print('saving labels')
        y_train = y_train.detach().cpu().numpy().flatten()
        y_test = y_test.detach().cpu().numpy().flatten()
        with open('tmp/y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open('tmp/y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)


    # pickle the latent dataset 
    print('pickling train latent data')
    with open('tmp/Z_train.pkl', 'wb') as f:
        pickle.dump(Z_train, f)

    print('pickling test latent data')
    with open('tmp/Z_test.pkl', 'wb') as f:
        pickle.dump(Z_test, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
