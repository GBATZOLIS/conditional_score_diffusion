import importlib
import os
import re
import subprocess

def read_config(config_path):
    module_name = re.findall('configs/[\w|/ | \.]+.py', config_path)[0][:-3].replace('/','.')
    module = importlib.import_module(module_name)
    config = module.get_config()
    return config


def get_path(path_type):
    user = subprocess.run(['whoami'], capture_output=True, text=True, shell=True, executable='/bin/bash').stdout
    machine = subprocess.run(['hostname'], capture_output=True, text=True, shell=True, executable='/bin/bash').stdout
    print(f'User: {user}\nMachine: {machine}')
    # Jan's configuration
    if user[:6] == 'js2164':
        # holiday
        if machine[:7] in ['holiday', 'holly-a']:
            path_dict = {
                'data_path': '/store/CIA/js2164/data'
            }
        # hpc
        elif machine[:5] == 'login' or machine[:3] == 'gpu':
            path_dict = {
                'data_path': '/rds/user/js2164/hpc-work/data/'
            }
        else:
            raise RuntimeError('Unknown machine. Please define the paths.')
    # Georgios's configuration
    elif user[:5] == 'gb511':
        print(machine)
        if machine[:11] == 'abg-h2ocool':
            path_dict = {
                'data_path': '/home/gb511/datasets'
            }
        elif machine[:5] == 'login': # hpc
            path_dict = {
                'data_path': '/home/gb511/datasets'
            }
        else:
            raise RuntimeError('Unknown machine. Please define the paths.')
    return path_dict[path_type]


def config_translator(config, mode):
    if mode == 'time_dependent_DDPM_encoder':
        config.model.scale_by_sigma = config.encoder.scale_by_sigma
        config.model.ema_rate = config.encoder.ema_rate
        config.model.dropout = config.encoder.dropout
        config.model.normalization = config.encoder.normalization
        config.model.nonlinearity = config.encoder.nonlinearity
        config.model.nf = config.encoder.nf
        config.model.ch_mult = config.encoder.ch_mult
        config.model.num_res_blocks = config.encoder.num_res_blocks
        config.model.attn_resolutions = config.encoder.attn_resolutions
        config.model.resamp_with_conv = config.encoder.resamp_with_conv
        config.model.time_conditional = config.encoder.time_conditional
        config.model.init_scale = config.encoder.init_scale
        config.model.embedding_type = config.encoder.embedding_type
        config.model.conv_size = config.encoder.conv_size
        config.model.input_channels = config.encoder.input_channels
        config.model.output_channels = config.encoder.output_channels
        config.model.encoder_input_channels = config.encoder.input_channels
        config.data.latent_dim = config.model.latent_dim
        config.training.variational = config.model.variational
    return config


def fix_rds_path(path):
    if path is None:
        return None
    home_path = os.path.expanduser('~')
    path = path.replace('/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/', f'{home_path}/rds_work/')
    path = path.replace('/home/gb511/', f'{home_path}/')
    return path


def fix_config(config):
    config.data.base_dir = fix_rds_path(config.data.base_dir)
    config.model.checkpoint_path = fix_rds_path(config.model.checkpoint_path)
    config.training.prior_checkpoint_path = fix_rds_path(config.training.prior_checkpoint_path)
    config.training.prior_config_path = fix_rds_path(config.training.prior_config_path)
    config.logging.log_path = fix_rds_path(config.logging.log_path)
    config.model.time_conditional = True
    return config
    
