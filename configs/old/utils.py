import importlib
import re
import subprocess
import pickle

def read_config(config_path):
    if config_path[-3:] == 'pkl':
        with open(config_path, 'rb') as file:
            config = pickle.load(file)
    elif config_path[-2:] == 'py':
        module_name = re.findall('configs/[\w|/ | \.]+.py', config_path)[0][:-3].replace('/','.')
        module = importlib.import_module(module_name)
        config = module.get_config()
    else:
        raise RuntimeError('Unsupported config format.')
    return config


def get_path(path_type):
    user = subprocess.run(['whoami'], capture_output=True, text=True, shell=True, executable='/bin/bash').stdout
    machine = subprocess.run(['hostname'], capture_output=True, text=True, shell=True, executable='/bin/bash').stdout
    print(f'User: {user}\nMachine: {machine}')
    # Jan's configuration
    if user[:6] == 'js2164':
        # holiday
        if machine[:7] == 'holiday':
            path_dict = {
                'data_path': '/store/CIA/js2164/data'
            }
        # hpc
        elif machine[:5] == 'login':
            path_dict = {
                'data_path': '/rds/user/js2164/hpc-work/data/'
            }
        else:
            raise RuntimeError('Unknown machine. Please define the paths.')
    # Georgios's configuration
    elif user[:5] == 'gb511':
        # hpc
        if machine[:5] == 'login':
            path_dict = {
                'data_path': 'georgios_hpc_data_path'
            }
        else:
            raise RuntimeError('Unknown machine. Please define the paths.')
    return path_dict[path_type]