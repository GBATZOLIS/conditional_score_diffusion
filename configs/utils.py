import importlib

def read_config(config_path):
    module = importlib.import_module(config_path[:-3].replace('/','.'))
    config = module.get_config()
    return config