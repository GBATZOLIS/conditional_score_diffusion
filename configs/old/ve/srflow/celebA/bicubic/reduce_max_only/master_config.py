from configs.ve.srflow.bicubic.reduce_max_only import config_160, config_80, config_40
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.config_40 = config_40.get_config()
    config.config_80 = config_80.get_config()
    config.config_160 = config_160.get_config()
    return config