import run_lib
import torch
import pandas as pd
import numpy as np
import pickle
from lightning_modules.VAE import VAE
from dim_reduction import inspect_VAE
from utils import fix_rds_path

# get config
from configs.VAE.celebA import get_config
config = get_config()

vae = VAE.load_from_checkpoint('logs/VAE/celeba/kl_0.01/latent_dim_512_v1/checkpoints/last.ckpt', config=config)
from lightning_modules.EncoderOnlyPretrainedScoreVAEmodel import EncoderOnlyPretrainedScoreVAEmodel
with open('/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/paper/pretrained/celebA_64/only_ResNetEncoder_VAE_KLweight_0.01/config.pkl', 'rb') as f:
  config = pickle.load(f)

config.model.time_conditional = True
#diff_vae = EncoderOnlyPretrainedScoreVAEmodel.load_from_checkpoint('/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/paper/pretrained/celebA_64/only_ResNetEncoder_VAE_KLweight_0.01/checkpoints/best/last.ckpt',
#                                                                    config=config)


from lightning_modules.utils import create_lightning_module
model = create_lightning_module(config)
pl_module = model.load_from_checkpoint(fix_rds_path(config.model.checkpoint_path), config=config)