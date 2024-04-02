#from models import ncsnpp, ddpm, ncsnv2, fcn, ddpm3D, fcn_potential, ddpm_potential, csdi #needed for model registration
from models import SkipMLP, ImprovedDiffusionUnet, BeatGANsEncoderModel, BeatGANsUNET_latent_conditioned, BeatGANsUNET, ddpm, ncsnv2, fcn, ddpm3D, fcn_potential, ddpm_potential, csdi, encoder, decoder, half_U #needed for model registration
import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin
import numpy as np

from torchvision.utils import make_grid

from lightning_callbacks import callbacks, ema, HaarMultiScaleCallback, PairedCallback, AttributeEncoder #needed for callback registration
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image, permute_channels, normalise, normalise_per_band, create_supergrid
from lightning_callbacks.utils import get_callbacks

from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset, SyntheticPairedDataset, Synthetic1DConditionalDataset, SyntheticTimeSeries, SRDataset, SRFLOWDataset, KSphereDataset, MammothDataset, LineDataset, GanDataset, guided_diff_datasets #needed for datamodule registration
from lightning_data_modules.utils import create_lightning_datamodule

from lightning_modules import No_MI_DisentangledScoreVAEmodel, DisentangledScoreVAEmodel, AttributeConditionalModel, AttributeEncoder, BaseSdeGenerativeModel, ScoreVAEmodel, PretrainedScoreVAEmodel, EncoderOnlyPretrainedScoreVAEmodel, CorrectedEncoderOnlyPretrainedScoreVAEmodel, ConditionalSdeGenerativeModel #need for lightning module registration
from lightning_modules.utils import create_lightning_module

from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Resize
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning.strategies import DDPStrategy

import create_dataset
from torch.nn import Upsample
import torch 

from pathlib import Path
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

#additions for manifold dimension estimation
import dim_reduction
import scoreVAE_testing

def train(config, log_path, checkpoint_path, log_name=None):
    print('RESUMING: ' + str(checkpoint_path))
    if config.data.create_dataset:
      create_dataset.create_dataset(config)

    DataModule = create_lightning_datamodule(config)
    callbacks = get_callbacks(config)
    LightningModule = create_lightning_module(config)

    if config.logging.log_path is not None:
      log_path = config.logging.log_path
    if config.logging.log_name is not None:
      log_name = config.logging.log_name

    logger = pl.loggers.TensorBoardLogger(log_path, name='', version=log_name)

    if checkpoint_path is not None or config.model.checkpoint_path is not None:
      if config.model.checkpoint_path is not None and checkpoint_path is None:
        checkpoint_path = config.model.checkpoint_path

    #from pytorch_lightning.callbacks import LearningRateFinder
    #callbacks.append(LearningRateFinder())

    trainer = pl.Trainer(accelerator = config.training.accelerator,
                          strategy = DDPStrategy(find_unused_parameters=True),
                          devices = config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters, 
                          max_epochs =config.training.num_epochs,
                          callbacks=callbacks, 
                          logger = logger,
                          num_sanity_val_steps=0
                          )
    
    #cuda_available = torch.cuda.is_available()
    #map_location = torch.device('cuda') if cuda_available else torch.device('cpu')

    #if checkpoint_path:
    #  LightningModule = LightningModule.load_from_checkpoint(checkpoint_path, config=config, map_location=map_location)

    #LightningModule = torch.compile(LightningModule)
    trainer.fit(LightningModule, datamodule=DataModule, ckpt_path=checkpoint_path)

def test(config, log_path, checkpoint_path):
    if config.data.create_dataset:
      create_dataset.create_dataset(config)

    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    callbacks = get_callbacks(config)

    if config.logging.log_path is not None:
      log_path = config.logging.log_path
    if config.logging.log_name is not None:
      log_name = config.logging.log_name
    logger = pl.loggers.TensorBoardLogger(log_path, name='test', version=log_name)

    checkpoint_path = config.model.checkpoint_path
    pl_module = create_lightning_module(config)
    pl_module = pl_module.load_from_checkpoint(checkpoint_path, config=config)
    pl_module = pl_module.to('cuda')

    scoreVAE_testing.check_changing_attributes(pl_module, DataModule, logger)
    '''
    trainer = pl.Trainer(accelerator = 'gpu' if config.training.gpus > 0 else 'cpu',
                          devices = config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters, 
                          max_epochs =config.training.num_epochs,
                          callbacks=callbacks, 
                          logger = logger                   
                          )
    trainer.test(pl_module, datamodule=DataModule, ckpt_path=checkpoint_path)
  '''


def inspect_corrected_VAE(config):
  dim_reduction.inspect_corrected_VAE(config)
  
def inspect_VAE(config):
  dim_reduction.inspect_VAE(config)

def scoreVAE_fidelity(config):
  dim_reduction.scoreVAE_fidelity(config)

def get_manifold_dimension(config, name=None):
  dim_reduction.get_manifold_dimension(config, name)

def get_conditional_manifold_dimension(config, name=None):
  dim_reduction.get_conditional_manifold_dimension(config, name)