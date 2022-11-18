#needed for model registration
from models import ddpm, ncsnv2, fcn, ddpm3D, fcn_potential, ddpm_potential, csdi 

import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin
import numpy as np

from torchvision.utils import make_grid

#needed for callback registration
from lightning_callbacks import callbacks
from lightning_callbacks.utils import get_callbacks

#needed for datamodule registration
from lightning_data_modules import ImageDatasets, PairedDataset, SyntheticDataset, SyntheticPairedDataset, Synthetic1DConditionalDataset, SyntheticTimeSeries, SRDataset
from lightning_data_modules.utils import create_lightning_datamodule

#need for lightning module registration
from lightning_modules import BaseSdeGenerativeModel, FokkerPlanckModel 
from lightning_modules.utils import create_lightning_module

from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

import create_dataset
import torch 

from pathlib import Path
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

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

      trainer = pl.Trainer(gpus=config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accelerator = config.training.accelerator, #plugins = DDPPlugin(find_unused_parameters=False) if config.training.accelerator=='ddp' else None,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters, 
                          max_epochs =config.training.num_epochs,
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=checkpoint_path
                          )
    else:  
      trainer = pl.Trainer(gpus=config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accelerator = config.training.accelerator,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters,
                          max_epochs =config.training.num_epochs,
                          callbacks=callbacks,
                          logger = logger                          
                          )

    trainer.fit(LightningModule, datamodule=DataModule)

def test(config, log_path, checkpoint_path):
    eval_log_path = os.path.join(config.eval.base_log_dir, config.data.task, config.data.dataset, config.training.conditioning_approach)
    Path(eval_log_path).mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=eval_log_path, name='test_metrics')

    DataModule = create_lightning_datamodule(config)
    DataModule.setup()

    callbacks = get_callbacks(config)

    if checkpoint_path is not None or config.model.checkpoint_path is not None:
      if config.model.checkpoint_path is not None and checkpoint_path is None:
        checkpoint_path = config.model.checkpoint_path
    else:
      return 'Testing cannot be completed because no checkpoint has been provided.'

    LightningModule = create_lightning_module(config, checkpoint_path)

    trainer = pl.Trainer(gpus=config.training.gpus,
                         num_nodes = config.training.num_nodes,
                         accelerator = 'ddp',
                         accumulate_grad_batches = config.training.accumulate_grad_batches,
                         gradient_clip_val = config.optim.grad_clip,
                         max_steps=config.training.n_iters, 
                         callbacks=callbacks, 
                         logger = logger)
    
    trainer.test(LightningModule, test_dataloaders = DataModule.test_dataloader())