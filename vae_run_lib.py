import os
import sys
import pickle
from pathlib import Path
repo_path=Path('.').absolute()
sys.path.append(f'{repo_path}/janutils')
import pytorch_lightning as pl
from configs.utils import read_config
from lightning_data_modules.SRFLOWDataset import UnpairedDataModule
from lightning_data_modules.ImageDatasets import Cifar10DataModule, ImageDataModule
from lightning_modules.VAE import VAE, MarkovianProjector
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
#from janutils.callbacks import VisualizationCallback
from argparse import ArgumentParser
from pytorch_lightning.callbacks import Callback
import torchvision
import numpy as np
from pathlib import Path
import torch

 
def train_sb(config):
    VAE_config = read_config(config.VAE_config_path)
    vae_model = VAE.load_from_checkpoint(checkpoint_path=VAE_config.model.checkpoint_path, config=VAE_config)

    if VAE_config.training.variational:
        kl_weight = VAE_config.model.kl_weight
    else:
        kl_weight = -1

    # create dataloaders
    DataModule = get_datamodule(VAE_config)

    # create log name
    dataset_name = VAE_config.data.dataset #f'{config.data.dataset}_{data_module.digit}' if config.data.dataset == 'MNIST' else config.data.dataset
    kl_weight = f'kl_{kl_weight}'

    if hasattr(VAE_config, 'log_path'):
        log_path = VAE_config.log_path
    else:
        log_path = 'logs/'
    
    tb_name = os.path.join('VAE', dataset_name, kl_weight)
    
    # create new version
    i = 0
    tb_version = f'latent_dim_{VAE_config.model.latent_dim}'
    while os.path.exists(os.path.join(log_path, tb_name, tb_version)):
        i +=1
        tb_version = f'latent_dim_{VAE_config.model.latent_dim}_v{i}'

    # pickle config
    Path(os.path.join(log_path, tb_name, tb_version)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_path, tb_name, tb_version, 'sb_config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    # logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path, 
                                             name=tb_name, 
                                             version=tb_version
                                             )
    # callbacks
    checkpoint_callback =  ModelCheckpoint(dirpath=os.path.join(log_path, 
                                                                tb_name,
                                                                tb_version,
                                                                'sb_checkpoints'),
                                            monitor='val_loss',
                                            filename='{epoch}--{val_loss:.3f}',
                                            save_last=True,
                                            save_top_k=3)


    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=100)
    lr_monitor = LearningRateMonitor()

    # trainer
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=100000,
                        logger=tb_logger,
                        #resume_from_checkpoint=args.checkpoint,
                        callbacks=[checkpoint_callback,
                                   lr_monitor,
                                   early_stop_callback
                                   ]
                        )

    model = MarkovianProjector(config, vae_model)
    trainer.fit(model, datamodule=DataModule, ckpt_path=config.model.checkpoint_path)

def train(config):
    #config_path = 'janutils/configs/mnist_mlp_autoencoder.py'
    #config = read_config(config_path)
    #config.model.latent_dim = args.latent_dim

    model = VAE(config)
    if config.training.variational:
        kl_weight = config.model.kl_weight
    else:
        kl_weight = -1

    data_module = get_datamodule(config)
    
    # create dataloaders
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # create log name
    dataset_name = config.data.dataset #f'{config.data.dataset}_{data_module.digit}' if config.data.dataset == 'MNIST' else config.data.dataset
    kl_weight = f'kl_{kl_weight}'

    if hasattr(config, 'log_path'):
        log_path = config.log_path
    else:
        log_path = 'logs/'
    
    tb_name = os.path.join('VAE', dataset_name, kl_weight)
    
    # create new version
    i = 0
    tb_version = f'latent_dim_{config.model.latent_dim}'
    while os.path.exists(os.path.join('logs', tb_name, tb_version)):
        i +=1
        tb_version = f'latent_dim_{config.model.latent_dim}_v{i}'

    # pickle config
    Path(os.path.join(log_path, tb_name, tb_version)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_path, tb_name, tb_version, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    # logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path, 
                                             name=tb_name, 
                                             version=tb_version
                                             )
    # callbacks
    checkpoint_callback =  ModelCheckpoint(dirpath=os.path.join(log_path, 
                                                                tb_name,
                                                                tb_version,
                                                                'checkpoints'),
                                            monitor='val_loss',
                                            filename='{epoch}--{val_loss:.3f}',
                                            save_last=True,
                                            save_top_k=5)
    
    #visualization_callback = VisualizationCallback()
    class VisualizationCallback(Callback):
        def __init__(self, freq):
            super().__init__()
            self.freq=freq

        def on_validation_epoch_end(self, trainer, pl_module):
            current_epoch = pl_module.current_epoch
            if (current_epoch+1) % self.freq == 0:
                dataloader = trainer.val_dataloaders
                batch = next(iter(dataloader))
                
                grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
                pl_module.logger.experiment.add_image('original', grid_batch)

                batch = batch.to(pl_module.device)

                B = batch.shape[0]
                if pl_module.config.training.variational:
                    mean_z, log_var_z = pl_module.encode(batch)
                    z = torch.randn((B, pl_module.latent_dim), device=pl_module.device) * torch.sqrt(log_var_z.exp()) + mean_z
                    mean_x, _ = pl_module.decode(z)
                else:
                    z, _ = pl_module.encode(batch)
                    mean_x, _ = pl_module.decode(z)
                
                sample = mean_x.cpu()
                grid_batch = torchvision.utils.make_grid(sample, nrow=int(np.sqrt(sample.size(0))), normalize=True, scale_each=True)
                pl_module.logger.experiment.add_image('reconstruction', grid_batch, current_epoch)

                

    visualization_callback = VisualizationCallback(freq=100)
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=100)
    lr_monitor = LearningRateMonitor()

    # trainer
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=100000,
                        logger=tb_logger,
                        #resume_from_checkpoint=args.checkpoint,
                        callbacks=[checkpoint_callback, 
                                   visualization_callback,
                                   lr_monitor,
                                   early_stop_callback
                                   ]
                        )

    trainer.fit(model, train_dataloader, val_dataloader)


def evaluate(config, eval_path = None):
    data_module = get_datamodule(config)

    trainer = pl.Trainer(accelerator='gpu', devices=1)
    if eval_path is None:
        eval_path = config.eval_path
    model = VAE.load_from_checkpoint(eval_path)
    output = trainer.test(datamodule=data_module, model=model)
    print(output)
    return output


def get_datamodule(config):
    if config.data.dataset == 'cifar10':
        data_module = Cifar10DataModule(config)
    elif config.data.dataset == 'celeba':
        data_module = UnpairedDataModule(config)
    elif config.data.dataset == 'mnist':
        data_module = ImageDataModule(config)
    return data_module


#config = read_config('configs/VAE/celebA.py')
#config = read_config('configs/VAE/cifar.py')
#train(config)