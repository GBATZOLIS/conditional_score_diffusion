import os
import sys
#repo_path='/rds/user/js2164/hpc-work/repos/autoencoder_dim_reduction' 
repo_path = '/store/CIA/js2164/repos/diffusion/score_sde_pytorch'
sys.path.append(f'{repo_path}/janutils')
import pytorch_lightning as pl
from configs.utils import read_config
from lightning_data_modules.SRFLOWDataset import UnpairedDataModule
from lightning_modules.VAE import VAE
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from janutils.callbacks import VisualizationCallback



def train(config):
    #config_path = 'janutils/configs/mnist_mlp_autoencoder.py'
    #config = read_config(config_path)
    #config.model.latent_dim = args.latent_dim

    model = VAE(config)
    kl_weight = config.model.kl_weight

    if config.data.dataset == 'cifar10':
        pass
        #data_module = Cifar10DataModule(config)
    elif config.data.dataset == 'celeba':
        data_module = UnpairedDataModule(config)
    #else:
        #data_module = ImageDataModule(config)
        
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    dataset_name = config.data.dataset #f'{config.data.dataset}_{data_module.digit}' if config.data.dataset == 'MNIST' else config.data.dataset
    kl_weight = f'kl_{kl_weight}'
    tb_name = os.path.join('VAE', dataset_name, kl_weight)
    
    i = 0
    tb_version = f'latent_dim_{config.model.latent_dim}'
    while os.path.exists(os.path.join('logs', tb_name, tb_version)):
        i +=1
        tb_version = f'latent_dim_{config.model.latent_dim}_v{i}'
        
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", 
                                             name=tb_name, 
                                             version=tb_version
                                             )
    
    checkpoint_callback =  ModelCheckpoint(dirpath=os.path.join('logs', 
                                                                tb_name,
                                                                tb_version,
                                                                'checkpoints'),
                                            monitor='val_loss',
                                            filename='{epoch}--{val_loss:.3f}',
                                            save_last=True,
                                            save_top_k=5)
    
    visualization_callback = VisualizationCallback()

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=10)
    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(gpus=1, 
                        max_epochs=1000,
                        logger=tb_logger,
                        #resume_from_checkpoint=args.checkpoint,
                        callbacks=[checkpoint_callback, 
                                   visualization_callback,
                                   lr_monitor
                                   #, early_stop_callback
                                   ]
                        )

    trainer.fit(model, train_dataloader, val_dataloader)


config = read_config('configs/VAE/celebA.py')
train(config)