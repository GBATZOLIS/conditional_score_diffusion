import torch
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from configs.utils import read_config
from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset, SyntheticPairedDataset, Synthetic1DConditionalDataset, SyntheticTimeSeries, SRDataset, SRFLOWDataset, CryptoDataset, KSphereDataset, MammothDataset, LineDataset
from lightning_data_modules.utils import create_lightning_datamodule

#Import R packages
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from rpy2.robjects.packages import importr
r_base = importr('base')
intdimr = importr('intrinsicDimension')


mammoth_config = read_config('configs/mammoth/vesde.py')
uniform_10_config = read_config('configs/ksphere/N_1/uniform_10.py')
uniform_50_config = read_config('configs/ksphere/N_1/uniform_50.py')
line_config = read_config('configs/line/vesde.py')

configs_dict = {
    'mammoth': mammoth_config,
    'uniform_10': uniform_10_config,
    'unifrom_50': uniform_50_config,
    'line': line_config
}

results = pd.DataFrame(columns=configs_dict.keys(), index=['mle_5', 'mle_20', 'lpca'])
results.index.name = 'method'

for name, config in configs_dict.items():
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    train_dataloader = DataModule.train_dataloader()
    X=[]
    for _, x in enumerate(train_dataloader):
        X.append(x)
    data_r = torch.cat(X, dim=0).numpy()
    dim = data_r.shape[1]

    #MLE estimator
    k=5
    #mle = intdimr.maxLikPointwiseDimEst(data_r, k=k).rx2('dim.est')
    mle = intdimr.maxLikGlobalDimEst(data_r, k=k).rx2('dim.est')
    results[name].loc['mle_5'] = mle[0]
    results.to_csv('benchmark.csv')

    k=20
    #mle = intdimr.maxLikPointwiseDimEst(data_r, k=k).rx2('dim.est')
    mle = intdimr.maxLikGlobalDimEst(data_r, k=k).rx2('dim.est')
    results[name].loc['mle_20'] = mle[0]
    results.to_csv('benchmark.csv')

    #Local PCE
    lpca = intdimr.pcaLocalDimEst(data_r, 'FO').rx2('dim.est')
    results[name].loc['lpca'] = lpca[0]
    results.to_csv('benchmark.csv')