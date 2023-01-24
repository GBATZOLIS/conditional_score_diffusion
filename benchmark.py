import torch
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from configs.utils import read_config
from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset, SyntheticPairedDataset, Synthetic1DConditionalDataset, SyntheticTimeSeries, SRDataset, SRFLOWDataset, CryptoDataset, KSphereDataset, MammothDataset, LineDataset
from lightning_data_modules.utils import create_lightning_datamodule
from sklearn.decomposition import PCA

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
non_uniform_10_1_config = read_config('configs/ksphere/N_1/non_uniform_1.py')
non_uniform_10_075_config = read_config('configs/ksphere/N_1/non_uniform_075.py')
non_uniform_10_05_config = read_config('configs/ksphere/N_1/non_uniform_05.py')
squares_10_3_5_config = read_config('configs/fixedsquaresmanifold/10_3_5.py')
squares_20_3_5_config = read_config('configs/fixedsquaresmanifold/20_3_5.py')
squares_100_3_5_config = read_config('configs/fixedsquaresmanifold/100_3_5.py')
gaussian_manifold_10 = read_config('configs/fixedgaussiansmanifold/10.py')
gaussian_manifold_20 = read_config('configs/fixedgaussiansmanifold/20.py')
gaussian_manifold_100 = read_config('configs/fixedgaussiansmanifold/100.py')

configs_dict = {
    'mammoth': mammoth_config,
    'uniform_10': uniform_10_config,
    'unifrom_50': uniform_50_config,
    'line': line_config,
    'non_uniform_10_1': non_uniform_10_1_config,
    'non_uniform_10_075': non_uniform_10_075_config,
    'non_uniform_10_05': non_uniform_10_05_config,
    'squares_10': squares_10_3_5_config,
    'squares_20': squares_20_3_5_config,
    'squares_100': squares_100_3_5_config,
    'gaussian_manifold_10': gaussian_manifold_10,
    'gaussian_manifold_20': gaussian_manifold_20,
    'gaussian_manifold_100': gaussian_manifold_100

}

# create a df for results
results = pd.DataFrame(columns=configs_dict.keys(), index=['mle_5', 'mle_20', 'lpca', 'ppca'])
results.index.name = 'method'

# load what is already saved
exisiting_results = pd.read_csv('benchmark.csv', index_col='method')
results.update(exisiting_results)



for name, config in configs_dict.items():
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    train_dataloader = DataModule.train_dataloader()
    X=[]
    for _, x in enumerate(train_dataloader):
        X.append(x.view(x.shape[0],-1))
    data_np = torch.cat(X, dim=0).numpy()
    data_np.reshape(data_np.shape[0],-1).shape
    dim = data_np.shape[1]

    #MLE estimator k=5
    if pd.isna(results[name].loc['mle_5']):
        k=5
        #mle = intdimr.maxLikPointwiseDimEst(data_r, k=k).rx2('dim.est')
        mle = intdimr.maxLikGlobalDimEst(data_np, k=k).rx2('dim.est')
        results[name].loc['mle_5'] = mle[0]
        results.to_csv('benchmark.csv')
    print(f'mle_5 on {name} DONE')

    #MLE estimator k=20
    if pd.isna(results[name].loc['mle_20']):
        k=20
        #mle = intdimr.maxLikPointwiseDimEst(data_r, k=k).rx2('dim.est')
        mle = intdimr.maxLikGlobalDimEst(data_np, k=k).rx2('dim.est')
        results[name].loc['mle_20'] = mle[0]
        results.to_csv('benchmark.csv')
    print(f'mle_20 on {name} DONE')

    #Local PCE
    if pd.isna(results[name].loc['lpca']):
        lpca = intdimr.pcaLocalDimEst(data_np, 'FO').rx2('dim.est')
        results[name].loc['lpca'] = lpca[0]
        results.to_csv('benchmark.csv')
    print(f'lpca on {name} DONE')

    #PPCA
    if pd.isna(results[name].loc['ppca']):
        pca = PCA(n_components='mle')
        pca.fit(data_np.astype(np.float64))
        results[name].loc['ppca'] = pca.n_components_
        results.to_csv('benchmark.csv')
    print(f'ppca on {name} DONE')

