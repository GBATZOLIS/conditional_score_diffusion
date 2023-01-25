import torch
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
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


<<<<<<< HEAD
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
file_name='benchmark.csv'
exisiting_results = pd.read_csv(file_name, index_col='method')
results.update(exisiting_results)


print('--------- STARTING BENCHAMRK -----------')

for name, config in configs_dict.items():
    print(f'------ Benchamrking on dataset {name} --------')

    if pd.isna(results[name]).any():
        print(f'------ Creating dataset: {name} --------')
        DataModule = create_lightning_datamodule(config)
        DataModule.setup()
        train_dataloader = DataModule.train_dataloader()
        X=[]
        for _, x in enumerate(train_dataloader):
            X.append(x.view(x.shape[0],-1))
        data_np = torch.cat(X, dim=0).numpy()
        data_np.reshape(data_np.shape[0],-1).shape
        dim = data_np.shape[1]
        print(f'------ Dataset {name} created --------')

    #MLE estimator k=5
    if pd.isna(results[name].loc['mle_5']):
        print(f'mle_5 on {name} START')
        k=5
        #mle = intdimr.maxLikPointwiseDimEst(data_r, k=k).rx2('dim.est')
        mle = intdimr.maxLikGlobalDimEst(data_np, k=k).rx2('dim.est')
        results[name].loc['mle_5'] = mle[0]
        results.to_csv(file_name)
    print(f'mle_5 on {name} DONE')

    #MLE estimator k=20
    if pd.isna(results[name].loc['mle_20']):
        print(f'mle_20 on {name} START')
        k=20
        #mle = intdimr.maxLikPointwiseDimEst(data_r, k=k).rx2('dim.est')
        mle = intdimr.maxLikGlobalDimEst(data_np, k=k).rx2('dim.est')
        results[name].loc['mle_20'] = mle[0]
        results.to_csv(file_name)
    print(f'mle_20 on {name} DONE')

    #Local PCA
    if pd.isna(results[name].loc['lpca']):
        print(f'lpca on {name} START')
        lpca = intdimr.pcaLocalDimEst(data_np, 'FO').rx2('dim.est')
        results[name].loc['lpca'] = lpca[0]
        results.to_csv(file_name)
    print(f'lpca on {name} DONE')

    #PPCA
    if pd.isna(results[name].loc['ppca']):
        print(f'ppca on {name} START')
        pca = PCA(n_components='mle')
        pca.fit(data_np.astype(np.float64))
        results[name].loc['ppca'] = pca.n_components_
        results.to_csv(file_name)
    print(f'ppca on {name} DONE')

    print(f'------ Benchamrking on dataset {name} compleated --------')
=======
class Benchmark():

    def __init__(self, file_name, configs_dict) -> None:
        self.file_name = file_name
        # create a df for results
        self.results = pd.DataFrame(columns=configs_dict.keys(), index=['mle_5', 'mle_20', 'lpca', 'ppca'])
        self.results.index.name = 'method'
        # load what is already saved
        if os.path.exists(self.file_name):
            file_name=self.file_name
            exisiting_results = pd.read_csv(file_name, index_col='method')
            self.results.update(exisiting_results)


    def run(self):
        print('--------- STARTING BENCHAMRK -----------')
        for dataset_name, config in self.configs_dict.items():
            print(f'------ Benchamrking on dataset {dataset_name} --------')
            try:
                data = self.create_dataset(dataset_name, config)
            except Exception as e:
                print(f'!!!!------ ERROR: Couldnt create dataset {dataset_name}------!!!!')
                print(e)
            for estimator_type in self.estimators:
                try:
                    self.evaluate_estimator(data, estimator_type=estimator_type, dataset_name=dataset_name)
                except Exception as e:
                    print(f'!!!!------ ERROR: Couldnt evaluate {estimator_type} on dataset {dataset_name}------!!!!')
                    print(e)
            print(f'------ Benchamrking on dataset {dataset_name} compleated --------')


    def evaluate_estimator(self, data, estimator_type, dataset_name):
        if pd.isna(self.results[dataset_name].loc['mle_5']):
            print(f'{estimator_type} on {dataset_name} START')
            if estimator_type == 'mle_5':
                k=5
                estimated_dim = intdimr.maxLikGlobalDimEst(data, k=k).rx2('dim.est')[0]
            elif estimator_type == 'mle_20':
                k=20
                estimated_dim = intdimr.maxLikGlobalDimEst(data, k=k).rx2('dim.est')[0]
            elif estimator_type == 'lpca':
                estimated_dim = intdimr.pcaLocalDimEst(data, 'FO').rx2('dim.est')[0]
            elif estimator_type == 'ppca':
                pca = PCA(n_components='mle')
                pca.fit(data.astype(np.float64))
                estimated_dim = pca.n_components_

            self.results[dataset_name].loc[f'{estimator_type}'] = estimated_dim
            self.results.to_csv(self.file_name)

            print(f'{estimator_type} on {dataset_name} DONE')
        else:
            print(f'{estimator_type} on {dataset_name} was already benchmarked')

    def create_dataset(self, dataset_name, config):        
        if pd.isna(self.results[dataset_name]).any():
                print(f'------ Creating dataset: {dataset_name} --------')
                DataModule = create_lightning_datamodule(config)
                DataModule.setup()
                train_dataloader = DataModule.train_dataloader()
                X=[]
                for _, x in enumerate(train_dataloader):
                    X.append(x.view(x.shape[0],-1))
                data_np = torch.cat(X, dim=0).numpy()
                data_np.reshape(data_np.shape[0],-1).shape
                print(f'------ Dataset {dataset_name} created --------')
                return data_np
        else:
            print(f'------ Dataset {dataset_name} was already benchamrked ------')
>>>>>>> 9b91bdaf8025d72215e10afc6248e3b3554245c8
