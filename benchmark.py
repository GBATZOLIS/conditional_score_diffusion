import torch
import numpy as np
import pandas as pd
import os
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


class Benchmark():

    def __init__(self, file_name, configs_dict) -> None:
        self.file_name = file_name
        self.estimators = ['mle_5', 'mle_20', 'lpca', 'ppca']
        self.configs_dict = configs_dict
        # create a df for results
        self.results = pd.DataFrame(columns=configs_dict.keys(), index=self.estimators)
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
        if pd.isna(self.results[dataset_name].loc[estimator_type]):
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
