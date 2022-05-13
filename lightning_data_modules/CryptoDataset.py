import torch
from . import utils
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset

class CryptoDataset(Dataset):

    def __init__(self, coin_name='BTC', time_freq='d', *args, **kwargs):
        super().__init__()
        path = '/store/CIA/js2164/data/crypto/binance/price/Binance_' + coin_name + 'USDT_' + time_freq + '.csv'
        price = pd.read_csv(path, parse_dates=True, header=1, index_col='date')
        price = price['close'].sort_index(ascending=True)
        returns = (price / price.shift(1) - 1).dropna()

        self.data_df = returns
        times = torch.tensor(range(self.data_df.size)) #torch.tensor([d.timestamp() for d in self.data_df.index]).float() #(N,)
        values = torch.tensor(self.data_df.to_numpy()).unsqueeze(-1).float() # (N, D)
        self.time_series = {'times': times, 'values': values}
        self.n = len(self.time_series['times'])
        self.sequences = self.generate_sequences(self.time_series, timestep=True, *args, **kwargs)

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        out = []
        y = sequence['input']
        if 'timesteps' in sequence.keys():
            y = {'input': sequence['input'], 'timesteps': sequence['timesteps']}
        x = sequence['label']
        out.append(y)
        out.append(x)
        return out

    def generate_sequences(self, time_series, L_1, L_2, timestep=False, *args, **kwargs):
        """
        Args:
            - time_series: time series data (L, D)
            - L_1: number of timesteps shown to the model
            - L_2: number of timesteps to be predicted by the model
            - timestep: if Ture the timestep is included in the input sequences
        Returns:
            - sequences: list of dicts, 
                each dict contains:
                    - 'input': input sequence to the model (L_1, D)
                    - 'label': the sequence to be predicted by the model (L_2, D)
                if timestep is true:
                    - 'timesteps': timesteps (L1 + L2)

        """

        n = len(time_series['times'])
        times = time_series['times']
        values = time_series['values']
        sequences = []
        for i in range(n-L_1):
            input = values[i:i+L_1] # (L_1, D)
            label = values[i+L_1:i+L_1+L_2] # (L_2, D) 
            if timestep:
                timesteps = times[i:i+L_1+L_2] # (L_1 + L2)
                sequences.append({'input': input,'label': label, 'timesteps': timesteps})        
            else:    
                sequences.append({'input': input,'label': label})        
         
        return sequences


@utils.register_lightning_datamodule(name='Crypto')            
class CryptoDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.batch_size = config.training.batch_size
        dataset = config.data.dataset
        coin_name = config.data.coin_name
        time_freq = config.data.time_frequency
        L_1 = config.data.L_1
        L_2 = config.data.L_2
        self.data = CryptoDataset(coin_name=coin_name, time_freq=time_freq, L_1=L_1, L_2=L_2)

    def setup(self, stage=None):
        n = self.data.__len__()
        n_train = int(0.8*n)
        n_val = int(0.1*n)
        idx = list(range(n))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:(n_train + n_val)]
        test_idx = idx[(n_train + n_val):]

        self.train_data = Subset(self.data, train_idx)
        self.val_data = Subset(self.data, val_idx)
        self.test_data = Subset(self.data, test_idx)


    def train_dataloader(self):
        return DataLoader(self.train_data,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=40)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=40)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=40)
