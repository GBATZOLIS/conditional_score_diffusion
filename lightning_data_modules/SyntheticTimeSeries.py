import torch
from . import utils
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset


class SyntheticTimeSeries(Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.time_series =  self.generate_data(*args, **kwargs)
        self.n = len(self.time_series['times'])
        self.sequences = self.generate_sequences(self.time_series, timestep=True, *args, **kwargs)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        #return self.sequences[index]
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
            input = torch.stack(values[i:i+L_1], dim=0) # (L_1, D)
            label = torch.stack(values[i+L_1:i+L_1+L_2], dim=0) # (L_2, D) 
            if timestep:
                timesteps = torch.stack(times[i:i+L_1+L_2]) # (L_1 + L2)
                sequences.append({'input': input,'label': label, 'timesteps': timesteps})        
            else:    
                sequences.append({'input': input,'label': label})        
         
        return sequences

    def generate_data(*args, **kwargs):
        raise NotImplemented

class SineWave(SyntheticTimeSeries):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def generate_data(self, n, dt=0.1, *args, **kwargs):
        times = [torch.tensor(i * dt) for i in range(n)] # each time ()
        values = [torch.sin(t).unsqueeze(-1) for t in times] # each value (1,)
        return {'times': times, 'values': values}

class GeometricBM(SyntheticTimeSeries):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_data(self,
                        n, 
                        mu=1,
                        sigma=0.1,
                        dt=1,
                        init_vals=None):

        t = torch.tensor([dt * i for i in range(n)])
        init_vals = init_vals if init_vals is not None else 100* torch.ones([1])
        increments = dt*torch.randn((n,))
        W_t = torch.cumsum(increments, dim=0)
        S_t = init_vals * torch.exp((mu - sigma**2 / 2)*t + sigma*W_t)

        time_series = S_t
        sequences = [[S_t[i], S_t[i+1]] for i in range(n-1)]

        return time_series, sequences


class ARProcess(SyntheticTimeSeries):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
         
    def generate_data(self,
                        n, 
                        k=1,
                        sigma=0.1,
                        init_vals=None):
        '''
        Args:
            n: number of vaules to be generated
            k: order of autoregressive process
            p: dimension of time series
            sigma: std of noise
            init_vals: initial values
        Returns:
            time_series: list of tensors of (p,)
            sequences: [(k,p),(p,)] pairs of k points and the succesive point
        '''

        # initialize 
        time_series = init_vals if init_vals is not None else [torch.tensor([0]).float()] * k 
        theta = 0.5 * torch.ones([k, 1]) # (k,1)
        theta[0] *= 0.5
        sequences=[]
        # simulate
        for i in range(n):
            x = torch.stack(time_series[-k:], dim=0) # (k, p)
            y = (x.transpose(0,1)).matmul(theta) + sigma * torch.randn_like(time_series[-1]) # (p, 1)
            sequences.append([x, y.squeeze(1)])
            time_series.append(y.squeeze(1))

        return time_series, sequences


@utils.register_lightning_datamodule(name='SyntheticTimeSeries')            
class SyntheticTimeSeriesDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.n = config.data.n_samples
        self.batch_size = config.training.batch_size
        dataset = config.data.dataset
        L_1 = config.data.L_1
        L_2 = config.data.L_2
        if dataset == 'Sine':
            self.data = SineWave(n=self.n, L_1=L_1, L_2=L_2)
        elif dataset == 'ARProcess':
            self.data = ARProcess(self.n)
        elif dataset == 'GeometricBM':
            raise GeometricBM(self.n)

    def setup(self, stage=None) -> None:
        n = self.n
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