import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader 
import lightning_data_modules.utils as utils

class KSphereDataset(Dataset):

    def __init__(self, config) -> None:
        super().__init__()
        self.data = self.generate_data(config.data.get('data_samples'), 
                                        config.data.get('n_spheres'), 
                                        config.data.get('ambient_dim'), 
                                        config.data.get('manifold_dim'),
                                        config.data.get('noise_std'), 
                                        config.data.get('embedding_type'),
                                        config.data.get('radii', []),
                                        config.data.get('angle_std', -1)
                                        )

    def generate_data(self, n_samples, n_spheres, ambient_dim, 
                        manifold_dim, noise_std, embedding_type,
                        radii, angle_std):
            if radii == []:
                radii = [1] * n_spheres

            if isinstance(manifold_dim, int):
                manifold_dims = [manifold_dim] * n_spheres
            elif isinstance(manifold_dim, list):
                manifold_dims = manifold_dim
                
            data = []
            for i in range(n_spheres):
                    manifold_dim = manifold_dims[i]
                    new_data = self.sample_sphere(n_samples, manifold_dim, angle_std)
                    new_data = new_data * radii[i]

                    if embedding_type == 'random_isometry':
                        # random isometric embedding
                        randomness_generator = torch.Generator().manual_seed(0)
                        embedding_matrix = torch.randn(size=(ambient_dim, manifold_dim+1), generator=randomness_generator)
                        q, r = np.linalg.qr(embedding_matrix)
                        q = torch.from_numpy(q)
                        new_data = (q @ new_data.T).T
                    elif embedding_type == 'first':
                        # embedding into first manifold_dim + 1 dimensions
                        suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                        new_data = torch.cat([new_data, suffix_zeros], dim=1)
                    elif embedding_type == 'separating':
                        # embbedding which puts spheres in non-intersecting dimensions
                        if n_spheres * (manifold_dim + 1) > ambient_dim:
                            raise RuntimeError('Cant fit that many spheres. Enusre that n_spheres * (manifold_dim + 1) <= ambient_dim')
                        prefix_zeros = torch.zeros((n_samples, i * (manifold_dim + 1)))
                        new_data = torch.cat([prefix_zeros, new_data], dim=1)
                        suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                        new_data = torch.cat([new_data, suffix_zeros], dim=1)
                    elif embedding_type == 'along_axis':
                        # embbedding which puts spheres in non-intersecting dimensions
                        if (n_spheres - 1) + (manifold_dim + 1) > ambient_dim:
                            raise RuntimeError('Cant fit that many spheres.')
                        prefix_zeros = torch.zeros((n_samples, i))
                        new_data = torch.cat([prefix_zeros, new_data], dim=1)
                        suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                        new_data = torch.cat([new_data, suffix_zeros], dim=1)    
                    else:
                        raise RuntimeError('Unknown embedding type.')
                        
                    # add noise
                    new_data = new_data + noise_std * torch.randn_like(new_data)
                    data.append(new_data)

            data = torch.cat(data, dim=0)
            return data

    def sample_sphere(self, n_samples, manifold_dim, std=-1):

        def polar_to_cartesian(angles):
            xs = []
            sin_prod=1
            for i in range(len(angles)):
                x_i = sin_prod * torch.cos(angles[i])
                xs.append(x_i)
                sin_prod *= torch.sin(angles[i])
            xs.append(sin_prod)
            return torch.stack(xs)[None, ...]

        if std == -1:
            new_data = torch.randn((n_samples, manifold_dim+1))
            norms = torch.linalg.norm(new_data, dim=1)
            new_data = new_data / norms[:,None]
            return new_data
        else:
            sampled_angles = std * torch.randn((n_samples,manifold_dim))
            return torch.cat([polar_to_cartesian(angles) for angles in sampled_angles], dim=0)    


    def __getitem__(self, index):
        item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)


@utils.register_lightning_datamodule(name='KSphere')
class KSphereDataModule(pl.LightningDataModule):
    def __init__(self, config): 
        super().__init__()
        #Synthetic Dataset arguments
        self.config = config
        self.split = config.data.split

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size
        
    def setup(self, stage=None): 

        self.dataset = KSphereDataset(self.config)
        l=len(self.dataset)
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True)  
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=True) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 