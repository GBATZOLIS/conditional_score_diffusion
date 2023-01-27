import torch.distributions as D
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset, DataLoader 
import numpy as np
from PIL import Image
#helper function for plotting samples from a 2D distribution.
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import io
from . import utils
from torchvision.transforms.functional import normalize
from sklearn.datasets import make_circles
import sde_lib
from utils import compute_grad
import random
from tqdm import tqdm

class SyntheticDataset(Dataset):
    def __init__(self, config):
        super(SyntheticDataset, self).__init__()
        self.return_labels = config.data.return_labels
        self.data, self.labels = self.create_dataset(config)
   
    def create_dataset(self, config):
        raise NotImplemented
        # return data, labels

    def log_prob(self, xs, ts):
        raise NotImplemented

    def ground_truth_score(self, xs, ts):
        log_prob_x = lambda x: self.log_prob(x,ts)
        return compute_grad(log_prob_x, xs)

    def __getitem__(self, index):
        if self.return_labels:
            item = self.data[index], self.labels[index]
        else:
            item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)

class SquaresManifold(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def create_dataset(self, config):
        #make sure we are generating the same dataset when we resume training
        random.seed(config.seed)

        num_samples = config.data.data_samples
        num_squares = config.data.num_squares #10
        square_range = config.data.square_range #[3, 5]
        img_size = config.data.image_size #32

        data = []
        for num in tqdm(range(num_samples)):
            img = torch.zeros(size=(img_size, img_size))
            for _ in range(num_squares):
                side = random.choice(square_range)
                start = (side+1)//2
                finish = img_size - (side+1)//2
                x = random.choice(np.arange(start, finish))
                y = random.choice(np.arange(start, finish))
                img = self.paint_the_square(img, x, y, side)   
            data.append(img.to(torch.float32).unsqueeze(0))
        
        data = torch.stack(data)
        return data, []
    
    def paint_the_square(self, img, center_x, center_y, side):
        for i in range(side):
            for j in range(side):
                img[center_x - ((side+1)//2 - 1) + i, center_y - ((side+1)//2 - 1) + j]+=1
        return img

class FixedSquaresManifold(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def get_the_squares(self, seed, num_squares, square_range, img_size):
        random.seed(seed)
        squares_info = []
        for _ in range(num_squares):
            side = random.choice(square_range)
            start = (side+1)//2
            finish = img_size - (side+1)//2
            x = random.choice(np.arange(start, finish))
            y = random.choice(np.arange(start, finish))
            squares_info.append([x, y, side])

        return squares_info

    def create_dataset(self, config):
        num_samples = config.data.data_samples
        num_squares = config.data.num_squares #10
        square_range = config.data.square_range #[3, 5]
        img_size = config.data.image_size #32
        seed = config.seed 

        squares_info = self.get_the_squares(seed, num_squares, square_range, img_size)

        data = []
        for num in range(num_samples):
            img = torch.zeros(size=(img_size, img_size))
            for i in range(num_squares):
                x, y, side = squares_info[i]
                img = self.paint_the_square(img, x, y, side)   
            data.append(img.to(torch.float32).unsqueeze(0))
        
        data = torch.stack(data)
        return data, []
    
    def paint_the_square(self, img, center_x, center_y, side):
        c = random.random()
        for i in range(side):
            for j in range(side):
                img[center_x - ((side+1)//2 - 1) + i, center_y - ((side+1)//2 - 1) + j]+=c
        return img

class FixedGaussiansManifold(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def get_the_gaussian_centers(self, seed, num_gaussians, std_range, img_size):
        random.seed(seed)
        guassians_info = []
        pairs = []
        for i in range(img_size):
            for j in range(img_size):
                pairs.append([i,j])
        
        #select the centers without replacement
        guassians_info = random.sample(pairs, k=num_gaussians)

        return guassians_info

    def create_dataset(self, config):
        num_samples = config.data.data_samples
        num_gaussians = config.data.num_gaussians #10
        std_range = config.data.std_range #[1, 5]
        img_size = config.data.image_size #32
        seed = config.seed 

        centers_info = self.get_the_gaussian_centers(seed, num_gaussians, std_range, img_size)

        data = []
        for num in range(num_samples):
            img = torch.zeros(size=(img_size, img_size))
            for i in range(num_gaussians):
                x, y = centers_info[i]

                #paint the gaussians efficiently
                img = self.paint_the_gaussian(img, x, y, std_range)    
            
            #scale the image to [0, 1] range
            min_val, max_val = torch.min(img), torch.max(img)
            img -= min_val
            img /= max_val-min_val

            data.append(img.to(torch.float32).unsqueeze(0))
        
        data = torch.stack(data)
        return data, []
    
    def paint_the_gaussian(self, img, center_x, center_y, std_range):
        std = random.uniform(std_range[0], std_range[1])
        c = 1/(np.sqrt(2*np.pi)*std)
        new_img = torch.zeros_like(img)
        
        x = torch.tensor(np.arange(img.size(0)))
        y = torch.tensor(np.arange(img.size(1)))
        xx, yy = torch.meshgrid((x,y), indexing='ij')

        d = -1/(2*std**2)
        new_img = np.exp(d*((xx-center_x)**2+(yy-center_y)**2))
        new_img *= c
        img += new_img
        return img

class GaussianBubbles(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
        self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=None)
    
    def create_dataset(self, config):
        data_samples = config.data.data_samples
        self.mixtures = config.data.mixtures
        self.std=config.data.std
        n=self.mixtures
        categorical = D.categorical.Categorical(torch.ones(n,)/n)
        distributions = []
        self.centres = self.calculate_centers(n)
        for center in self.centres:
            distributions.append(D.normal.Normal(loc=center, scale=self.std))
        mixtures_indices = categorical.sample(torch.Size([data_samples]))
        data = []
        for index in mixtures_indices:
            data.append(distributions[index].sample().to(torch.float32))
        data = torch.stack(data)
        # if normalize:
        #     data[:,0] = data[:,0] / torch.max(torch.abs(data[:,0]))
        #     data[:,1] = data[:,1] / torch.max(torch.abs(data[:,1]))
        return data, mixtures_indices

    def calculate_centers(self, num_mixtures):
                if num_mixtures==1:
                    return torch.zeros(1,2)
                else:
                    centers=[]
                    theta=0
                    for i in range(num_mixtures):
                        center=[np.cos(theta), np.sin(theta)]
                        centers.append(center)
                        theta+=2*np.pi/num_mixtures
                    centers=torch.tensor(centers)
                    return centers

    # def log_prob(self, xs, ts):
    
    #     def normal_density_2D(x, mu, sigma):
    #         '''
    #         x - vector of points (2,)
    #         mus - mean vector (2,)
    #         sigmas - standard deviation float
    #         returns a float with log_prob
    #         '''
    #         const = 2 * np.pi * sigma**2
    #         num = torch.exp( - torch.linalg.norm(x - mu)**2 / (2 * sigma**2)  )
    #         return num / const

    #     sigmas_t = self.sde.marginal_prob(torch.zeros_like(xs), ts)[1]
    #     sigmas = torch.sqrt(torch.tensor([self.std ** 2] * xs.shape[0]).type_as(sigmas_t) + sigmas_t ** 2)
    #     log_probs = []
    #     for x, sigma in zip(xs, sigmas):
    #          # float
    #         prob=0
    #         for mu in self.centres:
    #             prob += normal_density_2D(x, mu, sigma) / self.mixtures
    #         log_prob = torch.log(prob)
    #         log_probs.append(log_prob)
    #     return torch.stack(log_probs, dim=0)


    # def log_prob2(self, xs, ts):
    #     log_probs = []
    #     sigmas_t = self.sde.marginal_prob(torch.zeros_like(xs), ts)[1]
    #     sigmas = torch.sqrt(torch.tensor([self.std ** 2] * xs.shape[0]).type_as(sigmas_t) + sigmas_t ** 2)
    #     for x, sigma in zip(xs, sigmas):
    #         n=self.mixtures
    #         mus=torch.tensor(self.calculate_centers(n)).type_as(x)
    #         sigmas=torch.tensor([[sigma, sigma]]*n).type_as(x)
    #         mix = D.categorical.Categorical(torch.ones(n,).type_as(x))
    #         comp = D.independent.Independent(D.normal.Normal(
    #                     mus, sigmas), 1)
    #         gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
    #         log_prob = gmm.log_prob(x)
    #         log_probs.append(log_prob)
    #     return torch.stack(log_probs, dim=0)


    def normal_density_2D(self, xs, mus, sigmas):
                '''
                x - vector of points (N,2)
                mus - vector of means (K,2)
                sigmas - vector of simgas (N,)
                returns a vector of probabilites (N,K)
                '''
                const = 2 * np.pi * sigmas**2
                num = torch.exp(- torch.linalg.norm(xs[:, None, :] - mus[None, ...], dim=2)**2 / (2 * sigmas[:,None]**2) )   # (N,K)
                denum = const[:,None] # (N,1)
                return num / denum # (N,K)

    def log_prob(self, xs, ts):
        mus = self.centres.type_as(xs)
        sigmas_t = self.sde.marginal_prob(torch.zeros_like(xs), ts)[1]
        sigmas = torch.sqrt(torch.tensor([self.std ** 2] * xs.shape[0]).type_as(sigmas_t) + sigmas_t ** 2) # (N,)
        return torch.log(torch.mean(self.normal_density_2D(xs, mus, sigmas), dim=1)) #(N,)

    # def ground_truth_score(self, batch, ts):
    #         '''
    #         batch (N, 2)
    #         sigmas_t (N,)
    #         returns gt score (N,2)
    #         '''
    #         #assert self.sde == 'vesde'
            
    #         def gmm_score(xs, mus, sigmas):
    #             num = torch.sum(self.normal_density_2D(xs, mus, sigmas)[...,None] * (mus[None, ...] - xs[:, None, :]), dim=1) #(N,2)
    #             denum = torch.sum(self.normal_density_2D(xs, mus, sigmas), dim=1)[...,None] #(N,1)
    #             return num / (sigmas[...,None] ** 2 * denum) #(N,1)

    #         mus = self.centres.type_as(batch) #(K,2)
    #         sigmas_t = self.sde.marginal_prob(torch.zeros_like(batch), ts)[1]
    #         sigmas = torch.sqrt(torch.tensor([self.std ** 2] * batch.shape[0]).type_as(sigmas_t) + sigmas_t ** 2) # (N,)
    #         scores = gmm_score(batch, mus, sigmas)
    #         return scores
    

class Circles(SyntheticDataset):
    def __init__(self, config):
        self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=None)
        super().__init__(config)

    def create_dataset(self, config):
        num_data_samples = config.data.data_samples
        noise = config.data.noise
        #factor = config.data.factor
        labels =[]

        # radious distribution
        self.mus = torch.tensor([0.5, 1])
        self.std = noise
        mix = D.Categorical(torch.ones(2,))
        comp = D.Normal(self.mus, self.std* torch.ones(2,))
        radious_distro = D.MixtureSameFamily(mix, comp)

        # angle distribution
        angle_distro = D.Uniform(0, 2 * np.pi)

        r = radious_distro.sample((num_data_samples,))
        theta = angle_distro.sample((num_data_samples,))
        samples_polar = torch.stack([r,theta], dim=1)
        samples = self.cartesian(samples_polar)
        return samples, labels

    # def log_prob(self, xs, ts):
    #     N = 100
    #     mus, sigmas = self.sde.marginal_prob(xs, ts)
    #     def normal_density_2D(x, mu, sigma):
    #         '''
    #         x - vector of points (2,)
    #         mus - mean vector (2,)
    #         sigmas - standard deviation float
    #         returns a float with log_prob
    #         '''
    #         const = 2 * np.pi * sigma**2
    #         num = torch.exp( - torch.linalg.norm(x - mu)**2 / (2 * sigma**2)  )
    #         return num / const
        
    #     probs = []
    #     for mu, sigma in zip(mus, sigmas):
    #         x0s = self.data[:N]
    #         probs = [normal_density_2D(x, x0, sigma) for x0 in x0s]
    #         prob = probs.mean()




        # def normal_pdf(x, mu, std):
        #     c = 1 / (np.sqrt(2 * np.pi) * std)
        #     return c * torch.exp( - (x - mu) **2 / (2 * std **2) )
        
        # sigmas_t = self.sde.marginal_prob(torch.zeros_like(xs), ts)[1]
        # sigmas = torch.sqrt(torch.tensor([self.std ** 2] * xs.shape[0]).type_as(sigmas_t) + sigmas_t ** 2)
        # log_probs = []
        # for x, sigma in zip(xs, sigmas):
        #     r = torch.sqrt(x[0] ** 2 + x[1] ** 2)
        #     prob = 0.5 * normal_pdf(r, self.mus[0], sigma) + 0.5 * normal_pdf(r, self.mus[1], sigma)
        #     prob = prob / (2 * np.pi)
        #     log_prob = torch.log(prob)
        #     log_probs.append(log_prob)

        # return torch.stack(log_probs, dim=0)

    def cartesian(self, polar):
        #polar = torch.stack([r,theta], dim=1) # (N, 2)
        return torch.stack([torch.tensor([r*torch.cos(theta), r*torch.sin(theta)]) for (r ,theta) in polar], dim=0)

    def polar(self, cartesian):
        #cartesian = torch.stack([x, y], dim=1) # (N, 2)
        return torch.stack([torch.tensor([torch.sqrt(x**2 + y**2), y]) for (x ,y) in cartesian], dim=0)


    


        # points, labels = make_circles(n_samples=data_samples, noise=noise, factor=factor)
        # points = torch.from_numpy(points).float()
        # labels = torch.from_numpy(labels).float()
        return points, labels

@utils.register_lightning_datamodule(name='Synthetic')
class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config): 
        super().__init__()
        #Synthetic Dataset arguments
        self.config = config
        self.split = config.data.split
        self.dataset_type = self.config.data.dataset_type

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size
        
    def setup(self, stage=None): 
        if self.dataset_type == 'GaussianBubbles':
            self.dataset = GaussianBubbles(self.config)
        elif self.dataset_type == 'Circles':
            self.dataset = Circles(self.config)
        elif self.dataset_type == 'SquaresManifold':
            self.dataset = SquaresManifold(self.config)
        elif self.dataset_type == 'FixedSquaresManifold':
            self.dataset = FixedSquaresManifold(self.config)
        elif self.dataset_type == 'FixedGaussiansManifold':
            self.dataset = FixedGaussiansManifold(self.config)
        else:
            raise NotImplemented

        l=len(self.dataset)
        self.train_data, self.valid_data, self.test_data = random_split(self.dataset, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers, shuffle=False) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers, shuffle=False) 
    

def scatter_plot(x, x_lim=None, y_lim=None, labels=None, save=False):
    assert len(x.shape)==2, 'x must have 2 dimensions to create a scatter plot.'
    fig = plt.figure()
    x1 = x[:,0].cpu().numpy()
    x2 = x[:,1].cpu().numpy()
    plt.scatter(x1, x2, c=labels, s=8)
    if x_lim is not None and y_lim is not None:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    if save:
        plt.savefig('out.jpg', dpi=300)
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image