import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
from models.fcn import FCN
import pickle
from plot_utils import plot_spectrum

#path = 'logs/ksphere/n_1/dim_10/random_isometry/uniform_random/svd/svd_9000.pkl'
#path = 'logs/ksphere/n_1/dim_50/random_isometry/uniform_random/svd/svd_14000.pkl'
#path = 'logs/line/sine_line/svd/svd_26500.pkl'
#path = 'logs/mammoth/ve_random/svd/svd_16500.pkl'
path = 'logs/daniel/test/svd/svd.pkl'

with open(path, 'rb') as f:
    svd = pickle.load(f)

from plot_utils import plot_spectrum, plot_distribution, plot_dims
plot_spectrum(svd, mode='all')
plot_distribution(svd, mode='all')
plot_dims(svd)
plt.show()