import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import run_lib
import torch
import pandas as pd
import numpy as np
import pickle
import os
from lightning_modules.VAE import VAE
from utils import fix_rds_path
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from lightning_data_modules.ImageDatasets import CelebAAnnotatedDataset
from lightning_modules.utils import create_lightning_module
from utils import fix_config
from sklearn.linear_model import LogisticRegression
from semantic_manipulation.utils import spherical_to_cartesian, cartesian_to_spherical

#load picled files in tmp
with open('tmp/cls.pkl', 'rb') as f:
    cls = pickle.load(f)
with open('tmp/cls_spherical.pkl', 'rb') as f:
    cls_spherical = pickle.load(f)
with open('tmp/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('tmp/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('tmp/Z_train.pkl', 'rb') as f:
    Z_train = pickle.load(f)
with open('tmp/Z_test.pkl', 'rb') as f:
    Z_test = pickle.load(f)

# load model
# path = '/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/paper/pretrained/celebA_64/only_ResNetEncoder_VAE_KLweight_0.01/config.pkl'
path = '/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/gd_ffhq/only_encoder_ddpm_plus_smld_VAE_KLweight_0.01_DiffDecoders_continuous_prior/config.pkl'
with open(path, 'rb') as f:
  config = pickle.load(f)
config = fix_config(config)
model = create_lightning_module(config)
pl_module = model.load_from_checkpoint(config.model.checkpoint_path, config=config)
pl_module.configure_sde(config)
pl_module = pl_module.to('cuda')
pl_module.eval()


# edit config
home = os.path.expanduser('~')
config.data.base_dir = f'{home}/rds_work/datasets/'
config.data.dataset = 'celebA-HQ-160'
config.data.attributes = ['Male']


test_data = CelebAAnnotatedDataset(config, phase='val')
test_loader = DataLoader(test_data, batch_size=128, num_workers=8, shuffle=False)
X = next(iter(test_loader))[0].to('cuda')
spherical = cartesian_to_spherical(pl_module.encode(X).detach().cpu().numpy())

# manipulate
params = cls_spherical.coef_[0]
ts =np.linspace(0, 0.01, 100)
v = np.concatenate([[0], params[1:]])
#v = params
# choose random image with y = 0
#i = np.random.choice(np.where(y_test[:128] == 0)[0])

for i in range(128):
    sgn = 1 if y_test[i] == 0 else -1
    mainpulated_spherical = np.stack([spherical[i] + sgn * t * v for t in ts])
    print(np.round(cls_spherical.predict_proba(mainpulated_spherical)[:,1], 2))
    Z_manipulated = torch.tensor(spherical_to_cartesian(mainpulated_spherical))

    # decode
    X_manipulated = pl_module.decode(Z_manipulated.to(pl_module.device))

    # create dir for images
    if not os.path.exists(f'tmp/{i}'):
        os.makedirs(f'tmp/{i}')

    # plot X_0
    import matplotlib.pyplot as plt
    plt.imshow(X[i].detach().cpu().permute(1,2,0))
    plt.axis('off')
    # no background
    plt.savefig(f'tmp/{i}/X_0.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    # plot X_manipulated as grid
    from torchvision.utils import make_grid
    grid = make_grid(X_manipulated.detach().cpu(), nrow=10)
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.savefig(f'tmp/{i}/rec_manipulated_grid.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # plot rec_0
    plt.imshow(X_manipulated[0].detach().cpu().permute(1,2,0))
    plt.axis('off')
    plt.savefig(f'tmp/{i}/rec_0.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # plot rec_1
    plt.imshow(X_manipulated[-1].detach().cpu().permute(1,2,0))
    plt.axis('off')
    plt.savefig(f'tmp/{i}/rec_1.png', dpi=300, bbox_inches='tight', pad_inches=0)