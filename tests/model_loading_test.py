#%%
import os
os.chdir("..")
#%%
import pytorch_lightning as pl
import ml_collections
from matplotlib import pyplot as plt
from models.ncsnpp import NCSNpp
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
#%%
from configs.jan.holiday.ncsnpp_tets import get_config
config=get_config()
# %%
path = '/store/CIA/js2164/repos/diffusion_score/lightning_logs/version_12/checkpoints/epoch=43-step=142647.ckpt'
model = BaseSdeGenerativeModel.load_from_checkpoint(checkpoint_path=path)
# %%
model.configure_default_sampling_shape(config)
model.configure_sde(config)
model.eval()
# %%
x=model.sample(show_evolution=False)
# %%
print(x[0].shape)
from utils import batch_show
batch_show(x[0])
# %%
from lightning_callbacks.callbacks import ImageVisualizationCallback
imvc= ImageVisualizationCallback()
# %%
imvc.visualise_samples(x[0],model)
# %%
import numpy as np
import torchvision
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image
batch = normalise_per_image(x[0], [0,1])
image=np.transpose(torchvision.utils.make_grid(batch, normalize=False).cpu(),(1,2,0))
plt.imshow(image)

# %%
batch = x[0]
image=np.transpose(torchvision.utils.make_grid(batch, normalize=True).cpu(),(1,2,0))
plt.imshow(image)
# %%
samples = x[0].cpu()
grid=torchvision.utils.make_grid(samples, normalize=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/js2164/jan/repos/diffusion_score/TFB')
writer.add_image('images', grid, 0)
writer.close()
# %%
