import torch
import torchvision
import pickle
from utils import fix_rds_path
from lightning_modules.utils import create_lightning_module
from matplotlib import pyplot as plt
import os 
import run_lib

# get home path
home = os.path.expanduser("~")
# load config
with open(f'{home}/rds_work/projects/scoreVAE/experiments/paper/pretrained/FFHQ_128/only_encoder_ddpm_plus_smld_VAE_KLweight_0.01/config.pkl', 'rb') as f:
  config = pickle.load(f)

config.model.time_conditional = True
#diff_vae = EncoderOnlyPretrainedScoreVAEmodel.load_from_checkpoint('/store/CIA/js2164/rds/gb511/projects/scoreVAE/experiments/paper/pretrained/celebA_64/only_ResNetEncoder_VAE_KLweight_0.01/checkpoints/best/last.ckpt',
#                                                                    config=config)



model = create_lightning_module(config)
pl_module = model.load_from_checkpoint(fix_rds_path(config.model.checkpoint_path), config=config)
pl_module.configure_sde(config)
pl_module = pl_module.to('cuda')
pl_module.eval()


path_jan = 'images_for_manipulation/jan3.jpg'
#jan=torchvision.io.read_image(path_jan)[:,:150,:]
jan=torchvision.io.read_image(path_jan)
min_dim = min(jan.shape[1], jan.shape[2])
offset = 0
jan = jan[:, offset:(min_dim+offset), :min_dim]
georgios = torchvision.io.read_image('images_for_manipulation/georgios.jpg')[:,25:225,50:250]
resize = torchvision.transforms.Resize((128,128))
jan=resize(jan)
# normalize to 0, 1
jan = jan/255
georgios=resize(georgios)
georgios = georgios/255

#print min max of jan and georgios
print(jan.min(), jan.max())
print(georgios.min(), georgios.max())


# reapeat jan 128 times
B=1
jan_batch = torch.stack([jan]*B).to('cuda')
latent_mean = False
reconstructed_jan = pl_module.encode_n_decode(jan_batch, use_pretrained=config.training.use_pretrained,
                                                          encoder_only=config.training.encoder_only,
                                                          t_dependent=config.training.t_dependent)
# georgios
georgios_batch = torch.stack([georgios]*B).to('cuda')
reconstructed_georgios = pl_module.encode_n_decode(georgios_batch, use_pretrained=config.training.use_pretrained,
                                                          encoder_only=config.training.encoder_only,
                                                          t_dependent=config.training.t_dependent)

#print min max of reconstructed jan and georgios
print(reconstructed_jan.min(), reconstructed_jan.max())
print(reconstructed_georgios.min(), reconstructed_georgios.max())

# show jan and georgios on one figure
fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(jan.permute(1,2,0))
ax[0,1].imshow(georgios.permute(1,2,0))
ax[1,0].imshow(reconstructed_jan[0].detach().cpu().permute(1,2,0))
ax[1,1].imshow(reconstructed_georgios[0].detach().cpu().permute(1,2,0))
# tight layout
plt.tight_layout()
# no axis
for a in ax.flatten():
    a.axis('off')   
plt.savefig('reconstructed_jan_georgios.png', dpi=300)