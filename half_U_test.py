import torch
import run_lib
from models.half_U import HalfUEncoder, HalfUDecoder
from models.ddpm import DDPM
from configs.utils import read_config

# ddpm_config_path = 'configs/celebA/ddpm.py'
# ddpm_config = read_config(ddpm_config_path)
# ddpm = DDPM(ddpm_config)

config_path = 'configs/jan/halfU.py'
config = read_config(config_path)

encoder = HalfUEncoder(config)
decoder = HalfUDecoder(config)

x = torch.randn((1,3,64,64))
t = torch.randn((1,))

#y = ddpm(x,t)

z = encoder(x)
x_hat = decoder(z)

print(torch.linalg.norm(x-x_hat).item())