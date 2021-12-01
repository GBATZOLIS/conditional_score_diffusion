# Import required modules
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import fcn, fcn_potential
from vector_fields.vector_utils import calculate_centers, normal_score, curl, curl_backprop
from utils import compute_grad
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
from vector_fields.plot_utils import plot_curl, plot_streamlines, plot_curl_backprop

# non conservative
model = BaseSdeGenerativeModel.load_from_checkpoint('/home/js2164/jan/repos/diffusion_score/potential/lightning_logs/fcn/checkpoints/epoch=6249-step=499999.ckpt')
score = model.score_model
score = score.eval()
# conservative
model_conservative = BaseSdeGenerativeModel.load_from_checkpoint('/home/js2164/jan/repos/diffusion_score/potential/lightning_logs/fcn_potential/checkpoints/epoch=6249-step=499999.ckpt')
score_conservative = model_conservative.score_model
score_conservative= score_conservative.eval()

# plot_streamlines(score, 'standard_model_streamline')
# plot_curl(score, 'standard_model_curl')

# plot_streamlines(score_conservative, 'conservative_model_streamline')
# plot_curl(score_conservative, 'conservative_model_curl')


plot_curl_backprop(score, 'curl_backprop')

from vector_fields.sample_vector_fields import constant_curl_vf
# plot_streamlines(constant_curl_vf)
# plot_curl(constant_curl_vf, 'constant_curl')
#plot_curl_backprop(constant_curl_vf, 'constatnt_curl_backprop')