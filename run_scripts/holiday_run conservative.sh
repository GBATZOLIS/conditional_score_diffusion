#!/bin/bash
source /home/js2164/miniconda3/etc/profile.d/conda.sh
module list

nvidia-smi

conda activate score_sde

REPO=/home/js2164/jan/repos/diffusion_score
CONFIG=configs/jan/GaussianBubbles.py
LOG=logs/gaussian_bubbles

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG
