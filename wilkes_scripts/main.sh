#!/bin/bash

module unload miniconda/3
module load cuda/11.4

module list

nvidia-smi

source /home/js2164/.bashrc
conda activate score_sde

REPO=/rds/user/js2164/hpc-work/repos/score_sde_pytorch/
CONFIG=configs/jan/GaussianBubbles.py
LOG=logs/gaussian_bubbles

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG
