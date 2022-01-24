#!/bin/bash

module unload miniconda/3
module load cuda/11.4

module list

nvidia-smi

source /home/js2164/.bashrc
conda activate score_sde

REPO=/rds/user/js2164/hpc-work/repos/score_sde_pytorch/
CONFIG=configs/jan/circles/potential/circles_potential.py
LOG=logs/curl_penalty_new
CHECKPOINT=logs/curl_penalty_new/lightning_logs/potential/checkpoints/epoch=6249-step=499999.ckpt

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG \
               --checkpoint_path $CHECKPOINT