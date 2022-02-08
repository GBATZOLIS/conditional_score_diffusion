#!/bin/bash

module unload miniconda/3
module load cuda/11.4

module list

nvidia-smi

source /home/js2164/.bashrc
conda activate score_sde

REPO=/rds/user/js2164/hpc-work/repos/score_sde_pytorch/
#REPO=/home/js2164/jan/repos/diffusion_score/

CONFIG=configs/jan/circles/potential/circles_potential.py
#configs/jan/celebA/hpc.py
#configs/jan/circles/potential/circles_potential.py
#configs/jan/circles/curl_penalty/LAMBDA.py
LOG=logs/circles_new
#logs/circles_new
#logs/celebA

#CHECKPOINT=logs/curl_penalty_new/lightning_logs/LAMBDA_0/checkpoints/epoch=24999-step=1999999.ckpt

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG \
               #--checkpoint_path $CHECKPOINT