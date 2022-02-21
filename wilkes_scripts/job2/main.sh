#!/bin/bash

module unload miniconda/3
module load cuda/11.4

module list

nvidia-smi

source /home/js2164/.bashrc
conda activate score_sde

REPO=/rds/user/js2164/hpc-work/repos/score_sde_pytorch/
#REPO=/home/js2164/jan/repos/diffusion_score/

CONFIG=configs/jan/celebA/hpc/potential_snr.py
#configs/jan/circles/vanilla_vp.py
#configs/jan/celebA/potential_hpc.py
#configs/jan/celebA/hpc.py
#configs/jan/circles/potential/circles_potential.py
#configs/jan/circles/curl_penalty/LAMBDA.py

LOG=logs/celebA/potential
#logs/snrsde
#logs/circles_new
#logs/celebA

CHECKPOINT=logs/celebA/potential/ve_2/checkpoints/epoch=46-step=152373.ckpt
#logs/celebA/lightning_logs/potential_1/checkpoints/epoch=22-step=74565.ckpt

NAME=ve_3

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG \
               --checkpoint_path $CHECKPOINT \ 
               --log_name $NAME \        
               
               